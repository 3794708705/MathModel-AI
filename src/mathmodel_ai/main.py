from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from mathmodel_ai.agents import (
    CodeAgent,
    DataAgent,
    MathModeler,
    ModelExplorer,
    ModelJury,
    ProblemAgent,
)
from mathmodel_ai.api.routes.data_execution import router as data_execution_router
from mathmodel_ai.api.routes.health import router as health_router
from mathmodel_ai.api.routes.mathematical import router as mathematical_router
from mathmodel_ai.api.routes.reasoning import router as reasoning_router
from mathmodel_ai.api.routes.system import router as system_router
from mathmodel_ai.core.config import Settings, get_settings
from mathmodel_ai.core.errors import MathModelError, ResourceNotFoundError
from mathmodel_ai.core.logging import configure_logging
from mathmodel_ai.core.middleware import UploadBodyLimitMiddleware
from mathmodel_ai.data.profiler import DataProfiler
from mathmodel_ai.data.repository import DataRepository
from mathmodel_ai.data.workflow import DataExecutionWorkflow
from mathmodel_ai.db.session import create_database_engine, create_session_factory
from mathmodel_ai.files.parsers import ParserRegistry
from mathmodel_ai.files.pipeline import FilePipeline
from mathmodel_ai.files.storage import LocalFileStore
from mathmodel_ai.files.validation import FileValidator
from mathmodel_ai.mathematical.algorithms import AlgorithmSelector
from mathmodel_ai.mathematical.evidence import EvidenceIntegrityVerifier
from mathmodel_ai.mathematical.repository import MathematicalRepository
from mathmodel_ai.mathematical.strategy import ExecutionStrategySelector
from mathmodel_ai.mathematical.workflow import MathematicalWorkflow
from mathmodel_ai.providers.factory import ProviderRegistry, build_provider_registry
from mathmodel_ai.reasoning.prompts import PromptRegistry
from mathmodel_ai.reasoning.repository import ReasoningRepository
from mathmodel_ai.reasoning.workflow import ReasoningWorkflow
from mathmodel_ai.routing.router import ModelRouter
from mathmodel_ai.sandbox.executor import SandboxExecutor
from mathmodel_ai.schemas.execution import SandboxLimits
from mathmodel_ai.schemas.model_selection import ModelFamily, ModelJuryWeights
from mathmodel_ai.schemas.solver import ProblemSizeThresholds, SolverFamily
from mathmodel_ai.solvers.generated import GeneratedProgramExecutor
from mathmodel_ai.solvers.gurobi import GurobiSolver
from mathmodel_ai.solvers.ortools import ORToolsSolver
from mathmodel_ai.solvers.router import SolverRouter
from mathmodel_ai.solvers.scipy import SciPySolver


def create_app(
    settings: Settings | None = None,
    providers: ProviderRegistry | None = None,
) -> FastAPI:
    resolved = settings or get_settings()
    configure_logging(resolved.log_level)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await application.state.providers.aclose()
            application.state.engine.dispose()

    application = FastAPI(
        title=resolved.app_name,
        version=resolved.app_version,
        lifespan=lifespan,
    )
    application.add_middleware(
        UploadBodyLimitMiddleware,
        max_body_bytes=resolved.max_upload_bytes + 64 * 1024,
    )
    application.state.settings = resolved
    application.state.engine = create_database_engine(resolved)
    application.state.providers = providers or build_provider_registry(resolved)
    application.state.session_factory = create_session_factory(application.state.engine)
    application.state.reasoning_repository = ReasoningRepository(application.state.session_factory)
    application.state.data_repository = DataRepository(application.state.session_factory)
    evidence_verifier = EvidenceIntegrityVerifier(
        absolute_tolerance=resolved.evidence_abs_tolerance,
        relative_tolerance=resolved.evidence_rel_tolerance,
    )
    application.state.evidence_verifier = evidence_verifier
    application.state.mathematical_repository = MathematicalRepository(
        application.state.session_factory,
        evidence_verifier,
    )
    model_router = ModelRouter(resolved, available_providers=application.state.providers.available)
    prompts = PromptRegistry()
    weights = ModelJuryWeights(**resolved.model_jury_weights.model_dump(), version="configured-v1")
    shared = {
        "router": model_router,
        "providers": application.state.providers,
        "prompts": prompts,
        "max_retries": resolved.reasoning_max_retries,
    }
    application.state.reasoning_workflow = ReasoningWorkflow(
        repository=application.state.reasoning_repository,
        problem_agent=ProblemAgent(
            **shared,
            ambiguity_review_threshold=resolved.ambiguity_review_threshold,
        ),
        model_explorer=ModelExplorer(**shared),
        model_jury=ModelJury(**shared, weights=weights),
        weights=weights,
    )
    file_store = LocalFileStore(resolved.storage_root)
    file_pipeline = FilePipeline(
        store=file_store,
        validator=FileValidator(
            max_archive_entries=resolved.max_archive_entries,
            max_archive_uncompressed_bytes=resolved.max_archive_uncompressed_bytes,
            max_archive_ratio=resolved.max_archive_compression_ratio,
            max_image_pixels=resolved.max_image_pixels,
        ),
        parsers=ParserRegistry(
            max_pdf_pages=resolved.max_pdf_pages,
            max_pdf_page_images=resolved.max_pdf_page_images,
        ),
        profiler=DataProfiler(),
        max_upload_bytes=resolved.max_upload_bytes,
        max_multimodal_inline_bytes=resolved.max_multimodal_inline_bytes,
    )
    sandbox_limits = SandboxLimits(
        cpu_cores=resolved.sandbox_cpu_cores,
        memory_mb=resolved.sandbox_memory_mb,
        timeout_seconds=resolved.sandbox_timeout_seconds,
        pids_limit=resolved.sandbox_pids_limit,
        max_output_bytes=resolved.sandbox_max_output_bytes,
        max_artifacts=resolved.sandbox_max_artifacts,
        max_artifact_bytes=resolved.sandbox_max_artifact_bytes,
    )
    application.state.file_store = file_store
    application.state.file_pipeline = file_pipeline
    application.state.sandbox_executor = SandboxExecutor(
        store=file_store,
        root=resolved.sandbox_root,
        image=resolved.sandbox_image,
        limits=sandbox_limits,
    )
    application.state.data_execution_workflow = DataExecutionWorkflow(
        reasoning_repository=application.state.reasoning_repository,
        data_repository=application.state.data_repository,
        file_pipeline=file_pipeline,
        data_agent=DataAgent(**shared),
        sandbox=application.state.sandbox_executor,
    )
    solver_sandbox = SandboxExecutor(
        store=file_store,
        root=resolved.solver_sandbox_root,
        image=resolved.solver_sandbox_image,
        limits=sandbox_limits,
    )
    gurobi_image = resolved.gurobi_sandbox_image
    gurobi_sandbox = (
        solver_sandbox
        if gurobi_image is None or gurobi_image == resolved.solver_sandbox_image
        else SandboxExecutor(
            store=file_store,
            root=resolved.solver_sandbox_root,
            image=gurobi_image,
            limits=sandbox_limits,
        )
    )
    application.state.solver_sandbox_executor = solver_sandbox
    application.state.code_agent = CodeAgent(**shared)
    selector = AlgorithmSelector(
        {
            ModelFamily.LINEAR_PROGRAMMING: tuple(
                SolverFamily(item) for item in resolved.lp_solver_preference
            ),
            ModelFamily.MIXED_INTEGER_LINEAR_PROGRAMMING: tuple(
                SolverFamily(item) for item in resolved.milp_solver_preference
            ),
            ModelFamily.INTEGER_PROGRAMMING: tuple(
                SolverFamily(item) for item in resolved.integer_solver_preference
            ),
            ModelFamily.NONLINEAR_PROGRAMMING: tuple(
                SolverFamily(item) for item in resolved.nlp_solver_preference
            ),
        }
    )
    solver_router = SolverRouter(
        [
            GurobiSolver(
                sandbox=gurobi_sandbox,
                store=file_store,
                license_file=resolved.gurobi_license_file,
            ),
            SciPySolver(sandbox=solver_sandbox, store=file_store),
            ORToolsSolver(sandbox=solver_sandbox, store=file_store),
        ],
        size_thresholds=ProblemSizeThresholds(
            tiny_variables=resolved.solver_tiny_max_variables,
            tiny_constraints=resolved.solver_tiny_max_constraints,
            tiny_nonzeros=resolved.solver_tiny_max_nonzeros,
            small_variables=resolved.solver_small_max_variables,
            small_constraints=resolved.solver_small_max_constraints,
            small_nonzeros=resolved.solver_small_max_nonzeros,
            medium_variables=resolved.solver_medium_max_variables,
            medium_constraints=resolved.solver_medium_max_constraints,
            medium_nonzeros=resolved.solver_medium_max_nonzeros,
        ),
        deadline_runtime_caps={
            3: resolved.solver_deadline_pressure_3_seconds,
            4: resolved.solver_deadline_pressure_4_seconds,
            5: resolved.solver_deadline_pressure_5_seconds,
        },
    )
    application.state.mathematical_workflow = MathematicalWorkflow(
        reasoning_repository=application.state.reasoning_repository,
        repository=application.state.mathematical_repository,
        math_modeler=MathModeler(**shared),
        algorithm_selector=selector,
        solver_router=solver_router,
        code_agent=application.state.code_agent,
        strategy_selector=ExecutionStrategySelector(solver_router),
        generated_executor=GeneratedProgramExecutor(
            sandbox=solver_sandbox,
            store=file_store,
        ),
        evidence_verifier=evidence_verifier,
    )
    application.include_router(health_router)
    application.include_router(system_router)
    application.include_router(reasoning_router)
    application.include_router(data_execution_router)
    application.include_router(mathematical_router)

    @application.exception_handler(ResourceNotFoundError)
    async def not_found_handler(_request: Request, exc: ResourceNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @application.exception_handler(MathModelError)
    async def mathmodel_error_handler(_request: Request, exc: MathModelError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    return application


app = create_app()
