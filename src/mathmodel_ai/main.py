import subprocess
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

from mathmodel_ai.agents import (
    CitationAgent,
    CodeAgent,
    DataAgent,
    FinalJuryAgent,
    LiteratureAgent,
    MathModeler,
    ModelExplorer,
    ModelJury,
    ModelRepairAgent,
    PaperAgent,
    PaperFactualAuditAgent,
    ProblemAgent,
    RedTeamAgent,
)
from mathmodel_ai.api.routes.benchmarks import router as benchmarks_router
from mathmodel_ai.api.routes.data_execution import router as data_execution_router
from mathmodel_ai.api.routes.final_submission import router as final_submission_router
from mathmodel_ai.api.routes.health import router as health_router
from mathmodel_ai.api.routes.independent_verification import (
    router as independent_verification_router,
)
from mathmodel_ai.api.routes.mathematical import router as mathematical_router
from mathmodel_ai.api.routes.paper import router as paper_router
from mathmodel_ai.api.routes.providers import router as providers_router
from mathmodel_ai.api.routes.reasoning import router as reasoning_router
from mathmodel_ai.api.routes.system import router as system_router
from mathmodel_ai.api.routes.verification import router as verification_router
from mathmodel_ai.benchmark.causal_evaluation import CausalBenchmarkEvaluator
from mathmodel_ai.benchmark.executor import PipelineBenchmarkExecutor
from mathmodel_ai.benchmark.manifests import BenchmarkManifestRegistry
from mathmodel_ai.benchmark.profiles import comap_mcm_2024_profile
from mathmodel_ai.benchmark.repository import BenchmarkRepository
from mathmodel_ai.benchmark.sources import BenchmarkResourceCache, BenchmarkStructureInspector
from mathmodel_ai.benchmark.workflow import BenchmarkWorkflow, resolve_git_identity
from mathmodel_ai.core.config import Settings, get_settings
from mathmodel_ai.core.errors import MathModelError, ModelDiscoveryError, ResourceNotFoundError
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
from mathmodel_ai.paper.assets import FigureAgent, TableAgent
from mathmodel_ai.paper.bundle import PaperBundleBuilder
from mathmodel_ai.paper.compiler import PDFCompiler
from mathmodel_ai.paper.evidence import VerifiedEvidenceBuilder
from mathmodel_ai.paper.hashing import sha256_json
from mathmodel_ai.paper.literature import CrossrefLiteratureSource
from mathmodel_ai.paper.rendering import LaTeXRenderer
from mathmodel_ai.paper.repository import PaperRepository
from mathmodel_ai.paper.workflow import PaperWorkflow
from mathmodel_ai.providers.discovery import ProviderModelDiscovery
from mathmodel_ai.providers.factory import ProviderRegistry, build_provider_registry
from mathmodel_ai.providers.probe import ProviderCompatibilityProbe
from mathmodel_ai.providers.repository import ProviderModelRegistry
from mathmodel_ai.providers.secrets import (
    CompositeSecretResolver,
    EncryptedDatabaseSecretStore,
    EnvironmentSecretResolver,
)
from mathmodel_ai.providers.security import EndpointSecurityPolicy, redact_sensitive_text
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
from mathmodel_ai.solvers.scalar_response import ScalarResponseSolver
from mathmodel_ai.solvers.scipy import SciPySolver
from mathmodel_ai.submission.package import SubmissionIntegrityVerifier, SubmissionPackageBuilder
from mathmodel_ai.submission.profiles import (
    CompetitionProfileRegistry,
    generic_modeling_test_profile,
)
from mathmodel_ai.submission.repository import SubmissionRepository
from mathmodel_ai.submission.rules import RuleEngine
from mathmodel_ai.submission.workflow import FinalSubmissionWorkflow
from mathmodel_ai.verification.experiment_integrity import ExperimentIntegrityVerifier
from mathmodel_ai.verification.experiments import ExperimentEngine
from mathmodel_ai.verification.independent_repository import IndependentVerificationRepository
from mathmodel_ai.verification.independent_service import IndependentVerificationService
from mathmodel_ai.verification.red_team import RedTeamAnalyzer
from mathmodel_ai.verification.repository import VerificationRepository
from mathmodel_ai.verification.requirements import VerificationRequirementRegistry
from mathmodel_ai.verification.robustness import RobustnessAnalyzer
from mathmodel_ai.verification.scenario_replay import ScenarioReplayer
from mathmodel_ai.verification.sensitivity import SensitivityAnalyzer
from mathmodel_ai.verification.validation import IndependentValidator
from mathmodel_ai.verification.workflow import VerificationWorkflow


def _safe_validation_errors(errors: Sequence[object]) -> list[dict[str, object]]:
    sanitized: list[dict[str, object]] = []
    for raw_item in errors:
        item: Mapping[str, object]
        if isinstance(raw_item, Mapping):
            item = raw_item
        else:
            item = {}
        raw_location = item.get("loc", ())
        location: list[object]
        if isinstance(raw_location, (list, tuple)):
            location = list(raw_location)
        else:
            location = []
        sanitized.append(
            {
                "type": str(item.get("type", "validation_error")),
                "loc": location,
                "msg": redact_sensitive_text(str(item.get("msg", "invalid request"))),
            }
        )
    return sanitized


def create_app(
    settings: Settings | None = None,
    providers: ProviderRegistry | None = None,
    provider_security_policy: EndpointSecurityPolicy | None = None,
) -> FastAPI:
    resolved = settings or get_settings()
    configure_logging(resolved.log_level)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            application.state.benchmark_resource_cache.close()
            await application.state.literature_source.aclose()
            await application.state.providers.aclose()
            application.state.engine.dispose()

    application = FastAPI(
        title=resolved.app_name,
        version=resolved.app_version,
        lifespan=lifespan,
    )

    @application.middleware("http")
    async def prevent_credential_response_caching(
        request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        response = await call_next(request)
        if request.url.path.endswith("/credential"):
            response.headers["Cache-Control"] = "no-store"
            response.headers["Pragma"] = "no-cache"
        return response

    if resolved.cors_origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=resolved.cors_origins,
            allow_credentials=False,
            allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
            allow_headers=["Accept", "Content-Type"],
        )
    application.add_middleware(
        UploadBodyLimitMiddleware,
        max_body_bytes=resolved.max_upload_bytes + 64 * 1024,
    )
    application.state.settings = resolved
    application.state.engine = create_database_engine(resolved)
    application.state.session_factory = create_session_factory(application.state.engine)
    application.state.secret_store = EncryptedDatabaseSecretStore(
        application.state.session_factory,
        resolved.secret_master_key,
    )
    application.state.secret_resolver = CompositeSecretResolver(
        EnvironmentSecretResolver(),
        application.state.secret_store,
    )
    application.state.provider_security_policy = provider_security_policy or (
        EndpointSecurityPolicy(
            environment=resolved.environment,
            allow_local_model_endpoints=resolved.allow_local_model_endpoints,
            allow_insecure_provider_tls=resolved.allow_insecure_provider_tls,
        )
    )
    application.state.provider_configurations = ProviderModelRegistry(
        application.state.session_factory,
        secrets=application.state.secret_resolver,
        security_policy=application.state.provider_security_policy,
        resolve_dns_on_registration=True,
    )
    application.state.provider_model_discovery = ProviderModelDiscovery(
        secrets=application.state.secret_resolver,
        security_policy=application.state.provider_security_policy,
    )
    application.state.providers = providers or build_provider_registry(
        resolved,
        configurations=application.state.provider_configurations,
        secrets=application.state.secret_resolver,
        security_policy=application.state.provider_security_policy,
    )
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
    model_router = ModelRouter(
        resolved,
        available_providers=application.state.providers.available,
        registry=application.state.provider_configurations,
    )
    application.state.model_router = model_router
    application.state.provider_probe = ProviderCompatibilityProbe(
        configurations=application.state.provider_configurations,
        providers=application.state.providers,
    )
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
        max_generated_solve_attempts=max(1, min(3, resolved.reasoning_max_retries + 1)),
    )
    validator = IndependentValidator(
        absolute_tolerance=resolved.validation_abs_tolerance,
        relative_tolerance=resolved.validation_rel_tolerance,
    )
    experiment_integrity_verifier = ExperimentIntegrityVerifier(validator)
    experiment_engine = ExperimentEngine(
        algorithm_selector=selector,
        solver_router=solver_router,
        validator=validator,
        integrity_verifier=experiment_integrity_verifier,
        response_solver=ScalarResponseSolver(sandbox=solver_sandbox, store=file_store),
    )
    application.state.independent_validator = validator
    application.state.experiment_integrity_verifier = experiment_integrity_verifier
    application.state.experiment_engine = experiment_engine
    application.state.verification_repository = VerificationRepository(
        application.state.session_factory
    )
    application.state.verification_workflow = VerificationWorkflow(
        reasoning_repository=application.state.reasoning_repository,
        mathematical_repository=application.state.mathematical_repository,
        mathematical_workflow=application.state.mathematical_workflow,
        repository=application.state.verification_repository,
        validator=validator,
        sensitivity_analyzer=SensitivityAnalyzer(experiment_engine),
        robustness_analyzer=RobustnessAnalyzer(experiment_engine),
        red_team_agent=RedTeamAgent(**shared),
        red_team_analyzer=RedTeamAnalyzer(),
        model_repair_agent=ModelRepairAgent(**shared),
        algorithm_selector=selector,
        experiment_integrity_verifier=experiment_integrity_verifier,
        max_repair_cycles=resolved.phase5_max_repair_cycles,
    )
    application.state.paper_repository = PaperRepository(application.state.session_factory)
    literature_source = CrossrefLiteratureSource(
        base_url=resolved.crossref_base_url,
        mailto=resolved.crossref_mailto,
    )
    application.state.literature_source = literature_source
    application.state.paper_workflow = PaperWorkflow(
        repository=application.state.paper_repository,
        evidence_builder=VerifiedEvidenceBuilder(
            mathematical_repository=application.state.mathematical_repository,
            verification_repository=application.state.verification_repository,
            independent_repository=IndependentVerificationRepository(
                application.state.session_factory
            ),
        ),
        literature_agent=LiteratureAgent(**shared),
        literature_source=literature_source,
        citation_agent=CitationAgent(**shared),
        paper_agent=PaperAgent(**shared),
        audit_agent=PaperFactualAuditAgent(**shared),
        figure_agent=FigureAgent(file_store),
        table_agent=TableAgent(file_store),
        renderer=LaTeXRenderer(),
        compiler=PDFCompiler(
            store=file_store,
            image=resolved.paper_compiler_image,
            timeout_seconds=resolved.paper_compile_timeout_seconds,
        ),
        bundle_builder=PaperBundleBuilder(file_store),
        store=file_store,
        reasoning_repository=application.state.reasoning_repository,
    )
    real_benchmark_profile = comap_mcm_2024_profile()
    application.state.competition_profile_registry = CompetitionProfileRegistry(
        [generic_modeling_test_profile()]
    )
    application.state.benchmark_competition_profile_registry = CompetitionProfileRegistry(
        [real_benchmark_profile]
    )
    application.state.submission_repository = SubmissionRepository(
        application.state.session_factory
    )
    application.state.submission_integrity_verifier = SubmissionIntegrityVerifier(file_store)
    application.state.final_submission_workflow = FinalSubmissionWorkflow(
        reasoning_repository=application.state.reasoning_repository,
        paper_repository=application.state.paper_repository,
        repository=application.state.submission_repository,
        profiles=application.state.competition_profile_registry,
        jury_agent=FinalJuryAgent(**shared),
        rule_engine=RuleEngine(file_store),
        package_builder=SubmissionPackageBuilder(file_store),
        store=file_store,
    )
    application.state.benchmark_final_submission_workflow = FinalSubmissionWorkflow(
        reasoning_repository=application.state.reasoning_repository,
        paper_repository=application.state.paper_repository,
        repository=application.state.submission_repository,
        profiles=application.state.benchmark_competition_profile_registry,
        jury_agent=FinalJuryAgent(**shared),
        rule_engine=RuleEngine(file_store),
        package_builder=SubmissionPackageBuilder(file_store),
        store=file_store,
    )
    application.state.benchmark_repository = BenchmarkRepository(application.state.session_factory)
    benchmark_repository_root = resolved.benchmark_repository_root
    if benchmark_repository_root == Path("."):
        benchmark_repository_root = Path(__file__).resolve().parents[2]
    benchmark_manifest_root = resolved.benchmark_manifest_root
    if not benchmark_manifest_root.is_absolute():
        benchmark_manifest_root = benchmark_repository_root / benchmark_manifest_root
    benchmark_cache_root = resolved.benchmark_cache_root
    if not benchmark_cache_root.is_absolute():
        benchmark_cache_root = benchmark_repository_root / benchmark_cache_root
    benchmark_artifact_root = resolved.benchmark_artifact_root
    if not benchmark_artifact_root.is_absolute():
        benchmark_artifact_root = benchmark_repository_root / benchmark_artifact_root
    application.state.benchmark_manifest_registry = BenchmarkManifestRegistry(
        benchmark_manifest_root
    )
    verification_requirement_registry = VerificationRequirementRegistry(benchmark_manifest_root)
    application.state.independent_verification = IndependentVerificationService(
        repository=IndependentVerificationRepository(application.state.session_factory),
        benchmarks=application.state.benchmark_repository,
        mathematics=application.state.mathematical_repository,
        data=application.state.data_repository,
        store=file_store,
        replayer=ScenarioReplayer(
            store=file_store,
            root=resolved.solver_sandbox_root,
            image=resolved.solver_sandbox_image,
            solver_router=solver_router,
        ),
        requirements=verification_requirement_registry.get,
    )
    application.state.benchmark_resource_cache = BenchmarkResourceCache(
        benchmark_cache_root,
        max_resource_bytes=resolved.benchmark_max_resource_bytes,
        timeout_seconds=resolved.benchmark_download_timeout_seconds,
    )
    try:
        code_identity = resolve_git_identity(benchmark_repository_root)
        benchmark_commit = resolved.benchmark_code_commit or code_identity.commit
        benchmark_source_tree_digest = code_identity.source_tree_digest
        benchmark_working_tree_dirty = (
            code_identity.working_tree_dirty or benchmark_commit != code_identity.commit
        )
    except (OSError, ValueError, subprocess.SubprocessError):
        if resolved.benchmark_code_commit is None:
            raise
        benchmark_commit = resolved.benchmark_code_commit
        benchmark_source_tree_digest = sha256_json(
            {"declared_commit": benchmark_commit, "source_tree": "UNAVAILABLE"}
        )
        benchmark_working_tree_dirty = True
    application.state.benchmark_workflow = BenchmarkWorkflow(
        independent_verifier=application.state.independent_verification.view,
        independent_runner=application.state.independent_verification.prepare_and_run,
        repository=application.state.benchmark_repository,
        manifests=application.state.benchmark_manifest_registry,
        resource_cache=application.state.benchmark_resource_cache,
        inspector=BenchmarkStructureInspector(),
        providers=application.state.providers,
        literature_source=literature_source,
        profile=real_benchmark_profile,
        code_commit=benchmark_commit,
        source_tree_digest=benchmark_source_tree_digest,
        working_tree_dirty=benchmark_working_tree_dirty,
        output_root=benchmark_artifact_root,
        case_executor=PipelineBenchmarkExecutor(
            reasoning_repository=application.state.reasoning_repository,
            reasoning_workflow=application.state.reasoning_workflow,
            data_workflow=application.state.data_execution_workflow,
            mathematical_workflow=application.state.mathematical_workflow,
            verification_workflow=application.state.verification_workflow,
            paper_workflow=application.state.paper_workflow,
            final_workflow=application.state.benchmark_final_submission_workflow,
            benchmark_repository=application.state.benchmark_repository,
            reviewed_models=verification_requirement_registry,
            independent_runner=application.state.independent_verification.prepare_and_run,
            independent_verifier=application.state.independent_verification.view,
            causal_evaluator=CausalBenchmarkEvaluator(
                store=file_store,
                repository=application.state.data_repository,
                mathematics=application.state.mathematical_repository,
                root=resolved.solver_sandbox_root,
                image=resolved.solver_sandbox_image,
                limits=sandbox_limits,
            ),
        ),
        provider_configurations=application.state.provider_configurations,
    )
    application.include_router(health_router)
    application.include_router(system_router)
    application.include_router(reasoning_router)
    application.include_router(data_execution_router)
    application.include_router(mathematical_router)
    application.include_router(verification_router)
    application.include_router(paper_router)
    application.include_router(final_submission_router)
    application.include_router(benchmarks_router)
    application.include_router(independent_verification_router)
    application.include_router(providers_router)

    @application.exception_handler(ResourceNotFoundError)
    async def not_found_handler(_request: Request, exc: ResourceNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": redact_sensitive_text(str(exc))})

    @application.exception_handler(RequestValidationError)
    async def request_validation_handler(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422, content={"detail": _safe_validation_errors(exc.errors())}
        )

    @application.exception_handler(ValidationError)
    async def pydantic_validation_handler(_request: Request, exc: ValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422, content={"detail": _safe_validation_errors(exc.errors())}
        )

    @application.exception_handler(ModelDiscoveryError)
    async def model_discovery_error_handler(
        _request: Request, exc: ModelDiscoveryError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "detail": {
                    "code": exc.code,
                    "message": redact_sensitive_text(str(exc)),
                }
            },
        )

    @application.exception_handler(MathModelError)
    async def mathmodel_error_handler(_request: Request, exc: MathModelError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": redact_sensitive_text(str(exc))})

    return application
