from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from mathmodel_ai.agents import ModelExplorer, ModelJury, ProblemAgent
from mathmodel_ai.api.routes.health import router as health_router
from mathmodel_ai.api.routes.reasoning import router as reasoning_router
from mathmodel_ai.api.routes.system import router as system_router
from mathmodel_ai.core.config import Settings, get_settings
from mathmodel_ai.core.errors import MathModelError, ResourceNotFoundError
from mathmodel_ai.core.logging import configure_logging
from mathmodel_ai.db.session import create_database_engine, create_session_factory
from mathmodel_ai.providers.factory import ProviderRegistry, build_provider_registry
from mathmodel_ai.reasoning.prompts import PromptRegistry
from mathmodel_ai.reasoning.repository import ReasoningRepository
from mathmodel_ai.reasoning.workflow import ReasoningWorkflow
from mathmodel_ai.routing.router import ModelRouter
from mathmodel_ai.schemas.model_selection import ModelJuryWeights


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
    application.state.settings = resolved
    application.state.engine = create_database_engine(resolved)
    application.state.providers = providers or build_provider_registry(resolved)
    application.state.session_factory = create_session_factory(application.state.engine)
    application.state.reasoning_repository = ReasoningRepository(application.state.session_factory)
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
    application.include_router(health_router)
    application.include_router(system_router)
    application.include_router(reasoning_router)

    @application.exception_handler(ResourceNotFoundError)
    async def not_found_handler(_request: Request, exc: ResourceNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @application.exception_handler(MathModelError)
    async def mathmodel_error_handler(_request: Request, exc: MathModelError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    return application


app = create_app()
