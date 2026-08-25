from fastapi import APIRouter, Request

from mathmodel_ai.api.schemas import SystemInfoResponse
from mathmodel_ai.core.config import Settings
from mathmodel_ai.providers.factory import ProviderRegistry

router = APIRouter(prefix="/api/v1", tags=["system"])


@router.get("/system", response_model=SystemInfoResponse)
def system_info(request: Request) -> SystemInfoResponse:
    settings: Settings = request.app.state.settings
    providers: ProviderRegistry = request.app.state.providers
    return SystemInfoResponse(
        name=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
        default_provider=settings.default_provider,
        configured_providers=sorted(providers.available, key=lambda provider: provider.value),
    )
