from fastapi import APIRouter, Query, Request

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
from mathmodel_ai.api.provider_schemas import (
    AgentRoutePutRequest,
    CredentialPutRequest,
    CredentialStatusResponse,
    DefaultModelPutRequest,
    ModelCreateRequest,
    ModelDiscoveryCandidateView,
    ModelDiscoveryResponse,
    ModelPatchRequest,
    ProviderConnectionTestResponse,
    ProviderCreateRequest,
    ProviderEndpointView,
    ProviderPatchRequest,
    ProviderPresetView,
    RoutableAgentView,
    RoutingOverview,
    RoutingPreviewRequest,
    RoutingPreviewResponse,
)
from mathmodel_ai.core.errors import ConfigurationError, ModelDiscoveryError
from mathmodel_ai.providers.discovery import ProviderModelDiscovery
from mathmodel_ai.providers.presets import PROVIDER_PRESETS
from mathmodel_ai.providers.probe import ProviderCompatibilityProbe
from mathmodel_ai.providers.repository import DEFAULT_ROUTE_POLICY_NAME, ProviderModelRegistry
from mathmodel_ai.providers.secrets import (
    BaseSecretStore,
    new_provider_secret_reference,
    stored_secret_id,
)
from mathmodel_ai.routing.router import ModelRouter
from mathmodel_ai.routing.schemas import RouteAction, TaskProfile, TaskType
from mathmodel_ai.schemas.provider_registry import (
    AgentRoutePolicy,
    CapabilityProbeResult,
    ModelProfile,
    ProviderEndpoint,
)

router = APIRouter(prefix="/api/v1", tags=["provider-registry"])

ROUTABLE_AGENT_TYPES = (
    ProblemAgent,
    DataAgent,
    LiteratureAgent,
    ModelExplorer,
    ModelJury,
    MathModeler,
    CodeAgent,
    CitationAgent,
    PaperAgent,
    PaperFactualAuditAgent,
    RedTeamAgent,
    ModelRepairAgent,
    FinalJuryAgent,
)
ROUTABLE_AGENT_TASK_TYPES = {
    ProblemAgent.name: TaskType.PROBLEM_UNDERSTANDING,
    DataAgent.name: TaskType.DATA_UNDERSTANDING,
    LiteratureAgent.name: TaskType.PAPER_IR,
    ModelExplorer.name: TaskType.MODEL_EXPLORATION,
    ModelJury.name: TaskType.MODEL_JURY,
    MathModeler.name: TaskType.MATHEMATICAL_MODELING,
    CodeAgent.name: TaskType.CODE_GENERATION,
    CitationAgent.name: TaskType.CITATION_VERIFICATION,
    PaperAgent.name: TaskType.PAPER_IR,
    PaperFactualAuditAgent.name: TaskType.FINAL_ACCEPTANCE,
    RedTeamAgent.name: TaskType.RED_TEAM,
    ModelRepairAgent.name: TaskType.MODEL_REPAIR,
    FinalJuryAgent.name: TaskType.FINAL_ACCEPTANCE,
}


def _services(
    request: Request,
) -> tuple[ProviderModelRegistry, ProviderCompatibilityProbe, ModelRouter]:
    return (
        request.app.state.provider_configurations,
        request.app.state.provider_probe,
        request.app.state.model_router,
    )


def _provider_view(
    registry: ProviderModelRegistry,
    provider_id: str,
    *,
    endpoint: ProviderEndpoint | None = None,
) -> ProviderEndpointView:
    endpoint = registry.get_provider_for_management(provider_id) if endpoint is None else endpoint
    public = endpoint.model_dump(exclude={"credential_ref"})
    credential_configured = registry.credential_configured_for_management(provider_id)
    if endpoint.enabled and not registry.runtime_policy_allows(endpoint):
        public["health_status"] = "UNAVAILABLE"
    elif endpoint.enabled and not credential_configured:
        public["health_status"] = "UNCONFIGURED"
    return ProviderEndpointView(
        **public,
        credential_configured=credential_configured,
    )


def _routing_overview(request: Request, registry: ProviderModelRegistry) -> RoutingOverview:
    settings = request.app.state.settings
    routes = [
        route
        for route in registry.list_agent_routes()
        if route.agent_name != DEFAULT_ROUTE_POLICY_NAME
    ]
    return RoutingOverview(
        default_model_id=registry.get_default_model_id() or settings.default_model_id,
        legacy_default_provider=settings.default_provider.value,
        legacy_default_model=settings.default_provider_model,
        registered_model_count=len(registry.list_models()),
        agent_route_count=len(routes),
    )


@router.get("/providers/presets", response_model=list[ProviderPresetView])
def list_provider_presets() -> list[ProviderPresetView]:
    return [
        ProviderPresetView(
            preset_id=item.preset_id,
            display_name=item.display_name,
            protocol=item.protocol,
            base_url=item.base_url,
            recommended_credential_ref=item.recommended_credential_ref,
            trust_level=item.trust_level,
            credential_type=item.credential_type,
            model_hints=list(item.model_hints),
            capability_hints=item.capability_hints,
        )
        for item in PROVIDER_PRESETS.values()
    ]


@router.get("/providers", response_model=list[ProviderEndpointView])
def list_providers(request: Request) -> list[ProviderEndpointView]:
    registry, _, _ = _services(request)
    return [
        _provider_view(registry, item.provider_id, endpoint=item)
        for item in registry.list_providers_for_management()
    ]


@router.post("/providers", response_model=ProviderEndpointView, status_code=201)
def create_provider(payload: ProviderCreateRequest, request: Request) -> ProviderEndpointView:
    registry, _, _ = _services(request)
    endpoint = registry.create_provider(payload.to_endpoint())
    return _provider_view(registry, endpoint.provider_id, endpoint=endpoint)


@router.get("/providers/{provider_id}", response_model=ProviderEndpointView)
def get_provider(provider_id: str, request: Request) -> ProviderEndpointView:
    registry, _, _ = _services(request)
    return _provider_view(registry, provider_id)


@router.patch("/providers/{provider_id}", response_model=ProviderEndpointView)
def patch_provider(
    provider_id: str, payload: ProviderPatchRequest, request: Request
) -> ProviderEndpointView:
    registry, _, _ = _services(request)
    endpoint = registry.update_provider(provider_id, payload.model_dump(exclude_unset=True))
    return _provider_view(registry, provider_id, endpoint=endpoint)


@router.put(
    "/providers/{provider_id}/credential",
    response_model=CredentialStatusResponse,
)
def put_provider_credential(
    provider_id: str,
    payload: CredentialPutRequest,
    request: Request,
) -> CredentialStatusResponse:
    registry, _, _ = _services(request)
    store: BaseSecretStore = request.app.state.secret_store
    if not store.available:
        raise ConfigurationError(
            "encrypted credential storage is unavailable; configure MM_SECRET_MASTER_KEY"
        )
    current = registry.get_provider(provider_id)
    secret_id, credential_ref = new_provider_secret_reference(provider_id)
    store.put(secret_id, payload.api_key)
    try:
        registry.update_provider(provider_id, {"credential_ref": credential_ref})
    except Exception:
        store.delete(secret_id)
        raise
    if current.credential_ref is not None and current.credential_ref.startswith("secret:"):
        store.delete(stored_secret_id(current.credential_ref))
    return CredentialStatusResponse(credential_configured=True)


@router.delete(
    "/providers/{provider_id}/credential",
    response_model=CredentialStatusResponse,
)
def delete_provider_credential(
    provider_id: str,
    request: Request,
) -> CredentialStatusResponse:
    registry, _, _ = _services(request)
    store: BaseSecretStore = request.app.state.secret_store
    current = registry.get_provider(provider_id)
    registry.update_provider(provider_id, {"credential_ref": None})
    if current.credential_ref is not None and current.credential_ref.startswith("secret:"):
        store.delete(stored_secret_id(current.credential_ref))
    return CredentialStatusResponse(credential_configured=False)


@router.post(
    "/providers/{provider_id}/discover-models",
    response_model=ModelDiscoveryResponse,
)
async def discover_provider_models(
    provider_id: str,
    request: Request,
) -> ModelDiscoveryResponse:
    registry, _, _ = _services(request)
    endpoint = registry.get_provider(provider_id)
    discovery: ProviderModelDiscovery = request.app.state.provider_model_discovery
    models = await discovery.discover(endpoint)
    return ModelDiscoveryResponse(
        provider_id=provider_id,
        models=[
            ModelDiscoveryCandidateView(
                remote_model_id=item.remote_model_id,
                display_name=item.display_name,
            )
            for item in models
        ],
    )


@router.post(
    "/providers/{provider_id}/test-connection",
    response_model=ProviderConnectionTestResponse,
)
async def test_provider_connection(
    provider_id: str,
    request: Request,
) -> ProviderConnectionTestResponse:
    registry, _, _ = _services(request)
    endpoint = registry.get_provider(provider_id)
    discovery: ProviderModelDiscovery = request.app.state.provider_model_discovery
    try:
        await discovery.discover(endpoint)
    except ModelDiscoveryError as exc:
        if exc.code != "MODEL_DISCOVERY_UNSUPPORTED":
            raise
        if not exc.provider_reached:
            raise ModelDiscoveryError(
                "MODEL_DISCOVERY_UNSUPPORTED",
                "Connection testing is unavailable for this protocol. Probe a saved model instead.",
                status_code=409,
            ) from exc
        return ProviderConnectionTestResponse(
            provider_id=provider_id,
            model_discovery_supported=False,
        )
    return ProviderConnectionTestResponse(
        provider_id=provider_id,
        model_discovery_supported=True,
    )


@router.post("/providers/{provider_id}/probe", response_model=list[CapabilityProbeResult])
async def probe_provider(provider_id: str, request: Request) -> list[CapabilityProbeResult]:
    registry, probe, _ = _services(request)
    registry.get_provider(provider_id)
    return [
        await probe.run(model.model_id)
        for model in registry.list_models(provider_id=provider_id)
        if model.enabled
    ]


@router.post("/providers/{provider_id}/enable", response_model=ProviderEndpointView)
def enable_provider(provider_id: str, request: Request) -> ProviderEndpointView:
    registry, _, _ = _services(request)
    registry.set_provider_enabled(provider_id, True)
    return _provider_view(registry, provider_id)


@router.post("/providers/{provider_id}/disable", response_model=ProviderEndpointView)
def disable_provider(provider_id: str, request: Request) -> ProviderEndpointView:
    registry, _, _ = _services(request)
    registry.set_provider_enabled(provider_id, False)
    return _provider_view(registry, provider_id)


@router.get("/models", response_model=list[ModelProfile])
def list_models(
    request: Request, provider_id: str | None = Query(default=None)
) -> list[ModelProfile]:
    registry, _, _ = _services(request)
    return registry.list_models(provider_id=provider_id)


@router.post("/models", response_model=ModelProfile, status_code=201)
def create_model(payload: ModelCreateRequest, request: Request) -> ModelProfile:
    registry, _, _ = _services(request)
    return registry.create_model(payload.to_profile())


@router.get("/models/{model_id}", response_model=ModelProfile)
def get_model(model_id: str, request: Request) -> ModelProfile:
    registry, _, _ = _services(request)
    return registry.get_model(model_id)


@router.patch("/models/{model_id}", response_model=ModelProfile)
def patch_model(model_id: str, payload: ModelPatchRequest, request: Request) -> ModelProfile:
    registry, _, _ = _services(request)
    return registry.update_model(model_id, payload.model_dump(exclude_unset=True))


@router.post("/models/{model_id}/probe", response_model=CapabilityProbeResult)
async def probe_model(model_id: str, request: Request) -> CapabilityProbeResult:
    _, probe, _ = _services(request)
    return await probe.run(model_id)


@router.get("/models/{model_id}/probe", response_model=CapabilityProbeResult | None)
def get_latest_model_probe(model_id: str, request: Request) -> CapabilityProbeResult | None:
    registry, _, _ = _services(request)
    registry.get_model(model_id)
    return registry.latest_probe(model_id, current_only=True)


@router.get("/model-routing", response_model=RoutingOverview)
def routing_overview(request: Request) -> RoutingOverview:
    registry, _, _ = _services(request)
    return _routing_overview(request, registry)


@router.put("/model-routing/default", response_model=RoutingOverview)
def put_default_model(
    payload: DefaultModelPutRequest,
    request: Request,
) -> RoutingOverview:
    registry, _, _ = _services(request)
    registry.put_default_model(payload.model_id)
    return _routing_overview(request, registry)


@router.get("/model-routing/agents", response_model=list[AgentRoutePolicy])
def list_agent_routes(request: Request) -> list[AgentRoutePolicy]:
    registry, _, _ = _services(request)
    return [
        route
        for route in registry.list_agent_routes()
        if route.agent_name != DEFAULT_ROUTE_POLICY_NAME
    ]


@router.get("/model-routing/agent-catalog", response_model=list[RoutableAgentView])
def list_routable_agents(request: Request) -> list[RoutableAgentView]:
    registry, _, _ = _services(request)
    configured = {route.agent_name: route for route in registry.list_agent_routes()}
    return [
        RoutableAgentView(
            name=agent_type.name,
            role=agent_type.role,
            task_profile=TaskProfile(task_type=ROUTABLE_AGENT_TASK_TYPES[agent_type.name]),
            configured_model_id=(
                configured[agent_type.name].primary_model_id
                if agent_type.name in configured
                else None
            ),
        )
        for agent_type in ROUTABLE_AGENT_TYPES
    ]


@router.put("/model-routing/agents/{agent_name}", response_model=AgentRoutePolicy)
def put_agent_route(
    agent_name: str, payload: AgentRoutePutRequest, request: Request
) -> AgentRoutePolicy:
    registry, _, model_router = _services(request)
    task_type = ROUTABLE_AGENT_TASK_TYPES.get(agent_name)
    if task_type is None:
        raise ConfigurationError(f"agent {agent_name!r} is not routable")
    for model_id in [payload.primary_model_id, *payload.fallback_model_ids]:
        decision = model_router.route(
            TaskProfile(
                task_type=task_type,
                preferred_model_id=model_id,
                allow_model_fallback=False,
            )
        )
        if decision.action is not RouteAction.EXECUTE or decision.selected_model_id != model_id:
            raise ConfigurationError(f"route policy rejected model {model_id!r}: {decision.reason}")
    return registry.put_agent_route(AgentRoutePolicy(agent_name=agent_name, **payload.model_dump()))


@router.post("/model-routing/preview", response_model=RoutingPreviewResponse)
def preview_route(payload: RoutingPreviewRequest, request: Request) -> RoutingPreviewResponse:
    _, _, model_router = _services(request)
    return RoutingPreviewResponse(
        decision=model_router.route(payload.profile, agent_name=payload.agent_name)
    )
