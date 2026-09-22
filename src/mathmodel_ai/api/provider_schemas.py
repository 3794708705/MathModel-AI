from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from mathmodel_ai.routing.schemas import RouteDecision, TaskProfile
from mathmodel_ai.schemas.provider_registry import (
    CREDENTIAL_REF_PATTERN,
    AgentRoutePolicy,
    CapabilityEvidence,
    CapabilityProbeResult,
    EndpointTrustLevel,
    ModelCapability,
    ModelPricing,
    ModelProfile,
    ProviderEndpoint,
    ProviderHealthStatus,
    ProviderProtocol,
    QualityTier,
    QualityTierSource,
    StructuredOutputStrategy,
    ToolCallingStrategy,
)


class ProviderCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{1,99}$")
    display_name: str = Field(min_length=1, max_length=255)
    protocol: ProviderProtocol
    base_url: str = Field(min_length=1, max_length=2048)
    credential_ref: str | None = Field(default=None, pattern=CREDENTIAL_REF_PATTERN)
    enabled: bool = True
    trust_level: EndpointTrustLevel = EndpointTrustLevel.USER_MANAGED_PROXY
    timeout_seconds: float = Field(default=60.0, gt=0, le=600)
    connect_timeout_seconds: float = Field(default=10.0, gt=0, le=120)
    max_response_bytes: int = Field(default=8 * 1024 * 1024, ge=1024, le=64 * 1024 * 1024)
    verify_tls: bool = True
    allow_redirects: bool = False
    additional_headers: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_endpoint(self) -> ProviderEndpoint:
        return ProviderEndpoint(**self.model_dump())


class ProviderPatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    display_name: str | None = Field(default=None, min_length=1, max_length=255)
    protocol: ProviderProtocol | None = None
    base_url: str | None = Field(default=None, min_length=1, max_length=2048)
    credential_ref: str | None = Field(default=None, pattern=CREDENTIAL_REF_PATTERN)
    enabled: bool | None = None
    trust_level: EndpointTrustLevel | None = None
    timeout_seconds: float | None = Field(default=None, gt=0, le=600)
    connect_timeout_seconds: float | None = Field(default=None, gt=0, le=120)
    max_response_bytes: int | None = Field(default=None, ge=1024, le=64 * 1024 * 1024)
    verify_tls: bool | None = None
    allow_redirects: bool | None = None
    additional_headers: dict[str, str] | None = None
    metadata: dict[str, Any] | None = None


class ProviderEndpointView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider_id: str
    display_name: str
    protocol: ProviderProtocol
    base_url: str
    enabled: bool
    trust_level: EndpointTrustLevel
    timeout_seconds: float
    connect_timeout_seconds: float
    max_response_bytes: int
    verify_tls: bool
    allow_redirects: bool
    additional_headers: dict[str, str]
    metadata: dict[str, Any]
    health_status: ProviderHealthStatus
    consecutive_failures: int
    cooldown_until: datetime | None
    config_digest: str
    created_at: datetime
    updated_at: datetime
    credential_configured: bool


class ModelCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{1,99}$")
    provider_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{1,99}$")
    display_name: str = Field(min_length=1, max_length=255)
    remote_model: str = Field(min_length=1, max_length=500)
    enabled: bool = True
    quality_tier: QualityTier = QualityTier.BALANCED
    quality_tier_source: QualityTierSource = QualityTierSource.USER_DECLARED
    declared_capabilities: dict[ModelCapability, CapabilityEvidence] = Field(default_factory=dict)
    structured_output_strategy: StructuredOutputStrategy = StructuredOutputStrategy.UNSUPPORTED
    tool_calling_strategy: ToolCallingStrategy = ToolCallingStrategy.NO_TOOLS
    reasoning_mapping: dict[str, str] = Field(default_factory=dict)
    context_window: int | None = Field(default=None, ge=1)
    max_output_tokens: int | None = Field(default=None, ge=1)
    pricing: ModelPricing = Field(default_factory=ModelPricing)
    trust_level: EndpointTrustLevel = EndpointTrustLevel.UNKNOWN
    configuration: dict[str, Any] = Field(default_factory=dict)

    def to_profile(self) -> ModelProfile:
        return ModelProfile(**self.model_dump())


class ModelPatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    display_name: str | None = Field(default=None, min_length=1, max_length=255)
    remote_model: str | None = Field(default=None, min_length=1, max_length=500)
    enabled: bool | None = None
    quality_tier: QualityTier | None = None
    quality_tier_source: QualityTierSource | None = None
    declared_capabilities: dict[ModelCapability, CapabilityEvidence] | None = None
    structured_output_strategy: StructuredOutputStrategy | None = None
    tool_calling_strategy: ToolCallingStrategy | None = None
    reasoning_mapping: dict[str, str] | None = None
    context_window: int | None = Field(default=None, ge=1)
    max_output_tokens: int | None = Field(default=None, ge=1)
    pricing: ModelPricing | None = None
    trust_level: EndpointTrustLevel | None = None
    configuration: dict[str, Any] | None = None


class RoutingOverview(BaseModel):
    default_model_id: str | None
    legacy_default_provider: str
    legacy_default_model: str
    legacy_config_deprecated: bool = True
    registered_model_count: int = Field(ge=0)
    agent_route_count: int = Field(ge=0)


class DefaultModelPutRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{1,99}$")


class CredentialPutRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    api_key: SecretStr = Field(min_length=1, max_length=65_536)


class CredentialStatusResponse(BaseModel):
    credential_configured: bool


class ModelDiscoveryCandidateView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    remote_model_id: str = Field(min_length=1, max_length=500)
    display_name: str | None = Field(default=None, max_length=255)


class ModelDiscoveryResponse(BaseModel):
    provider_id: str
    models: list[ModelDiscoveryCandidateView]
    capabilities_probed: bool = False


class ProviderConnectionTestResponse(BaseModel):
    provider_id: str
    connected: bool = True
    credential_accepted: bool = True
    model_discovery_supported: bool


class RoutableAgentView(BaseModel):
    name: str
    role: str
    task_profile: TaskProfile
    configured_model_id: str | None = None


class RoutingPreviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile: TaskProfile
    agent_name: str | None = Field(default=None, max_length=100)


class RoutingPreviewResponse(BaseModel):
    decision: RouteDecision
    provider_called: bool = False


class AgentRoutePutRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    primary_model_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{1,99}$")
    fallback_model_ids: list[str] = Field(default_factory=list, max_length=10)


class ProviderPresetView(BaseModel):
    preset_id: str
    display_name: str
    protocol: ProviderProtocol
    base_url: str | None
    recommended_credential_ref: str
    trust_level: EndpointTrustLevel
    credential_type: str
    model_hints: list[str] = Field(default_factory=list)
    capability_hints: dict[ModelCapability, CapabilityEvidence] = Field(default_factory=dict)


class ProviderHealthView(BaseModel):
    provider_id: str
    status: ProviderHealthStatus
    credential_configured: bool
    latest_probe_at: datetime | None = None
    probes: list[CapabilityProbeResult] = Field(default_factory=list)


AgentRouteResponse = AgentRoutePolicy
