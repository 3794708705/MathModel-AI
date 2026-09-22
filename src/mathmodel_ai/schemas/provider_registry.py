from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from urllib.parse import unquote, urlsplit, urlunsplit
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from mathmodel_ai.providers.schemas import ModelUsage

REGISTRY_ID_PATTERN = r"^[a-z0-9][a-z0-9._-]{1,99}$"
CREDENTIAL_REF_PATTERN = (
    r"^(?:env:[A-Z_][A-Z0-9_]{1,126}|"
    r"secret:provider/[a-z0-9][a-z0-9._-]{1,99}/[a-f0-9]{12})$"
)
_SENSITIVE_KEY = re.compile(
    r"(^|[_-])(api[_-]?key|secret|password|credential|authorization|cookie|token)($|[_-])",
    re.IGNORECASE,
)
_CREDENTIAL_VALUE = re.compile(
    r"(?:sk-[A-Za-z0-9_-]{12,}|Bearer\s+[A-Za-z0-9._~+/-]{8,}|-----BEGIN [A-Z ]+PRIVATE KEY-----)",
    re.IGNORECASE,
)
_SAFE_TOKEN_FIELD_NAMES = {
    "max_tokens",
    "input_tokens",
    "output_tokens",
    "cached_input_tokens",
    "reasoning_tokens",
    "total_tokens",
}


class ProviderProtocol(StrEnum):
    OPENAI_CHAT_COMPLETIONS = "openai_chat_completions"
    OPENAI_RESPONSES = "openai_responses"
    ANTHROPIC_MESSAGES = "anthropic_messages"
    GOOGLE_GENERATE_CONTENT = "google_generate_content"
    CUSTOM_JSON_HTTP = "custom_json_http"


class EndpointTrustLevel(StrEnum):
    OFFICIAL_VENDOR = "OFFICIAL_VENDOR"
    USER_MANAGED_PROXY = "USER_MANAGED_PROXY"
    LOCAL_ENDPOINT = "LOCAL_ENDPOINT"
    UNKNOWN = "UNKNOWN"


class ModelIdentityConfidence(StrEnum):
    VENDOR_ENDPOINT = "VENDOR_ENDPOINT"
    NOT_INDEPENDENTLY_VERIFIED = "NOT_INDEPENDENTLY_VERIFIED"
    UNKNOWN = "UNKNOWN"


class ProviderHealthStatus(StrEnum):
    UNCONFIGURED = "UNCONFIGURED"
    READY = "READY"
    DEGRADED = "DEGRADED"
    AUTH_FAILED = "AUTH_FAILED"
    UNAVAILABLE = "UNAVAILABLE"
    DISABLED = "DISABLED"


class ModelCapability(StrEnum):
    TEXT = "TEXT"
    VISION = "VISION"
    STRUCTURED_OUTPUT = "STRUCTURED_OUTPUT"
    JSON_MODE = "JSON_MODE"
    JSON_SCHEMA = "JSON_SCHEMA"
    TOOLS = "TOOLS"
    STREAMING = "STREAMING"
    REASONING_CONTROL = "REASONING_CONTROL"
    SYSTEM_ROLE = "SYSTEM_ROLE"
    DEVELOPER_ROLE = "DEVELOPER_ROLE"
    LONG_CONTEXT = "LONG_CONTEXT"
    USAGE_REPORTING = "USAGE_REPORTING"


class CapabilityStatus(StrEnum):
    SUPPORTED = "SUPPORTED"
    PARTIAL = "PARTIAL"
    UNSUPPORTED = "UNSUPPORTED"
    UNKNOWN = "UNKNOWN"


class CapabilitySource(StrEnum):
    BUILTIN_VERIFIED = "BUILTIN_VERIFIED"
    USER_DECLARED = "USER_DECLARED"
    PROBED = "PROBED"
    UNKNOWN = "UNKNOWN"


class QualityTier(StrEnum):
    ROUTINE = "ROUTINE"
    BALANCED = "BALANCED"
    FLAGSHIP_HIGH = "FLAGSHIP_HIGH"
    FLAGSHIP_XHIGH = "FLAGSHIP_XHIGH"
    FLAGSHIP_MAX = "FLAGSHIP_MAX"


class QualityTierSource(StrEnum):
    USER_DECLARED = "USER_DECLARED"
    BUILTIN = "BUILTIN"
    BENCHMARKED = "BENCHMARKED"


class StructuredOutputStrategy(StrEnum):
    NATIVE_JSON_SCHEMA = "NATIVE_JSON_SCHEMA"
    JSON_MODE = "JSON_MODE"
    PROMPT_JSON_FALLBACK = "PROMPT_JSON_FALLBACK"
    UNSUPPORTED = "UNSUPPORTED"


class ToolCallingStrategy(StrEnum):
    NATIVE_TOOLS = "NATIVE_TOOLS"
    NO_TOOLS = "NO_TOOLS"


class CustomAuthScheme(StrEnum):
    BEARER = "BEARER"
    API_KEY_HEADER = "API_KEY_HEADER"


class ProbeAuthenticationStatus(StrEnum):
    PASS = "PASS"
    AUTH_FAILED = "AUTH_FAILED"
    NOT_CONFIGURED = "NOT_CONFIGURED"
    UNAVAILABLE = "UNAVAILABLE"


class PricingSource(StrEnum):
    USER_CONFIGURED = "USER_CONFIGURED"
    BUILTIN_VERIFIED = "BUILTIN_VERIFIED"
    UNKNOWN = "UNKNOWN"


class CapabilityEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: CapabilityStatus = CapabilityStatus.UNKNOWN
    source: CapabilitySource = CapabilitySource.UNKNOWN
    detail: str | None = Field(default=None, max_length=500)


class ModelPricing(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_per_million: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    output_per_million: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    cached_input_per_million: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    currency: str | None = Field(default=None, pattern=r"^[A-Z]{3}$")
    effective_from: datetime | None = None
    source: PricingSource = PricingSource.UNKNOWN


class CustomJSONRequestMapping(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: str = Field(default="POST", pattern=r"^POST$")
    endpoint_path: str = Field(default="/generate", min_length=1, max_length=500)
    body: dict[str, Any] = Field(default_factory=lambda: {"model": "{model}", "prompt": "{prompt}"})

    @field_validator("endpoint_path")
    @classmethod
    def safe_path(cls, value: str) -> str:
        decoded = value
        for _ in range(3):
            unquoted = unquote(decoded)
            if unquoted == decoded:
                break
            decoded = unquoted
        path_segments = decoded.replace("\\", "/").split("/")
        if (
            not value.startswith("/")
            or value.startswith("//")
            or decoded.startswith("//")
            or any(segment in {".", ".."} for segment in path_segments)
            or "\\" in decoded
            or "://" in decoded
            or "?" in decoded
            or "#" in decoded
            or any(ord(character) < 32 for character in decoded)
        ):
            raise ValueError("custom endpoint_path must be an absolute URL path without query")
        return value

    @field_validator("body")
    @classmethod
    def safe_template(cls, value: dict[str, Any]) -> dict[str, Any]:
        _validate_safe_mapping(
            value,
            allowed_placeholders={
                "model",
                "system",
                "messages",
                "prompt",
                "max_tokens",
            },
        )
        return value


class CustomJSONResponseMapping(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text_path: str = Field(min_length=1, max_length=300)
    input_tokens_path: str | None = Field(default=None, max_length=300)
    output_tokens_path: str | None = Field(default=None, max_length=300)
    cached_input_tokens_path: str | None = Field(default=None, max_length=300)
    reasoning_tokens_path: str | None = Field(default=None, max_length=300)
    total_tokens_path: str | None = Field(default=None, max_length=300)

    @field_validator(
        "text_path",
        "input_tokens_path",
        "output_tokens_path",
        "cached_input_tokens_path",
        "reasoning_tokens_path",
        "total_tokens_path",
    )
    @classmethod
    def json_field_path(cls, value: str | None) -> str | None:
        if value is not None and not re.fullmatch(r"[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*", value):
            raise ValueError("response paths may contain only dotted JSON field names")
        return value


class CustomJSONConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request: CustomJSONRequestMapping
    response: CustomJSONResponseMapping
    auth_scheme: CustomAuthScheme = CustomAuthScheme.BEARER
    api_key_header: str | None = Field(default=None, max_length=100)

    @model_validator(mode="after")
    def api_key_header_required(self) -> CustomJSONConfiguration:
        if self.auth_scheme is CustomAuthScheme.API_KEY_HEADER and not self.api_key_header:
            raise ValueError("api_key_header is required for API_KEY_HEADER auth")
        if self.api_key_header and not re.fullmatch(r"[A-Za-z0-9-]+", self.api_key_header):
            raise ValueError("api_key_header contains invalid characters")
        if self.api_key_header and self.api_key_header.casefold() in {
            "authorization",
            "proxy-authorization",
            "cookie",
            "host",
            "content-length",
            "connection",
        }:
            raise ValueError("api_key_header is reserved")
        return self


class ProviderEndpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider_id: str = Field(pattern=REGISTRY_ID_PATTERN)
    display_name: str = Field(min_length=1, max_length=255)
    protocol: ProviderProtocol
    base_url: str = Field(min_length=1, max_length=2048)
    credential_ref: str | None = Field(default=None, pattern=CREDENTIAL_REF_PATTERN)
    enabled: bool = True
    trust_level: EndpointTrustLevel = EndpointTrustLevel.UNKNOWN
    timeout_seconds: float = Field(default=60.0, gt=0, le=600)
    connect_timeout_seconds: float = Field(default=10.0, gt=0, le=120)
    max_response_bytes: int = Field(default=8 * 1024 * 1024, ge=1024, le=64 * 1024 * 1024)
    verify_tls: bool = True
    allow_redirects: bool = False
    additional_headers: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    health_status: ProviderHealthStatus = ProviderHealthStatus.UNCONFIGURED
    consecutive_failures: int = Field(default=0, ge=0)
    cooldown_until: datetime | None = None
    config_digest: str = Field(default="", pattern=r"^$|^[a-f0-9]{64}$")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("base_url")
    @classmethod
    def canonical_base_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if not parsed.scheme or not parsed.hostname:
            raise ValueError("base_url must be an absolute URL")
        if parsed.username or parsed.password:
            raise ValueError("base_url must not contain credentials")
        if parsed.query or parsed.fragment:
            raise ValueError("base_url must not contain query or fragment data")
        path = parsed.path.rstrip("/")
        host = parsed.hostname.lower()
        host = f"[{host}]" if ":" in host else host
        port = f":{parsed.port}" if parsed.port is not None else ""
        return urlunsplit((parsed.scheme.lower(), f"{host}{port}", path, "", ""))

    @field_validator("additional_headers")
    @classmethod
    def safe_headers(cls, value: dict[str, str]) -> dict[str, str]:
        forbidden = {
            "authorization",
            "host",
            "cookie",
            "content-length",
            "connection",
            "proxy-authorization",
            "x-api-key",
            "api-key",
            "x-auth-token",
        }
        normalized: dict[str, str] = {}
        for name, header_value in value.items():
            if not re.fullmatch(r"[A-Za-z0-9-]+", name) or name.lower() in forbidden:
                raise ValueError(f"reserved or invalid additional header: {name}")
            if "\r" in header_value or "\n" in header_value:
                raise ValueError("additional header values must not contain line breaks")
            if _CREDENTIAL_VALUE.search(header_value):
                raise ValueError("additional headers must not contain credentials")
            normalized[name] = header_value
        return normalized

    @field_validator("metadata")
    @classmethod
    def safe_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        _reject_sensitive_data(value)
        return value

    @model_validator(mode="after")
    def enforce_and_digest(self) -> ProviderEndpoint:
        if self.allow_redirects:
            raise ValueError("redirects are disabled for provider endpoints")
        if self.connect_timeout_seconds > self.timeout_seconds:
            raise ValueError("connect_timeout_seconds cannot exceed timeout_seconds")
        expected = compute_provider_config_digest(self)
        if self.config_digest and self.config_digest != expected:
            raise ValueError("provider config_digest does not match canonical configuration")
        self.config_digest = expected
        return self

    def digest_payload(self) -> dict[str, Any]:
        """Backward-compatible access to the canonical provider configuration."""

        return canonical_provider_config(self)


def canonical_provider_config(provider: ProviderEndpoint) -> dict[str, Any]:
    """Return the sole canonical input for ProviderEndpoint identity.

    Runtime health, probe/discovery results, timestamps, and resolved credential
    values are intentionally absent. ``credential_ref`` remains part of the
    configuration identity because it selects the server-side credential source.
    """

    return {
        "provider_id": provider.provider_id,
        "display_name": provider.display_name,
        "protocol": provider.protocol.value,
        "base_url": provider.base_url,
        "credential_ref": provider.credential_ref,
        "enabled": provider.enabled,
        "trust_level": provider.trust_level.value,
        "timeout_seconds": provider.timeout_seconds,
        "connect_timeout_seconds": provider.connect_timeout_seconds,
        "max_response_bytes": provider.max_response_bytes,
        "verify_tls": provider.verify_tls,
        "allow_redirects": provider.allow_redirects,
        "additional_headers": dict(provider.additional_headers),
        "metadata": dict(provider.metadata),
    }


def compute_provider_config_digest(provider: ProviderEndpoint) -> str:
    """Compute a ProviderEndpoint digest from its canonical configuration."""

    return _digest(canonical_provider_config(provider))


class ModelProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(pattern=REGISTRY_ID_PATTERN)
    provider_id: str = Field(pattern=REGISTRY_ID_PATTERN)
    display_name: str = Field(min_length=1, max_length=255)
    remote_model: str = Field(min_length=1, max_length=500)
    enabled: bool = True
    quality_tier: QualityTier = QualityTier.BALANCED
    quality_tier_source: QualityTierSource = QualityTierSource.USER_DECLARED
    declared_capabilities: dict[ModelCapability, CapabilityEvidence] = Field(default_factory=dict)
    observed_capabilities: dict[ModelCapability, CapabilityEvidence] = Field(default_factory=dict)
    effective_capabilities: dict[ModelCapability, CapabilityEvidence] = Field(default_factory=dict)
    structured_output_strategy: StructuredOutputStrategy = StructuredOutputStrategy.UNSUPPORTED
    tool_calling_strategy: ToolCallingStrategy = ToolCallingStrategy.NO_TOOLS
    reasoning_mapping: dict[str, str] = Field(default_factory=dict)
    context_window: int | None = Field(default=None, ge=1)
    max_output_tokens: int | None = Field(default=None, ge=1)
    pricing: ModelPricing = Field(default_factory=ModelPricing)
    trust_level: EndpointTrustLevel = EndpointTrustLevel.UNKNOWN
    configuration: dict[str, Any] = Field(default_factory=dict)
    config_digest: str = Field(default="", pattern=r"^$|^[a-f0-9]{64}$")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("configuration")
    @classmethod
    def safe_configuration(cls, value: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(value)
        custom_json = normalized.pop("custom_json", None)
        _reject_sensitive_data(normalized)
        if custom_json is not None:
            normalized["custom_json"] = CustomJSONConfiguration.model_validate(
                custom_json
            ).model_dump(mode="json")
        return normalized

    @field_validator("observed_capabilities")
    @classmethod
    def observed_sources_are_verified(
        cls, value: dict[ModelCapability, CapabilityEvidence]
    ) -> dict[ModelCapability, CapabilityEvidence]:
        if any(
            item.source not in {CapabilitySource.PROBED, CapabilitySource.BUILTIN_VERIFIED}
            for item in value.values()
        ):
            raise ValueError("observed capabilities must be PROBED or BUILTIN_VERIFIED")
        return value

    @model_validator(mode="after")
    def effective_and_digest(self) -> ModelProfile:
        effective: dict[ModelCapability, CapabilityEvidence] = {}
        for capability in ModelCapability:
            observed = self.observed_capabilities.get(capability)
            declared = self.declared_capabilities.get(capability)
            if observed is not None:
                effective[capability] = observed
            elif declared is not None and declared.source is CapabilitySource.BUILTIN_VERIFIED:
                effective[capability] = declared
            elif declared is not None:
                status = (
                    CapabilityStatus.PARTIAL
                    if declared.status is CapabilityStatus.SUPPORTED
                    else declared.status
                )
                effective[capability] = CapabilityEvidence(
                    status=status,
                    source=declared.source,
                    detail=declared.detail,
                )
            else:
                effective[capability] = CapabilityEvidence()
        self.effective_capabilities = effective
        expected = _digest(self.digest_payload())
        if self.config_digest and self.config_digest != expected:
            raise ValueError("model config_digest does not match canonical configuration")
        self.config_digest = expected
        return self

    def digest_payload(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "provider_id": self.provider_id,
            "display_name": self.display_name,
            "remote_model": self.remote_model,
            "enabled": self.enabled,
            "quality_tier": self.quality_tier.value,
            "quality_tier_source": self.quality_tier_source.value,
            "declared_capabilities": _capability_json(self.declared_capabilities),
            "structured_output_strategy": self.structured_output_strategy.value,
            "tool_calling_strategy": self.tool_calling_strategy.value,
            "reasoning_mapping": self.reasoning_mapping,
            "context_window": self.context_window,
            "max_output_tokens": self.max_output_tokens,
            "pricing": self.pricing.model_dump(mode="json"),
            "trust_level": self.trust_level.value,
            "configuration": self.configuration,
        }

    def supports(self, capability: ModelCapability, *, allow_partial: bool = False) -> bool:
        status = self.effective_capabilities[capability].status
        return status is CapabilityStatus.SUPPORTED or (
            allow_partial and status is CapabilityStatus.PARTIAL
        )


class CapabilityProbeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    probe_id: UUID = Field(default_factory=uuid4)
    provider_id: str = Field(pattern=REGISTRY_ID_PATTERN)
    model_id: str = Field(pattern=REGISTRY_ID_PATTERN)
    provider_config_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    model_config_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    capabilities: dict[ModelCapability, CapabilityEvidence]
    authentication_status: ProbeAuthenticationStatus
    latency_ms: int = Field(ge=0)
    usage: ModelUsage = Field(default_factory=ModelUsage)
    errors: list[str] = Field(default_factory=list, max_length=50)
    performed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    probe_digest: str = Field(default="", pattern=r"^$|^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def enforce_digest(self) -> CapabilityProbeResult:
        expected = _digest(self.digest_payload())
        if self.probe_digest and self.probe_digest != expected:
            raise ValueError("probe_digest does not match canonical probe evidence")
        self.probe_digest = expected
        return self

    def digest_payload(self) -> dict[str, Any]:
        performed_at = self.performed_at
        if performed_at.tzinfo is None:
            performed_at = performed_at.replace(tzinfo=UTC)
        else:
            performed_at = performed_at.astimezone(UTC)
        return {
            "probe_id": str(self.probe_id),
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "provider_config_digest": self.provider_config_digest,
            "model_config_digest": self.model_config_digest,
            "capabilities": _capability_json(self.capabilities),
            "authentication_status": self.authentication_status.value,
            "latency_ms": self.latency_ms,
            "usage": self.usage.model_dump(mode="json"),
            "errors": self.errors,
            "performed_at": performed_at.isoformat(),
        }

    def is_current(self, provider: ProviderEndpoint, model: ModelProfile) -> bool:
        return (
            self.provider_id == provider.provider_id
            and self.model_id == model.model_id
            and self.provider_config_digest == provider.config_digest
            and self.model_config_digest == model.config_digest
        )


class AgentRoutePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_name: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_.-]+$")
    primary_model_id: str = Field(pattern=REGISTRY_ID_PATTERN)
    fallback_model_ids: list[str] = Field(default_factory=list, max_length=20)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("fallback_model_ids")
    @classmethod
    def valid_unique_fallbacks(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("fallback model IDs must be unique")
        if any(not re.fullmatch(REGISTRY_ID_PATTERN, item) for item in value):
            raise ValueError("fallback model ID is invalid")
        return value

    @model_validator(mode="after")
    def primary_not_in_fallbacks(self) -> AgentRoutePolicy:
        if self.primary_model_id in self.fallback_model_ids:
            raise ValueError("primary model cannot also be a fallback")
        return self


def _capability_json(
    capabilities: dict[ModelCapability, CapabilityEvidence],
) -> dict[str, dict[str, Any]]:
    return {
        key.value: value.model_dump(mode="json")
        for key, value in sorted(capabilities.items(), key=lambda item: item[0].value)
    }


def _digest(value: dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_safe_mapping(value: Any, *, allowed_placeholders: set[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or _is_sensitive_key(key):
                raise ValueError("custom mappings cannot contain secret-bearing fields")
            _validate_safe_mapping(item, allowed_placeholders=allowed_placeholders)
        return
    if isinstance(value, list):
        for item in value:
            _validate_safe_mapping(item, allowed_placeholders=allowed_placeholders)
        return
    if isinstance(value, str):
        placeholders = set(re.findall(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", value))
        if not placeholders <= allowed_placeholders:
            raise ValueError("custom mapping contains an unsupported placeholder")
        if "{{" in value or "{%" in value or "%}" in value:
            raise ValueError("executable templates are not supported")


def _reject_sensitive_data(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if _is_sensitive_key(str(key)):
                raise ValueError("configuration cannot contain secret-bearing fields")
            _reject_sensitive_data(item)
    elif isinstance(value, list):
        for item in value:
            _reject_sensitive_data(item)
    elif isinstance(value, str) and _CREDENTIAL_VALUE.search(value):
        raise ValueError("configuration cannot contain credential-like values")


def _is_sensitive_key(value: str) -> bool:
    normalized = value.casefold().replace("-", "_")
    return normalized not in _SAFE_TOKEN_FIELD_NAMES and _SENSITIVE_KEY.search(value) is not None
