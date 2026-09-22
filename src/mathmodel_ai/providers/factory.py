from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from mathmodel_ai.core.config import Settings
from mathmodel_ai.core.errors import ConfigurationError
from mathmodel_ai.core.types import Environment, ProviderName
from mathmodel_ai.providers.anthropic import AnthropicProvider
from mathmodel_ai.providers.base import BaseModelProvider
from mathmodel_ai.providers.bound import BoundModelProvider, ExecutionGuardedProvider
from mathmodel_ai.providers.compatible import (
    CustomJSONHttpProvider,
    OpenAICompatibleProvider,
)
from mathmodel_ai.providers.google import GoogleProvider
from mathmodel_ai.providers.mock import MockProvider
from mathmodel_ai.providers.openai import OpenAIProvider
from mathmodel_ai.providers.repository import ProviderModelRegistry
from mathmodel_ai.providers.secrets import BaseSecretResolver, EnvironmentSecretResolver
from mathmodel_ai.providers.security import EndpointSecurityPolicy
from mathmodel_ai.schemas.provider_registry import (
    CapabilityProbeResult,
    ModelProfile,
    ProbeAuthenticationStatus,
    ProviderEndpoint,
    ProviderHealthStatus,
    ProviderProtocol,
)


class ProviderRegistry:
    """Runtime protocol-adapter registry keyed by arbitrary stable provider IDs."""

    def __init__(
        self,
        providers: list[BaseModelProvider],
        *,
        configurations: ProviderModelRegistry | None = None,
        secrets: BaseSecretResolver | None = None,
        security_policy: EndpointSecurityPolicy | None = None,
        anthropic_version: str = "2023-06-01",
        configured_clients: Mapping[str, Any] | None = None,
    ) -> None:
        self._providers = {str(provider.name): provider for provider in providers}
        self._configurations = configurations
        self._secrets = secrets
        self._security_policy = security_policy
        self._anthropic_version = anthropic_version
        self._configured_clients = configured_clients or {}
        self._bound: dict[tuple[str, str, str, str, str], BaseModelProvider] = {}

    def register(self, provider_id: str, provider: BaseModelProvider) -> None:
        if provider_id in self._providers:
            raise ConfigurationError(f"provider {provider_id!r} is already registered")
        self._providers[provider_id] = provider

    def get(
        self,
        name: str | ProviderName,
        *,
        model_id: str | None = None,
        for_probe: bool = False,
        expected_provider_config_digest: str | None = None,
        expected_model_config_digest: str | None = None,
        expected_probe_id: UUID | None = None,
        expected_probe_digest: str | None = None,
    ) -> BaseModelProvider:
        provider_id = str(name)
        if model_id is None:
            try:
                return self._providers[provider_id]
            except KeyError as exc:
                raise ConfigurationError(
                    f"provider {provider_id!r} requires a registered model_id"
                ) from exc
        if self._configurations is None or self._secrets is None or self._security_policy is None:
            raise ConfigurationError("configurable provider registry is not initialized")
        model = self._configurations.get_model(model_id)
        endpoint = self._configurations.get_provider(model.provider_id)
        if endpoint.provider_id != provider_id:
            raise ConfigurationError("route provider/model identity mismatch")
        probe_key = "PROBE"
        runtime_model = model
        if for_probe:
            if not endpoint.enabled or not model.enabled:
                raise ConfigurationError("provider or model is disabled")
            if not self._configurations.credential_configured(provider_id):
                raise ConfigurationError("provider credential is not configured")
            model_payload = model.model_dump(mode="python")
            model_payload.update(
                {
                    "observed_capabilities": {},
                    "effective_capabilities": {},
                }
            )
            runtime_model = ModelProfile.model_validate(model_payload)
        else:
            if (
                expected_provider_config_digest is None
                or expected_model_config_digest is None
                or expected_probe_id is None
                or expected_probe_digest is None
            ):
                raise ConfigurationError("registry execution requires an exact route binding")
            endpoint, runtime_model, probe = self._validate_execution_binding(
                provider_id=provider_id,
                model_id=model_id,
                expected_provider_config_digest=expected_provider_config_digest,
                expected_model_config_digest=expected_model_config_digest,
                expected_probe_id=expected_probe_id,
                expected_probe_digest=expected_probe_digest,
            )
            probe_key = probe.probe_digest
            bound_provider_digest = expected_provider_config_digest
            bound_model_digest = expected_model_config_digest
            bound_probe_id = expected_probe_id
            bound_probe_digest = expected_probe_digest
        cache_key = (
            provider_id,
            model_id,
            endpoint.config_digest,
            runtime_model.config_digest,
            probe_key,
        )
        cached = self._bound.get(cache_key)
        if cached is not None:
            return cached
        provider = build_configured_provider(
            endpoint=endpoint,
            model=runtime_model,
            secrets=self._secrets,
            security_policy=self._security_policy,
            anthropic_version=self._anthropic_version,
            client=self._configured_clients.get(provider_id),
        )
        if not for_probe:

            def execution_guard() -> None:
                self._validate_execution_binding(
                    provider_id=provider_id,
                    model_id=model_id,
                    expected_provider_config_digest=bound_provider_digest,
                    expected_model_config_digest=bound_model_digest,
                    expected_probe_id=bound_probe_id,
                    expected_probe_digest=bound_probe_digest,
                )

            provider = ExecutionGuardedProvider(
                provider,
                execution_guard,
            )
        self._bound[cache_key] = provider
        return provider

    def _validate_execution_binding(
        self,
        *,
        provider_id: str,
        model_id: str,
        expected_provider_config_digest: str,
        expected_model_config_digest: str,
        expected_probe_id: UUID,
        expected_probe_digest: str,
    ) -> tuple[ProviderEndpoint, ModelProfile, CapabilityProbeResult]:
        if self._configurations is None:
            raise ConfigurationError("configurable provider registry is not initialized")
        endpoint = self._configurations.get_provider(provider_id)
        model = self._configurations.get_model(model_id)
        if model.provider_id != endpoint.provider_id:
            raise ConfigurationError("route provider/model identity mismatch")
        if not endpoint.enabled or not model.enabled:
            raise ConfigurationError("provider or model was disabled after routing")
        if endpoint.config_digest != expected_provider_config_digest:
            raise ConfigurationError("provider configuration changed after routing")
        if model.config_digest != expected_model_config_digest:
            raise ConfigurationError("model configuration changed after routing")
        if endpoint.health_status is not ProviderHealthStatus.READY:
            raise ConfigurationError(
                f"provider {provider_id!r} is not READY ({endpoint.health_status.value})"
            )
        if endpoint.cooldown_until is not None and endpoint.cooldown_until > datetime.now(UTC):
            raise ConfigurationError("provider circuit breaker is in cooldown")
        try:
            credential_ready = self._configurations.credential_configured(provider_id)
        except Exception as exc:
            raise ConfigurationError("provider credential lookup failed") from exc
        if not credential_ready:
            raise ConfigurationError("provider credential was removed after routing")
        probe = self._configurations.latest_probe(model_id, current_only=True)
        if probe is None:
            raise ConfigurationError("capability probe became stale after routing")
        if probe.probe_id != expected_probe_id or probe.probe_digest != expected_probe_digest:
            raise ConfigurationError("capability evidence changed after routing")
        if probe.authentication_status is not ProbeAuthenticationStatus.PASS:
            raise ConfigurationError("current capability probe authentication is not valid")
        payload = model.model_dump(mode="python")
        payload.update(
            {
                "observed_capabilities": probe.capabilities,
                "config_digest": model.config_digest,
            }
        )
        return endpoint, ModelProfile.model_validate(payload), probe

    @property
    def available(self) -> frozenset[str]:
        # This property is used during application construction and therefore must
        # not open a database connection. Registry-backed models are discovered by
        # ModelRouter only when a route is evaluated.
        return frozenset(self._providers)

    async def aclose(self) -> None:
        closed: set[int] = set()
        for provider in [*self._providers.values(), *self._bound.values()]:
            if id(provider) in closed:
                continue
            closed.add(id(provider))
            await provider.aclose()


def build_provider_registry(
    settings: Settings,
    *,
    configurations: ProviderModelRegistry | None = None,
    secrets: BaseSecretResolver | None = None,
    security_policy: EndpointSecurityPolicy | None = None,
) -> ProviderRegistry:
    providers: list[BaseModelProvider] = []
    if settings.environment is not Environment.PRODUCTION:
        providers.append(MockProvider())
    timeout = settings.provider_timeout_seconds
    if settings.openai_api_key is not None:
        providers.append(
            OpenAIProvider(
                api_key=settings.openai_api_key.get_secret_value(),
                base_url=settings.openai_base_url,
                timeout_seconds=timeout,
                max_response_bytes=settings.provider_max_response_bytes,
            )
        )
    if settings.google_api_key is not None:
        providers.append(
            GoogleProvider(
                api_key=settings.google_api_key.get_secret_value(),
                base_url=settings.google_base_url,
                timeout_seconds=timeout,
                max_response_bytes=settings.provider_max_response_bytes,
            )
        )
    if settings.anthropic_api_key is not None:
        providers.append(
            AnthropicProvider(
                api_key=settings.anthropic_api_key.get_secret_value(),
                base_url=settings.anthropic_base_url,
                api_version=settings.anthropic_version,
                timeout_seconds=timeout,
                max_response_bytes=settings.provider_max_response_bytes,
            )
        )
    resolved_secrets = secrets or EnvironmentSecretResolver()
    resolved_security = security_policy or EndpointSecurityPolicy(
        environment=settings.environment,
        allow_local_model_endpoints=settings.allow_local_model_endpoints,
        allow_insecure_provider_tls=settings.allow_insecure_provider_tls,
    )
    return ProviderRegistry(
        providers,
        configurations=configurations,
        secrets=resolved_secrets,
        security_policy=resolved_security,
        anthropic_version=settings.anthropic_version,
    )


def build_configured_provider(
    *,
    endpoint: ProviderEndpoint,
    model: ModelProfile,
    secrets: BaseSecretResolver,
    security_policy: EndpointSecurityPolicy,
    anthropic_version: str = "2023-06-01",
    client: Any | None = None,
) -> BaseModelProvider:
    if endpoint.provider_id != model.provider_id:
        raise ConfigurationError("provider endpoint and model profile do not match")
    security_policy.validate_configuration(endpoint, resolve_dns=False)
    if endpoint.credential_ref is None:
        raise ConfigurationError("provider endpoint has no credential reference")

    def supply_credential() -> str:
        if endpoint.credential_ref is None:  # pragma: no cover - guarded above
            raise ConfigurationError("provider endpoint has no credential reference")
        return secrets.resolve(endpoint.credential_ref).get_secret_value()

    common: dict[str, Any] = {
        "endpoint": endpoint,
        "model_profile": model,
        "api_key": None,
        "credential_supplier": supply_credential,
        "security_policy": security_policy,
    }
    if client is not None:
        common["client"] = client
    if endpoint.protocol is ProviderProtocol.OPENAI_CHAT_COMPLETIONS:
        return OpenAICompatibleProvider(**common)
    if endpoint.protocol is ProviderProtocol.CUSTOM_JSON_HTTP:
        return CustomJSONHttpProvider(**common)

    native_common: dict[str, Any] = {
        "api_key": None,
        "credential_supplier": supply_credential,
        "base_url": endpoint.base_url,
        "timeout_seconds": endpoint.timeout_seconds,
        "endpoint": endpoint,
        "security_policy": security_policy,
        "max_response_bytes": endpoint.max_response_bytes,
    }
    if client is not None:
        native_common["client"] = client
    if endpoint.protocol is ProviderProtocol.OPENAI_RESPONSES:
        native: BaseModelProvider = OpenAIProvider(**native_common)
    elif endpoint.protocol is ProviderProtocol.ANTHROPIC_MESSAGES:
        native = AnthropicProvider(
            **native_common,
            api_version=str(endpoint.metadata.get("api_version", anthropic_version)),
        )
    elif endpoint.protocol is ProviderProtocol.GOOGLE_GENERATE_CONTENT:
        native = GoogleProvider(**native_common)
    else:  # pragma: no cover - enum exhaustiveness guard
        raise ConfigurationError(f"unsupported provider protocol {endpoint.protocol.value}")
    return BoundModelProvider(native, endpoint=endpoint, model_profile=model)
