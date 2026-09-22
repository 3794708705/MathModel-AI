from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from mathmodel_ai.core.errors import ConfigurationError, ResourceNotFoundError
from mathmodel_ai.db.models import (
    AgentRoutePolicyRecordModel,
    CapabilityProbeRunRecordModel,
    ModelProfileRecordModel,
    ProviderEndpointRecordModel,
)
from mathmodel_ai.db.session import session_scope
from mathmodel_ai.providers.secrets import BaseSecretResolver
from mathmodel_ai.providers.security import EndpointSecurityPolicy
from mathmodel_ai.schemas.provider_registry import (
    AgentRoutePolicy,
    CapabilityProbeResult,
    EndpointTrustLevel,
    ModelProfile,
    ProviderEndpoint,
    ProviderHealthStatus,
)

DEFAULT_ROUTE_POLICY_NAME = "__default__"


class ProviderModelRegistry:
    """Transactional provider/model registry. Secret values never cross this boundary."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        *,
        secrets: BaseSecretResolver,
        security_policy: EndpointSecurityPolicy,
        resolve_dns_on_registration: bool = True,
    ) -> None:
        self._session_factory = session_factory
        self._secrets = secrets
        self._security = security_policy
        self._resolve_dns_on_registration = resolve_dns_on_registration

    def create_provider(self, endpoint: ProviderEndpoint) -> ProviderEndpoint:
        self._security.validate_configuration(
            endpoint, resolve_dns=self._resolve_dns_on_registration
        )
        persisted = self._with_health(endpoint)
        with session_scope(self._session_factory) as session:
            if session.get(ProviderEndpointRecordModel, endpoint.provider_id) is not None:
                raise ConfigurationError(f"provider {endpoint.provider_id!r} already exists")
            session.add(self._provider_row(persisted))
        return persisted

    def update_provider(self, provider_id: str, changes: dict[str, Any]) -> ProviderEndpoint:
        immutable = {"provider_id", "created_at", "config_digest", "health_status"}
        if immutable & changes.keys():
            raise ConfigurationError("provider identity, digest, and health are service managed")
        with session_scope(self._session_factory) as session:
            row = self._provider_row_or_404(session, provider_id, for_update=True)
            # An integrity-valid configuration that is blocked by the current
            # runtime policy must remain editable so an operator can repair its
            # endpoint. The replacement itself is still policy validated below.
            current = self._provider_from_row(row, enforce_runtime_policy=False)
            payload = current.model_dump(mode="python")
            payload.update(changes)
            payload["updated_at"] = datetime.now(UTC)
            payload["config_digest"] = ""
            candidate = ProviderEndpoint.model_validate(payload)
            self._security.validate_configuration(
                candidate, resolve_dns=self._resolve_dns_on_registration
            )
            candidate = self._with_health(candidate)
            self._apply_provider(row, candidate)
            return candidate

    def set_provider_enabled(self, provider_id: str, enabled: bool) -> ProviderEndpoint:
        return self.update_provider(provider_id, {"enabled": enabled})

    def get_provider(self, provider_id: str) -> ProviderEndpoint:
        with session_scope(self._session_factory) as session:
            return self._provider_from_row(self._provider_row_or_404(session, provider_id))

    def get_provider_for_management(self, provider_id: str) -> ProviderEndpoint:
        """Load an integrity-verified provider without granting execution eligibility."""

        with session_scope(self._session_factory) as session:
            return self._provider_from_row(
                self._provider_row_or_404(session, provider_id),
                enforce_runtime_policy=False,
            )

    def list_providers(self) -> list[ProviderEndpoint]:
        with session_scope(self._session_factory) as session:
            rows = session.scalars(
                select(ProviderEndpointRecordModel).order_by(
                    ProviderEndpointRecordModel.provider_id
                )
            )
            return [self._provider_from_row(row) for row in rows]

    def list_providers_for_management(self) -> list[ProviderEndpoint]:
        """List integrity-verified configurations, including policy-blocked endpoints."""

        with session_scope(self._session_factory) as session:
            rows = session.scalars(
                select(ProviderEndpointRecordModel).order_by(
                    ProviderEndpointRecordModel.provider_id
                )
            )
            return [self._provider_from_row(row, enforce_runtime_policy=False) for row in rows]

    def credential_configured(self, provider_id: str) -> bool:
        endpoint = self.get_provider(provider_id)
        return self._secrets.is_configured(endpoint.credential_ref)

    def credential_configured_for_management(self, provider_id: str) -> bool:
        endpoint = self.get_provider_for_management(provider_id)
        return self._secrets.is_configured(endpoint.credential_ref)

    def runtime_policy_allows(self, endpoint: ProviderEndpoint) -> bool:
        """Return whether current policy permits execution without resolving DNS."""

        try:
            self._security.validate_configuration(endpoint, resolve_dns=False)
        except ConfigurationError:
            return False
        return True

    def create_model(self, profile: ModelProfile) -> ModelProfile:
        with session_scope(self._session_factory) as session:
            provider = self._provider_from_row(
                self._provider_row_or_404(session, profile.provider_id)
            )
            if session.get(ModelProfileRecordModel, profile.model_id) is not None:
                raise ConfigurationError(f"model {profile.model_id!r} already exists")
            candidate = self._normalize_model_trust(profile, provider)
            session.add(self._model_row(candidate))
            return candidate

    def update_model(self, model_id: str, changes: dict[str, Any]) -> ModelProfile:
        immutable = {
            "model_id",
            "provider_id",
            "created_at",
            "config_digest",
            "observed_capabilities",
        }
        if immutable & changes.keys():
            raise ConfigurationError(
                "model identity, provider, digest, and observations are service managed"
            )
        with session_scope(self._session_factory) as session:
            row = self._model_row_or_404(session, model_id, for_update=True)
            current = self._model_from_row(row)
            provider = self._provider_from_row(
                self._provider_row_or_404(session, current.provider_id)
            )
            payload = current.model_dump(mode="python")
            payload.update(changes)
            payload["updated_at"] = datetime.now(UTC)
            payload["config_digest"] = ""
            candidate = self._normalize_model_trust(ModelProfile.model_validate(payload), provider)
            self._apply_model(row, candidate)
            return candidate

    def set_model_enabled(self, model_id: str, enabled: bool) -> ModelProfile:
        return self.update_model(model_id, {"enabled": enabled})

    def get_model(self, model_id: str) -> ModelProfile:
        with session_scope(self._session_factory) as session:
            return self._model_from_row(self._model_row_or_404(session, model_id))

    def list_models(self, *, provider_id: str | None = None) -> list[ModelProfile]:
        statement = select(ModelProfileRecordModel)
        if provider_id is not None:
            statement = statement.where(ModelProfileRecordModel.provider_id == provider_id)
        statement = statement.order_by(ModelProfileRecordModel.model_id)
        with session_scope(self._session_factory) as session:
            return [self._model_from_row(row) for row in session.scalars(statement)]

    def record_probe(self, result: CapabilityProbeResult) -> CapabilityProbeResult:
        with session_scope(self._session_factory) as session:
            provider_row = self._provider_row_or_404(session, result.provider_id, for_update=True)
            model_row = self._model_row_or_404(session, result.model_id, for_update=True)
            provider = self._provider_from_row(provider_row)
            model = self._model_from_row(model_row)
            if not result.is_current(provider, model):
                raise ConfigurationError("probe digests do not match current provider/model config")
            if session.get(CapabilityProbeRunRecordModel, result.probe_id) is not None:
                raise ConfigurationError(f"probe {result.probe_id} already exists")
            session.add(self._probe_row(result))
            model_payload = model.model_dump(mode="python")
            model_payload.update(
                {
                    "observed_capabilities": {
                        capability: evidence
                        for capability, evidence in result.capabilities.items()
                        if evidence.status.value != "UNKNOWN"
                    },
                    "config_digest": "",
                    "updated_at": datetime.now(UTC),
                }
            )
            observed_model = ModelProfile.model_validate(model_payload)
            self._apply_model(model_row, observed_model)
            provider_payload = provider.model_dump(mode="python")
            if not provider.enabled:
                health_status = ProviderHealthStatus.DISABLED
            elif not model.enabled:
                health_status = provider.health_status
            else:
                health_status = _health_from_probe(result)
            provider_payload.update(
                {
                    "health_status": health_status,
                    "consecutive_failures": (
                        provider.consecutive_failures
                        if not provider.enabled or not model.enabled
                        else (
                            0
                            if result.authentication_status.value == "PASS"
                            else provider.consecutive_failures + 1
                        )
                    ),
                    "updated_at": datetime.now(UTC),
                    "config_digest": "",
                }
            )
            probed_provider = ProviderEndpoint.model_validate(provider_payload)
            self._apply_provider(provider_row, probed_provider)
        return result

    def latest_probe(
        self, model_id: str, *, current_only: bool = True
    ) -> CapabilityProbeResult | None:
        with session_scope(self._session_factory) as session:
            model = self._model_from_row(self._model_row_or_404(session, model_id))
            provider = self._provider_from_row(
                self._provider_row_or_404(session, model.provider_id)
            )
            statement = (
                select(CapabilityProbeRunRecordModel)
                .where(CapabilityProbeRunRecordModel.model_id == model_id)
                .order_by(
                    CapabilityProbeRunRecordModel.performed_at.desc(),
                    CapabilityProbeRunRecordModel.id.desc(),
                )
            )
            for row in session.scalars(statement):
                probe = self._probe_from_row(row)
                if not current_only or probe.is_current(provider, model):
                    return probe
            return None

    def put_agent_route(self, policy: AgentRoutePolicy) -> AgentRoutePolicy:
        with session_scope(self._session_factory) as session:
            all_ids = [policy.primary_model_id, *policy.fallback_model_ids]
            missing = [
                item for item in all_ids if session.get(ModelProfileRecordModel, item) is None
            ]
            if missing:
                raise ResourceNotFoundError(
                    f"route references unknown models: {', '.join(missing)}"
                )
            row = session.get(AgentRoutePolicyRecordModel, policy.agent_name)
            if row is None:
                row = AgentRoutePolicyRecordModel(
                    agent_name=policy.agent_name,
                    primary_model_id=policy.primary_model_id,
                    fallback_model_ids=policy.fallback_model_ids,
                    updated_at=policy.updated_at,
                )
                session.add(row)
            else:
                row.primary_model_id = policy.primary_model_id
                row.fallback_model_ids = policy.fallback_model_ids
                row.updated_at = datetime.now(UTC)
            return AgentRoutePolicy(
                agent_name=row.agent_name,
                primary_model_id=row.primary_model_id,
                fallback_model_ids=row.fallback_model_ids,
                updated_at=row.updated_at,
            )

    def get_agent_route(self, agent_name: str) -> AgentRoutePolicy | None:
        with session_scope(self._session_factory) as session:
            row = session.get(AgentRoutePolicyRecordModel, agent_name)
            if row is None:
                return None
            return AgentRoutePolicy(
                agent_name=row.agent_name,
                primary_model_id=row.primary_model_id,
                fallback_model_ids=row.fallback_model_ids,
                updated_at=row.updated_at,
            )

    def list_agent_routes(self) -> list[AgentRoutePolicy]:
        with session_scope(self._session_factory) as session:
            rows = session.scalars(
                select(AgentRoutePolicyRecordModel).order_by(AgentRoutePolicyRecordModel.agent_name)
            )
            return [
                AgentRoutePolicy(
                    agent_name=row.agent_name,
                    primary_model_id=row.primary_model_id,
                    fallback_model_ids=row.fallback_model_ids,
                    updated_at=row.updated_at,
                )
                for row in rows
            ]

    def get_default_model_id(self) -> str | None:
        policy = self.get_agent_route(DEFAULT_ROUTE_POLICY_NAME)
        return policy.primary_model_id if policy is not None else None

    def put_default_model(self, model_id: str) -> str:
        self.put_agent_route(
            AgentRoutePolicy(
                agent_name=DEFAULT_ROUTE_POLICY_NAME,
                primary_model_id=model_id,
            )
        )
        return model_id

    def _with_health(self, endpoint: ProviderEndpoint) -> ProviderEndpoint:
        status = (
            ProviderHealthStatus.DISABLED
            if not endpoint.enabled
            else (
                ProviderHealthStatus.UNAVAILABLE
                if self._secrets.is_configured(endpoint.credential_ref)
                else ProviderHealthStatus.UNCONFIGURED
            )
        )
        payload = endpoint.model_dump(mode="python")
        payload.update({"health_status": status, "config_digest": ""})
        return ProviderEndpoint.model_validate(payload)

    @staticmethod
    def _normalize_model_trust(profile: ModelProfile, provider: ProviderEndpoint) -> ModelProfile:
        if (
            profile.trust_level is EndpointTrustLevel.OFFICIAL_VENDOR
            and provider.trust_level is not EndpointTrustLevel.OFFICIAL_VENDOR
        ):
            raise ConfigurationError("a model on a proxy endpoint cannot claim official trust")
        payload = profile.model_dump(mode="python")
        payload.update({"trust_level": provider.trust_level, "config_digest": ""})
        return ModelProfile.model_validate(payload)

    @staticmethod
    def _provider_row_or_404(
        session: Session, provider_id: str, *, for_update: bool = False
    ) -> ProviderEndpointRecordModel:
        statement = select(ProviderEndpointRecordModel).where(
            ProviderEndpointRecordModel.provider_id == provider_id
        )
        if for_update:
            statement = statement.with_for_update()
        row = session.scalar(statement)
        if row is None:
            raise ResourceNotFoundError(f"provider {provider_id!r} was not found")
        return row

    @staticmethod
    def _model_row_or_404(
        session: Session, model_id: str, *, for_update: bool = False
    ) -> ModelProfileRecordModel:
        statement = select(ModelProfileRecordModel).where(
            ModelProfileRecordModel.model_id == model_id
        )
        if for_update:
            statement = statement.with_for_update()
        row = session.scalar(statement)
        if row is None:
            raise ResourceNotFoundError(f"model {model_id!r} was not found")
        return row

    @staticmethod
    def _provider_row(endpoint: ProviderEndpoint) -> ProviderEndpointRecordModel:
        return ProviderEndpointRecordModel(**_provider_values(endpoint))

    @staticmethod
    def _apply_provider(row: ProviderEndpointRecordModel, endpoint: ProviderEndpoint) -> None:
        for name, value in _provider_values(endpoint).items():
            setattr(row, name, value)

    @staticmethod
    def _model_row(profile: ModelProfile) -> ModelProfileRecordModel:
        return ModelProfileRecordModel(**_model_values(profile))

    @staticmethod
    def _apply_model(row: ModelProfileRecordModel, profile: ModelProfile) -> None:
        for name, value in _model_values(profile).items():
            setattr(row, name, value)

    def _provider_from_row(
        self,
        row: ProviderEndpointRecordModel,
        *,
        enforce_runtime_policy: bool = True,
    ) -> ProviderEndpoint:
        try:
            endpoint = ProviderEndpoint(
                provider_id=row.provider_id,
                display_name=row.display_name,
                protocol=row.protocol,
                base_url=row.base_url,
                credential_ref=row.credential_ref,
                enabled=row.enabled,
                trust_level=row.trust_level,
                timeout_seconds=row.timeout_seconds,
                connect_timeout_seconds=row.connect_timeout_seconds,
                max_response_bytes=row.max_response_bytes,
                verify_tls=row.verify_tls,
                allow_redirects=row.allow_redirects,
                additional_headers=row.additional_headers,
                metadata=row.metadata_json,
                health_status=row.health_status,
                consecutive_failures=row.consecutive_failures,
                cooldown_until=row.cooldown_until,
                config_digest=row.config_digest,
                created_at=row.created_at,
                updated_at=row.updated_at,
            )
        except ValidationError as exc:
            raise ConfigurationError(
                "persisted provider configuration failed integrity checks"
            ) from exc
        if enforce_runtime_policy:
            try:
                self._security.validate_configuration(endpoint, resolve_dns=False)
            except ConfigurationError as exc:
                raise ConfigurationError(
                    "persisted provider endpoint is blocked by the current security policy"
                ) from exc
        return endpoint

    @staticmethod
    def _model_from_row(row: ModelProfileRecordModel) -> ModelProfile:
        try:
            return ModelProfile(
                model_id=row.model_id,
                provider_id=row.provider_id,
                display_name=row.display_name,
                remote_model=row.remote_model,
                enabled=row.enabled,
                quality_tier=row.quality_tier,
                quality_tier_source=row.quality_tier_source,
                declared_capabilities=row.declared_capabilities,
                observed_capabilities=row.observed_capabilities,
                structured_output_strategy=row.structured_output_strategy,
                tool_calling_strategy=row.tool_calling_strategy,
                reasoning_mapping=row.reasoning_mapping,
                context_window=row.context_window,
                max_output_tokens=row.max_output_tokens,
                pricing=row.pricing_json,
                trust_level=row.trust_level,
                configuration=row.configuration,
                config_digest=row.config_digest,
                created_at=row.created_at,
                updated_at=row.updated_at,
            )
        except ValidationError as exc:
            raise ConfigurationError(
                "persisted model configuration failed integrity checks"
            ) from exc

    @staticmethod
    def _probe_row(result: CapabilityProbeResult) -> CapabilityProbeRunRecordModel:
        return CapabilityProbeRunRecordModel(
            id=result.probe_id,
            provider_id=result.provider_id,
            model_id=result.model_id,
            provider_config_digest=result.provider_config_digest,
            model_config_digest=result.model_config_digest,
            probe_digest=result.probe_digest,
            capabilities={
                key.value: value.model_dump(mode="json")
                for key, value in result.capabilities.items()
            },
            authentication_status=result.authentication_status.value,
            latency_ms=result.latency_ms,
            usage_json=result.usage.model_dump(mode="json"),
            errors=result.errors,
            performed_at=result.performed_at,
        )

    @staticmethod
    def _probe_from_row(row: CapabilityProbeRunRecordModel) -> CapabilityProbeResult:
        try:
            return CapabilityProbeResult(
                probe_id=row.id,
                provider_id=row.provider_id,
                model_id=row.model_id,
                provider_config_digest=row.provider_config_digest,
                model_config_digest=row.model_config_digest,
                probe_digest=row.probe_digest,
                capabilities=row.capabilities,
                authentication_status=row.authentication_status,
                latency_ms=row.latency_ms,
                usage=row.usage_json,
                errors=row.errors,
                performed_at=row.performed_at,
            )
        except ValidationError as exc:
            raise ConfigurationError("persisted capability probe failed integrity checks") from exc


ProviderRegistryRepository = ProviderModelRegistry
ModelRegistry = ProviderModelRegistry


def _provider_values(endpoint: ProviderEndpoint) -> dict[str, Any]:
    return {
        "provider_id": endpoint.provider_id,
        "display_name": endpoint.display_name,
        "protocol": endpoint.protocol.value,
        "base_url": endpoint.base_url,
        "credential_ref": endpoint.credential_ref,
        "enabled": endpoint.enabled,
        "trust_level": endpoint.trust_level.value,
        "timeout_seconds": endpoint.timeout_seconds,
        "connect_timeout_seconds": endpoint.connect_timeout_seconds,
        "max_response_bytes": endpoint.max_response_bytes,
        "verify_tls": endpoint.verify_tls,
        "allow_redirects": endpoint.allow_redirects,
        "additional_headers": endpoint.additional_headers,
        "metadata_json": endpoint.metadata,
        "health_status": endpoint.health_status.value,
        "consecutive_failures": endpoint.consecutive_failures,
        "cooldown_until": endpoint.cooldown_until,
        "config_digest": endpoint.config_digest,
        "created_at": endpoint.created_at,
        "updated_at": endpoint.updated_at,
    }


def _model_values(profile: ModelProfile) -> dict[str, Any]:
    return {
        "model_id": profile.model_id,
        "provider_id": profile.provider_id,
        "display_name": profile.display_name,
        "remote_model": profile.remote_model,
        "enabled": profile.enabled,
        "quality_tier": profile.quality_tier.value,
        "quality_tier_source": profile.quality_tier_source.value,
        "declared_capabilities": {
            key.value: value.model_dump(mode="json")
            for key, value in profile.declared_capabilities.items()
        },
        "observed_capabilities": {
            key.value: value.model_dump(mode="json")
            for key, value in profile.observed_capabilities.items()
        },
        "structured_output_strategy": profile.structured_output_strategy.value,
        "tool_calling_strategy": profile.tool_calling_strategy.value,
        "reasoning_mapping": profile.reasoning_mapping,
        "context_window": profile.context_window,
        "max_output_tokens": profile.max_output_tokens,
        "pricing_json": profile.pricing.model_dump(mode="json"),
        "trust_level": profile.trust_level.value,
        "configuration": profile.configuration,
        "config_digest": profile.config_digest,
        "created_at": profile.created_at,
        "updated_at": profile.updated_at,
    }


def _health_from_probe(result: CapabilityProbeResult) -> ProviderHealthStatus:
    mapping = {
        "PASS": ProviderHealthStatus.READY,
        "AUTH_FAILED": ProviderHealthStatus.AUTH_FAILED,
        "NOT_CONFIGURED": ProviderHealthStatus.UNCONFIGURED,
        "UNAVAILABLE": ProviderHealthStatus.UNAVAILABLE,
    }
    return mapping[result.authentication_status.value]
