import asyncio
import gzip
import json
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
from pydantic import BaseModel, Field, SecretStr, ValidationError
from sqlalchemy import select

from mathmodel_ai.core.config import Settings
from mathmodel_ai.core.errors import (
    ConfigurationError,
    ProviderError,
    ProviderResponseError,
    ProviderResponseTooLargeError,
    ProviderTimeoutError,
)
from mathmodel_ai.core.types import Environment, ReasoningEffort
from mathmodel_ai.db.base import Base
from mathmodel_ai.db.models import (
    CapabilityProbeRunRecordModel,
    ModelProfileRecordModel,
    ProviderEndpointRecordModel,
)
from mathmodel_ai.db.session import (
    create_database_engine,
    create_session_factory,
    session_scope,
)
from mathmodel_ai.providers.compatible import OpenAICompatibleProvider
from mathmodel_ai.providers.factory import ProviderRegistry
from mathmodel_ai.providers.probe import ProviderCompatibilityProbe
from mathmodel_ai.providers.repository import ProviderModelRegistry
from mathmodel_ai.providers.schemas import GenerationRequest, ModelMessage, ModelUsage
from mathmodel_ai.providers.secrets import BaseSecretResolver, EnvironmentSecretResolver
from mathmodel_ai.providers.security import EndpointSecurityPolicy, safe_error
from mathmodel_ai.routing.router import ModelRouter
from mathmodel_ai.routing.schemas import RouteAction, TaskProfile, TaskType
from mathmodel_ai.schemas.provider_registry import (
    AgentRoutePolicy,
    CapabilityEvidence,
    CapabilityProbeResult,
    CapabilitySource,
    CapabilityStatus,
    CustomJSONConfiguration,
    CustomJSONRequestMapping,
    CustomJSONResponseMapping,
    EndpointTrustLevel,
    ModelCapability,
    ModelPricing,
    ModelProfile,
    ProbeAuthenticationStatus,
    ProviderEndpoint,
    ProviderHealthStatus,
    ProviderProtocol,
    QualityTier,
    StructuredOutputStrategy,
)


class _PositiveAnswer(BaseModel):
    value: int = Field(gt=0)


def _policy(
    resolver: Callable[[str, int | None], list[str]] | None = None,
) -> EndpointSecurityPolicy:
    return EndpointSecurityPolicy(
        environment=Environment.TEST,
        resolver=resolver or (lambda _host, _port: ["93.184.216.34"]),
    )


def _endpoint(**changes: object) -> ProviderEndpoint:
    payload: dict[str, object] = {
        "provider_id": "audit-provider",
        "display_name": "Audit Provider",
        "protocol": ProviderProtocol.OPENAI_CHAT_COMPLETIONS,
        "base_url": "https://audit.example.test/v1",
        "credential_ref": "env:AUDIT_KEY",
        "trust_level": EndpointTrustLevel.USER_MANAGED_PROXY,
    }
    payload.update(changes)
    return ProviderEndpoint.model_validate(payload)


def _evidence(status: CapabilityStatus) -> CapabilityEvidence:
    return CapabilityEvidence(status=status, source=CapabilitySource.PROBED)


def _model(**changes: object) -> ModelProfile:
    declared = {
        capability: CapabilityEvidence(
            status=CapabilityStatus.SUPPORTED,
            source=CapabilitySource.USER_DECLARED,
        )
        for capability in (
            ModelCapability.TEXT,
            ModelCapability.STRUCTURED_OUTPUT,
            ModelCapability.JSON_SCHEMA,
        )
    }
    payload: dict[str, object] = {
        "model_id": "audit-model",
        "provider_id": "audit-provider",
        "display_name": "Audit Model",
        "remote_model": "remote-audit-model",
        "quality_tier": QualityTier.FLAGSHIP_MAX,
        "declared_capabilities": declared,
        "structured_output_strategy": StructuredOutputStrategy.NATIVE_JSON_SCHEMA,
        "context_window": 10_000_000,
    }
    payload.update(changes)
    return ModelProfile.model_validate(payload)


def _compatible_response(
    content: str = "OK", *, model: str = "remote-audit-model", tool_name: str | None = None
) -> httpx.Response:
    tool_calls: list[dict[str, Any]] = []
    if tool_name is not None:
        tool_calls.append(
            {
                "id": "call-audit",
                "type": "function",
                "function": {"name": tool_name, "arguments": '{"value":1}'},
            }
        )
    return httpx.Response(
        200,
        json={
            "id": "response-audit",
            "model": model,
            "choices": [
                {
                    "message": {"content": content, "tool_calls": tool_calls},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        },
    )


def _stack(
    *,
    environment: dict[str, str] | None = None,
    handler: Callable[[httpx.Request], httpx.Response] | None = None,
) -> tuple[
    ProviderModelRegistry,
    ProviderRegistry,
    dict[str, str],
    httpx.AsyncClient,
    Any,
]:
    settings = Settings(environment="test", database_url="sqlite+pysqlite:///:memory:")
    engine = create_database_engine(settings)
    Base.metadata.create_all(engine)
    session_factory = create_session_factory(engine)
    secrets = environment if environment is not None else {"AUDIT_KEY": "secret-a"}
    security = _policy()
    registry = ProviderModelRegistry(
        session_factory,
        secrets=EnvironmentSecretResolver(secrets),
        security_policy=security,
    )
    registry.create_provider(_endpoint())
    registry.create_model(_model())
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler or (lambda _request: _compatible_response())),
        base_url="https://audit.example.test/v1",
    )
    providers = ProviderRegistry(
        [],
        configurations=registry,
        secrets=EnvironmentSecretResolver(secrets),
        security_policy=security,
        configured_clients={"audit-provider": client},
    )
    return registry, providers, secrets, client, session_factory


def _record_probe(
    registry: ProviderModelRegistry,
    *,
    authentication: ProbeAuthenticationStatus = ProbeAuthenticationStatus.PASS,
    json_schema: CapabilityStatus = CapabilityStatus.SUPPORTED,
    extra: dict[ModelCapability, CapabilityEvidence] | None = None,
) -> CapabilityProbeResult:
    endpoint = registry.get_provider("audit-provider")
    model = registry.get_model("audit-model")
    capabilities = {
        ModelCapability.TEXT: _evidence(CapabilityStatus.SUPPORTED),
        ModelCapability.STRUCTURED_OUTPUT: _evidence(CapabilityStatus.SUPPORTED),
        ModelCapability.JSON_SCHEMA: _evidence(json_schema),
    }
    capabilities.update(extra or {})
    return registry.record_probe(
        CapabilityProbeResult(
            provider_id=endpoint.provider_id,
            model_id=model.model_id,
            provider_config_digest=endpoint.config_digest,
            model_config_digest=model.config_digest,
            capabilities=capabilities,
            authentication_status=authentication,
            latency_ms=1,
        )
    )


def _decision(registry: ProviderModelRegistry, **profile: object) -> Any:
    payload: dict[str, object] = {
        "task_type": TaskType.DOCUMENTATION,
        "preferred_model_id": "audit-model",
    }
    payload.update(profile)
    return ModelRouter(
        Settings(environment="test", default_model_id="audit-model"),
        registry=registry,
        available_providers=set(),
    ).route(TaskProfile.model_validate(payload))


def _generation() -> GenerationRequest:
    return GenerationRequest(
        model="ignored",
        messages=[ModelMessage(role="user", content="Reply OK")],
        max_output_tokens=8,
    )


@pytest.mark.parametrize(
    "hostname",
    ["2130706433", "0177.0.0.1", "0x7f000001"],
)
def test_encoded_ipv4_forms_fail_closed_after_resolution(hostname: str) -> None:
    endpoint = _endpoint(base_url=f"https://{hostname}/v1")
    with pytest.raises(ProviderError, match="private or local"):
        _policy(lambda _host, _port: ["127.0.0.1"]).validate_configuration(endpoint)


@pytest.mark.parametrize(
    "address",
    ["::1", "fc00::1", "fe80::1", "::ffff:127.0.0.1", "169.254.169.254"],
)
def test_ipv6_mapped_private_and_link_local_addresses_are_rejected(address: str) -> None:
    with pytest.raises(ProviderError, match="private or local"):
        _policy(lambda _host, _port: [address]).validate_configuration(_endpoint())


def test_mixed_dns_answer_is_rejected() -> None:
    with pytest.raises(ProviderError, match="mixed public and private"):
        _policy(lambda _host, _port: ["93.184.216.34", "10.0.0.5"]).validate_configuration(
            _endpoint()
        )


@pytest.mark.parametrize(
    "base_url",
    [
        "https://api.deepseek.com.evil.example/v1",
        "https://deepseek-api.example/v1",
        "https://xn--deepsek-9za.example/v1",
        "https://deepseék.example/v1",
    ],
)
def test_lookalike_and_unicode_domains_cannot_claim_official_trust(base_url: str) -> None:
    with pytest.raises(ConfigurationError, match="OFFICIAL_VENDOR"):
        _policy().validate_configuration(
            _endpoint(base_url=base_url, trust_level=EndpointTrustLevel.OFFICIAL_VENDOR),
            resolve_dns=False,
        )


def test_custom_paths_reject_encoded_traversal_scheme_relative_and_backslash() -> None:
    for path in ("/../admin", "/%2e%2e/admin", "/%252e%252e/admin", "//evil/x", "/a\\b"):
        with pytest.raises(ValidationError, match="endpoint_path"):
            CustomJSONRequestMapping(endpoint_path=path)


def test_custom_mapping_rejects_executable_templates_and_unapproved_expansion() -> None:
    for body in (
        {"input": "{{ cycler.__init__.__globals__.os.system('id') }}"},
        {"input": "{% import os %}"},
        {"input": "{environment}"},
    ):
        with pytest.raises(ValidationError):
            CustomJSONRequestMapping(body=body)
    for path in ("result[0].text", "result.__class__()", "$.result.text"):
        with pytest.raises(ValidationError, match="response paths"):
            CustomJSONResponseMapping(text_path=path)


def test_custom_auth_header_cannot_override_reserved_transport_headers() -> None:
    for header in ("Authorization", "Proxy-Authorization", "Cookie", "Host"):
        with pytest.raises(ValidationError, match="reserved"):
            CustomJSONConfiguration(
                request=CustomJSONRequestMapping(),
                response=CustomJSONResponseMapping(text_path="text"),
                auth_scheme="API_KEY_HEADER",
                api_key_header=header,
            )


def test_pricing_rejects_nan_infinity_and_negative_values() -> None:
    for value in (float("nan"), float("inf"), float("-inf"), -0.01):
        with pytest.raises(ValidationError):
            ModelPricing(input_per_million=value)


@pytest.mark.parametrize(
    "identifier",
    ["Audit-Provider", " audit-provider", "audit-provider ", "é-provider", "é-provider"],
)
def test_registry_ids_reject_case_whitespace_and_unicode_collisions(identifier: str) -> None:
    with pytest.raises(ValidationError):
        _endpoint(provider_id=identifier)
    with pytest.raises(ValidationError):
        _model(model_id=identifier)


def test_duplicate_stable_ids_are_rejected_but_remote_model_aliases_are_distinct() -> None:
    registry, providers, _secrets, client, _session_factory = _stack()
    with pytest.raises(ConfigurationError, match="already exists"):
        registry.create_provider(_endpoint())
    with pytest.raises(ConfigurationError, match="already exists"):
        registry.create_model(_model())
    second = registry.create_model(
        _model(model_id="audit-model-two", display_name="Audit Model Two")
    )
    assert second.remote_model == registry.get_model("audit-model").remote_model
    assert second.model_id != "audit-model"
    assert not hasattr(registry, "delete_provider")
    assert not hasattr(registry, "delete_model")
    asyncio.run(providers.aclose())
    asyncio.run(client.aclose())


def test_agent_fallback_policy_rejects_loops_and_duplicates() -> None:
    with pytest.raises(ValidationError, match="must be unique"):
        AgentRoutePolicy(
            agent_name="problem_agent",
            primary_model_id="model-a",
            fallback_model_ids=["model-b", "model-b"],
        )
    with pytest.raises(ValidationError, match="cannot also be a fallback"):
        AgentRoutePolicy(
            agent_name="problem_agent",
            primary_model_id="model-a",
            fallback_model_ids=["model-a", "model-b"],
        )


def test_usage_rejects_bool_negative_and_extreme_direct_values() -> None:
    for value in (True, -1, 1_000_000_001):
        with pytest.raises(ValidationError):
            ModelUsage(input_tokens=value)  # type: ignore[arg-type]


def test_empty_and_whitespace_credentials_are_not_configured() -> None:
    resolver = EnvironmentSecretResolver({"EMPTY": "", "SPACE": "   "})
    assert resolver.is_configured("env:EMPTY") is False
    assert resolver.is_configured("env:SPACE") is False
    with pytest.raises(ConfigurationError, match="not configured"):
        resolver.resolve("env:SPACE")


def test_secret_resolver_failure_makes_routing_fail_closed() -> None:
    class FailingResolver(BaseSecretResolver):
        def resolve(self, credential_ref: str) -> SecretStr:
            raise OSError("secret backend unavailable")

        def is_configured(self, credential_ref: str | None) -> bool:
            raise OSError("secret backend unavailable")

    registry, providers, _secrets, client, _session_factory = _stack()
    _record_probe(registry)
    registry._secrets = FailingResolver()  # type: ignore[attr-defined]
    decision = _decision(registry)
    assert decision.action is RouteAction.HUMAN_REVIEW
    assert "credential lookup failed" in " ".join(decision.rejected_models["audit-model"])
    asyncio.run(providers.aclose())
    asyncio.run(client.aclose())


def test_persisted_fake_official_endpoint_is_revalidated_on_read() -> None:
    registry, providers, _secrets, client, session_factory = _stack()
    forged = _endpoint(trust_level=EndpointTrustLevel.OFFICIAL_VENDOR)
    with session_scope(session_factory) as session:
        row = session.get(ProviderEndpointRecordModel, "audit-provider")
        assert row is not None
        row.trust_level = EndpointTrustLevel.OFFICIAL_VENDOR.value
        row.config_digest = forged.config_digest
    with pytest.raises(ConfigurationError, match="blocked by the current security policy"):
        registry.get_provider("audit-provider")
    asyncio.run(providers.aclose())
    asyncio.run(client.aclose())


def test_persisted_model_config_digest_tamper_is_rejected() -> None:
    registry, providers, _secrets, client, session_factory = _stack()
    with session_scope(session_factory) as session:
        row = session.get(ModelProfileRecordModel, "audit-model")
        assert row is not None
        row.remote_model = "tampered-remote-model"
    with pytest.raises(ConfigurationError, match="model configuration failed integrity"):
        registry.get_model("audit-model")
    asyncio.run(providers.aclose())
    asyncio.run(client.aclose())


def test_persisted_provider_config_digest_tamper_is_rejected() -> None:
    registry, providers, _secrets, client, session_factory = _stack()
    with session_scope(session_factory) as session:
        row = session.get(ProviderEndpointRecordModel, "audit-provider")
        assert row is not None
        row.base_url = "https://tampered.example.test/v1"
    with pytest.raises(ConfigurationError, match="provider configuration failed integrity checks"):
        registry.get_provider("audit-provider")
    asyncio.run(providers.aclose())
    asyncio.run(client.aclose())


@pytest.mark.parametrize(
    ("target", "changes"),
    [
        ("provider", {"base_url": "https://replacement.example.test/v1"}),
        ("provider", {"credential_ref": "env:ROTATED_KEY"}),
        ("model", {"reasoning_mapping": {ReasoningEffort.MAX: "max"}}),
    ],
)
def test_security_relevant_config_changes_make_old_probe_stale(
    target: str, changes: dict[str, object]
) -> None:
    registry, providers, secrets, client, _session_factory = _stack()
    secrets["ROTATED_KEY"] = "secret-b"
    _record_probe(registry)
    if target == "provider":
        registry.update_provider("audit-provider", changes)
    else:
        registry.update_model("audit-model", changes)
    assert registry.latest_probe("audit-model", current_only=True) is None
    decision = _decision(registry)
    assert decision.action is RouteAction.HUMAN_REVIEW
    asyncio.run(providers.aclose())
    asyncio.run(client.aclose())


def test_atomic_probe_digest_detects_capability_tampering() -> None:
    registry, providers, _secrets, client, session_factory = _stack()
    probe = _record_probe(registry, json_schema=CapabilityStatus.UNSUPPORTED)
    with session_scope(session_factory) as session:
        row = session.get(CapabilityProbeRunRecordModel, probe.probe_id)
        assert row is not None
        row.capabilities[ModelCapability.JSON_SCHEMA.value]["status"] = "SUPPORTED"
        from sqlalchemy.orm.attributes import flag_modified

        flag_modified(row, "capabilities")
    with pytest.raises(ConfigurationError, match="probe failed integrity"):
        registry.latest_probe("audit-model")
    decision = _decision(registry, requires_json_schema=True)
    assert decision.action is RouteAction.HUMAN_REVIEW
    assert "integrity" in " ".join(decision.rejected_models["audit-model"])
    asyncio.run(providers.aclose())
    asyncio.run(client.aclose())


def test_tampered_capability_summary_cannot_override_atomic_probe() -> None:
    registry, providers, _secrets, client, session_factory = _stack()
    _record_probe(registry, json_schema=CapabilityStatus.UNSUPPORTED)
    with session_scope(session_factory) as session:
        row = session.get(ModelProfileRecordModel, "audit-model")
        assert row is not None
        row.observed_capabilities[ModelCapability.JSON_SCHEMA.value]["status"] = "SUPPORTED"
        from sqlalchemy.orm.attributes import flag_modified

        flag_modified(row, "observed_capabilities")
    decision = _decision(registry, requires_json_schema=True)
    assert decision.action is RouteAction.HUMAN_REVIEW
    assert "JSON_SCHEMA" in " ".join(decision.rejected_models["audit-model"])
    asyncio.run(providers.aclose())
    asyncio.run(client.aclose())


def test_ready_health_tamper_cannot_override_failed_probe_authentication() -> None:
    registry, providers, _secrets, client, session_factory = _stack()
    _record_probe(registry, authentication=ProbeAuthenticationStatus.AUTH_FAILED)
    with session_scope(session_factory) as session:
        row = session.get(ProviderEndpointRecordModel, "audit-provider")
        assert row is not None
        row.health_status = ProviderHealthStatus.READY.value
    decision = _decision(registry)
    assert decision.action is RouteAction.HUMAN_REVIEW
    assert "authentication is AUTH_FAILED" in " ".join(decision.rejected_models["audit-model"])
    asyncio.run(providers.aclose())
    asyncio.run(client.aclose())


def test_user_context_window_claim_requires_probed_long_context_evidence() -> None:
    registry, providers, _secrets, client, _session_factory = _stack()
    _record_probe(registry)
    decision = _decision(registry, minimum_context_tokens=100_000)
    assert decision.action is RouteAction.HUMAN_REVIEW
    assert "LONG_CONTEXT" in " ".join(decision.rejected_models["audit-model"])
    asyncio.run(providers.aclose())
    asyncio.run(client.aclose())


@pytest.mark.asyncio
async def test_route_binding_revalidates_disable_before_network_call() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _compatible_response()

    registry, providers, _secrets, client, _session_factory = _stack(handler=handler)
    _record_probe(registry)
    route = _decision(registry)
    assert route.capability_probe_id is not None
    provider = providers.get(
        route.selected_provider or "",
        model_id=route.selected_model_id,
        expected_provider_config_digest=route.provider_config_digest,
        expected_model_config_digest=route.model_config_digest,
        expected_probe_id=route.capability_probe_id,
        expected_probe_digest=route.capability_probe_digest,
    )
    registry.set_provider_enabled("audit-provider", False)
    with pytest.raises(ConfigurationError, match="disabled after routing"):
        await provider.generate(_generation())
    assert requests == []
    await providers.aclose()
    await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["model_disabled", "provider_unavailable"])
async def test_route_binding_revalidates_model_and_health_races_before_network_call(
    mutation: str,
) -> None:
    requests: list[httpx.Request] = []
    registry, providers, _secrets, client, session_factory = _stack(
        handler=lambda request: requests.append(request) or _compatible_response()
    )
    _record_probe(registry)
    route = _decision(registry)
    provider = providers.get(
        route.selected_provider or "",
        model_id=route.selected_model_id,
        expected_provider_config_digest=route.provider_config_digest,
        expected_model_config_digest=route.model_config_digest,
        expected_probe_id=route.capability_probe_id,
        expected_probe_digest=route.capability_probe_digest,
    )
    if mutation == "model_disabled":
        registry.set_model_enabled("audit-model", False)
    else:
        with session_scope(session_factory) as session:
            row = session.get(ProviderEndpointRecordModel, "audit-provider")
            assert row is not None
            row.health_status = ProviderHealthStatus.UNAVAILABLE.value
    with pytest.raises(ConfigurationError):
        await provider.generate(_generation())
    assert requests == []
    await providers.aclose()
    await client.aclose()


@pytest.mark.asyncio
async def test_route_binding_revalidates_removed_credential_before_network_call() -> None:
    requests: list[httpx.Request] = []
    registry, providers, secrets, client, _session_factory = _stack(
        handler=lambda request: requests.append(request) or _compatible_response()
    )
    _record_probe(registry)
    route = _decision(registry)
    provider = providers.get(
        route.selected_provider or "",
        model_id=route.selected_model_id,
        expected_provider_config_digest=route.provider_config_digest,
        expected_model_config_digest=route.model_config_digest,
        expected_probe_id=route.capability_probe_id,
        expected_probe_digest=route.capability_probe_digest,
    )
    del secrets["AUDIT_KEY"]
    with pytest.raises(ConfigurationError, match="credential was removed"):
        await provider.generate(_generation())
    assert requests == []
    await providers.aclose()
    await client.aclose()


@pytest.mark.asyncio
async def test_same_cached_provider_uses_rotated_secret_value() -> None:
    authorizations: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        authorizations.append(request.headers["authorization"])
        return _compatible_response()

    registry, providers, secrets, client, _session_factory = _stack(handler=handler)
    _record_probe(registry)
    route = _decision(registry)
    provider = providers.get(
        route.selected_provider or "",
        model_id=route.selected_model_id,
        expected_provider_config_digest=route.provider_config_digest,
        expected_model_config_digest=route.model_config_digest,
        expected_probe_id=route.capability_probe_id,
        expected_probe_digest=route.capability_probe_digest,
    )
    await provider.generate(_generation())
    secrets["AUDIT_KEY"] = "secret-b"
    await provider.generate(_generation())
    assert authorizations == ["Bearer secret-a", "Bearer secret-b"]
    assert "secret-a" not in _endpoint().config_digest
    assert "secret-b" not in _endpoint().config_digest
    await providers.aclose()
    await client.aclose()


@pytest.mark.asyncio
async def test_validated_dns_address_is_pinned_with_original_host_and_tls_sni() -> None:
    requests: list[httpx.Request] = []
    registry, providers, _secrets, client, _session_factory = _stack(
        handler=lambda request: requests.append(request) or _compatible_response()
    )
    _record_probe(registry)
    route = _decision(registry)
    provider = providers.get(
        route.selected_provider or "",
        model_id=route.selected_model_id,
        expected_provider_config_digest=route.provider_config_digest,
        expected_model_config_digest=route.model_config_digest,
        expected_probe_id=route.capability_probe_id,
        expected_probe_digest=route.capability_probe_digest,
    )
    await provider.generate(_generation())
    assert len(requests) == 1
    assert requests[0].url.host == "93.184.216.34"
    assert requests[0].headers["host"] == "audit.example.test"
    assert requests[0].extensions["sni_hostname"] == "audit.example.test"
    await providers.aclose()
    await client.aclose()


@pytest.mark.asyncio
async def test_new_probe_invalidates_an_already_bound_provider() -> None:
    requests: list[httpx.Request] = []
    registry, providers, _secrets, client, _session_factory = _stack(
        handler=lambda request: requests.append(request) or _compatible_response()
    )
    _record_probe(registry)
    route = _decision(registry)
    provider = providers.get(
        route.selected_provider or "",
        model_id=route.selected_model_id,
        expected_provider_config_digest=route.provider_config_digest,
        expected_model_config_digest=route.model_config_digest,
        expected_probe_id=route.capability_probe_id,
        expected_probe_digest=route.capability_probe_digest,
    )
    _record_probe(registry, json_schema=CapabilityStatus.UNSUPPORTED)
    with pytest.raises(ConfigurationError, match="evidence changed"):
        await provider.generate(_generation())
    assert requests == []
    await providers.aclose()
    await client.aclose()


@pytest.mark.asyncio
async def test_probe_rejects_fake_tool_name() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        if payload.get("stream"):
            return httpx.Response(200, content=b"data: [DONE]\n\n")
        if payload.get("tools"):
            return _compatible_response("", tool_name="arbitrary_side_effect")
        if payload.get("response_format"):
            return _compatible_response('{"ok":true,"value":1}')
        return _compatible_response()

    registry, providers, _secrets, client, _session_factory = _stack(handler=handler)
    model = registry.get_model("audit-model")
    registry.update_model(
        "audit-model",
        {
            "declared_capabilities": {
                **model.declared_capabilities,
                ModelCapability.TOOLS: CapabilityEvidence(
                    status=CapabilityStatus.SUPPORTED,
                    source=CapabilitySource.USER_DECLARED,
                ),
            },
            "tool_calling_strategy": "NATIVE_TOOLS",
        },
    )
    probe = await ProviderCompatibilityProbe(configurations=registry, providers=providers).run(
        "audit-model"
    )
    assert probe.capabilities[ModelCapability.TOOLS].status is CapabilityStatus.UNSUPPORTED
    await providers.aclose()
    await client.aclose()


@pytest.mark.asyncio
async def test_partial_probe_timeout_leaves_untested_capability_unknown() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        if payload.get("response_format"):
            raise httpx.ReadTimeout("simulated", request=request)
        if payload.get("stream"):
            return httpx.Response(200, content=b"data: [DONE]\n\n")
        return _compatible_response()

    registry, providers, _secrets, client, _session_factory = _stack(handler=handler)
    probe = await ProviderCompatibilityProbe(configurations=registry, providers=providers).run(
        "audit-model"
    )
    assert probe.authentication_status is ProbeAuthenticationStatus.PASS
    assert probe.capabilities[ModelCapability.STRUCTURED_OUTPUT].status is CapabilityStatus.UNKNOWN
    assert probe.capabilities[ModelCapability.JSON_SCHEMA].status is CapabilityStatus.UNKNOWN
    await providers.aclose()
    await client.aclose()


@pytest.mark.asyncio
async def test_gzip_decompression_limit_applies_to_decoded_bytes() -> None:
    compressed = gzip.compress(b"x" * 5000)
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200, content=compressed, headers={"content-encoding": "gzip"}
            )
        ),
        base_url="https://audit.example.test/v1",
    )
    provider = OpenAICompatibleProvider(
        endpoint=_endpoint(max_response_bytes=1024),
        model_profile=_model(),
        api_key="test-secret",
        security_policy=_policy(),
        client=client,
    )
    with pytest.raises(ProviderResponseTooLargeError, match="RESPONSE_TOO_LARGE"):
        await provider.generate(_generation())
    await provider.aclose()
    await client.aclose()


@pytest.mark.asyncio
async def test_connect_timeout_is_classified_and_cancellation_is_not_timeout() -> None:
    def connect_timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("simulated", request=request)

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(connect_timeout),
        base_url="https://audit.example.test/v1",
    )
    provider = OpenAICompatibleProvider(
        endpoint=_endpoint(),
        model_profile=_model(),
        api_key="test-secret",
        security_policy=_policy(),
        client=client,
    )
    with pytest.raises(ProviderTimeoutError, match="PROVIDER_TIMEOUT_CONNECT"):
        await provider.generate(_generation())
    await client.aclose()

    started = asyncio.Event()

    async def wait_forever(_request: httpx.Request) -> httpx.Response:
        started.set()
        await asyncio.Event().wait()
        return _compatible_response()  # pragma: no cover

    cancel_client = httpx.AsyncClient(
        transport=httpx.MockTransport(wait_forever),
        base_url="https://audit.example.test/v1",
    )
    cancel_provider = OpenAICompatibleProvider(
        endpoint=_endpoint(),
        model_profile=_model(),
        api_key="test-secret",
        security_policy=_policy(),
        client=cancel_client,
    )
    task = asyncio.create_task(cancel_provider.generate(_generation()))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await cancel_client.aclose()


@pytest.mark.asyncio
async def test_invalid_remote_usage_is_normalized_to_unknown() -> None:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                json={
                    "id": "usage-spoof",
                    "model": "claimed-model",
                    "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": -1,
                        "completion_tokens": 1_000_000_001,
                        "total_tokens": True,
                    },
                },
            )
        ),
        base_url="https://audit.example.test/v1",
    )
    provider = OpenAICompatibleProvider(
        endpoint=_endpoint(),
        model_profile=_model(),
        api_key="test-secret",
        security_policy=_policy(),
        client=client,
    )
    response = await provider.generate(_generation())
    assert response.usage.input_tokens is None
    assert response.usage.output_tokens is None
    assert response.usage.total_tokens is None
    await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["almost JSON", '{"value":0}'])
async def test_prompt_json_fallback_rejects_malformed_and_semantically_invalid_json(
    content: str,
) -> None:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: _compatible_response(content)),
        base_url="https://audit.example.test/v1",
    )
    model = _model(
        structured_output_strategy=StructuredOutputStrategy.PROMPT_JSON_FALLBACK,
        declared_capabilities={
            ModelCapability.TEXT: CapabilityEvidence(
                status=CapabilityStatus.SUPPORTED,
                source=CapabilitySource.USER_DECLARED,
            )
        },
    )
    provider = OpenAICompatibleProvider(
        endpoint=_endpoint(),
        model_profile=model,
        api_key="test-secret",
        security_policy=_policy(),
        client=client,
    )
    with pytest.raises(ProviderResponseError, match="schema validation"):
        await provider.structured_generate(_generation(), _PositiveAnswer)
    await client.aclose()


@pytest.mark.asyncio
async def test_remote_error_headers_cannot_inject_secret_into_exception() -> None:
    secret = "REMOTE_REQUEST_SECRET"
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                500,
                headers={"request-id": f"Authorization: Bearer {secret}"},
            )
        ),
        base_url="https://audit.example.test/v1",
    )
    provider = OpenAICompatibleProvider(
        endpoint=_endpoint(),
        model_profile=_model(),
        api_key="test-secret",
        security_policy=_policy(),
        client=client,
    )
    with pytest.raises(ProviderError) as error:
        await provider.generate(_generation())
    assert secret not in str(error.value)
    await client.aclose()


def test_secret_shaped_exception_is_redacted() -> None:
    secret = "TEST_SECRET_VALUE"
    rendered = safe_error(RuntimeError(f"401 Authorization: Bearer {secret}"))
    assert secret not in rendered
    assert "[REDACTED]" in rendered


def test_probe_history_is_retained_and_latest_tie_break_is_deterministic() -> None:
    registry, providers, _secrets, client, session_factory = _stack()
    timestamp = datetime(2026, 8, 31, tzinfo=UTC)
    first = _record_probe(registry)
    second = _record_probe(registry, json_schema=CapabilityStatus.UNSUPPORTED)
    with session_scope(session_factory) as session:
        rows = list(
            session.scalars(
                select(CapabilityProbeRunRecordModel).order_by(CapabilityProbeRunRecordModel.id)
            )
        )
        assert len(rows) == 2
        for row in rows:
            row.performed_at = timestamp
            probe = first if row.id == first.probe_id else second
            payload = probe.model_copy(update={"performed_at": timestamp, "probe_digest": ""})
            row.probe_digest = CapabilityProbeResult.model_validate(payload).probe_digest
    latest = registry.latest_probe("audit-model")
    assert latest is not None
    assert latest.probe_id == max(first.probe_id, second.probe_id)
    asyncio.run(providers.aclose())
    asyncio.run(client.aclose())
