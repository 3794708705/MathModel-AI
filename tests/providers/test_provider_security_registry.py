from collections.abc import Iterable

import pytest
from pydantic import ValidationError
from sqlalchemy import select

from mathmodel_ai.core.config import Settings
from mathmodel_ai.core.errors import ConfigurationError, ProviderError
from mathmodel_ai.core.types import Environment
from mathmodel_ai.db.base import Base
from mathmodel_ai.db.models import ProviderEndpointRecordModel
from mathmodel_ai.db.session import (
    create_database_engine,
    create_session_factory,
    session_scope,
)
from mathmodel_ai.providers.repository import ProviderModelRegistry
from mathmodel_ai.providers.secrets import EnvironmentSecretResolver
from mathmodel_ai.providers.security import (
    EndpointSecurityPolicy,
    redact_sensitive_text,
    safe_error,
)
from mathmodel_ai.schemas.provider_registry import (
    CapabilityEvidence,
    CapabilitySource,
    CapabilityStatus,
    EndpointTrustLevel,
    ModelCapability,
    ModelProfile,
    ProviderEndpoint,
    ProviderProtocol,
    canonical_provider_config,
    compute_provider_config_digest,
)


def _endpoint(**changes: object) -> ProviderEndpoint:
    payload: dict[str, object] = {
        "provider_id": "secure-provider",
        "display_name": "Secure Provider",
        "protocol": ProviderProtocol.OPENAI_CHAT_COMPLETIONS,
        "base_url": "https://public.example.test/v1",
        "credential_ref": "env:SECURE_PROVIDER_KEY",
        "trust_level": EndpointTrustLevel.USER_MANAGED_PROXY,
    }
    payload.update(changes)
    return ProviderEndpoint.model_validate(payload)


def _model(**changes: object) -> ModelProfile:
    payload: dict[str, object] = {
        "model_id": "secure-main",
        "provider_id": "secure-provider",
        "display_name": "Secure Main",
        "remote_model": "remote-main",
        "declared_capabilities": {
            ModelCapability.TEXT: CapabilityEvidence(
                status=CapabilityStatus.SUPPORTED,
                source=CapabilitySource.USER_DECLARED,
            )
        },
    }
    payload.update(changes)
    return ModelProfile.model_validate(payload)


def _policy(
    addresses: Iterable[str] = ("93.184.216.34",),
    *,
    allow_local: bool = False,
    allow_insecure_tls: bool = False,
    environment: Environment = Environment.TEST,
) -> EndpointSecurityPolicy:
    return EndpointSecurityPolicy(
        environment=environment,
        allow_local_model_endpoints=allow_local,
        allow_insecure_provider_tls=allow_insecure_tls,
        resolver=lambda _host, _port: addresses,
    )


def _registry(*, environment: dict[str, str] | None = None) -> tuple[ProviderModelRegistry, object]:
    engine = create_database_engine(
        Settings(environment="test", database_url="sqlite+pysqlite:///:memory:")
    )
    Base.metadata.create_all(engine)
    factory = create_session_factory(engine)
    return (
        ProviderModelRegistry(
            factory,
            secrets=EnvironmentSecretResolver(environment or {}),
            security_policy=_policy(),
        ),
        factory,
    )


def test_f_secret_resolver_registry_and_errors_never_expose_secret() -> None:
    secret = "sk-unit-secret-never-persist"
    registry, factory = _registry(environment={"SECURE_PROVIDER_KEY": secret})
    stored = registry.create_provider(_endpoint())
    assert registry.credential_configured(stored.provider_id) is True
    with session_scope(factory) as session:  # type: ignore[arg-type]
        row = session.scalar(select(ProviderEndpointRecordModel))
        assert row is not None
        persisted = repr(
            {
                "credential_ref": row.credential_ref,
                "headers": row.additional_headers,
                "metadata": row.metadata_json,
            }
        )
    assert secret not in persisted
    assert stored.credential_ref == "env:SECURE_PROVIDER_KEY"
    assert secret not in safe_error(RuntimeError(f"Authorization: Bearer {secret}"))
    assert secret not in redact_sensitive_text(f"api_key={secret}")


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8000/v1",
        "https://127.0.0.1/v1",
        "https://localhost/v1",
        "https://[::1]/v1",
        "https://[fc00::1]/v1",
        "https://[::ffff:127.0.0.1]/v1",
    ],
)
def test_g_localhost_is_rejected_by_default(url: str) -> None:
    with pytest.raises(ConfigurationError, match=r"private|local|HTTPS"):
        _policy().validate_configuration(_endpoint(base_url=url))


@pytest.mark.parametrize(
    "address",
    ["169.254.169.254", "10.0.0.1", "172.16.0.1", "192.168.0.1", "::1"],
)
def test_h_metadata_and_private_addresses_are_rejected(address: str) -> None:
    with pytest.raises((ConfigurationError, ProviderError), match=r"private|local"):
        _policy([address]).validate_configuration(_endpoint())


def test_j_local_endpoint_requires_explicit_server_mode() -> None:
    endpoint = _endpoint(
        base_url="http://localhost:8000/v1",
        trust_level=EndpointTrustLevel.LOCAL_ENDPOINT,
    )
    _policy(["127.0.0.1"], allow_local=True).validate_configuration(endpoint)
    with pytest.raises(ConfigurationError):
        _policy(["127.0.0.1"]).validate_configuration(endpoint)


def test_dns_destination_is_rechecked_and_rebinding_is_blocked() -> None:
    answers = iter([("93.184.216.34",), ("127.0.0.1",)])
    policy = EndpointSecurityPolicy(
        environment=Environment.TEST,
        resolver=lambda _host, _port: next(answers),
    )
    endpoint = _endpoint()
    policy.validate_configuration(endpoint)
    with pytest.raises(ProviderError, match="private or local"):
        policy.validate_destination(endpoint)


def test_t_proxy_cannot_claim_official_vendor_or_model_identity() -> None:
    with pytest.raises(ConfigurationError, match="OFFICIAL_VENDOR"):
        _policy().validate_configuration(_endpoint(trust_level=EndpointTrustLevel.OFFICIAL_VENDOR))

    registry, _ = _registry(environment={"SECURE_PROVIDER_KEY": "configured"})
    registry.create_provider(_endpoint())
    with pytest.raises(ConfigurationError, match="cannot claim official trust"):
        registry.create_model(_model(trust_level=EndpointTrustLevel.OFFICIAL_VENDOR))
    stored = registry.create_model(_model())
    assert stored.trust_level is EndpointTrustLevel.USER_MANAGED_PROXY


@pytest.mark.parametrize(
    "header",
    [
        "Authorization",
        "Host",
        "Cookie",
        "Content-Length",
        "Proxy-Authorization",
        "X-Api-Key",
        "Api-Key",
        "X-Auth-Token",
    ],
)
def test_w_reserved_header_injection_is_rejected(header: str) -> None:
    with pytest.raises(ValidationError, match="reserved or invalid"):
        _endpoint(additional_headers={header: "attacker-controlled"})


def test_header_line_break_and_credential_value_are_rejected() -> None:
    with pytest.raises(ValidationError, match="line breaks"):
        _endpoint(additional_headers={"X-Custom": "value\r\nHost: attacker"})
    with pytest.raises(ValidationError, match="credentials"):
        _endpoint(additional_headers={"X-Custom": "Bearer sk-attacker-secret-value"})


def test_r_provider_and_model_digests_track_configuration_not_secret_values() -> None:
    baseline = _endpoint()
    assert baseline.config_digest != _endpoint(base_url="https://other.example.test").config_digest
    assert (
        baseline.config_digest
        != _endpoint(protocol=ProviderProtocol.CUSTOM_JSON_HTTP).config_digest
    )
    first_resolver = EnvironmentSecretResolver({"SECURE_PROVIDER_KEY": "secret-one"})
    second_resolver = EnvironmentSecretResolver({"SECURE_PROVIDER_KEY": "secret-two"})
    assert first_resolver.resolve(baseline.credential_ref or "").get_secret_value() != (
        second_resolver.resolve(baseline.credential_ref or "").get_secret_value()
    )
    assert baseline.config_digest == _endpoint().config_digest

    model = _model()
    assert model.config_digest != _model(remote_model="other-remote").config_digest
    assert (
        model.config_digest
        != _model(
            declared_capabilities={
                ModelCapability.TEXT: CapabilityEvidence(
                    status=CapabilityStatus.UNSUPPORTED,
                    source=CapabilitySource.USER_DECLARED,
                )
            }
        ).config_digest
    )

    tampered = baseline.model_dump(mode="python")
    tampered["base_url"] = "https://tampered.example.test"
    with pytest.raises(ValidationError, match="config_digest"):
        ProviderEndpoint.model_validate(tampered)


def test_provider_canonical_digest_is_stable_across_defaults_and_roundtrip_shapes() -> None:
    baseline = _endpoint(base_url="https://public.example.test/v1/")
    explicit = _endpoint(
        base_url="https://public.example.test/v1",
        enabled=True,
        timeout_seconds=60,
        connect_timeout_seconds=10.0,
        max_response_bytes=8 * 1024 * 1024,
        verify_tls=True,
        allow_redirects=False,
        additional_headers={},
        metadata={},
    )
    enum_strings = _endpoint(
        protocol="openai_chat_completions",
        trust_level="USER_MANAGED_PROXY",
    )
    ordered_headers = _endpoint(
        additional_headers={"X-Zeta": "z", "X-Alpha": "a"},
    )
    reversed_headers = _endpoint(
        additional_headers={"X-Alpha": "a", "X-Zeta": "z"},
    )
    without_ref = ProviderEndpoint(
        provider_id="no-ref-provider",
        display_name="No Ref Provider",
        protocol=ProviderProtocol.OPENAI_CHAT_COMPLETIONS,
        base_url="https://public.example.test/v1",
        trust_level=EndpointTrustLevel.USER_MANAGED_PROXY,
    )
    explicit_none_ref = ProviderEndpoint(
        provider_id="no-ref-provider",
        display_name="No Ref Provider",
        protocol="openai_chat_completions",
        base_url="https://public.example.test/v1/",
        credential_ref=None,
        trust_level="USER_MANAGED_PROXY",
        additional_headers={},
        metadata={},
    )
    runtime_state = baseline.model_dump(mode="python")
    runtime_state.update(
        {
            "health_status": "READY",
            "consecutive_failures": 3,
            "config_digest": "",
        }
    )
    after_runtime_state_change = ProviderEndpoint.model_validate(runtime_state)

    assert baseline.base_url == "https://public.example.test/v1"
    assert baseline.config_digest == explicit.config_digest == enum_strings.config_digest
    assert ordered_headers.config_digest == reversed_headers.config_digest
    assert without_ref.config_digest == explicit_none_ref.config_digest
    assert after_runtime_state_change.config_digest == baseline.config_digest
    assert baseline.digest_payload() == canonical_provider_config(baseline)
    assert baseline.config_digest == compute_provider_config_digest(baseline)
    assert {
        "health_status",
        "consecutive_failures",
        "cooldown_until",
        "created_at",
        "updated_at",
    }.isdisjoint(canonical_provider_config(baseline))


def test_policy_blocked_provider_remains_manageable_but_not_executable() -> None:
    engine = create_database_engine(
        Settings(environment="test", database_url="sqlite+pysqlite:///:memory:")
    )
    Base.metadata.create_all(engine)
    factory = create_session_factory(engine)
    secrets = EnvironmentSecretResolver({"SECURE_PROVIDER_KEY": "configured"})
    allowed_registry = ProviderModelRegistry(
        factory,
        secrets=secrets,
        security_policy=_policy(["127.0.0.1"], allow_local=True),
    )
    endpoint = allowed_registry.create_provider(
        _endpoint(
            base_url="http://127.0.0.1:18081/v1/",
            trust_level=EndpointTrustLevel.LOCAL_ENDPOINT,
        )
    )
    blocked_registry = ProviderModelRegistry(
        factory,
        secrets=secrets,
        security_policy=_policy(["127.0.0.1"], allow_local=False),
    )

    managed = blocked_registry.get_provider_for_management(endpoint.provider_id)
    assert managed.config_digest == endpoint.config_digest
    assert blocked_registry.list_providers_for_management() == [managed]
    assert blocked_registry.runtime_policy_allows(managed) is False
    with pytest.raises(ConfigurationError, match="blocked by the current security policy"):
        blocked_registry.get_provider(endpoint.provider_id)

    with session_scope(factory) as session:
        row = session.get(ProviderEndpointRecordModel, endpoint.provider_id)
        assert row is not None
        row.base_url = "https://tampered.example.test/v1"
    with pytest.raises(ConfigurationError, match="integrity checks"):
        blocked_registry.get_provider_for_management(endpoint.provider_id)


def test_base_url_cannot_embed_credentials_query_or_non_http_scheme() -> None:
    with pytest.raises(ValidationError, match="credentials"):
        _endpoint(base_url="https://user:password@public.example.test/v1")
    with pytest.raises(ValidationError, match="query"):
        _endpoint(base_url="https://public.example.test/v1?token=secret")
    with pytest.raises(ConfigurationError, match="HTTPS"):
        _policy().validate_configuration(_endpoint(base_url="ftp://public.example.test/v1"))


def test_unknown_protocol_is_rejected_by_schema() -> None:
    payload = _endpoint().model_dump(mode="python")
    payload.update({"protocol": "generic_http", "config_digest": ""})
    with pytest.raises(ValidationError, match="protocol"):
        ProviderEndpoint.model_validate(payload)


def test_tls_verification_requires_explicit_nonproduction_server_override() -> None:
    endpoint = _endpoint(verify_tls=False)
    with pytest.raises(ConfigurationError, match="TLS verification"):
        _policy().validate_configuration(endpoint)
    _policy(allow_insecure_tls=True).validate_configuration(endpoint)
    with pytest.raises(ConfigurationError, match="TLS verification"):
        _policy(
            allow_insecure_tls=True,
            environment=Environment.PRODUCTION,
        ).validate_configuration(endpoint)
