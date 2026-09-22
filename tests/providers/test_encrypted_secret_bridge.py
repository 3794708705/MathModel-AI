import base64
import logging

from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import select

from mathmodel_ai.core.config import Settings
from mathmodel_ai.core.types import Environment
from mathmodel_ai.db.base import Base
from mathmodel_ai.db.models import EncryptedSecretRecordModel
from mathmodel_ai.db.session import session_scope
from mathmodel_ai.main import create_app
from mathmodel_ai.providers.secrets import (
    CompositeSecretResolver,
    EncryptedDatabaseSecretStore,
    EnvironmentSecretResolver,
    stored_secret_id,
)
from mathmodel_ai.providers.security import EndpointSecurityPolicy
from mathmodel_ai.schemas.provider_registry import (
    CapabilityEvidence,
    CapabilityProbeResult,
    CapabilitySource,
    CapabilityStatus,
    ModelCapability,
    ProbeAuthenticationStatus,
)

MASTER_KEY = base64.urlsafe_b64encode(b"phase-8-2-test-master-key-32b!!!").decode()


def _app(*, master_key: str | None = MASTER_KEY, database_echo: bool = False) -> object:
    app = create_app(
        Settings(
            environment="test",
            database_url="sqlite+pysqlite:///:memory:",
            database_echo=database_echo,
            default_provider="mock",
            secret_master_key=master_key,
        ),
        provider_security_policy=EndpointSecurityPolicy(
            environment=Environment.TEST,
            resolver=lambda _host, _port: ["93.184.216.34"],
        ),
    )
    Base.metadata.create_all(app.state.engine)
    return app


def _provider() -> dict[str, object]:
    return {
        "provider_id": "secret-provider",
        "display_name": "Secret Provider",
        "protocol": "openai_chat_completions",
        "base_url": "https://secret-provider.example.test/v1",
        "trust_level": "USER_MANAGED_PROXY",
    }


def _model() -> dict[str, object]:
    return {
        "model_id": "secret-model",
        "provider_id": "secret-provider",
        "display_name": "Secret Model",
        "remote_model": "remote-secret-model",
        "quality_tier": "FLAGSHIP_MAX",
        "structured_output_strategy": "NATIVE_JSON_SCHEMA",
    }


def test_secret_bridge_never_persists_returns_or_logs_plaintext(caplog) -> None:
    secret = "sk-phase82-plaintext-must-never-leak"
    caplog.set_level(logging.DEBUG)
    app = _app(database_echo=True)
    with TestClient(app) as client:
        assert client.post("/api/v1/providers", json=_provider()).status_code == 201
        response = client.put(
            "/api/v1/providers/secret-provider/credential",
            json={"api_key": secret},
        )
        provider = client.get("/api/v1/providers/secret-provider")
        with session_scope(app.state.session_factory) as session:
            row = session.scalar(select(EncryptedSecretRecordModel))
            assert row is not None
            assert secret.encode() not in bytes(row.ciphertext)
            assert secret.encode() not in bytes(row.nonce)
            assert not hasattr(row, "plaintext")
    assert response.status_code == 200, response.text
    assert response.json() == {"credential_configured": True}
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["pragma"] == "no-cache"
    assert "credential_ref" not in provider.text
    assert secret not in response.text
    assert secret not in provider.text
    assert secret not in caplog.text


def test_secret_rotation_invalidates_probe_and_deletes_old_ciphertext() -> None:
    app = _app()
    with TestClient(app) as client:
        assert client.post("/api/v1/providers", json=_provider()).status_code == 201
        assert client.post("/api/v1/models", json=_model()).status_code == 201
        assert (
            client.put(
                "/api/v1/providers/secret-provider/credential",
                json={"api_key": "first-secret-value"},
            ).status_code
            == 200
        )
        registry = app.state.provider_configurations
        first_endpoint = registry.get_provider("secret-provider")
        first_ref = first_endpoint.credential_ref
        assert first_ref is not None
        model = registry.get_model("secret-model")
        registry.record_probe(
            CapabilityProbeResult(
                provider_id=first_endpoint.provider_id,
                model_id=model.model_id,
                provider_config_digest=first_endpoint.config_digest,
                model_config_digest=model.config_digest,
                capabilities={
                    ModelCapability.TEXT: CapabilityEvidence(
                        status=CapabilityStatus.SUPPORTED,
                        source=CapabilitySource.PROBED,
                    )
                },
                authentication_status=ProbeAuthenticationStatus.PASS,
                latency_ms=1,
            )
        )
        assert registry.latest_probe("secret-model", current_only=True) is not None

        rotated = client.put(
            "/api/v1/providers/secret-provider/credential",
            json={"api_key": "second-secret-value"},
        )
        assert rotated.json() == {"credential_configured": True}
        second_endpoint = registry.get_provider("secret-provider")
        assert second_endpoint.credential_ref not in {None, first_ref}
        assert second_endpoint.health_status.value == "UNAVAILABLE"
        assert registry.latest_probe("secret-model", current_only=True) is None
        assert app.state.secret_store.contains(stored_secret_id(first_ref)) is False
        assert (
            app.state.secret_resolver.resolve(second_endpoint.credential_ref).get_secret_value()
            == "second-secret-value"
        )


def test_secret_deletion_clears_reference_and_ciphertext() -> None:
    app = _app()
    with TestClient(app) as client:
        client.post("/api/v1/providers", json=_provider())
        client.put(
            "/api/v1/providers/secret-provider/credential",
            json={"api_key": "delete-me-secret"},
        )
        credential_ref = app.state.provider_configurations.get_provider(
            "secret-provider"
        ).credential_ref
        assert credential_ref is not None
        response = client.delete("/api/v1/providers/secret-provider/credential")
        public = client.get("/api/v1/providers/secret-provider")
        assert response.json() == {"credential_configured": False}
        assert public.json()["credential_configured"] is False
        endpoint = app.state.provider_configurations.get_provider("secret-provider")
        assert endpoint.credential_ref is None
        assert endpoint.health_status.value == "UNCONFIGURED"
        assert app.state.secret_store.contains(stored_secret_id(credential_ref)) is False


def test_secret_write_without_master_key_fails_closed_without_echo() -> None:
    app = _app(master_key=None)
    secret = "secret-without-server-key"
    with TestClient(app, raise_server_exceptions=False) as client:
        client.post("/api/v1/providers", json=_provider())
        response = client.put(
            "/api/v1/providers/secret-provider/credential",
            json={"api_key": secret},
        )
        assert response.status_code == 400
        assert response.headers["cache-control"] == "no-store"
        assert "MM_SECRET_MASTER_KEY" in response.text
        assert secret not in response.text
        assert (
            app.state.provider_configurations.get_provider("secret-provider").credential_ref is None
        )


def test_environment_secret_resolution_remains_supported() -> None:
    resolver = EnvironmentSecretResolver({"PHASE82_ENV_KEY": "environment-secret"})
    assert resolver.is_configured("env:PHASE82_ENV_KEY") is True
    assert resolver.resolve("env:PHASE82_ENV_KEY").get_secret_value() == "environment-secret"


def test_tampered_ciphertext_and_wrong_master_key_are_not_reported_as_configured() -> None:
    app = _app()
    with TestClient(app) as client:
        assert client.post("/api/v1/providers", json=_provider()).status_code == 201
        assert client.post("/api/v1/models", json=_model()).status_code == 201
        assert (
            client.put(
                "/api/v1/providers/secret-provider/credential",
                json={"api_key": "encrypted-secret-value"},
            ).status_code
            == 200
        )
        registry = app.state.provider_configurations
        endpoint = registry.get_provider("secret-provider")
        model = registry.get_model("secret-model")
        registry.record_probe(
            CapabilityProbeResult(
                provider_id=endpoint.provider_id,
                model_id=model.model_id,
                provider_config_digest=endpoint.config_digest,
                model_config_digest=model.config_digest,
                capabilities={
                    ModelCapability.STRUCTURED_OUTPUT: CapabilityEvidence(
                        status=CapabilityStatus.SUPPORTED,
                        source=CapabilitySource.PROBED,
                    )
                },
                authentication_status=ProbeAuthenticationStatus.PASS,
                latency_ms=1,
            )
        )
        assert registry.get_provider("secret-provider").health_status.value == "READY"
        credential_ref = registry.get_provider("secret-provider").credential_ref
        assert credential_ref is not None

        wrong_key = base64.urlsafe_b64encode(bytes(range(32))).decode()
        wrong_store = EncryptedDatabaseSecretStore(
            app.state.session_factory,
            SecretStr(wrong_key),
        )
        wrong_resolver = CompositeSecretResolver(EnvironmentSecretResolver({}), wrong_store)
        assert wrong_resolver.is_configured(credential_ref) is False

        with session_scope(app.state.session_factory) as session:
            row = session.scalar(select(EncryptedSecretRecordModel))
            assert row is not None
            ciphertext = bytearray(row.ciphertext)
            ciphertext[-1] ^= 1
            row.ciphertext = bytes(ciphertext)

        public = client.get("/api/v1/providers/secret-provider")
        assert public.status_code == 200
        assert public.json()["credential_configured"] is False
        assert public.json()["health_status"] == "UNCONFIGURED"
        assert registry.credential_configured("secret-provider") is False
