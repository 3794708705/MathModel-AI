import json

import httpx
from fastapi.testclient import TestClient

from mathmodel_ai.core.config import Settings
from mathmodel_ai.core.types import Environment
from mathmodel_ai.db.base import Base
from mathmodel_ai.main import create_app
from mathmodel_ai.providers.discovery import ProviderModelDiscovery
from mathmodel_ai.providers.repository import ProviderModelRegistry
from mathmodel_ai.providers.security import EndpointSecurityPolicy
from mathmodel_ai.schemas.provider_registry import (
    CapabilityEvidence,
    CapabilityProbeResult,
    CapabilitySource,
    CapabilityStatus,
    ModelCapability,
    ProbeAuthenticationStatus,
)


def _app() -> object:
    app = create_app(
        Settings(
            environment="test",
            database_url="sqlite+pysqlite:///:memory:",
            default_provider="mock",
        ),
        provider_security_policy=EndpointSecurityPolicy(
            environment=Environment.TEST,
            resolver=lambda _host, _port: ["93.184.216.34"],
        ),
    )
    Base.metadata.create_all(app.state.engine)
    return app


def _provider_payload() -> dict[str, object]:
    return {
        "provider_id": "api-provider",
        "display_name": "API Provider",
        "protocol": "openai_chat_completions",
        "base_url": "https://api-provider.example.test/v1",
        "credential_ref": "env:API_PROVIDER_KEY",
        "trust_level": "USER_MANAGED_PROXY",
    }


def _model_payload() -> dict[str, object]:
    evidence = {"status": "SUPPORTED", "source": "USER_DECLARED"}
    return {
        "model_id": "api-model",
        "provider_id": "api-provider",
        "display_name": "API Model",
        "remote_model": "remote-api-model",
        "quality_tier": "FLAGSHIP_MAX",
        "declared_capabilities": {
            "TEXT": evidence,
            "STRUCTURED_OUTPUT": evidence,
            "JSON_SCHEMA": evidence,
        },
        "structured_output_strategy": "NATIVE_JSON_SCHEMA",
    }


def test_provider_model_and_route_apis_are_secret_safe(monkeypatch) -> None:
    secret = "sk-api-secret-never-return"
    monkeypatch.setenv("API_PROVIDER_KEY", secret)
    app = _app()
    with TestClient(app) as client:
        created_provider = client.post("/api/v1/providers", json=_provider_payload())
        assert created_provider.status_code == 201, created_provider.text
        provider_json = created_provider.json()
        assert provider_json["credential_configured"] is True
        assert "credential_ref" not in provider_json
        assert secret not in created_provider.text

        created_model = client.post("/api/v1/models", json=_model_payload())
        assert created_model.status_code == 201, created_model.text
        model_json = created_model.json()
        assert model_json["model_id"] == "api-model"
        assert model_json["remote_model"] == "remote-api-model"
        assert secret not in created_model.text

        endpoint = app.state.provider_configurations.get_provider("api-provider")
        model = app.state.provider_configurations.get_model("api-model")
        app.state.provider_configurations.record_probe(
            CapabilityProbeResult(
                provider_id=endpoint.provider_id,
                model_id=model.model_id,
                provider_config_digest=endpoint.config_digest,
                model_config_digest=model.config_digest,
                capabilities={
                    capability: CapabilityEvidence(
                        status=CapabilityStatus.SUPPORTED,
                        source=CapabilitySource.PROBED,
                    )
                    for capability in (
                        ModelCapability.TEXT,
                        ModelCapability.STRUCTURED_OUTPUT,
                        ModelCapability.JSON_SCHEMA,
                    )
                },
                authentication_status=ProbeAuthenticationStatus.PASS,
                latency_ms=1,
            )
        )

        route = client.put(
            "/api/v1/model-routing/agents/problem_agent",
            json={"primary_model_id": "api-model", "fallback_model_ids": []},
        )
        assert route.status_code == 200, route.text
        preview = client.post(
            "/api/v1/model-routing/preview",
            json={
                "agent_name": "problem_agent",
                "profile": {"task_type": "problem_understanding"},
            },
        )
        assert preview.status_code == 200, preview.text
        assert preview.json()["provider_called"] is False
        assert preview.json()["decision"]["selected_model_id"] == "api-model"

        listed = client.get("/api/v1/providers")
        models = client.get("/api/v1/models?provider_id=api-provider")
        overview = client.get("/api/v1/model-routing")
        routes = client.get("/api/v1/model-routing/agents")
        all_responses = [listed, models, overview, routes, preview]
        assert all(item.status_code == 200 for item in all_responses)
        assert secret not in json.dumps([item.json() for item in all_responses])


def test_provider_probe_without_credential_is_explicitly_not_configured(monkeypatch) -> None:
    monkeypatch.delenv("API_PROVIDER_KEY", raising=False)
    app = _app()
    with TestClient(app) as client:
        assert client.post("/api/v1/providers", json=_provider_payload()).status_code == 201
        assert client.post("/api/v1/models", json=_model_payload()).status_code == 201
        response = client.post("/api/v1/models/api-model/probe")
    assert response.status_code == 200, response.text
    assert response.json()["authentication_status"] == "NOT_CONFIGURED"
    assert response.json()["errors"] == ["credential reference is not configured"]


def test_provider_validation_errors_do_not_echo_secret_input() -> None:
    app = _app()
    secret = "TEST_SECRET_MUST_NOT_ECHO"
    payload = _provider_payload()
    payload["credential_ref"] = f"Bearer {secret}"
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/api/v1/providers", json=payload)
        url_payload = _provider_payload()
        url_payload["base_url"] = f"https://user:{secret}@example.test/v1"
        url_response = client.post("/api/v1/providers", json=url_payload)
    assert response.status_code == 422
    assert url_response.status_code == 422
    assert secret not in response.text
    assert secret not in url_response.text


def test_provider_registry_openapi_exposes_required_non_destructive_routes() -> None:
    app = _app()
    with TestClient(app) as client:
        presets = {
            item["preset_id"]: item for item in client.get("/api/v1/providers/presets").json()
        }
    assert presets["deepseek_official"]["base_url"] == "https://api.deepseek.com"
    assert presets["qwen_modelstudio"]["base_url"].endswith("/compatible-mode/v1")
    assert {
        "deepseek_official",
        "qwen_modelstudio",
        "openai_official",
        "google_ai",
        "anthropic_official",
    } <= set(presets)
    assert presets["deepseek_official"]["credential_type"] == "API_KEY"
    assert presets["deepseek_official"]["model_hints"] == [
        "deepseek-v4-flash",
        "deepseek-v4-pro",
    ]
    assert presets["deepseek_official"]["capability_hints"]["TEXT"]["status"] == "PARTIAL"
    paths = set(app.openapi()["paths"])
    assert {
        "/api/v1/providers",
        "/api/v1/providers/{provider_id}",
        "/api/v1/providers/{provider_id}/probe",
        "/api/v1/providers/{provider_id}/discover-models",
        "/api/v1/providers/{provider_id}/test-connection",
        "/api/v1/providers/{provider_id}/enable",
        "/api/v1/providers/{provider_id}/disable",
        "/api/v1/models",
        "/api/v1/models/{model_id}",
        "/api/v1/models/{model_id}/probe",
        "/api/v1/model-routing",
        "/api/v1/model-routing/agents",
        "/api/v1/model-routing/agents/{agent_name}",
        "/api/v1/model-routing/preview",
    } <= paths
    assert not any(path.endswith("/delete") for path in paths)


def test_saved_provider_discovery_and_connection_use_structured_errors(monkeypatch) -> None:
    monkeypatch.setenv("API_PROVIDER_KEY", "synthetic-discovery-key")
    app = _app()
    remote_status = 200

    def remote(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer synthetic-discovery-key"
        if remote_status == 200:
            return httpx.Response(200, json={"data": [{"id": "model-a"}, {"id": "model-b"}]})
        return httpx.Response(remote_status)

    app.state.provider_model_discovery = ProviderModelDiscovery(
        secrets=app.state.secret_resolver,
        security_policy=app.state.provider_security_policy,
        client=httpx.AsyncClient(transport=httpx.MockTransport(remote)),
    )
    with TestClient(app) as client:
        created = client.post("/api/v1/providers", json=_provider_payload())
        assert created.status_code == 201
        provider_digest = created.json()["config_digest"]
        discovered = client.post("/api/v1/providers/api-provider/discover-models")
        assert discovered.status_code == 200
        assert [item["remote_model_id"] for item in discovered.json()["models"]] == [
            "model-a",
            "model-b",
        ]
        assert discovered.json()["capabilities_probed"] is False

        remote_status = 404
        unsupported = client.post("/api/v1/providers/api-provider/discover-models")
        assert unsupported.status_code == 409
        assert unsupported.json()["detail"]["code"] == "MODEL_DISCOVERY_UNSUPPORTED"
        connection = client.post("/api/v1/providers/api-provider/test-connection")
        assert connection.status_code == 200
        assert connection.json()["model_discovery_supported"] is False

        remote_status = 401
        rejected = client.post("/api/v1/providers/api-provider/discover-models")
        assert rejected.status_code == 401
        assert rejected.json()["detail"]["code"] == "AUTHENTICATION_FAILED"
        assert "synthetic-discovery-key" not in rejected.text
        assert (
            client.get("/api/v1/providers/api-provider").json()["config_digest"] == provider_digest
        )


def test_policy_blocked_provider_does_not_poison_management_or_weaken_execution(
    monkeypatch,
) -> None:
    monkeypatch.setenv("API_PROVIDER_KEY", "local-policy-test-key")
    allowed_policy = EndpointSecurityPolicy(
        environment=Environment.TEST,
        allow_local_model_endpoints=True,
        resolver=lambda host, _port: ["127.0.0.1" if host == "127.0.0.1" else "93.184.216.34"],
    )
    app = create_app(
        Settings(
            environment="test",
            database_url="sqlite+pysqlite:///:memory:",
            default_provider="mock",
        ),
        provider_security_policy=allowed_policy,
    )
    Base.metadata.create_all(app.state.engine)
    payload = _provider_payload()
    payload.update(
        {
            "provider_id": "local-policy-provider",
            "display_name": "Local Policy Provider",
            "base_url": "http://127.0.0.1:18081/v1/",
            "trust_level": "LOCAL_ENDPOINT",
        }
    )
    with TestClient(app) as client:
        created = client.post("/api/v1/providers", json=payload)
        assert created.status_code == 201, created.text
        original_digest = created.json()["config_digest"]

        blocked_policy = EndpointSecurityPolicy(
            environment=Environment.TEST,
            allow_local_model_endpoints=False,
            resolver=lambda host, _port: ["127.0.0.1" if host == "127.0.0.1" else "93.184.216.34"],
        )
        app.state.provider_configurations = ProviderModelRegistry(
            app.state.session_factory,
            secrets=app.state.secret_resolver,
            security_policy=blocked_policy,
        )

        listed = client.get("/api/v1/providers")
        fetched = client.get("/api/v1/providers/local-policy-provider")
        blocked_discovery = client.post("/api/v1/providers/local-policy-provider/discover-models")
        assert listed.status_code == 200, listed.text
        assert fetched.status_code == 200, fetched.text
        assert listed.json()[0]["config_digest"] == original_digest
        assert fetched.json()["health_status"] == "UNAVAILABLE"
        assert blocked_discovery.status_code == 400
        assert "blocked by the current security policy" in blocked_discovery.text
        assert "integrity checks" not in blocked_discovery.text

        repaired = client.patch(
            "/api/v1/providers/local-policy-provider",
            json={
                "base_url": "https://public.example.test/v1",
                "trust_level": "USER_MANAGED_PROXY",
            },
        )
        assert repaired.status_code == 200, repaired.text
        assert repaired.json()["config_digest"] != original_digest
        assert (
            app.state.provider_configurations.get_provider("local-policy-provider").base_url
            == "https://public.example.test/v1"
        )


def test_agent_route_save_revalidates_after_preview_and_rejects_unknown_agent(
    monkeypatch,
) -> None:
    monkeypatch.setenv("API_PROVIDER_KEY", "route-race-secret")
    app = _app()
    with TestClient(app) as client:
        assert client.post("/api/v1/providers", json=_provider_payload()).status_code == 201
        assert client.post("/api/v1/models", json=_model_payload()).status_code == 201
        registry = app.state.provider_configurations
        endpoint = registry.get_provider("api-provider")
        model = registry.get_model("api-model")
        registry.record_probe(
            CapabilityProbeResult(
                provider_id=endpoint.provider_id,
                model_id=model.model_id,
                provider_config_digest=endpoint.config_digest,
                model_config_digest=model.config_digest,
                capabilities={
                    capability: CapabilityEvidence(
                        status=CapabilityStatus.SUPPORTED,
                        source=CapabilitySource.PROBED,
                    )
                    for capability in (
                        ModelCapability.TEXT,
                        ModelCapability.STRUCTURED_OUTPUT,
                        ModelCapability.JSON_SCHEMA,
                    )
                },
                authentication_status=ProbeAuthenticationStatus.PASS,
                latency_ms=1,
            )
        )
        preview = client.post(
            "/api/v1/model-routing/preview",
            json={
                "agent_name": "problem_agent",
                "profile": {
                    "task_type": "problem_understanding",
                    "preferred_model_id": "api-model",
                },
            },
        )
        assert preview.status_code == 200
        assert preview.json()["decision"]["action"] == "EXECUTE"

        assert client.patch("/api/v1/models/api-model", json={"enabled": False}).status_code == 200
        raced = client.put(
            "/api/v1/model-routing/agents/problem_agent",
            json={"primary_model_id": "api-model", "fallback_model_ids": []},
        )
        unknown_agent = client.put(
            "/api/v1/model-routing/agents/invented_agent",
            json={"primary_model_id": "api-model", "fallback_model_ids": []},
        )
        nonexistent_model = client.put(
            "/api/v1/model-routing/agents/problem_agent",
            json={"primary_model_id": "missing-model", "fallback_model_ids": []},
        )
        assert raced.status_code == 400
        assert "route policy rejected" in raced.text
        assert unknown_agent.status_code == 400
        assert "not routable" in unknown_agent.text
        assert nonexistent_model.status_code == 400
        assert client.get("/api/v1/model-routing/agents").json() == []


def test_agent_route_direct_api_rejects_stale_probe_and_mock_provider(monkeypatch) -> None:
    monkeypatch.setenv("API_PROVIDER_KEY", "stale-route-secret")
    monkeypatch.setenv("MOCK_DIRECT_KEY", "mock-route-secret")
    app = _app()
    with TestClient(app) as client:
        assert client.post("/api/v1/providers", json=_provider_payload()).status_code == 201
        assert client.post("/api/v1/models", json=_model_payload()).status_code == 201
        stale = client.put(
            "/api/v1/model-routing/agents/problem_agent",
            json={"primary_model_id": "api-model", "fallback_model_ids": []},
        )
        assert stale.status_code == 400
        assert "route policy rejected" in stale.text

        mock_provider = _provider_payload()
        mock_provider.update(
            {
                "provider_id": "mock",
                "display_name": "Registry Mock",
                "credential_ref": "env:MOCK_DIRECT_KEY",
            }
        )
        mock_model = _model_payload()
        mock_model.update(
            {
                "model_id": "mock-direct-model",
                "provider_id": "mock",
                "display_name": "Mock Direct Model",
            }
        )
        assert client.post("/api/v1/providers", json=mock_provider).status_code == 201
        assert client.post("/api/v1/models", json=mock_model).status_code == 201
        registry = app.state.provider_configurations
        endpoint = registry.get_provider("mock")
        model = registry.get_model("mock-direct-model")
        registry.record_probe(
            CapabilityProbeResult(
                provider_id="mock",
                model_id="mock-direct-model",
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
        mock_route = client.put(
            "/api/v1/model-routing/agents/problem_agent",
            json={"primary_model_id": "mock-direct-model", "fallback_model_ids": []},
        )
        assert mock_route.status_code == 400
        assert "route policy rejected" in mock_route.text


def test_public_provider_api_rejects_service_owned_field_assignment() -> None:
    app = _app()
    with TestClient(app) as client:
        assert client.post("/api/v1/providers", json=_provider_payload()).status_code == 201
        health = client.patch(
            "/api/v1/providers/api-provider",
            json={"health_status": "READY"},
        )
        credential_status = client.patch(
            "/api/v1/providers/api-provider",
            json={"credential_configured": True},
        )
        credential_readback = client.get("/api/v1/providers/api-provider/credential")
    assert health.status_code == 422
    assert credential_status.status_code == 422
    assert credential_readback.status_code == 405
    assert credential_readback.headers["cache-control"] == "no-store"


def test_official_trust_cannot_be_spoofed_for_proxy_domain() -> None:
    app = _app()
    payload = _provider_payload()
    payload["trust_level"] = "OFFICIAL_VENDOR"
    with TestClient(app) as client:
        response = client.post("/api/v1/providers", json=payload)
        providers = client.get("/api/v1/providers")
    assert response.status_code == 400
    assert "OFFICIAL_VENDOR" in response.text
    assert providers.json() == []
