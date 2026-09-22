import httpx
import pytest

from mathmodel_ai.core.errors import ModelDiscoveryError
from mathmodel_ai.core.types import Environment
from mathmodel_ai.providers.discovery import ProviderModelDiscovery
from mathmodel_ai.providers.secrets import EnvironmentSecretResolver
from mathmodel_ai.providers.security import EndpointSecurityPolicy
from mathmodel_ai.schemas.provider_registry import (
    EndpointTrustLevel,
    ProviderEndpoint,
    ProviderProtocol,
)


def _endpoint(**changes: object) -> ProviderEndpoint:
    payload: dict[str, object] = {
        "provider_id": "discovery-provider",
        "display_name": "Discovery Provider",
        "protocol": ProviderProtocol.OPENAI_CHAT_COMPLETIONS,
        "base_url": "https://models.example.test/v1",
        "credential_ref": "env:DISCOVERY_KEY",
        "trust_level": EndpointTrustLevel.USER_MANAGED_PROXY,
    }
    payload.update(changes)
    return ProviderEndpoint.model_validate(payload)


def _service(handler: httpx.MockTransport) -> ProviderModelDiscovery:
    return ProviderModelDiscovery(
        secrets=EnvironmentSecretResolver({"DISCOVERY_KEY": "synthetic-key"}),
        security_policy=EndpointSecurityPolicy(
            environment=Environment.TEST,
            resolver=lambda _host, _port: ["93.184.216.34"],
        ),
        client=httpx.AsyncClient(transport=handler),
    )


@pytest.mark.asyncio
async def test_openai_compatible_discovery_is_bounded_and_does_not_import_capabilities() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url == "https://93.184.216.34/v1/models"
        assert request.headers["host"] == "models.example.test"
        assert request.headers["authorization"] == "Bearer synthetic-key"
        return httpx.Response(
            200,
            json={
                "data": [
                    {"id": "model-a", "vision": True},
                    {"id": "model-b", "display_name": "Model B", "tools": True},
                    {"id": "model-a"},
                ]
            },
        )

    models = await _service(httpx.MockTransport(handler)).discover(_endpoint())

    assert [item.remote_model_id for item in models] == ["model-a", "model-b"]
    assert models[1].display_name == "Model B"
    assert not hasattr(models[0], "capabilities")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "code", "response_status"),
    [
        (401, "AUTHENTICATION_FAILED", 401),
        (404, "MODEL_DISCOVERY_UNSUPPORTED", 409),
        (503, "MODEL_DISCOVERY_FAILED", 502),
    ],
)
async def test_discovery_has_stable_secret_safe_error_taxonomy(
    status: int, code: str, response_status: int
) -> None:
    service = _service(
        httpx.MockTransport(lambda _request: httpx.Response(status, text="credential=secret"))
    )

    with pytest.raises(ModelDiscoveryError) as caught:
        await service.discover(_endpoint())

    assert caught.value.code == code
    assert caught.value.status_code == response_status
    assert "secret" not in str(caught.value)


@pytest.mark.asyncio
async def test_custom_json_discovery_is_unsupported_without_marking_endpoint_invalid() -> None:
    service = _service(httpx.MockTransport(lambda _request: httpx.Response(500)))

    with pytest.raises(ModelDiscoveryError) as caught:
        await service.discover(_endpoint(protocol=ProviderProtocol.CUSTOM_JSON_HTTP))

    assert caught.value.code == "MODEL_DISCOVERY_UNSUPPORTED"
    assert caught.value.status_code == 409
