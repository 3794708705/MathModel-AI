import asyncio
import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from mathmodel_ai.core.config import Settings
from mathmodel_ai.core.errors import (
    ProviderRedirectError,
    ProviderResponseError,
    ProviderResponseTooLargeError,
    ProviderTimeoutError,
)
from mathmodel_ai.core.types import Environment, ReasoningEffort
from mathmodel_ai.db.base import Base
from mathmodel_ai.db.session import create_database_engine, create_session_factory
from mathmodel_ai.providers.compatible import (
    CustomJSONHttpProvider,
    OpenAICompatibleProvider,
)
from mathmodel_ai.providers.factory import ProviderRegistry, build_configured_provider
from mathmodel_ai.providers.probe import ProviderCompatibilityProbe
from mathmodel_ai.providers.repository import ProviderModelRegistry
from mathmodel_ai.providers.schemas import (
    GenerationRequest,
    ModelMessage,
    ToolDefinition,
    ToolFunction,
)
from mathmodel_ai.providers.secrets import EnvironmentSecretResolver
from mathmodel_ai.providers.security import EndpointSecurityPolicy
from mathmodel_ai.routing.router import ModelRouter
from mathmodel_ai.routing.schemas import RouteAction, TaskProfile, TaskType
from mathmodel_ai.schemas.provider_registry import (
    CapabilityEvidence,
    CapabilitySource,
    CapabilityStatus,
    CustomAuthScheme,
    CustomJSONConfiguration,
    CustomJSONRequestMapping,
    CustomJSONResponseMapping,
    ModelCapability,
    ModelProfile,
    ProbeAuthenticationStatus,
    ProviderEndpoint,
    ProviderHealthStatus,
    ProviderProtocol,
    QualityTier,
    StructuredOutputStrategy,
    ToolCallingStrategy,
)


class Answer(BaseModel):
    value: int


def test_generation_request_default_supports_large_structured_agent_schemas() -> None:
    request = GenerationRequest(
        model="test-model",
        messages=[ModelMessage(role="user", content="Return JSON")],
    )

    assert request.max_output_tokens == 32_768


def _public_policy() -> EndpointSecurityPolicy:
    return EndpointSecurityPolicy(
        environment=Environment.TEST,
        resolver=lambda _host, _port: ["93.184.216.34"],
    )


def _endpoint(
    *,
    protocol: ProviderProtocol = ProviderProtocol.OPENAI_CHAT_COMPLETIONS,
    max_response_bytes: int = 4096,
) -> ProviderEndpoint:
    return ProviderEndpoint(
        provider_id="compatible-test",
        display_name="Compatible Test",
        protocol=protocol,
        base_url="https://models.example.test/v1",
        credential_ref="env:TEST_MODEL_KEY",
        max_response_bytes=max_response_bytes,
    )


def _capability(status: CapabilityStatus = CapabilityStatus.SUPPORTED) -> CapabilityEvidence:
    return CapabilityEvidence(status=status, source=CapabilitySource.USER_DECLARED)


def _model(
    *,
    strategy: StructuredOutputStrategy = StructuredOutputStrategy.NATIVE_JSON_SCHEMA,
    capabilities: set[ModelCapability] | None = None,
    configuration: dict[str, Any] | None = None,
) -> ModelProfile:
    selected = capabilities or {
        ModelCapability.TEXT,
        ModelCapability.STRUCTURED_OUTPUT,
        ModelCapability.JSON_SCHEMA,
        ModelCapability.TOOLS,
        ModelCapability.STREAMING,
        ModelCapability.REASONING_CONTROL,
        ModelCapability.SYSTEM_ROLE,
        ModelCapability.USAGE_REPORTING,
    }
    return ModelProfile(
        model_id="compatible-main",
        provider_id="compatible-test",
        display_name="Compatible Main",
        remote_model="remote-compatible-model",
        quality_tier=QualityTier.FLAGSHIP_XHIGH,
        declared_capabilities={item: _capability() for item in selected},
        structured_output_strategy=strategy,
        tool_calling_strategy=(
            ToolCallingStrategy.NATIVE_TOOLS
            if ModelCapability.TOOLS in selected
            else ToolCallingStrategy.NO_TOOLS
        ),
        reasoning_mapping={"LOW": "low", "XHIGH": "high", "MAX": "provider-max"},
        configuration=configuration or {},
    )


def _registry(
    endpoint: ProviderEndpoint,
    model: ModelProfile,
    *,
    client: httpx.AsyncClient,
) -> tuple[ProviderModelRegistry, ProviderRegistry]:
    settings = Settings(
        environment="test",
        database_url="sqlite+pysqlite:///:memory:",
        default_provider="mock",
    )
    engine = create_database_engine(settings)
    Base.metadata.create_all(engine)
    secrets = EnvironmentSecretResolver({"TEST_MODEL_KEY": "unit-secret-value"})
    policy = _public_policy()
    configurations = ProviderModelRegistry(
        create_session_factory(engine),
        secrets=secrets,
        security_policy=policy,
    )
    configurations.create_provider(endpoint)
    configurations.create_model(model)
    providers = ProviderRegistry(
        [],
        configurations=configurations,
        secrets=secrets,
        security_policy=policy,
        configured_clients={endpoint.provider_id: client},
    )
    return configurations, providers


def _compatible_response(
    content: str = "OK",
    *,
    tool_calls: list[dict[str, Any]] | None = None,
) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "response-test",
            "model": "remote-compatible-model",
            "choices": [
                {
                    "message": {"content": content, "tool_calls": tool_calls or []},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "total_tokens": 5,
                "completion_tokens_details": {"reasoning_tokens": 1},
            },
        },
    )


def _probe_handler(seen: list[dict[str, Any]]) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        seen.append(payload)
        if payload.get("stream"):
            return httpx.Response(
                200,
                content=(
                    b'data: {"choices":[{"delta":{"content":"O"}}]}\n\n'
                    b'data: {"choices":[],"usage":{"prompt_tokens":1,'
                    b'"completion_tokens":1,"total_tokens":2}}\n\n'
                    b"data: [DONE]\n\n"
                ),
                headers={"content-type": "text/event-stream"},
            )
        if payload.get("tools"):
            return _compatible_response(
                "",
                tool_calls=[
                    {
                        "id": "call-test",
                        "type": "function",
                        "function": {"name": "dummy_test_tool", "arguments": '{"value":1}'},
                    }
                ],
            )
        if payload.get("response_format"):
            return _compatible_response('{"ok":true,"value":1}')
        return _compatible_response()

    return handler


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("protocol", "remote_response"),
    [
        (
            ProviderProtocol.OPENAI_RESPONSES,
            {
                "id": "resp-registry",
                "model": "remote-compatible-model",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": '{"value":2}'}],
                    }
                ],
                "usage": {"input_tokens": 4, "output_tokens": 3},
            },
        ),
        (
            ProviderProtocol.ANTHROPIC_MESSAGES,
            {
                "id": "msg-registry",
                "model": "remote-compatible-model",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": '{"value":2}'}],
                "usage": {"input_tokens": 4, "output_tokens": 3},
            },
        ),
        (
            ProviderProtocol.GOOGLE_GENERATE_CONTENT,
            {
                "responseId": "gemini-registry",
                "modelVersion": "remote-compatible-model",
                "candidates": [
                    {
                        "finishReason": "STOP",
                        "content": {"parts": [{"text": '{"value":2}'}]},
                    }
                ],
                "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 3},
            },
        ),
    ],
)
async def test_native_protocol_adapters_are_bound_to_registry_identity(
    protocol: ProviderProtocol, remote_response: dict[str, Any]
) -> None:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, json=remote_response)),
        base_url="https://models.example.test/v1",
    )
    endpoint = _endpoint(protocol=protocol)
    model = _model()
    provider = build_configured_provider(
        endpoint=endpoint,
        model=model,
        secrets=EnvironmentSecretResolver({"TEST_MODEL_KEY": "unit-secret-value"}),
        security_policy=_public_policy(),
        client=client,
    )

    result = await provider.structured_generate(
        GenerationRequest(
            model="ignored",
            messages=[ModelMessage(role="user", content="Return two")],
        ),
        Answer,
    )

    assert result.parsed.value == 2
    assert result.response.provider == endpoint.provider_id
    assert result.response.model_id == model.model_id
    assert result.response.provider_config_digest == endpoint.config_digest
    assert result.response.model_config_digest == model.config_digest
    assert result.response.protocol == protocol.value
    await provider.aclose()
    await client.aclose()


@pytest.mark.asyncio
async def test_a_deepseek_style_probe_detects_supported_capabilities() -> None:
    seen: list[dict[str, Any]] = []
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(_probe_handler(seen)),
        base_url="https://models.example.test/v1",
    )
    configurations, providers = _registry(
        _endpoint(),
        _model(configuration={"allowed_parameters": ["temperature"]}),
        client=client,
    )
    result = await ProviderCompatibilityProbe(
        configurations=configurations, providers=providers
    ).run("compatible-main")

    assert result.authentication_status is ProbeAuthenticationStatus.PASS
    for capability in (
        ModelCapability.TEXT,
        ModelCapability.JSON_SCHEMA,
        ModelCapability.TOOLS,
        ModelCapability.REASONING_CONTROL,
        ModelCapability.STREAMING,
        ModelCapability.USAGE_REPORTING,
    ):
        assert result.capabilities[capability].status is CapabilityStatus.SUPPORTED
    assert any("reasoning_effort" in payload for payload in seen)
    structured_requests = [payload for payload in seen if payload.get("response_format")]
    assert structured_requests
    assert all(
        "json"
        in " ".join(
            str(message.get("content", "")) for message in payload.get("messages", [])
        ).lower()
        for payload in structured_requests
    )
    assert len(seen) <= 6
    assert all(int(payload.get("max_tokens", 32)) <= 256 for payload in seen)
    assert all(
        int(payload.get("max_tokens", 32)) <= 32
        for payload in seen
        if not payload.get("response_format")
    )
    assert result.capabilities[ModelCapability.VISION].status is CapabilityStatus.UNKNOWN
    assert result.capabilities[ModelCapability.LONG_CONTEXT].status is CapabilityStatus.UNKNOWN
    assert (
        configurations.get_provider("compatible-test").health_status is ProviderHealthStatus.READY
    )
    await providers.aclose()


@pytest.mark.asyncio
async def test_probe_can_reverify_a_previously_unsupported_capability() -> None:
    seen: list[dict[str, Any]] = []
    accepts_structured = False
    success_handler = _probe_handler(seen)

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        if payload.get("response_format") and not accepts_structured:
            seen.append(payload)
            return httpx.Response(400, json={"error": "temporarily unsupported"})
        return success_handler(request)

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://models.example.test/v1",
    )
    configurations, first_providers = _registry(_endpoint(), _model(), client=client)
    first = await ProviderCompatibilityProbe(
        configurations=configurations, providers=first_providers
    ).run("compatible-main")
    assert (
        first.capabilities[ModelCapability.STRUCTURED_OUTPUT].status is CapabilityStatus.UNSUPPORTED
    )

    accepts_structured = True
    second_providers = ProviderRegistry(
        [],
        configurations=configurations,
        secrets=EnvironmentSecretResolver({"TEST_MODEL_KEY": "unit-secret-value"}),
        security_policy=_public_policy(),
        configured_clients={"compatible-test": client},
    )
    second = await ProviderCompatibilityProbe(
        configurations=configurations, providers=second_providers
    ).run("compatible-main")

    assert (
        second.capabilities[ModelCapability.STRUCTURED_OUTPUT].status is CapabilityStatus.SUPPORTED
    )
    assert second.capabilities[ModelCapability.JSON_SCHEMA].status is CapabilityStatus.SUPPORTED
    await first_providers.aclose()
    await second_providers.aclose()
    await client.aclose()


@pytest.mark.asyncio
async def test_json_mode_injects_an_explicit_json_instruction() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return _compatible_response('{"value":2}')

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://models.example.test/v1",
    )
    provider = OpenAICompatibleProvider(
        endpoint=_endpoint(),
        model_profile=_model(
            strategy=StructuredOutputStrategy.JSON_MODE,
            capabilities={
                ModelCapability.TEXT,
                ModelCapability.STRUCTURED_OUTPUT,
                ModelCapability.JSON_MODE,
            },
        ),
        api_key="unit-secret-value",
        security_policy=_public_policy(),
        client=client,
    )

    result = await provider.structured_generate(
        GenerationRequest(
            model="ignored",
            messages=[ModelMessage(role="user", content="Return the requested structure.")],
        ),
        Answer,
    )

    assert result.parsed == Answer(value=2)
    assert seen[0]["response_format"] == {"type": "json_object"}
    structured_instruction = " ".join(
        str(message.get("content", "")) for message in seen[0]["messages"]
    ).lower()
    assert "json schema" in structured_instruction
    assert "every property listed in required" in structured_instruction
    assert '"value"' in structured_instruction
    await provider.aclose()
    await client.aclose()


@pytest.mark.asyncio
async def test_structured_validation_error_reports_only_safe_field_metadata() -> None:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: _compatible_response('{"value":"sensitive-invalid-value"}')
        ),
        base_url="https://models.example.test/v1",
    )
    provider = OpenAICompatibleProvider(
        endpoint=_endpoint(),
        model_profile=_model(),
        api_key="unit-secret-value",
        security_policy=_public_policy(),
        client=client,
    )

    with pytest.raises(ProviderResponseError) as error:
        await provider.structured_generate(
            GenerationRequest(
                model="ignored",
                messages=[ModelMessage(role="user", content="Return a value")],
            ),
            Answer,
        )

    assert "value:int_parsing" in str(error.value)
    assert "sensitive-invalid-value" not in str(error.value)
    await provider.aclose()
    await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    [
        '{"value": "private-sentinel",}',
        '{"value": "private-sentinel\\q"}',
        '```json\n{"value": 2}\n```',
        'Here is the answer: {"value": 2}',
        '{"value": 2} {"value": 3}',
    ],
)
async def test_malformed_structured_json_is_rejected_with_safe_syntax_location(
    content: str,
) -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: _compatible_response(content)),
        base_url="https://models.example.test/v1",
    ) as client:
        provider = OpenAICompatibleProvider(
            endpoint=_endpoint(),
            model_profile=_model(),
            api_key="unit-secret-value",
            security_policy=_public_policy(),
            client=client,
        )
        with pytest.raises(ProviderResponseError) as error:
            await provider.structured_generate(
                GenerationRequest(
                    model="ignored", messages=[ModelMessage(role="user", content="Return JSON")]
                ),
                Answer,
            )

        message = str(error.value)
        assert "json_syntax=" in message
        assert "line=" in message and "column=" in message and "position=" in message
        assert "finish_reason=stop" in message
        assert "private-sentinel" not in message
        assert "unit-secret-value" not in message
        assert provider.get_usage().requests == 1


@pytest.mark.asyncio
async def test_b_parameter_mapper_omits_unverified_optional_parameters() -> None:
    seen: list[dict[str, Any]] = []
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(_probe_handler(seen)),
        base_url="https://models.example.test/v1",
    )
    provider = OpenAICompatibleProvider(
        endpoint=_endpoint(),
        model_profile=_model(
            strategy=StructuredOutputStrategy.UNSUPPORTED,
            capabilities={ModelCapability.TEXT},
        ),
        api_key="unit-secret-value",
        security_policy=_public_policy(),
        client=client,
    )
    await provider.generate(
        GenerationRequest(
            model="ignored-business-slug",
            messages=[ModelMessage(role="user", content="Reply OK")],
            temperature=0.7,
            reasoning_effort=ReasoningEffort.XHIGH,
        )
    )

    assert "temperature" not in seen[0]
    assert "reasoning_effort" not in seen[0]
    assert "reasoning" not in seen[0]
    assert "thinking" not in seen[0]
    assert seen[0]["model"] == "remote-compatible-model"
    await provider.aclose()


@pytest.mark.asyncio
async def test_max_reasoning_is_normalized_then_mapped_by_model_profile() -> None:
    seen: list[dict[str, Any]] = []
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(_probe_handler(seen)),
        base_url="https://models.example.test/v1",
    )
    provider = OpenAICompatibleProvider(
        endpoint=_endpoint(),
        model_profile=_model(),
        api_key="unit-secret-value",
        security_policy=_public_policy(),
        client=client,
    )

    response = await provider.generate(
        GenerationRequest(
            model="ignored",
            messages=[ModelMessage(role="user", content="Reply OK")],
            reasoning_effort=ReasoningEffort.MAX,
        )
    )

    assert seen[0]["reasoning_effort"] == "provider-max"
    assert response.reasoning_effective == "provider-max"
    await provider.aclose()


@pytest.mark.asyncio
async def test_c_failed_json_schema_probe_blocks_schema_route() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        if payload.get("response_format"):
            return httpx.Response(400, json={"error": "unsupported response_format"})
        if payload.get("stream"):
            return httpx.Response(200, content=b"data: [DONE]\n\n")
        return _compatible_response()

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://models.example.test/v1"
    )
    configurations, providers = _registry(_endpoint(), _model(), client=client)
    result = await ProviderCompatibilityProbe(
        configurations=configurations, providers=providers
    ).run("compatible-main")

    assert result.capabilities[ModelCapability.TEXT].status is CapabilityStatus.SUPPORTED
    assert result.capabilities[ModelCapability.JSON_SCHEMA].status is CapabilityStatus.UNSUPPORTED
    decision = ModelRouter(
        Settings(default_model_id="compatible-main"),
        available_providers=providers.available,
        registry=configurations,
    ).route(
        TaskProfile(
            task_type=TaskType.DOCUMENTATION,
            requires_json_schema=True,
        )
    )
    assert decision.action is RouteAction.HUMAN_REVIEW
    assert "JSON_SCHEMA" in " ".join(decision.rejected_models["compatible-main"])
    await providers.aclose()


@pytest.mark.asyncio
async def test_d_prompt_json_fallback_is_validated_and_traced() -> None:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: _compatible_response('{"value":7}')),
        base_url="https://models.example.test/v1",
    )
    provider = OpenAICompatibleProvider(
        endpoint=_endpoint(),
        model_profile=_model(
            strategy=StructuredOutputStrategy.PROMPT_JSON_FALLBACK,
            capabilities={ModelCapability.TEXT},
        ),
        api_key="unit-secret-value",
        security_policy=_public_policy(),
        client=client,
    )
    result = await provider.structured_generate(
        GenerationRequest(
            model="ignored",
            messages=[ModelMessage(role="user", content="Return a value")],
        ),
        Answer,
    )
    assert result.parsed == Answer(value=7)
    assert (
        result.response.structured_output_mode
        == StructuredOutputStrategy.PROMPT_JSON_FALLBACK.value
    )
    await provider.aclose()


@pytest.mark.asyncio
async def test_e_authentication_failure_is_not_misclassified() -> None:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(401)),
        base_url="https://models.example.test/v1",
    )
    configurations, providers = _registry(_endpoint(), _model(), client=client)
    result = await ProviderCompatibilityProbe(
        configurations=configurations, providers=providers
    ).run("compatible-main")
    assert result.authentication_status is ProbeAuthenticationStatus.AUTH_FAILED
    assert (
        configurations.get_provider("compatible-test").health_status
        is ProviderHealthStatus.AUTH_FAILED
    )
    await providers.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "location",
    [
        "https://evil.example/steal",
        "https://models.example.test/other",
        "http://models.example.test/downgrade",
    ],
)
async def test_i_redirect_is_blocked_without_forwarding_authorization(location: str) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(302, headers={"location": location})

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://models.example.test/v1"
    )
    provider = OpenAICompatibleProvider(
        endpoint=_endpoint(),
        model_profile=_model(),
        api_key="unit-secret-value",
        security_policy=_public_policy(),
        client=client,
    )
    with pytest.raises(ProviderRedirectError, match="PROVIDER_REDIRECT_BLOCKED"):
        await provider.generate(
            GenerationRequest(
                model="ignored", messages=[ModelMessage(role="user", content="Reply OK")]
            )
        )
    assert len(requests) == 1
    assert requests[0].url.host == "93.184.216.34"
    assert requests[0].headers["host"] == "models.example.test"
    await provider.aclose()


@pytest.mark.asyncio
async def test_u_oversized_response_has_stable_failure_classification() -> None:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, content=b"x" * 1025)),
        base_url="https://models.example.test/v1",
    )
    provider = OpenAICompatibleProvider(
        endpoint=_endpoint(max_response_bytes=1024),
        model_profile=_model(),
        api_key="unit-secret-value",
        security_policy=_public_policy(),
        client=client,
    )
    with pytest.raises(ProviderResponseTooLargeError, match="RESPONSE_TOO_LARGE"):
        await provider.generate(
            GenerationRequest(
                model="ignored", messages=[ModelMessage(role="user", content="Reply OK")]
            )
        )
    await provider.aclose()


@pytest.mark.asyncio
async def test_v_timeout_has_stable_failure_classification() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("simulated timeout", request=request)

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://models.example.test/v1"
    )
    provider = OpenAICompatibleProvider(
        endpoint=_endpoint(),
        model_profile=_model(),
        api_key="unit-secret-value",
        security_policy=_public_policy(),
        client=client,
    )
    with pytest.raises(ProviderTimeoutError, match="PROVIDER_TIMEOUT"):
        await provider.generate(
            GenerationRequest(
                model="ignored", messages=[ModelMessage(role="user", content="Reply OK")]
            )
        )
    await provider.aclose()


@pytest.mark.asyncio
async def test_total_timeout_bounds_the_complete_provider_request() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.05)
        return _compatible_response()

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://models.example.test/v1"
    )
    endpoint = ProviderEndpoint(
        provider_id="compatible-test",
        display_name="Compatible Test",
        protocol=ProviderProtocol.OPENAI_CHAT_COMPLETIONS,
        base_url="https://models.example.test/v1",
        credential_ref="env:TEST_MODEL_KEY",
        timeout_seconds=0.01,
        connect_timeout_seconds=0.01,
        max_response_bytes=4096,
    )
    provider = OpenAICompatibleProvider(
        endpoint=endpoint,
        model_profile=_model(),
        api_key="unit-secret-value",
        security_policy=_public_policy(),
        client=client,
    )
    with pytest.raises(ProviderTimeoutError, match="PROVIDER_TIMEOUT"):
        await provider.generate(
            GenerationRequest(
                model="ignored", messages=[ModelMessage(role="user", content="Reply OK")]
            )
        )
    await provider.aclose()


@pytest.mark.asyncio
async def test_custom_json_http_uses_constrained_mapping_without_secret_template() -> None:
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"result": {"text": '{"value":9}'}, "usage": {"in": 2, "out": 3}},
        )

    endpoint = _endpoint(protocol=ProviderProtocol.CUSTOM_JSON_HTTP)
    mapping = CustomJSONConfiguration(
        request=CustomJSONRequestMapping(
            endpoint_path="/generate",
            body={
                "model": "{model}",
                "input": "{prompt}",
                "max_tokens": "{max_tokens}",
            },
        ),
        response=CustomJSONResponseMapping(
            text_path="result.text",
            input_tokens_path="usage.in",
            output_tokens_path="usage.out",
        ),
    )
    model = _model(
        strategy=StructuredOutputStrategy.PROMPT_JSON_FALLBACK,
        capabilities={ModelCapability.TEXT},
        configuration={"custom_json": mapping.model_dump(mode="json")},
    )
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://models.example.test/v1"
    )
    provider = CustomJSONHttpProvider(
        endpoint=endpoint,
        model_profile=model,
        api_key="unit-secret-value",
        security_policy=_public_policy(),
        client=client,
    )
    result = await provider.structured_generate(
        GenerationRequest(
            model="ignored",
            messages=[ModelMessage(role="user", content="Return a value")],
            max_output_tokens=17,
        ),
        Answer,
    )
    assert result.parsed.value == 9
    assert captured[0]["model"] == "remote-compatible-model"
    assert captured[0]["max_tokens"] == 17
    assert "unit-secret-value" not in json.dumps(captured)
    await provider.aclose()


@pytest.mark.asyncio
async def test_custom_json_api_key_header_is_adapter_controlled() -> None:
    observed_header: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        observed_header.append(request.headers["x-model-key"])
        return httpx.Response(200, json={"text": "OK"})

    mapping = CustomJSONConfiguration(
        request=CustomJSONRequestMapping(),
        response=CustomJSONResponseMapping(text_path="text"),
        auth_scheme=CustomAuthScheme.API_KEY_HEADER,
        api_key_header="X-Model-Key",
    )
    provider = CustomJSONHttpProvider(
        endpoint=_endpoint(protocol=ProviderProtocol.CUSTOM_JSON_HTTP),
        model_profile=_model(
            strategy=StructuredOutputStrategy.PROMPT_JSON_FALLBACK,
            capabilities={ModelCapability.TEXT},
            configuration={"custom_json": mapping.model_dump(mode="json")},
        ),
        api_key="unit-secret-value",
        security_policy=_public_policy(),
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="https://models.example.test/v1",
        ),
    )
    await provider.generate(
        GenerationRequest(
            model="ignored",
            messages=[ModelMessage(role="user", content="Reply OK")],
        )
    )
    assert observed_header == ["unit-secret-value"]
    await provider.aclose()


def test_tool_mapping_requires_both_strategy_and_capability() -> None:
    profile = _model(
        capabilities={ModelCapability.TEXT, ModelCapability.TOOLS},
    ).model_copy(update={"tool_calling_strategy": ToolCallingStrategy.NO_TOOLS})
    provider = OpenAICompatibleProvider(
        endpoint=_endpoint(),
        model_profile=profile,
        api_key="unit-secret-value",
        security_policy=_public_policy(),
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _request: _compatible_response()),
            base_url="https://models.example.test/v1",
        ),
    )
    request = GenerationRequest(
        model="ignored",
        messages=[ModelMessage(role="user", content="Call tool")],
        tools=[
            ToolDefinition(
                function=ToolFunction(
                    name="dummy_test_tool",
                    description="No side effects",
                    parameters={"type": "object"},
                )
            )
        ],
    )
    with pytest.raises(Exception, match="does not support native tools"):
        provider._payload(request)
