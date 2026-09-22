import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from mathmodel_ai.core.errors import ProviderError
from mathmodel_ai.providers.anthropic import AnthropicProvider
from mathmodel_ai.providers.google import GoogleProvider
from mathmodel_ai.providers.http import create_secure_async_client
from mathmodel_ai.providers.openai import OpenAIProvider
from mathmodel_ai.providers.schemas import GenerationRequest, MediaPart, ModelMessage


class Answer(BaseModel):
    value: int


def request() -> GenerationRequest:
    return GenerationRequest(
        model="configured-model",
        messages=[
            ModelMessage(role="system", content="Return a checked answer."),
            ModelMessage(role="user", content="Compute one plus one."),
        ],
        max_output_tokens=128,
    )


def client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://provider.example/v1"
    )


@pytest.mark.asyncio
async def test_secure_default_client_never_inherits_ambient_proxy_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actual_client = httpx.AsyncClient
    captured: dict[str, Any] = {}

    def client_factory(**kwargs: Any) -> httpx.AsyncClient:
        captured.update(kwargs)
        return actual_client(transport=httpx.MockTransport(lambda _: httpx.Response(200)))

    monkeypatch.setattr(httpx, "AsyncClient", client_factory)
    secure_client = create_secure_async_client(
        timeout=httpx.Timeout(1),
        verify=True,
        base_url="https://provider.example/v1",
    )
    try:
        assert captured["trust_env"] is False
        assert captured["follow_redirects"] is False
        assert captured["verify"] is True
    finally:
        await secure_client.aclose()


@pytest.mark.asyncio
async def test_openai_structured_request_and_response_contract() -> None:
    def handler(http_request: httpx.Request) -> httpx.Response:
        payload = json.loads(http_request.content)
        assert http_request.url.path == "/v1/responses"
        assert payload["text"]["format"]["type"] == "json_schema"
        assert payload["store"] is False
        return httpx.Response(
            200,
            json={
                "id": "resp-1",
                "model": "configured-model",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": '{"value":2}'}],
                    }
                ],
                "usage": {"input_tokens": 4, "output_tokens": 3},
            },
        )

    http_client = client(handler)
    provider = OpenAIProvider(
        api_key="test",
        base_url="https://provider.example/v1",
        timeout_seconds=1,
        client=http_client,
    )
    result = await provider.structured_generate(request(), Answer)
    await http_client.aclose()
    assert result.parsed.value == 2
    assert result.response.usage.input_tokens == 4


@pytest.mark.asyncio
async def test_anthropic_structured_request_and_response_contract() -> None:
    def handler(http_request: httpx.Request) -> httpx.Response:
        payload = json.loads(http_request.content)
        assert http_request.url.path == "/v1/messages"
        assert payload["output_config"]["format"]["type"] == "json_schema"
        assert http_request.headers["anthropic-version"] == "2023-06-01"
        return httpx.Response(
            200,
            json={
                "id": "msg-1",
                "model": "configured-model",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": '{"value":2}'}],
                "usage": {"input_tokens": 4, "output_tokens": 3},
            },
        )

    http_client = client(handler)
    provider = AnthropicProvider(
        api_key="test",
        base_url="https://provider.example/v1",
        api_version="2023-06-01",
        timeout_seconds=1,
        client=http_client,
    )
    result = await provider.structured_generate(request(), Answer)
    await http_client.aclose()
    assert result.parsed.value == 2
    assert result.response.usage.output_tokens == 3


@pytest.mark.asyncio
async def test_google_structured_request_and_response_contract() -> None:
    def handler(http_request: httpx.Request) -> httpx.Response:
        payload: dict[str, Any] = json.loads(http_request.content)
        assert http_request.url.path == "/v1/models/configured-model:generateContent"
        assert payload["generationConfig"]["responseMimeType"] == "application/json"
        assert "responseJsonSchema" in payload["generationConfig"]
        assert http_request.headers["x-goog-api-key"] == "test"
        return httpx.Response(
            200,
            json={
                "responseId": "gemini-1",
                "modelVersion": "configured-model",
                "candidates": [
                    {
                        "finishReason": "STOP",
                        "content": {"parts": [{"text": '{"value":2}'}]},
                    }
                ],
                "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 3},
            },
        )

    http_client = client(handler)
    provider = GoogleProvider(
        api_key="test",
        base_url="https://provider.example/v1",
        timeout_seconds=1,
        client=http_client,
    )
    result = await provider.structured_generate(request(), Answer)
    await http_client.aclose()
    assert result.parsed.value == 2
    assert result.response.finish_reason == "STOP"


@pytest.mark.asyncio
async def test_google_sends_inline_multimodal_bytes_with_declared_mime() -> None:
    def handler(http_request: httpx.Request) -> httpx.Response:
        payload: dict[str, Any] = json.loads(http_request.content)
        parts = payload["contents"][0]["parts"]
        assert parts[1] == {"inlineData": {"mimeType": "image/png", "data": "aW1hZ2U="}}
        return httpx.Response(
            200,
            json={
                "candidates": [{"content": {"parts": [{"text": '{"value":2}'}]}}],
                "usageMetadata": {},
            },
        )

    http_client = client(handler)
    provider = GoogleProvider(
        api_key="test",
        base_url="https://provider.example/v1",
        timeout_seconds=1,
        client=http_client,
    )
    media_request = request().model_copy(
        update={
            "messages": [
                ModelMessage(
                    role="user",
                    content="Inspect the attachment.",
                    media=[
                        MediaPart(
                            mime_type="image/png",
                            data_base64="aW1hZ2U=",
                            source_id="file-1",
                        )
                    ],
                )
            ]
        }
    )
    result = await provider.structured_generate(media_request, Answer)
    await http_client.aclose()
    assert result.parsed.value == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_name", ["openai", "anthropic"])
async def test_non_multimodal_provider_paths_reject_media_instead_of_dropping_it(
    provider_name: str,
) -> None:
    http_client = client(lambda _request: httpx.Response(500))
    if provider_name == "openai":
        provider = OpenAIProvider(
            api_key="test",
            base_url="https://provider.example/v1",
            timeout_seconds=1,
            client=http_client,
        )
    else:
        provider = AnthropicProvider(
            api_key="test",
            base_url="https://provider.example/v1",
            api_version="2023-06-01",
            timeout_seconds=1,
            client=http_client,
        )
    media_request = request().model_copy(
        update={
            "messages": [
                ModelMessage(
                    role="user",
                    media=[
                        MediaPart(
                            mime_type="image/png",
                            data_base64="aW1hZ2U=",
                            source_id="file-1",
                        )
                    ],
                )
            ]
        }
    )
    with pytest.raises(ProviderError, match="Google provider path"):
        await provider.generate(media_request)
    await http_client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_name", ["openai", "anthropic", "google"])
async def test_http_provider_streams_text_deltas(provider_name: str) -> None:
    events = {
        "openai": [
            {"type": "response.output_text.delta", "delta": "a"},
            {"type": "response.output_text.delta", "delta": "b"},
            {
                "type": "response.completed",
                "response": {"usage": {"input_tokens": 4, "output_tokens": 3}},
            },
        ],
        "anthropic": [
            {
                "type": "message_start",
                "message": {"usage": {"input_tokens": 4, "output_tokens": 0}},
            },
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "a"}},
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "b"}},
            {"type": "message_delta", "usage": {"output_tokens": 3}},
        ],
        "google": [
            {"candidates": [{"content": {"parts": [{"text": "a"}]}}]},
            {
                "candidates": [{"content": {"parts": [{"text": "b"}]}}],
                "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 3},
            },
        ],
    }
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events[provider_name])

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    http_client = client(handler)
    if provider_name == "openai":
        provider = OpenAIProvider(
            api_key="test",
            base_url="https://provider.example/v1",
            timeout_seconds=1,
            client=http_client,
        )
    elif provider_name == "anthropic":
        provider = AnthropicProvider(
            api_key="test",
            base_url="https://provider.example/v1",
            api_version="2023-06-01",
            timeout_seconds=1,
            client=http_client,
        )
    else:
        provider = GoogleProvider(
            api_key="test",
            base_url="https://provider.example/v1",
            timeout_seconds=1,
            client=http_client,
        )
    chunks = [chunk async for chunk in provider.stream(request())]
    usage = provider.get_usage()
    await http_client.aclose()
    assert chunks == ["a", "b"]
    assert usage.requests == 1
    assert usage.input_tokens == 4
    assert usage.output_tokens == 3
