from collections.abc import AsyncIterator

import httpx
from pydantic import BaseModel

from mathmodel_ai.core.errors import ProviderResponseError
from mathmodel_ai.core.types import ProviderName
from mathmodel_ai.providers.http import HttpModelProvider
from mathmodel_ai.providers.schemas import (
    GenerationRequest,
    JsonObject,
    ModelResponse,
    ModelUsage,
    StructuredModelResponse,
)


class AnthropicProvider(HttpModelProvider):
    name = ProviderName.ANTHROPIC

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        api_version: str,
        timeout_seconds: float,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(base_url=base_url, timeout_seconds=timeout_seconds, client=client)
        self._headers = {
            "x-api-key": api_key,
            "anthropic-version": api_version,
            "content-type": "application/json",
        }

    @staticmethod
    def _payload(request: GenerationRequest) -> JsonObject:
        system = "\n\n".join(
            message.content
            for message in request.messages
            if message.role in {"system", "developer"}
        )
        messages = [
            {"role": message.role, "content": message.content}
            for message in request.messages
            if message.role in {"user", "assistant"}
        ]
        payload: JsonObject = {
            "model": request.model,
            "max_tokens": request.max_output_tokens,
            "messages": messages,
        }
        if system:
            payload["system"] = system
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        return payload

    def _response(self, data: JsonObject, requested_model: str) -> ModelResponse:
        content = data.get("content")
        if not isinstance(content, list):
            raise ProviderResponseError("anthropic response content is not a list")
        pieces = [
            block["text"]
            for block in content
            if isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        ]
        if not pieces:
            raise ProviderResponseError("anthropic response contained no text block")
        usage_value = data.get("usage")
        usage_data = usage_value if isinstance(usage_value, dict) else {}
        usage = ModelUsage(
            input_tokens=self._integer(usage_data.get("input_tokens")),
            output_tokens=self._integer(usage_data.get("output_tokens")),
            requests=1,
        )
        self._record_usage(usage)
        return ModelResponse(
            content="".join(pieces),
            provider=self.name,
            model=str(data.get("model", requested_model)),
            usage=usage,
            response_id=str(data["id"]) if data.get("id") else None,
            finish_reason=str(data["stop_reason"]) if data.get("stop_reason") else None,
        )

    async def generate(self, request: GenerationRequest) -> ModelResponse:
        data = await self._post_json(
            "/messages", headers=self._headers, payload=self._payload(request)
        )
        return self._response(data, request.model)

    async def structured_generate[T: BaseModel](
        self, request: GenerationRequest, response_model: type[T]
    ) -> StructuredModelResponse[T]:
        payload = self._payload(request)
        payload["output_config"] = {
            "format": {"type": "json_schema", "schema": response_model.model_json_schema()}
        }
        data = await self._post_json("/messages", headers=self._headers, payload=payload)
        response = self._response(data, request.model)
        return StructuredModelResponse(
            parsed=response_model.model_validate_json(response.content), response=response
        )

    def stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        async def iterator() -> AsyncIterator[str]:
            payload = self._payload(request)
            payload["stream"] = True
            input_tokens = 0
            output_tokens = 0
            async for event in self._stream_events(
                "/messages", headers=self._headers, payload=payload
            ):
                if event.get("type") == "message_start":
                    message = event.get("message")
                    if isinstance(message, dict) and isinstance(message.get("usage"), dict):
                        input_tokens = self._integer(message["usage"].get("input_tokens"))
                elif event.get("type") == "content_block_delta":
                    delta = event.get("delta")
                    if (
                        isinstance(delta, dict)
                        and delta.get("type") == "text_delta"
                        and isinstance(delta.get("text"), str)
                    ):
                        yield delta["text"]
                elif event.get("type") == "message_delta":
                    usage = event.get("usage")
                    if isinstance(usage, dict):
                        output_tokens = self._integer(usage.get("output_tokens"))
            self._record_usage(
                ModelUsage(input_tokens=input_tokens, output_tokens=output_tokens, requests=1)
            )

        return iterator()
