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


class OpenAIProvider(HttpModelProvider):
    name = ProviderName.OPENAI

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        timeout_seconds: float,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(base_url=base_url, timeout_seconds=timeout_seconds, client=client)
        self._headers = {"Authorization": f"Bearer {api_key}"}

    @staticmethod
    def _payload(request: GenerationRequest) -> JsonObject:
        instructions = "\n\n".join(
            message.content
            for message in request.messages
            if message.role in {"system", "developer"}
        )
        messages = [
            {"role": message.role, "content": message.content}
            for message in request.messages
            if message.role not in {"system", "developer"}
        ]
        payload: JsonObject = {
            "model": request.model,
            "input": messages,
            "max_output_tokens": request.max_output_tokens,
            "store": False,
        }
        if instructions:
            payload["instructions"] = instructions
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.reasoning_effort is not None and request.reasoning_effort.value != "none":
            payload["reasoning"] = {"effort": request.reasoning_effort.value}
        if request.metadata:
            payload["metadata"] = request.metadata
        return payload

    @staticmethod
    def _content(data: JsonObject) -> str:
        pieces: list[str] = []
        output = data.get("output", [])
        if not isinstance(output, list):
            raise ProviderResponseError("openai response output is not a list")
        for item in output:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            content = item.get("content", [])
            if not isinstance(content, list):
                continue
            for block in content:
                if isinstance(block, dict) and block.get("type") == "output_text":
                    text = block.get("text")
                    if isinstance(text, str):
                        pieces.append(text)
        if not pieces:
            raise ProviderResponseError("openai response contained no output_text")
        return "".join(pieces)

    def _response(self, data: JsonObject, requested_model: str) -> ModelResponse:
        usage_value = data.get("usage")
        usage_data = usage_value if isinstance(usage_value, dict) else {}
        details_value = usage_data.get("input_tokens_details")
        input_details = details_value if isinstance(details_value, dict) else {}
        usage = ModelUsage(
            input_tokens=self._integer(usage_data.get("input_tokens")),
            output_tokens=self._integer(usage_data.get("output_tokens")),
            cached_input_tokens=self._integer(input_details.get("cached_tokens")),
            requests=1,
        )
        self._record_usage(usage)
        return ModelResponse(
            content=self._content(data),
            provider=self.name,
            model=str(data.get("model", requested_model)),
            usage=usage,
            response_id=str(data["id"]) if data.get("id") else None,
            finish_reason=str(data["status"]) if data.get("status") else None,
        )

    async def generate(self, request: GenerationRequest) -> ModelResponse:
        data = await self._post_json(
            "/responses", headers=self._headers, payload=self._payload(request)
        )
        return self._response(data, request.model)

    async def structured_generate[T: BaseModel](
        self, request: GenerationRequest, response_model: type[T]
    ) -> StructuredModelResponse[T]:
        payload = self._payload(request)
        payload["text"] = {
            "format": {
                "type": "json_schema",
                "name": response_model.__name__,
                "schema": response_model.model_json_schema(),
                "strict": True,
            }
        }
        data = await self._post_json("/responses", headers=self._headers, payload=payload)
        response = self._response(data, request.model)
        return StructuredModelResponse(
            parsed=response_model.model_validate_json(response.content), response=response
        )

    def stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        async def iterator() -> AsyncIterator[str]:
            payload = self._payload(request)
            payload["stream"] = True
            async for event in self._stream_events(
                "/responses", headers=self._headers, payload=payload
            ):
                event_type = event.get("type")
                if event_type == "response.output_text.delta" and isinstance(
                    event.get("delta"), str
                ):
                    yield event["delta"]
                elif event_type == "response.completed":
                    response = event.get("response")
                    if isinstance(response, dict):
                        usage = response.get("usage")
                        if isinstance(usage, dict):
                            self._record_usage(
                                ModelUsage(
                                    input_tokens=self._integer(usage.get("input_tokens")),
                                    output_tokens=self._integer(usage.get("output_tokens")),
                                    requests=1,
                                )
                            )

        return iterator()
