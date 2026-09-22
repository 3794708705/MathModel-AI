from collections.abc import AsyncIterator
from urllib.parse import quote

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
from mathmodel_ai.providers.secrets import (
    CredentialSupplier,
)
from mathmodel_ai.providers.secrets import (
    credential_supplier as build_credential_supplier,
)
from mathmodel_ai.providers.security import EndpointSecurityPolicy
from mathmodel_ai.schemas.provider_registry import ProviderEndpoint


class GoogleProvider(HttpModelProvider):
    name = ProviderName.GOOGLE

    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str,
        timeout_seconds: float,
        client: httpx.AsyncClient | None = None,
        endpoint: ProviderEndpoint | None = None,
        security_policy: EndpointSecurityPolicy | None = None,
        max_response_bytes: int = 8 * 1024 * 1024,
        credential_supplier: CredentialSupplier | None = None,
    ) -> None:
        super().__init__(
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            client=client,
            endpoint=endpoint,
            security_policy=security_policy,
            max_response_bytes=max_response_bytes,
        )
        self._credential_supplier = build_credential_supplier(
            api_key=api_key, supplier=credential_supplier
        )

    def _headers(self) -> dict[str, str]:
        return {
            "x-goog-api-key": self._credential_supplier(),
            "content-type": "application/json",
        }

    @staticmethod
    def _payload(request: GenerationRequest) -> JsonObject:
        system = "\n\n".join(
            message.content
            for message in request.messages
            if message.role in {"system", "developer"}
        )
        contents = []
        for message in request.messages:
            if message.role in {"system", "developer"}:
                continue
            parts: list[JsonObject] = []
            if message.content:
                parts.append({"text": message.content})
            parts.extend(
                {
                    "inlineData": {
                        "mimeType": media.mime_type,
                        "data": media.data_base64,
                    }
                }
                for media in message.media
            )
            contents.append(
                {
                    "role": "model" if message.role == "assistant" else "user",
                    "parts": parts,
                }
            )
        generation: JsonObject = {"maxOutputTokens": request.max_output_tokens}
        if request.temperature is not None:
            generation["temperature"] = request.temperature
        payload: JsonObject = {"contents": contents, "generationConfig": generation}
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        return payload

    @staticmethod
    def _text(data: JsonObject) -> str:
        candidates = data.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ProviderResponseError("google response contained no candidates")
        candidate = candidates[0]
        if not isinstance(candidate, dict) or not isinstance(candidate.get("content"), dict):
            raise ProviderResponseError("google response candidate has no content")
        parts = candidate["content"].get("parts")
        if not isinstance(parts, list):
            raise ProviderResponseError("google response content has no parts")
        pieces = [
            part["text"]
            for part in parts
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        ]
        if not pieces:
            raise ProviderResponseError("google response contained no text")
        return "".join(pieces)

    def _response(self, data: JsonObject, requested_model: str) -> ModelResponse:
        usage_value = data.get("usageMetadata")
        usage_data = usage_value if isinstance(usage_value, dict) else {}
        usage = ModelUsage(
            input_tokens=self._integer(usage_data.get("promptTokenCount")),
            output_tokens=self._integer(usage_data.get("candidatesTokenCount")),
            cached_input_tokens=self._integer(usage_data.get("cachedContentTokenCount")),
            requests=1,
        )
        self._record_usage(usage)
        finish_reason: str | None = None
        candidates = data.get("candidates")
        if isinstance(candidates, list) and candidates and isinstance(candidates[0], dict):
            value = candidates[0].get("finishReason")
            finish_reason = str(value) if value else None
        return ModelResponse(
            content=self._text(data),
            provider=self.name,
            model=str(data.get("modelVersion", requested_model)),
            usage=usage,
            response_id=str(data["responseId"]) if data.get("responseId") else None,
            finish_reason=finish_reason,
        )

    @staticmethod
    def _path(model: str, method: str) -> str:
        return f"/models/{quote(model, safe='')}:{method}"

    async def generate(self, request: GenerationRequest) -> ModelResponse:
        data = await self._post_json(
            self._path(request.model, "generateContent"),
            headers=self._headers(),
            payload=self._payload(request),
        )
        return self._response(data, request.model)

    async def structured_generate[T: BaseModel](
        self, request: GenerationRequest, response_model: type[T]
    ) -> StructuredModelResponse[T]:
        payload = self._payload(request)
        generation = self._required_dict(payload["generationConfig"], "generationConfig")
        generation["responseMimeType"] = "application/json"
        generation["responseJsonSchema"] = response_model.model_json_schema()
        data = await self._post_json(
            self._path(request.model, "generateContent"), headers=self._headers(), payload=payload
        )
        response = self._response(data, request.model)
        return StructuredModelResponse(
            parsed=_validated_response(response_model, response.content), response=response
        )

    def stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        async def iterator() -> AsyncIterator[str]:
            path = f"{self._path(request.model, 'streamGenerateContent')}?alt=sse"
            last_usage: dict[str, object] = {}
            async for event in self._stream_events(
                path, headers=self._headers(), payload=self._payload(request)
            ):
                try:
                    yield self._text(event)
                except ProviderResponseError:
                    continue
                usage = event.get("usageMetadata")
                if isinstance(usage, dict):
                    last_usage = usage
            self._record_usage(
                ModelUsage(
                    input_tokens=self._integer(last_usage.get("promptTokenCount")),
                    output_tokens=self._integer(last_usage.get("candidatesTokenCount")),
                    requests=1,
                )
            )

        return iterator()


def _validated_response[T: BaseModel](model: type[T], content: str) -> T:
    try:
        return model.model_validate_json(content)
    except ValueError as exc:
        raise ProviderResponseError("google structured response failed validation") from exc
