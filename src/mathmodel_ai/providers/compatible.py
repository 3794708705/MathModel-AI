from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from mathmodel_ai.core.errors import (
    ProviderAuthenticationError,
    ProviderError,
    ProviderRedirectError,
    ProviderResponseError,
    ProviderResponseTooLargeError,
    ProviderTimeoutError,
)
from mathmodel_ai.providers.base import BaseModelProvider
from mathmodel_ai.providers.http import create_secure_async_client, secure_request_target
from mathmodel_ai.providers.parameters import ProtocolParameterMapper
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
from mathmodel_ai.schemas.provider_registry import (
    CustomAuthScheme,
    CustomJSONConfiguration,
    ModelCapability,
    ModelProfile,
    ProviderEndpoint,
    StructuredOutputStrategy,
)


class SecureEndpointProvider(BaseModelProvider):
    def __init__(
        self,
        *,
        endpoint: ProviderEndpoint,
        model_profile: ModelProfile,
        api_key: str | None,
        security_policy: EndpointSecurityPolicy,
        client: httpx.AsyncClient | None = None,
        credential_supplier: CredentialSupplier | None = None,
    ) -> None:
        self.name = endpoint.provider_id
        self.endpoint = endpoint
        self.model_profile = model_profile
        self._credential_supplier = build_credential_supplier(
            api_key=api_key, supplier=credential_supplier
        )
        self._security_policy = security_policy
        self._owns_client = client is None
        timeout = httpx.Timeout(
            endpoint.timeout_seconds,
            connect=endpoint.connect_timeout_seconds,
        )
        self._client = client or create_secure_async_client(
            base_url=endpoint.base_url,
            timeout=timeout,
            verify=endpoint.verify_tls,
        )
        self._usage = ModelUsage()

    def get_usage(self) -> ModelUsage:
        return self._usage.model_copy(deep=True)

    def _record_usage(self, usage: ModelUsage) -> None:
        self._usage = self._usage.plus(usage)

    def _base_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._credential_supplier()}",
            "Content-Type": "application/json",
            **self.endpoint.additional_headers,
        }

    async def _post_json(
        self,
        path: str,
        *,
        payload: JsonObject,
        headers: dict[str, str] | None = None,
    ) -> JsonObject:
        request_target, request_headers, request_extensions = secure_request_target(
            self.endpoint,
            self._security_policy,
            path,
            headers or self._base_headers(),
        )
        try:
            async with asyncio.timeout(self.endpoint.timeout_seconds):
                async with self._client.stream(
                    "POST",
                    request_target,
                    headers=request_headers,
                    json=payload,
                    extensions=request_extensions,
                ) as response:
                    if 300 <= response.status_code < 400:
                        raise ProviderRedirectError("PROVIDER_REDIRECT_BLOCKED")
                    if response.status_code in {401, 403}:
                        raise ProviderAuthenticationError("AUTH_FAILED")
                    response.raise_for_status()
                    body = bytearray()
                    async for chunk in response.aiter_bytes():
                        body.extend(chunk)
                        if len(body) > self.endpoint.max_response_bytes:
                            raise ProviderResponseTooLargeError("RESPONSE_TOO_LARGE")
        except ProviderError:
            raise
        except httpx.ConnectTimeout as exc:
            raise ProviderTimeoutError("PROVIDER_TIMEOUT_CONNECT") from exc
        except httpx.ReadTimeout as exc:
            raise ProviderTimeoutError("PROVIDER_TIMEOUT_READ") from exc
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError("PROVIDER_TIMEOUT_HTTP") from exc
        except TimeoutError as exc:
            raise ProviderTimeoutError("PROVIDER_TIMEOUT_TOTAL") from exc
        except httpx.HTTPStatusError as exc:
            raise ProviderError(f"provider returned HTTP {exc.response.status_code}") from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"provider transport failure: {type(exc).__name__}") from exc
        try:
            data = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProviderResponseError("provider returned invalid JSON") from exc
        if not isinstance(data, dict):
            raise ProviderResponseError("provider returned a non-object response")
        return data

    async def _stream_sse(self, path: str, *, payload: JsonObject) -> AsyncIterator[JsonObject]:
        request_target, request_headers, request_extensions = secure_request_target(
            self.endpoint,
            self._security_policy,
            path,
            self._base_headers(),
        )
        try:
            async with asyncio.timeout(self.endpoint.timeout_seconds):
                async with self._client.stream(
                    "POST",
                    request_target,
                    headers=request_headers,
                    json=payload,
                    extensions=request_extensions,
                ) as response:
                    if 300 <= response.status_code < 400:
                        raise ProviderRedirectError("PROVIDER_REDIRECT_BLOCKED")
                    if response.status_code in {401, 403}:
                        raise ProviderAuthenticationError("AUTH_FAILED")
                    response.raise_for_status()
                    seen = 0
                    async for line in response.aiter_lines():
                        seen += len(line.encode("utf-8"))
                        if seen > self.endpoint.max_response_bytes:
                            raise ProviderResponseTooLargeError("RESPONSE_TOO_LARGE")
                        if not line.startswith("data:"):
                            continue
                        raw = line[5:].strip()
                        if not raw or raw == "[DONE]":
                            continue
                        try:
                            item = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(item, dict):
                            yield item
        except ProviderError:
            raise
        except httpx.ConnectTimeout as exc:
            raise ProviderTimeoutError("PROVIDER_TIMEOUT_CONNECT") from exc
        except httpx.ReadTimeout as exc:
            raise ProviderTimeoutError("PROVIDER_TIMEOUT_READ") from exc
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError("PROVIDER_TIMEOUT_HTTP") from exc
        except TimeoutError as exc:
            raise ProviderTimeoutError("PROVIDER_TIMEOUT_TOTAL") from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"provider stream failure: {type(exc).__name__}") from exc

    def _response_metadata(
        self, *, structured_output_mode: str | None = None, reasoning_effective: str | None = None
    ) -> dict[str, Any]:
        return {
            "provider_config_digest": self.endpoint.config_digest,
            "model_id": self.model_profile.model_id,
            "model_config_digest": self.model_profile.config_digest,
            "protocol": self.endpoint.protocol.value,
            "endpoint_trust": self.endpoint.trust_level.value,
            "structured_output_mode": structured_output_mode,
            "reasoning_effective": reasoning_effective,
        }

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


class OpenAICompatibleProvider(SecureEndpointProvider):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._parameters = ProtocolParameterMapper()

    def _messages(self, request: GenerationRequest) -> list[JsonObject]:
        messages: list[JsonObject] = []
        supports_developer = self.model_profile.supports(
            ModelCapability.DEVELOPER_ROLE, allow_partial=True
        )
        supports_vision = self.model_profile.supports(ModelCapability.VISION, allow_partial=True)
        for message in request.messages:
            role = (
                "system" if message.role == "developer" and not supports_developer else message.role
            )
            if not message.media:
                messages.append({"role": role, "content": message.content})
                continue
            if not supports_vision:
                raise ProviderError("configured model does not support vision input")
            content: list[JsonObject] = []
            if message.content:
                content.append({"type": "text", "text": message.content})
            content.extend(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{media.mime_type};base64,{media.data_base64}"},
                }
                for media in message.media
            )
            messages.append({"role": role, "content": content})
        return messages

    def _payload(self, request: GenerationRequest) -> tuple[JsonObject, str | None]:
        payload: JsonObject = {
            "model": self.model_profile.remote_model,
            "messages": self._messages(request),
            "max_tokens": min(
                request.max_output_tokens,
                self.model_profile.max_output_tokens or request.max_output_tokens,
            ),
        }
        if request.temperature is not None and self._parameters.allows_parameter(
            self.model_profile, "temperature"
        ):
            payload["temperature"] = request.temperature
        reasoning = self._parameters.reasoning_value(self.model_profile, request.reasoning_effort)
        if reasoning is not None:
            parameter_name = str(
                self.model_profile.configuration.get("reasoning_parameter", "reasoning_effort")
            )
            if parameter_name not in {"reasoning_effort", "reasoning", "thinking"}:
                raise ProviderError("unsupported configured reasoning parameter")
            payload[parameter_name] = reasoning
        if request.tools:
            if not self._parameters.allows_tools(self.model_profile):
                raise ProviderError("configured model does not support native tools")
            payload["tools"] = [tool.model_dump(mode="json") for tool in request.tools]
            payload["tool_choice"] = "auto"
        return payload, reasoning

    @staticmethod
    def _parse_response(data: JsonObject) -> tuple[str, JsonObject, JsonObject]:
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ProviderResponseError("compatible response contained no choice")
        choice = choices[0]
        message = choice.get("message")
        if not isinstance(message, dict):
            raise ProviderResponseError("compatible response contained no message")
        content = message.get("content")
        if content is None:
            content = ""
        if not isinstance(content, str):
            raise ProviderResponseError("compatible response content is not text")
        return content, choice, message

    def _response(
        self,
        data: JsonObject,
        *,
        structured_output_mode: str | None = None,
        reasoning_effective: str | None = None,
    ) -> ModelResponse:
        content, choice, message = self._parse_response(data)
        usage_value = data.get("usage")
        usage_data = usage_value if isinstance(usage_value, dict) else {}
        completion_details = usage_data.get("completion_tokens_details")
        details = completion_details if isinstance(completion_details, dict) else {}
        usage = ModelUsage(
            input_tokens=_optional_nonnegative_int(usage_data.get("prompt_tokens")),
            output_tokens=_optional_nonnegative_int(usage_data.get("completion_tokens")),
            cached_input_tokens=_optional_nonnegative_int(usage_data.get("cached_prompt_tokens")),
            reasoning_tokens=_optional_nonnegative_int(details.get("reasoning_tokens")),
            total_tokens=_optional_nonnegative_int(usage_data.get("total_tokens")),
            requests=1,
        )
        self._record_usage(usage)
        tool_calls = message.get("tool_calls")
        tool_call_names = _tool_call_names(tool_calls)
        return ModelResponse(
            content=content,
            provider=self.name,
            model=str(data.get("model", self.model_profile.remote_model)),
            usage=usage,
            response_id=str(data["id"]) if data.get("id") else None,
            finish_reason=str(choice["finish_reason"]) if choice.get("finish_reason") else None,
            tool_call_count=len(tool_calls) if isinstance(tool_calls, list) else 0,
            tool_call_names=tool_call_names,
            **self._response_metadata(
                structured_output_mode=structured_output_mode,
                reasoning_effective=reasoning_effective,
            ),
        )

    async def generate(self, request: GenerationRequest) -> ModelResponse:
        payload, reasoning = self._payload(request)
        data = await self._post_json("/chat/completions", payload=payload)
        return self._response(data, reasoning_effective=reasoning)

    async def structured_generate[T: BaseModel](
        self, request: GenerationRequest, response_model: type[T]
    ) -> StructuredModelResponse[T]:
        payload, reasoning = self._payload(request)
        strategy = self._parameters.structured_strategy(self.model_profile)
        if strategy is StructuredOutputStrategy.NATIVE_JSON_SCHEMA:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "schema": response_model.model_json_schema(),
                    "strict": True,
                },
            }
        elif strategy is StructuredOutputStrategy.JSON_MODE:
            payload["response_format"] = {"type": "json_object"}
            payload["messages"] = [
                {
                    "role": "system",
                    "content": (
                        "Return concise, valid JSON matching this JSON Schema. "
                        "Include every property listed in required and do not invent extra "
                        "properties: "
                    )
                    + json.dumps(response_model.model_json_schema(), separators=(",", ":")),
                },
                *payload["messages"],
            ]
        elif strategy is StructuredOutputStrategy.PROMPT_JSON_FALLBACK:
            payload["messages"] = [
                {
                    "role": "system",
                    "content": "Return only JSON matching this schema: "
                    + json.dumps(response_model.model_json_schema(), separators=(",", ":")),
                },
                *payload["messages"],
            ]
        else:
            raise ProviderError("structured output is unsupported by the configured model")
        data = await self._post_json("/chat/completions", payload=payload)
        response = self._response(
            data,
            structured_output_mode=strategy.value,
            reasoning_effective=reasoning,
        )
        try:
            parsed = response_model.model_validate_json(response.content)
        except ValidationError as exc:
            syntax = ""
            if any(error["type"] == "json_invalid" for error in exc.errors()):
                try:
                    json.loads(response.content)
                except json.JSONDecodeError as decode_error:
                    # JSONDecodeError.msg is parser-authored; never include its
                    # doc (the provider response) in persisted retry diagnostics.
                    syntax = (
                        f"json_syntax={decode_error.msg}; line={decode_error.lineno}; "
                        f"column={decode_error.colno}; position={decode_error.pos}; "
                    )
            details = ", ".join(
                f"{'.'.join(str(part) for part in error['loc']) or '<root>'}:{error['type']}"
                for error in exc.errors(include_input=False, include_url=False)[:8]
            )
            raise ProviderResponseError(
                f"structured response failed schema validation "
                f"({exc.error_count()} errors: {details}; {syntax}"
                f"finish_reason={response.finish_reason or 'unknown'}; "
                f"content_chars={len(response.content)}; "
                f"content_non_whitespace={bool(response.content.strip())})"
            ) from exc
        return StructuredModelResponse(parsed=parsed, response=response)

    def stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        async def iterator() -> AsyncIterator[str]:
            payload, _ = self._payload(request)
            payload["stream"] = True
            last_usage: JsonObject = {}
            async for event in self._stream_sse("/chat/completions", payload=payload):
                usage = event.get("usage")
                if isinstance(usage, dict):
                    last_usage = usage
                choices = event.get("choices")
                if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                    continue
                delta = choices[0].get("delta")
                if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                    yield delta["content"]
            self._record_usage(
                ModelUsage(
                    input_tokens=_optional_nonnegative_int(last_usage.get("prompt_tokens")),
                    output_tokens=_optional_nonnegative_int(last_usage.get("completion_tokens")),
                    total_tokens=_optional_nonnegative_int(last_usage.get("total_tokens")),
                    requests=1,
                )
            )

        return iterator()


class CustomJSONHttpProvider(SecureEndpointProvider):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        try:
            self._mapping = CustomJSONConfiguration.model_validate(
                self.model_profile.configuration["custom_json"]
            )
        except (KeyError, ValueError) as exc:
            raise ProviderError(
                "CUSTOM_JSON_HTTP requires valid custom_json configuration"
            ) from exc

    def _base_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **self.endpoint.additional_headers}
        if self._mapping.auth_scheme is CustomAuthScheme.BEARER:
            headers["Authorization"] = f"Bearer {self._credential_supplier()}"
        else:
            if self._mapping.api_key_header is None:
                raise ProviderError("custom API key header is missing")
            headers[self._mapping.api_key_header] = self._credential_supplier()
        return headers

    def _payload(self, request: GenerationRequest) -> JsonObject:
        system = "\n\n".join(
            item.content for item in request.messages if item.role in {"system", "developer"}
        )
        messages = [
            {"role": item.role, "content": item.content}
            for item in request.messages
            if item.role not in {"system", "developer"}
        ]
        prompt = "\n\n".join(item["content"] for item in messages)
        context: dict[str, Any] = {
            "model": self.model_profile.remote_model,
            "system": system,
            "messages": messages,
            "prompt": prompt,
            "max_tokens": request.max_output_tokens,
        }
        rendered = _render_static_template(self._mapping.request.body, context)
        if not isinstance(rendered, dict):
            raise ProviderError("custom request mapping must render a JSON object")
        return rendered

    def _response(self, data: JsonObject, *, mode: str | None = None) -> ModelResponse:
        text = _field_path(data, self._mapping.response.text_path)
        if not isinstance(text, str):
            raise ProviderResponseError("custom response text_path did not resolve to text")
        usage = ModelUsage(
            input_tokens=_path_int(data, self._mapping.response.input_tokens_path),
            output_tokens=_path_int(data, self._mapping.response.output_tokens_path),
            cached_input_tokens=_path_int(data, self._mapping.response.cached_input_tokens_path),
            reasoning_tokens=_path_optional_int(data, self._mapping.response.reasoning_tokens_path),
            total_tokens=_path_optional_int(data, self._mapping.response.total_tokens_path),
            requests=1,
        )
        self._record_usage(usage)
        return ModelResponse(
            content=text,
            provider=self.name,
            model=self.model_profile.remote_model,
            usage=usage,
            **self._response_metadata(structured_output_mode=mode),
        )

    async def generate(self, request: GenerationRequest) -> ModelResponse:
        data = await self._post_json(
            self._mapping.request.endpoint_path,
            payload=self._payload(request),
        )
        return self._response(data)

    async def structured_generate[T: BaseModel](
        self, request: GenerationRequest, response_model: type[T]
    ) -> StructuredModelResponse[T]:
        if self.model_profile.structured_output_strategy is not (
            StructuredOutputStrategy.PROMPT_JSON_FALLBACK
        ):
            raise ProviderError("CUSTOM_JSON_HTTP structured output requires prompt fallback")
        schema_message = GenerationRequest(
            **request.model_dump(exclude={"messages"}),
            messages=[
                *request.messages,
                {
                    "role": "user",
                    "content": "Return only JSON matching: "
                    + json.dumps(response_model.model_json_schema(), separators=(",", ":")),
                },
            ],
        )
        data = await self._post_json(
            self._mapping.request.endpoint_path,
            payload=self._payload(schema_message),
        )
        response = self._response(data, mode=StructuredOutputStrategy.PROMPT_JSON_FALLBACK.value)
        try:
            parsed = response_model.model_validate_json(response.content)
        except ValueError as exc:
            raise ProviderResponseError("custom structured response failed validation") from exc
        return StructuredModelResponse(parsed=parsed, response=response)

    def stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        async def unsupported() -> AsyncIterator[str]:
            raise ProviderError("CUSTOM_JSON_HTTP streaming is unsupported")
            yield ""  # pragma: no cover

        return unsupported()


def _render_static_template(value: Any, context: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        return {key: _render_static_template(item, context) for key, item in value.items()}
    if isinstance(value, list):
        return [_render_static_template(item, context) for item in value]
    if isinstance(value, str):
        exact = re_full_placeholder(value)
        if exact is not None:
            return context[exact]
        rendered = value
        for key, item in context.items():
            rendered = rendered.replace("{" + key + "}", str(item))
        return rendered
    return value


def re_full_placeholder(value: str) -> str | None:
    if len(value) >= 3 and value[0] == "{" and value[-1] == "}":
        key = value[1:-1]
        if key in {"model", "system", "messages", "prompt", "max_tokens"}:
            return key
    return None


def _field_path(data: JsonObject, path: str) -> Any:
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _path_int(data: JsonObject, path: str | None) -> int | None:
    return _optional_nonnegative_int(_field_path(data, path)) if path else None


def _path_optional_int(data: JsonObject, path: str | None) -> int | None:
    return _optional_nonnegative_int(_field_path(data, path)) if path else None


def _optional_nonnegative_int(value: Any) -> int | None:
    return value if type(value) is int and 0 <= value <= 1_000_000_000 else None


def _tool_call_names(value: Any) -> list[str]:
    if not isinstance(value, list) or len(value) > 32:
        return []
    names: list[str] = []
    for item in value:
        if not isinstance(item, dict) or not isinstance(item.get("function"), dict):
            return []
        name = item["function"].get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,99}", name):
            return []
        names.append(name)
    return names
