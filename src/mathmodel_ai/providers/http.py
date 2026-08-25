import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from mathmodel_ai.core.errors import ProviderError, ProviderResponseError
from mathmodel_ai.providers.base import BaseModelProvider
from mathmodel_ai.providers.schemas import JsonObject, ModelUsage


class HttpModelProvider(BaseModelProvider):
    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"), timeout=timeout_seconds
        )
        self._usage = ModelUsage()

    def get_usage(self) -> ModelUsage:
        return self._usage.model_copy(deep=True)

    def _record_usage(self, usage: ModelUsage) -> None:
        self._usage = self._usage.plus(usage)

    async def _post_json(
        self, path: str, *, headers: dict[str, str], payload: JsonObject
    ) -> JsonObject:
        try:
            response = await self._client.post(path, headers=headers, json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            request_id = exc.response.headers.get("request-id", "unknown")
            raise ProviderError(
                f"{self.name.value} returned HTTP {exc.response.status_code}; "
                f"request_id={request_id}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"{self.name.value} transport failure: {type(exc).__name__}"
            ) from exc
        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderResponseError(f"{self.name.value} returned invalid JSON") from exc
        if not isinstance(data, dict):
            raise ProviderResponseError(f"{self.name.value} returned a non-object response")
        return data

    async def _stream_events(
        self, path: str, *, headers: dict[str, str], payload: JsonObject
    ) -> AsyncIterator[JsonObject]:
        try:
            async with self._client.stream("POST", path, headers=headers, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if not raw or raw == "[DONE]":
                        continue
                    try:
                        event = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(event, dict):
                        yield event
        except httpx.HTTPStatusError as exc:
            request_id = exc.response.headers.get("request-id", "unknown")
            raise ProviderError(
                f"{self.name.value} stream returned HTTP {exc.response.status_code}; "
                f"request_id={request_id}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"{self.name.value} stream transport failure: {type(exc).__name__}"
            ) from exc

    @staticmethod
    def _required_dict(value: Any, context: str) -> JsonObject:
        if not isinstance(value, dict):
            raise ProviderResponseError(f"missing or invalid {context}")
        return value

    @staticmethod
    def _integer(value: Any) -> int:
        return value if isinstance(value, int) and value >= 0 else 0

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
