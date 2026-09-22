import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from mathmodel_ai.core.errors import (
    ProviderAuthenticationError,
    ProviderError,
    ProviderRedirectError,
    ProviderResponseError,
    ProviderResponseTooLargeError,
    ProviderTimeoutError,
)
from mathmodel_ai.providers.base import BaseModelProvider
from mathmodel_ai.providers.schemas import JsonObject, ModelUsage
from mathmodel_ai.providers.security import EndpointSecurityPolicy
from mathmodel_ai.schemas.provider_registry import ProviderEndpoint


def create_secure_async_client(
    *,
    timeout: httpx.Timeout,
    verify: bool,
    base_url: str | None = None,
) -> httpx.AsyncClient:
    """Create a direct client for an endpoint already guarded by our SSRF policy.

    Ambient proxy settings cannot preserve the validated DNS destination pin and
    may redirect credentials to an endpoint outside the provider trust contract.
    Explicit proxy support therefore requires a separate validated configuration.
    """

    if base_url is None:
        return httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=False,
            verify=verify,
            trust_env=False,
        )
    return httpx.AsyncClient(
        base_url=base_url,
        timeout=timeout,
        follow_redirects=False,
        verify=verify,
        trust_env=False,
    )


def secure_request_target(
    endpoint: ProviderEndpoint,
    security_policy: EndpointSecurityPolicy,
    path: str,
    headers: dict[str, str],
) -> tuple[httpx.URL, dict[str, str], dict[str, Any]]:
    """Resolve once, validate every address, and connect to that exact destination."""

    relative = httpx.URL(path)
    if not relative.is_relative_url:
        raise ProviderError("provider request path must be relative")
    addresses = security_policy.validate_destination(endpoint)
    base = httpx.URL(endpoint.base_url)
    base_path = base.raw_path.split(b"?", maxsplit=1)[0]
    if not base_path.endswith(b"/"):
        base_path += b"/"
    relative_path = relative.raw_path.split(b"?", maxsplit=1)[0].lstrip(b"/")
    original = base.copy_with(raw_path=base_path + relative_path, query=relative.query)
    pinned = original.copy_with(host=addresses[0])
    secured_headers = dict(headers)
    secured_headers["Host"] = original.netloc.decode("ascii")
    return pinned, secured_headers, {"sni_hostname": original.host}


class HttpModelProvider(BaseModelProvider):
    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float,
        client: httpx.AsyncClient | None = None,
        endpoint: ProviderEndpoint | None = None,
        security_policy: EndpointSecurityPolicy | None = None,
        max_response_bytes: int = 8 * 1024 * 1024,
    ) -> None:
        self._owns_client = client is None
        timeout = httpx.Timeout(
            timeout_seconds,
            connect=(endpoint.connect_timeout_seconds if endpoint is not None else timeout_seconds),
        )
        self._client = client or create_secure_async_client(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            verify=endpoint.verify_tls if endpoint is not None else True,
        )
        self._usage = ModelUsage()
        self._endpoint = endpoint
        self._security_policy = security_policy
        self._max_response_bytes = max_response_bytes
        self._total_timeout_seconds = timeout_seconds

    def get_usage(self) -> ModelUsage:
        return self._usage.model_copy(deep=True)

    def _record_usage(self, usage: ModelUsage) -> None:
        self._usage = self._usage.plus(usage)

    async def _post_json(
        self, path: str, *, headers: dict[str, str], payload: JsonObject
    ) -> JsonObject:
        request_target: str | httpx.URL = path
        request_headers = headers
        request_extensions: dict[str, Any] | None = None
        if self._endpoint is not None and self._security_policy is not None:
            request_target, request_headers, request_extensions = secure_request_target(
                self._endpoint, self._security_policy, path, headers
            )
        try:
            async with asyncio.timeout(self._total_timeout_seconds):
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
                        if len(body) > self._max_response_bytes:
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
            raise ProviderError(f"{self.name} returned HTTP {exc.response.status_code}") from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"{self.name} transport failure: {type(exc).__name__}") from exc
        try:
            data = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProviderResponseError(f"{self.name} returned invalid JSON") from exc
        if not isinstance(data, dict):
            raise ProviderResponseError(f"{self.name} returned a non-object response")
        return data

    async def _stream_events(
        self, path: str, *, headers: dict[str, str], payload: JsonObject
    ) -> AsyncIterator[JsonObject]:
        request_target: str | httpx.URL = path
        request_headers = headers
        request_extensions: dict[str, Any] | None = None
        if self._endpoint is not None and self._security_policy is not None:
            request_target, request_headers, request_extensions = secure_request_target(
                self._endpoint, self._security_policy, path, headers
            )
        try:
            async with asyncio.timeout(self._total_timeout_seconds):
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
                        if seen > self._max_response_bytes:
                            raise ProviderResponseTooLargeError("RESPONSE_TOO_LARGE")
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
            raise ProviderError(
                f"{self.name} stream returned HTTP {exc.response.status_code}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"{self.name} stream transport failure: {type(exc).__name__}"
            ) from exc

    @staticmethod
    def _required_dict(value: Any, context: str) -> JsonObject:
        if not isinstance(value, dict):
            raise ProviderResponseError(f"missing or invalid {context}")
        return value

    @staticmethod
    def _integer(value: Any) -> int | None:
        return value if type(value) is int and 0 <= value <= 1_000_000_000 else None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
