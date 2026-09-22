from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

import httpx

from mathmodel_ai.core.errors import ConfigurationError, ModelDiscoveryError
from mathmodel_ai.providers.http import create_secure_async_client, secure_request_target
from mathmodel_ai.providers.secrets import BaseSecretResolver
from mathmodel_ai.providers.security import EndpointSecurityPolicy
from mathmodel_ai.schemas.provider_registry import ProviderEndpoint, ProviderProtocol


@dataclass(frozen=True)
class DiscoveredModel:
    remote_model_id: str
    display_name: str | None = None


class ProviderModelDiscovery:
    """Bounded, server-side model discovery for already persisted providers."""

    def __init__(
        self,
        *,
        secrets: BaseSecretResolver,
        security_policy: EndpointSecurityPolicy,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._secrets = secrets
        self._security = security_policy
        self._client = client

    async def discover(self, endpoint: ProviderEndpoint) -> list[DiscoveredModel]:
        if endpoint.credential_ref is None:
            raise ModelDiscoveryError(
                "MISSING_CREDENTIAL",
                "Configure a credential before fetching models.",
                status_code=400,
            )
        try:
            credential = self._secrets.resolve(endpoint.credential_ref).get_secret_value()
        except ConfigurationError as exc:
            raise ModelDiscoveryError(
                "MISSING_CREDENTIAL",
                "The configured credential is unavailable.",
                status_code=400,
            ) from exc

        path, headers = self._request(endpoint, credential)
        request_target, request_headers, request_extensions = secure_request_target(
            endpoint,
            self._security,
            path,
            {**headers, **endpoint.additional_headers},
        )
        owns_client = self._client is None
        client = self._client or create_secure_async_client(
            timeout=httpx.Timeout(
                endpoint.timeout_seconds,
                connect=endpoint.connect_timeout_seconds,
            ),
            verify=endpoint.verify_tls,
        )
        try:
            body = await self._get(
                client,
                request_target,
                request_headers,
                request_extensions,
                endpoint,
            )
        finally:
            if owns_client:
                await client.aclose()
        return self._parse(endpoint.protocol, body)

    @staticmethod
    def _request(endpoint: ProviderEndpoint, credential: str) -> tuple[str, dict[str, str]]:
        if endpoint.protocol in {
            ProviderProtocol.OPENAI_CHAT_COMPLETIONS,
            ProviderProtocol.OPENAI_RESPONSES,
        }:
            return "/models", {"Authorization": f"Bearer {credential}"}
        if endpoint.protocol is ProviderProtocol.ANTHROPIC_MESSAGES:
            return "/models", {
                "x-api-key": credential,
                "anthropic-version": "2023-06-01",
            }
        if endpoint.protocol is ProviderProtocol.GOOGLE_GENERATE_CONTENT:
            return "/models", {"x-goog-api-key": credential}
        raise ModelDiscoveryError(
            "MODEL_DISCOVERY_UNSUPPORTED",
            "Model discovery is unavailable for this protocol. Enter model IDs manually.",
            status_code=409,
        )

    @staticmethod
    async def _get(
        client: httpx.AsyncClient,
        request_target: httpx.URL,
        headers: dict[str, str],
        extensions: dict[str, Any],
        endpoint: ProviderEndpoint,
    ) -> bytes:
        try:
            async with asyncio.timeout(endpoint.timeout_seconds):
                async with client.stream(
                    "GET",
                    request_target,
                    headers=headers,
                    extensions=extensions,
                ) as response:
                    if response.status_code in {401, 403}:
                        raise ModelDiscoveryError(
                            "AUTHENTICATION_FAILED",
                            "Credential rejected by the provider.",
                            status_code=401,
                        )
                    if response.status_code in {404, 405}:
                        raise ModelDiscoveryError(
                            "MODEL_DISCOVERY_UNSUPPORTED",
                            "Model discovery is unavailable. Enter model IDs manually.",
                            status_code=409,
                            provider_reached=True,
                        )
                    if 300 <= response.status_code < 400:
                        raise ModelDiscoveryError(
                            "MODEL_DISCOVERY_FAILED",
                            "Model discovery redirects are blocked.",
                        )
                    if response.status_code < 200 or response.status_code >= 300:
                        raise ModelDiscoveryError(
                            "MODEL_DISCOVERY_FAILED",
                            "The provider could not return a model catalog.",
                        )
                    body = bytearray()
                    async for chunk in response.aiter_bytes():
                        body.extend(chunk)
                        if len(body) > endpoint.max_response_bytes:
                            raise ModelDiscoveryError(
                                "MODEL_DISCOVERY_FAILED",
                                "The provider model catalog exceeded the response limit.",
                            )
                    return bytes(body)
        except ModelDiscoveryError:
            raise
        except (
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.TimeoutException,
            TimeoutError,
        ) as exc:
            raise ModelDiscoveryError(
                "PROVIDER_UNREACHABLE",
                "The provider did not respond before the timeout.",
            ) from exc
        except httpx.HTTPError as exc:
            raise ModelDiscoveryError(
                "PROVIDER_UNREACHABLE",
                "The provider endpoint could not be reached.",
            ) from exc

    @staticmethod
    def _parse(protocol: ProviderProtocol, body: bytes) -> list[DiscoveredModel]:
        try:
            data = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ModelDiscoveryError(
                "MODEL_DISCOVERY_FAILED",
                "The provider returned an invalid model catalog.",
            ) from exc
        if not isinstance(data, dict):
            raise ModelDiscoveryError(
                "MODEL_DISCOVERY_FAILED",
                "The provider returned an invalid model catalog.",
            )
        raw_models = data.get(
            "models" if protocol is ProviderProtocol.GOOGLE_GENERATE_CONTENT else "data"
        )
        if not isinstance(raw_models, list):
            raise ModelDiscoveryError(
                "MODEL_DISCOVERY_FAILED",
                "The provider returned an invalid model catalog.",
            )

        discovered: list[DiscoveredModel] = []
        seen: set[str] = set()
        for item in raw_models[:10_000]:
            if not isinstance(item, dict):
                continue
            raw_identifier = item.get(
                "name" if protocol is ProviderProtocol.GOOGLE_GENERATE_CONTENT else "id"
            )
            if not isinstance(raw_identifier, str):
                continue
            identifier = raw_identifier.strip()
            if protocol is ProviderProtocol.GOOGLE_GENERATE_CONTENT:
                identifier = identifier.removeprefix("models/")
            if not identifier or len(identifier) > 500 or identifier in seen:
                continue
            raw_display_name = item.get("displayName", item.get("display_name"))
            display_name = (
                raw_display_name.strip()[:255]
                if isinstance(raw_display_name, str) and raw_display_name.strip()
                else None
            )
            seen.add(identifier)
            discovered.append(
                DiscoveredModel(remote_model_id=identifier, display_name=display_name)
            )
        return discovered
