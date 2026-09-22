from __future__ import annotations

import ipaddress
import re
import socket
from collections.abc import Callable, Iterable
from urllib.parse import urlsplit

from mathmodel_ai.core.errors import ConfigurationError, ProviderError
from mathmodel_ai.core.types import Environment
from mathmodel_ai.schemas.provider_registry import (
    EndpointTrustLevel,
    ModelIdentityConfidence,
    ProviderEndpoint,
    ProviderProtocol,
)

AddressResolver = Callable[[str, int | None], Iterable[str]]

_SECRET_PATTERNS = (
    re.compile(r"Bearer\s+[A-Za-z0-9._~+/-]+", re.IGNORECASE),
    re.compile(r"(?:sk|key)-[A-Za-z0-9_-]{8,}", re.IGNORECASE),
    re.compile(
        r"(?i)(authorization|api[_-]?key|secret|password|cookie|credential)"
        r"\s*[:=]\s*[^\s,;]+"
    ),
    re.compile(r"-----BEGIN [A-Z ]+PRIVATE KEY-----[\s\S]*?-----END [A-Z ]+PRIVATE KEY-----"),
)

_OFFICIAL_ENDPOINTS: dict[str, frozenset[ProviderProtocol]] = {
    "api.openai.com": frozenset(
        {ProviderProtocol.OPENAI_RESPONSES, ProviderProtocol.OPENAI_CHAT_COMPLETIONS}
    ),
    "api.anthropic.com": frozenset({ProviderProtocol.ANTHROPIC_MESSAGES}),
    "generativelanguage.googleapis.com": frozenset({ProviderProtocol.GOOGLE_GENERATE_CONTENT}),
    "api.deepseek.com": frozenset({ProviderProtocol.OPENAI_CHAT_COMPLETIONS}),
}


class EndpointSecurityPolicy:
    def __init__(
        self,
        *,
        environment: Environment,
        allow_local_model_endpoints: bool = False,
        allow_insecure_provider_tls: bool = False,
        resolver: AddressResolver | None = None,
    ) -> None:
        self._environment = environment
        self._allow_local = allow_local_model_endpoints
        self._allow_insecure_tls = allow_insecure_provider_tls
        self._resolver = resolver or _resolve_addresses

    def validate_configuration(
        self, endpoint: ProviderEndpoint, *, resolve_dns: bool = True
    ) -> None:
        parsed = urlsplit(endpoint.base_url)
        hostname = parsed.hostname
        if hostname is None:
            raise ConfigurationError("provider endpoint has no hostname")
        scheme = parsed.scheme.lower()
        literal = _parse_ip(hostname)
        local_destination = hostname.lower() == "localhost" or (
            literal is not None and _is_forbidden_address(literal)
        )
        if scheme != "https":
            if not (scheme == "http" and self._allow_local and local_destination):
                raise ConfigurationError("provider endpoints require HTTPS by default")
        if not endpoint.verify_tls and (
            self._environment is Environment.PRODUCTION or not self._allow_insecure_tls
        ):
            raise ConfigurationError("TLS verification cannot be disabled by provider config")
        if endpoint.allow_redirects:
            raise ConfigurationError("provider redirects are disabled")
        self._validate_trust(endpoint, hostname)
        if local_destination and not self._allow_local:
            raise ConfigurationError("private or local provider destinations are disabled")
        if resolve_dns:
            self.validate_destination(endpoint)

    def validate_destination(self, endpoint: ProviderEndpoint) -> tuple[str, ...]:
        parsed = urlsplit(endpoint.base_url)
        hostname = parsed.hostname
        if hostname is None:
            raise ProviderError("provider destination has no hostname")
        try:
            addresses = tuple(dict.fromkeys(self._resolver(hostname, parsed.port)))
        except OSError as exc:
            raise ProviderError("provider DNS resolution failed") from exc
        if not addresses:
            raise ProviderError("provider DNS resolution returned no addresses")
        forbidden: list[str] = []
        for value in addresses:
            try:
                address = ipaddress.ip_address(value)
            except ValueError as exc:
                raise ProviderError("provider DNS returned an invalid address") from exc
            if _is_forbidden_address(address):
                forbidden.append(value)
        if forbidden and len(forbidden) != len(addresses):
            raise ProviderError("provider DNS returned mixed public and private destinations")
        if forbidden and not self._allow_local:
            raise ProviderError("provider DNS resolved to a private or local destination")
        return addresses

    @staticmethod
    def _validate_trust(endpoint: ProviderEndpoint, hostname: str) -> None:
        if endpoint.trust_level is not EndpointTrustLevel.OFFICIAL_VENDOR:
            return
        protocol_set = _OFFICIAL_ENDPOINTS.get(hostname.lower())
        is_qwen_official = (
            hostname.lower() == "dashscope.aliyuncs.com"
            or hostname.lower().endswith(".dashscope.aliyuncs.com")
        )
        if is_qwen_official:
            protocol_set = frozenset({ProviderProtocol.OPENAI_CHAT_COMPLETIONS})
        if protocol_set is None or endpoint.protocol not in protocol_set:
            raise ConfigurationError(
                "OFFICIAL_VENDOR requires a built-in official domain and matching protocol"
            )


def endpoint_identity_confidence(endpoint: ProviderEndpoint) -> ModelIdentityConfidence:
    if endpoint.trust_level is EndpointTrustLevel.OFFICIAL_VENDOR:
        return ModelIdentityConfidence.VENDOR_ENDPOINT
    if endpoint.trust_level is EndpointTrustLevel.USER_MANAGED_PROXY:
        return ModelIdentityConfidence.NOT_INDEPENDENTLY_VERIFIED
    return ModelIdentityConfidence.UNKNOWN


def redact_sensitive_text(value: str, *, known_secrets: Iterable[str] = ()) -> str:
    redacted = value
    for secret in known_secrets:
        if secret:
            redacted = redacted.replace(secret, "[REDACTED]")
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def safe_error(exc: BaseException, *, known_secrets: Iterable[str] = ()) -> str:
    return redact_sensitive_text(f"{type(exc).__name__}: {exc}", known_secrets=known_secrets)


def _resolve_addresses(host: str, port: int | None) -> Iterable[str]:
    return (
        str(item[4][0]) for item in socket.getaddrinfo(host, port or 443, type=socket.SOCK_STREAM)
    )


def _parse_ip(value: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return ipaddress.ip_address(value)
    except ValueError:
        return None


def _is_forbidden_address(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        return _is_forbidden_address(address.ipv4_mapped)
    return bool(
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    )
