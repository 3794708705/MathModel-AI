from __future__ import annotations

import time
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from mathmodel_ai.core.errors import (
    ConfigurationError,
    ProviderAuthenticationError,
    ProviderError,
    ProviderTimeoutError,
)
from mathmodel_ai.core.types import ReasoningEffort
from mathmodel_ai.providers.factory import ProviderRegistry
from mathmodel_ai.providers.repository import ProviderModelRegistry
from mathmodel_ai.providers.schemas import (
    GenerationRequest,
    MediaPart,
    ModelMessage,
    ToolDefinition,
    ToolFunction,
)
from mathmodel_ai.providers.security import safe_error
from mathmodel_ai.schemas.provider_registry import (
    CapabilityEvidence,
    CapabilityProbeResult,
    CapabilitySource,
    CapabilityStatus,
    ModelCapability,
    ProbeAuthenticationStatus,
    StructuredOutputStrategy,
)

_ONE_PIXEL_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl6nE8AAAAASUVORK5CYII="
)


class _ProbeAnswer(BaseModel):
    ok: bool
    value: int = Field(ge=1, le=1)


class ProviderCompatibilityProbe:
    """Low-cost behavioral probe; it proves compatibility, not vendor model identity."""

    def __init__(
        self,
        *,
        configurations: ProviderModelRegistry,
        providers: ProviderRegistry,
    ) -> None:
        self._configurations = configurations
        self._providers = providers

    async def run(
        self, model_id: str, *, probe_long_context: bool = False
    ) -> CapabilityProbeResult:
        model = self._configurations.get_model(model_id)
        endpoint = self._configurations.get_provider(model.provider_id)
        started = time.perf_counter()
        capabilities = {
            capability: CapabilityEvidence(
                status=CapabilityStatus.UNKNOWN,
                source=CapabilitySource.PROBED,
            )
            for capability in ModelCapability
        }
        errors: list[str] = []
        authentication = ProbeAuthenticationStatus.UNAVAILABLE

        if not endpoint.enabled or not model.enabled:
            errors.append("provider or model is disabled")
            return self._persist(
                endpoint.config_digest,
                model.config_digest,
                endpoint.provider_id,
                model.model_id,
                capabilities,
                ProbeAuthenticationStatus.UNAVAILABLE,
                started,
                errors,
            )
        try:
            credential_configured = self._configurations.credential_configured(endpoint.provider_id)
        except Exception as exc:
            errors.append("credential reference could not be resolved")
            errors.append(safe_error(exc))
            credential_configured = False
        if not credential_configured:
            errors.append("credential reference is not configured")
            return self._persist(
                endpoint.config_digest,
                model.config_digest,
                endpoint.provider_id,
                model.model_id,
                capabilities,
                ProbeAuthenticationStatus.NOT_CONFIGURED,
                started,
                errors,
            )

        try:
            provider = self._providers.get(
                endpoint.provider_id, model_id=model.model_id, for_probe=True
            )
            basic = await provider.generate(self._basic_request(model.remote_model))
            authentication = ProbeAuthenticationStatus.PASS
            capabilities[ModelCapability.TEXT] = _supported("basic text response succeeded")
            capabilities[ModelCapability.SYSTEM_ROLE] = _supported("system-role request succeeded")
            if basic.usage.requests > 0 and (
                (basic.usage.input_tokens or 0) > 0 or (basic.usage.output_tokens or 0) > 0
            ):
                capabilities[ModelCapability.USAGE_REPORTING] = _supported(
                    "provider returned token usage"
                )
        except ProviderAuthenticationError as exc:
            errors.append(safe_error(exc))
            authentication = ProbeAuthenticationStatus.AUTH_FAILED
            return self._persist(
                endpoint.config_digest,
                model.config_digest,
                endpoint.provider_id,
                model.model_id,
                capabilities,
                authentication,
                started,
                errors,
            )
        except (ProviderError, ConfigurationError) as exc:
            errors.append(safe_error(exc))
            return self._persist(
                endpoint.config_digest,
                model.config_digest,
                endpoint.provider_id,
                model.model_id,
                capabilities,
                authentication,
                started,
                errors,
            )

        await self._probe_structured(provider, model, capabilities, errors)
        await self._probe_developer(provider, model, capabilities, errors)
        await self._probe_tools(provider, model, capabilities, errors)
        await self._probe_reasoning(provider, model, capabilities, errors)
        await self._probe_streaming(provider, model.remote_model, capabilities, errors)
        await self._probe_vision(provider, model, capabilities, errors)
        if probe_long_context:
            await self._probe_long_context(provider, model.remote_model, capabilities, errors)
        return self._persist(
            endpoint.config_digest,
            model.config_digest,
            endpoint.provider_id,
            model.model_id,
            capabilities,
            authentication,
            started,
            errors,
            usage=provider.get_usage(),
        )

    @staticmethod
    def _basic_request(remote_model: str) -> GenerationRequest:
        return GenerationRequest(
            model=remote_model,
            messages=[
                ModelMessage(role="system", content="Reply briefly."),
                ModelMessage(role="user", content="Reply OK."),
            ],
            max_output_tokens=16,
            temperature=0,
        )

    @staticmethod
    async def _probe_structured(
        provider: Any,
        model: Any,
        capabilities: dict[ModelCapability, CapabilityEvidence],
        errors: list[str],
    ) -> None:
        try:
            result = await provider.structured_generate(
                GenerationRequest(
                    model=model.remote_model,
                    messages=[
                        ModelMessage(
                            role="user",
                            content='Return JSON exactly matching {"ok":true,"value":1}.',
                        )
                    ],
                    max_output_tokens=256,
                    temperature=0,
                    reasoning_effort=ReasoningEffort.LOW,
                ),
                _ProbeAnswer,
            )
            capabilities[ModelCapability.STRUCTURED_OUTPUT] = _supported(
                "structured response passed Pydantic validation"
            )
            mode = result.response.structured_output_mode
            if mode == StructuredOutputStrategy.NATIVE_JSON_SCHEMA.value:
                capabilities[ModelCapability.JSON_SCHEMA] = _supported(
                    "native JSON Schema response passed"
                )
            elif mode == StructuredOutputStrategy.JSON_MODE.value:
                capabilities[ModelCapability.JSON_MODE] = _supported("native JSON mode passed")
                capabilities[ModelCapability.JSON_SCHEMA] = _unsupported(
                    "JSON mode is not JSON Schema enforcement"
                )
            elif mode == StructuredOutputStrategy.PROMPT_JSON_FALLBACK.value:
                capabilities[ModelCapability.STRUCTURED_OUTPUT] = CapabilityEvidence(
                    status=CapabilityStatus.PARTIAL,
                    source=CapabilitySource.PROBED,
                    detail="prompt-constrained JSON passed; no native schema claim",
                )
                capabilities[ModelCapability.JSON_SCHEMA] = _unsupported(
                    "prompt fallback is not native JSON Schema"
                )
        except ProviderTimeoutError as exc:
            errors.append(safe_error(exc))
        except (ProviderError, ValueError) as exc:
            capabilities[ModelCapability.STRUCTURED_OUTPUT] = _unsupported(
                "structured-output probe failed"
            )
            if model.structured_output_strategy is StructuredOutputStrategy.NATIVE_JSON_SCHEMA:
                capabilities[ModelCapability.JSON_SCHEMA] = _unsupported("JSON Schema probe failed")
            elif model.structured_output_strategy is StructuredOutputStrategy.JSON_MODE:
                capabilities[ModelCapability.JSON_MODE] = _unsupported("JSON mode probe failed")
            errors.append(safe_error(exc))

    @classmethod
    async def _probe_developer(
        cls,
        provider: Any,
        model: Any,
        capabilities: dict[ModelCapability, CapabilityEvidence],
        errors: list[str],
    ) -> None:
        declared = model.declared_capabilities.get(ModelCapability.DEVELOPER_ROLE)
        if declared is None or declared.status is not CapabilityStatus.SUPPORTED:
            return
        try:
            await provider.generate(
                GenerationRequest(
                    model=model.remote_model,
                    messages=[
                        ModelMessage(role="developer", content="Reply briefly."),
                        ModelMessage(role="user", content="Reply OK."),
                    ],
                    max_output_tokens=8,
                )
            )
            capabilities[ModelCapability.DEVELOPER_ROLE] = _supported(
                "developer-role request succeeded"
            )
        except ProviderTimeoutError as exc:
            errors.append(safe_error(exc))
        except ProviderError as exc:
            capabilities[ModelCapability.DEVELOPER_ROLE] = _unsupported(
                "developer-role request failed"
            )
            errors.append(safe_error(exc))

    @staticmethod
    async def _probe_tools(
        provider: Any,
        model: Any,
        capabilities: dict[ModelCapability, CapabilityEvidence],
        errors: list[str],
    ) -> None:
        declared = model.declared_capabilities.get(ModelCapability.TOOLS)
        if declared is None or declared.status is not CapabilityStatus.SUPPORTED:
            return
        try:
            response = await provider.generate(
                GenerationRequest(
                    model=model.remote_model,
                    messages=[
                        ModelMessage(role="user", content="Call dummy_test_tool with value 1.")
                    ],
                    max_output_tokens=16,
                    tools=[
                        ToolDefinition(
                            function=ToolFunction(
                                name="dummy_test_tool",
                                description="A no-side-effect compatibility marker.",
                                parameters={
                                    "type": "object",
                                    "properties": {"value": {"type": "integer"}},
                                    "required": ["value"],
                                },
                            )
                        )
                    ],
                )
            )
            capabilities[ModelCapability.TOOLS] = (
                _supported("exact inert dummy tool call was returned without execution")
                if response.tool_call_count == 1 and response.tool_call_names == ["dummy_test_tool"]
                else _unsupported("provider returned a missing, unexpected, or malformed tool call")
            )
        except ProviderTimeoutError as exc:
            errors.append(safe_error(exc))
        except ProviderError as exc:
            capabilities[ModelCapability.TOOLS] = _unsupported("tool-call request failed")
            errors.append(safe_error(exc))

    @staticmethod
    async def _probe_reasoning(
        provider: Any,
        model: Any,
        capabilities: dict[ModelCapability, CapabilityEvidence],
        errors: list[str],
    ) -> None:
        declared = model.declared_capabilities.get(ModelCapability.REASONING_CONTROL)
        if declared is None or declared.status is not CapabilityStatus.SUPPORTED:
            return
        try:
            await provider.generate(
                GenerationRequest(
                    model=model.remote_model,
                    messages=[ModelMessage(role="user", content="Reply OK.")],
                    max_output_tokens=8,
                    reasoning_effort=ReasoningEffort.LOW,
                )
            )
            capabilities[ModelCapability.REASONING_CONTROL] = _supported(
                "mapped reasoning parameter was accepted"
            )
        except ProviderTimeoutError as exc:
            errors.append(safe_error(exc))
        except ProviderError as exc:
            capabilities[ModelCapability.REASONING_CONTROL] = _unsupported(
                "reasoning-control request failed"
            )
            errors.append(safe_error(exc))

    @classmethod
    async def _probe_streaming(
        cls,
        provider: Any,
        remote_model: str,
        capabilities: dict[ModelCapability, CapabilityEvidence],
        errors: list[str],
    ) -> None:
        try:
            chunks = [chunk async for chunk in provider.stream(cls._basic_request(remote_model))]
            capabilities[ModelCapability.STREAMING] = (
                _supported("stream returned text deltas")
                if chunks
                else _unsupported("stream returned no text deltas")
            )
        except ProviderTimeoutError as exc:
            errors.append(safe_error(exc))
        except ProviderError as exc:
            capabilities[ModelCapability.STREAMING] = _unsupported("streaming request failed")
            errors.append(safe_error(exc))

    @staticmethod
    async def _probe_vision(
        provider: Any,
        model: Any,
        capabilities: dict[ModelCapability, CapabilityEvidence],
        errors: list[str],
    ) -> None:
        declared = model.declared_capabilities.get(ModelCapability.VISION)
        if declared is None or declared.status is not CapabilityStatus.SUPPORTED:
            return
        try:
            await provider.generate(
                GenerationRequest(
                    model=model.remote_model,
                    messages=[
                        ModelMessage(
                            role="user",
                            content="Reply OK after inspecting this one-pixel image.",
                            media=[
                                MediaPart(
                                    mime_type="image/png",
                                    data_base64=_ONE_PIXEL_PNG,
                                    source_id="capability-probe",
                                )
                            ],
                        )
                    ],
                    max_output_tokens=8,
                )
            )
            capabilities[ModelCapability.VISION] = _supported("minimal vision request succeeded")
        except ProviderTimeoutError as exc:
            errors.append(safe_error(exc))
        except ProviderError as exc:
            capabilities[ModelCapability.VISION] = _unsupported("vision request failed")
            errors.append(safe_error(exc))

    @staticmethod
    async def _probe_long_context(
        provider: Any,
        remote_model: str,
        capabilities: dict[ModelCapability, CapabilityEvidence],
        errors: list[str],
    ) -> None:
        """Opt-in, costly behavioral check; never infer capacity from a model label."""
        marker = uuid4().hex
        filler = "\n".join(
            f"Record {index:05d}: "
            + " ".join(f"{(index * 7919 + offset * 104729) % 999983:06d}" for offset in range(8))
            for index in range(1800)
        )
        try:
            response = await provider.generate(
                GenerationRequest(
                    model=remote_model,
                    messages=[
                        ModelMessage(
                            role="user",
                            content=(
                                f"The private marker is {marker}.\n"
                                "Read the records, then return only the private marker.\n"
                                f"{filler}\nWhat is the private marker? Return only its value."
                            ),
                        )
                    ],
                    max_output_tokens=512,
                    temperature=0,
                )
            )
            actual = response.content.strip()
            input_tokens = response.usage.input_tokens or 0
            capabilities[ModelCapability.LONG_CONTEXT] = (
                _supported(f"exact early-marker recall with {input_tokens} reported input tokens")
                if actual == marker and input_tokens >= 8192
                else _unsupported(
                    "long-context marker recall or minimum reported input-token evidence failed"
                )
            )
        except ProviderTimeoutError as exc:
            errors.append(safe_error(exc))
        except ProviderError as exc:
            capabilities[ModelCapability.LONG_CONTEXT] = _unsupported("long-context request failed")
            errors.append(safe_error(exc))

    def _persist(
        self,
        provider_digest: str,
        model_digest: str,
        provider_id: str,
        model_id: str,
        capabilities: dict[ModelCapability, CapabilityEvidence],
        authentication: ProbeAuthenticationStatus,
        started: float,
        errors: list[str],
        *,
        usage: Any | None = None,
    ) -> CapabilityProbeResult:
        result = CapabilityProbeResult(
            provider_id=provider_id,
            model_id=model_id,
            provider_config_digest=provider_digest,
            model_config_digest=model_digest,
            capabilities=capabilities,
            authentication_status=authentication,
            latency_ms=int((time.perf_counter() - started) * 1000),
            usage=usage or {},
            errors=errors[:50],
        )
        return self._configurations.record_probe(result)


def _supported(detail: str) -> CapabilityEvidence:
    return CapabilityEvidence(
        status=CapabilityStatus.SUPPORTED,
        source=CapabilitySource.PROBED,
        detail=detail,
    )


def _unsupported(detail: str) -> CapabilityEvidence:
    return CapabilityEvidence(
        status=CapabilityStatus.UNSUPPORTED,
        source=CapabilitySource.PROBED,
        detail=detail,
    )
