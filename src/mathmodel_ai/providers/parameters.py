from __future__ import annotations

from mathmodel_ai.core.types import ReasoningEffort
from mathmodel_ai.schemas.provider_registry import (
    ModelCapability,
    ModelProfile,
    StructuredOutputStrategy,
    ToolCallingStrategy,
)


class ProtocolParameterMapper:
    """Emit only parameters supported by current, verified model capability evidence."""

    def reasoning_value(
        self, profile: ModelProfile, requested: ReasoningEffort | None
    ) -> str | None:
        if requested is None or requested is ReasoningEffort.NONE:
            return None
        if not profile.supports(ModelCapability.REASONING_CONTROL, allow_partial=True):
            return None
        return profile.reasoning_mapping.get(
            requested.value.upper()
        ) or profile.reasoning_mapping.get(requested.value)

    @staticmethod
    def allows_tools(profile: ModelProfile) -> bool:
        return (
            profile.tool_calling_strategy is ToolCallingStrategy.NATIVE_TOOLS
            and profile.supports(ModelCapability.TOOLS, allow_partial=True)
        )

    @staticmethod
    def allows_parameter(profile: ModelProfile, parameter: str) -> bool:
        configured = profile.configuration.get("allowed_parameters", [])
        return isinstance(configured, list) and parameter in configured

    @staticmethod
    def structured_strategy(profile: ModelProfile) -> StructuredOutputStrategy:
        strategy = profile.structured_output_strategy
        if strategy is StructuredOutputStrategy.NATIVE_JSON_SCHEMA and not profile.supports(
            ModelCapability.JSON_SCHEMA, allow_partial=True
        ):
            return StructuredOutputStrategy.UNSUPPORTED
        if strategy is StructuredOutputStrategy.JSON_MODE and not profile.supports(
            ModelCapability.JSON_MODE, allow_partial=True
        ):
            return StructuredOutputStrategy.UNSUPPORTED
        if strategy is StructuredOutputStrategy.PROMPT_JSON_FALLBACK and not profile.supports(
            ModelCapability.TEXT, allow_partial=True
        ):
            return StructuredOutputStrategy.UNSUPPORTED
        return strategy
