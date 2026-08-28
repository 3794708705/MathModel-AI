from collections.abc import Collection

from mathmodel_ai.core.config import ModelTargetSettings, Settings
from mathmodel_ai.core.types import ProviderName, ReasoningEffort
from mathmodel_ai.routing.schemas import (
    EscalationLevel,
    RouteAction,
    RouteDecision,
    TaskProfile,
    TaskType,
)

TASK_MINIMUMS: dict[TaskType, EscalationLevel] = {
    TaskType.ARCHITECTURE_REDESIGN: EscalationLevel.FLAGSHIP_XHIGH,
    TaskType.MATHEMATICAL_MODELING: EscalationLevel.FLAGSHIP_XHIGH,
    TaskType.MODEL_REPAIR: EscalationLevel.FLAGSHIP_XHIGH,
    TaskType.SANDBOX_SECURITY: EscalationLevel.FLAGSHIP_XHIGH,
    TaskType.E2E_ROOT_CAUSE: EscalationLevel.FLAGSHIP_XHIGH,
    TaskType.FINAL_ACCEPTANCE: EscalationLevel.FLAGSHIP_XHIGH,
    TaskType.PROBLEM_UNDERSTANDING: EscalationLevel.FLAGSHIP_HIGH,
    TaskType.MODEL_EXPLORATION: EscalationLevel.FLAGSHIP_XHIGH,
    TaskType.MODEL_JURY: EscalationLevel.FLAGSHIP_XHIGH,
    TaskType.DATA_UNDERSTANDING: EscalationLevel.FLAGSHIP_HIGH,
    TaskType.DATABASE_ARCHITECTURE: EscalationLevel.FLAGSHIP_HIGH,
    TaskType.CODE_GENERATION: EscalationLevel.FLAGSHIP_HIGH,
    TaskType.VALIDATION: EscalationLevel.FLAGSHIP_HIGH,
    TaskType.RED_TEAM: EscalationLevel.FLAGSHIP_XHIGH,
    TaskType.PAPER_IR: EscalationLevel.FLAGSHIP_HIGH,
    TaskType.CITATION_VERIFICATION: EscalationLevel.FLAGSHIP_HIGH,
    TaskType.API: EscalationLevel.BALANCED,
    TaskType.CRUD: EscalationLevel.BALANCED,
    TaskType.PARSER: EscalationLevel.BALANCED,
    TaskType.PROVIDER_ADAPTER: EscalationLevel.BALANCED,
    TaskType.TESTING: EscalationLevel.BALANCED,
    TaskType.TABLE: EscalationLevel.BALANCED,
    TaskType.LATEX: EscalationLevel.BALANCED,
    TaskType.REFACTOR: EscalationLevel.BALANCED,
    TaskType.DOCUMENTATION: EscalationLevel.FAST,
    TaskType.FORMATTING: EscalationLevel.FAST,
    TaskType.CLEANUP: EscalationLevel.FAST,
    TaskType.OTHER: EscalationLevel.BALANCED,
}


class TaskProfiler:
    """Build an explicit profile; it does not infer requirements from keywords."""

    def profile(self, task_type: TaskType, **requirements: int) -> TaskProfile:
        return TaskProfile(task_type=task_type, **requirements)


class ModelRouter:
    def __init__(
        self,
        settings: Settings,
        available_providers: Collection[ProviderName] | None = None,
    ) -> None:
        self._settings = settings
        self._available = (
            frozenset(available_providers)
            if available_providers is not None
            else frozenset(ProviderName)
        )

    @staticmethod
    def _demand_level(profile: TaskProfile) -> EscalationLevel:
        demand = max(
            profile.complexity,
            profile.reasoning_requirement,
            profile.math_requirement,
            profile.coding_requirement,
            profile.multimodal_requirement,
            profile.long_context_requirement,
            profile.review_requirement,
        )
        if demand <= 1:
            return EscalationLevel.FAST
        if demand == 2:
            return EscalationLevel.BALANCED
        if demand == 3:
            return EscalationLevel.FLAGSHIP_HIGH
        if demand == 4:
            return EscalationLevel.FLAGSHIP_XHIGH
        return EscalationLevel.FLAGSHIP_MAX

    def _base_level(self, profile: TaskProfile) -> EscalationLevel:
        minimum = profile.minimum_level or TASK_MINIMUMS[profile.task_type]
        level = max(minimum, self._demand_level(profile))
        if profile.security_risk >= 4 or profile.blast_radius >= 5:
            level = max(level, EscalationLevel.FLAGSHIP_XHIGH)
        if profile.deadline_pressure >= 5 and level < EscalationLevel.FLAGSHIP_HIGH:
            level = EscalationLevel.FLAGSHIP_HIGH
        escalated_value = min(level.value + profile.retry_count, EscalationLevel.HUMAN_REVIEW.value)
        return EscalationLevel(escalated_value)

    def _target(self, level: EscalationLevel) -> ModelTargetSettings:
        catalog = self._settings.model_catalog
        mapping = {
            EscalationLevel.FAST: catalog.fast,
            EscalationLevel.BALANCED: catalog.balanced,
            EscalationLevel.FLAGSHIP_HIGH: catalog.flagship_high,
            EscalationLevel.FLAGSHIP_XHIGH: catalog.flagship_xhigh,
            EscalationLevel.FLAGSHIP_MAX: catalog.flagship_max,
        }
        return mapping[min(level, EscalationLevel.FLAGSHIP_MAX)]

    def route(self, profile: TaskProfile) -> RouteDecision:
        level = self._base_level(profile)
        if level is EscalationLevel.HUMAN_REVIEW:
            return RouteDecision(
                level=level,
                action=RouteAction.HUMAN_REVIEW,
                reason="automatic escalation exhausted; human judgment is required",
            )

        target = self._target(level)
        if profile.task_type is TaskType.DATA_UNDERSTANDING and (
            profile.multimodal_requirement > 0 or profile.long_context_requirement >= 4
        ):
            level = max(level, EscalationLevel.FLAGSHIP_XHIGH)
            target = self._settings.model_catalog.multimodal
        action = (
            RouteAction.MULTI_MODEL_REVIEW
            if level is EscalationLevel.MULTI_MODEL_REVIEW
            else RouteAction.EXECUTE
        )
        fallback_used = target.provider not in self._available
        selected_provider = target.provider
        selected_model = target.model
        selected_reasoning: ReasoningEffort | None = target.reasoning
        reason = f"selected {level.name} from task minimum, explicit requirements, and retries"
        if fallback_used:
            if self._settings.default_provider not in self._available:
                return RouteDecision(
                    level=EscalationLevel.HUMAN_REVIEW,
                    action=RouteAction.HUMAN_REVIEW,
                    recommended_provider=target.provider,
                    recommended_model=target.model,
                    recommended_reasoning=target.reasoning,
                    reason="recommended and fallback providers are unavailable",
                )
            selected_provider = self._settings.default_provider
            selected_model = self._settings.default_provider_model
            selected_reasoning = None
            reason += "; configured provider unavailable, using explicit fallback"

        return RouteDecision(
            level=level,
            action=action,
            recommended_provider=target.provider,
            recommended_model=target.model,
            recommended_reasoning=target.reasoning,
            selected_provider=selected_provider,
            selected_model=selected_model,
            selected_reasoning=selected_reasoning,
            fallback_used=fallback_used,
            reason=reason,
        )
