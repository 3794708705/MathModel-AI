from __future__ import annotations

from collections.abc import Collection
from datetime import UTC, datetime

from mathmodel_ai.core.config import ModelTargetSettings, Settings
from mathmodel_ai.core.types import ProviderName, ReasoningEffort
from mathmodel_ai.providers.repository import ProviderModelRegistry
from mathmodel_ai.routing.schemas import (
    EscalationLevel,
    RouteAction,
    RouteDecision,
    TaskProfile,
    TaskType,
)
from mathmodel_ai.schemas.provider_registry import (
    CapabilityEvidence,
    CapabilityProbeResult,
    CapabilitySource,
    CapabilityStatus,
    ModelCapability,
    ModelProfile,
    ProbeAuthenticationStatus,
    ProviderEndpoint,
    ProviderHealthStatus,
    QualityTier,
    StructuredOutputStrategy,
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

_TIER_RANK = {
    QualityTier.ROUTINE: 1,
    QualityTier.BALANCED: 2,
    QualityTier.FLAGSHIP_HIGH: 3,
    QualityTier.FLAGSHIP_XHIGH: 4,
    QualityTier.FLAGSHIP_MAX: 5,
}


class TaskProfiler:
    """Build an explicit profile; it does not infer requirements from keywords."""

    def profile(self, task_type: TaskType, **requirements: int) -> TaskProfile:
        return TaskProfile(task_type=task_type, **requirements)


class ModelRouter:
    def __init__(
        self,
        settings: Settings,
        available_providers: Collection[str | ProviderName] | None = None,
        *,
        registry: ProviderModelRegistry | None = None,
    ) -> None:
        self._settings = settings
        self._available = (
            frozenset(str(item) for item in available_providers)
            if available_providers is not None
            else frozenset(item.value for item in ProviderName)
        )
        self._registry = registry

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

    def route(self, profile: TaskProfile, *, agent_name: str | None = None) -> RouteDecision:
        level = self._base_level(profile)
        if level is EscalationLevel.HUMAN_REVIEW:
            return RouteDecision(
                level=level,
                action=RouteAction.HUMAN_REVIEW,
                task_profile_digest=profile.requirements_digest,
                reason="automatic escalation exhausted; human judgment is required",
            )
        if level is EscalationLevel.MULTI_MODEL_REVIEW:
            return RouteDecision(
                level=level,
                action=RouteAction.MULTI_MODEL_REVIEW,
                task_profile_digest=profile.requirements_digest,
                reason="retry escalation requires independent multi-model review",
            )
        try:
            registry_models = self._registry_models()
            configured_default_model_id = self._configured_default_model_id()
        except Exception:
            return RouteDecision(
                level=EscalationLevel.HUMAN_REVIEW,
                action=RouteAction.HUMAN_REVIEW,
                task_profile_digest=profile.requirements_digest,
                reason="REGISTRY_INTEGRITY_FAILURE: model registry could not be validated",
            )
        if configured_default_model_id is not None and not registry_models:
            return RouteDecision(
                level=EscalationLevel.HUMAN_REVIEW,
                action=RouteAction.HUMAN_REVIEW,
                recommended_model=configured_default_model_id,
                task_profile_digest=profile.requirements_digest,
                reason="configured default_model_id is not present in the model registry",
            )
        use_registry = bool(registry_models) and (
            configured_default_model_id is not None
            or profile.preferred_model_id is not None
            or agent_name is not None
        )
        if use_registry:
            return self._route_registry(
                profile,
                level,
                registry_models,
                agent_name=agent_name,
                default_model_id=configured_default_model_id,
            )
        return self._route_legacy(profile, level)

    def _registry_models(self) -> list[ModelProfile]:
        if self._registry is None:
            return []
        return self._registry.list_models()

    def _configured_default_model_id(self) -> str | None:
        if self._registry is not None:
            persisted = self._registry.get_default_model_id()
            if persisted is not None:
                return persisted
        return self._settings.default_model_id

    def _route_registry(
        self,
        profile: TaskProfile,
        level: EscalationLevel,
        models: list[ModelProfile],
        *,
        agent_name: str | None,
        default_model_id: str | None,
    ) -> RouteDecision:
        if self._registry is None:  # pragma: no cover - guarded by caller
            raise RuntimeError("registry routing requires a registry")
        try:
            policy = self._registry.get_agent_route(agent_name) if agent_name else None
        except Exception:
            return RouteDecision(
                level=EscalationLevel.HUMAN_REVIEW,
                action=RouteAction.HUMAN_REVIEW,
                task_profile_digest=profile.requirements_digest,
                reason="REGISTRY_INTEGRITY_FAILURE: agent routing policy could not be validated",
            )
        preferred = profile.preferred_model_id or (
            policy.primary_model_id if policy is not None else default_model_id
        )
        ordered_ids: list[str] = []
        if preferred is not None:
            ordered_ids.append(preferred)
        if policy is not None:
            ordered_ids.extend(policy.fallback_model_ids)
        remaining = (
            []
            if policy is not None
            else sorted(
                (item for item in models if item.model_id not in ordered_ids),
                key=lambda item: self._score(item, profile, level),
                reverse=True,
            )
        )
        by_id = {item.model_id: item for item in models}
        if preferred is not None and preferred not in by_id:
            return RouteDecision(
                level=EscalationLevel.HUMAN_REVIEW,
                action=RouteAction.HUMAN_REVIEW,
                recommended_model=preferred,
                task_profile_digest=profile.requirements_digest,
                reason="NO_ELIGIBLE_MODEL: configured preferred model is absent from registry",
            )
        candidates = [by_id[item] for item in ordered_ids if item in by_id] + remaining
        rejected: dict[str, list[str]] = {}
        selected: tuple[ModelProfile, ProviderEndpoint, CapabilityProbeResult] | None = None
        for model in candidates:
            try:
                endpoint = self._registry.get_provider(model.provider_id)
                reasons, probe = self._hard_rejections(model, endpoint, profile)
            except Exception:
                rejected[model.model_id] = ["registry evidence failed integrity validation"]
                continue
            if reasons:
                rejected[model.model_id] = reasons
                continue
            if probe is None:  # pragma: no cover - guarded by hard rejections
                rejected[model.model_id] = ["current capability probe is unavailable"]
                continue
            selected = (model, endpoint, probe)
            break
        if selected is None:
            preferred_reason = (
                f" preferred model {preferred!r} was ineligible;" if preferred is not None else ""
            )
            return RouteDecision(
                level=EscalationLevel.HUMAN_REVIEW,
                action=RouteAction.HUMAN_REVIEW,
                recommended_model=preferred,
                rejected_models=rejected,
                task_profile_digest=profile.requirements_digest,
                reason=(
                    "NO_ELIGIBLE_MODEL: no registered model satisfies hard routing requirements;"
                    f"{preferred_reason}"
                ),
            )
        model, endpoint, probe = selected
        fallback_used = preferred is not None and model.model_id != preferred
        fallback_reason = None
        if fallback_used:
            if not profile.allow_model_fallback:
                return RouteDecision(
                    level=EscalationLevel.HUMAN_REVIEW,
                    action=RouteAction.HUMAN_REVIEW,
                    recommended_model=preferred,
                    rejected_models=rejected,
                    task_profile_digest=profile.requirements_digest,
                    reason="preferred model is ineligible and model fallback is disabled",
                )
            fallback_reason = "; ".join(
                rejected.get(preferred or "", ["preferred model unavailable"])
            )
        requested_reasoning = _cap_reasoning_effort(
            _reasoning_for_level(level), profile.maximum_reasoning_effort
        )
        reasoning_evidence = probe.capabilities.get(ModelCapability.REASONING_CONTROL)
        reasoning_supported = (
            reasoning_evidence is not None
            and reasoning_evidence.source is CapabilitySource.PROBED
            and reasoning_evidence.status is CapabilityStatus.SUPPORTED
        )
        selected_reasoning = requested_reasoning if reasoning_supported else None
        effective_reasoning = (
            model.reasoning_mapping.get(requested_reasoning.value.upper())
            if reasoning_supported and requested_reasoning is not ReasoningEffort.NONE
            else None
        )
        return RouteDecision(
            level=level,
            action=RouteAction.EXECUTE,
            recommended_provider=endpoint.provider_id,
            recommended_model=preferred or model.model_id,
            recommended_reasoning=requested_reasoning,
            selected_provider=endpoint.provider_id,
            selected_model=model.remote_model,
            selected_model_id=model.model_id,
            provider_config_digest=endpoint.config_digest,
            model_config_digest=model.config_digest,
            capability_probe_id=probe.probe_id,
            capability_probe_digest=probe.probe_digest,
            task_profile_digest=profile.requirements_digest,
            protocol=endpoint.protocol.value,
            endpoint_trust=endpoint.trust_level,
            structured_output_mode=model.structured_output_strategy,
            selected_reasoning=selected_reasoning,
            reasoning_effective=effective_reasoning,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
            rejected_models=rejected,
            reason=(
                "selected by capability hard filters and configured model policy"
                + ("; recorded real-model fallback" if fallback_used else "")
            ),
        )

    def _hard_rejections(
        self,
        model: ModelProfile,
        endpoint: ProviderEndpoint,
        profile: TaskProfile,
    ) -> tuple[list[str], CapabilityProbeResult | None]:
        registry = self._registry
        if registry is None:  # pragma: no cover - only called from registry routing
            raise RuntimeError("registry routing requires a registry")
        reasons: list[str] = []
        if model.provider_id == ProviderName.MOCK.value:
            reasons.append("Mock is not an eligible registry route")
        if not endpoint.enabled:
            reasons.append("provider is disabled")
        if not model.enabled:
            reasons.append("model is disabled")
        try:
            credential_configured = registry.credential_configured(endpoint.provider_id)
        except Exception:
            credential_configured = False
            reasons.append("provider credential lookup failed")
        if not credential_configured:
            reasons.append("provider credential is not configured")
        if endpoint.health_status is not ProviderHealthStatus.READY:
            reasons.append(f"provider health is {endpoint.health_status.value}")
        if endpoint.cooldown_until is not None and endpoint.cooldown_until > datetime.now(UTC):
            reasons.append("provider circuit breaker is in cooldown")
        # Quality tier is a ranking preference, not proof that a model can or cannot do this task.
        required = set(profile.required_capabilities)
        required.add(ModelCapability.STRUCTURED_OUTPUT)
        if profile.multimodal_requirement > 0:
            required.add(ModelCapability.VISION)
        if profile.requires_json_schema:
            required.add(ModelCapability.JSON_SCHEMA)
        if profile.requires_native_tools:
            required.add(ModelCapability.TOOLS)
        if profile.requires_reasoning_control:
            required.add(ModelCapability.REASONING_CONTROL)
        if profile.minimum_context_tokens is not None or profile.long_context_requirement > 0:
            required.add(ModelCapability.LONG_CONTEXT)
        current_probe = registry.latest_probe(model.model_id, current_only=True)
        if current_probe is None:
            reasons.append("current capability probe is missing or stale")
        elif current_probe.authentication_status is not ProbeAuthenticationStatus.PASS:
            reasons.append(
                "current capability probe authentication is "
                f"{current_probe.authentication_status.value}"
            )
        for capability in sorted(required, key=lambda item: item.value):
            evidence = (
                current_probe.capabilities.get(capability, CapabilityEvidence())
                if current_probe is not None
                else CapabilityEvidence()
            )
            if (
                evidence.source is not CapabilitySource.PROBED
                or evidence.status is not CapabilityStatus.SUPPORTED
            ):
                if (
                    capability is ModelCapability.STRUCTURED_OUTPUT
                    and profile.allows_prompt_json_fallback
                    and model.structured_output_strategy
                    is StructuredOutputStrategy.PROMPT_JSON_FALLBACK
                    and current_probe is not None
                    and _probe_supports(current_probe, ModelCapability.TEXT)
                    and evidence.source is CapabilitySource.PROBED
                    and evidence.status is CapabilityStatus.PARTIAL
                ):
                    continue
                reasons.append(f"required capability {capability.value} is not supported")
        if (
            profile.minimum_context_tokens is not None
            and (model.context_window or 0) < profile.minimum_context_tokens
        ):
            reasons.append("model context window is insufficient")
        if (
            profile.allowed_endpoint_trust is not None
            and endpoint.trust_level not in profile.allowed_endpoint_trust
        ):
            reasons.append("endpoint trust level is disallowed")
        return reasons, current_probe

    @staticmethod
    def _score(
        model: ModelProfile, profile: TaskProfile, level: EscalationLevel
    ) -> tuple[float, float, float]:
        tier = float(_TIER_RANK[model.quality_tier])
        quality_fit = -abs(tier - min(level.value, 5))
        input_cost = model.pricing.input_per_million
        cost = -(input_cost if input_cost is not None else 1_000_000)
        stable_id = -float(sum(ord(character) for character in model.model_id))
        return (quality_fit, cost if profile.cost_sensitivity > 0 else 0.0, stable_id)

    def _route_legacy(self, profile: TaskProfile, level: EscalationLevel) -> RouteDecision:
        target = self._target(level)
        if profile.task_type is TaskType.DATA_UNDERSTANDING and (
            profile.multimodal_requirement > 0 or profile.long_context_requirement >= 4
        ):
            level = max(level, EscalationLevel.FLAGSHIP_XHIGH)
            target = self._settings.model_catalog.multimodal
        fallback_used = target.provider.value not in self._available
        selected_provider: ProviderName = target.provider
        selected_model = target.model
        requested_reasoning = _cap_reasoning_effort(
            target.reasoning, profile.maximum_reasoning_effort
        )
        selected_reasoning: ReasoningEffort | None = requested_reasoning
        reason = f"selected {level.name} from task minimum, explicit requirements, and retries"
        fallback_reason = None
        if fallback_used:
            if self._settings.default_provider.value not in self._available:
                return RouteDecision(
                    level=EscalationLevel.HUMAN_REVIEW,
                    action=RouteAction.HUMAN_REVIEW,
                    recommended_provider=target.provider,
                    recommended_model=target.model,
                    recommended_reasoning=requested_reasoning,
                    task_profile_digest=profile.requirements_digest,
                    reason="recommended and fallback providers are unavailable",
                    legacy_config_used=True,
                )
            selected_provider = self._settings.default_provider
            selected_model = self._settings.default_provider_model
            selected_reasoning = None
            fallback_reason = "configured catalog provider unavailable"
            reason += "; configured provider unavailable, using explicit fallback"
        return RouteDecision(
            level=level,
            action=RouteAction.EXECUTE,
            recommended_provider=target.provider,
            recommended_model=target.model,
            recommended_reasoning=requested_reasoning,
            selected_provider=selected_provider,
            selected_model=selected_model,
            selected_reasoning=selected_reasoning,
            task_profile_digest=profile.requirements_digest,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
            legacy_config_used=True,
            reason=reason,
        )


def _reasoning_for_level(level: EscalationLevel) -> ReasoningEffort:
    return {
        EscalationLevel.FAST: ReasoningEffort.LOW,
        EscalationLevel.BALANCED: ReasoningEffort.MEDIUM,
        EscalationLevel.FLAGSHIP_HIGH: ReasoningEffort.HIGH,
        EscalationLevel.FLAGSHIP_XHIGH: ReasoningEffort.XHIGH,
        EscalationLevel.FLAGSHIP_MAX: ReasoningEffort.MAX,
    }[min(level, EscalationLevel.FLAGSHIP_MAX)]


def _cap_reasoning_effort(
    requested: ReasoningEffort,
    maximum: ReasoningEffort | None,
) -> ReasoningEffort:
    if maximum is None:
        return requested
    order = {
        ReasoningEffort.NONE: 0,
        ReasoningEffort.LOW: 1,
        ReasoningEffort.MEDIUM: 2,
        ReasoningEffort.HIGH: 3,
        ReasoningEffort.XHIGH: 4,
        ReasoningEffort.MAX: 5,
    }
    return requested if order[requested] <= order[maximum] else maximum


def _probe_supports(probe: CapabilityProbeResult, capability: ModelCapability) -> bool:
    evidence = probe.capabilities.get(capability)
    return bool(
        evidence is not None
        and evidence.source is CapabilitySource.PROBED
        and evidence.status is CapabilityStatus.SUPPORTED
    )
