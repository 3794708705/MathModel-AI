from dataclasses import dataclass

import pytest

from mathmodel_ai.core.config import Settings
from mathmodel_ai.core.types import Environment, ProviderName, ReasoningEffort
from mathmodel_ai.db.base import Base
from mathmodel_ai.db.session import create_database_engine, create_session_factory
from mathmodel_ai.providers.repository import ProviderModelRegistry
from mathmodel_ai.providers.secrets import EnvironmentSecretResolver
from mathmodel_ai.providers.security import EndpointSecurityPolicy
from mathmodel_ai.routing.router import ModelRouter
from mathmodel_ai.routing.schemas import EscalationLevel, RouteAction, TaskProfile, TaskType
from mathmodel_ai.schemas.provider_registry import (
    AgentRoutePolicy,
    CapabilityEvidence,
    CapabilityProbeResult,
    CapabilitySource,
    CapabilityStatus,
    EndpointTrustLevel,
    ModelCapability,
    ModelProfile,
    ProbeAuthenticationStatus,
    ProviderEndpoint,
    ProviderProtocol,
    QualityTier,
    StructuredOutputStrategy,
)


@dataclass
class RegistryHarness:
    registry: ProviderModelRegistry
    secrets: dict[str, str]

    def add_provider(self, provider_id: str, *, enabled: bool = True) -> ProviderEndpoint:
        credential_name = f"KEY_{provider_id.upper().replace('-', '_')}"
        self.secrets[credential_name] = f"secret-for-{provider_id}"
        return self.registry.create_provider(
            ProviderEndpoint(
                provider_id=provider_id,
                display_name=provider_id,
                protocol=ProviderProtocol.OPENAI_CHAT_COMPLETIONS,
                base_url=f"https://{provider_id}.example.test/v1",
                credential_ref=f"env:{credential_name}",
                enabled=enabled,
                trust_level=EndpointTrustLevel.USER_MANAGED_PROXY,
            )
        )

    def add_model(
        self,
        model_id: str,
        provider_id: str,
        *,
        enabled: bool = True,
        statuses: dict[ModelCapability, CapabilityStatus] | None = None,
        quality_tier: QualityTier = QualityTier.FLAGSHIP_MAX,
    ) -> ModelProfile:
        selected = statuses or {
            ModelCapability.TEXT: CapabilityStatus.SUPPORTED,
            ModelCapability.STRUCTURED_OUTPUT: CapabilityStatus.SUPPORTED,
            ModelCapability.JSON_SCHEMA: CapabilityStatus.SUPPORTED,
        }
        return self.registry.create_model(
            ModelProfile(
                model_id=model_id,
                provider_id=provider_id,
                display_name=model_id,
                remote_model=f"remote-{model_id}",
                enabled=enabled,
                quality_tier=quality_tier,
                declared_capabilities={
                    capability: CapabilityEvidence(
                        status=status,
                        source=CapabilitySource.USER_DECLARED,
                    )
                    for capability, status in selected.items()
                },
                structured_output_strategy=StructuredOutputStrategy.NATIVE_JSON_SCHEMA,
                reasoning_mapping={
                    "LOW": "low",
                    "MEDIUM": "medium",
                    "HIGH": "high",
                    "XHIGH": "xhigh",
                    "MAX": "max",
                },
            )
        )

    def probe(
        self,
        model_id: str,
        *,
        statuses: dict[ModelCapability, CapabilityStatus] | None = None,
    ) -> CapabilityProbeResult:
        model = self.registry.get_model(model_id)
        endpoint = self.registry.get_provider(model.provider_id)
        selected = statuses or {
            ModelCapability.TEXT: CapabilityStatus.SUPPORTED,
            ModelCapability.STRUCTURED_OUTPUT: CapabilityStatus.SUPPORTED,
            ModelCapability.JSON_SCHEMA: CapabilityStatus.SUPPORTED,
        }
        return self.registry.record_probe(
            CapabilityProbeResult(
                provider_id=endpoint.provider_id,
                model_id=model.model_id,
                provider_config_digest=endpoint.config_digest,
                model_config_digest=model.config_digest,
                capabilities={
                    capability: CapabilityEvidence(
                        status=status,
                        source=CapabilitySource.PROBED,
                    )
                    for capability, status in selected.items()
                },
                authentication_status=ProbeAuthenticationStatus.PASS,
                latency_ms=1,
            )
        )


def _harness() -> RegistryHarness:
    settings = Settings(
        environment="test",
        database_url="sqlite+pysqlite:///:memory:",
        default_provider="mock",
    )
    engine = create_database_engine(settings)
    Base.metadata.create_all(engine)
    secrets: dict[str, str] = {}
    registry = ProviderModelRegistry(
        create_session_factory(engine),
        secrets=EnvironmentSecretResolver(secrets),
        security_policy=EndpointSecurityPolicy(
            environment=Environment.TEST,
            resolver=lambda _host, _port: ["93.184.216.34"],
        ),
    )
    return RegistryHarness(registry=registry, secrets=secrets)


def _router(
    harness: RegistryHarness,
    *,
    default_model_id: str | None = None,
    default_provider: ProviderName = ProviderName.MOCK,
) -> ModelRouter:
    return ModelRouter(
        Settings(
            environment="test",
            default_model_id=default_model_id,
            default_provider=default_provider,
        ),
        available_providers={default_provider},
        registry=harness.registry,
    )


def _profile(**changes: object) -> TaskProfile:
    payload: dict[str, object] = {"task_type": TaskType.DOCUMENTATION}
    payload.update(changes)
    return TaskProfile.model_validate(payload)


def test_k_disabled_provider_is_never_selected() -> None:
    harness = _harness()
    harness.add_provider("provider-a", enabled=False)
    harness.add_model("model-a", "provider-a")
    harness.add_provider("provider-b")
    harness.add_model("model-b", "provider-b")
    harness.probe("model-b")

    decision = _router(harness).route(_profile(preferred_model_id="model-a"))
    assert decision.selected_model_id == "model-b"
    assert decision.fallback_used is True
    assert "provider is disabled" in decision.rejected_models["model-a"]


def test_l_disabled_model_is_never_selected() -> None:
    harness = _harness()
    harness.add_provider("provider-main")
    harness.add_model("model-disabled", "provider-main", enabled=False)
    harness.add_model("model-ready", "provider-main")
    harness.probe("model-ready")

    decision = _router(harness).route(_profile(preferred_model_id="model-disabled"))
    assert decision.selected_model_id == "model-ready"
    assert "model is disabled" in decision.rejected_models["model-disabled"]


def test_m_required_capability_hard_filter_selects_supported_model() -> None:
    harness = _harness()
    harness.add_provider("provider-main")
    harness.add_model("model-without-schema", "provider-main")
    harness.add_model("model-with-schema", "provider-main")
    harness.probe(
        "model-without-schema",
        statuses={
            ModelCapability.TEXT: CapabilityStatus.SUPPORTED,
            ModelCapability.STRUCTURED_OUTPUT: CapabilityStatus.PARTIAL,
            ModelCapability.JSON_SCHEMA: CapabilityStatus.UNSUPPORTED,
        },
    )
    harness.probe("model-with-schema")

    decision = _router(harness).route(
        _profile(
            preferred_model_id="model-without-schema",
            requires_json_schema=True,
        )
    )
    assert decision.selected_model_id == "model-with-schema"
    assert "required capability JSON_SCHEMA" in " ".join(
        decision.rejected_models["model-without-schema"]
    )


def test_unprobed_reasoning_control_is_not_sent() -> None:
    harness = _harness()
    harness.add_provider("provider-a")
    harness.add_model("model-a", "provider-a")
    harness.probe("model-a")

    decision = _router(harness, default_model_id="model-a").route(_profile(complexity=5))

    assert decision.action is RouteAction.EXECUTE
    assert decision.selected_reasoning is None


def test_probed_reasoning_control_uses_normalized_max_reasoning() -> None:
    harness = _harness()
    harness.add_provider("provider-a")
    harness.add_model("model-a", "provider-a")
    harness.probe(
        "model-a",
        statuses={
            ModelCapability.TEXT: CapabilityStatus.SUPPORTED,
            ModelCapability.STRUCTURED_OUTPUT: CapabilityStatus.SUPPORTED,
            ModelCapability.JSON_SCHEMA: CapabilityStatus.SUPPORTED,
            ModelCapability.REASONING_CONTROL: CapabilityStatus.SUPPORTED,
        },
    )

    decision = _router(harness, default_model_id="model-a").route(_profile(complexity=5))

    assert decision.action is RouteAction.EXECUTE
    assert decision.selected_reasoning is ReasoningEffort.MAX


def test_registry_reasoning_cap_preserves_quality_eligibility() -> None:
    harness = _harness()
    harness.add_provider("provider-a")
    harness.add_model("model-a", "provider-a")
    harness.probe(
        "model-a",
        statuses={
            ModelCapability.TEXT: CapabilityStatus.SUPPORTED,
            ModelCapability.STRUCTURED_OUTPUT: CapabilityStatus.SUPPORTED,
            ModelCapability.JSON_SCHEMA: CapabilityStatus.SUPPORTED,
            ModelCapability.REASONING_CONTROL: CapabilityStatus.SUPPORTED,
        },
    )

    decision = _router(harness, default_model_id="model-a").route(
        _profile(complexity=5, maximum_reasoning_effort=ReasoningEffort.LOW)
    )

    assert decision.level.value == 5
    assert decision.selected_reasoning is ReasoningEffort.LOW
    assert decision.reasoning_effective == "low"


def test_user_declared_supported_capability_cannot_bypass_probe() -> None:
    harness = _harness()
    harness.add_provider("provider-main")
    harness.add_model("model-unprobed", "provider-main")
    harness.add_model("model-health-seed", "provider-main")
    harness.probe("model-health-seed")

    decision = _router(harness).route(
        _profile(
            preferred_model_id="model-unprobed",
            allow_model_fallback=False,
            requires_json_schema=True,
        )
    )
    assert decision.action is RouteAction.HUMAN_REVIEW
    rejected = " ".join(decision.rejected_models["model-unprobed"])
    assert "not supported" in rejected


def test_n_explicit_preference_is_honored_or_rejected_explicitly() -> None:
    harness = _harness()
    harness.add_provider("provider-main")
    harness.add_model("model-a", "provider-main")
    harness.add_model("model-b", "provider-main")
    harness.probe("model-a")
    harness.probe("model-b")

    selected = _router(harness).route(_profile(preferred_model_id="model-a"))
    assert selected.selected_model_id == "model-a"
    assert selected.fallback_used is False

    harness.registry.set_model_enabled("model-a", False)
    rejected = _router(harness).route(
        _profile(preferred_model_id="model-a", allow_model_fallback=False)
    )
    assert rejected.action is RouteAction.HUMAN_REVIEW
    assert rejected.recommended_model == "model-a"
    assert "fallback is disabled" in rejected.reason


def test_o_real_model_policy_fallback_records_reason_and_never_uses_mock() -> None:
    harness = _harness()
    harness.add_provider("provider-primary")
    harness.add_model("model-primary", "provider-primary")
    harness.add_provider("provider-backup")
    harness.add_model("model-backup", "provider-backup")
    harness.probe("model-backup")
    harness.registry.put_agent_route(
        AgentRoutePolicy(
            agent_name="problem_agent",
            primary_model_id="model-primary",
            fallback_model_ids=["model-backup"],
        )
    )

    decision = _router(harness).route(
        _profile(task_type=TaskType.PROBLEM_UNDERSTANDING),
        agent_name="problem_agent",
    )
    assert decision.selected_model_id == "model-backup"
    assert decision.selected_provider == "provider-backup"
    assert decision.fallback_used is True
    assert decision.fallback_reason
    assert decision.selected_provider != "mock"


def test_p_legacy_provider_and_model_configuration_remains_operational() -> None:
    decision = ModelRouter(
        Settings(
            environment="test",
            default_provider="openai",
            default_provider_model="legacy-model",
        ),
        available_providers={ProviderName.OPENAI},
    ).route(_profile())
    assert decision.action is RouteAction.EXECUTE
    assert decision.selected_provider is ProviderName.OPENAI
    assert decision.legacy_config_used is True


def test_q_default_model_id_has_priority_over_legacy_configuration() -> None:
    harness = _harness()
    harness.add_provider("provider-new")
    harness.add_model("model-new", "provider-new")
    harness.probe("model-new")
    decision = _router(
        harness,
        default_model_id="model-new",
        default_provider=ProviderName.MOCK,
    ).route(_profile())
    assert decision.selected_model_id == "model-new"
    assert decision.selected_provider == "provider-new"
    assert decision.legacy_config_used is False


def test_q_missing_new_default_fails_closed_instead_of_using_legacy_mock() -> None:
    harness = _harness()
    decision = _router(
        harness,
        default_model_id="missing-model",
        default_provider=ProviderName.MOCK,
    ).route(_profile())
    assert decision.action is RouteAction.HUMAN_REVIEW
    assert decision.selected_provider is None
    assert "not present" in decision.reason


def test_s_model_change_invalidates_probe_and_blocks_route() -> None:
    harness = _harness()
    harness.add_provider("provider-main")
    before = harness.add_model("model-main", "provider-main")
    probe = harness.probe("model-main")
    assert probe.model_config_digest == before.config_digest
    assert harness.registry.latest_probe("model-main", current_only=True) is not None

    updated = harness.registry.update_model("model-main", {"remote_model": "new-remote-model"})
    assert updated.config_digest != before.config_digest
    assert harness.registry.latest_probe("model-main", current_only=True) is None
    assert harness.registry.latest_probe("model-main", current_only=False) is not None
    decision = _router(harness, default_model_id="model-main").route(_profile())
    assert decision.action is RouteAction.HUMAN_REVIEW
    assert "stale" in " ".join(decision.rejected_models["model-main"])


def test_y_agent_specific_routes_select_distinct_model_profiles() -> None:
    harness = _harness()
    harness.add_provider("provider-agents")
    routes = {
        "problem_agent": ("model-problem", TaskType.PROBLEM_UNDERSTANDING),
        "math_modeler": ("model-math", TaskType.MATHEMATICAL_MODELING),
        "paper_agent": ("model-paper", TaskType.PAPER_IR),
        "red_team": ("model-red-team", TaskType.RED_TEAM),
        "final_jury": ("model-final", TaskType.FINAL_ACCEPTANCE),
    }
    for agent_name, (model_id, _task_type) in routes.items():
        harness.add_model(model_id, "provider-agents")
        harness.probe(model_id)
        harness.registry.put_agent_route(
            AgentRoutePolicy(agent_name=agent_name, primary_model_id=model_id)
        )

    router = _router(harness)
    decisions = {
        agent_name: router.route(_profile(task_type=task_type), agent_name=agent_name)
        for agent_name, (_model_id, task_type) in routes.items()
    }
    assert {
        agent_name: decision.selected_model_id for agent_name, decision in decisions.items()
    } == {agent_name: model_id for agent_name, (model_id, _task) in routes.items()}


def test_runtime_default_and_agent_route_changes_apply_to_next_decision_without_restart() -> None:
    harness = _harness()
    harness.add_provider("provider-runtime")
    for model_id in ("model-a", "model-b"):
        harness.add_model(model_id, "provider-runtime")
        harness.probe(model_id)
    router = _router(harness)

    harness.registry.put_default_model("model-a")
    first_default = router.route(_profile())
    harness.registry.put_default_model("model-b")
    next_default = router.route(_profile())

    harness.registry.put_agent_route(
        AgentRoutePolicy(agent_name="math_modeler", primary_model_id="model-a")
    )
    first_agent = router.route(
        _profile(task_type=TaskType.MATHEMATICAL_MODELING),
        agent_name="math_modeler",
    )
    harness.registry.put_agent_route(
        AgentRoutePolicy(agent_name="math_modeler", primary_model_id="model-b")
    )
    next_agent = router.route(
        _profile(task_type=TaskType.MATHEMATICAL_MODELING),
        agent_name="math_modeler",
    )

    assert first_default.selected_model_id == "model-a"
    assert next_default.selected_model_id == "model-b"
    assert first_agent.selected_model_id == "model-a"
    assert next_agent.selected_model_id == "model-b"
    assert first_agent.selected_model_id == "model-a"  # prior decision remains immutable


def test_registry_model_named_mock_is_never_a_live_fallback() -> None:
    harness = _harness()
    harness.add_provider("mock")
    harness.add_model("fake-live-mock", "mock")
    harness.probe("fake-live-mock")
    decision = _router(harness, default_model_id="fake-live-mock").route(_profile())
    assert decision.action is RouteAction.HUMAN_REVIEW
    assert "Mock is not an eligible" in " ".join(decision.rejected_models["fake-live-mock"])


def test_route_decision_binds_exact_task_and_probe_snapshot() -> None:
    harness = _harness()
    harness.add_provider("provider-main")
    harness.add_model("model-main", "provider-main")
    probe = harness.probe("model-main")
    profile = _profile(preferred_model_id="model-main", requires_json_schema=True)

    decision = _router(harness).route(profile)

    assert decision.action is RouteAction.EXECUTE
    assert decision.task_profile_digest == profile.requirements_digest
    assert decision.capability_probe_id == probe.probe_id
    assert decision.capability_probe_digest == probe.probe_digest


def test_missing_explicit_model_does_not_fall_back_to_an_unrequested_registry_model() -> None:
    harness = _harness()
    harness.add_provider("provider-main")
    harness.add_model("model-ready", "provider-main")
    harness.probe("model-ready")

    decision = _router(harness).route(_profile(preferred_model_id="missing-model"))

    assert decision.action is RouteAction.HUMAN_REVIEW
    assert decision.selected_model_id is None
    assert "NO_ELIGIBLE_MODEL" in decision.reason


def test_registry_routing_is_deterministic_for_equal_candidates() -> None:
    harness = _harness()
    harness.add_provider("provider-main")
    harness.add_model("model-a", "provider-main")
    harness.add_model("model-b", "provider-main")
    harness.probe("model-a")
    harness.probe("model-b")
    router = _router(harness)
    profile = _profile()

    decisions = [router.route(profile).model_dump(mode="json") for _ in range(10)]

    assert all(item == decisions[0] for item in decisions)


def test_deadline_pressure_and_low_cost_never_bypass_hard_capability_filter() -> None:
    harness = _harness()
    harness.add_provider("provider-main")
    harness.add_model("model-cheap", "provider-main")
    harness.add_model("model-capable", "provider-main")
    harness.probe(
        "model-cheap",
        statuses={
            ModelCapability.TEXT: CapabilityStatus.SUPPORTED,
            ModelCapability.STRUCTURED_OUTPUT: CapabilityStatus.PARTIAL,
            ModelCapability.JSON_SCHEMA: CapabilityStatus.UNSUPPORTED,
        },
    )
    harness.probe("model-capable")

    decision = _router(harness).route(
        _profile(
            preferred_model_id="model-cheap",
            requires_json_schema=True,
            deadline_pressure=5,
            cost_sensitivity=5,
        )
    )

    assert decision.selected_model_id == "model-capable"
    assert "JSON_SCHEMA" in " ".join(decision.rejected_models["model-cheap"])


def test_all_unavailable_models_return_explicit_no_eligible_model() -> None:
    harness = _harness()
    harness.add_provider("provider-main", enabled=False)
    harness.add_model("model-main", "provider-main")

    decision = _router(harness, default_model_id="model-main").route(_profile())

    assert decision.action is RouteAction.HUMAN_REVIEW
    assert decision.selected_provider is None
    assert "NO_ELIGIBLE_MODEL" in decision.reason


@pytest.mark.parametrize("tier", list(QualityTier))
def test_red_team_accepts_every_quality_tier_when_live_capabilities_pass(
    tier: QualityTier,
) -> None:
    harness = _harness()
    harness.add_provider("provider-any-tier")
    harness.add_model("model-any-tier", "provider-any-tier", quality_tier=tier)
    harness.probe("model-any-tier")
    router = _router(harness, default_model_id="model-any-tier")
    profile = _profile(
        task_type=TaskType.RED_TEAM,
        complexity=4,
        reasoning_requirement=4,
        math_requirement=4,
        review_requirement=5,
        minimum_level=EscalationLevel.FLAGSHIP_XHIGH,
    )

    decision = router.route(profile, agent_name="red_team_agent")

    assert decision.action is RouteAction.EXECUTE
    assert decision.selected_model_id == "model-any-tier"
    assert decision.level is EscalationLevel.FLAGSHIP_MAX


def test_low_tier_still_cannot_bypass_required_structured_output() -> None:
    harness = _harness()
    harness.add_provider("provider-low")
    harness.add_model("model-low", "provider-low", quality_tier=QualityTier.ROUTINE)
    harness.probe(
        "model-low",
        statuses={
            ModelCapability.TEXT: CapabilityStatus.SUPPORTED,
            ModelCapability.STRUCTURED_OUTPUT: CapabilityStatus.UNSUPPORTED,
        },
    )

    decision = _router(harness, default_model_id="model-low").route(
        _profile(task_type=TaskType.RED_TEAM), agent_name="red_team_agent"
    )

    assert decision.action is RouteAction.HUMAN_REVIEW
    assert "STRUCTURED_OUTPUT" in " ".join(decision.rejected_models["model-low"])
