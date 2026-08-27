from mathmodel_ai.core.config import Settings
from mathmodel_ai.core.types import ProviderName
from mathmodel_ai.routing import EscalationLevel, ModelRouter, RouteAction, TaskProfile, TaskType


def test_mathematical_modeling_starts_at_phase_four_xhigh_minimum() -> None:
    router = ModelRouter(Settings(), available_providers={ProviderName.OPENAI})
    decision = router.route(TaskProfile(task_type=TaskType.MATHEMATICAL_MODELING))
    assert decision.level is EscalationLevel.FLAGSHIP_XHIGH
    assert decision.recommended_model == "gpt-5.6-sol"


def test_sandbox_security_starts_at_xhigh() -> None:
    router = ModelRouter(Settings(), available_providers={ProviderName.OPENAI})
    decision = router.route(TaskProfile(task_type=TaskType.SANDBOX_SECURITY))
    assert decision.level is EscalationLevel.FLAGSHIP_XHIGH


def test_model_exploration_and_jury_start_at_xhigh() -> None:
    router = ModelRouter(Settings(), available_providers={ProviderName.OPENAI})
    for task_type in (TaskType.MODEL_EXPLORATION, TaskType.MODEL_JURY):
        decision = router.route(TaskProfile(task_type=task_type))
        assert decision.level is EscalationLevel.FLAGSHIP_XHIGH


def test_documentation_can_use_fast_level() -> None:
    router = ModelRouter(Settings(), available_providers={ProviderName.OPENAI})
    decision = router.route(TaskProfile(task_type=TaskType.DOCUMENTATION))
    assert decision.level is EscalationLevel.FAST


def test_unavailable_recommended_provider_uses_explicit_mock_fallback() -> None:
    settings = Settings(default_provider="mock", default_provider_model="mock-foundation")
    router = ModelRouter(settings, available_providers={ProviderName.MOCK})
    decision = router.route(TaskProfile(task_type=TaskType.DOCUMENTATION))
    assert decision.fallback_used is True
    assert decision.recommended_provider is ProviderName.OPENAI
    assert decision.selected_provider is ProviderName.MOCK
    assert decision.selected_model == "mock-foundation"


def test_retries_escalate_to_review_then_human() -> None:
    router = ModelRouter(Settings(), available_providers={ProviderName.OPENAI})
    multi = router.route(TaskProfile(task_type=TaskType.DOCUMENTATION, retry_count=5))
    human = router.route(TaskProfile(task_type=TaskType.DOCUMENTATION, retry_count=6))
    assert multi.action is RouteAction.MULTI_MODEL_REVIEW
    assert human.action is RouteAction.HUMAN_REVIEW


def test_multimodal_data_understanding_routes_to_configured_gemini_path() -> None:
    router = ModelRouter(Settings(), available_providers={ProviderName.GOOGLE})
    decision = router.route(
        TaskProfile(
            task_type=TaskType.DATA_UNDERSTANDING,
            multimodal_requirement=4,
        )
    )
    assert decision.level is EscalationLevel.FLAGSHIP_XHIGH
    assert decision.recommended_provider is ProviderName.GOOGLE
    assert decision.selected_provider is ProviderName.GOOGLE
    assert decision.selected_model == "gemini-3.7-flash"
