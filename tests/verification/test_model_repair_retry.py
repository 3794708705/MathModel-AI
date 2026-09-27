from mathmodel_ai.agents.model_repair import ModelRepairAgent
from mathmodel_ai.reasoning.prompts import PromptRegistry
from mathmodel_ai.schemas.verification import ModelRepairInput


def test_repair_retry_receives_exact_structural_feedback() -> None:
    agent = ModelRepairAgent(router=None, providers=None, prompts=PromptRegistry())
    original = ModelRepairInput.model_construct(
        assigned_version=2,
        repair_cycle=1,
        current_model=None,
        red_team_report=None,
        validation=None,
        sensitivity=None,
        robustness=None,
        user_guidance=["retain official data bindings"],
    )
    error = "revised_model: variable index_sets must reference declared sets"

    assert agent.prepare_attempt_input(original, None, ()) is original
    retry = agent.prepare_attempt_input(original, None, (error,))

    assert retry.user_guidance[0] == "retain official data bindings"
    assert error in retry.user_guidance[1]
    assert original.user_guidance == ["retain official data bindings"]


def test_repair_prompt_preserves_scientific_obligations_when_schema_fails() -> None:
    prompt = PromptRegistry().get("model_repair_agent")
    assert "declared in revised_model.sets" in prompt.system
    assert "valid value (finite if numeric) or a valid data_binding" in prompt.system
    assert "preserve the critical" in prompt.system
