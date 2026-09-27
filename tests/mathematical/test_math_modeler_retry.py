from mathmodel_ai.agents.math_modeler import MathModeler
from mathmodel_ai.reasoning.prompts import PromptRegistry
from mathmodel_ai.schemas.mathematical import MathModelerInput


def test_modeler_retry_retains_distinct_prior_gate_errors() -> None:
    agent = MathModeler(router=None, providers=None, prompts=PromptRegistry())
    original = MathModelerInput.model_construct(user_guidance=["retain official data"])
    errors = (
        "MODEL_GATE_FAIL:data_bindings_reach_core",
        "MODEL_GATE_FAIL:state_relations_sufficient",
    )

    assert agent.prepare_attempt_input(original, None, ()) is original
    retry = agent.prepare_attempt_input(original, None, errors)

    assert retry.user_guidance[0] == "retain official data"
    assert all(error in retry.user_guidance[1] for error in errors)
    assert original.user_guidance == ["retain official data"]
