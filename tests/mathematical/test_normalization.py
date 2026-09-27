from mathmodel_ai.mathematical.normalization import canonicalize_scalar_optimization_family
from mathmodel_ai.mathematical.quality_gates import model_quality_gate
from mathmodel_ai.schemas.mathematical import VariableDomain
from mathmodel_ai.schemas.model_selection import ModelFamily
from mathmodel_ai.schemas.quality import QualityGateStatus
from tests.mathematical.helpers import lp_model, selected_state


def test_scalar_continuous_objective_family_is_normalized_without_changing_math() -> None:
    state = selected_state()
    base = lp_model(
        project_id=state.project_id,
        problem_id=state.problem_id,
        source_selected_model_id="CAND-lp",
    )
    draft = base.model_copy(update={"model_family": ModelFamily.REGRESSION})
    normalized = canonicalize_scalar_optimization_family(draft)
    assert normalized.model_family is ModelFamily.NONLINEAR_PROGRAMMING
    assert normalized.objective == draft.objective
    assert normalized.decision_variables == draft.decision_variables
    assert normalized.equations == draft.equations
    assert normalized.parameters == draft.parameters
    assert any("normalized from regression" in item for item in normalized.limitations)
    assert model_quality_gate(normalized, state).status is QualityGateStatus.PASS
    assert canonicalize_scalar_optimization_family(normalized) == normalized


def test_non_scalar_or_discrete_models_are_not_silently_reclassified() -> None:
    base = lp_model().model_copy(update={"model_family": ModelFamily.MODEL_CHAIN})
    discrete = base.decision_variables[0].model_copy(update={"domain": VariableDomain.INTEGER})
    assert (
        canonicalize_scalar_optimization_family(
            base.model_copy(update={"decision_variables": [discrete, base.decision_variables[1]]})
        ).model_family
        is ModelFamily.MODEL_CHAIN
    )
    indexed = base.decision_variables[0].model_copy(update={"index_sets": ["SET-I"]})
    assert (
        canonicalize_scalar_optimization_family(
            base.model_copy(update={"decision_variables": [indexed, base.decision_variables[1]]})
        ).model_family
        is ModelFamily.MODEL_CHAIN
    )
