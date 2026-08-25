from mathmodel_ai.reasoning.deduplication import deduplicate_candidates
from mathmodel_ai.reasoning.quality_gates import explore_quality_gate
from mathmodel_ai.schemas.model_selection import ModelExploration, ModelFamily
from mathmodel_ai.schemas.quality import QualityGateStatus
from tests.reasoning.helpers import candidate_fixture, exploration_fixture


def test_model_chain_candidate_covers_prediction_optimization_and_evaluation() -> None:
    exploration = exploration_fixture()
    required = {"Q1", "Q2", "Q3"}
    gate = explore_quality_gate(exploration, required)
    assert gate.status is QualityGateStatus.PASS
    assert exploration.candidates[0].family is ModelFamily.MODEL_CHAIN
    assert set(exploration.candidates[0].target_subproblems) == required


def test_semantic_deduplication_handles_milp_aliases() -> None:
    variants = [
        candidate_fixture("CAND-a", name="MILP", family=ModelFamily.MILP, targets=["Q2"]),
        candidate_fixture(
            "CAND-b",
            name="Mixed Integer Linear Programming",
            family=ModelFamily.MILP,
            targets=["Q2"],
        ),
        candidate_fixture(
            "CAND-c", name="整数线性规划模型", family=ModelFamily.MILP, targets=["Q2"]
        ),
    ]
    result = deduplicate_candidates(variants)
    assert [item.candidate_id for item in result.candidates] == ["CAND-a"]
    assert result.removed_ids == ["CAND-b", "CAND-c"]


def test_explore_gate_rejects_duplicate_candidates() -> None:
    first = candidate_fixture("CAND-one", name="MILP", family=ModelFamily.MILP, targets=["Q2"])
    duplicate = candidate_fixture(
        "CAND-two",
        name="Mixed Integer Linear Programming",
        family=ModelFamily.MILP,
        targets=["Q2"],
    )
    exploration = ModelExploration(
        candidates=[first, duplicate],
        fewer_than_three_reason="Only one model family is represented in this test.",
        exploration_summary="Deliberately duplicated candidate set.",
    )
    gate = explore_quality_gate(exploration, {"Q2"})
    assert gate.status is QualityGateStatus.RETRY
    assert "semantically_distinct" in gate.errors
