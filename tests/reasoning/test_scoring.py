import pytest

from mathmodel_ai.reasoning.quality_gates import explore_quality_gate, select_quality_gate
from mathmodel_ai.reasoning.scoring import build_model_selection, calculate_score
from mathmodel_ai.schemas.model_selection import (
    CandidateJuryAssessment,
    HardFailure,
    JuryAssessment,
    ModelExploration,
    ModelFamily,
    ModelJuryWeights,
)
from mathmodel_ai.schemas.problem_analysis import DataAvailability
from mathmodel_ai.schemas.quality import QualityGateStatus
from tests.reasoning.helpers import candidate_fixture, candidate_set, jury_fixture


def test_weighted_score_is_calculated_by_python() -> None:
    assessment = CandidateJuryAssessment(
        candidate_id="CAND-a",
        problem_fit=10,
        data_fit=8,
        mathematical_validity=6,
        explainability=4,
        validation_potential=2,
        innovation_potential=0,
        competition_feasibility=10,
        computational_cost=8,
        rationale="Controlled dimensions for exact weighted arithmetic.",
    )
    score = calculate_score(assessment, ModelJuryWeights())
    assert score.weighted_total == 66.0


def test_data_poor_transformer_is_ranked_below_traceable_models() -> None:
    selection = build_model_selection(candidate_set(), jury_fixture(), ModelJuryWeights())
    scores = {score.candidate_id: score for score in selection.score_matrix}
    transformer = scores["CAND-transformer"]
    assert transformer.data_fit == 1
    assert transformer.competition_feasibility == 2
    assert transformer.computational_cost == 1
    assert selection.ranking[-1] == "CAND-transformer"
    assert selection.selected_model_id != "CAND-transformer"


def test_missing_required_data_is_a_deterministic_hard_failure() -> None:
    valid = candidate_fixture("CAND-valid", name="Linear Regression", family=ModelFamily.REGRESSION)
    backup = candidate_fixture("CAND-backup", name="Grey Forecast", family=ModelFamily.GREY_MODEL)
    impossible = candidate_fixture(
        "CAND-missing",
        name="Satellite Feature Transformer",
        family=ModelFamily.NEURAL_NETWORK,
        data_availability=DataAvailability.MISSING,
    )
    assessments = [
        CandidateJuryAssessment(
            candidate_id=item.candidate_id,
            problem_fit=9,
            data_fit=9,
            mathematical_validity=9,
            explainability=9,
            validation_potential=9,
            innovation_potential=9,
            competition_feasibility=9,
            computational_cost=9,
            rationale="Intentionally optimistic LLM assessment.",
        )
        for item in [valid, backup, impossible]
    ]
    selection = build_model_selection(
        [valid, backup, impossible],
        JuryAssessment(
            assessments=assessments,
            overall_rationale="Application hard checks override optimistic scoring.",
            confidence=0.8,
        ),
        ModelJuryWeights(),
    )
    missing_score = next(
        score for score in selection.score_matrix if score.candidate_id == "CAND-missing"
    )
    assert missing_score.eligible is False
    assert HardFailure.DATA_NOT_AVAILABLE in missing_score.hard_failures
    assert selection.selected_model_id != "CAND-missing"
    assert selection.backup_model_id != "CAND-missing"


def test_optimistic_jury_scores_cannot_select_an_incomplete_candidate() -> None:
    candidates = candidate_set()
    candidates[0] = candidates[0].model_copy(update={"target_subproblems": ["Q1"]})
    selection = build_model_selection(
        candidates,
        jury_fixture(),
        ModelJuryWeights(),
        required_subproblem_ids={"Q1", "Q2", "Q3"},
    )
    rejected = next(score for score in selection.score_matrix if score.candidate_id == "CAND-chain")
    assert not rejected.eligible
    assert HardFailure.VIOLATES_PROBLEM_REQUIREMENT in rejected.hard_failures
    assert selection.selected_model_id != "CAND-chain"
    assert selection.backup_model_id != "CAND-chain"


def test_complementary_partial_modules_cannot_be_primary_and_backup() -> None:
    candidates = [
        candidate.model_copy(update={"target_subproblems": [f"Q{index}"]})
        for index, candidate in enumerate(candidate_set(), start=1)
    ]
    exploration = ModelExploration(
        candidates=candidates, exploration_summary="Complementary modules"
    )
    gate = explore_quality_gate(exploration, {"Q1", "Q2", "Q3"})
    assert gate.checks["all_subproblems_covered"]
    assert not gate.checks["two_end_to_end_candidates"]
    assert gate.status is QualityGateStatus.RETRY
    with pytest.raises(ValueError, match="at least two eligible"):
        build_model_selection(candidates, jury_fixture(), ModelJuryWeights())


def test_select_gate_rechecks_coverage_despite_unchanged_eligible_scores() -> None:
    candidates = candidate_set()
    selection = build_model_selection(candidates, jury_fixture(), ModelJuryWeights())
    modified = [
        candidate.model_copy(update={"target_subproblems": [f"Q{index}"]})
        for index, candidate in enumerate(candidates, start=1)
    ]
    gate = select_quality_gate(selection, modified, {"Q1", "Q2", "Q3"})
    assert gate.checks["primary_eligible"] and gate.checks["backup_eligible"]
    assert not gate.checks["primary_covers_all_subproblems"]
    assert not gate.checks["backup_covers_all_subproblems"]
    assert gate.status is QualityGateStatus.RETRY
