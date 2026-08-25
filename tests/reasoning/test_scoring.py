from mathmodel_ai.reasoning.scoring import build_model_selection, calculate_score
from mathmodel_ai.schemas.model_selection import (
    CandidateJuryAssessment,
    HardFailure,
    JuryAssessment,
    ModelFamily,
    ModelJuryWeights,
)
from mathmodel_ai.schemas.problem_analysis import DataAvailability
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
