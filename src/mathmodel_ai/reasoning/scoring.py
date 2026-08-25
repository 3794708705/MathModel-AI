from mathmodel_ai.schemas.model_selection import (
    CandidateJuryAssessment,
    HardFailure,
    JuryAssessment,
    ModelCandidate,
    ModelJuryWeights,
    ModelScore,
    ModelSelection,
    RejectedModel,
)
from mathmodel_ai.schemas.problem_analysis import DataAvailability

_SCORE_FIELDS = (
    "problem_fit",
    "data_fit",
    "mathematical_validity",
    "explainability",
    "validation_potential",
    "innovation_potential",
    "competition_feasibility",
    "computational_cost",
)


def calculate_score(assessment: CandidateJuryAssessment, weights: ModelJuryWeights) -> ModelScore:
    total = sum(
        getattr(assessment, field_name) * getattr(weights, field_name) / 10
        for field_name in _SCORE_FIELDS
    )
    return ModelScore(
        **assessment.model_dump(),
        weighted_total=round(total, 4),
        eligible=not assessment.hard_failures,
    )


def build_model_selection(
    candidates: list[ModelCandidate],
    assessment: JuryAssessment,
    weights: ModelJuryWeights,
) -> ModelSelection:
    candidate_ids = {candidate.candidate_id for candidate in candidates}
    assessment_ids = {item.candidate_id for item in assessment.assessments}
    if candidate_ids != assessment_ids:
        missing = sorted(candidate_ids - assessment_ids)
        unknown = sorted(assessment_ids - candidate_ids)
        raise ValueError(f"jury coverage mismatch: missing={missing}, unknown={unknown}")

    candidates_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    normalized_assessments = []
    for item in assessment.assessments:
        candidate = candidates_by_id[item.candidate_id]
        hard_failures = list(item.hard_failures)
        if (
            any(
                requirement.availability is DataAvailability.MISSING
                for requirement in candidate.data_requirements
            )
            and HardFailure.DATA_NOT_AVAILABLE not in hard_failures
        ):
            hard_failures.append(HardFailure.DATA_NOT_AVAILABLE)
        normalized_assessments.append(item.model_copy(update={"hard_failures": hard_failures}))

    scores = [calculate_score(item, weights) for item in normalized_assessments]
    scores.sort(key=lambda item: (-item.eligible, -item.weighted_total, item.candidate_id))
    eligible = [score for score in scores if score.eligible]
    if len(eligible) < 2:
        raise ValueError("at least two eligible candidates are required for primary and backup")

    rejected = [
        RejectedModel(
            candidate_id=score.candidate_id,
            reason=(
                f"hard failure: {', '.join(item.value for item in score.hard_failures)}"
                if score.hard_failures
                else "lower deterministic jury ranking"
            ),
            hard_failures=score.hard_failures,
        )
        for score in scores[2:]
    ]
    primary, backup = eligible[:2]
    return ModelSelection(
        selected_model_id=primary.candidate_id,
        backup_model_id=backup.candidate_id,
        ranking=[score.candidate_id for score in scores],
        score_matrix=scores,
        decision_reason=(
            f"{assessment.overall_rationale} Deterministic weighted ranking selected "
            f"{primary.candidate_id} ({primary.weighted_total:.2f}) with "
            f"{backup.candidate_id} ({backup.weighted_total:.2f}) as backup."
        ),
        rejected_models=rejected,
        critical_risks=assessment.critical_risks,
        confidence=assessment.confidence,
        weights_version=weights.version,
    )
