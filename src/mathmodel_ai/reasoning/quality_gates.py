from mathmodel_ai.reasoning.deduplication import deduplicate_candidates
from mathmodel_ai.reasoning.scoring import covers_required_subproblems, has_missing_required_data
from mathmodel_ai.schemas.model_selection import ModelCandidate, ModelExploration, ModelSelection
from mathmodel_ai.schemas.problem_analysis import (
    AmbiguityReviewStatus,
    EvidenceStatus,
    ProblemAnalysis,
)
from mathmodel_ai.schemas.quality import QualityGateResult, QualityGateStatus


def understand_quality_gate(analysis: ProblemAnalysis) -> QualityGateResult:
    checks = {
        "core_problem_present": bool(analysis.core_problem.statement),
        "objectives_present": bool(analysis.objectives),
        "subproblems_present": bool(analysis.subproblems),
        "facts_or_reason": bool(analysis.facts or analysis.no_facts_reason),
        "assumptions_are_proposed": all(
            item.status is EvidenceStatus.PROPOSED for item in analysis.assumptions_required
        ),
        "dependency_graph_valid": True,
    }
    errors = [name for name, passed in checks.items() if not passed]
    warnings = [
        f"ambiguity {item.ambiguity_id} should be reviewed by a human"
        for item in analysis.ambiguities
        if item.review_status is AmbiguityReviewStatus.HUMAN_REVIEW_RECOMMENDED
    ]
    if analysis.human_review_recommended and not warnings:
        warnings.append("analysis recommends human review")
    return QualityGateResult(
        gate="UNDERSTAND",
        status=QualityGateStatus.PASS if not errors else QualityGateStatus.RETRY,
        checks=checks,
        errors=errors,
        warnings=warnings,
    )


def explore_quality_gate(
    exploration: ModelExploration, required_subproblem_ids: set[str]
) -> QualityGateResult:
    deduplication = deduplicate_candidates(exploration.candidates)
    covered = {
        target for candidate in exploration.candidates for target in candidate.target_subproblems
    }
    blocked = [
        candidate.candidate_id
        for candidate in exploration.candidates
        if has_missing_required_data(candidate)
    ]
    checks = {
        "candidate_count": 2 <= len(exploration.candidates) <= 5,
        "fewer_than_three_explained": (
            len(exploration.candidates) >= 3 or bool(exploration.fewer_than_three_reason)
        ),
        "semantically_distinct": not deduplication.removed_ids,
        "two_candidates_without_missing_required_data": (
            len(exploration.candidates) - len(blocked) >= 2
        ),
        "two_end_to_end_candidates": sum(
            not has_missing_required_data(candidate)
            and covers_required_subproblems(candidate, required_subproblem_ids)
            for candidate in exploration.candidates
        )
        >= 2,
        "all_subproblems_covered": required_subproblem_ids <= covered,
        "candidate_targets_known": all(
            set(candidate.target_subproblems) <= required_subproblem_ids
            for candidate in exploration.candidates
        ),
    }
    errors = [name for name, passed in checks.items() if not passed]
    if not checks["two_candidates_without_missing_required_data"]:
        errors.extend(f"MISSING_REQUIRED_DATA:{candidate_id}" for candidate_id in blocked)
    if not checks["two_end_to_end_candidates"]:
        errors.extend(
            f"INCOMPLETE_CANDIDATE:{candidate.candidate_id}:missing="
            + ",".join(sorted(required_subproblem_ids - set(candidate.target_subproblems)))
            for candidate in exploration.candidates
            if not covers_required_subproblems(candidate, required_subproblem_ids)
        )
    return QualityGateResult(
        gate="EXPLORE",
        status=QualityGateStatus.PASS if not errors else QualityGateStatus.RETRY,
        checks=checks,
        errors=errors,
        warnings=[],
    )


def select_quality_gate(
    selection: ModelSelection,
    candidates: list[ModelCandidate],
    required_subproblem_ids: set[str],
) -> QualityGateResult:
    candidates_by_id = {item.candidate_id: item for item in candidates}
    candidate_ids = set(candidates_by_id)
    score_by_id = {score.candidate_id: score for score in selection.score_matrix}
    primary = score_by_id.get(selection.selected_model_id)
    backup = score_by_id.get(selection.backup_model_id)
    checks = {
        "primary_exists": selection.selected_model_id in candidate_ids,
        "backup_exists": selection.backup_model_id in candidate_ids,
        "models_differ": selection.selected_model_id != selection.backup_model_id,
        "primary_eligible": primary is not None and primary.eligible,
        "backup_eligible": backup is not None and backup.eligible,
        "primary_covers_all_subproblems": (
            selection.selected_model_id in candidates_by_id
            and covers_required_subproblems(
                candidates_by_id[selection.selected_model_id], required_subproblem_ids
            )
        ),
        "backup_covers_all_subproblems": (
            selection.backup_model_id in candidates_by_id
            and covers_required_subproblems(
                candidates_by_id[selection.backup_model_id], required_subproblem_ids
            )
        ),
        "all_candidates_scored": set(score_by_id) == candidate_ids,
    }
    errors = [name for name, passed in checks.items() if not passed]
    return QualityGateResult(
        gate="SELECT",
        status=QualityGateStatus.PASS if not errors else QualityGateStatus.RETRY,
        checks=checks,
        errors=errors,
        warnings=selection.critical_risks,
    )
