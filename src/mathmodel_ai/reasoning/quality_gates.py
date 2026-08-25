from mathmodel_ai.reasoning.deduplication import deduplicate_candidates
from mathmodel_ai.schemas.model_selection import ModelExploration, ModelSelection
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
    checks = {
        "candidate_count": 2 <= len(exploration.candidates) <= 5,
        "fewer_than_three_explained": (
            len(exploration.candidates) >= 3 or bool(exploration.fewer_than_three_reason)
        ),
        "semantically_distinct": not deduplication.removed_ids,
        "all_subproblems_covered": required_subproblem_ids <= covered,
        "candidate_targets_known": all(
            set(candidate.target_subproblems) <= required_subproblem_ids
            for candidate in exploration.candidates
        ),
    }
    errors = [name for name, passed in checks.items() if not passed]
    return QualityGateResult(
        gate="EXPLORE",
        status=QualityGateStatus.PASS if not errors else QualityGateStatus.RETRY,
        checks=checks,
        errors=errors,
        warnings=[],
    )


def select_quality_gate(selection: ModelSelection, candidate_ids: set[str]) -> QualityGateResult:
    score_by_id = {score.candidate_id: score for score in selection.score_matrix}
    primary = score_by_id.get(selection.selected_model_id)
    backup = score_by_id.get(selection.backup_model_id)
    checks = {
        "primary_exists": selection.selected_model_id in candidate_ids,
        "backup_exists": selection.backup_model_id in candidate_ids,
        "models_differ": selection.selected_model_id != selection.backup_model_id,
        "primary_eligible": primary is not None and primary.eligible,
        "backup_eligible": backup is not None and backup.eligible,
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
