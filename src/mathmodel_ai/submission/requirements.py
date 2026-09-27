from __future__ import annotations

from mathmodel_ai.schemas.paper import ClaimType, ClaimVerificationStatus, PaperIR
from mathmodel_ai.schemas.problem_analysis import ProblemTaskType, SubProblem
from mathmodel_ai.schemas.submission import (
    RequirementCoverage,
    RequirementCoverageStatus,
    SubmissionArtifact,
    SubmissionArtifactRole,
    SubmissionRequirement,
)
from mathmodel_ai.submission.integrity import requirement_digest


class RequirementRegistry:
    @staticmethod
    def from_subproblems(subproblems: list[SubProblem]) -> list[SubmissionRequirement]:
        requirements: list[SubmissionRequirement] = []
        for subproblem in sorted(subproblems, key=lambda item: item.order):
            for index, output in enumerate(subproblem.output_required, start=1):
                requirements.append(
                    SubmissionRequirement(
                        requirement_id=f"REQ-{subproblem.subproblem_id}-{index}",
                        source_text=subproblem.original_text,
                        normalized_requirement=subproblem.normalized_goal,
                        subproblem_id=subproblem.subproblem_id,
                        required_output=output,
                        allowed_claim_types=_allowed_claim_types(output, subproblem.task_types),
                    )
                )
        return requirements


class RequirementCoverageValidator:
    """Require real claim/evidence mapping; a named section alone is insufficient."""

    def validate(
        self,
        requirements: list[SubmissionRequirement],
        paper: PaperIR,
        artifacts: list[SubmissionArtifact],
        *,
        require_paper_artifact: bool = True,
    ) -> list[RequirementCoverage]:
        claims = {item.claim_id: item for item in paper.claims}
        sections = {item.section_id: item for item in [*paper.sections, *paper.appendices]}
        visible_claims = {
            item.section_id: {
                block.claim_ref for block in item.blocks if block.claim_ref is not None
            }
            for item in [*paper.sections, *paper.appendices]
        }
        coverage_by_subproblem = {item.subproblem_id: item for item in paper.subproblem_coverage}
        paper_artifacts = [
            item.artifact_id for item in artifacts if item.role is SubmissionArtifactRole.PAPER_PDF
        ]
        results: list[RequirementCoverage] = []
        for requirement in requirements:
            declared = coverage_by_subproblem.get(requirement.subproblem_id)
            reasons: list[str] = []
            valid_sections: list[str] = []
            valid_claims: list[str] = []
            evidence_refs = []
            if declared is None:
                reasons.append("paper has no structured subproblem coverage record")
            else:
                if requirement.required_output not in declared.required_outputs:
                    reasons.append("required output is absent from structured coverage")
                valid_sections = [
                    item
                    for item in declared.section_ids
                    if item in sections
                    and requirement.subproblem_id in sections[item].subproblem_refs
                ]
                if not valid_sections:
                    reasons.append(
                        "coverage does not reference an existing section bound to the subproblem"
                    )
                output_claim_refs = declared.output_claim_refs.get(requirement.required_output, [])
                if not output_claim_refs:
                    reasons.append("required output has no explicit claim mapping")
                for claim_id in output_claim_refs:
                    claim = claims.get(claim_id)
                    if claim is None:
                        continue
                    if claim_id not in declared.claim_refs:
                        continue
                    if claim.claim_type not in requirement.allowed_claim_types:
                        continue
                    if claim.section_id not in valid_sections:
                        continue
                    if (
                        claim_id not in sections[claim.section_id].claim_refs
                        and claim_id not in visible_claims[claim.section_id]
                    ):
                        continue
                    if claim.verification_status is not ClaimVerificationStatus.SUPPORTED:
                        continue
                    if not claim.evidence_refs:
                        continue
                    valid_claims.append(claim_id)
                    evidence_refs.extend(claim.evidence_refs)
                if not valid_claims:
                    reasons.append("coverage has no supported evidence-linked claim")
            if require_paper_artifact and not paper_artifacts:
                reasons.append("formal paper artifact is missing")
            if not requirement.required:
                status = RequirementCoverageStatus.NOT_APPLICABLE
            elif not reasons:
                status = RequirementCoverageStatus.COVERED
            elif declared is not None and (valid_sections or valid_claims):
                status = RequirementCoverageStatus.PARTIAL
            else:
                status = RequirementCoverageStatus.MISSING
            results.append(
                RequirementCoverage(
                    requirement_id=requirement.requirement_id,
                    requirement_digest=requirement_digest(requirement),
                    subproblem_id=requirement.subproblem_id,
                    required=requirement.required,
                    status=status,
                    evidence_refs=sorted(set(evidence_refs), key=str),
                    paper_refs=[*valid_sections, *valid_claims],
                    artifact_refs=paper_artifacts,
                    reasons=reasons,
                )
            )
        return results


def _allowed_claim_types(output: str, task_types: list[ProblemTaskType]) -> list[ClaimType]:
    normalized = output.casefold()
    conceptual_output = any(
        marker in normalized
        for marker in (
            "analysis",
            "definition",
            "criterion",
            "discussion",
            "interpretation",
            "treatment",
            "model",
            "formulation",
            "equation",
        )
    ) or (
        any(marker in normalized for marker in ("identification", "identify"))
        and any(
            marker in normalized
            for marker in ("species", "interaction", "component", "entity", "population")
        )
    )
    if any(
        marker in normalized
        for marker in ("risk", "robust", "stability", "uncert", "风险", "稳健", "敏感")
    ):
        allowed = [
            ClaimType.ROBUSTNESS,
            ClaimType.SENSITIVITY,
            ClaimType.COMPARISON,
            ClaimType.CONCLUSION,
        ]
        if "stability" in normalized:
            allowed.append(ClaimType.NUMERIC)
        if any(marker in normalized for marker in ("definition", "criterion")):
            allowed.append(ClaimType.MODEL)
        return allowed
    if any(
        item in {ProblemTaskType.PREDICTION, ProblemTaskType.TIME_SERIES} for item in task_types
    ):
        allowed = [ClaimType.NUMERIC, ClaimType.RESULT, ClaimType.COMPARISON]
        if conceptual_output:
            allowed.append(ClaimType.MODEL)
        if any(marker in normalized for marker in ("discussion", "interpretation", "limitation")):
            allowed.append(ClaimType.CONCLUSION)
        return allowed
    if any(
        item
        in {
            ProblemTaskType.OPTIMIZATION,
            ProblemTaskType.DECISION,
            ProblemTaskType.MULTI_OBJECTIVE,
            ProblemTaskType.NETWORK,
        }
        for item in task_types
    ):
        allowed = [ClaimType.RESULT, ClaimType.NUMERIC, ClaimType.COMPARISON]
        if conceptual_output:
            allowed.append(ClaimType.MODEL)
        if any(marker in normalized for marker in ("discussion", "interpretation", "limitation")):
            allowed.append(ClaimType.CONCLUSION)
        return allowed
    if ProblemTaskType.EVALUATION in task_types:
        allowed = [
            ClaimType.NUMERIC,
            ClaimType.RESULT,
            ClaimType.COMPARISON,
            ClaimType.CONCLUSION,
            ClaimType.ROBUSTNESS,
            ClaimType.SENSITIVITY,
        ]
        if conceptual_output:
            allowed.append(ClaimType.MODEL)
        return allowed
    return [
        ClaimType.FACTUAL,
        ClaimType.NUMERIC,
        ClaimType.RESULT,
        ClaimType.COMPARISON,
        ClaimType.SENSITIVITY,
        ClaimType.ROBUSTNESS,
        ClaimType.CONCLUSION,
        *([ClaimType.MODEL] if conceptual_output else []),
    ]
