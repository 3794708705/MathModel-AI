from __future__ import annotations

from uuid import UUID

from mathmodel_ai.schemas.paper import PaperQualityStatus
from mathmodel_ai.schemas.submission import (
    CompetitionProfile,
    FinalJuryDraft,
    FinalJuryReport,
    JuryDecision,
    JuryDimensionScore,
    JuryFinding,
    JurySeverity,
    RequirementCoverage,
    RequirementCoverageStatus,
    RuleResult,
    RuleResultStatus,
    RuleSeverity,
    SubmissionCandidate,
)
from mathmodel_ai.submission.integrity import (
    artifact_set_digest,
    final_jury_report_digest,
    requirement_snapshot_digest,
    rule_result_snapshot_digest,
    submission_candidate_digest,
    validate_competition_profile,
)


class FinalJuryGate:
    """Recompute hard facts and score; persisted summaries have no authority."""

    def build_report(
        self,
        *,
        draft: FinalJuryDraft,
        profile: CompetitionProfile,
        candidate: SubmissionCandidate,
        requirements: list[RequirementCoverage],
        rule_results: list[RuleResult],
        reviewer_is_mock: bool,
        agent_run_id: UUID | None,
    ) -> FinalJuryReport:
        findings = self._deterministic_findings(candidate, requirements, rule_results)
        findings.extend(draft.findings)
        findings = self._deduplicate(findings)
        score, dimensions_valid = self.recompute_score(draft.dimensions, profile)
        decision = self._decision(
            profile=profile,
            candidate=candidate,
            requirements=requirements,
            rules=rule_results,
            findings=findings,
            score=score,
            dimensions_valid=dimensions_valid,
            reviewer_is_mock=reviewer_is_mock,
            independent_reviewer_used=False,
        )
        report = FinalJuryReport(
            project_id=candidate.project_id,
            paper_id=candidate.paper_id,
            paper_version=candidate.paper_version,
            paper_manifest_hash=candidate.paper_manifest_hash,
            paper_ir_hash=candidate.paper_ir_hash,
            verified_model_id=candidate.verified_model_id,
            verified_model_version=candidate.verified_model_version,
            verified_model_digest=candidate.verified_model_digest,
            verified_result_id=candidate.verified_result_id,
            competition_profile_id=profile.profile_id,
            competition_profile_version=profile.version,
            competition_profile_digest=validate_competition_profile(profile),
            candidate_digest=submission_candidate_digest(candidate),
            requirement_snapshot_digest=requirement_snapshot_digest(requirements),
            rule_result_snapshot_digest=rule_result_snapshot_digest(rule_results),
            artifact_set_digest=artifact_set_digest(candidate.artifacts),
            dimensions=draft.dimensions,
            findings=findings,
            claimed_score=score,
            decision=decision,
            summary=draft.summary,
            reviewer_is_mock=reviewer_is_mock,
            independent_reviewer_used=False,
            agent_run_id=agent_run_id,
            report_digest="0" * 64,
        )
        return report.model_copy(update={"report_digest": final_jury_report_digest(report)})

    def verify_report(
        self,
        report: FinalJuryReport,
        *,
        profile: CompetitionProfile,
        candidate: SubmissionCandidate,
        requirements: list[RequirementCoverage],
        rule_results: list[RuleResult],
    ) -> JuryDecision:
        if final_jury_report_digest(report) != report.report_digest:
            return JuryDecision.FAIL
        if (
            report.project_id != candidate.project_id
            or report.paper_id != candidate.paper_id
            or report.paper_version != candidate.paper_version
            or report.verified_result_id != candidate.verified_result_id
            or report.competition_profile_id != profile.profile_id
            or report.competition_profile_version != profile.version
            or report.paper_manifest_hash != candidate.paper_manifest_hash
            or report.paper_ir_hash != candidate.paper_ir_hash
            or report.verified_model_id != candidate.verified_model_id
            or report.verified_model_version != candidate.verified_model_version
            or report.verified_model_digest != candidate.verified_model_digest
            or report.competition_profile_digest != validate_competition_profile(profile)
            or report.competition_profile_digest != candidate.competition_profile_digest
            or report.candidate_digest != submission_candidate_digest(candidate)
            or report.requirement_snapshot_digest != requirement_snapshot_digest(requirements)
            or report.rule_result_snapshot_digest != rule_result_snapshot_digest(rule_results)
            or report.artifact_set_digest != artifact_set_digest(candidate.artifacts)
        ):
            return JuryDecision.FAIL
        score, dimensions_valid = self.recompute_score(report.dimensions, profile)
        if abs(score - report.claimed_score) > 1e-9:
            return JuryDecision.FAIL
        expected = self._decision(
            profile=profile,
            candidate=candidate,
            requirements=requirements,
            rules=rule_results,
            findings=report.findings,
            score=score,
            dimensions_valid=dimensions_valid,
            reviewer_is_mock=report.reviewer_is_mock,
            independent_reviewer_used=report.independent_reviewer_used,
        )
        return expected if report.decision is expected else JuryDecision.FAIL

    @staticmethod
    def recompute_score(
        dimensions: list[JuryDimensionScore], profile: CompetitionProfile
    ) -> tuple[float, bool]:
        expected = profile.jury_weights
        names = [item.dimension for item in dimensions]
        if len(names) != len(set(names)) or set(names) != set(expected):
            return 0.0, False
        if any(abs(item.maximum - expected[item.dimension]) > 1e-9 for item in dimensions):
            return 0.0, False
        return sum(item.score for item in dimensions), True

    @staticmethod
    def _decision(
        *,
        profile: CompetitionProfile,
        candidate: SubmissionCandidate,
        requirements: list[RequirementCoverage],
        rules: list[RuleResult],
        findings: list[JuryFinding],
        score: float,
        dimensions_valid: bool,
        reviewer_is_mock: bool,
        independent_reviewer_used: bool,
    ) -> JuryDecision:
        if not dimensions_valid or not candidate.phase5_verified:
            return JuryDecision.FAIL
        if candidate.paper_status is not PaperQualityStatus.READY_FOR_FINAL_JURY:
            return JuryDecision.FAIL
        if any(
            item.severity is RuleSeverity.BLOCKING and item.status is RuleResultStatus.FAIL
            for item in rules
        ):
            return JuryDecision.FAIL
        if any(
            item.severity is RuleSeverity.BLOCKING
            and item.status in {RuleResultStatus.HUMAN_REVIEW, RuleResultStatus.NOT_EVALUABLE}
            for item in rules
        ):
            return JuryDecision.HUMAN_REVIEW
        if profile.require_independent_reviewer and (
            reviewer_is_mock or not independent_reviewer_used
        ):
            return JuryDecision.HUMAN_REVIEW
        if any(
            item.required and item.status is not RequirementCoverageStatus.COVERED
            for item in requirements
        ):
            return JuryDecision.FAIL
        if any(item.severity is JurySeverity.CRITICAL and not item.resolved for item in findings):
            return JuryDecision.FAIL
        if score < profile.minimum_jury_score:
            return JuryDecision.FAIL
        return JuryDecision.PASS

    @staticmethod
    def _deterministic_findings(
        candidate: SubmissionCandidate,
        requirements: list[RequirementCoverage],
        rules: list[RuleResult],
    ) -> list[JuryFinding]:
        findings: list[JuryFinding] = []
        if not candidate.phase5_verified:
            findings.append(_critical("phase5", "Official result is not Phase 5 VERIFIED"))
        if candidate.paper_status is not PaperQualityStatus.READY_FOR_FINAL_JURY:
            findings.append(_critical("paper", "Paper did not pass the Phase 6 final gate"))
        for coverage in requirements:
            if coverage.required and coverage.status is not RequirementCoverageStatus.COVERED:
                findings.append(
                    _critical(
                        coverage.requirement_id,
                        f"Required output is {coverage.status.value.lower()}",
                        paper_refs=coverage.paper_refs,
                        evidence_refs=[str(item) for item in coverage.evidence_refs],
                    )
                )
        for result in rules:
            if (
                result.severity is RuleSeverity.BLOCKING
                and result.status is not RuleResultStatus.PASS
            ):
                findings.append(
                    _critical(
                        result.rule_id,
                        f"Blocking competition rule is {result.status.value}",
                        rule_refs=[result.rule_id],
                    )
                )
        return findings

    @staticmethod
    def _deduplicate(findings: list[JuryFinding]) -> list[JuryFinding]:
        result: list[JuryFinding] = []
        seen: set[str] = set()
        for finding in findings:
            if finding.finding_id not in seen:
                seen.add(finding.finding_id)
                result.append(finding)
        return result


def _critical(
    suffix: str,
    title: str,
    *,
    evidence_refs: list[str] | None = None,
    paper_refs: list[str] | None = None,
    rule_refs: list[str] | None = None,
) -> JuryFinding:
    safe_suffix = "".join(
        character if character.isalnum() or character in "_-" else "-" for character in suffix
    )
    return JuryFinding(
        finding_id=f"JURY-{safe_suffix}",
        category="deterministic_gate",
        severity=JurySeverity.CRITICAL,
        title=title,
        description=title,
        evidence_refs=evidence_refs or [],
        paper_refs=paper_refs or [],
        rule_refs=rule_refs or [],
        impact="Submission cannot be frozen.",
        recommendation="Correct the source issue and rerun required upstream validation.",
        confidence=1.0,
    )
