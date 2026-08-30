from __future__ import annotations

from uuid import UUID

from mathmodel_ai.paper.hashing import sha256_bytes
from mathmodel_ai.schemas.paper import PaperQualityStatus
from mathmodel_ai.schemas.submission import (
    CompetitionProfile,
    FinalJuryReport,
    JuryDecision,
    RequirementCoverage,
    RequirementCoverageStatus,
    RuleResult,
    RuleResultStatus,
    RuleSeverity,
    SubmissionCandidate,
    SubmissionCheckCategory,
    SubmissionCheckIssue,
    SubmissionCheckResult,
    SubmissionCheckStatus,
)
from mathmodel_ai.submission.integrity import (
    final_jury_report_digest,
    requirement_snapshot_digest,
    rule_result_snapshot_digest,
    submission_candidate_digest,
    submission_check_digest,
    validate_competition_profile,
)
from mathmodel_ai.submission.jury import FinalJuryGate
from mathmodel_ai.submission.rules import SubmissionSecurityScanner, is_allowed_path


class SubmissionCheck:
    def __init__(self) -> None:
        self._security = SubmissionSecurityScanner()
        self._jury = FinalJuryGate()

    def run(
        self,
        *,
        profile: CompetitionProfile,
        candidate: SubmissionCandidate,
        requirements: list[RequirementCoverage],
        rules: list[RuleResult],
        jury: FinalJuryReport,
        artifact_bytes: dict[UUID, bytes],
    ) -> SubmissionCheckResult:
        issues = self._issues(profile, candidate, requirements, rules, jury, artifact_bytes)
        status = self._status(issues, rules, jury)
        check = SubmissionCheckResult(
            project_id=candidate.project_id,
            paper_id=candidate.paper_id,
            paper_version=candidate.paper_version,
            verified_result_id=candidate.verified_result_id,
            jury_report_id=jury.report_id,
            competition_profile_digest=validate_competition_profile(profile),
            candidate_digest=submission_candidate_digest(candidate),
            requirement_snapshot_digest=requirement_snapshot_digest(requirements),
            rule_result_snapshot_digest=rule_result_snapshot_digest(rules),
            jury_report_digest=final_jury_report_digest(jury),
            status=status,
            issues=issues,
            rule_results=rules,
            requirement_coverage=requirements,
            claimed_failed_rule_count=self.failed_rule_count(rules),
            claimed_missing_requirement_count=self.missing_requirement_count(requirements),
            check_digest="0" * 64,
        )
        return check.model_copy(update={"check_digest": submission_check_digest(check)})

    def verify(
        self,
        check: SubmissionCheckResult,
        *,
        profile: CompetitionProfile,
        candidate: SubmissionCandidate,
        requirements: list[RequirementCoverage],
        rules: list[RuleResult],
        jury: FinalJuryReport,
        artifact_bytes: dict[UUID, bytes],
    ) -> SubmissionCheckStatus:
        if submission_check_digest(check) != check.check_digest:
            return SubmissionCheckStatus.FAIL
        if (
            check.project_id != candidate.project_id
            or check.paper_id != candidate.paper_id
            or check.paper_version != candidate.paper_version
            or check.verified_result_id != candidate.verified_result_id
            or check.jury_report_id != jury.report_id
            or check.competition_profile_digest != validate_competition_profile(profile)
            or check.competition_profile_digest != candidate.competition_profile_digest
            or check.candidate_digest != submission_candidate_digest(candidate)
            or check.requirement_snapshot_digest != requirement_snapshot_digest(requirements)
            or check.rule_result_snapshot_digest != rule_result_snapshot_digest(rules)
            or check.jury_report_digest != final_jury_report_digest(jury)
        ):
            return SubmissionCheckStatus.FAIL
        if check.claimed_failed_rule_count != self.failed_rule_count(rules):
            return SubmissionCheckStatus.FAIL
        if check.claimed_missing_requirement_count != self.missing_requirement_count(requirements):
            return SubmissionCheckStatus.FAIL
        expected_issues = self._issues(
            profile, candidate, requirements, rules, jury, artifact_bytes
        )
        if expected_issues != check.issues:
            return SubmissionCheckStatus.FAIL
        expected_status = self._status(expected_issues, rules, jury)
        return expected_status if check.status is expected_status else SubmissionCheckStatus.FAIL

    def _issues(
        self,
        profile: CompetitionProfile,
        candidate: SubmissionCandidate,
        requirements: list[RequirementCoverage],
        rules: list[RuleResult],
        jury: FinalJuryReport,
        artifact_bytes: dict[UUID, bytes],
    ) -> list[SubmissionCheckIssue]:
        issues: list[SubmissionCheckIssue] = []
        if not candidate.phase5_verified:
            issues.append(
                _issue(
                    "RESULT_NOT_VERIFIED",
                    SubmissionCheckCategory.RESULT,
                    "formal result is not Phase 5 VERIFIED",
                )
            )
        if candidate.paper_status is not PaperQualityStatus.READY_FOR_FINAL_JURY:
            issues.append(
                _issue(
                    "PAPER_NOT_READY", SubmissionCheckCategory.PAPER, "paper did not pass Phase 6"
                )
            )
        if candidate.paper_manifest_hash == "0" * 64:
            issues.append(
                _issue(
                    "PAPER_MANIFEST_INVALID",
                    SubmissionCheckCategory.MANIFEST,
                    "paper manifest is invalid",
                )
            )
        for result in rules:
            if result.status is RuleResultStatus.FAIL:
                issues.append(
                    _issue(
                        (
                            "BLOCKING_RULE_FAILED"
                            if result.severity is RuleSeverity.BLOCKING
                            else "COMPETITION_RULE_FAILED"
                        ),
                        SubmissionCheckCategory.FORMAT,
                        result.message,
                        result.rule_id,
                        result.severity,
                    )
                )
        for coverage in requirements:
            if coverage.required and coverage.status is not RequirementCoverageStatus.COVERED:
                issues.append(
                    _issue(
                        "REQUIREMENT_NOT_COVERED",
                        SubmissionCheckCategory.PROBLEM_COVERAGE,
                        "required output is not fully covered",
                        coverage.requirement_id,
                    )
                )
        jury_decision = self._jury.verify_report(
            jury,
            profile=profile,
            candidate=candidate,
            requirements=requirements,
            rule_results=rules,
        )
        if jury_decision is not JuryDecision.PASS:
            issues.append(
                _issue(
                    "FINAL_JURY_NOT_PASS",
                    SubmissionCheckCategory.PAPER,
                    f"final jury decision is {jury_decision.value}",
                    severity=(
                        RuleSeverity.MAJOR
                        if jury_decision is JuryDecision.HUMAN_REVIEW
                        else RuleSeverity.BLOCKING
                    ),
                )
            )
        for artifact in candidate.artifacts:
            data = artifact_bytes.get(artifact.artifact_id)
            if (
                data is None
                or len(data) != artifact.size_bytes
                or sha256_bytes(data) != artifact.sha256
            ):
                issues.append(
                    _issue(
                        "ARTIFACT_INTEGRITY_FAILED",
                        SubmissionCheckCategory.FILES,
                        "artifact bytes do not match the registry",
                        artifact.relative_path,
                    )
                )
            if not is_allowed_path(artifact.relative_path, profile):
                issues.append(
                    _issue(
                        "FILE_NOT_ALLOWED",
                        SubmissionCheckCategory.FILES,
                        "file is outside the profile allow-list",
                        artifact.relative_path,
                    )
                )
        for code, path in self._security.scan(candidate, artifact_bytes):
            category = SubmissionCheckCategory.SECURITY
            if code == "FORBIDDEN_FILE":
                category = SubmissionCheckCategory.FILES
            issues.append(_issue(code, category, f"submission security scan rejected {path}", path))
        return issues

    @staticmethod
    def _status(
        issues: list[SubmissionCheckIssue],
        rules: list[RuleResult],
        jury: FinalJuryReport,
    ) -> SubmissionCheckStatus:
        if jury.decision is JuryDecision.HUMAN_REVIEW or any(
            item.severity is RuleSeverity.BLOCKING
            and item.status in {RuleResultStatus.HUMAN_REVIEW, RuleResultStatus.NOT_EVALUABLE}
            for item in rules
        ):
            return SubmissionCheckStatus.HUMAN_REVIEW
        if any(item.severity is RuleSeverity.BLOCKING for item in issues):
            return SubmissionCheckStatus.FAIL
        if issues:
            return SubmissionCheckStatus.WARNING
        return SubmissionCheckStatus.PASS

    @staticmethod
    def failed_rule_count(rules: list[RuleResult]) -> int:
        return sum(item.status is RuleResultStatus.FAIL for item in rules)

    @staticmethod
    def missing_requirement_count(requirements: list[RequirementCoverage]) -> int:
        return sum(
            item.required and item.status is not RequirementCoverageStatus.COVERED
            for item in requirements
        )


def _issue(
    code: str,
    category: SubmissionCheckCategory,
    message: str,
    object_ref: str | None = None,
    severity: RuleSeverity = RuleSeverity.BLOCKING,
) -> SubmissionCheckIssue:
    return SubmissionCheckIssue(
        code=code,
        category=category,
        severity=severity,
        message=message,
        object_ref=object_ref,
    )
