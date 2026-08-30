from __future__ import annotations

from threading import Lock
from uuid import UUID

from mathmodel_ai.schemas.paper import PaperIR
from mathmodel_ai.schemas.submission import (
    CompetitionProfile,
    FinalJuryReport,
    JuryDecision,
    RequirementCoverage,
    RuleResult,
    SubmissionCandidate,
    SubmissionCheckResult,
    SubmissionCheckStatus,
    SubmissionRequirement,
    SubmissionStatus,
)
from mathmodel_ai.submission.checks import SubmissionCheck
from mathmodel_ai.submission.integrity import (
    requirement_snapshot_digest,
    rule_result_snapshot_digest,
)
from mathmodel_ai.submission.jury import FinalJuryGate
from mathmodel_ai.submission.package import SubmissionPackage, SubmissionPackageBuilder
from mathmodel_ai.submission.requirements import RequirementCoverageValidator
from mathmodel_ai.submission.rules import RuleEngine


class SubmissionFreeze:
    """The only coordinator allowed to turn a checked candidate into a frozen package."""

    def __init__(self, builder: SubmissionPackageBuilder, rule_engine: RuleEngine) -> None:
        self._builder = builder
        self._rules = rule_engine
        self._jury = FinalJuryGate()
        self._check = SubmissionCheck()
        self._coverage = RequirementCoverageValidator()
        self._frozen: dict[UUID, SubmissionPackage] = {}
        self._freeze_lock = Lock()

    def freeze(
        self,
        *,
        profile: CompetitionProfile,
        candidate: SubmissionCandidate,
        paper: PaperIR,
        submission_requirements: list[SubmissionRequirement],
        requirements: list[RequirementCoverage],
        rules: list[RuleResult],
        jury: FinalJuryReport,
        check: SubmissionCheckResult,
        artifact_bytes: dict[UUID, bytes],
    ) -> SubmissionPackage:
        if jury.decision is not JuryDecision.PASS:
            raise ValueError("Final Jury must pass before submission freeze")
        if check.status is not SubmissionCheckStatus.PASS:
            raise ValueError("SubmissionCheck must pass before submission freeze")
        if (
            jury.project_id != candidate.project_id
            or jury.paper_id != candidate.paper_id
            or jury.paper_version != candidate.paper_version
            or jury.verified_result_id != candidate.verified_result_id
            or check.project_id != candidate.project_id
            or check.paper_id != candidate.paper_id
            or check.paper_version != candidate.paper_version
            or check.verified_result_id != candidate.verified_result_id
            or check.jury_report_id != jury.report_id
        ):
            raise ValueError("freeze inputs do not bind the same approved candidate")
        recomputed_rules = self._rules.evaluate(profile, candidate, artifact_bytes=artifact_bytes)
        if rule_result_snapshot_digest(recomputed_rules) != rule_result_snapshot_digest(rules):
            raise ValueError("competition rule results changed before freeze")
        recomputed_coverage = self._coverage.validate(
            submission_requirements, paper, candidate.artifacts
        )
        if requirement_snapshot_digest(recomputed_coverage) != requirement_snapshot_digest(
            requirements
        ):
            raise ValueError("problem requirement coverage changed before freeze")
        if (
            self._jury.verify_report(
                jury,
                profile=profile,
                candidate=candidate,
                requirements=recomputed_coverage,
                rule_results=recomputed_rules,
            )
            is not JuryDecision.PASS
        ):
            raise ValueError("Final Jury integrity failed before freeze")
        if (
            self._check.verify(
                check,
                profile=profile,
                candidate=candidate,
                requirements=recomputed_coverage,
                rules=recomputed_rules,
                jury=jury,
                artifact_bytes=artifact_bytes,
            )
            is not SubmissionCheckStatus.PASS
        ):
            raise ValueError("SubmissionCheck integrity failed before freeze")
        with self._freeze_lock:
            cached = self._frozen.get(candidate.candidate_id)
            if cached is not None:
                if self._builder.verify(cached) is not SubmissionStatus.FROZEN:
                    raise ValueError("previous frozen package became dirty")
                return cached
            package = self._builder.build(
                profile=profile,
                candidate=candidate,
                requirements=recomputed_coverage,
                jury=jury,
                check=check,
            )
            self._frozen[candidate.candidate_id] = package
            return package
