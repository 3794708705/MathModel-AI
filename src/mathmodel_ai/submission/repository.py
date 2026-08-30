from __future__ import annotations

from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from mathmodel_ai.core.errors import ResourceNotFoundError
from mathmodel_ai.db.models import (
    CompetitionProfileRecord,
    CompetitionRuleRecord,
    CorrectionPlanRecord,
    FinalJuryReportRecord,
    JuryFindingRecord,
    PaperVersionRecord,
    Problem,
    ProblemStateRecord,
    RequirementCoverageRecordModel,
    ResultRecordModel,
    SubmissionArtifactRecordModel,
    SubmissionCheckRecord,
    SubmissionManifestRecord,
    SubmissionSnapshotRecord,
)
from mathmodel_ai.db.session import session_scope
from mathmodel_ai.paper.hashing import sha256_json
from mathmodel_ai.schemas.paper import PaperQualityStatus, PaperVersion
from mathmodel_ai.schemas.problem_state import ProblemState
from mathmodel_ai.schemas.results import ResultRecord
from mathmodel_ai.schemas.submission import (
    CompetitionProfile,
    CorrectionPlan,
    FinalJuryReport,
    JuryFinding,
    RequirementCoverage,
    SubmissionArtifact,
    SubmissionCheckResult,
    SubmissionManifest,
    SubmissionSnapshot,
)
from mathmodel_ai.submission.integrity import (
    final_jury_report_digest,
    requirement_digest,
    requirement_snapshot_digest,
    submission_check_digest,
    submission_snapshot_digest,
    validate_competition_profile,
)
from mathmodel_ai.submission.package import SubmissionPackage
from mathmodel_ai.submission.requirements import RequirementRegistry


class SubmissionRepository:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def persist_profile(self, profile: CompetitionProfile) -> None:
        profile_digest = validate_competition_profile(profile)
        with session_scope(self._session_factory) as session:
            existing = self._profile_row(session, profile.profile_id, profile.version)
            payload = profile.model_dump(mode="json")
            if existing is not None:
                if existing.profile_json != payload or existing.profile_digest != profile_digest:
                    raise ValueError(
                        "immutable competition profile version has conflicting content"
                    )
                return
            row = CompetitionProfileRecord(
                profile_id=profile.profile_id,
                version=profile.version,
                name=profile.name,
                verification_status=profile.verification_status.value,
                profile_digest=profile_digest,
                profile_json=payload,
            )
            session.add(row)
            session.flush()
            for rule in profile.rules:
                session.add(
                    CompetitionRuleRecord(
                        profile_record_id=row.id,
                        rule_id=rule.rule_id,
                        rule_type=rule.rule_type.value,
                        severity=rule.severity.value,
                        verification_status=rule.verification_status.value,
                        rule_json=rule.model_dump(mode="json"),
                    )
                )

    def list_profiles(self) -> list[CompetitionProfile]:
        with session_scope(self._session_factory) as session:
            rows = list(
                session.scalars(
                    select(CompetitionProfileRecord).order_by(
                        CompetitionProfileRecord.name, CompetitionProfileRecord.version
                    )
                )
            )
            return [self._validated_profile_row(session, item) for item in rows]

    def get_profile(self, profile_id: UUID, version: int | None = None) -> CompetitionProfile:
        with session_scope(self._session_factory) as session:
            statement = select(CompetitionProfileRecord).where(
                CompetitionProfileRecord.profile_id == profile_id
            )
            if version is not None:
                statement = statement.where(CompetitionProfileRecord.version == version)
            else:
                statement = statement.order_by(CompetitionProfileRecord.version.desc()).limit(1)
            row = session.scalar(statement)
            if row is None:
                raise ResourceNotFoundError("competition profile was not found")
            return self._validated_profile_row(session, row)

    def persist_run(
        self,
        *,
        profile: CompetitionProfile,
        coverage: list[RequirementCoverage],
        jury: FinalJuryReport,
        check: SubmissionCheckResult,
        package: SubmissionPackage | None,
        states: list[ProblemState],
    ) -> None:
        if not states:
            raise ValueError("Phase 7 persistence requires at least one state revision")
        profile_digest = validate_competition_profile(profile)
        coverage_digest = requirement_snapshot_digest(coverage)
        if final_jury_report_digest(jury) != jury.report_digest:
            raise ValueError("final jury report digest is invalid")
        if submission_check_digest(check) != check.check_digest:
            raise ValueError("submission check digest is invalid")
        if (
            jury.competition_profile_digest != profile_digest
            or check.competition_profile_digest != profile_digest
            or jury.requirement_snapshot_digest != coverage_digest
            or check.requirement_snapshot_digest != coverage_digest
            or check.jury_report_id != jury.report_id
            or check.jury_report_digest != jury.report_digest
        ):
            raise ValueError("Phase 7 approval records do not share one immutable snapshot")
        state = states[-1]
        with session_scope(self._session_factory) as session:
            current = self._lock_current(session, state.project_id)
            expected_versions = list(
                range(current.revision + 1, current.revision + len(states) + 1)
            )
            if [item.version for item in states] != expected_versions:
                raise ValueError("stale or discontinuous Phase 7 state revisions")
            if any(
                item.project_id != state.project_id or item.problem_id != state.problem_id
                for item in states
            ):
                raise ValueError("Phase 7 state history crosses project identity")
            profile_row = self._profile_row(session, profile.profile_id, profile.version)
            if (
                profile_row is None
                or profile_row.profile_json != profile.model_dump(mode="json")
                or profile_row.profile_digest != profile_digest
            ):
                raise ValueError("persisted competition profile version is missing or changed")
            paper_row = session.scalar(
                select(PaperVersionRecord).where(
                    PaperVersionRecord.project_id == state.project_id,
                    PaperVersionRecord.paper_id == jury.paper_id,
                    PaperVersionRecord.version == jury.paper_version,
                )
            )
            if paper_row is None:
                raise ResourceNotFoundError("approved paper version was not found")
            for item in coverage:
                session.add(
                    RequirementCoverageRecordModel(
                        id=item.coverage_id,
                        project_id=state.project_id,
                        paper_version_record_id=paper_row.id,
                        requirement_id=item.requirement_id,
                        subproblem_id=item.subproblem_id,
                        status=item.status.value,
                        coverage_json=item.model_dump(mode="json"),
                    )
                )
            jury_row = FinalJuryReportRecord(
                id=jury.report_id,
                project_id=state.project_id,
                paper_version_record_id=paper_row.id,
                verified_result_id=jury.verified_result_id,
                profile_record_id=profile_row.id,
                decision=jury.decision.value,
                claimed_score=jury.claimed_score,
                reviewer_is_mock=jury.reviewer_is_mock,
                report_digest=jury.report_digest,
                agent_run_id=jury.agent_run_id,
                report_json=jury.model_dump(mode="json"),
                created_at=jury.created_at,
            )
            session.add(jury_row)
            for finding in jury.findings:
                session.add(
                    JuryFindingRecord(
                        jury_report_id=jury.report_id,
                        finding_id=finding.finding_id,
                        severity=finding.severity.value,
                        resolved=finding.resolved,
                        finding_json=finding.model_dump(mode="json"),
                    )
                )
            session.add(
                SubmissionCheckRecord(
                    id=check.check_id,
                    project_id=state.project_id,
                    jury_report_id=jury.report_id,
                    status=check.status.value,
                    check_digest=check.check_digest,
                    check_json=check.model_dump(mode="json"),
                    created_at=check.created_at,
                )
            )
            if package is not None:
                self._persist_package(
                    session,
                    package=package,
                    profile_row=profile_row,
                    paper_row=paper_row,
                    check=check,
                )
            current.is_current = False
            session.flush()
            for index, revision_state in enumerate(states):
                session.add(
                    ProblemStateRecord(
                        problem_id=revision_state.problem_id,
                        revision=revision_state.version,
                        schema_version=revision_state.schema_version,
                        current_stage=revision_state.current_stage.value,
                        status=revision_state.status.value,
                        is_current=index == len(states) - 1,
                        state_json=revision_state.model_dump(mode="json"),
                        updated_by=revision_state.updated_by,
                        update_reason=revision_state.update_reason,
                    )
                )

    def get_latest_jury(self, project_id: UUID) -> FinalJuryReport:
        with session_scope(self._session_factory) as session:
            row = session.scalar(
                select(FinalJuryReportRecord)
                .where(FinalJuryReportRecord.project_id == project_id)
                .order_by(FinalJuryReportRecord.created_at.desc())
                .limit(1)
            )
            if row is None:
                raise ResourceNotFoundError("final jury report was not found")
            report = FinalJuryReport.model_validate(row.report_json)
            if (
                row.id != report.report_id
                or row.project_id != report.project_id
                or row.verified_result_id != report.verified_result_id
                or row.decision != report.decision.value
                or row.claimed_score != report.claimed_score
                or row.reviewer_is_mock != report.reviewer_is_mock
                or row.report_digest != report.report_digest
                or final_jury_report_digest(report) != report.report_digest
            ):
                raise ValueError("final jury report record failed integrity validation")
            finding_rows = list(
                session.scalars(
                    select(JuryFindingRecord)
                    .where(JuryFindingRecord.jury_report_id == row.id)
                    .order_by(JuryFindingRecord.finding_id)
                )
            )
            persisted_findings = sorted(
                (JuryFinding.model_validate(item.finding_json) for item in finding_rows),
                key=lambda item: item.finding_id,
            )
            if persisted_findings != sorted(report.findings, key=lambda item: item.finding_id):
                raise ValueError("jury report and finding records disagree")
            return report

    def get_latest_check(self, project_id: UUID) -> SubmissionCheckResult:
        with session_scope(self._session_factory) as session:
            row = session.scalar(
                select(SubmissionCheckRecord)
                .where(SubmissionCheckRecord.project_id == project_id)
                .order_by(SubmissionCheckRecord.created_at.desc())
                .limit(1)
            )
            if row is None:
                raise ResourceNotFoundError("submission check was not found")
            check = SubmissionCheckResult.model_validate(row.check_json)
            if (
                row.id != check.check_id
                or row.project_id != check.project_id
                or row.jury_report_id != check.jury_report_id
                or row.status != check.status.value
                or row.check_digest != check.check_digest
                or submission_check_digest(check) != check.check_digest
            ):
                raise ValueError("submission check record failed integrity validation")
            return check

    def get_latest_snapshot(self, project_id: UUID) -> SubmissionSnapshot:
        with session_scope(self._session_factory) as session:
            row = session.scalar(
                select(SubmissionSnapshotRecord)
                .where(SubmissionSnapshotRecord.project_id == project_id)
                .order_by(SubmissionSnapshotRecord.created_at.desc())
                .limit(1)
            )
            if row is None:
                raise ResourceNotFoundError("submission snapshot was not found")
            snapshot = SubmissionSnapshot.model_validate(row.snapshot_json)
            if (
                row.id != snapshot.submission_id
                or row.project_id != snapshot.project_id
                or row.verified_result_id != snapshot.verified_result_id
                or row.submission_check_id != snapshot.submission_check_id
                or row.status != snapshot.status.value
                or row.manifest_hash != snapshot.manifest_hash
                or row.package_hash != snapshot.package_hash
                or row.artifact_set_digest != snapshot.artifact_set_digest
                or row.snapshot_digest != snapshot.snapshot_digest
                or submission_snapshot_digest(snapshot) != snapshot.snapshot_digest
            ):
                raise ValueError("submission snapshot record failed integrity validation")
            return snapshot

    def get_manifest(self, submission_id: UUID) -> SubmissionManifest:
        with session_scope(self._session_factory) as session:
            row = session.scalar(
                select(SubmissionManifestRecord).where(
                    SubmissionManifestRecord.submission_id == submission_id
                )
            )
            if row is None:
                raise ResourceNotFoundError("submission manifest was not found")
            manifest = SubmissionManifest.model_validate(row.manifest_json)
            expected_hash = sha256_json(manifest.model_dump(mode="json", exclude={"manifest_hash"}))
            if (
                row.submission_id != manifest.submission_id
                or row.manifest_hash != manifest.manifest_hash
                or row.package_hash != manifest.package_hash
                or expected_hash != manifest.manifest_hash
            ):
                raise ValueError("submission manifest record failed integrity validation")
            return manifest

    def snapshot_context_valid(self, project_id: UUID, snapshot: SubmissionSnapshot) -> bool:
        """Re-prove that a frozen snapshot still names the current official chain."""

        try:
            with session_scope(self._session_factory) as session:
                state_row = session.scalar(
                    select(ProblemStateRecord)
                    .join(Problem, Problem.id == ProblemStateRecord.problem_id)
                    .where(
                        Problem.project_id == project_id,
                        ProblemStateRecord.is_current.is_(True),
                    )
                )
                if state_row is None:
                    return False
                state = ProblemState.model_validate(state_row.state_json)
                if (
                    state.project_id != project_id
                    or state.version != state_row.revision
                    or state.verified_result_id != snapshot.verified_result_id
                ):
                    return False
                if not any(
                    item.gate == "VERIFIED"
                    and item.status.value == "PASS"
                    and item.subject_ref == f"result:{snapshot.verified_result_id}"
                    for item in state.quality_gates
                ):
                    return False
                result_row = session.get(ResultRecordModel, snapshot.verified_result_id)
                if result_row is None:
                    return False
                result = ResultRecord.model_validate(result_row.record_json)
                if (
                    result.result_id != result_row.id
                    or result.project_id != project_id
                    or result.model_id != snapshot.verified_model_id
                    or result.model_version != snapshot.verified_model_version
                    or result.model_digest != snapshot.verified_model_digest
                    or result_row.model_id != result.model_id
                    or result_row.model_version != result.model_version
                    or result_row.model_digest != result.model_digest
                    or result_row.status != result.status.value
                ):
                    return False
                paper_row = session.scalar(
                    select(PaperVersionRecord).where(
                        PaperVersionRecord.project_id == project_id,
                        PaperVersionRecord.paper_id == snapshot.paper_id,
                        PaperVersionRecord.version == snapshot.paper_version,
                    )
                )
                if paper_row is None:
                    return False
                paper = PaperVersion.model_validate(paper_row.version_json)
                if (
                    paper.paper_id != paper_row.paper_id
                    or paper.version != paper_row.version
                    or paper.project_id != project_id
                    or paper.status is not PaperQualityStatus.READY_FOR_FINAL_JURY
                    or paper_row.status != paper.status.value
                    or paper.manifest_hash != snapshot.paper_manifest_hash
                    or sha256_json(paper.paper_ir.model_dump(mode="json", exclude={"status"}))
                    != snapshot.paper_ir_hash
                    or paper_row.manifest_hash != snapshot.paper_manifest_hash
                    or paper.evidence_snapshot.verified_result_id != snapshot.verified_result_id
                    or paper_row.verified_result_id != snapshot.verified_result_id
                ):
                    return False
                paper_refs = [
                    item
                    for item in state.paper_versions
                    if (item.paper_id, item.version) == (snapshot.paper_id, snapshot.paper_version)
                ]
                if (
                    len(paper_refs) != 1
                    or paper_refs[0].status is not PaperQualityStatus.READY_FOR_FINAL_JURY
                    or paper_refs[0].manifest_hash != snapshot.paper_manifest_hash
                ):
                    return False
                snapshot_refs = [
                    item
                    for item in state.submission_snapshots
                    if item.submission_id == snapshot.submission_id
                ]
                if (
                    len(snapshot_refs) != 1
                    or snapshot_refs[0].manifest_hash != snapshot.manifest_hash
                    or snapshot_refs[0].package_hash != snapshot.package_hash
                ):
                    return False
                profile_row = self._profile_row(
                    session,
                    snapshot.competition_profile_id,
                    snapshot.competition_profile_version,
                )
                if (
                    profile_row is None
                    or validate_competition_profile(
                        self._validated_profile_row(session, profile_row)
                    )
                    != snapshot.competition_profile_digest
                ):
                    return False
                jury_row = session.get(FinalJuryReportRecord, snapshot.jury_report_id)
                check_row = session.get(SubmissionCheckRecord, snapshot.submission_check_id)
                if jury_row is None or check_row is None:
                    return False
                jury = FinalJuryReport.model_validate(jury_row.report_json)
                check = SubmissionCheckResult.model_validate(check_row.check_json)
                if (
                    final_jury_report_digest(jury) != snapshot.jury_report_digest
                    or jury_row.report_digest != snapshot.jury_report_digest
                    or submission_check_digest(check) != snapshot.submission_check_digest
                    or check_row.check_digest != snapshot.submission_check_digest
                    or check.jury_report_id != jury.report_id
                    or check.requirement_snapshot_digest != snapshot.requirement_snapshot_digest
                    or requirement_snapshot_digest(check.requirement_coverage)
                    != snapshot.requirement_snapshot_digest
                ):
                    return False
                subproblems = (
                    state.problem_analysis.subproblems
                    if state.problem_analysis is not None
                    else state.subproblems
                )
                expected_requirements = RequirementRegistry.from_subproblems(subproblems)
                expected_digests = {
                    item.requirement_id: requirement_digest(item) for item in expected_requirements
                }
                actual_digests = {
                    item.requirement_id: item.requirement_digest
                    for item in check.requirement_coverage
                }
                return expected_digests == actual_digests
        except (KeyError, TypeError, ValueError):
            return False

    def list_artifacts(self, submission_id: UUID) -> list[SubmissionArtifact]:
        with session_scope(self._session_factory) as session:
            rows = list(
                session.scalars(
                    select(SubmissionArtifactRecordModel)
                    .where(SubmissionArtifactRecordModel.submission_id == submission_id)
                    .order_by(SubmissionArtifactRecordModel.relative_path)
                )
            )
            if not rows:
                raise ResourceNotFoundError("submission artifacts were not found")
            artifacts = [SubmissionArtifact.model_validate(item.artifact_json) for item in rows]
            for row, artifact in zip(rows, artifacts, strict=True):
                if (
                    row.role != artifact.role.value
                    or row.relative_path != artifact.relative_path
                    or row.mime_type != artifact.mime_type
                    or row.size_bytes != artifact.size_bytes
                    or row.sha256 != artifact.sha256
                    or row.storage_key != artifact.storage_key
                ):
                    raise ValueError("submission artifact record failed integrity validation")
            return artifacts

    def persist_correction(
        self, plan: CorrectionPlan, *, submission_check_id: UUID | None = None
    ) -> None:
        with session_scope(self._session_factory) as session:
            existing = session.get(CorrectionPlanRecord, plan.correction_id)
            payload = plan.model_dump(mode="json")
            if existing is not None:
                if existing.plan_json != payload:
                    raise ValueError("immutable correction plan has conflicting content")
                return
            session.add(
                CorrectionPlanRecord(
                    id=plan.correction_id,
                    project_id=plan.project_id,
                    submission_check_id=submission_check_id,
                    scope=plan.scope.value,
                    priority=plan.priority,
                    plan_json=payload,
                )
            )

    def list_corrections(self, project_id: UUID) -> list[CorrectionPlan]:
        with session_scope(self._session_factory) as session:
            rows = list(
                session.scalars(
                    select(CorrectionPlanRecord)
                    .where(CorrectionPlanRecord.project_id == project_id)
                    .order_by(CorrectionPlanRecord.priority, CorrectionPlanRecord.id)
                )
            )
            return [CorrectionPlan.model_validate(item.plan_json) for item in rows]

    @staticmethod
    def _persist_package(
        session: Session,
        *,
        package: SubmissionPackage,
        profile_row: CompetitionProfileRecord,
        paper_row: PaperVersionRecord,
        check: SubmissionCheckResult,
    ) -> None:
        snapshot = package.snapshot
        session.add(
            SubmissionSnapshotRecord(
                id=snapshot.submission_id,
                project_id=snapshot.project_id,
                paper_version_record_id=paper_row.id,
                verified_result_id=snapshot.verified_result_id,
                profile_record_id=profile_row.id,
                submission_check_id=check.check_id,
                status=snapshot.status.value,
                manifest_hash=snapshot.manifest_hash,
                package_hash=snapshot.package_hash,
                artifact_set_digest=snapshot.artifact_set_digest,
                snapshot_digest=snapshot.snapshot_digest,
                snapshot_json=snapshot.model_dump(mode="json"),
                created_at=snapshot.created_at,
            )
        )
        artifacts = [
            *package.protected_artifacts,
            package.manifest_artifact,
            package.package_artifact,
        ]
        for artifact in artifacts:
            session.add(
                SubmissionArtifactRecordModel(
                    id=uuid4(),
                    project_id=snapshot.project_id,
                    submission_id=snapshot.submission_id,
                    role=artifact.role.value,
                    relative_path=artifact.relative_path,
                    mime_type=artifact.mime_type,
                    size_bytes=artifact.size_bytes,
                    sha256=artifact.sha256,
                    storage_key=artifact.storage_key,
                    artifact_json=artifact.model_dump(mode="json"),
                )
            )
        session.add(
            SubmissionManifestRecord(
                submission_id=snapshot.submission_id,
                manifest_hash=package.manifest.manifest_hash,
                package_hash=package.manifest.package_hash,
                manifest_json=package.manifest.model_dump(mode="json"),
            )
        )

    @staticmethod
    def _profile_row(
        session: Session, profile_id: UUID, version: int
    ) -> CompetitionProfileRecord | None:
        return session.scalar(
            select(CompetitionProfileRecord).where(
                CompetitionProfileRecord.profile_id == profile_id,
                CompetitionProfileRecord.version == version,
            )
        )

    @staticmethod
    def _validated_profile_row(
        session: Session, row: CompetitionProfileRecord
    ) -> CompetitionProfile:
        profile = CompetitionProfile.model_validate(row.profile_json)
        digest = validate_competition_profile(profile)
        if (
            row.profile_id != profile.profile_id
            or row.version != profile.version
            or row.name != profile.name
            or row.verification_status != profile.verification_status.value
            or row.profile_digest != digest
        ):
            raise ValueError("COMPETITION_PROFILE_INTEGRITY_ERROR: persisted record changed")
        rule_rows = list(
            session.scalars(
                select(CompetitionRuleRecord)
                .where(CompetitionRuleRecord.profile_record_id == row.id)
                .order_by(CompetitionRuleRecord.rule_id)
            )
        )
        expected = sorted(
            (item.model_dump(mode="json") for item in profile.rules),
            key=lambda item: str(item["rule_id"]),
        )
        actual = sorted(
            (item.rule_json for item in rule_rows), key=lambda item: str(item["rule_id"])
        )
        if actual != expected:
            raise ValueError(
                "COMPETITION_PROFILE_INTEGRITY_ERROR: rule records disagree with profile"
            )
        return profile

    @staticmethod
    def _lock_current(session: Session, project_id: UUID) -> ProblemStateRecord:
        row = session.scalar(
            select(ProblemStateRecord)
            .join(Problem, Problem.id == ProblemStateRecord.problem_id)
            .where(Problem.project_id == project_id, ProblemStateRecord.is_current.is_(True))
            .with_for_update()
        )
        if row is None:
            raise ResourceNotFoundError(f"project {project_id} was not found")
        return row
