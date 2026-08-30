from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from io import BytesIO
from uuid import UUID

from pydantic import ValidationError
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from mathmodel_ai.agents.base import AgentRunResult, AgentRunStatus
from mathmodel_ai.agents.final_jury import FinalJuryAgent
from mathmodel_ai.files.storage import FileStore
from mathmodel_ai.paper.bundle import PaperBundleBuilder
from mathmodel_ai.paper.hashing import sha256_bytes, sha256_json
from mathmodel_ai.paper.repository import PaperRepository
from mathmodel_ai.reasoning.repository import ReasoningRepository
from mathmodel_ai.reasoning.state_machine import ensure_transition
from mathmodel_ai.routing.schemas import EscalationLevel, TaskProfile, TaskType
from mathmodel_ai.schemas.paper import (
    PaperArtifact,
    PaperArtifactKind,
    PaperManifest,
    PaperQualityStatus,
    PaperVersion,
)
from mathmodel_ai.schemas.problem_state import ProblemState, WorkflowStage, WorkflowStatus
from mathmodel_ai.schemas.submission import (
    CompetitionProfile,
    FinalJuryDraft,
    FinalJuryInput,
    FinalJuryReport,
    JuryDecision,
    RequirementCoverage,
    RuleResult,
    SubmissionArtifact,
    SubmissionArtifactRole,
    SubmissionCandidate,
    SubmissionCheckResult,
    SubmissionCheckStatus,
    SubmissionStatus,
    SubmissionSummaryRef,
)
from mathmodel_ai.submission.checks import SubmissionCheck
from mathmodel_ai.submission.freeze import SubmissionFreeze
from mathmodel_ai.submission.integrity import validate_competition_profile
from mathmodel_ai.submission.jury import FinalJuryGate
from mathmodel_ai.submission.package import SubmissionPackage, SubmissionPackageBuilder
from mathmodel_ai.submission.profiles import CompetitionProfileRegistry
from mathmodel_ai.submission.repository import SubmissionRepository
from mathmodel_ai.submission.requirements import RequirementCoverageValidator, RequirementRegistry
from mathmodel_ai.submission.rules import RuleEngine


class FinalSubmissionError(ValueError):
    pass


@dataclass(frozen=True)
class FinalWorkflowOutcome:
    profile: CompetitionProfile
    candidate: SubmissionCandidate
    requirements: list[RequirementCoverage]
    rule_results: list[RuleResult]
    jury: FinalJuryReport
    check: SubmissionCheckResult
    package: SubmissionPackage | None
    state: ProblemState
    agent_run: AgentRunResult[FinalJuryDraft]


class FinalSubmissionWorkflow:
    def __init__(
        self,
        *,
        reasoning_repository: ReasoningRepository,
        paper_repository: PaperRepository,
        repository: SubmissionRepository,
        profiles: CompetitionProfileRegistry,
        jury_agent: FinalJuryAgent,
        rule_engine: RuleEngine,
        package_builder: SubmissionPackageBuilder,
        store: FileStore,
    ) -> None:
        self._reasoning = reasoning_repository
        self._papers = paper_repository
        self._repository = repository
        self._profiles = profiles
        self._jury_agent = jury_agent
        self._rules = rule_engine
        self._requirements = RequirementCoverageValidator()
        self._jury_gate = FinalJuryGate()
        self._submission_check = SubmissionCheck()
        self._freeze = SubmissionFreeze(package_builder, rule_engine)
        self._store = store

    async def run(
        self,
        project_id: UUID,
        *,
        profile_id: UUID,
        profile_version: int,
        paper_id: UUID,
        paper_version: int,
        freeze_on_pass: bool = False,
    ) -> FinalWorkflowOutcome:
        state = self._reasoning.load_current(project_id)
        if state.current_stage not in {
            WorkflowStage.PAPER,
            WorkflowStage.FINAL_JURY,
            WorkflowStage.SUBMISSION,
            WorkflowStage.FINAL,
        }:
            raise FinalSubmissionError("Phase 7 requires an accepted PAPER or Phase 7 state")
        profile = self._profiles.get(profile_id, profile_version)
        self._repository.persist_profile(profile)
        paper = self._papers.get_version(project_id, paper_id, paper_version)
        self._validate_explicit_paper(state, paper)
        paper_artifacts = self._papers.list_version_artifacts(project_id, paper_id, paper_version)
        candidate, artifact_bytes = self._candidate(
            state=state,
            paper=paper,
            artifacts=paper_artifacts,
            profile=profile,
        )
        subproblems = (
            state.problem_analysis.subproblems
            if state.problem_analysis is not None
            else state.subproblems
        )
        if not subproblems:
            raise FinalSubmissionError("final review requires structured problem requirements")
        requirement_records = RequirementRegistry.from_subproblems(subproblems)
        coverage = self._requirements.validate(
            requirement_records, paper.paper_ir, candidate.artifacts
        )
        rule_results = self._rules.evaluate(
            profile,
            candidate,
            artifact_bytes=artifact_bytes,
        )
        jury_run = await self._jury_agent.run(
            FinalJuryInput(
                profile=profile,
                paper=paper.paper_ir,
                candidate=candidate,
                requirements=coverage,
                rule_results=rule_results,
            ),
            state,
            TaskProfile(
                task_type=TaskType.FINAL_ACCEPTANCE,
                complexity=5,
                reasoning_requirement=5,
                review_requirement=5,
                security_risk=4,
                blast_radius=5,
                minimum_level=EscalationLevel.FLAGSHIP_XHIGH,
            ),
        )
        draft = self._require_jury_output(jury_run)
        self._reasoning.record_run(project_id, state.problem_id, jury_run)
        jury = self._jury_gate.build_report(
            draft=draft,
            profile=profile,
            candidate=candidate,
            requirements=coverage,
            rule_results=rule_results,
            reviewer_is_mock=jury_run.is_mock,
            agent_run_id=jury_run.run_id,
        )
        check = self._submission_check.run(
            profile=profile,
            candidate=candidate,
            requirements=coverage,
            rules=rule_results,
            jury=jury,
            artifact_bytes=artifact_bytes,
        )
        package = None
        if freeze_on_pass:
            if (
                jury.decision is not JuryDecision.PASS
                or check.status is not SubmissionCheckStatus.PASS
            ):
                issue_codes = ",".join(item.code for item in check.issues) or "none"
                failed_rules = (
                    ",".join(
                        f"{item.rule_id}:{item.status.value}"
                        for item in rule_results
                        if item.status.value != "PASS"
                    )
                    or "none"
                )
                raise FinalSubmissionError(
                    "submission cannot freeze until jury and checks pass: "
                    f"jury={jury.decision.value}; check={check.status.value}; "
                    f"issues={issue_codes}; rules={failed_rules}"
                )
            package = self._freeze.freeze(
                profile=profile,
                candidate=candidate,
                paper=paper.paper_ir,
                submission_requirements=requirement_records,
                requirements=coverage,
                rules=rule_results,
                jury=jury,
                check=check,
                artifact_bytes=artifact_bytes,
            )
        states = self._state_history(state, jury, check, package)
        self._repository.persist_run(
            profile=profile,
            coverage=coverage,
            jury=jury,
            check=check,
            package=package,
            states=states,
        )
        return FinalWorkflowOutcome(
            profile=profile,
            candidate=candidate,
            requirements=coverage,
            rule_results=rule_results,
            jury=jury,
            check=check,
            package=package,
            state=states[-1],
            agent_run=jury_run,
        )

    @staticmethod
    def _validate_explicit_paper(state: ProblemState, paper: PaperVersion) -> None:
        if state.verified_result_id is None:
            raise FinalSubmissionError("Phase 7 requires an explicit verified_result_id")
        matching = [
            item
            for item in state.paper_versions
            if (item.paper_id, item.version) == (paper.paper_id, paper.version)
        ]
        if len(matching) != 1:
            raise FinalSubmissionError("explicit paper version is not present in ProblemState")
        reference = matching[0]
        if (
            reference.status is not PaperQualityStatus.READY_FOR_FINAL_JURY
            or paper.status is not PaperQualityStatus.READY_FOR_FINAL_JURY
        ):
            raise FinalSubmissionError("paper version is not READY_FOR_FINAL_JURY")
        if (
            reference.verified_result_id != state.verified_result_id
            or paper.evidence_snapshot.verified_result_id != state.verified_result_id
        ):
            raise FinalSubmissionError("paper does not bind the official verified result")
        if reference.manifest_hash != paper.manifest_hash or paper.manifest_hash is None:
            raise FinalSubmissionError("paper manifest identity is absent or changed")

    def _candidate(
        self,
        *,
        state: ProblemState,
        paper: PaperVersion,
        artifacts: list[PaperArtifact],
        profile: CompetitionProfile,
    ) -> tuple[SubmissionCandidate, dict[UUID, bytes]]:
        all_bytes = {
            item.artifact_id: self._store.read_bytes(item.storage_key) for item in artifacts
        }
        for artifact in artifacts:
            data = all_bytes[artifact.artifact_id]
            if len(data) != artifact.size_bytes or sha256_bytes(data) != artifact.sha256:
                raise FinalSubmissionError("Phase 6 artifact bytes changed after acceptance")
        manifest_artifacts = [item for item in artifacts if item.kind is PaperArtifactKind.MANIFEST]
        if len(manifest_artifacts) != 1:
            raise FinalSubmissionError("approved paper must have one exact Phase 6 manifest")
        try:
            paper_manifest = PaperManifest.model_validate_json(
                all_bytes[manifest_artifacts[0].artifact_id]
            )
        except (ValidationError, ValueError) as exc:
            raise FinalSubmissionError("Phase 6 manifest is invalid") from exc
        if (
            not PaperBundleBuilder.verify_manifest(paper_manifest)
            or paper_manifest.manifest_hash != paper.manifest_hash
            or paper_manifest.paper_id != paper.paper_id
            or paper_manifest.paper_version != paper.version
            or paper_manifest.verified_result_id != paper.evidence_snapshot.verified_result_id
            or paper_manifest.verified_model_version
            != paper.evidence_snapshot.verified_model_version
            or paper_manifest.verified_model_digest != paper.evidence_snapshot.verified_model_digest
            or paper_manifest.evidence_snapshot_hash != paper.evidence_snapshot.snapshot_hash
            or paper_manifest.paper_ir_hash
            != sha256_json(paper.paper_ir.model_dump(mode="json", exclude={"status"}))
            or paper_manifest.claim_set_hash
            != sha256_json([item.model_dump(mode="json") for item in paper.paper_ir.claims])
            or paper_manifest.document_registry_hash
            != sha256_json(
                [item.model_dump(mode="json") for item in paper.paper_ir.document_registry]
            )
        ):
            raise FinalSubmissionError("Phase 6 manifest does not bind the approved paper")
        by_kind = {item.kind: item for item in artifacts}
        required_artifacts = {
            PaperArtifactKind.TEX: (
                paper_manifest.tex_artifact_id,
                paper_manifest.tex_hash,
            ),
            PaperArtifactKind.BIB: (
                paper_manifest.bib_artifact_id,
                paper_manifest.bib_hash,
            ),
            PaperArtifactKind.PDF: (
                paper_manifest.pdf_artifact_id,
                paper_manifest.pdf_hash,
            ),
        }
        if any(
            kind not in by_kind
            or by_kind[kind].artifact_id != expected_id
            or by_kind[kind].sha256 != expected_hash
            for kind, (expected_id, expected_hash) in required_artifacts.items()
        ):
            raise FinalSubmissionError("Phase 6 manifest artifact bindings changed")
        selected = [
            converted
            for artifact in artifacts
            if (converted := self._convert_artifact(artifact)) is not None
        ]
        pdf = [item for item in selected if item.role is SubmissionArtifactRole.PAPER_PDF]
        if len(pdf) != 1:
            raise FinalSubmissionError("approved paper requires one PDF artifact")
        pdf_bytes = all_bytes[pdf[0].artifact_id]
        try:
            reader = PdfReader(BytesIO(pdf_bytes), strict=True)
            page_count = len(reader.pages)
            pdf_text = "\n".join(page.extract_text() or "" for page in reader.pages)
            metadata = {str(key): str(value) for key, value in (reader.metadata or {}).items()}
        except (PdfReadError, ValueError, TypeError, KeyError) as exc:
            raise FinalSubmissionError("approved paper PDF failed strict parsing") from exc
        compile_record = self._papers.get_compile(state.project_id, paper.paper_id, paper.version)
        if page_count < 1 or compile_record.page_count != page_count:
            raise FinalSubmissionError("paper PDF page count changed after compilation")
        candidate = SubmissionCandidate(
            project_id=state.project_id,
            problem_id=state.problem_id,
            competition_profile_id=profile.profile_id,
            competition_profile_version=profile.version,
            competition_profile_digest=validate_competition_profile(profile),
            paper_id=paper.paper_id,
            paper_version=paper.version,
            paper_status=paper.status,
            paper_manifest_hash=paper.manifest_hash,
            paper_ir_hash=paper_manifest.paper_ir_hash,
            verified_model_id=paper.evidence_snapshot.verified_model_id,
            verified_model_version=paper.evidence_snapshot.verified_model_version,
            verified_model_digest=paper.evidence_snapshot.verified_model_digest,
            verified_result_id=paper.evidence_snapshot.verified_result_id,
            phase5_verified=self._phase5_verified(state),
            page_count=page_count,
            section_types=[
                item.section_type.value
                for item in [*paper.paper_ir.sections, *paper.paper_ir.appendices]
            ],
            paper_text="\n".join([self._paper_ir_text(paper), pdf_text]),
            paper_metadata=metadata,
            artifacts=selected,
        )
        selected_bytes = {item.artifact_id: all_bytes[item.artifact_id] for item in selected}
        return candidate, selected_bytes

    @staticmethod
    def _convert_artifact(artifact: PaperArtifact) -> SubmissionArtifact | None:
        mapping = {
            PaperArtifactKind.PDF: (SubmissionArtifactRole.PAPER_PDF, "paper.pdf"),
            PaperArtifactKind.TEX: (SubmissionArtifactRole.PAPER_TEX, "paper.tex"),
            PaperArtifactKind.BIB: (SubmissionArtifactRole.REFERENCES, "references.bib"),
            PaperArtifactKind.FIGURE_IMAGE: (
                SubmissionArtifactRole.FIGURE,
                f"figures/{artifact.name}",
            ),
            PaperArtifactKind.TABLE_DATA: (
                SubmissionArtifactRole.TABLE,
                f"tables/{artifact.name}",
            ),
        }
        mapped = mapping.get(artifact.kind)
        if mapped is None:
            return None
        role, path = mapped
        return SubmissionArtifact(
            artifact_id=artifact.artifact_id,
            project_id=artifact.project_id,
            role=role,
            relative_path=path,
            mime_type=artifact.mime_type,
            size_bytes=artifact.size_bytes,
            sha256=artifact.sha256,
            storage_key=artifact.storage_key,
            source_artifact_id=artifact.artifact_id,
            paper_id=artifact.paper_id,
            paper_version=artifact.paper_version,
        )

    @staticmethod
    def _phase5_verified(state: ProblemState) -> bool:
        result_id = state.verified_result_id
        if result_id is None:
            return False
        return any(
            gate.gate == "VERIFIED"
            and gate.status.value == "PASS"
            and gate.subject_ref == f"result:{result_id}"
            for gate in state.quality_gates
        )

    @staticmethod
    def _paper_ir_text(paper: PaperVersion) -> str:
        ir = paper.paper_ir
        text = [ir.title, *(claim.text for claim in ir.claims)]
        for block in [
            *ir.abstract,
            *(block for section in [*ir.sections, *ir.appendices] for block in section.blocks),
        ]:
            if block.text is not None:
                text.append(block.text)
        return "\n".join(text)

    @staticmethod
    def _require_jury_output(run: AgentRunResult[FinalJuryDraft]) -> FinalJuryDraft:
        if run.status is not AgentRunStatus.SUCCEEDED or run.output is None:
            raise FinalSubmissionError("FinalJuryAgent did not produce a structured report")
        return run.output

    @staticmethod
    def _state_history(
        current: ProblemState,
        jury: FinalJuryReport,
        check: SubmissionCheckResult,
        package: SubmissionPackage | None,
    ) -> list[ProblemState]:
        states: list[ProblemState] = []
        base = current
        if current.current_stage is not WorkflowStage.PAPER:
            ensure_transition(current.current_stage, WorkflowStage.PAPER)
            base = current.model_copy(
                update={
                    "version": current.version + 1,
                    "current_stage": WorkflowStage.PAPER,
                    "status": WorkflowStatus.RUNNING,
                    "submission_state": {
                        **current.submission_state,
                        "jury_decision": JuryDecision.RECHECK_REQUIRED.value,
                        "status": SubmissionStatus.DIRTY.value,
                    },
                    "updated_by": "final_submission_workflow",
                    "update_reason": "Phase 7 recheck invalidated current submission readiness",
                    "updated_at": datetime.now(UTC),
                }
            )
            states.append(base)
        ensure_transition(base.current_stage, WorkflowStage.FINAL_JURY)
        jury_status = (
            WorkflowStatus.SUCCEEDED
            if jury.decision is JuryDecision.PASS
            else WorkflowStatus.HUMAN_REVIEW
            if jury.decision is JuryDecision.HUMAN_REVIEW
            else WorkflowStatus.FAILED
        )
        jury_state = base.model_copy(
            update={
                "schema_version": 7,
                "version": base.version + 1,
                "current_stage": WorkflowStage.FINAL_JURY,
                "status": jury_status,
                "submission_state": {
                    "jury_report_id": str(jury.report_id),
                    "jury_decision": jury.decision.value,
                    "jury_score": jury.claimed_score,
                    "status": SubmissionStatus.CHECKING.value,
                },
                "updated_by": "final_submission_workflow",
                "update_reason": "Phase 7 deterministic Final Jury completed",
                "updated_at": datetime.now(UTC),
            }
        )
        ensure_transition(jury_state.current_stage, WorkflowStage.SUBMISSION)
        check_status = (
            WorkflowStatus.SUCCEEDED
            if check.status is SubmissionCheckStatus.PASS
            else WorkflowStatus.HUMAN_REVIEW
            if check.status is SubmissionCheckStatus.HUMAN_REVIEW
            else WorkflowStatus.FAILED
        )
        submission_state = jury_state.model_copy(
            update={
                "version": jury_state.version + 1,
                "current_stage": WorkflowStage.SUBMISSION,
                "status": check_status,
                "submission_state": {
                    **jury_state.submission_state,
                    "submission_check_id": str(check.check_id),
                    "submission_check_status": check.status.value,
                    "status": (
                        SubmissionStatus.READY.value
                        if check.status is SubmissionCheckStatus.PASS
                        else SubmissionStatus.HUMAN_REVIEW.value
                        if check.status is SubmissionCheckStatus.HUMAN_REVIEW
                        else SubmissionStatus.FAILED.value
                    ),
                },
                "updated_by": "final_submission_workflow",
                "update_reason": "Phase 7 deterministic submission check completed",
                "updated_at": datetime.now(UTC),
            }
        )
        states.extend([jury_state, submission_state])
        if package is not None:
            ensure_transition(submission_state.current_stage, WorkflowStage.FINAL)
            snapshot = package.snapshot
            final_state = submission_state.model_copy(
                update={
                    "version": submission_state.version + 1,
                    "current_stage": WorkflowStage.FINAL,
                    "status": WorkflowStatus.SUCCEEDED,
                    "submission_snapshots": [
                        *submission_state.submission_snapshots,
                        SubmissionSummaryRef(
                            submission_id=snapshot.submission_id,
                            paper_id=snapshot.paper_id,
                            paper_version=snapshot.paper_version,
                            verified_result_id=snapshot.verified_result_id,
                            competition_profile_id=snapshot.competition_profile_id,
                            competition_profile_version=snapshot.competition_profile_version,
                            status=snapshot.status,
                            manifest_hash=snapshot.manifest_hash,
                            package_hash=snapshot.package_hash,
                        ),
                    ],
                    "submission_state": {
                        **submission_state.submission_state,
                        "submission_id": str(snapshot.submission_id),
                        "manifest_hash": snapshot.manifest_hash,
                        "package_hash": snapshot.package_hash,
                        "status": snapshot.status.value,
                    },
                    "updated_by": "final_submission_workflow",
                    "update_reason": "Phase 7 submission package frozen and revalidated",
                    "updated_at": datetime.now(UTC),
                }
            )
            states.append(final_state)
        return states
