from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from mathmodel_ai.schemas.data import (
    CrossDatasetRelationship,
    DataProfile,
    DatasetRecord,
    DataUnderstanding,
)
from mathmodel_ai.schemas.execution import ExecutionRecord
from mathmodel_ai.schemas.files import ArtifactRecord, RegisteredFile
from mathmodel_ai.schemas.mathematical import MathematicalModelRef
from mathmodel_ai.schemas.model_selection import (
    ModelCandidate,
    ModelDecisionEvidence,
    ModelScore,
    ModelSelection,
)
from mathmodel_ai.schemas.paper import PaperQualityStatus, PaperVersionRef
from mathmodel_ai.schemas.problem_analysis import (
    Ambiguity,
    EvidenceItem,
    EvidenceType,
    ProblemAnalysis,
    SubProblem,
)
from mathmodel_ai.schemas.program import GeneratedProgramRef
from mathmodel_ai.schemas.quality import (
    DataStageHistoryEntry,
    DataWorkflowStage,
    QualityGateResult,
    StageHistoryEntry,
)
from mathmodel_ai.schemas.results import ResultRecordRef
from mathmodel_ai.schemas.solver import AlgorithmPlan, SolverRunRef
from mathmodel_ai.schemas.submission import SubmissionStatus, SubmissionSummaryRef
from mathmodel_ai.schemas.subproblem_identity import SubproblemIdentityResolution
from mathmodel_ai.schemas.verification import (
    RedTeamReportRef,
    RepairCycleRef,
    RobustnessReportRef,
    SensitivityReportRef,
    ValidationReportRef,
)

EvidenceKind = EvidenceType


class VerificationStatus(StrEnum):
    UNVERIFIED = "UNVERIFIED"
    VERIFIED = "VERIFIED"
    REJECTED = "REJECTED"


class WorkflowStage(StrEnum):
    INGEST = "INGEST"
    UNDERSTAND = "UNDERSTAND"
    DATA = "DATA"
    LITERATURE = "LITERATURE"
    EXPLORE = "EXPLORE"
    SELECT = "SELECT"
    MODEL = "MODEL"
    SOLVE = "SOLVE"
    VALIDATE = "VALIDATE"
    SENSITIVITY = "SENSITIVITY"
    ROBUSTNESS = "ROBUSTNESS"
    RED_TEAM = "RED_TEAM"
    MODEL_REPAIR = "MODEL_REPAIR"
    PAPER = "PAPER"
    FINAL_JURY = "FINAL_JURY"
    SUBMISSION = "SUBMISSION"
    FINAL = "FINAL"


class WorkflowStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    RETRY = "RETRY"
    ESCALATED = "ESCALATED"
    HUMAN_REVIEW = "HUMAN_REVIEW"


class ProjectSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: UUID
    name: str = Field(min_length=1, max_length=255)
    title: str = Field(min_length=1, max_length=500)
    current_stage: WorkflowStage
    status: WorkflowStatus
    version: int = Field(ge=0)
    created_at: datetime
    updated_at: datetime


class TraceableItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str = Field(min_length=1)
    kind: EvidenceType
    statement: str = Field(min_length=1)
    source_refs: list[str] = Field(default_factory=list)
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED
    metadata: dict[str, Any] = Field(default_factory=dict)


class ArtifactRef(BaseModel):
    artifact_id: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    path: str = Field(min_length=1)
    content_hash: str | None = None
    is_original: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelRef(BaseModel):
    model_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    score: float | None = None
    rationale: str | None = None


class SymbolRef(BaseModel):
    symbol_id: str = Field(min_length=1)
    symbol: str = Field(min_length=1)
    meaning: str = Field(min_length=1)
    unit: str | None = None


class EquationRef(BaseModel):
    equation_id: str = Field(pattern=r"^EQ-[A-Za-z0-9_-]+$")
    latex: str = Field(min_length=1)
    meaning: str = Field(min_length=1)
    source_refs: list[str] = Field(default_factory=list)
    variables: list[str] = Field(default_factory=list)
    parameters: list[str] = Field(default_factory=list)
    units: list[str] = Field(default_factory=list)


class ExecutionRecordRef(BaseModel):
    run_id: UUID
    code_hash: str = Field(min_length=1)
    exit_code: int
    artifact_ids: list[str] = Field(default_factory=list)
    is_mock: bool = False


class ReportRef(BaseModel):
    report_id: str = Field(min_length=1)
    status: VerificationStatus = VerificationStatus.UNVERIFIED
    artifact_id: str | None = None
    summary: str | None = None


class ProblemState(BaseModel):
    """Canonical shared state; agents may not maintain competing private facts."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    schema_version: int = Field(default=5, ge=1)
    version: int = Field(default=0, ge=0)
    problem_id: UUID = Field(default_factory=uuid4)
    project_id: UUID
    title: str = Field(min_length=1)
    raw_problem: str = Field(min_length=1)

    competition: str | None = None
    deadline: datetime | None = None
    remaining_hours: float | None = Field(default=None, ge=0)

    files: list[ArtifactRef] = Field(default_factory=list)
    registered_files: list[RegisteredFile] = Field(default_factory=list)
    tracked_artifacts: list[ArtifactRecord] = Field(default_factory=list)
    background: list[TraceableItem] = Field(default_factory=list)
    objectives: list[TraceableItem] = Field(default_factory=list)
    subproblems: list[SubProblem] = Field(default_factory=list)

    facts: list[TraceableItem] = Field(default_factory=list)
    data_sources: list[TraceableItem] = Field(default_factory=list)
    assumptions: list[TraceableItem] = Field(default_factory=list)
    ambiguities: list[Ambiguity] = Field(default_factory=list)
    constraints: list[TraceableItem] = Field(default_factory=list)

    problem_analysis: ProblemAnalysis | None = None
    subproblem_identity: SubproblemIdentityResolution | None = None
    evidence_items: list[EvidenceItem] = Field(default_factory=list)
    proposed_assumptions: list[EvidenceItem] = Field(default_factory=list)

    candidate_models: list[ModelCandidate] = Field(default_factory=list)
    model_scores: list[ModelScore] = Field(default_factory=list)
    selected_model: ModelCandidate | None = None
    backup_model: ModelCandidate | None = None
    model_selection: ModelSelection | None = None
    model_selection_version: str | None = None
    decision_evidence: list[ModelDecisionEvidence] = Field(default_factory=list)

    datasets: list[DatasetRecord] = Field(default_factory=list)
    data_profiles: list[DataProfile] = Field(default_factory=list)
    cross_dataset_relationships: list[CrossDatasetRelationship] = Field(default_factory=list)
    data_understanding: DataUnderstanding | None = None
    data_stage: DataWorkflowStage = DataWorkflowStage.PENDING
    data_stage_history: list[DataStageHistoryEntry] = Field(default_factory=list)
    quality_gates: list[QualityGateResult] = Field(default_factory=list)
    stage_history: list[StageHistoryEntry] = Field(default_factory=list)

    variables: list[SymbolRef] = Field(default_factory=list)
    parameters: list[SymbolRef] = Field(default_factory=list)
    units: dict[str, str] = Field(default_factory=dict)
    equations: list[EquationRef] = Field(default_factory=list)

    objective: TraceableItem | None = None
    model_constraints: list[TraceableItem] = Field(default_factory=list)
    algorithm: TraceableItem | None = None

    mathematical_model: MathematicalModelRef | None = None
    algorithm_plan: AlgorithmPlan | None = None
    generated_programs: list[GeneratedProgramRef] = Field(default_factory=list)
    solver_runs: list[SolverRunRef] = Field(default_factory=list)
    result_records: list[ResultRecordRef] = Field(default_factory=list)
    verified_result_id: UUID | None = None

    code_files: list[ArtifactRef] = Field(default_factory=list)
    execution_records: list[ExecutionRecord] = Field(default_factory=list)
    results: list[TraceableItem] = Field(default_factory=list)

    validation_results: list[ValidationReportRef] = Field(default_factory=list)
    sensitivity_results: list[SensitivityReportRef] = Field(default_factory=list)
    robustness_results: list[RobustnessReportRef] = Field(default_factory=list)

    literature: list[TraceableItem] = Field(default_factory=list)
    citations: list[TraceableItem] = Field(default_factory=list)
    red_team_reports: list[RedTeamReportRef] = Field(default_factory=list)
    revisions: list[RepairCycleRef] = Field(default_factory=list)

    figures: list[ArtifactRef] = Field(default_factory=list)
    tables: list[ArtifactRef] = Field(default_factory=list)
    paper_state: dict[str, Any] = Field(default_factory=dict)
    paper_versions: list[PaperVersionRef] = Field(default_factory=list)
    submission_state: dict[str, Any] = Field(default_factory=dict)
    submission_snapshots: list[SubmissionSummaryRef] = Field(default_factory=list)

    current_stage: WorkflowStage = WorkflowStage.INGEST
    status: WorkflowStatus = WorkflowStatus.PENDING
    updated_by: str = "system"
    update_reason: str = "initial problem ingestion"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("deadline")
    @classmethod
    def deadline_must_be_timezone_aware(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("deadline must include a timezone")
        return value

    @model_validator(mode="after")
    def enforce_evidence_categories(self) -> "ProblemState":
        if self.subproblem_identity is not None:
            canonical = [item.canonical_subproblem_id for item in self.subproblem_identity.bindings]
            if [item.subproblem_id for item in self.subproblems] != canonical:
                raise ValueError(
                    "state subproblems must match the resolved canonical identity order"
                )
            if (
                self.problem_analysis is None
                or [item.subproblem_id for item in self.problem_analysis.subproblems] != canonical
            ):
                raise ValueError(
                    "problem analysis must match the resolved canonical identity order"
                )
        expected = {
            "facts": EvidenceType.FACT,
            "data_sources": EvidenceType.DATA,
            "assumptions": EvidenceType.ASSUMPTION,
            "results": EvidenceType.RESULT,
            "literature": EvidenceType.EXTERNAL_EVIDENCE,
            "citations": EvidenceType.EXTERNAL_EVIDENCE,
        }
        for field_name, kind in expected.items():
            items: list[TraceableItem] = getattr(self, field_name)
            if any(item.kind is not kind for item in items):
                raise ValueError(f"{field_name} entries must have kind={kind.value}")
        if self.schema_version >= 5:
            verified_items = [
                item
                for item in self.results
                if item.verification_status is VerificationStatus.VERIFIED
            ]
            if self.verified_result_id is None and verified_items:
                raise ValueError("VERIFIED result entries require verified_result_id")
            if self.verified_result_id is not None:
                if not any(
                    item.result_id == self.verified_result_id for item in self.result_records
                ):
                    raise ValueError("verified_result_id must reference a persisted result")
                expected_item_id = f"RESULT-{self.verified_result_id}"
                if [item.item_id for item in verified_items] != [expected_item_id]:
                    raise ValueError("exactly the verified_result_id result entry may be VERIFIED")
                if not any(
                    gate.gate == "VERIFIED"
                    and gate.status.value == "PASS"
                    and gate.subject_ref == f"result:{self.verified_result_id}"
                    for gate in self.quality_gates
                ):
                    raise ValueError("verified_result_id requires a passing VERIFIED gate")
        if self.schema_version >= 6:
            identities = [(item.paper_id, item.version) for item in self.paper_versions]
            if len(identities) != len(set(identities)):
                raise ValueError("paper version identities must be unique")
            ready = [
                item
                for item in self.paper_versions
                if item.status is PaperQualityStatus.READY_FOR_FINAL_JURY
            ]
            if ready and self.verified_result_id is None:
                raise ValueError("ready paper requires verified_result_id")
            if any(item.verified_result_id != self.verified_result_id for item in ready):
                raise ValueError("ready paper must use the explicit verified_result_id")
        if self.schema_version >= 7:
            submission_ids = [item.submission_id for item in self.submission_snapshots]
            if len(submission_ids) != len(set(submission_ids)):
                raise ValueError("submission snapshot identifiers must be unique")
            frozen = [
                item for item in self.submission_snapshots if item.status is SubmissionStatus.FROZEN
            ]
            ready_papers = {
                (item.paper_id, item.version)
                for item in self.paper_versions
                if item.status is PaperQualityStatus.READY_FOR_FINAL_JURY
            }
            if any(item.verified_result_id != self.verified_result_id for item in frozen):
                raise ValueError("frozen submission must use the explicit verified_result_id")
            if any((item.paper_id, item.paper_version) not in ready_papers for item in frozen):
                raise ValueError("frozen submission must bind a ready paper version")
            if any(item.manifest_hash is None or item.package_hash is None for item in frozen):
                raise ValueError("frozen submission requires manifest and package hashes")
        return self
