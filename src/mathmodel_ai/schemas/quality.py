from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from mathmodel_ai.schemas.problem_analysis import Ambiguity, EvidenceItem, SubProblem


class QualityGateStatus(StrEnum):
    PASS = "PASS"
    RETRY = "RETRY"
    ESCALATE = "ESCALATE"
    HUMAN_REVIEW = "HUMAN_REVIEW"


class QualityGateResult(BaseModel):
    gate: str
    status: QualityGateStatus
    checks: dict[str, bool]
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class StageHistoryEntry(BaseModel):
    from_stage: str
    to_stage: str
    status: str
    input_version: int = Field(ge=0)
    output_version: int = Field(ge=1)
    updated_by: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    agent_run_id: UUID | None = None
    execution_run_id: UUID | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def one_run_reference_is_required(self) -> "StageHistoryEntry":
        if (self.agent_run_id is None) == (self.execution_run_id is None):
            raise ValueError("stage history requires exactly one agent or execution run reference")
        return self


class ReasoningStateSummary(BaseModel):
    version: int
    subproblems: list[SubProblem]
    evidence_items: list[EvidenceItem]
    ambiguities: list[Ambiguity]


class DataWorkflowStage(StrEnum):
    PENDING = "PENDING"
    FILES = "FILES"
    DATA = "DATA"
    EXECUTION = "EXECUTION"


class DataStageHistoryEntry(BaseModel):
    from_stage: DataWorkflowStage
    to_stage: DataWorkflowStage
    status: QualityGateStatus
    input_version: int = Field(ge=0)
    output_version: int = Field(ge=1)
    updated_by: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    source_id: UUID | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
