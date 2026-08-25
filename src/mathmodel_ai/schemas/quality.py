from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, Field

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
    agent_run_id: UUID
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ReasoningStateSummary(BaseModel):
    version: int
    subproblems: list[SubProblem]
    evidence_items: list[EvidenceItem]
    ambiguities: list[Ambiguity]
