from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mathmodel_ai.schemas.solver import SolverName, SolverStatus


class EvidenceErrorCode(StrEnum):
    OBJECTIVE_MISMATCH = "OBJECTIVE_MISMATCH"
    STATUS_MISMATCH = "STATUS_MISMATCH"
    KEY_OUTPUT_MISMATCH = "KEY_OUTPUT_MISMATCH"
    RESULT_REF_MISMATCH = "RESULT_REF_MISMATCH"
    MODEL_VERSION_MISMATCH = "MODEL_VERSION_MISMATCH"
    MODEL_DIGEST_MISMATCH = "MODEL_DIGEST_MISMATCH"
    CODE_HASH_MISMATCH = "CODE_HASH_MISMATCH"
    EXECUTION_FAILED = "EXECUTION_FAILED"
    MOCK_EXECUTION = "MOCK_EXECUTION"
    MISSING_EXECUTION = "MISSING_EXECUTION"
    MISSING_MODEL = "MISSING_MODEL"
    MISSING_SOLVER_RUN = "MISSING_SOLVER_RUN"
    MISSING_GENERATED_PROGRAM = "MISSING_GENERATED_PROGRAM"
    SOLVER_IDENTITY_MISMATCH = "SOLVER_IDENTITY_MISMATCH"
    BROKEN_RESULT_CHAIN = "BROKEN_RESULT_CHAIN"


class EvidenceIssue(BaseModel):
    code: EvidenceErrorCode
    message: str = Field(min_length=1)
    reference: str | None = None


class ResultRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    result_id: UUID = Field(default_factory=uuid4)
    project_id: UUID
    problem_id: UUID
    model_id: UUID
    model_version: int = Field(ge=1)
    model_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    solver_run_id: UUID
    execution_record_id: UUID
    solver: SolverName
    objective: float | None = Field(default=None, allow_inf_nan=False)
    key_outputs: dict[str, float] = Field(default_factory=dict)
    status: SolverStatus
    evidence_refs: list[str] = Field(min_length=3)
    artifact_refs: list[UUID] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def evidence_refs_cover_required_chain(self) -> ResultRecord:
        required = {
            f"model:{self.model_id}:v{self.model_version}",
            f"solver_run:{self.solver_run_id}",
            f"execution:{self.execution_record_id}",
        }
        if not required <= set(self.evidence_refs):
            raise ValueError("result evidence_refs must cover model, solver run, and execution")
        return self


class ResultRecordRef(BaseModel):
    result_id: UUID
    model_id: UUID
    model_version: int = Field(ge=1)
    model_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    solver_run_id: UUID
    status: SolverStatus


class EvidenceChainReport(BaseModel):
    result_id: UUID
    valid: bool
    model_found: bool
    solver_run_found: bool
    execution_found: bool
    generated_program_found: bool | None = None
    exact_model_version: bool
    model_digest_match: bool
    code_hash_match: bool | None = None
    errors: list[EvidenceIssue] = Field(default_factory=list)

    @property
    def error_codes(self) -> list[EvidenceErrorCode]:
        return [item.code for item in self.errors]
