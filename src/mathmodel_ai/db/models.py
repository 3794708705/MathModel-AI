from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from mathmodel_ai.db.base import Base

JSON_TYPE = JSON().with_variant(JSONB(), "postgresql")


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    problems: Mapped[list["Problem"]] = relationship(back_populates="project")


class Problem(Base):
    __tablename__ = "problems"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    project_id: Mapped[UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    title: Mapped[str] = mapped_column(String(500))
    raw_problem: Mapped[str] = mapped_column(Text)
    competition: Mapped[str | None] = mapped_column(String(255))
    deadline: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    project: Mapped[Project] = relationship(back_populates="problems")
    states: Mapped[list["ProblemStateRecord"]] = relationship(back_populates="problem")


class ProblemStateRecord(Base):
    __tablename__ = "problem_states"
    __table_args__ = (
        UniqueConstraint("problem_id", "revision", name="uq_problem_states_problem_revision"),
        Index(
            "uq_problem_states_one_current",
            "problem_id",
            unique=True,
            postgresql_where=text("is_current"),
            sqlite_where=text("is_current = 1"),
        ),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    problem_id: Mapped[UUID] = mapped_column(
        ForeignKey("problems.id", ondelete="CASCADE"), index=True
    )
    revision: Mapped[int]
    schema_version: Mapped[int] = mapped_column(default=1)
    current_stage: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(32))
    is_current: Mapped[bool] = mapped_column(default=True)
    state_json: Mapped[dict[str, Any]] = mapped_column("state", JSON_TYPE)
    updated_by: Mapped[str] = mapped_column(String(100), default="system")
    update_reason: Mapped[str] = mapped_column(Text, default="state revision")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    problem: Mapped[Problem] = relationship(back_populates="states")


class AgentRunRecord(Base):
    __tablename__ = "agent_runs"

    id: Mapped[UUID] = mapped_column(primary_key=True)
    project_id: Mapped[UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    problem_id: Mapped[UUID] = mapped_column(
        ForeignKey("problems.id", ondelete="CASCADE"), index=True
    )
    agent_name: Mapped[str] = mapped_column(String(100), index=True)
    input_state_version: Mapped[int]
    output_state_version: Mapped[int | None]
    provider: Mapped[str | None] = mapped_column(String(50))
    model: Mapped[str | None] = mapped_column(String(255))
    reasoning_level: Mapped[str | None] = mapped_column(String(32))
    prompt_version: Mapped[str | None] = mapped_column(String(64))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    latency_ms: Mapped[int]
    token_usage: Mapped[dict[str, Any]] = mapped_column(JSON_TYPE)
    status: Mapped[str] = mapped_column(String(32))
    retry_count: Mapped[int]
    error: Mapped[str | None] = mapped_column(Text)
    is_mock: Mapped[bool] = mapped_column(Boolean, default=False)
    route_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON_TYPE)


class ModelDecisionRecord(Base):
    __tablename__ = "model_decisions"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    project_id: Mapped[UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    problem_id: Mapped[UUID] = mapped_column(
        ForeignKey("problems.id", ondelete="CASCADE"), index=True
    )
    state_version: Mapped[int]
    agent_run_id: Mapped[UUID] = mapped_column(
        ForeignKey("agent_runs.id", ondelete="RESTRICT"), unique=True
    )
    selected_model_id: Mapped[str] = mapped_column(String(100))
    backup_model_id: Mapped[str] = mapped_column(String(100))
    weights_version: Mapped[str] = mapped_column(String(64))
    decision_json: Mapped[dict[str, Any]] = mapped_column(JSON_TYPE)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class FileRecordModel(Base):
    __tablename__ = "files"

    id: Mapped[UUID] = mapped_column(primary_key=True)
    project_id: Mapped[UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    problem_id: Mapped[UUID] = mapped_column(
        ForeignKey("problems.id", ondelete="CASCADE"), index=True
    )
    original_name: Mapped[str] = mapped_column(String(255))
    safe_name: Mapped[str] = mapped_column(String(255))
    extension: Mapped[str] = mapped_column(String(20))
    kind: Mapped[str] = mapped_column(String(32))
    declared_mime_type: Mapped[str | None] = mapped_column(String(255))
    detected_mime_type: Mapped[str] = mapped_column(String(255))
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    storage_key: Mapped[str] = mapped_column(String(1000), unique=True)
    status: Mapped[str] = mapped_column(String(32))
    validation_warnings: Mapped[list[str]] = mapped_column(JSON_TYPE)
    parser_name: Mapped[str | None] = mapped_column(String(100))
    parser_version: Mapped[str | None] = mapped_column(String(100))
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ArtifactRecordModel(Base):
    __tablename__ = "artifacts"

    id: Mapped[UUID] = mapped_column(primary_key=True)
    project_id: Mapped[UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    problem_id: Mapped[UUID] = mapped_column(
        ForeignKey("problems.id", ondelete="CASCADE"), index=True
    )
    source_file_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("files.id", ondelete="CASCADE"), index=True
    )
    execution_run_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("execution_records.id", ondelete="CASCADE"), index=True
    )
    kind: Mapped[str] = mapped_column(String(64))
    name: Mapped[str] = mapped_column(String(255))
    mime_type: Mapped[str] = mapped_column(String(255))
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    storage_key: Mapped[str] = mapped_column(String(1000), unique=True)
    artifact_metadata: Mapped[dict[str, Any]] = mapped_column("metadata", JSON_TYPE)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class DatasetRecordModel(Base):
    __tablename__ = "datasets"

    id: Mapped[UUID] = mapped_column(primary_key=True)
    project_id: Mapped[UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    problem_id: Mapped[UUID] = mapped_column(
        ForeignKey("problems.id", ondelete="CASCADE"), index=True
    )
    source_file_id: Mapped[UUID] = mapped_column(
        ForeignKey("files.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(255))
    sheet_name: Mapped[str | None] = mapped_column(String(255))
    layer: Mapped[str] = mapped_column(String(32))
    row_count: Mapped[int] = mapped_column(BigInteger)
    column_count: Mapped[int]
    columns_json: Mapped[list[str]] = mapped_column("columns", JSON_TYPE)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class DataProfileRecordModel(Base):
    __tablename__ = "data_profiles"

    id: Mapped[UUID] = mapped_column(primary_key=True)
    project_id: Mapped[UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    problem_id: Mapped[UUID] = mapped_column(
        ForeignKey("problems.id", ondelete="CASCADE"), index=True
    )
    dataset_id: Mapped[UUID] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), unique=True
    )
    source_file_id: Mapped[UUID] = mapped_column(
        ForeignKey("files.id", ondelete="CASCADE"), index=True
    )
    profile_json: Mapped[dict[str, Any]] = mapped_column(JSON_TYPE)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ExecutionRecordModel(Base):
    __tablename__ = "execution_records"

    id: Mapped[UUID] = mapped_column(primary_key=True)
    project_id: Mapped[UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    problem_id: Mapped[UUID] = mapped_column(
        ForeignKey("problems.id", ondelete="CASCADE"), index=True
    )
    code_hash: Mapped[str] = mapped_column(String(64), index=True)
    executed_bundle_hash: Mapped[str | None] = mapped_column(String(64), index=True)
    execution_origin: Mapped[str] = mapped_column(String(48), default="USER_CODE", index=True)
    model_digest: Mapped[str | None] = mapped_column(String(64), index=True)
    generated_program_id: Mapped[UUID | None] = mapped_column(index=True)
    image: Mapped[str] = mapped_column(String(255))
    image_id: Mapped[str | None] = mapped_column(String(255))
    start_time: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    end_time: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    runtime_seconds: Mapped[float]
    status: Mapped[str] = mapped_column(String(32), index=True)
    exit_code: Mapped[int | None]
    is_mock: Mapped[bool] = mapped_column(Boolean, default=False)
    record_json: Mapped[dict[str, Any]] = mapped_column(JSON_TYPE)


class MathematicalModelRecord(Base):
    __tablename__ = "mathematical_models"
    __table_args__ = (
        UniqueConstraint(
            "model_id",
            "version",
            name="uq_mathematical_models_model_version",
        ),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    project_id: Mapped[UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    problem_id: Mapped[UUID] = mapped_column(
        ForeignKey("problems.id", ondelete="CASCADE"), index=True
    )
    model_id: Mapped[UUID] = mapped_column(index=True)
    version: Mapped[int]
    model_digest: Mapped[str] = mapped_column(String(64), index=True)
    state_version: Mapped[int]
    selected_model_id: Mapped[str] = mapped_column(String(100), index=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    model_json: Mapped[dict[str, Any]] = mapped_column(JSON_TYPE)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class GeneratedProgramRecord(Base):
    __tablename__ = "generated_programs"

    id: Mapped[UUID] = mapped_column(primary_key=True)
    project_id: Mapped[UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    problem_id: Mapped[UUID] = mapped_column(
        ForeignKey("problems.id", ondelete="CASCADE"), index=True
    )
    mathematical_model_record_id: Mapped[UUID] = mapped_column(
        ForeignKey("mathematical_models.id", ondelete="RESTRICT"), index=True
    )
    model_id: Mapped[UUID] = mapped_column(index=True)
    model_version: Mapped[int]
    model_digest: Mapped[str] = mapped_column(String(64), index=True)
    solver_target: Mapped[str] = mapped_column(String(64), index=True)
    code_hash: Mapped[str] = mapped_column(String(64), index=True)
    execution_origin: Mapped[str] = mapped_column(String(48), index=True)
    generator_agent_run_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("agent_runs.id", ondelete="RESTRICT"), index=True
    )
    status: Mapped[str] = mapped_column(String(32), index=True)
    is_mock: Mapped[bool] = mapped_column(Boolean, default=False)
    program_json: Mapped[dict[str, Any]] = mapped_column(JSON_TYPE)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class SolverRunRecord(Base):
    __tablename__ = "solver_runs"

    id: Mapped[UUID] = mapped_column(primary_key=True)
    project_id: Mapped[UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    problem_id: Mapped[UUID] = mapped_column(
        ForeignKey("problems.id", ondelete="CASCADE"), index=True
    )
    mathematical_model_record_id: Mapped[UUID] = mapped_column(
        ForeignKey("mathematical_models.id", ondelete="RESTRICT"), index=True
    )
    model_id: Mapped[UUID] = mapped_column(index=True)
    model_version: Mapped[int]
    model_digest: Mapped[str] = mapped_column(String(64), index=True)
    generated_program_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("generated_programs.id", ondelete="RESTRICT"), index=True
    )
    execution_record_id: Mapped[UUID] = mapped_column(
        ForeignKey("execution_records.id", ondelete="RESTRICT"), unique=True
    )
    execution_origin: Mapped[str] = mapped_column(String(48), index=True)
    routing_decision_id: Mapped[UUID | None] = mapped_column(index=True)
    routing_json: Mapped[dict[str, Any]] = mapped_column(JSON_TYPE)
    solver: Mapped[str] = mapped_column(String(32), index=True)
    solver_version: Mapped[str | None] = mapped_column(String(100))
    status: Mapped[str] = mapped_column(String(32), index=True)
    objective: Mapped[float | None] = mapped_column(Float)
    runtime_seconds: Mapped[float] = mapped_column(Float)
    options_json: Mapped[dict[str, Any]] = mapped_column(JSON_TYPE)
    result_json: Mapped[dict[str, Any]] = mapped_column(JSON_TYPE)
    error: Mapped[str | None] = mapped_column(Text)
    start_time: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    end_time: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class ResultRecordModel(Base):
    __tablename__ = "results"

    id: Mapped[UUID] = mapped_column(primary_key=True)
    project_id: Mapped[UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    problem_id: Mapped[UUID] = mapped_column(
        ForeignKey("problems.id", ondelete="CASCADE"), index=True
    )
    mathematical_model_record_id: Mapped[UUID] = mapped_column(
        ForeignKey("mathematical_models.id", ondelete="RESTRICT"), index=True
    )
    model_id: Mapped[UUID] = mapped_column(index=True)
    model_version: Mapped[int]
    model_digest: Mapped[str] = mapped_column(String(64), index=True)
    solver_run_id: Mapped[UUID] = mapped_column(
        ForeignKey("solver_runs.id", ondelete="RESTRICT"), unique=True
    )
    execution_record_id: Mapped[UUID] = mapped_column(
        ForeignKey("execution_records.id", ondelete="RESTRICT"), index=True
    )
    status: Mapped[str] = mapped_column(String(32), index=True)
    objective: Mapped[float | None] = mapped_column(Float)
    record_json: Mapped[dict[str, Any]] = mapped_column(JSON_TYPE)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ValidationRunRecord(Base):
    __tablename__ = "validation_runs"

    id: Mapped[UUID] = mapped_column(primary_key=True)
    project_id: Mapped[UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    problem_id: Mapped[UUID] = mapped_column(
        ForeignKey("problems.id", ondelete="CASCADE"), index=True
    )
    mathematical_model_record_id: Mapped[UUID] = mapped_column(
        ForeignKey("mathematical_models.id", ondelete="RESTRICT"), index=True
    )
    model_id: Mapped[UUID] = mapped_column(index=True)
    model_version: Mapped[int]
    model_digest: Mapped[str] = mapped_column(String(64), index=True)
    result_id: Mapped[UUID] = mapped_column(
        ForeignKey("results.id", ondelete="RESTRICT"), index=True
    )
    solver_run_id: Mapped[UUID] = mapped_column(
        ForeignKey("solver_runs.id", ondelete="RESTRICT"), index=True
    )
    execution_record_id: Mapped[UUID] = mapped_column(
        ForeignKey("execution_records.id", ondelete="RESTRICT"), index=True
    )
    status: Mapped[str] = mapped_column(String(32), index=True)
    report_json: Mapped[dict[str, Any]] = mapped_column(JSON_TYPE)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class SensitivityRunRecord(Base):
    __tablename__ = "sensitivity_runs"

    id: Mapped[UUID] = mapped_column(primary_key=True)
    project_id: Mapped[UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    problem_id: Mapped[UUID] = mapped_column(
        ForeignKey("problems.id", ondelete="CASCADE"), index=True
    )
    mathematical_model_record_id: Mapped[UUID] = mapped_column(
        ForeignKey("mathematical_models.id", ondelete="RESTRICT"), index=True
    )
    model_id: Mapped[UUID] = mapped_column(index=True)
    model_version: Mapped[int]
    model_digest: Mapped[str] = mapped_column(String(64), index=True)
    result_id: Mapped[UUID] = mapped_column(
        ForeignKey("results.id", ondelete="RESTRICT"), index=True
    )
    validation_run_id: Mapped[UUID] = mapped_column(
        ForeignKey("validation_runs.id", ondelete="RESTRICT"), index=True
    )
    status: Mapped[str] = mapped_column(String(32), index=True)
    config_json: Mapped[dict[str, Any]] = mapped_column(JSON_TYPE)
    report_json: Mapped[dict[str, Any]] = mapped_column(JSON_TYPE)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class RobustnessRunRecord(Base):
    __tablename__ = "robustness_runs"

    id: Mapped[UUID] = mapped_column(primary_key=True)
    project_id: Mapped[UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    problem_id: Mapped[UUID] = mapped_column(
        ForeignKey("problems.id", ondelete="CASCADE"), index=True
    )
    mathematical_model_record_id: Mapped[UUID] = mapped_column(
        ForeignKey("mathematical_models.id", ondelete="RESTRICT"), index=True
    )
    model_id: Mapped[UUID] = mapped_column(index=True)
    model_version: Mapped[int]
    model_digest: Mapped[str] = mapped_column(String(64), index=True)
    result_id: Mapped[UUID] = mapped_column(
        ForeignKey("results.id", ondelete="RESTRICT"), index=True
    )
    validation_run_id: Mapped[UUID] = mapped_column(
        ForeignKey("validation_runs.id", ondelete="RESTRICT"), index=True
    )
    sensitivity_run_id: Mapped[UUID] = mapped_column(
        ForeignKey("sensitivity_runs.id", ondelete="RESTRICT"), index=True
    )
    method: Mapped[str] = mapped_column(String(32), index=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    config_json: Mapped[dict[str, Any]] = mapped_column(JSON_TYPE)
    report_json: Mapped[dict[str, Any]] = mapped_column(JSON_TYPE)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class VerificationExperimentRecord(Base):
    __tablename__ = "verification_experiments"

    id: Mapped[UUID] = mapped_column(primary_key=True)
    project_id: Mapped[UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    problem_id: Mapped[UUID] = mapped_column(
        ForeignKey("problems.id", ondelete="CASCADE"), index=True
    )
    mathematical_model_record_id: Mapped[UUID] = mapped_column(
        ForeignKey("mathematical_models.id", ondelete="RESTRICT"), index=True
    )
    sensitivity_run_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("sensitivity_runs.id", ondelete="CASCADE"), index=True
    )
    robustness_run_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("robustness_runs.id", ondelete="CASCADE"), index=True
    )
    execution_record_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("execution_records.id", ondelete="RESTRICT"), unique=True
    )
    experiment_type: Mapped[str] = mapped_column(String(64), index=True)
    base_model_digest: Mapped[str] = mapped_column(String(64), index=True)
    scenario_model_digest: Mapped[str] = mapped_column(String(64), index=True)
    solver: Mapped[str | None] = mapped_column(String(32), index=True)
    solver_status: Mapped[str | None] = mapped_column(String(32), index=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    objective: Mapped[float | None] = mapped_column(Float)
    perturbation_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON_TYPE)
    program_json: Mapped[dict[str, Any] | None] = mapped_column(JSON_TYPE)
    solver_result_json: Mapped[dict[str, Any] | None] = mapped_column(JSON_TYPE)
    record_json: Mapped[dict[str, Any]] = mapped_column(JSON_TYPE)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class RedTeamReportRecord(Base):
    __tablename__ = "red_team_reports"

    id: Mapped[UUID] = mapped_column(primary_key=True)
    project_id: Mapped[UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    problem_id: Mapped[UUID] = mapped_column(
        ForeignKey("problems.id", ondelete="CASCADE"), index=True
    )
    mathematical_model_record_id: Mapped[UUID] = mapped_column(
        ForeignKey("mathematical_models.id", ondelete="RESTRICT"), index=True
    )
    model_id: Mapped[UUID] = mapped_column(index=True)
    model_version: Mapped[int]
    model_digest: Mapped[str] = mapped_column(String(64), index=True)
    result_id: Mapped[UUID] = mapped_column(
        ForeignKey("results.id", ondelete="RESTRICT"), index=True
    )
    validation_run_id: Mapped[UUID] = mapped_column(
        ForeignKey("validation_runs.id", ondelete="RESTRICT"), index=True
    )
    sensitivity_run_id: Mapped[UUID] = mapped_column(
        ForeignKey("sensitivity_runs.id", ondelete="RESTRICT"), index=True
    )
    robustness_run_id: Mapped[UUID] = mapped_column(
        ForeignKey("robustness_runs.id", ondelete="RESTRICT"), index=True
    )
    reviewer_agent_run_id: Mapped[UUID] = mapped_column(
        ForeignKey("agent_runs.id", ondelete="RESTRICT"), index=True
    )
    status: Mapped[str] = mapped_column(String(32), index=True)
    critical_count: Mapped[int]
    major_count: Mapped[int]
    minor_count: Mapped[int]
    review_is_mock: Mapped[bool] = mapped_column(Boolean, default=False)
    report_json: Mapped[dict[str, Any]] = mapped_column(JSON_TYPE)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class RepairCycleRecordModel(Base):
    __tablename__ = "repair_cycles"
    __table_args__ = (
        UniqueConstraint(
            "stable_model_id",
            "repair_cycle",
            name="uq_repair_cycles_model_cycle",
        ),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True)
    project_id: Mapped[UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    problem_id: Mapped[UUID] = mapped_column(
        ForeignKey("problems.id", ondelete="CASCADE"), index=True
    )
    stable_model_id: Mapped[UUID] = mapped_column(index=True)
    source_model_record_id: Mapped[UUID] = mapped_column(
        ForeignKey("mathematical_models.id", ondelete="RESTRICT"), index=True
    )
    target_model_record_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("mathematical_models.id", ondelete="RESTRICT"), index=True
    )
    red_team_report_id: Mapped[UUID] = mapped_column(
        ForeignKey("red_team_reports.id", ondelete="RESTRICT"), index=True
    )
    agent_run_id: Mapped[UUID] = mapped_column(
        ForeignKey("agent_runs.id", ondelete="RESTRICT"), index=True
    )
    repair_cycle: Mapped[int]
    status: Mapped[str] = mapped_column(String(32), index=True)
    is_mock: Mapped[bool] = mapped_column(Boolean, default=False)
    record_json: Mapped[dict[str, Any]] = mapped_column(JSON_TYPE)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
