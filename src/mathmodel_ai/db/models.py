from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
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
