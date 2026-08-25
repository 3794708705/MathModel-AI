"""Add Phase 2 reasoning trace and decision persistence.

Revision ID: 20260825_0002
Revises: 20260825_0001
Create Date: 2026-08-25
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260825_0002"
down_revision: str | None = "20260825_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "problem_states",
        sa.Column("updated_by", sa.String(length=100), nullable=False, server_default="system"),
    )
    op.add_column(
        "problem_states",
        sa.Column(
            "update_reason", sa.Text(), nullable=False, server_default="legacy state revision"
        ),
    )
    op.create_table(
        "agent_runs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("problem_id", sa.Uuid(), nullable=False),
        sa.Column("agent_name", sa.String(length=100), nullable=False),
        sa.Column("input_state_version", sa.Integer(), nullable=False),
        sa.Column("output_state_version", sa.Integer(), nullable=True),
        sa.Column("provider", sa.String(length=50), nullable=True),
        sa.Column("model", sa.String(length=255), nullable=True),
        sa.Column("reasoning_level", sa.String(length=32), nullable=True),
        sa.Column("prompt_version", sa.String(length=64), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column("token_usage", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("retry_count", sa.Integer(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("is_mock", sa.Boolean(), nullable=False),
        sa.Column("route_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.ForeignKeyConstraint(
            ["problem_id"],
            ["problems.id"],
            ondelete="CASCADE",
            name=op.f("fk_agent_runs_problem_id_problems"),
        ),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
            ondelete="CASCADE",
            name=op.f("fk_agent_runs_project_id_projects"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_agent_runs")),
    )
    op.create_index(op.f("ix_agent_runs_agent_name"), "agent_runs", ["agent_name"])
    op.create_index(op.f("ix_agent_runs_problem_id"), "agent_runs", ["problem_id"])
    op.create_index(op.f("ix_agent_runs_project_id"), "agent_runs", ["project_id"])
    op.create_table(
        "model_decisions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("problem_id", sa.Uuid(), nullable=False),
        sa.Column("state_version", sa.Integer(), nullable=False),
        sa.Column("agent_run_id", sa.Uuid(), nullable=False),
        sa.Column("selected_model_id", sa.String(length=100), nullable=False),
        sa.Column("backup_model_id", sa.String(length=100), nullable=False),
        sa.Column("weights_version", sa.String(length=64), nullable=False),
        sa.Column("decision_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(
            ["agent_run_id"],
            ["agent_runs.id"],
            ondelete="RESTRICT",
            name=op.f("fk_model_decisions_agent_run_id_agent_runs"),
        ),
        sa.ForeignKeyConstraint(
            ["problem_id"],
            ["problems.id"],
            ondelete="CASCADE",
            name=op.f("fk_model_decisions_problem_id_problems"),
        ),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
            ondelete="CASCADE",
            name=op.f("fk_model_decisions_project_id_projects"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_model_decisions")),
        sa.UniqueConstraint("agent_run_id", name=op.f("uq_model_decisions_agent_run_id")),
    )
    op.create_index(op.f("ix_model_decisions_problem_id"), "model_decisions", ["problem_id"])
    op.create_index(op.f("ix_model_decisions_project_id"), "model_decisions", ["project_id"])


def downgrade() -> None:
    op.drop_index(op.f("ix_model_decisions_project_id"), table_name="model_decisions")
    op.drop_index(op.f("ix_model_decisions_problem_id"), table_name="model_decisions")
    op.drop_table("model_decisions")
    op.drop_index(op.f("ix_agent_runs_project_id"), table_name="agent_runs")
    op.drop_index(op.f("ix_agent_runs_problem_id"), table_name="agent_runs")
    op.drop_index(op.f("ix_agent_runs_agent_name"), table_name="agent_runs")
    op.drop_table("agent_runs")
    op.drop_column("problem_states", "update_reason")
    op.drop_column("problem_states", "updated_by")
