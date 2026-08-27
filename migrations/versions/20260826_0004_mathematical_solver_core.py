"""Add Phase 4 mathematical model, program, solver run, and result registries.

Revision ID: 20260826_0004
Revises: 20260826_0003
Create Date: 2026-08-26
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260826_0004"
down_revision: str | None = "20260826_0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _jsonb() -> postgresql.JSONB:
    return postgresql.JSONB(astext_type=sa.Text())


def upgrade() -> None:
    op.create_table(
        "mathematical_models",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("problem_id", sa.Uuid(), nullable=False),
        sa.Column("model_id", sa.Uuid(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("state_version", sa.Integer(), nullable=False),
        sa.Column("selected_model_id", sa.String(length=100), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("model_json", _jsonb(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["problem_id"], ["problems.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "model_id",
            "version",
            name="uq_mathematical_models_model_version",
        ),
    )
    for column in ("model_id", "problem_id", "project_id", "selected_model_id", "status"):
        op.create_index(
            op.f(f"ix_mathematical_models_{column}"),
            "mathematical_models",
            [column],
        )

    op.create_table(
        "generated_programs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("problem_id", sa.Uuid(), nullable=False),
        sa.Column("mathematical_model_record_id", sa.Uuid(), nullable=False),
        sa.Column("model_id", sa.Uuid(), nullable=False),
        sa.Column("model_version", sa.Integer(), nullable=False),
        sa.Column("solver_target", sa.String(length=64), nullable=False),
        sa.Column("code_hash", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("is_mock", sa.Boolean(), nullable=False),
        sa.Column("program_json", _jsonb(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["mathematical_model_record_id"],
            ["mathematical_models.id"],
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(["problem_id"], ["problems.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    for column in (
        "code_hash",
        "mathematical_model_record_id",
        "model_id",
        "problem_id",
        "project_id",
        "solver_target",
        "status",
    ):
        op.create_index(
            op.f(f"ix_generated_programs_{column}"),
            "generated_programs",
            [column],
        )

    op.create_table(
        "solver_runs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("problem_id", sa.Uuid(), nullable=False),
        sa.Column("mathematical_model_record_id", sa.Uuid(), nullable=False),
        sa.Column("model_id", sa.Uuid(), nullable=False),
        sa.Column("model_version", sa.Integer(), nullable=False),
        sa.Column("generated_program_id", sa.Uuid(), nullable=True),
        sa.Column("execution_record_id", sa.Uuid(), nullable=False),
        sa.Column("solver", sa.String(length=32), nullable=False),
        sa.Column("solver_version", sa.String(length=100), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("objective", sa.Float(), nullable=True),
        sa.Column("runtime_seconds", sa.Float(), nullable=False),
        sa.Column("options_json", _jsonb(), nullable=False),
        sa.Column("result_json", _jsonb(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("start_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("end_time", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["execution_record_id"], ["execution_records.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["generated_program_id"], ["generated_programs.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["mathematical_model_record_id"],
            ["mathematical_models.id"],
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(["problem_id"], ["problems.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("execution_record_id"),
    )
    for column in (
        "generated_program_id",
        "mathematical_model_record_id",
        "model_id",
        "problem_id",
        "project_id",
        "solver",
        "status",
    ):
        op.create_index(op.f(f"ix_solver_runs_{column}"), "solver_runs", [column])

    op.create_table(
        "results",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("problem_id", sa.Uuid(), nullable=False),
        sa.Column("mathematical_model_record_id", sa.Uuid(), nullable=False),
        sa.Column("model_id", sa.Uuid(), nullable=False),
        sa.Column("model_version", sa.Integer(), nullable=False),
        sa.Column("solver_run_id", sa.Uuid(), nullable=False),
        sa.Column("execution_record_id", sa.Uuid(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("objective", sa.Float(), nullable=True),
        sa.Column("record_json", _jsonb(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["execution_record_id"], ["execution_records.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["mathematical_model_record_id"],
            ["mathematical_models.id"],
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(["problem_id"], ["problems.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["solver_run_id"], ["solver_runs.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("solver_run_id"),
    )
    for column in (
        "execution_record_id",
        "mathematical_model_record_id",
        "model_id",
        "problem_id",
        "project_id",
        "status",
    ):
        op.create_index(op.f(f"ix_results_{column}"), "results", [column])


def downgrade() -> None:
    for column in (
        "status",
        "project_id",
        "problem_id",
        "model_id",
        "mathematical_model_record_id",
        "execution_record_id",
    ):
        op.drop_index(op.f(f"ix_results_{column}"), table_name="results")
    op.drop_table("results")
    for column in (
        "status",
        "solver",
        "project_id",
        "problem_id",
        "model_id",
        "mathematical_model_record_id",
        "generated_program_id",
    ):
        op.drop_index(op.f(f"ix_solver_runs_{column}"), table_name="solver_runs")
    op.drop_table("solver_runs")
    for column in (
        "status",
        "solver_target",
        "project_id",
        "problem_id",
        "model_id",
        "mathematical_model_record_id",
        "code_hash",
    ):
        op.drop_index(op.f(f"ix_generated_programs_{column}"), table_name="generated_programs")
    op.drop_table("generated_programs")
    for column in ("status", "selected_model_id", "project_id", "problem_id", "model_id"):
        op.drop_index(op.f(f"ix_mathematical_models_{column}"), table_name="mathematical_models")
    op.drop_table("mathematical_models")
