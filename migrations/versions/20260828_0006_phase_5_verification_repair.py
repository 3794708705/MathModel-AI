"""Add Phase 5 verification, experiment, red-team, and repair registries.

Revision ID: 20260828_0006
Revises: 20260827_0005
Create Date: 2026-08-28
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260828_0006"
down_revision: str | None = "20260827_0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _jsonb() -> postgresql.JSONB:
    return postgresql.JSONB(astext_type=sa.Text())


def _create_indexes(table: str, columns: tuple[str, ...]) -> None:
    for column in columns:
        op.create_index(op.f(f"ix_{table}_{column}"), table, [column])


def _drop_indexes(table: str, columns: tuple[str, ...]) -> None:
    for column in reversed(columns):
        op.drop_index(op.f(f"ix_{table}_{column}"), table_name=table)


def upgrade() -> None:
    op.create_table(
        "validation_runs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("problem_id", sa.Uuid(), nullable=False),
        sa.Column("mathematical_model_record_id", sa.Uuid(), nullable=False),
        sa.Column("model_id", sa.Uuid(), nullable=False),
        sa.Column("model_version", sa.Integer(), nullable=False),
        sa.Column("model_digest", sa.String(length=64), nullable=False),
        sa.Column("result_id", sa.Uuid(), nullable=False),
        sa.Column("solver_run_id", sa.Uuid(), nullable=False),
        sa.Column("execution_record_id", sa.Uuid(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("report_json", _jsonb(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["execution_record_id"], ["execution_records.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["mathematical_model_record_id"], ["mathematical_models.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(["problem_id"], ["problems.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["result_id"], ["results.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["solver_run_id"], ["solver_runs.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
    )
    _create_indexes(
        "validation_runs",
        (
            "project_id",
            "problem_id",
            "mathematical_model_record_id",
            "model_id",
            "model_digest",
            "result_id",
            "solver_run_id",
            "execution_record_id",
            "status",
        ),
    )

    op.create_table(
        "sensitivity_runs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("problem_id", sa.Uuid(), nullable=False),
        sa.Column("mathematical_model_record_id", sa.Uuid(), nullable=False),
        sa.Column("model_id", sa.Uuid(), nullable=False),
        sa.Column("model_version", sa.Integer(), nullable=False),
        sa.Column("model_digest", sa.String(length=64), nullable=False),
        sa.Column("result_id", sa.Uuid(), nullable=False),
        sa.Column("validation_run_id", sa.Uuid(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("config_json", _jsonb(), nullable=False),
        sa.Column("report_json", _jsonb(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["mathematical_model_record_id"], ["mathematical_models.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(["problem_id"], ["problems.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["result_id"], ["results.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["validation_run_id"], ["validation_runs.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
    )
    _create_indexes(
        "sensitivity_runs",
        (
            "project_id",
            "problem_id",
            "mathematical_model_record_id",
            "model_id",
            "model_digest",
            "result_id",
            "validation_run_id",
            "status",
        ),
    )

    op.create_table(
        "robustness_runs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("problem_id", sa.Uuid(), nullable=False),
        sa.Column("mathematical_model_record_id", sa.Uuid(), nullable=False),
        sa.Column("model_id", sa.Uuid(), nullable=False),
        sa.Column("model_version", sa.Integer(), nullable=False),
        sa.Column("model_digest", sa.String(length=64), nullable=False),
        sa.Column("result_id", sa.Uuid(), nullable=False),
        sa.Column("validation_run_id", sa.Uuid(), nullable=False),
        sa.Column("sensitivity_run_id", sa.Uuid(), nullable=False),
        sa.Column("method", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("config_json", _jsonb(), nullable=False),
        sa.Column("report_json", _jsonb(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["mathematical_model_record_id"], ["mathematical_models.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(["problem_id"], ["problems.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["result_id"], ["results.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(
            ["sensitivity_run_id"], ["sensitivity_runs.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(["validation_run_id"], ["validation_runs.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
    )
    _create_indexes(
        "robustness_runs",
        (
            "project_id",
            "problem_id",
            "mathematical_model_record_id",
            "model_id",
            "model_digest",
            "result_id",
            "validation_run_id",
            "sensitivity_run_id",
            "method",
            "status",
        ),
    )

    op.create_table(
        "verification_experiments",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("problem_id", sa.Uuid(), nullable=False),
        sa.Column("mathematical_model_record_id", sa.Uuid(), nullable=False),
        sa.Column("sensitivity_run_id", sa.Uuid(), nullable=True),
        sa.Column("robustness_run_id", sa.Uuid(), nullable=True),
        sa.Column("execution_record_id", sa.Uuid(), nullable=True),
        sa.Column("experiment_type", sa.String(length=64), nullable=False),
        sa.Column("base_model_digest", sa.String(length=64), nullable=False),
        sa.Column("scenario_model_digest", sa.String(length=64), nullable=False),
        sa.Column("solver", sa.String(length=32), nullable=True),
        sa.Column("solver_status", sa.String(length=32), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("objective", sa.Float(), nullable=True),
        sa.Column("perturbation_json", _jsonb(), nullable=False),
        sa.Column("program_json", _jsonb(), nullable=True),
        sa.Column("solver_result_json", _jsonb(), nullable=True),
        sa.Column("record_json", _jsonb(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["execution_record_id"], ["execution_records.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["mathematical_model_record_id"], ["mathematical_models.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(["problem_id"], ["problems.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["robustness_run_id"], ["robustness_runs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["sensitivity_run_id"], ["sensitivity_runs.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("execution_record_id"),
    )
    _create_indexes(
        "verification_experiments",
        (
            "project_id",
            "problem_id",
            "mathematical_model_record_id",
            "sensitivity_run_id",
            "robustness_run_id",
            "experiment_type",
            "base_model_digest",
            "scenario_model_digest",
            "solver",
            "solver_status",
            "status",
        ),
    )

    op.create_table(
        "red_team_reports",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("problem_id", sa.Uuid(), nullable=False),
        sa.Column("mathematical_model_record_id", sa.Uuid(), nullable=False),
        sa.Column("model_id", sa.Uuid(), nullable=False),
        sa.Column("model_version", sa.Integer(), nullable=False),
        sa.Column("model_digest", sa.String(length=64), nullable=False),
        sa.Column("result_id", sa.Uuid(), nullable=False),
        sa.Column("validation_run_id", sa.Uuid(), nullable=False),
        sa.Column("sensitivity_run_id", sa.Uuid(), nullable=False),
        sa.Column("robustness_run_id", sa.Uuid(), nullable=False),
        sa.Column("reviewer_agent_run_id", sa.Uuid(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("critical_count", sa.Integer(), nullable=False),
        sa.Column("major_count", sa.Integer(), nullable=False),
        sa.Column("minor_count", sa.Integer(), nullable=False),
        sa.Column("review_is_mock", sa.Boolean(), nullable=False),
        sa.Column("report_json", _jsonb(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["mathematical_model_record_id"], ["mathematical_models.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(["problem_id"], ["problems.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["result_id"], ["results.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["reviewer_agent_run_id"], ["agent_runs.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["robustness_run_id"], ["robustness_runs.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(
            ["sensitivity_run_id"], ["sensitivity_runs.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(["validation_run_id"], ["validation_runs.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
    )
    _create_indexes(
        "red_team_reports",
        (
            "project_id",
            "problem_id",
            "mathematical_model_record_id",
            "model_id",
            "model_digest",
            "result_id",
            "validation_run_id",
            "sensitivity_run_id",
            "robustness_run_id",
            "reviewer_agent_run_id",
            "status",
        ),
    )

    op.create_table(
        "repair_cycles",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("problem_id", sa.Uuid(), nullable=False),
        sa.Column("stable_model_id", sa.Uuid(), nullable=False),
        sa.Column("source_model_record_id", sa.Uuid(), nullable=False),
        sa.Column("target_model_record_id", sa.Uuid(), nullable=True),
        sa.Column("red_team_report_id", sa.Uuid(), nullable=False),
        sa.Column("agent_run_id", sa.Uuid(), nullable=False),
        sa.Column("repair_cycle", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("is_mock", sa.Boolean(), nullable=False),
        sa.Column("record_json", _jsonb(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["agent_run_id"], ["agent_runs.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["problem_id"], ["problems.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["red_team_report_id"], ["red_team_reports.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["source_model_record_id"], ["mathematical_models.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["target_model_record_id"], ["mathematical_models.id"], ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("stable_model_id", "repair_cycle", name="uq_repair_cycles_model_cycle"),
    )
    _create_indexes(
        "repair_cycles",
        (
            "project_id",
            "problem_id",
            "stable_model_id",
            "source_model_record_id",
            "target_model_record_id",
            "red_team_report_id",
            "agent_run_id",
            "status",
        ),
    )


def downgrade() -> None:
    _drop_indexes(
        "repair_cycles",
        (
            "project_id",
            "problem_id",
            "stable_model_id",
            "source_model_record_id",
            "target_model_record_id",
            "red_team_report_id",
            "agent_run_id",
            "status",
        ),
    )
    op.drop_table("repair_cycles")
    _drop_indexes(
        "red_team_reports",
        (
            "project_id",
            "problem_id",
            "mathematical_model_record_id",
            "model_id",
            "model_digest",
            "result_id",
            "validation_run_id",
            "sensitivity_run_id",
            "robustness_run_id",
            "reviewer_agent_run_id",
            "status",
        ),
    )
    op.drop_table("red_team_reports")
    _drop_indexes(
        "verification_experiments",
        (
            "project_id",
            "problem_id",
            "mathematical_model_record_id",
            "sensitivity_run_id",
            "robustness_run_id",
            "experiment_type",
            "base_model_digest",
            "scenario_model_digest",
            "solver",
            "solver_status",
            "status",
        ),
    )
    op.drop_table("verification_experiments")
    _drop_indexes(
        "robustness_runs",
        (
            "project_id",
            "problem_id",
            "mathematical_model_record_id",
            "model_id",
            "model_digest",
            "result_id",
            "validation_run_id",
            "sensitivity_run_id",
            "method",
            "status",
        ),
    )
    op.drop_table("robustness_runs")
    _drop_indexes(
        "sensitivity_runs",
        (
            "project_id",
            "problem_id",
            "mathematical_model_record_id",
            "model_id",
            "model_digest",
            "result_id",
            "validation_run_id",
            "status",
        ),
    )
    op.drop_table("sensitivity_runs")
    _drop_indexes(
        "validation_runs",
        (
            "project_id",
            "problem_id",
            "mathematical_model_record_id",
            "model_id",
            "model_digest",
            "result_id",
            "solver_run_id",
            "execution_record_id",
            "status",
        ),
    )
    op.drop_table("validation_runs")
