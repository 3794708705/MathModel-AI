"""add phase 8 immutable benchmark history

Revision ID: 20260830_0009
Revises: 20260829_0008
Create Date: 2026-08-30 10:00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260830_0009"
down_revision: str | None = "20260829_0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

JSON_TYPE = sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql")


def upgrade() -> None:
    op.create_table(
        "benchmark_runs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("code_commit", sa.String(length=40), nullable=False),
        sa.Column("source_tree_digest", sa.String(length=64), nullable=False),
        sa.Column("working_tree_dirty", sa.Boolean(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("provider", sa.String(length=64), nullable=False),
        sa.Column("model", sa.String(length=255), nullable=False),
        sa.Column("live_provider_status", sa.String(length=16), nullable=False),
        sa.Column("live_literature_status", sa.String(length=16), nullable=False),
        sa.Column("run_digest", sa.String(length=64), nullable=False),
        sa.Column("run_json", JSON_TYPE, nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_benchmark_runs")),
    )
    _indexes(
        "benchmark_runs",
        "code_commit",
        "source_tree_digest",
        "working_tree_dirty",
        "status",
        "provider",
        "live_provider_status",
        "live_literature_status",
        "run_digest",
    )
    op.create_table(
        "benchmark_attempts",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("benchmark_id", sa.String(length=100), nullable=False),
        sa.Column("attempt_number", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=40), nullable=False),
        sa.Column("project_id", sa.Uuid(), nullable=True),
        sa.Column("official", sa.Boolean(), nullable=False),
        sa.Column("solve_input_digest", sa.String(length=64), nullable=False),
        sa.Column("attempt_digest", sa.String(length=64), nullable=False),
        sa.Column("attempt_json", JSON_TYPE, nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
            name=op.f("fk_benchmark_attempts_project_id_projects"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["benchmark_runs.id"],
            name=op.f("fk_benchmark_attempts_run_id_benchmark_runs"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_benchmark_attempts")),
        sa.UniqueConstraint(
            "run_id",
            "benchmark_id",
            "attempt_number",
            name="uq_benchmark_attempts_run_case_number",
        ),
    )
    _indexes(
        "benchmark_attempts",
        "run_id",
        "benchmark_id",
        "status",
        "project_id",
        "official",
        "solve_input_digest",
        "attempt_digest",
    )
    op.create_table(
        "benchmark_case_results",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("attempt_id", sa.Uuid(), nullable=False),
        sa.Column("benchmark_id", sa.String(length=100), nullable=False),
        sa.Column("status", sa.String(length=40), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("result_digest", sa.String(length=64), nullable=False),
        sa.Column("result_json", JSON_TYPE, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["attempt_id"],
            ["benchmark_attempts.id"],
            name=op.f("fk_benchmark_case_results_attempt_id_benchmark_attempts"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_benchmark_case_results")),
    )
    op.create_index(
        op.f("ix_benchmark_case_results_attempt_id"),
        "benchmark_case_results",
        ["attempt_id"],
        unique=True,
    )
    _indexes(
        "benchmark_case_results",
        "benchmark_id",
        "status",
        "result_digest",
    )
    op.create_table(
        "benchmark_metrics",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("attempt_id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("value", sa.Float(), nullable=False),
        sa.Column("metric_digest", sa.String(length=64), nullable=False),
        sa.Column("metric_json", JSON_TYPE, nullable=False),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["attempt_id"],
            ["benchmark_attempts.id"],
            name=op.f("fk_benchmark_metrics_attempt_id_benchmark_attempts"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_benchmark_metrics")),
        sa.UniqueConstraint("attempt_id", "name", name="uq_benchmark_metrics_attempt_name"),
    )
    _indexes(
        "benchmark_metrics",
        "attempt_id",
        "name",
        "kind",
        "metric_digest",
    )
    op.create_table(
        "benchmark_failures",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("attempt_id", sa.Uuid(), nullable=False),
        sa.Column("category", sa.String(length=40), nullable=False),
        sa.Column("severity", sa.String(length=8), nullable=False),
        sa.Column("generic_issue", sa.Boolean(), nullable=False),
        sa.Column("failure_digest", sa.String(length=64), nullable=False),
        sa.Column("failure_json", JSON_TYPE, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["attempt_id"],
            ["benchmark_attempts.id"],
            name=op.f("fk_benchmark_failures_attempt_id_benchmark_attempts"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_benchmark_failures")),
    )
    _indexes(
        "benchmark_failures",
        "attempt_id",
        "category",
        "severity",
        "generic_issue",
        "failure_digest",
    )
    op.create_table(
        "benchmark_human_interventions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("attempt_id", sa.Uuid(), nullable=False),
        sa.Column("intervention_type", sa.String(length=40), nullable=False),
        sa.Column("intervention_digest", sa.String(length=64), nullable=False),
        sa.Column("intervention_json", JSON_TYPE, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["attempt_id"],
            ["benchmark_attempts.id"],
            name=op.f("fk_benchmark_human_interventions_attempt_id_benchmark_attempts"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_benchmark_human_interventions")),
    )
    _indexes(
        "benchmark_human_interventions",
        "attempt_id",
        "intervention_type",
        "intervention_digest",
    )


def downgrade() -> None:
    for table in (
        "benchmark_human_interventions",
        "benchmark_failures",
        "benchmark_metrics",
        "benchmark_case_results",
        "benchmark_attempts",
        "benchmark_runs",
    ):
        op.drop_table(table)


def _indexes(table: str, *columns: str) -> None:
    for column in columns:
        op.create_index(op.f(f"ix_{table}_{column}"), table, [column], unique=False)
