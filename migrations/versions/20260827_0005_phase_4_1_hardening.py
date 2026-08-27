"""Add Phase 4.1 evidence, execution-origin, and routing trace fields.

Revision ID: 20260827_0005
Revises: 20260826_0004
Create Date: 2026-08-27
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260827_0005"
down_revision: str | None = "20260826_0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_UNKNOWN_DIGEST = "0" * 64


def _jsonb() -> postgresql.JSONB:
    return postgresql.JSONB(astext_type=sa.Text())


def upgrade() -> None:
    op.add_column(
        "execution_records",
        sa.Column(
            "execution_origin",
            sa.String(length=48),
            server_default="USER_CODE",
            nullable=False,
        ),
    )
    op.add_column("execution_records", sa.Column("model_digest", sa.String(64), nullable=True))
    op.add_column(
        "execution_records", sa.Column("executed_bundle_hash", sa.String(64), nullable=True)
    )
    op.add_column("execution_records", sa.Column("generated_program_id", sa.Uuid(), nullable=True))

    op.add_column(
        "mathematical_models",
        sa.Column("model_digest", sa.String(64), server_default=_UNKNOWN_DIGEST, nullable=False),
    )
    op.alter_column("mathematical_models", "model_digest", server_default=None)

    op.add_column(
        "generated_programs",
        sa.Column("model_digest", sa.String(64), server_default=_UNKNOWN_DIGEST, nullable=False),
    )
    op.alter_column("generated_programs", "model_digest", server_default=None)
    op.add_column(
        "generated_programs",
        sa.Column(
            "execution_origin",
            sa.String(length=48),
            server_default="DETERMINISTIC_SOLVER_ADAPTER",
            nullable=False,
        ),
    )
    op.add_column(
        "generated_programs",
        sa.Column("generator_agent_run_id", sa.Uuid(), nullable=True),
    )
    op.create_foreign_key(
        "fk_generated_programs_generator_agent_run_id_agent_runs",
        "generated_programs",
        "agent_runs",
        ["generator_agent_run_id"],
        ["id"],
        ondelete="RESTRICT",
    )

    op.add_column(
        "solver_runs",
        sa.Column("model_digest", sa.String(64), server_default=_UNKNOWN_DIGEST, nullable=False),
    )
    op.alter_column("solver_runs", "model_digest", server_default=None)
    op.add_column(
        "solver_runs",
        sa.Column(
            "execution_origin",
            sa.String(length=48),
            server_default="DETERMINISTIC_SOLVER_ADAPTER",
            nullable=False,
        ),
    )
    op.add_column("solver_runs", sa.Column("routing_decision_id", sa.Uuid(), nullable=True))
    op.add_column(
        "solver_runs",
        sa.Column("routing_json", _jsonb(), server_default=sa.text("'{}'::jsonb"), nullable=False),
    )
    op.alter_column("solver_runs", "routing_json", server_default=None)

    op.add_column(
        "results",
        sa.Column("model_digest", sa.String(64), server_default=_UNKNOWN_DIGEST, nullable=False),
    )
    op.alter_column("results", "model_digest", server_default=None)

    op.execute(
        """
        UPDATE execution_records AS execution
        SET execution_origin = 'DETERMINISTIC_SOLVER_ADAPTER'
        WHERE EXISTS (
            SELECT 1 FROM solver_runs AS run
            WHERE run.execution_record_id = execution.id
        )
        """
    )

    for table, columns in {
        "execution_records": (
            "execution_origin",
            "model_digest",
            "executed_bundle_hash",
            "generated_program_id",
        ),
        "mathematical_models": ("model_digest",),
        "generated_programs": (
            "model_digest",
            "execution_origin",
            "generator_agent_run_id",
        ),
        "solver_runs": ("model_digest", "execution_origin", "routing_decision_id"),
        "results": ("model_digest",),
    }.items():
        for column in columns:
            op.create_index(op.f(f"ix_{table}_{column}"), table, [column])


def downgrade() -> None:
    for table, columns in {
        "results": ("model_digest",),
        "solver_runs": ("routing_decision_id", "execution_origin", "model_digest"),
        "generated_programs": (
            "generator_agent_run_id",
            "execution_origin",
            "model_digest",
        ),
        "mathematical_models": ("model_digest",),
        "execution_records": (
            "generated_program_id",
            "executed_bundle_hash",
            "model_digest",
            "execution_origin",
        ),
    }.items():
        for column in columns:
            op.drop_index(op.f(f"ix_{table}_{column}"), table_name=table)

    op.drop_column("results", "model_digest")
    op.drop_column("solver_runs", "routing_json")
    op.drop_column("solver_runs", "routing_decision_id")
    op.drop_column("solver_runs", "execution_origin")
    op.drop_column("solver_runs", "model_digest")
    op.drop_constraint(
        "fk_generated_programs_generator_agent_run_id_agent_runs",
        "generated_programs",
        type_="foreignkey",
    )
    op.drop_column("generated_programs", "generator_agent_run_id")
    op.drop_column("generated_programs", "execution_origin")
    op.drop_column("generated_programs", "model_digest")
    op.drop_column("mathematical_models", "model_digest")
    op.drop_column("execution_records", "generated_program_id")
    op.drop_column("execution_records", "executed_bundle_hash")
    op.drop_column("execution_records", "model_digest")
    op.drop_column("execution_records", "execution_origin")
