"""Add immutable independent plans and idempotent execution jobs.

Revision ID: 20260906_0012
Revises: 20260831_0011
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260906_0012"
down_revision: str | None = "20260831_0011"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    json_type = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")
    op.create_table(
        "independent_verification_plans",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "attempt_id",
            sa.Uuid(),
            sa.ForeignKey("benchmark_attempts.id", ondelete="RESTRICT"),
            nullable=False,
            unique=True,
        ),
        sa.Column(
            "result_id", sa.Uuid(), sa.ForeignKey("results.id", ondelete="RESTRICT"), nullable=False
        ),
        sa.Column("plan_digest", sa.String(64), nullable=False),
        sa.Column("requirements_digest", sa.String(64), nullable=False),
        sa.Column("plan_json", json_type, nullable=False),
    )
    op.create_table(
        "independent_verification_jobs",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "plan_id",
            sa.Uuid(),
            sa.ForeignKey("independent_verification_plans.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("operation", sa.String(100), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("payload_digest", sa.String(64)),
        sa.Column("payload_json", json_type),
        sa.Column(
            "execution_id", sa.Uuid(), sa.ForeignKey("execution_records.id", ondelete="RESTRICT")
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.UniqueConstraint("plan_id", "operation", name="uq_independent_job_key"),
    )


def downgrade() -> None:
    op.drop_table("independent_verification_jobs")
    op.drop_table("independent_verification_plans")
