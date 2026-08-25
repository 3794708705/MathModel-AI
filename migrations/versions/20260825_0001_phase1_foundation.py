"""Create Phase 1 project, problem, and state tables.

Revision ID: 20260825_0001
Revises:
Create Date: 2026-08-25
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260825_0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "projects",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_projects")),
    )
    op.create_table(
        "problems",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("title", sa.String(length=500), nullable=False),
        sa.Column("raw_problem", sa.Text(), nullable=False),
        sa.Column("competition", sa.String(length=255), nullable=True),
        sa.Column("deadline", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
            name=op.f("fk_problems_project_id_projects"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_problems")),
    )
    op.create_index(op.f("ix_problems_project_id"), "problems", ["project_id"], unique=False)
    op.create_table(
        "problem_states",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("problem_id", sa.Uuid(), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.Column("current_stage", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("is_current", sa.Boolean(), nullable=False),
        sa.Column("state", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(
            ["problem_id"],
            ["problems.id"],
            name=op.f("fk_problem_states_problem_id_problems"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_problem_states")),
        sa.UniqueConstraint("problem_id", "revision", name="uq_problem_states_problem_revision"),
    )
    op.create_index(
        "uq_problem_states_one_current",
        "problem_states",
        ["problem_id"],
        unique=True,
        postgresql_where=sa.text("is_current"),
    )
    op.create_index(
        op.f("ix_problem_states_problem_id"), "problem_states", ["problem_id"], unique=False
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_problem_states_problem_id"), table_name="problem_states")
    op.drop_index("uq_problem_states_one_current", table_name="problem_states")
    op.drop_table("problem_states")
    op.drop_index(op.f("ix_problems_project_id"), table_name="problems")
    op.drop_table("problems")
    op.drop_table("projects")
