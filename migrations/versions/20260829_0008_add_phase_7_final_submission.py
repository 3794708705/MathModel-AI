"""add phase 7 final submission pipeline

Revision ID: 20260829_0008
Revises: 20260828_0007
Create Date: 2026-08-29 16:00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260829_0008"
down_revision: str | None = "20260828_0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

JSON_TYPE = sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql")


def upgrade() -> None:
    op.create_table(
        "competition_profiles",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("profile_id", sa.Uuid(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("verification_status", sa.String(length=32), nullable=False),
        sa.Column("profile_digest", sa.String(length=64), nullable=False),
        sa.Column("profile_json", JSON_TYPE, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_competition_profiles")),
        sa.UniqueConstraint("profile_id", "version", name="uq_competition_profiles_identity"),
    )
    _indexes("competition_profiles", "profile_id", "name", "verification_status", "profile_digest")
    op.create_table(
        "competition_rules",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("profile_record_id", sa.Uuid(), nullable=False),
        sa.Column("rule_id", sa.String(length=100), nullable=False),
        sa.Column("rule_type", sa.String(length=40), nullable=False),
        sa.Column("severity", sa.String(length=16), nullable=False),
        sa.Column("verification_status", sa.String(length=32), nullable=False),
        sa.Column("rule_json", JSON_TYPE, nullable=False),
        sa.ForeignKeyConstraint(
            ["profile_record_id"],
            ["competition_profiles.id"],
            name=op.f("fk_competition_rules_profile_record_id_competition_profiles"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_competition_rules")),
        sa.UniqueConstraint(
            "profile_record_id", "rule_id", name="uq_competition_rules_profile_rule"
        ),
    )
    _indexes(
        "competition_rules",
        "profile_record_id",
        "rule_id",
        "rule_type",
        "severity",
        "verification_status",
    )
    op.create_table(
        "requirement_coverage",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("paper_version_record_id", sa.Uuid(), nullable=False),
        sa.Column("requirement_id", sa.String(length=100), nullable=False),
        sa.Column("subproblem_id", sa.String(length=100), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("coverage_json", JSON_TYPE, nullable=False),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
            name=op.f("fk_requirement_coverage_project_id_projects"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["paper_version_record_id"],
            ["paper_versions.id"],
            name=op.f("fk_requirement_coverage_paper_version_record_id_paper_versions"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_requirement_coverage")),
    )
    _indexes(
        "requirement_coverage",
        "project_id",
        "paper_version_record_id",
        "requirement_id",
        "subproblem_id",
        "status",
    )
    op.create_table(
        "final_jury_reports",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("paper_version_record_id", sa.Uuid(), nullable=False),
        sa.Column("verified_result_id", sa.Uuid(), nullable=False),
        sa.Column("profile_record_id", sa.Uuid(), nullable=False),
        sa.Column("decision", sa.String(length=32), nullable=False),
        sa.Column("claimed_score", sa.Float(), nullable=False),
        sa.Column("reviewer_is_mock", sa.Boolean(), nullable=False),
        sa.Column("report_digest", sa.String(length=64), nullable=False),
        sa.Column("agent_run_id", sa.Uuid(), nullable=True),
        sa.Column("report_json", JSON_TYPE, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["agent_run_id"],
            ["agent_runs.id"],
            name=op.f("fk_final_jury_reports_agent_run_id_agent_runs"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["paper_version_record_id"],
            ["paper_versions.id"],
            name=op.f("fk_final_jury_reports_paper_version_record_id_paper_versions"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["profile_record_id"],
            ["competition_profiles.id"],
            name=op.f("fk_final_jury_reports_profile_record_id_competition_profiles"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
            name=op.f("fk_final_jury_reports_project_id_projects"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["verified_result_id"],
            ["results.id"],
            name=op.f("fk_final_jury_reports_verified_result_id_results"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_final_jury_reports")),
    )
    _indexes(
        "final_jury_reports",
        "project_id",
        "paper_version_record_id",
        "verified_result_id",
        "profile_record_id",
        "decision",
        "reviewer_is_mock",
        "report_digest",
        "agent_run_id",
    )
    op.create_table(
        "jury_findings",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("jury_report_id", sa.Uuid(), nullable=False),
        sa.Column("finding_id", sa.String(length=100), nullable=False),
        sa.Column("severity", sa.String(length=16), nullable=False),
        sa.Column("resolved", sa.Boolean(), nullable=False),
        sa.Column("finding_json", JSON_TYPE, nullable=False),
        sa.ForeignKeyConstraint(
            ["jury_report_id"],
            ["final_jury_reports.id"],
            name=op.f("fk_jury_findings_jury_report_id_final_jury_reports"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_jury_findings")),
        sa.UniqueConstraint("jury_report_id", "finding_id", name="uq_jury_findings_report_finding"),
    )
    _indexes("jury_findings", "jury_report_id", "finding_id", "severity", "resolved")
    op.create_table(
        "submission_checks",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("jury_report_id", sa.Uuid(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("check_digest", sa.String(length=64), nullable=False),
        sa.Column("check_json", JSON_TYPE, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["jury_report_id"],
            ["final_jury_reports.id"],
            name=op.f("fk_submission_checks_jury_report_id_final_jury_reports"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
            name=op.f("fk_submission_checks_project_id_projects"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_submission_checks")),
    )
    _indexes("submission_checks", "project_id", "jury_report_id", "status", "check_digest")
    op.create_table(
        "submission_snapshots",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("paper_version_record_id", sa.Uuid(), nullable=False),
        sa.Column("verified_result_id", sa.Uuid(), nullable=False),
        sa.Column("profile_record_id", sa.Uuid(), nullable=False),
        sa.Column("submission_check_id", sa.Uuid(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("manifest_hash", sa.String(length=64), nullable=False),
        sa.Column("package_hash", sa.String(length=64), nullable=False),
        sa.Column("artifact_set_digest", sa.String(length=64), nullable=False),
        sa.Column("snapshot_digest", sa.String(length=64), nullable=False),
        sa.Column("snapshot_json", JSON_TYPE, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["paper_version_record_id"],
            ["paper_versions.id"],
            name=op.f("fk_submission_snapshots_paper_version_record_id_paper_versions"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["profile_record_id"],
            ["competition_profiles.id"],
            name=op.f("fk_submission_snapshots_profile_record_id_competition_profiles"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
            name=op.f("fk_submission_snapshots_project_id_projects"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["submission_check_id"],
            ["submission_checks.id"],
            name=op.f("fk_submission_snapshots_submission_check_id_submission_checks"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["verified_result_id"],
            ["results.id"],
            name=op.f("fk_submission_snapshots_verified_result_id_results"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_submission_snapshots")),
    )
    _indexes(
        "submission_snapshots",
        "project_id",
        "paper_version_record_id",
        "verified_result_id",
        "profile_record_id",
        "submission_check_id",
        "status",
        "manifest_hash",
        "package_hash",
        "artifact_set_digest",
        "snapshot_digest",
    )
    op.create_table(
        "submission_artifacts",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("submission_id", sa.Uuid(), nullable=False),
        sa.Column("role", sa.String(length=32), nullable=False),
        sa.Column("relative_path", sa.String(length=1024), nullable=False),
        sa.Column("mime_type", sa.String(length=255), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("storage_key", sa.String(length=1024), nullable=False),
        sa.Column("artifact_json", JSON_TYPE, nullable=False),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
            name=op.f("fk_submission_artifacts_project_id_projects"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["submission_id"],
            ["submission_snapshots.id"],
            name=op.f("fk_submission_artifacts_submission_id_submission_snapshots"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_submission_artifacts")),
        sa.UniqueConstraint("submission_id", "relative_path", name="uq_submission_artifacts_path"),
    )
    _indexes("submission_artifacts", "project_id", "submission_id", "role", "sha256")
    op.create_table(
        "submission_manifests",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("submission_id", sa.Uuid(), nullable=False),
        sa.Column("manifest_hash", sa.String(length=64), nullable=False),
        sa.Column("package_hash", sa.String(length=64), nullable=False),
        sa.Column("manifest_json", JSON_TYPE, nullable=False),
        sa.ForeignKeyConstraint(
            ["submission_id"],
            ["submission_snapshots.id"],
            name=op.f("fk_submission_manifests_submission_id_submission_snapshots"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_submission_manifests")),
    )
    op.create_index(
        op.f("ix_submission_manifests_submission_id"),
        "submission_manifests",
        ["submission_id"],
        unique=True,
    )
    _indexes("submission_manifests", "manifest_hash", "package_hash")
    op.create_table(
        "correction_plans",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("submission_check_id", sa.Uuid(), nullable=True),
        sa.Column("scope", sa.String(length=32), nullable=False),
        sa.Column("priority", sa.Integer(), nullable=False),
        sa.Column("plan_json", JSON_TYPE, nullable=False),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
            name=op.f("fk_correction_plans_project_id_projects"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["submission_check_id"],
            ["submission_checks.id"],
            name=op.f("fk_correction_plans_submission_check_id_submission_checks"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_correction_plans")),
    )
    _indexes("correction_plans", "project_id", "submission_check_id", "scope")


def downgrade() -> None:
    for table in (
        "correction_plans",
        "submission_manifests",
        "submission_artifacts",
        "submission_snapshots",
        "submission_checks",
        "jury_findings",
        "final_jury_reports",
        "requirement_coverage",
        "competition_rules",
        "competition_profiles",
    ):
        op.drop_table(table)


def _indexes(table: str, *columns: str) -> None:
    for column in columns:
        op.create_index(op.f(f"ix_{table}_{column}"), table, [column], unique=False)
