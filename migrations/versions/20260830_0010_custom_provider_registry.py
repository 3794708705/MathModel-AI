"""add configurable provider and model registry

Revision ID: 20260830_0010
Revises: 20260830_0009
Create Date: 2026-08-30 18:00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260830_0010"
down_revision: str | None = "20260830_0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

JSON_TYPE = sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql")


def upgrade() -> None:
    op.create_table(
        "provider_endpoints",
        sa.Column("provider_id", sa.String(length=100), nullable=False),
        sa.Column("display_name", sa.String(length=255), nullable=False),
        sa.Column("protocol", sa.String(length=50), nullable=False),
        sa.Column("base_url", sa.String(length=2048), nullable=False),
        sa.Column("credential_ref", sa.String(length=132), nullable=True),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("trust_level", sa.String(length=40), nullable=False),
        sa.Column("timeout_seconds", sa.Float(), nullable=False),
        sa.Column("connect_timeout_seconds", sa.Float(), nullable=False),
        sa.Column("max_response_bytes", sa.BigInteger(), nullable=False),
        sa.Column("verify_tls", sa.Boolean(), nullable=False),
        sa.Column("allow_redirects", sa.Boolean(), nullable=False),
        sa.Column("additional_headers", JSON_TYPE, nullable=False),
        sa.Column("metadata", JSON_TYPE, nullable=False),
        sa.Column("health_status", sa.String(length=32), nullable=False),
        sa.Column("consecutive_failures", sa.Integer(), nullable=False),
        sa.Column("cooldown_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("config_digest", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("provider_id", name=op.f("pk_provider_endpoints")),
    )
    _indexes(
        "provider_endpoints",
        "protocol",
        "enabled",
        "trust_level",
        "health_status",
        "config_digest",
    )
    op.create_table(
        "model_profiles",
        sa.Column("model_id", sa.String(length=100), nullable=False),
        sa.Column("provider_id", sa.String(length=100), nullable=False),
        sa.Column("display_name", sa.String(length=255), nullable=False),
        sa.Column("remote_model", sa.String(length=500), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("quality_tier", sa.String(length=32), nullable=False),
        sa.Column("quality_tier_source", sa.String(length=32), nullable=False),
        sa.Column("declared_capabilities", JSON_TYPE, nullable=False),
        sa.Column("observed_capabilities", JSON_TYPE, nullable=False),
        sa.Column("structured_output_strategy", sa.String(length=40), nullable=False),
        sa.Column("tool_calling_strategy", sa.String(length=32), nullable=False),
        sa.Column("reasoning_mapping", JSON_TYPE, nullable=False),
        sa.Column("context_window", sa.Integer(), nullable=True),
        sa.Column("max_output_tokens", sa.Integer(), nullable=True),
        sa.Column("pricing", JSON_TYPE, nullable=False),
        sa.Column("trust_level", sa.String(length=40), nullable=False),
        sa.Column("configuration", JSON_TYPE, nullable=False),
        sa.Column("config_digest", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["provider_id"],
            ["provider_endpoints.provider_id"],
            name=op.f("fk_model_profiles_provider_id_provider_endpoints"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("model_id", name=op.f("pk_model_profiles")),
    )
    _indexes(
        "model_profiles",
        "provider_id",
        "enabled",
        "quality_tier",
        "trust_level",
        "config_digest",
    )
    op.create_table(
        "capability_probe_runs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("provider_id", sa.String(length=100), nullable=False),
        sa.Column("model_id", sa.String(length=100), nullable=False),
        sa.Column("provider_config_digest", sa.String(length=64), nullable=False),
        sa.Column("model_config_digest", sa.String(length=64), nullable=False),
        sa.Column("probe_digest", sa.String(length=64), nullable=False),
        sa.Column("capabilities", JSON_TYPE, nullable=False),
        sa.Column("authentication_status", sa.String(length=32), nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column("usage", JSON_TYPE, nullable=False),
        sa.Column("errors", JSON_TYPE, nullable=False),
        sa.Column("performed_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["provider_id"],
            ["provider_endpoints.provider_id"],
            name=op.f("fk_capability_probe_runs_provider_id_provider_endpoints"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["model_id"],
            ["model_profiles.model_id"],
            name=op.f("fk_capability_probe_runs_model_id_model_profiles"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_capability_probe_runs")),
    )
    _indexes(
        "capability_probe_runs",
        "provider_id",
        "model_id",
        "provider_config_digest",
        "model_config_digest",
        "probe_digest",
        "authentication_status",
    )
    op.create_table(
        "agent_route_policies",
        sa.Column("agent_name", sa.String(length=100), nullable=False),
        sa.Column("primary_model_id", sa.String(length=100), nullable=False),
        sa.Column("fallback_model_ids", JSON_TYPE, nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["primary_model_id"],
            ["model_profiles.model_id"],
            name=op.f("fk_agent_route_policies_primary_model_id_model_profiles"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("agent_name", name=op.f("pk_agent_route_policies")),
    )
    op.create_index(
        op.f("ix_agent_route_policies_primary_model_id"),
        "agent_route_policies",
        ["primary_model_id"],
        unique=False,
    )
    _add_trace_columns()


def downgrade() -> None:
    _drop_trace_columns()
    op.drop_table("agent_route_policies")
    op.drop_table("capability_probe_runs")
    op.drop_table("model_profiles")
    op.drop_table("provider_endpoints")


def _add_trace_columns() -> None:
    agent_columns = (
        ("provider_id", sa.String(length=100), True),
        ("provider_config_digest", sa.String(length=64), True),
        ("model_id", sa.String(length=100), True),
        ("remote_model", sa.String(length=500), False),
        ("model_config_digest", sa.String(length=64), True),
        ("protocol", sa.String(length=50), False),
        ("structured_output_mode", sa.String(length=40), False),
        ("reasoning_requested", sa.String(length=32), False),
        ("reasoning_effective", sa.String(length=100), False),
        ("endpoint_trust", sa.String(length=40), False),
    )
    for name, type_, indexed in agent_columns:
        op.add_column("agent_runs", sa.Column(name, type_, nullable=True))
        if indexed:
            op.create_index(op.f(f"ix_agent_runs_{name}"), "agent_runs", [name], unique=False)
    benchmark_columns = (
        ("provider_id", sa.String(length=100), True),
        ("provider_config_digest", sa.String(length=64), True),
        ("model_id", sa.String(length=100), True),
        ("model_config_digest", sa.String(length=64), True),
        ("protocol", sa.String(length=50), False),
        ("endpoint_trust", sa.String(length=40), False),
        ("model_identity_confidence", sa.String(length=40), False),
    )
    for name, type_, indexed in benchmark_columns:
        op.add_column("benchmark_runs", sa.Column(name, type_, nullable=True))
        if indexed:
            op.create_index(
                op.f(f"ix_benchmark_runs_{name}"), "benchmark_runs", [name], unique=False
            )


def _drop_trace_columns() -> None:
    for table, columns, indexed in (
        (
            "benchmark_runs",
            (
                "provider_id",
                "provider_config_digest",
                "model_id",
                "model_config_digest",
                "protocol",
                "endpoint_trust",
                "model_identity_confidence",
            ),
            {"provider_id", "provider_config_digest", "model_id", "model_config_digest"},
        ),
        (
            "agent_runs",
            (
                "provider_id",
                "provider_config_digest",
                "model_id",
                "remote_model",
                "model_config_digest",
                "protocol",
                "structured_output_mode",
                "reasoning_requested",
                "reasoning_effective",
                "endpoint_trust",
            ),
            {"provider_id", "provider_config_digest", "model_id", "model_config_digest"},
        ),
    ):
        for name in reversed(columns):
            if name in indexed:
                op.drop_index(op.f(f"ix_{table}_{name}"), table_name=table)
            op.drop_column(table, name)


def _indexes(table: str, *columns: str) -> None:
    for column in columns:
        op.create_index(op.f(f"ix_{table}_{column}"), table, [column], unique=False)
