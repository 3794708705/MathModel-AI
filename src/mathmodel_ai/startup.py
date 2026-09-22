from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TextIO

from sqlalchemy import func, inspect, select
from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import ArgumentError
from sqlalchemy.orm import Session

from mathmodel_ai.core.config import Settings
from mathmodel_ai.core.errors import ConfigurationError
from mathmodel_ai.db.models import EncryptedSecretRecordModel, ProviderEndpointRecordModel
from mathmodel_ai.db.session import (
    create_database_engine,
    create_session_factory,
    database_is_ready,
)
from mathmodel_ai.providers.secrets import (
    CompositeSecretResolver,
    EncryptedDatabaseSecretStore,
    EnvironmentSecretResolver,
)

PREFLIGHT_DATABASE_CONNECT_TIMEOUT_SECONDS = 3


class StartupPreflightError(RuntimeError):
    """A local startup condition that cannot be handled safely."""


class SecretStoreStatus(StrEnum):
    CONFIGURED = "CONFIGURED"
    NOT_CONFIGURED = "NOT_CONFIGURED"
    STORED_SECRETS_UNAVAILABLE = "STORED_SECRETS_UNAVAILABLE"
    DATABASE_UNAVAILABLE = "DATABASE_UNAVAILABLE"


@dataclass(frozen=True)
class StartupPreflightReport:
    python_version: str
    python_executable: Path
    database_url: str
    database_reachable: bool
    secret_store_status: SecretStoreStatus
    stored_secret_count: int | None
    live_provider_configured: bool
    directories: tuple[Path, ...]
    warnings: tuple[str, ...]


def run_startup_preflight(settings: Settings) -> StartupPreflightReport:
    """Inspect local prerequisites without requiring a live model provider."""

    warnings: list[str] = []
    python_version = ".".join(str(item) for item in sys.version_info[:3])
    if sys.version_info[:2] != (3, 12):
        warnings.append(
            f"Python {python_version} is running; the tested local baseline is Python 3.12."
        )
    python_executable = Path(sys.executable).resolve()
    if ".venv" not in {part.lower() for part in python_executable.parts}:
        warnings.append(
            "The repository .venv is not active; use scripts/dev-backend.ps1 to avoid PATH drift."
        )

    directories = _ensure_runtime_directories(settings)
    try:
        parsed_database_url = make_url(settings.database_url)
        database_url = parsed_database_url.render_as_string(hide_password=True)
    except ArgumentError as exc:
        raise StartupPreflightError("MM_DATABASE_URL is invalid") from exc

    try:
        engine = create_database_engine(
            _settings_with_bounded_database_connect_timeout(settings, parsed_database_url)
        )
    except (ArgumentError, ModuleNotFoundError, ValueError) as exc:
        raise StartupPreflightError("MM_DATABASE_URL cannot initialize a database engine") from exc

    session_factory = create_session_factory(engine)
    try:
        try:
            secret_store = EncryptedDatabaseSecretStore(
                session_factory,
                settings.secret_master_key,
            )
        except ConfigurationError as exc:
            raise StartupPreflightError(str(exc)) from exc

        database_reachable = database_is_ready(engine)
        stored_secret_count: int | None = None
        provider_refs: list[tuple[str, str | None]] = []
        if database_reachable:
            inspector = inspect(engine)
            if inspector.has_table(EncryptedSecretRecordModel.__tablename__):
                with Session(engine) as session:
                    stored_secret_count = int(
                        session.scalar(select(func.count()).select_from(EncryptedSecretRecordModel))
                        or 0
                    )
            else:
                warnings.append(
                    "Database is reachable but encrypted_secrets is missing; "
                    "run Alembic migrations."
                )
            if inspector.has_table(ProviderEndpointRecordModel.__tablename__):
                with Session(engine) as session:
                    provider_refs = [
                        (str(provider_id), credential_ref)
                        for provider_id, credential_ref in session.execute(
                            select(
                                ProviderEndpointRecordModel.provider_id,
                                ProviderEndpointRecordModel.credential_ref,
                            ).where(ProviderEndpointRecordModel.enabled.is_(True))
                        )
                    ]
        else:
            warnings.append(
                "Database is unavailable; liveness and API docs can start, "
                "but readiness is blocked."
            )

        secret_store_status = _secret_store_status(
            configured=secret_store.available,
            database_reachable=database_reachable,
            stored_secret_count=stored_secret_count,
        )
        if secret_store_status is SecretStoreStatus.STORED_SECRETS_UNAVAILABLE:
            warnings.append(
                "Encrypted credentials exist but MM_SECRET_MASTER_KEY is not configured; "
                "stored providers remain unavailable."
            )
        elif secret_store_status is SecretStoreStatus.NOT_CONFIGURED:
            warnings.append(
                "MM_SECRET_MASTER_KEY is not configured; encrypted credential entry is unavailable."
            )
        elif secret_store_status is SecretStoreStatus.DATABASE_UNAVAILABLE:
            warnings.append(
                "MM_SECRET_MASTER_KEY is not configured and stored-secret state cannot be checked."
            )

        resolver = CompositeSecretResolver(EnvironmentSecretResolver(), secret_store)
        live_provider_configured = _built_in_live_provider_configured(
            settings
        ) or _registry_live_provider_configured(
            provider_refs,
            resolver,
        )
        if not live_provider_configured:
            warnings.append("Live provider not configured; the Web UI can still start.")
    finally:
        engine.dispose()

    return StartupPreflightReport(
        python_version=python_version,
        python_executable=python_executable,
        database_url=database_url,
        database_reachable=database_reachable,
        secret_store_status=secret_store_status,
        stored_secret_count=stored_secret_count,
        live_provider_configured=live_provider_configured,
        directories=directories,
        warnings=tuple(warnings),
    )


def _settings_with_bounded_database_connect_timeout(
    settings: Settings,
    parsed_database_url: URL,
) -> Settings:
    """Keep the developer preflight responsive when local PostgreSQL is down."""

    database_url = make_url(parsed_database_url)
    if database_url.get_backend_name() != "postgresql":
        return settings
    bounded_url = database_url.update_query_dict(
        {"connect_timeout": str(PREFLIGHT_DATABASE_CONNECT_TIMEOUT_SECONDS)}
    )
    return settings.model_copy(
        update={"database_url": bounded_url.render_as_string(hide_password=False)}
    )


def print_startup_preflight(report: StartupPreflightReport, stream: TextIO | None = None) -> None:
    output = stream or sys.stdout
    print("MathModel AI startup preflight", file=output)
    print(f"  Python: {report.python_version} ({report.python_executable})", file=output)
    database_state = "reachable" if report.database_reachable else "unavailable"
    print(f"  Database: {database_state} ({report.database_url})", file=output)
    print(f"  Secret store: {report.secret_store_status.value}", file=output)
    live_state = "configured" if report.live_provider_configured else "not configured"
    print(f"  Live provider: {live_state}", file=output)
    print(f"  Runtime directories: ready ({len(report.directories)})", file=output)
    for warning in report.warnings:
        print(f"  WARNING: {warning}", file=output)


def _ensure_runtime_directories(settings: Settings) -> tuple[Path, ...]:
    configured = (
        settings.storage_root,
        settings.sandbox_root,
        settings.solver_sandbox_root,
        settings.benchmark_cache_root,
        settings.benchmark_artifact_root,
    )
    resolved: list[Path] = []
    for raw_path in configured:
        path = raw_path.resolve()
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise StartupPreflightError(f"runtime directory cannot be created: {path}") from exc
        if not path.is_dir() or not os.access(path, os.W_OK):
            raise StartupPreflightError(f"runtime directory is not writable: {path}")
        resolved.append(path)
    return tuple(dict.fromkeys(resolved))


def _secret_store_status(
    *, configured: bool, database_reachable: bool, stored_secret_count: int | None
) -> SecretStoreStatus:
    if configured:
        return SecretStoreStatus.CONFIGURED
    if not database_reachable or stored_secret_count is None:
        return SecretStoreStatus.DATABASE_UNAVAILABLE
    if stored_secret_count > 0:
        return SecretStoreStatus.STORED_SECRETS_UNAVAILABLE
    return SecretStoreStatus.NOT_CONFIGURED


def _built_in_live_provider_configured(settings: Settings) -> bool:
    return any(
        secret is not None and bool(secret.get_secret_value().strip())
        for secret in (
            settings.openai_api_key,
            settings.google_api_key,
            settings.anthropic_api_key,
        )
    )


def _registry_live_provider_configured(
    provider_refs: list[tuple[str, str | None]],
    resolver: CompositeSecretResolver,
) -> bool:
    for provider_id, credential_ref in provider_refs:
        if provider_id == "mock":
            continue
        try:
            if resolver.is_configured(credential_ref):
                return True
        except ConfigurationError:
            continue
    return False
