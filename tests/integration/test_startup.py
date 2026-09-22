from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

from mathmodel_ai.core.config import Settings
from mathmodel_ai.db.base import Base
from mathmodel_ai.db.models import EncryptedSecretRecordModel
from mathmodel_ai.main import create_app
from mathmodel_ai.startup import (
    PREFLIGHT_DATABASE_CONNECT_TIMEOUT_SECONDS,
    SecretStoreStatus,
    _settings_with_bounded_database_connect_timeout,
    print_startup_preflight,
    run_startup_preflight,
)


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        environment="test",
        database_url=f"sqlite+pysqlite:///{(tmp_path / 'startup.db').as_posix()}",
        default_provider="mock",
        storage_root=tmp_path / "storage",
        sandbox_root=tmp_path / "sandbox",
        solver_sandbox_root=tmp_path / "solver",
        benchmark_cache_root=tmp_path / "benchmark-cache",
        benchmark_artifact_root=tmp_path / "benchmark-runs",
    )


def test_factory_creates_fastapi_openapi_docs_and_liveness(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    app = create_app(settings)
    Base.metadata.create_all(app.state.engine)

    assert isinstance(app, FastAPI)
    assert callable(create_app)
    assert "/health/live" in app.openapi()["paths"]
    with TestClient(app) as client:
        assert client.get("/openapi.json").status_code == 200
        assert client.get("/docs").status_code == 200
        assert client.get("/health/live").json() == {"status": "ok"}
        system = client.get("/api/v1/system")
    assert system.status_code == 200
    assert system.json()["secret_store_configured"] is False


def test_preflight_allows_new_database_without_master_key(tmp_path: Path, capsys) -> None:
    settings = _settings(tmp_path)
    engine = create_engine(settings.database_url)
    Base.metadata.create_all(engine)
    engine.dispose()

    report = run_startup_preflight(settings)
    print_startup_preflight(report)
    output = capsys.readouterr().out

    assert report.database_reachable is True
    assert report.secret_store_status is SecretStoreStatus.NOT_CONFIGURED
    assert report.stored_secret_count == 0
    assert report.live_provider_configured is False
    assert "Live provider not configured" in output
    assert "MM_SECRET_MASTER_KEY is not configured" in output


def test_preflight_warns_when_encrypted_rows_exist_without_master_key(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    engine = create_engine(settings.database_url)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(
            EncryptedSecretRecordModel(
                secret_id="provider/startup-test/0123456789ab",
                ciphertext=b"authenticated-ciphertext",
                nonce=b"0123456789ab",
            )
        )
        session.commit()
    engine.dispose()

    report = run_startup_preflight(settings)

    assert report.secret_store_status is SecretStoreStatus.STORED_SECRETS_UNAVAILABLE
    assert report.stored_secret_count == 1
    assert any("stored providers remain unavailable" in warning for warning in report.warnings)


def test_preflight_never_prints_master_key(tmp_path: Path, capsys) -> None:
    key = "c3RhcnR1cC1yZWxpYWJpbGl0eS10ZXN0LWtleS0wMDA="
    settings = _settings(tmp_path).model_copy(update={"secret_master_key": SecretStr(key)})
    engine = create_engine(settings.database_url)
    Base.metadata.create_all(engine)
    engine.dispose()

    report = run_startup_preflight(settings)
    print_startup_preflight(report)

    assert report.secret_store_status is SecretStoreStatus.CONFIGURED
    assert key not in capsys.readouterr().out


def test_preflight_bounds_postgresql_connect_wait_without_mutating_runtime_settings(
    tmp_path: Path,
) -> None:
    database_url = (
        "postgresql+psycopg://mathmodel:mathmodel@localhost:5432/mathmodel?connect_timeout=90"
    )
    settings = _settings(tmp_path).model_copy(update={"database_url": database_url})

    preflight_settings = _settings_with_bounded_database_connect_timeout(
        settings,
        make_url(database_url),
    )

    assert settings.database_url == database_url
    assert make_url(preflight_settings.database_url).query["connect_timeout"] == str(
        PREFLIGHT_DATABASE_CONNECT_TIMEOUT_SECONDS
    )
