import pytest
from fastapi.testclient import TestClient

from mathmodel_ai.core.config import Settings
from mathmodel_ai.db.base import Base
from mathmodel_ai.main import create_app
from mathmodel_ai.submission.profiles import generic_modeling_test_profile


@pytest.mark.integration
def test_database_outage_is_explicit_and_does_not_change_liveness(monkeypatch) -> None:
    app = create_app(Settings(environment="test", database_url="sqlite+pysqlite:///:memory:"))
    with TestClient(app) as client:
        monkeypatch.setattr(
            "mathmodel_ai.api.routes.health.database_is_ready", lambda _engine: False
        )
        response = client.get("/health/ready")
        assert response.status_code == 503
        assert response.json() == {
            "detail": {
                "code": "DATABASE_UNAVAILABLE",
                "message": "Database is unavailable. Check PostgreSQL and MM_DATABASE_URL.",
            }
        }
        assert client.get("/health/live").json() == {"status": "ok"}
        monkeypatch.setattr(
            "mathmodel_ai.api.routes.health.database_is_ready", lambda _engine: True
        )
        assert client.get("/health/ready").json() == {"status": "ready", "database": "ready"}


@pytest.mark.integration
def test_health_and_system_endpoints_do_not_expose_secrets() -> None:
    settings = Settings(
        environment="test",
        database_url="sqlite+pysqlite:///:memory:",
        openai_api_key="must-not-leak",
        default_provider="mock",
    )
    app = create_app(settings)
    with TestClient(app) as client:
        assert client.get("/health/live").json() == {"status": "ok"}
        assert client.get("/health/ready").json() == {
            "status": "ready",
            "database": "ready",
        }
        response = client.get("/api/v1/system")
    assert response.status_code == 200
    assert response.json()["configured_providers"] == ["mock", "openai"]
    assert "must-not-leak" not in response.text


@pytest.mark.integration
def test_phase7_profile_versioning_and_final_routes(tmp_path) -> None:
    app = create_app(
        Settings(
            environment="test",
            database_url="sqlite+pysqlite:///:memory:",
            default_provider="mock",
            storage_root=tmp_path / "storage",
        )
    )
    Base.metadata.create_all(app.state.engine)
    profile_v1 = generic_modeling_test_profile()
    profile_v2 = profile_v1.model_copy(
        update={"version": 2, "name": "GENERIC_MODELING_TEST_PROFILE_V2"}
    )
    with TestClient(app) as client:
        created = client.post(
            "/api/v1/competition-profiles", json=profile_v2.model_dump(mode="json")
        )
        assert created.status_code == 200, created.text
        listed = client.get("/api/v1/competition-profiles")
        assert {(item["profile_id"], item["version"]) for item in listed.json()} >= {
            (str(profile_v1.profile_id), 1),
            (str(profile_v2.profile_id), 2),
        }
        explicit = client.get(f"/api/v1/competition-profiles/{profile_v1.profile_id}?version=1")
        assert explicit.json()["version"] == 1
        latest = client.get(f"/api/v1/competition-profiles/{profile_v1.profile_id}")
        assert latest.json()["version"] == 2

    paths = set(app.openapi()["paths"])
    assert {
        "/api/v1/projects/{project_id}/final/run",
        "/api/v1/projects/{project_id}/final-jury",
        "/api/v1/projects/{project_id}/submission/check",
        "/api/v1/projects/{project_id}/submission/freeze",
        "/api/v1/projects/{project_id}/submission/build",
        "/api/v1/projects/{project_id}/submission",
        "/api/v1/projects/{project_id}/submission/artifacts",
    } <= paths
