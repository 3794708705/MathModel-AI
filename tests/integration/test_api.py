import pytest
from fastapi.testclient import TestClient

from mathmodel_ai.core.config import Settings
from mathmodel_ai.main import create_app


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
