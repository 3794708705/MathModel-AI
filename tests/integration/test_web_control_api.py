from fastapi.testclient import TestClient

from mathmodel_ai.core.config import Settings
from mathmodel_ai.core.types import Environment
from mathmodel_ai.db.base import Base
from mathmodel_ai.main import create_app
from mathmodel_ai.providers.security import EndpointSecurityPolicy
from mathmodel_ai.schemas.benchmark import (
    BenchmarkConfig,
    BenchmarkPricing,
    BenchmarkRun,
    BenchmarkRunStatus,
    benchmark_run_digest,
)

ZERO = "0" * 64


def _app(tmp_path) -> object:
    app = create_app(
        Settings(
            environment="test",
            database_url="sqlite+pysqlite:///:memory:",
            default_provider="mock",
            storage_root=tmp_path / "storage",
            benchmark_artifact_root=tmp_path / "benchmark-runs",
        ),
        provider_security_policy=EndpointSecurityPolicy(
            environment=Environment.TEST,
            resolver=lambda _host, _port: ["93.184.216.34"],
        ),
    )
    Base.metadata.create_all(app.state.engine)
    return app


def _provider_payload() -> dict[str, object]:
    return {
        "provider_id": "ui-provider",
        "display_name": "UI Provider",
        "protocol": "openai_chat_completions",
        "base_url": "https://ui-provider.example.test/v1",
        "trust_level": "USER_MANAGED_PROXY",
    }


def _model_payload() -> dict[str, object]:
    return {
        "model_id": "ui-model",
        "provider_id": "ui-provider",
        "display_name": "UI Model",
        "remote_model": "remote-ui-model",
        "quality_tier": "FLAGSHIP_MAX",
        "structured_output_strategy": "NATIVE_JSON_SCHEMA",
    }


def test_project_list_and_current_state_are_backend_backed(tmp_path) -> None:
    app = _app(tmp_path)
    payload = {
        "name": "First project",
        "title": "Allocation planning",
        "raw_problem": "Determine a feasible allocation plan from the supplied demand data.",
    }
    with TestClient(app) as client:
        created = client.post("/api/v1/projects", json=payload)
        project_id = created.json()["project_id"]
        listed = client.get("/api/v1/projects")
        current = client.get(f"/api/v1/projects/{project_id}")
    assert listed.status_code == 200
    assert listed.json() == [
        {
            "project_id": project_id,
            "name": "First project",
            "title": "Allocation planning",
            "current_stage": "INGEST",
            "status": "PENDING",
            "version": 0,
            "created_at": created.json()["created_at"],
            "updated_at": created.json()["updated_at"],
        }
    ]
    assert current.status_code == 200
    assert current.json()["project_id"] == project_id
    assert current.json()["raw_problem"] == payload["raw_problem"]


def test_runtime_default_remains_a_router_preference_not_a_bypass(tmp_path) -> None:
    app = _app(tmp_path)
    with TestClient(app) as client:
        client.post("/api/v1/providers", json=_provider_payload())
        client.post("/api/v1/models", json=_model_payload())
        saved = client.put("/api/v1/model-routing/default", json={"model_id": "ui-model"})
        overview = client.get("/api/v1/model-routing")
        preview = client.post(
            "/api/v1/model-routing/preview",
            json={"profile": {"task_type": "documentation"}},
        )
        catalog = client.get("/api/v1/model-routing/agent-catalog")
        assert saved.status_code == 200
        assert overview.json()["default_model_id"] == "ui-model"
        assert preview.json()["decision"]["action"] == "HUMAN_REVIEW"
        assert "credential is not configured" in str(
            preview.json()["decision"]["rejected_models"]["ui-model"]
        )
        names = {item["name"] for item in catalog.json()}
        assert {"problem_agent", "data_agent", "final_jury_agent"} <= names
        assert "__default__" not in names
        assert client.get("/api/v1/model-routing/agents").json() == []


def test_benchmark_history_and_case_catalog_are_read_only_views(tmp_path) -> None:
    app = _app(tmp_path)
    run = BenchmarkRun(
        code_commit="a" * 40,
        source_tree_digest="b" * 64,
        working_tree_dirty=True,
        config=BenchmarkConfig(
            provider="unconfigured-live-provider",
            model="unconfigured-live-model",
            reasoning_tier="high",
            pricing=BenchmarkPricing(version="ui-test"),
        ),
        case_manifest_digests={"BENCH-ui": "c" * 64},
        status=BenchmarkRunStatus.RUNNING,
        run_digest=ZERO,
    )
    run = run.model_copy(update={"run_digest": benchmark_run_digest(run)})
    app.state.benchmark_repository.create_run(run)
    with TestClient(app) as client:
        history = client.get("/api/v1/benchmarks/runs")
        cases = client.get("/api/v1/benchmarks/cases")
    assert history.status_code == 200
    assert [item["run_id"] for item in history.json()] == [str(run.run_id)]
    assert history.json()[0]["status"] == "RUNNING"
    assert cases.status_code == 200
    assert all("benchmark_id" in item and "title" in item for item in cases.json())


def test_cors_allows_only_configured_frontend_origin(tmp_path) -> None:
    app = _app(tmp_path)
    with TestClient(app) as client:
        allowed = client.options(
            "/api/v1/providers",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
            },
        )
        rejected = client.options(
            "/api/v1/providers",
            headers={
                "Origin": "https://attacker.example",
                "Access-Control-Request-Method": "GET",
            },
        )
    assert allowed.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert "access-control-allow-credentials" not in allowed.headers
    assert "access-control-allow-origin" not in rejected.headers
