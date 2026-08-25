from uuid import UUID

from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from mathmodel_ai.core.config import Settings
from mathmodel_ai.db.base import Base
from mathmodel_ai.db.models import AgentRunRecord, ModelDecisionRecord, ProblemStateRecord
from mathmodel_ai.main import create_app
from mathmodel_ai.providers.factory import ProviderRegistry
from mathmodel_ai.providers.mock import MockProvider
from mathmodel_ai.schemas.problem_state import ProblemState, WorkflowStage
from tests.reasoning.helpers import analysis_fixture, exploration_fixture, jury_fixture


def test_mock_reasoning_api_persists_complete_phase_two_chain() -> None:
    mock = MockProvider(
        [
            analysis_fixture(low_confidence_ambiguity=True).model_dump_json(),
            exploration_fixture().model_dump_json(),
            jury_fixture().model_dump_json(),
        ]
    )
    app = create_app(
        Settings(
            environment="test",
            database_url="sqlite+pysqlite:///:memory:",
            default_provider="mock",
            reasoning_max_retries=0,
        ),
        providers=ProviderRegistry([mock]),
    )
    Base.metadata.create_all(app.state.engine)

    with TestClient(app) as client:
        created = client.post(
            "/api/v1/projects",
            json={
                "name": "Phase 2 benchmark",
                "title": "Demand allocation planning",
                "raw_problem": (
                    "Forecast next-period demand, optimize allocation under capacity, "
                    "and evaluate the resulting plan."
                ),
                "competition": "Fixture Competition",
            },
        )
        assert created.status_code == 201, created.text
        project_id = created.json()["project_id"]
        project_uuid = UUID(project_id)
        assert created.json()["version"] == 0

        response = client.post(
            f"/api/v1/projects/{project_id}/reasoning/run",
            json={
                "user_notes": [],
                "user_guidance": ["Prefer traceable models."],
                "jury_notes": ["Respect the short competition deadline."],
            },
        )
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["is_mock"] is True
        assert payload["state"]["version"] == 3
        assert payload["state"]["current_stage"] == "SELECT"
        assert payload["state"]["selected_model"]["candidate_id"] == "CAND-chain"
        assert payload["state"]["backup_model"]["candidate_id"] == "CAND-regression"
        assert [run["output_state_version"] for run in payload["agent_runs"]] == [1, 2, 3]
        assert all(run["is_mock"] for run in payload["agent_runs"])
        assert all(run["prompt_version"] == "2.0.0" for run in payload["agent_runs"])

        analysis = client.get(f"/api/v1/projects/{project_id}/problem-analysis")
        models = client.get(f"/api/v1/projects/{project_id}/models")
        selection = client.get(f"/api/v1/projects/{project_id}/model-selection")
        assert analysis.status_code == models.status_code == selection.status_code == 200
        assert analysis.json()["human_review_recommended"] is True
        assert len(models.json()) == 3
        assert selection.json()["selected_model_id"] == "CAND-chain"

        revisions = app.state.reasoning_repository.list_state_revisions(project_uuid)
        assert [state.version for state in revisions] == [0, 1, 2, 3]
        assert [state.current_stage for state in revisions] == [
            WorkflowStage.INGEST,
            WorkflowStage.UNDERSTAND,
            WorkflowStage.EXPLORE,
            WorkflowStage.SELECT,
        ]
        restored = app.state.reasoning_repository.load_current(project_uuid)
        assert ProblemState.model_validate_json(restored.model_dump_json()) == restored

        with Session(app.state.engine) as session:
            assert session.scalar(select(func.count()).select_from(AgentRunRecord)) == 3
            assert session.scalar(select(func.count()).select_from(ModelDecisionRecord)) == 1
            assert session.scalar(select(func.count()).select_from(ProblemStateRecord)) == 4
            decision = session.scalar(select(ModelDecisionRecord))
            assert decision is not None
            assert decision.decision_json["selected_model_id"] == "CAND-chain"


def test_reasoning_api_rejects_skipped_selection_stage() -> None:
    app = create_app(
        Settings(environment="test", database_url="sqlite+pysqlite:///:memory:"),
        providers=ProviderRegistry([MockProvider()]),
    )
    Base.metadata.create_all(app.state.engine)
    with TestClient(app) as client:
        created = client.post(
            "/api/v1/projects",
            json={
                "name": "Invalid transition",
                "title": "Skip test",
                "raw_problem": "This sufficiently long problem statement tests stage skipping.",
            },
        )
        project_id = created.json()["project_id"]
        response = client.post(f"/api/v1/projects/{project_id}/models/select", json={"notes": []})
    assert response.status_code == 400
    assert "INGEST -> SELECT" in response.json()["detail"]


def test_individual_reasoning_stage_endpoints_advance_one_version_each() -> None:
    mock = MockProvider(
        [
            analysis_fixture().model_dump_json(),
            exploration_fixture().model_dump_json(),
            jury_fixture().model_dump_json(),
        ]
    )
    app = create_app(
        Settings(
            environment="test",
            database_url="sqlite+pysqlite:///:memory:",
            reasoning_max_retries=0,
        ),
        providers=ProviderRegistry([mock]),
    )
    Base.metadata.create_all(app.state.engine)
    with TestClient(app) as client:
        created = client.post(
            "/api/v1/projects",
            json={
                "name": "Stage endpoints",
                "title": "Three-stage test",
                "raw_problem": "Forecast demand, optimize capacity, and evaluate the final plan.",
            },
        )
        project_id = created.json()["project_id"]
        analyze = client.post(f"/api/v1/projects/{project_id}/problem/analyze", json={"notes": []})
        explore = client.post(f"/api/v1/projects/{project_id}/models/explore", json={"notes": []})
        select_model = client.post(
            f"/api/v1/projects/{project_id}/models/select", json={"notes": []}
        )

    assert [
        analyze.json()["state_version"],
        explore.json()["state_version"],
        select_model.json()["state_version"],
    ] == [1, 2, 3]
    assert all(response.status_code == 200 for response in (analyze, explore, select_model))
