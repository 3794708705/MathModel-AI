from __future__ import annotations

import os
from pathlib import Path
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from mathmodel_ai.core.config import Settings
from mathmodel_ai.core.types import Environment, ProviderName
from mathmodel_ai.db.base import Base
from mathmodel_ai.db.models import AgentRunRecord
from mathmodel_ai.main import create_app
from mathmodel_ai.mathematical.expressions import evaluate_expression
from mathmodel_ai.providers.factory import build_provider_registry
from mathmodel_ai.schemas.mathematical import MathematicalModel, ObjectiveSense
from mathmodel_ai.schemas.model_selection import ModelFamily
from mathmodel_ai.schemas.problem_state import WorkflowStage
from mathmodel_ai.solvers.feasibility import check_feasibility


def _has_any_live_credential() -> bool:
    settings = Settings()
    return any(
        secret is not None
        for secret in (
            settings.openai_api_key,
            settings.google_api_key,
            settings.anthropic_api_key,
        )
    )


pytestmark = [
    pytest.mark.integration,
    pytest.mark.solver,
    pytest.mark.skipif(
        os.getenv("MM_RUN_LIVE_PROVIDER_TESTS") != "1",
        reason="paid live-provider tests require explicit MM_RUN_LIVE_PROVIDER_TESTS=1",
    ),
    pytest.mark.skipif(
        not _has_any_live_credential(),
        reason="no live provider credentials are configured",
    ),
]


def test_live_problem_agent_to_real_solver_evidence_chain(tmp_path: Path) -> None:
    base = Settings()
    assert base.default_provider is not ProviderName.MOCK
    configured_key = {
        ProviderName.OPENAI: base.openai_api_key,
        ProviderName.GOOGLE: base.google_api_key,
        ProviderName.ANTHROPIC: base.anthropic_api_key,
    }[base.default_provider]
    assert configured_key is not None
    assert base.default_provider_model != "mock-foundation"
    catalog = base.model_catalog.model_copy(
        update={
            name: target.model_copy(
                update={
                    "provider": base.default_provider,
                    "model": base.default_provider_model,
                }
            )
            for name, target in base.model_catalog
        }
    )
    settings = base.model_copy(
        update={
            "environment": Environment.TEST,
            "database_url": "sqlite+pysqlite:///:memory:",
            "reasoning_max_retries": 2,
            "model_catalog": catalog,
            "storage_root": tmp_path / "storage",
            "sandbox_root": tmp_path / "sandbox",
            "solver_sandbox_root": tmp_path / "solver-sandbox",
            "sandbox_memory_mb": 768,
            "sandbox_timeout_seconds": 30,
        }
    )
    registry = build_provider_registry(settings)
    app = create_app(settings, providers=registry)
    Base.metadata.create_all(app.state.engine)

    with TestClient(app) as client:
        created = client.post(
            "/api/v1/projects",
            json={
                "name": "Live Phase 4 LP acceptance",
                "title": "Minimum-cost allocation",
                "raw_problem": (
                    "Minimize 3x + 4y subject to x + y >= 10, x >= 0, y >= 0. "
                    "Construct an exact linear optimization model without adding assumptions."
                ),
            },
        )
        assert created.status_code == 201, created.text
        project_id = created.json()["project_id"]

        reasoning = client.post(
            f"/api/v1/projects/{project_id}/reasoning/run",
            json={"user_guidance": ["Interpret every inequality literally."]},
        )
        assert reasoning.status_code == 200, reasoning.text
        assert reasoning.json()["state"]["current_stage"] == "SELECT"
        assert reasoning.json()["is_mock"] is False

        response = client.post(
            f"/api/v1/projects/{project_id}/mathematical/run",
            json={
                "execution_strategy": "DETERMINISTIC",
                "user_guidance": [
                    "Use variables x and y and preserve the exact coefficients and inequality."
                ],
            },
        )
        assert response.status_code == 200, response.text
        payload = response.json()
        model = MathematicalModel.model_validate(payload["model_stage"]["mathematical_model"])
        solve = payload["solve_stage"]

        assert payload["model_stage"]["is_mock"] is False
        assert payload["model_stage"]["gate"]["status"] == "PASS"
        assert model.model_family is ModelFamily.LINEAR_PROGRAMMING
        assert model.objective is not None
        assert model.objective.sense is ObjectiveSense.MINIMIZE
        assert evaluate_expression(model.objective.expression, {"x": 10, "y": 0}) == pytest.approx(
            30
        )
        assert evaluate_expression(model.objective.expression, {"x": 0, "y": 10}) == pytest.approx(
            40
        )
        feasible = check_feasibility(model, {"x": 10, "y": 0}, tolerance=1e-7)
        infeasible = check_feasibility(model, {"x": 0, "y": 0}, tolerance=1e-7)
        assert not feasible.violated_constraints
        assert infeasible.violated_constraints
        assert solve["execution_strategy"]["selected"] == "DETERMINISTIC"
        assert solve["result"]["status"] == "OPTIMAL"
        assert solve["result"]["objective"] == pytest.approx(30, abs=1e-6)
        assert solve["execution"]["is_mock"] is False
        assert solve["execution"]["status"] == "SUCCEEDED"
        assert solve["gate"]["status"] == "PASS"

        evidence = client.get(
            f"/api/v1/projects/{project_id}/results/{solve['result']['result_id']}/evidence"
        )
        assert evidence.status_code == 200, evidence.text
        assert evidence.json()["valid"] is True
        state = app.state.reasoning_repository.load_current(UUID(project_id))
        assert state.current_stage is WorkflowStage.SOLVE

        with Session(app.state.engine) as session:
            runs = list(session.scalars(select(AgentRunRecord)))
            assert len(runs) == 4
            assert all(not run.is_mock for run in runs)
