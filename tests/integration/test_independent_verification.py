"""Mock reasoning scaffolds state only; numeric/replay evidence is real Docker."""

import hashlib
from io import BytesIO
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from mathmodel_ai.core.config import Settings
from mathmodel_ai.db.base import Base
from mathmodel_ai.db.models import (
    IndependentVerificationJobRecord,
    IndependentVerificationPlanRecord,
)
from mathmodel_ai.main import create_app
from mathmodel_ai.mathematical.digests import mathematical_model_digest
from mathmodel_ai.providers.factory import ProviderRegistry
from mathmodel_ai.providers.mock import MockProvider
from mathmodel_ai.schemas.independent_verification import (
    CsvObservationSpec,
    MetricSpec,
    ScenarioSpec,
    VerificationPlan,
    VerificationRequirements,
)
from mathmodel_ai.schemas.mathematical import MathematicalModelDraft
from mathmodel_ai.verification.independent_repository import IndependentVerificationRepository
from mathmodel_ai.verification.metric_recompute import content_digest
from tests.benchmark.test_repository import _running_attempt, _running_run
from tests.reasoning.helpers import analysis_fixture, exploration_fixture, jury_fixture
from tests.verification.helpers import phase5_model


def create_verification_app(tmp_path, database_url="sqlite+pysqlite:///:memory:"):
    draft = MathematicalModelDraft.model_validate(
        phase5_model().model_dump(
            exclude={
                "model_id",
                "project_id",
                "problem_id",
                "version",
                "source_selected_model_id",
                "status",
            }
        )
    )
    mock = MockProvider(
        [
            analysis_fixture().model_dump_json(),
            exploration_fixture().model_dump_json(),
            jury_fixture().model_dump_json(),
            draft.model_dump_json(),
        ]
    )
    app = create_app(
        Settings(
            environment="test",
            database_url=database_url,
            reasoning_max_retries=0,
            default_provider="mock",
            storage_root=tmp_path / "store",
            sandbox_root=tmp_path / "sandbox",
            solver_sandbox_root=tmp_path / "solver",
            solver_sandbox_image="mathmodel-ai-solver:phase4",
            sandbox_memory_mb=768,
            sandbox_timeout_seconds=30,
            sandbox_max_artifact_bytes=1048576,
        ),
        providers=ProviderRegistry([mock]),
    )
    if database_url.startswith("sqlite"):
        Base.metadata.create_all(app.state.engine)
    return app


def prepare_formal_plan(app, client):
    created = client.post(
        "/api/v1/projects",
        json={
            "name": "Independent replay fixture",
            "title": "Demand allocation",
            "raw_problem": "Minimize allocation cost with sourced demand.",
        },
    )
    assert created.status_code == 201, created.text
    project_id = UUID(created.json()["project_id"])
    assert client.post(f"/api/v1/projects/{project_id}/reasoning/run", json={}).status_code == 200
    solved = client.post(f"/api/v1/projects/{project_id}/mathematical/run", json={})
    assert solved.status_code == 200, solved.text
    context = app.state.mathematical_repository.get_result_context(project_id)
    assert context.evidence.valid
    benchmarks = app.state.benchmark_repository
    run = _running_run()
    benchmarks.create_run(run)
    attempt = _running_attempt(run, 1)
    benchmarks.create_attempt(attempt)
    attempt = benchmarks.bind_attempt_project(attempt.attempt_id, project_id)
    artifact = next(
        a
        for a in app.state.data_repository.list_artifacts(project_id)
        if a.name == "result.json" and a.execution_run_id == context.execution.run_id
    )
    model_digest = mathematical_model_digest(context.model)
    common = {
        "unit": "dimensionless",
        "tolerance_provenance": "synthetic fixture numerical contract",
        "model_binding": model_digest,
    }
    metrics = [
        MetricSpec(
            metric_id="objective",
            key="objective",
            version="1",
            reported_key="objective",
            quantity="synthetic objective",
            calculation="formal objective AST",
            inputs=["x", "y"],
            **common,
        ),
        MetricSpec(
            metric_id="feasible",
            key="feasible",
            version="1",
            quantity="synthetic feasibility",
            calculation="formal domains and constraints",
            inputs=["x", "y"],
            **common,
        ),
    ]
    spec = ScenarioSpec(
        scenario_id="demand_stress",
        version="1",
        parameter_values={"demand": 11.0},
        metrics=metrics,
        baseline="fixture baseline demand",
        perturbation="demand=11",
        reason="exercise reviewed scenario binding",
        input_changes=["demand"],
        comparison_quantity="objective and feasibility",
        acceptance_criterion="both required metrics pass",
        criterion_provenance="synthetic fixture contract",
    )
    plan = VerificationPlan(
        attempt_id=attempt.attempt_id,
        result_id=context.result.result_id,
        version="1",
        source_artifact_id=artifact.artifact_id,
        source_sha256=artifact.sha256,
        metrics=metrics,
        scenarios=[spec],
        scientific_scope="Explicit synthetic linear allocation acceptance only.",
    )
    policy = VerificationRequirements(
        version="2",
        review_status="REVIEWED",
        production_eligible=True,
        benchmark_id=attempt.benchmark_id,
        manifest_digest=attempt.manifest_digest,
        model_digest=model_digest,
        metrics=metrics,
        scenarios=[spec],
        scientific_scope=plan.scientific_scope,
    )
    service = app.state.independent_verification
    service._requirements = lambda _: policy
    service.register(plan)
    return plan, context, policy


@pytest.mark.integration
@pytest.mark.solver
def test_real_api_recompute_replay_persistence_idempotency_and_tampering(tmp_path):
    app = create_verification_app(tmp_path)
    with TestClient(app) as client:
        plan, context, policy = prepare_formal_plan(app, client)
        service = app.state.independent_verification
        url = f"/api/v1/benchmarks/attempts/{plan.attempt_id}/verification"
        assert client.get(url).json()["status"] == "NOT_READY"
        baseline = client.post(url + "/recompute")
        assert baseline.status_code == 200, baseline.text
        assert baseline.json()["report"]["passed_metrics"] == 2
        assert baseline.json()["status"] == "NOT_READY"
        completed = client.post(url + "/scenarios/demand_stress/replay")
        assert completed.status_code == 200, completed.text
        assert completed.json()["status"] == "PASS", completed.text
        first = completed.json()
        from mathmodel_ai.benchmark.reporting import BenchmarkReportBuilder
        from tests.benchmark.test_evaluation_reporting import _claimed_result

        claimed = _claimed_result(app.state.benchmark_repository.get_attempt(plan.attempt_id))
        claimed = claimed.model_copy(
            update={"verified_result_id": plan.result_id, "status": "PASS"}
        )
        builder = BenchmarkReportBuilder(independent_verifier=service.view)
        assert builder._independent_gate(claimed) == claimed
        previous_failure = claimed.model_copy(
            update={"status": "FAIL", "hard_failures": ["CRITICAL"]}
        )
        assert builder._independent_gate(previous_failure).status == "FAIL"
        assert client.post(url + "/scenarios/demand_stress/replay").json() == first
        assert client.post(url + "/recompute").json() == first
        state = app.state.reasoning_repository.load_current(context.model.project_id)
        assert state.verified_result_id is None
        assert client.post(url + "/scenarios/unknown/replay").status_code == 409
        with app.state.session_factory() as session:
            assert (
                session.scalar(select(func.count()).select_from(IndependentVerificationJobRecord))
                == 2
            )
        service.repository = IndependentVerificationRepository(app.state.session_factory)
        assert client.get(url).json() == first
        old = app.state.mathematical_repository.get_result_context(
            context.model.project_id, plan.result_id
        )
        assert (
            old.model == context.model
            and old.result == context.result
            and old.execution == context.execution
        )
        with pytest.raises(ValueError, match="EXACT_REVIEWED"):
            service.register(plan.model_copy(update={"metrics": [plan.metrics[0]]}))
        service._requirements = lambda _: policy.model_copy(update={"manifest_digest": "f" * 64})
        assert client.get(url).json()["status"] == "NOT_READY"
        service._requirements = lambda _: policy.model_copy(update={"model_digest": "f" * 64})
        mismatch = client.get(url).json()
        assert mismatch["status"] == "NOT_READY"
        assert mismatch["blockers"] == ["STALE_VERIFICATION_POLICY"]
        service._requirements = lambda _: policy.model_copy(update={"problem_sha256": "f" * 64})
        missing_problem = client.get(url).json()
        assert missing_problem["status"] == "NOT_READY"
        assert missing_problem["blockers"] == ["STALE_VERIFICATION_POLICY"]
        service._requirements = lambda _: policy
        with app.state.session_factory() as session:
            row = session.scalar(
                select(IndependentVerificationJobRecord).where(
                    IndependentVerificationJobRecord.operation == "baseline"
                )
            )
            row.payload_json = {"batch": {"metrics": []}}
            session.commit()
        assert client.get(url).json()["status"] == "NOT_READY"
        with app.state.session_factory() as session:
            row = session.get(IndependentVerificationPlanRecord, plan.plan_id)
            row.plan_digest = "f" * 64
            session.commit()
        assert client.get(url).json()["status"] == "NOT_READY"
        assert client.get(f"/api/v1/benchmarks/attempts/{uuid4()}/verification").status_code == 404


@pytest.mark.integration
@pytest.mark.solver
def test_reviewed_policy_auto_binds_exact_result_and_runs_all_obligations(tmp_path):
    app = create_verification_app(tmp_path)
    with TestClient(app) as client:
        _plan, context, policy = prepare_formal_plan(app, client)
        # Exercise the production auto-registration path with a fresh attempt.
        run = _running_run()
        app.state.benchmark_repository.create_run(run)
        attempt = _running_attempt(run, 1)
        app.state.benchmark_repository.create_attempt(attempt)
        app.state.benchmark_repository.bind_attempt_project(
            attempt.attempt_id, context.model.project_id
        )
        rebound = policy.model_copy(update={"manifest_digest": attempt.manifest_digest})
        service = app.state.independent_verification
        service._requirements = lambda _: rebound.model_copy(update={"model_digest": "f" * 64})
        with pytest.raises(ValueError, match="VERIFICATION_POLICY_MODEL_MISMATCH"):
            service.prepare_and_run(attempt.attempt_id, context.result.result_id)
        service._requirements = lambda _: rebound.model_copy(update={"problem_sha256": "f" * 64})
        with pytest.raises(ValueError, match="REVIEWED_PROBLEM_FILE_IS_NOT_UNIQUE"):
            service.prepare_and_run(attempt.attempt_id, context.result.result_id)
        service._requirements = lambda _: rebound
        service.prepare_and_run(attempt.attempt_id, context.result.result_id)
        view = service.view(attempt.attempt_id)
        assert view.status == "PASS"
        assert view.report is not None
        assert view.report.result_id == context.result.result_id
        assert view.report.passed_metrics == view.report.required_metrics == 2
        assert view.report.passed_scenarios == view.report.required_scenarios == 1
        stored, digest = service.repository.get(attempt.attempt_id)
        assert stored.result_id == context.result.result_id
        assert stored.source_artifact_id != context.execution.code_artifact_id
        assert digest == content_digest(rebound)


@pytest.mark.integration
@pytest.mark.solver
def test_reviewed_csv_observations_derive_from_registered_file_without_json(tmp_path):
    app = create_verification_app(tmp_path)
    with TestClient(app) as client:
        plan, context, policy = prepare_formal_plan(app, client)
        source = b"point_victor,server\n1,2\n2,1\n1,1\n"
        ingested = app.state.data_execution_workflow.ingest_file(
            context.model.project_id,
            BytesIO(source),
            original_name="points.csv",
            declared_mime_type="text/csv",
        )
        registered = ingested.processed.parsed_file.file
        assert registered.sha256 == hashlib.sha256(source).hexdigest()
        spec = CsvObservationSpec(
            source_csv_sha256=registered.sha256,
            source_column="point_victor",
            positive_value="1",
            negative_value="2",
        )
        bound = plan.model_copy(
            update={
                "observation_file_id": registered.file_id,
                "observation_sha256": registered.sha256,
                "csv_observation": spec,
            }
        )
        service = app.state.independent_verification
        service._requirements = lambda _: policy.model_copy(update={"csv_observation": spec})
        _context, _raw, observations = service._inputs(bound)
        assert observations == [1.0, 0.0, 1.0]
        with pytest.raises(ValueError, match="PLAN_DOES_NOT_COVER_EXACT_REVIEWED"):
            service._check_policy(plan, service.policy(plan.attempt_id))
        run = _running_run()
        app.state.benchmark_repository.create_run(run)
        attempt = _running_attempt(run, 1)
        app.state.benchmark_repository.create_attempt(attempt)
        app.state.benchmark_repository.bind_attempt_project(
            attempt.attempt_id, context.model.project_id
        )
        rebound = policy.model_copy(
            update={"manifest_digest": attempt.manifest_digest, "csv_observation": spec}
        )
        service._requirements = lambda _: rebound
        service.prepare_and_run(attempt.attempt_id, context.result.result_id)
        view = service.view(attempt.attempt_id)
        assert view.status == "PASS"
        assert view.plan is not None and view.plan.csv_observation == spec
        assert view.report is not None and view.report.passed_metrics == 2
