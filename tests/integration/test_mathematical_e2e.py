import subprocess
from copy import deepcopy
from pathlib import Path
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from mathmodel_ai.core.config import Settings
from mathmodel_ai.db.base import Base
from mathmodel_ai.db.models import (
    AgentRunRecord,
    ExecutionRecordModel,
    GeneratedProgramRecord,
    MathematicalModelRecord,
    ResultRecordModel,
    SolverRunRecord,
)
from mathmodel_ai.main import create_app
from mathmodel_ai.mathematical.digests import mathematical_model_digest
from mathmodel_ai.providers.factory import ProviderRegistry
from mathmodel_ai.providers.mock import MockProvider
from mathmodel_ai.sandbox.executor import SandboxExecutor
from mathmodel_ai.schemas.execution import SandboxLimits
from mathmodel_ai.schemas.mathematical import MathematicalModelDraft
from mathmodel_ai.schemas.problem_state import WorkflowStage
from mathmodel_ai.schemas.program import GeneratedProgramDraft, GeneratedSourceFile
from mathmodel_ai.solvers.router import SolverRouter
from mathmodel_ai.solvers.scipy import SciPySolver
from tests.mathematical.helpers import lp_model
from tests.reasoning.helpers import analysis_fixture, exploration_fixture, jury_fixture

SOLVER_IMAGE = "mathmodel-ai-solver:phase4"


class RoutedTimeoutRunner:
    def __call__(self, command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        if command[1:3] == ["image", "inspect"]:
            return subprocess.CompletedProcess(command, 0, stdout=b"sha256:solver", stderr=b"")
        if command[1] == "run" and "--rm" in command:
            return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")
        if command[1:3] == ["rm", "--force"]:
            return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")
        raise subprocess.TimeoutExpired(command, timeout=float(kwargs["timeout"]))


def _solver_image_available() -> bool:
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", SOLVER_IMAGE],
            capture_output=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def _generated_lp_program(model_digest: str) -> GeneratedProgramDraft:
    source = f'''import json
from pathlib import Path

from scipy.optimize import linprog

COST = [3.0, 4.0]
solution = linprog(
    COST,
    A_ub=[[-1.0, -1.0]],
    b_ub=[-10.0],
    bounds=[(0.0, None), (0.0, None)],
    method="highs",
)
status_map = {{
    0: "OPTIMAL",
    1: "ITERATION_LIMIT",
    2: "INFEASIBLE",
    3: "UNBOUNDED",
    4: "NUMERICAL_ERROR",
}}
variables = {{}}
if solution.x is not None:
    variables = {{"x": float(solution.x[0]), "y": float(solution.x[1])}}
objective = float(solution.fun) if solution.fun is not None else None
is_optimal = solution.status == 0
is_feasible = solution.x is not None and solution.status in {{0, 1}}
payload = {{
    "solver_name": "SCIPY",
    "solver_version": None,
    "model_digest": "{model_digest}",
    "status": status_map.get(solution.status, "UNKNOWN"),
    "objective": objective,
    "variable_values": variables,
    "metrics": {{"iterations": int(solution.nit)}},
    "warnings": [],
    "message": str(solution.message),
    "raw_status": str(solution.status),
    "is_optimal": is_optimal,
    "is_feasible": is_feasible,
}}
Path("/output/result.json").write_text(
    json.dumps(payload, allow_nan=False), encoding="utf-8"
)
'''
    return GeneratedProgramDraft(
        entrypoint="generated_solver.py",
        files=[GeneratedSourceFile(path="generated_solver.py", content=source)],
        dependencies=["scipy"],
        solver_target="SCIPY_HIGHS",
        explanation="Translate the immutable LP to SciPy and emit the required JSON contract.",
    )


@pytest.mark.integration
def test_model_endpoint_rejects_skipping_reasoning_selection(tmp_path: Path) -> None:
    app = create_app(
        Settings(
            environment="test",
            database_url="sqlite+pysqlite:///:memory:",
            storage_root=tmp_path / "storage",
            sandbox_root=tmp_path / "sandbox",
            solver_sandbox_root=tmp_path / "solver-sandbox",
        ),
        providers=ProviderRegistry([MockProvider()]),
    )
    Base.metadata.create_all(app.state.engine)
    with TestClient(app) as client:
        created = client.post(
            "/api/v1/projects",
            json={
                "name": "No stage skipping",
                "title": "Reject premature model construction",
                "raw_problem": "A sufficiently detailed fixture must complete reasoning first.",
            },
        )
        project_id = created.json()["project_id"]
        response = client.post(
            f"/api/v1/projects/{project_id}/model/build",
            json={"user_guidance": []},
        )

    assert response.status_code == 400
    assert "INGEST -> MODEL" in response.json()["detail"]


@pytest.mark.integration
def test_model_gate_failure_is_audited_without_persisting_model_revision(tmp_path: Path) -> None:
    template = lp_model().model_copy(update={"objective": None})
    invalid_draft = MathematicalModelDraft.model_validate(
        template.model_dump(
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
            invalid_draft.model_dump_json(),
        ]
    )
    app = create_app(
        Settings(
            environment="test",
            database_url="sqlite+pysqlite:///:memory:",
            reasoning_max_retries=0,
            storage_root=tmp_path / "storage",
            sandbox_root=tmp_path / "sandbox",
            solver_sandbox_root=tmp_path / "solver-sandbox",
        ),
        providers=ProviderRegistry([mock]),
    )
    Base.metadata.create_all(app.state.engine)
    with TestClient(app) as client:
        created = client.post(
            "/api/v1/projects",
            json={
                "name": "MODEL gate rejection",
                "title": "Optimization without an objective",
                "raw_problem": "Reject an incomplete optimization model before any solver runs.",
            },
        )
        project_id = created.json()["project_id"]
        project_uuid = UUID(project_id)
        reasoning = client.post(f"/api/v1/projects/{project_id}/reasoning/run", json={})
        assert reasoning.status_code == 200, reasoning.text

        rejected = client.post(
            f"/api/v1/projects/{project_id}/model/build",
            json={"user_guidance": []},
        )

        assert rejected.status_code == 400
        assert "MODEL quality gate rejected output" in rejected.json()["detail"]
        restored = app.state.reasoning_repository.load_current(project_uuid)
        assert restored.current_stage is WorkflowStage.SELECT
        assert restored.version == 3
        with Session(app.state.engine) as session:
            assert session.scalar(select(func.count()).select_from(MathematicalModelRecord)) == 0
            runs = list(session.scalars(select(AgentRunRecord).order_by(AgentRunRecord.started_at)))
            assert len(runs) == 4
            assert runs[-1].agent_name == "math_modeler"
            assert runs[-1].status == "RETRY"


@pytest.mark.integration
def test_solve_timeout_is_persisted_as_unverified_retry_at_model_stage(tmp_path: Path) -> None:
    template = lp_model()
    draft = MathematicalModelDraft.model_validate(
        template.model_dump(
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
            database_url="sqlite+pysqlite:///:memory:",
            reasoning_max_retries=0,
            storage_root=tmp_path / "storage",
            sandbox_root=tmp_path / "sandbox",
            solver_sandbox_root=tmp_path / "solver-sandbox",
        ),
        providers=ProviderRegistry([mock]),
    )
    Base.metadata.create_all(app.state.engine)
    timeout_sandbox = SandboxExecutor(
        store=app.state.file_store,
        root=tmp_path / "timeout-solver",
        image="solver:test",
        limits=SandboxLimits(
            cpu_cores=1,
            memory_mb=128,
            timeout_seconds=0.01,
            pids_limit=16,
            max_output_bytes=1024,
            max_artifacts=2,
            max_artifact_bytes=4096,
        ),
        runner=RoutedTimeoutRunner(),
    )
    app.state.mathematical_workflow._solver_router = SolverRouter(
        [SciPySolver(sandbox=timeout_sandbox, store=app.state.file_store)]
    )

    with TestClient(app) as client:
        created = client.post(
            "/api/v1/projects",
            json={
                "name": "SOLVE timeout rollback",
                "title": "Trace a timed-out LP attempt",
                "raw_problem": "Persist a timed-out solver attempt without claiming success.",
            },
        )
        project_id = created.json()["project_id"]
        project_uuid = UUID(project_id)
        assert (
            client.post(f"/api/v1/projects/{project_id}/reasoning/run", json={}).status_code == 200
        )
        assert (
            client.post(
                f"/api/v1/projects/{project_id}/model/build",
                json={"user_guidance": []},
            ).status_code
            == 200
        )

        timed_out = client.post(
            f"/api/v1/projects/{project_id}/solve",
            json={"options": {}},
        )

        assert timed_out.status_code == 200, timed_out.text
        assert timed_out.json()["execution"]["status"] == "TIMEOUT"
        assert timed_out.json()["result"]["status"] == "TIME_LIMIT"
        assert timed_out.json()["gate"]["status"] == "RETRY"
        assert "SOLVE_GATE_FAIL:TIME_LIMIT" in timed_out.json()["gate"]["errors"]
        restored = app.state.reasoning_repository.load_current(project_uuid)
        assert restored.current_stage is WorkflowStage.MODEL
        assert restored.status.value == "RETRY"
        assert restored.results[-1].verification_status.value == "UNVERIFIED"
        assert restored.stage_history[-1].from_stage == "MODEL"
        assert restored.stage_history[-1].to_stage == "MODEL"


@pytest.mark.integration
@pytest.mark.solver
@pytest.mark.skipif(
    not _solver_image_available(),
    reason=f"Docker image {SOLVER_IMAGE} is unavailable",
)
def test_individual_model_and_solve_endpoints_advance_exact_stages(tmp_path: Path) -> None:
    template = lp_model()
    draft = MathematicalModelDraft.model_validate(
        template.model_dump(
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
            database_url="sqlite+pysqlite:///:memory:",
            default_provider="mock",
            reasoning_max_retries=0,
            storage_root=tmp_path / "storage",
            sandbox_root=tmp_path / "sandbox",
            solver_sandbox_root=tmp_path / "solver-sandbox",
            solver_sandbox_image=SOLVER_IMAGE,
            sandbox_memory_mb=768,
            sandbox_timeout_seconds=20,
            sandbox_max_artifact_bytes=1_048_576,
        ),
        providers=ProviderRegistry([mock]),
    )
    Base.metadata.create_all(app.state.engine)

    with TestClient(app) as client:
        created = client.post(
            "/api/v1/projects",
            json={
                "name": "Individual Phase 4 endpoints",
                "title": "Minimum-cost allocation",
                "raw_problem": (
                    "Minimize 3x + 4y subject to x + y being at least 10, with x and y nonnegative."
                ),
            },
        )
        project_id = created.json()["project_id"]
        reasoning = client.post(
            f"/api/v1/projects/{project_id}/reasoning/run",
            json={},
        )
        assert reasoning.status_code == 200, reasoning.text

        built = client.post(
            f"/api/v1/projects/{project_id}/model/build",
            json={"user_guidance": []},
        )
        fetched = client.get(f"/api/v1/projects/{project_id}/mathematical-model")
        solved = client.post(
            f"/api/v1/projects/{project_id}/solve",
            json={"options": {"feasibility_tolerance": 1e-7}},
        )

    assert built.status_code == 200, built.text
    assert built.json()["state_version"] == 4
    assert fetched.status_code == 200, fetched.text
    assert fetched.json()["model_id"] == built.json()["mathematical_model"]["model_id"]
    assert solved.status_code == 200, solved.text
    assert solved.json()["state_version"] == 5
    assert solved.json()["result"]["status"] == "OPTIMAL"
    assert solved.json()["execution"]["is_mock"] is False


@pytest.mark.integration
@pytest.mark.solver
@pytest.mark.skipif(
    not _solver_image_available(),
    reason=f"Docker image {SOLVER_IMAGE} is unavailable",
)
def test_code_agent_generated_program_runs_in_real_sandbox_and_passes_evidence(
    tmp_path: Path,
) -> None:
    template = lp_model()
    draft = MathematicalModelDraft.model_validate(
        template.model_dump(
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
            _generated_lp_program(mathematical_model_digest(template)).model_dump_json(),
        ]
    )
    app = create_app(
        Settings(
            environment="test",
            database_url="sqlite+pysqlite:///:memory:",
            default_provider="mock",
            reasoning_max_retries=0,
            storage_root=tmp_path / "storage",
            sandbox_root=tmp_path / "sandbox",
            solver_sandbox_root=tmp_path / "solver-sandbox",
            solver_sandbox_image=SOLVER_IMAGE,
            sandbox_memory_mb=768,
            sandbox_timeout_seconds=20,
            sandbox_max_artifact_bytes=1_048_576,
        ),
        providers=ProviderRegistry([mock]),
    )
    Base.metadata.create_all(app.state.engine)

    with TestClient(app) as client:
        created = client.post(
            "/api/v1/projects",
            json={
                "name": "Generated-program LP",
                "title": "Minimum-cost allocation",
                "raw_problem": (
                    "Minimize 3x + 4y subject to x + y being at least 10, with x and y nonnegative."
                ),
            },
        )
        project_id = created.json()["project_id"]
        reasoning = client.post(f"/api/v1/projects/{project_id}/reasoning/run", json={})
        assert reasoning.status_code == 200, reasoning.text

        response = client.post(
            f"/api/v1/projects/{project_id}/mathematical/run",
            json={"execution_strategy": "GENERATED"},
        )

        assert response.status_code == 200, response.text
        solve = response.json()["solve_stage"]
        assert solve["execution_strategy"] == {
            "requested": "GENERATED",
            "selected": "GENERATED",
            "reason": "caller explicitly requested the generated-program strategy",
        }
        assert solve["code_agent_run"] is not None
        assert solve["code_agent_run"]["is_mock"] is True
        assert solve["generated_program"]["execution_origin"] == "GENERATED_PROGRAM"
        assert (
            solve["generated_program"]["generator_agent_run_id"]
            == solve["code_agent_run"]["run_id"]
        )
        assert solve["execution"]["execution_origin"] == "GENERATED_PROGRAM"
        assert solve["execution"]["is_mock"] is False
        assert solve["execution"]["status"] == "SUCCEEDED"
        assert solve["execution"]["executed_bundle_hash"] == solve["generated_program"]["code_hash"]
        assert solve["result"]["status"] == "OPTIMAL"
        assert solve["result"]["objective"] == pytest.approx(30)
        assert solve["result"]["key_outputs"] == pytest.approx({"x": 10, "y": 0})
        assert solve["gate"]["status"] == "PASS"

        evidence = client.get(
            f"/api/v1/projects/{project_id}/results/{solve['result']['result_id']}/evidence"
        )
        assert evidence.status_code == 200, evidence.text
        assert evidence.json()["valid"] is True
        assert evidence.json()["code_hash_match"] is True

        with Session(app.state.engine) as session:
            runs = list(session.scalars(select(AgentRunRecord).order_by(AgentRunRecord.started_at)))
            assert len(runs) == 5
            assert runs[-1].agent_name == "code_agent"
            assert runs[-1].is_mock is True


@pytest.mark.integration
@pytest.mark.solver
@pytest.mark.skipif(
    not _solver_image_available(),
    reason=f"Docker image {SOLVER_IMAGE} is unavailable",
)
def test_problem_to_real_solver_result_has_complete_persisted_evidence_chain(
    tmp_path: Path,
) -> None:
    template = lp_model()
    draft = MathematicalModelDraft.model_validate(
        template.model_dump(
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
            database_url="sqlite+pysqlite:///:memory:",
            default_provider="mock",
            reasoning_max_retries=0,
            storage_root=tmp_path / "storage",
            sandbox_root=tmp_path / "sandbox",
            solver_sandbox_root=tmp_path / "solver-sandbox",
            solver_sandbox_image=SOLVER_IMAGE,
            sandbox_memory_mb=768,
            sandbox_timeout_seconds=20,
            sandbox_max_artifact_bytes=1_048_576,
        ),
        providers=ProviderRegistry([mock]),
    )
    Base.metadata.create_all(app.state.engine)

    with TestClient(app) as client:
        created = client.post(
            "/api/v1/projects",
            json={
                "name": "Phase 4 real LP benchmark",
                "title": "Minimum-cost allocation",
                "raw_problem": (
                    "Minimize 3x + 4y subject to x + y being at least 10, with x and y nonnegative."
                ),
            },
        )
        assert created.status_code == 201, created.text
        project_id = created.json()["project_id"]
        project_uuid = UUID(project_id)

        reasoning = client.post(
            f"/api/v1/projects/{project_id}/reasoning/run",
            json={"user_guidance": ["Use the deterministic LP fixture."]},
        )
        assert reasoning.status_code == 200, reasoning.text
        assert reasoning.json()["state"]["current_stage"] == "SELECT"

        mathematical = client.post(
            f"/api/v1/projects/{project_id}/mathematical/run",
            json={"user_guidance": ["Keep all coefficients traceable."]},
        )
        assert mathematical.status_code == 200, mathematical.text
        payload = mathematical.json()
        model_stage = payload["model_stage"]
        solve_stage = payload["solve_stage"]
        result = solve_stage["result"]
        execution = solve_stage["execution"]

        assert model_stage["is_mock"] is True
        assert model_stage["gate"]["status"] == "PASS"
        assert solve_stage["gate"]["status"] == "PASS"
        assert solve_stage["route"]["selected_solver"] == "SCIPY"
        assert solve_stage["route"]["selected_family"] == "SCIPY_HIGHS"
        assert result["status"] == "OPTIMAL"
        assert result["objective"] == pytest.approx(30)
        assert result["key_outputs"] == pytest.approx({"x": 10, "y": 0})
        assert solve_stage["solver_run"]["result"]["is_optimal"] is True
        assert solve_stage["solver_run"]["result"]["is_feasible"] is True
        assert solve_stage["solver_run"]["result"]["feasibility"][
            "max_constraint_violation"
        ] == pytest.approx(0)
        assert execution["status"] == "SUCCEEDED"
        assert execution["is_mock"] is False
        assert execution["image_id"].startswith("sha256:")
        assert solve_stage["generated_program"]["is_mock"] is False
        assert solve_stage["generated_program"]["status"] == "EXECUTED"

        result_id = result["result_id"]
        evidence = client.get(f"/api/v1/projects/{project_id}/results/{result_id}/evidence")
        assert evidence.status_code == 200, evidence.text
        assert evidence.json() == {
            "result_id": result_id,
            "valid": True,
            "model_found": True,
            "solver_run_found": True,
            "execution_found": True,
            "generated_program_found": True,
            "exact_model_version": True,
            "model_digest_match": True,
            "code_hash_match": True,
            "errors": [],
        }

        assert len(client.get(f"/api/v1/projects/{project_id}/results").json()) == 1
        assert len(client.get(f"/api/v1/projects/{project_id}/solver-runs").json()) == 1
        assert len(client.get(f"/api/v1/projects/{project_id}/generated-programs").json()) == 1

        restored = app.state.reasoning_repository.load_current(project_uuid)
        assert restored.current_stage is WorkflowStage.SOLVE
        assert restored.version == 5
        assert restored.mathematical_model is not None
        assert restored.solver_runs[0].execution_ref == UUID(execution["run_id"])
        assert restored.result_records[0].result_id == UUID(result_id)
        assert restored.execution_records[0].is_mock is False
        assert restored.results[0].verification_status.value == "UNVERIFIED"
        assert restored.verified_result_id is None

        with Session(app.state.engine) as session:
            assert session.scalar(select(func.count()).select_from(AgentRunRecord)) == 4
            assert session.scalar(select(func.count()).select_from(MathematicalModelRecord)) == 1
            assert session.scalar(select(func.count()).select_from(GeneratedProgramRecord)) == 1
            assert session.scalar(select(func.count()).select_from(SolverRunRecord)) == 1
            assert session.scalar(select(func.count()).select_from(ResultRecordModel)) == 1
            assert session.scalar(select(func.count()).select_from(ExecutionRecordModel)) == 1

            model_row = session.scalar(select(MathematicalModelRecord))
            program_row = session.scalar(select(GeneratedProgramRecord))
            solver_row = session.scalar(select(SolverRunRecord))
            result_row = session.scalar(select(ResultRecordModel))
            assert model_row is not None
            assert program_row is not None
            assert solver_row is not None
            assert result_row is not None
            assert {
                (program_row.model_id, program_row.model_version),
                (solver_row.model_id, solver_row.model_version),
                (result_row.model_id, result_row.model_version),
            } == {(model_row.model_id, model_row.version)}
            assert solver_row.execution_record_id == result_row.execution_record_id
            assert result_row.mathematical_model_record_id == model_row.id

        tamper_cases = [
            ("result_objective", "OBJECTIVE_MISMATCH"),
            ("result_status", "STATUS_MISMATCH"),
            ("result_key_outputs", "KEY_OUTPUT_MISMATCH"),
            ("solver_result_ref", "RESULT_REF_MISMATCH"),
            ("model_version", "MODEL_VERSION_MISMATCH"),
            ("model_digest", "MODEL_DIGEST_MISMATCH"),
            ("program_code_hash", "CODE_HASH_MISMATCH"),
            ("execution_code_hash", "CODE_HASH_MISMATCH"),
            ("execution_exit_code", "EXECUTION_FAILED"),
            ("execution_is_mock", "MOCK_EXECUTION"),
        ]
        for tamper, expected_code in tamper_cases:
            with Session(app.state.engine) as session:
                program_row = session.scalar(select(GeneratedProgramRecord))
                solver_row = session.scalar(select(SolverRunRecord))
                result_row = session.scalar(select(ResultRecordModel))
                execution_row = session.scalar(select(ExecutionRecordModel))
                assert program_row is not None
                assert solver_row is not None
                assert result_row is not None
                assert execution_row is not None
                originals = {
                    "program_code_hash": program_row.code_hash,
                    "program_json": deepcopy(program_row.program_json),
                    "solver_model_version": solver_row.model_version,
                    "solver_json": deepcopy(solver_row.result_json),
                    "result_objective": result_row.objective,
                    "result_status": result_row.status,
                    "result_digest": result_row.model_digest,
                    "result_json": deepcopy(result_row.record_json),
                    "execution_code_hash": execution_row.code_hash,
                    "execution_exit_code": execution_row.exit_code,
                    "execution_is_mock": execution_row.is_mock,
                    "execution_json": deepcopy(execution_row.record_json),
                }
                if tamper == "result_objective":
                    result_row.objective = 999
                    payload = deepcopy(result_row.record_json)
                    payload["objective"] = 999
                    result_row.record_json = payload
                elif tamper == "result_status":
                    result_row.status = "TIME_LIMIT"
                    payload = deepcopy(result_row.record_json)
                    payload["status"] = "TIME_LIMIT"
                    result_row.record_json = payload
                elif tamper == "result_key_outputs":
                    payload = deepcopy(result_row.record_json)
                    payload["key_outputs"]["x"] = 999
                    result_row.record_json = payload
                elif tamper == "solver_result_ref":
                    payload = deepcopy(solver_row.result_json)
                    payload["result_ref"] = str(UUID(int=0))
                    solver_row.result_json = payload
                elif tamper == "model_version":
                    solver_row.model_version += 1
                    payload = deepcopy(solver_row.result_json)
                    payload["model_version"] += 1
                    solver_row.result_json = payload
                elif tamper == "model_digest":
                    result_row.model_digest = "f" * 64
                    payload = deepcopy(result_row.record_json)
                    payload["model_digest"] = "f" * 64
                    result_row.record_json = payload
                elif tamper == "program_code_hash":
                    program_row.code_hash = "f" * 64
                    payload = deepcopy(program_row.program_json)
                    payload["code_hash"] = "f" * 64
                    program_row.program_json = payload
                elif tamper == "execution_code_hash":
                    execution_row.code_hash = "f" * 64
                    payload = deepcopy(execution_row.record_json)
                    payload["code_hash"] = "f" * 64
                    execution_row.record_json = payload
                elif tamper == "execution_exit_code":
                    execution_row.exit_code = 1
                    payload = deepcopy(execution_row.record_json)
                    payload["exit_code"] = 1
                    execution_row.record_json = payload
                elif tamper == "execution_is_mock":
                    execution_row.is_mock = True
                    payload = deepcopy(execution_row.record_json)
                    payload["is_mock"] = True
                    execution_row.record_json = payload
                session.commit()

            broken = client.get(f"/api/v1/projects/{project_id}/results/{result_id}/evidence")
            assert broken.status_code == 200, broken.text
            assert broken.json()["valid"] is False
            assert expected_code in {issue["code"] for issue in broken.json()["errors"]}

            with Session(app.state.engine) as session:
                program_row = session.scalar(select(GeneratedProgramRecord))
                solver_row = session.scalar(select(SolverRunRecord))
                result_row = session.scalar(select(ResultRecordModel))
                execution_row = session.scalar(select(ExecutionRecordModel))
                assert program_row is not None
                assert solver_row is not None
                assert result_row is not None
                assert execution_row is not None
                program_row.code_hash = originals["program_code_hash"]
                program_row.program_json = originals["program_json"]
                solver_row.model_version = originals["solver_model_version"]
                solver_row.result_json = originals["solver_json"]
                result_row.objective = originals["result_objective"]
                result_row.status = originals["result_status"]
                result_row.model_digest = originals["result_digest"]
                result_row.record_json = originals["result_json"]
                execution_row.code_hash = originals["execution_code_hash"]
                execution_row.exit_code = originals["execution_exit_code"]
                execution_row.is_mock = originals["execution_is_mock"]
                execution_row.record_json = originals["execution_json"]
                session.commit()

            restored_evidence = client.get(
                f"/api/v1/projects/{project_id}/results/{result_id}/evidence"
            )
            assert restored_evidence.status_code == 200, restored_evidence.text
            assert restored_evidence.json()["valid"] is True
