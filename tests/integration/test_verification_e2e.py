from __future__ import annotations

import subprocess
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from mathmodel_ai.agents import AgentRunResult, AgentRunStatus
from mathmodel_ai.core.config import Settings
from mathmodel_ai.db.base import Base
from mathmodel_ai.db.models import (
    ExecutionRecordModel,
    MathematicalModelRecord,
    RedTeamReportRecord,
    RepairCycleRecordModel,
    ResultRecordModel,
    RobustnessRunRecord,
    SensitivityRunRecord,
    SolverRunRecord,
    ValidationRunRecord,
    VerificationExperimentRecord,
)
from mathmodel_ai.main import create_app
from mathmodel_ai.mathematical.algorithms import AlgorithmSelector
from mathmodel_ai.providers.factory import ProviderRegistry
from mathmodel_ai.providers.mock import MockProvider
from mathmodel_ai.schemas.mathematical import MathematicalModelDraft
from mathmodel_ai.schemas.problem_state import WorkflowStage, WorkflowStatus
from mathmodel_ai.schemas.verification import (
    ModelRepairInput,
    ModelRepairOutput,
    RedTeamCategory,
    RedTeamDraft,
    RedTeamFinding,
    RedTeamInput,
    RedTeamSeverity,
    RepairAction,
    RepairTargetType,
)
from mathmodel_ai.verification.experiment_integrity import ExperimentIntegrityVerifier
from mathmodel_ai.verification.quality_gates import verified_result_quality_gate
from mathmodel_ai.verification.red_team import RedTeamAnalyzer
from mathmodel_ai.verification.robustness import RobustnessAnalyzer
from mathmodel_ai.verification.sensitivity import SensitivityAnalyzer
from mathmodel_ai.verification.validation import IndependentValidator
from mathmodel_ai.verification.workflow import VerificationWorkflow
from tests.reasoning.helpers import analysis_fixture, exploration_fixture, jury_fixture
from tests.verification.helpers import phase5_model

SOLVER_IMAGE = "mathmodel-ai-solver:phase4"


def _agent_result(name: str, output, state) -> AgentRunResult:
    now = datetime.now(UTC)
    return AgentRunResult(
        agent_name=name,
        status=AgentRunStatus.SUCCEEDED,
        output=output,
        attempts=1,
        is_mock=False,
        input_state_version=state.version,
        provider="fixture",
        model="fixture-reviewer",
        reasoning="high",
        prompt_version="test-5.0.0",
        latency_ms=1,
        started_at=now,
        ended_at=now,
    )


class StatefulRedTeamAgent:
    def __init__(self) -> None:
        self.calls = 0

    async def run(self, input_data: RedTeamInput, state, profile):
        del input_data, profile
        self.calls += 1
        findings = (
            [
                RedTeamFinding(
                    finding_id="RTF-reserve",
                    severity=RedTeamSeverity.CRITICAL,
                    category=RedTeamCategory.CONSTRAINT,
                    title="Demand reserve is omitted",
                    attack="The fixture reviewer requires a reserve adjustment.",
                    evidence_refs=["EVID-fact-1"],
                    affected_refs=["PAR-demand"],
                    recommendation="Repair the demand parameter and rerun every computation.",
                )
            ]
            if self.calls == 1
            else []
        )
        return _agent_result(
            "red_team_agent",
            RedTeamDraft(
                findings=findings,
                summary=(
                    "One critical reserve omission."
                    if findings
                    else "The repaired fixture has no remaining critical finding."
                ),
            ),
            state,
        )


class FixtureRepairAgent:
    async def run(self, input_data: ModelRepairInput, state, profile):
        del profile
        parameter = input_data.current_model.parameters[0]
        revised_parameter = parameter.model_copy(update={"value": float(parameter.value) * 1.05})
        revised = input_data.current_model.model_copy(
            update={
                "version": input_data.assigned_version,
                "parameters": [revised_parameter],
            }
        )
        output = ModelRepairOutput(
            revised_model=revised,
            actions=[
                RepairAction(
                    action_id="REPAIR-reserve",
                    finding_ids=["RTF-reserve"],
                    target_type=RepairTargetType.PARAMETER,
                    target_ref="PAR-demand",
                    description="Apply the fixture reserve adjustment.",
                    rationale="Directly addresses the critical fixture finding.",
                )
            ],
            addressed_finding_ids=["RTF-reserve"],
        )
        return _agent_result("model_repair_agent", output, state)


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


@pytest.mark.integration
@pytest.mark.solver
@pytest.mark.skipif(
    not _solver_image_available(),
    reason=f"Docker image {SOLVER_IMAGE} is unavailable",
)
def test_real_result_through_phase5_reports_and_mock_review_boundary(tmp_path: Path) -> None:
    template = phase5_model()
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
            RedTeamDraft(
                summary="Mock review adds no finding; it cannot establish Red Team acceptance."
            ).model_dump_json(),
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
                "name": "Phase 5 real solver fixture",
                "title": "Demand-sensitive allocation",
                "raw_problem": (
                    "Minimize allocation cost while satisfying the sourced demand parameter."
                ),
            },
        )
        project_id = created.json()["project_id"]
        project_uuid = UUID(project_id)
        reasoning = client.post(f"/api/v1/projects/{project_id}/reasoning/run", json={})
        mathematical = client.post(f"/api/v1/projects/{project_id}/mathematical/run", json={})
        verification = client.post(
            f"/api/v1/projects/{project_id}/verification/run",
            json={
                "sensitivity": {"perturbation_fractions": [0.05]},
                "robustness": {
                    "method": "SCENARIO_ANALYSIS",
                    "scenario_fractions": [-0.05, 0.05],
                },
            },
        )

        assert created.status_code == 201, created.text
        assert reasoning.status_code == 200, reasoning.text
        assert mathematical.status_code == 200, mathematical.text
        assert verification.status_code == 200, verification.text
        payload = verification.json()
        assert payload["validation"]["gate"]["status"] == "PASS"
        assert payload["validation"]["state_version"] == 6
        assert payload["sensitivity"]["gate"]["status"] == "PASS"
        assert payload["sensitivity"]["state_version"] == 7
        assert payload["sensitivity"]["report"]["successful_runs"] == 2
        assert payload["robustness"]["gate"]["status"] == "PASS"
        assert payload["robustness"]["state_version"] == 8
        assert payload["robustness"]["report"]["summary"]["feasibility_rate"] == 1
        assert payload["red_team"]["state_version"] == 9
        assert payload["red_team"]["report"]["review_is_mock"] is True
        assert payload["red_team"]["report"]["status"] == "INCONCLUSIVE"
        assert payload["red_team"]["gate"]["status"] == "HUMAN_REVIEW"
        assert payload["red_team"]["verified_gate"]["status"] == "HUMAN_REVIEW"
        assert payload["red_team"]["verified_result_id"] is None

        assert (
            client.get(f"/api/v1/projects/{project_id}/validation-runs/latest").status_code == 200
        )
        assert (
            client.get(f"/api/v1/projects/{project_id}/sensitivity-runs/latest").status_code == 200
        )
        assert (
            client.get(f"/api/v1/projects/{project_id}/robustness-runs/latest").status_code == 200
        )
        assert (
            client.get(f"/api/v1/projects/{project_id}/red-team-reports/latest").status_code == 200
        )

        restored = app.state.reasoning_repository.load_current(project_uuid)
        assert restored.current_stage is WorkflowStage.RED_TEAM
        assert restored.status is WorkflowStatus.HUMAN_REVIEW
        assert restored.schema_version == 5
        assert len(restored.validation_results) == 1
        assert len(restored.sensitivity_results) == 1
        assert len(restored.robustness_results) == 1
        assert len(restored.red_team_reports) == 1
        assert restored.verified_result_id is None
        assert restored.results[-1].verification_status.value == "UNVERIFIED"

        with Session(app.state.engine) as session:
            assert session.scalar(select(func.count()).select_from(ValidationRunRecord)) == 1
            assert session.scalar(select(func.count()).select_from(SensitivityRunRecord)) == 1
            assert session.scalar(select(func.count()).select_from(RobustnessRunRecord)) == 1
            assert session.scalar(select(func.count()).select_from(RedTeamReportRecord)) == 1
            assert (
                session.scalar(select(func.count()).select_from(VerificationExperimentRecord)) == 4
            )
            assert session.scalar(select(func.count()).select_from(ExecutionRecordModel)) == 5

        sensitivity_report = app.state.verification_repository.get_sensitivity(project_uuid)
        robustness_report = app.state.verification_repository.get_robustness(project_uuid)
        validation_report = app.state.verification_repository.get_validation(project_uuid)
        context = app.state.mathematical_repository.get_result_context(
            project_uuid,
            validation_report.result_id,
        )
        tampered_experiment = sensitivity_report.experiments[0]
        assert tampered_experiment.execution_record_id is not None
        with Session(app.state.engine) as session:
            execution_row = session.get(
                ExecutionRecordModel,
                tampered_experiment.execution_record_id,
            )
            assert execution_row is not None
            tampered_execution = deepcopy(execution_row.record_json)
            tampered_execution["model_digest"] = validation_report.model_digest
            execution_row.record_json = tampered_execution
            execution_row.model_digest = validation_report.model_digest
            session.commit()
        integrity_errors = app.state.verification_repository.audit_experiment_report(
            project_id=project_uuid,
            model=context.model,
            experiments=sensitivity_report.experiments,
            verifier=ExperimentIntegrityVerifier(IndependentValidator()),
            sensitivity_run_id=sensitivity_report.sensitivity_id,
        )
        non_mock_review = RedTeamAnalyzer().compile(
            model=context.model,
            validation=validation_report,
            sensitivity=sensitivity_report,
            robustness=robustness_report,
            draft=RedTeamDraft(summary="Fixture review has no additional finding."),
            reviewer_agent_run_id=UUID(int=1),
            review_is_mock=False,
        )
        tampered_gate = verified_result_quality_gate(
            validation_report,
            sensitivity_report,
            robustness_report,
            non_mock_review,
            validation_integrity_errors=[],
            sensitivity_integrity_errors=integrity_errors,
            robustness_integrity_errors=[],
        )
        assert any("EXECUTION_MODEL_DIGEST_MISMATCH" in item for item in integrity_errors)
        assert tampered_gate.status.value == "RETRY"
        assert tampered_gate.checks["sensitivity_execution_integrity"] is False


@pytest.mark.integration
@pytest.mark.solver
@pytest.mark.skipif(
    not _solver_image_available(),
    reason=f"Docker image {SOLVER_IMAGE} is unavailable",
)
def test_non_mock_critical_finding_creates_new_model_version_and_revalidates(
    tmp_path: Path,
) -> None:
    template = phase5_model()
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
    validator = app.state.independent_validator
    integrity_verifier = app.state.experiment_integrity_verifier
    app.state.verification_workflow = VerificationWorkflow(
        reasoning_repository=app.state.reasoning_repository,
        mathematical_repository=app.state.mathematical_repository,
        mathematical_workflow=app.state.mathematical_workflow,
        repository=app.state.verification_repository,
        validator=validator,
        sensitivity_analyzer=SensitivityAnalyzer(app.state.experiment_engine),
        robustness_analyzer=RobustnessAnalyzer(app.state.experiment_engine),
        red_team_agent=StatefulRedTeamAgent(),  # type: ignore[arg-type]
        red_team_analyzer=RedTeamAnalyzer(),
        model_repair_agent=FixtureRepairAgent(),  # type: ignore[arg-type]
        algorithm_selector=AlgorithmSelector(),
        experiment_integrity_verifier=integrity_verifier,
    )

    with TestClient(app) as client:
        project_id = client.post(
            "/api/v1/projects",
            json={
                "name": "Phase 5 repair fixture",
                "title": "Repair demand-sensitive allocation",
                "raw_problem": "Minimize cost while satisfying demand and reserve evidence.",
            },
        ).json()["project_id"]
        assert (
            client.post(f"/api/v1/projects/{project_id}/reasoning/run", json={}).status_code == 200
        )
        mathematical = client.post(f"/api/v1/projects/{project_id}/mathematical/run", json={})
        assert mathematical.status_code == 200, mathematical.text
        stable_model_id = mathematical.json()["model_stage"]["mathematical_model"]["model_id"]
        with Session(app.state.engine) as session:
            original_model = session.scalar(select(MathematicalModelRecord))
            original_result = session.scalar(select(ResultRecordModel))
            original_solver = session.scalar(select(SolverRunRecord))
            assert original_model is not None
            assert original_result is not None
            assert original_solver is not None
            original_execution = session.get(
                ExecutionRecordModel,
                original_result.execution_record_id,
            )
            assert original_execution is not None
            immutable_snapshot = {
                "model_id": original_model.id,
                "model_json": deepcopy(original_model.model_json),
                "result_id": original_result.id,
                "result_json": deepcopy(original_result.record_json),
                "solver_id": original_solver.id,
                "solver_json": deepcopy(original_solver.result_json),
                "execution_id": original_execution.id,
                "execution_json": deepcopy(original_execution.record_json),
            }

        attacked = client.post(
            f"/api/v1/projects/{project_id}/verification/run",
            json={
                "sensitivity": {"perturbation_fractions": [0.05]},
                "robustness": {"scenario_fractions": [-0.05, 0.05]},
            },
        )
        assert attacked.status_code == 200, attacked.text
        assert attacked.json()["red_team"]["report"]["critical_count"] == 1
        assert attacked.json()["red_team"]["gate"]["status"] == "RETRY"
        assert attacked.json()["red_team"]["verified_gate"]["status"] == "RETRY"
        assert attacked.json()["red_team"]["verified_result_id"] is None

        repaired = client.post(
            f"/api/v1/projects/{project_id}/model/repair-loop",
            json={
                "sensitivity": {"perturbation_fractions": [0.05]},
                "robustness": {"scenario_fractions": [-0.05, 0.05]},
            },
        )
        assert repaired.status_code == 200, repaired.text
        payload = repaired.json()
        assert payload["resolved"] is True
        assert payload["exhausted"] is False
        assert len(payload["cycles"]) == 1
        assert payload["cycles"][0]["source_model_version"] == 1
        assert payload["cycles"][0]["target_model_version"] == 2
        assert payload["cycles"][0]["status"] == "ACCEPTED"
        assert payload["final_red_team"]["critical_count"] == 0
        assert payload["final_red_team"]["status"] == "PASS"
        assert payload["state"]["current_stage"] == "RED_TEAM"
        assert payload["state"]["status"] == "SUCCEEDED"
        assert payload["state"]["mathematical_model"]["version"] == 2
        assert payload["state"]["mathematical_model"]["model_id"] == stable_model_id
        assert payload["state"]["verified_result_id"] == payload["final_red_team"]["result_id"]
        assert [item["verification_status"] for item in payload["state"]["results"]] == [
            "UNVERIFIED",
            "VERIFIED",
        ]

        with Session(app.state.engine) as session:
            models = list(
                session.scalars(
                    select(MathematicalModelRecord).order_by(MathematicalModelRecord.version)
                )
            )
            assert len(models) == 2
            assert models[0].model_id == models[1].model_id
            assert models[0].model_digest != models[1].model_digest
            assert models[0].id == immutable_snapshot["model_id"]
            assert models[0].model_json == immutable_snapshot["model_json"]
            formal_results = list(
                session.scalars(select(ResultRecordModel).order_by(ResultRecordModel.created_at))
            )
            solver_runs = list(
                session.scalars(select(SolverRunRecord).order_by(SolverRunRecord.start_time))
            )
            assert len(formal_results) == 2
            assert len(solver_runs) == 2
            assert formal_results[0].id == immutable_snapshot["result_id"]
            assert formal_results[0].record_json == immutable_snapshot["result_json"]
            assert solver_runs[0].id == immutable_snapshot["solver_id"]
            assert solver_runs[0].result_json == immutable_snapshot["solver_json"]
            old_execution = session.get(
                ExecutionRecordModel,
                immutable_snapshot["execution_id"],
            )
            assert old_execution is not None
            assert old_execution.record_json == immutable_snapshot["execution_json"]
            assert formal_results[1].id != formal_results[0].id
            assert formal_results[1].execution_record_id != formal_results[0].execution_record_id
            assert formal_results[1].solver_run_id != formal_results[0].solver_run_id
            assert formal_results[1].id == UUID(payload["state"]["verified_result_id"])
            assert session.scalar(select(func.count()).select_from(RepairCycleRecordModel)) == 1
            assert session.scalar(select(func.count()).select_from(ValidationRunRecord)) == 2
            assert session.scalar(select(func.count()).select_from(SensitivityRunRecord)) == 2
            assert session.scalar(select(func.count()).select_from(RobustnessRunRecord)) == 2
            assert session.scalar(select(func.count()).select_from(RedTeamReportRecord)) == 2
