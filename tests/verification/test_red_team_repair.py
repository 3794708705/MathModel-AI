from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from mathmodel_ai.agents import AgentRunResult, AgentRunStatus
from mathmodel_ai.core.errors import AgentRunError
from mathmodel_ai.schemas.problem_state import ProblemState, WorkflowStage, WorkflowStatus
from mathmodel_ai.schemas.verification import (
    ModelRepairOutput,
    RedTeamCategory,
    RedTeamDraft,
    RedTeamFinding,
    RedTeamSeverity,
    RepairAction,
    RepairCycleRecord,
    RepairCycleStatus,
    RepairTargetType,
    RobustnessConfig,
    SensitivityConfig,
    ValidationStatus,
)
from mathmodel_ai.verification.quality_gates import (
    red_team_quality_gate,
    repair_quality_gate,
)
from mathmodel_ai.verification.red_team import RedTeamAnalyzer
from mathmodel_ai.verification.robustness import RobustnessAnalyzer
from mathmodel_ai.verification.sensitivity import SensitivityAnalyzer
from mathmodel_ai.verification.workflow import VerificationWorkflow
from tests.verification.helpers import experiment_engine, valid_report


def _phase5_reports():
    model, result, validation, _ = valid_report()
    engine = experiment_engine()
    sensitivity, _ = SensitivityAnalyzer(engine).analyze(
        model=model,
        result=result,
        validation=validation,
        config=SensitivityConfig(perturbation_fractions=[0.05]),
    )
    robustness, _ = RobustnessAnalyzer(engine).analyze(
        model=model,
        result=result,
        validation=validation,
        sensitivity=sensitivity,
        config=RobustnessConfig(scenario_fractions=[-0.05, 0.05]),
    )
    return model, validation, sensitivity, robustness


def test_red_team_mock_review_cannot_pass_quality_gate() -> None:
    model, validation, sensitivity, robustness = _phase5_reports()
    report = RedTeamAnalyzer().compile(
        model=model,
        validation=validation,
        sensitivity=sensitivity,
        robustness=robustness,
        draft=RedTeamDraft(summary="No additional finding in the supplied fixture."),
        reviewer_agent_run_id=uuid4(),
        review_is_mock=True,
    )

    assert report.status is ValidationStatus.INCONCLUSIVE
    assert report.critical_count == 0
    assert red_team_quality_gate(report).status.value == "HUMAN_REVIEW"
    invalid = report.model_dump()
    invalid["status"] = ValidationStatus.PASS
    with pytest.raises(ValidationError, match="Mock red-team review cannot pass"):
        type(report).model_validate(invalid)


def test_critical_red_team_finding_requires_scoped_non_mock_repair() -> None:
    model, validation, sensitivity, robustness = _phase5_reports()
    finding = RedTeamFinding(
        finding_id="RTF-critical-demand",
        severity=RedTeamSeverity.CRITICAL,
        category=RedTeamCategory.CONSTRAINT,
        title="Demand constraint omits a required reserve",
        attack="The accepted evidence requires reserve capacity that the model omits.",
        evidence_refs=["EVID-fact-1"],
        affected_refs=["CON-1"],
        recommendation="Add the evidenced reserve to the demand constraint.",
    )
    report = RedTeamAnalyzer().compile(
        model=model,
        validation=validation,
        sensitivity=sensitivity,
        robustness=robustness,
        draft=RedTeamDraft(findings=[finding], summary="One critical omission."),
        reviewer_agent_run_id=uuid4(),
        review_is_mock=False,
    )
    revised_parameter = model.parameters[0].model_copy(
        update={"value": float(model.parameters[0].value) * 1.05}
    )
    revised = model.model_copy(update={"version": 2, "parameters": [revised_parameter]})
    action = RepairAction(
        action_id="REPAIR-demand",
        finding_ids=[finding.finding_id],
        target_type=RepairTargetType.CONSTRAINT,
        target_ref="CON-1",
        description="Add the evidenced reserve term.",
        rationale="Directly addresses the critical Red Team finding.",
    )
    output = ModelRepairOutput(
        revised_model=revised,
        actions=[action],
        addressed_finding_ids=[finding.finding_id],
    )
    no_op = output.model_copy(update={"revised_model": model.model_copy(update={"version": 2})})

    assert report.status is ValidationStatus.FAIL
    assert red_team_quality_gate(report).status.value == "RETRY"
    assert repair_quality_gate(output, report, is_mock=False).status.value == "PASS"
    assert repair_quality_gate(no_op, report, is_mock=False).status.value == "RETRY"
    assert repair_quality_gate(output, report, is_mock=True).status.value == "HUMAN_REVIEW"
    incomplete = output.model_copy(update={"addressed_finding_ids": ["RTF-other"]})
    assert repair_quality_gate(incomplete, report, is_mock=False).status.value == "RETRY"
    with pytest.raises(ValidationError, match="Mock repair cannot be accepted"):
        RepairCycleRecord(
            project_id=model.project_id,
            problem_id=model.problem_id,
            stable_model_id=model.model_id,
            source_model_version=model.version,
            source_model_digest=validation.model_digest,
            target_model_version=model.version + 1,
            target_model_digest=validation.model_digest,
            red_team_report_id=report.report_id,
            repair_cycle=1,
            agent_run_id=uuid4(),
            is_mock=True,
            status=RepairCycleStatus.ACCEPTED,
        )


@pytest.mark.asyncio
async def test_three_existing_repair_cycles_force_human_review() -> None:
    model, validation, sensitivity, robustness = _phase5_reports()
    finding = RedTeamFinding(
        finding_id="RTF-still-critical",
        severity=RedTeamSeverity.CRITICAL,
        category=RedTeamCategory.CONSTRAINT,
        title="Critical issue remains after three cycles",
        attack="The accepted model still omits a required constraint.",
        evidence_refs=["EVID-fact-1"],
        recommendation="Require human review.",
    )
    report = RedTeamAnalyzer().compile(
        model=model,
        validation=validation,
        sensitivity=sensitivity,
        robustness=robustness,
        draft=RedTeamDraft(findings=[finding], summary="Critical issue remains."),
        reviewer_agent_run_id=uuid4(),
        review_is_mock=False,
    )
    state = ProblemState(
        schema_version=5,
        project_id=model.project_id,
        problem_id=model.problem_id,
        title="repair limit",
        raw_problem="fixture",
        version=9,
        current_stage=WorkflowStage.RED_TEAM,
        status=WorkflowStatus.RETRY,
    )

    class Repository:
        saved: ProblemState | None = None

        def get_red_team(self, project_id):
            assert project_id == model.project_id
            return report

        def repair_cycle_count(self, model_id):
            assert model_id == model.model_id
            return 3

        def persist_state_only(self, next_state):
            self.saved = next_state

    class StateRepository:
        def __init__(self, repository: Repository) -> None:
            self.repository = repository

        def load_current(self, project_id):
            assert project_id == model.project_id
            return self.repository.saved or state

    repository = Repository()
    workflow = object.__new__(VerificationWorkflow)
    workflow._repository = repository  # type: ignore[attr-defined]
    workflow._reasoning_repository = StateRepository(repository)  # type: ignore[attr-defined]
    workflow._max_repair_cycles = 3  # type: ignore[attr-defined]

    outcome = await workflow.repair_until_clear(model.project_id)

    assert outcome.exhausted is True
    assert outcome.resolved is False
    assert outcome.iterations == []
    assert outcome.state.status is WorkflowStatus.HUMAN_REVIEW
    assert outcome.state.version == 10
    assert outcome.state.quality_gates[-1].gate == "MODEL_REPAIR"
    assert outcome.state.quality_gates[-1].status.value == "HUMAN_REVIEW"


def test_failed_agent_output_is_audited_before_error() -> None:
    model, _, _, _ = _phase5_reports()
    state = ProblemState(
        schema_version=5,
        project_id=model.project_id,
        problem_id=model.problem_id,
        title="agent audit",
        raw_problem="fixture",
        current_stage=WorkflowStage.RED_TEAM,
    )

    class StateRepository:
        recorded = None

        def record_run(self, project_id, problem_id, run):
            self.recorded = (project_id, problem_id, run)

    repository = StateRepository()
    workflow = object.__new__(VerificationWorkflow)
    workflow._reasoning_repository = repository  # type: ignore[attr-defined]
    now = datetime.now(UTC)
    run = AgentRunResult[RedTeamDraft](
        agent_name="red_team_agent",
        status=AgentRunStatus.FAILED,
        attempts=1,
        is_mock=False,
        input_state_version=state.version,
        latency_ms=0,
        started_at=now,
        ended_at=now,
        errors=["fixture failure"],
    )

    with pytest.raises(AgentRunError, match="fixture failure"):
        workflow._require_agent_output(state, run, "Red Team")

    assert repository.recorded == (state.project_id, state.problem_id, run)
