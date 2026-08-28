from types import SimpleNamespace
from uuid import uuid4

import pytest
from pydantic import ValidationError

from mathmodel_ai.mathematical.algorithms import AlgorithmSelector
from mathmodel_ai.schemas.verification import (
    ExperimentReportStatus,
    ModelRepairOutput,
    RedTeamCategory,
    RedTeamDraft,
    RedTeamFinding,
    RedTeamSeverity,
    RepairAction,
    RepairTargetType,
    RobustnessConfig,
    RobustnessMethod,
    SensitivityConfig,
    ValidationStatus,
)
from mathmodel_ai.verification.experiments import ExperimentEngine
from mathmodel_ai.verification.quality_gates import (
    red_team_quality_gate,
    verified_result_quality_gate,
)
from mathmodel_ai.verification.red_team import RedTeamAnalyzer
from mathmodel_ai.verification.robustness import RobustnessAnalyzer
from mathmodel_ai.verification.sensitivity import SensitivityAnalyzer
from mathmodel_ai.verification.validation import IndependentValidator
from tests.verification.helpers import (
    ScenarioSolver,
    experiment_engine,
    result_bundle,
    valid_report,
)


def _passing_chain():
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
    red_team = RedTeamAnalyzer().compile(
        model=model,
        validation=validation,
        sensitivity=sensitivity,
        robustness=robustness,
        draft=RedTeamDraft(summary="No unresolved finding in the fixture."),
        reviewer_agent_run_id=uuid4(),
        review_is_mock=False,
    )
    return model, result, validation, sensitivity, robustness, red_team


def _critical_report(model, validation, sensitivity, robustness):
    finding = RedTeamFinding(
        finding_id="RTF-adversarial-critical",
        severity=RedTeamSeverity.CRITICAL,
        category=RedTeamCategory.CONSTRAINT,
        title="Critical constraint omission",
        attack="The model omits a required evidenced constraint.",
        evidence_refs=["EVID-fact-1"],
        affected_refs=["CON-1"],
        recommendation="Repair the model and rerun the complete chain.",
    )
    return RedTeamAnalyzer().compile(
        model=model,
        validation=validation,
        sensitivity=sensitivity,
        robustness=robustness,
        draft=RedTeamDraft(findings=[finding], summary="One critical finding."),
        reviewer_agent_run_id=uuid4(),
        review_is_mock=False,
    )


def _verified_gate(validation, sensitivity, robustness, red_team):
    return verified_result_quality_gate(
        validation,
        sensitivity,
        robustness,
        red_team,
        validation_integrity_errors=[],
        sensitivity_integrity_errors=[],
        robustness_integrity_errors=[],
    )


def test_a_critical_red_team_finding_cannot_be_verified() -> None:
    model, _, validation, sensitivity, robustness, _ = _passing_chain()
    critical = _critical_report(model, validation, sensitivity, robustness)

    assert red_team_quality_gate(critical).status.value == "RETRY"
    assert _verified_gate(validation, sensitivity, robustness, critical).status.value == "RETRY"


def test_b_not_evaluable_validation_cannot_be_verified() -> None:
    _, _, validation, sensitivity, robustness, red_team = _passing_chain()
    not_evaluable = validation.model_copy(update={"status": ValidationStatus.NOT_EVALUABLE})

    gate = _verified_gate(not_evaluable, sensitivity, robustness, red_team)

    assert gate.status.value == "HUMAN_REVIEW"
    assert gate.checks["validation_pass"] is False


def test_c_solver_objective_tamper_fails_independent_recalculation() -> None:
    model, result, solver_run, _, _, evidence = result_bundle()
    tampered_result = solver_run.result.model_copy(update={"objective_value": 999.0})
    tampered_run = solver_run.model_copy(update={"result": tampered_result})

    report = IndependentValidator().validate(
        model=model,
        result=result,
        solver_run=tampered_run,
        evidence=evidence,
    )

    assert report.status is ValidationStatus.FAIL
    assert any(
        item.metric_id.endswith(":solver_run") and item.status.value == "FAIL"
        for item in report.metric_recalculations
    )


class _BaselineReplayRouter:
    def __init__(self, baseline_model) -> None:
        self._baseline_model = baseline_model
        self._solver = ScenarioSolver()

    def route(self, model, plan, options):
        del model, plan
        solver = self._solver
        baseline = self._baseline_model

        class ReplaySolver:
            def solve(self, ignored_model, selected_options):
                del ignored_model
                return solver.solve(baseline, selected_options)

        return SimpleNamespace(solver=ReplaySolver(), options=options)


def _replay_engine(model) -> ExperimentEngine:
    validator = IndependentValidator()
    return ExperimentEngine(
        algorithm_selector=AlgorithmSelector(),
        solver_router=_BaselineReplayRouter(model),  # type: ignore[arg-type]
        validator=validator,
    )


def test_d_sensitivity_metadata_replay_is_detected() -> None:
    model, result, validation, _ = valid_report()

    report, _ = SensitivityAnalyzer(_replay_engine(model)).analyze(
        model=model,
        result=result,
        validation=validation,
        config=SensitivityConfig(perturbation_fractions=[0.10]),
    )

    assert report.status is ExperimentReportStatus.FAIL
    assert all(
        item.error is not None and "EXECUTION_MODEL_DIGEST_MISMATCH" in item.error
        for item in report.experiments
    )


def test_e_robustness_stress_replay_is_detected() -> None:
    model, result, validation, _ = valid_report()
    sensitivity, _ = SensitivityAnalyzer(experiment_engine()).analyze(
        model=model,
        result=result,
        validation=validation,
        config=SensitivityConfig(perturbation_fractions=[0.05]),
    )

    report, _ = RobustnessAnalyzer(_replay_engine(model)).analyze(
        model=model,
        result=result,
        validation=validation,
        sensitivity=sensitivity,
        config=RobustnessConfig(scenario_fractions=[-0.10, 0.10]),
    )

    assert report.status is ExperimentReportStatus.FAIL
    assert all(
        item.error is not None and "EXECUTED_SCENARIO_PAYLOAD_MISMATCH" in item.error
        for item in report.experiments
    )


def test_f_tampered_persisted_critical_count_is_recomputed() -> None:
    model, _, validation, sensitivity, robustness, _ = _passing_chain()
    critical = _critical_report(model, validation, sensitivity, robustness)
    tampered = critical.model_copy(update={"critical_count": 0, "status": ValidationStatus.PASS})

    red_gate = red_team_quality_gate(tampered)
    verified_gate = _verified_gate(validation, sensitivity, robustness, tampered)

    assert red_gate.status.value == "RETRY"
    assert red_gate.checks["severity_counts_recomputed"] is False
    assert red_gate.checks["no_unresolved_critical_findings"] is False
    assert verified_gate.status.value == "RETRY"


def test_model_repair_contract_forbids_result_mutation() -> None:
    model, result, validation, sensitivity, robustness, _ = _passing_chain()
    critical = _critical_report(model, validation, sensitivity, robustness)
    action = RepairAction(
        action_id="REPAIR-model-only",
        finding_ids=[critical.findings[0].finding_id],
        target_type=RepairTargetType.CONSTRAINT,
        target_ref="CON-1",
        description="Repair only the mathematical constraint.",
        rationale="The result must be regenerated by a new solve.",
    )
    payload = {
        "revised_model": model.model_copy(update={"version": 2}),
        "actions": [action],
        "addressed_finding_ids": [critical.findings[0].finding_id],
        "result": result,
    }

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ModelRepairOutput.model_validate(payload)


def test_i_cross_result_chain_cannot_be_verified() -> None:
    _, _, validation, sensitivity, robustness, red_team = _passing_chain()
    stale = robustness.model_copy(update={"result_id": uuid4()})

    gate = _verified_gate(validation, sensitivity, stale, red_team)

    assert gate.status.value == "RETRY"
    assert gate.checks["single_formal_result"] is False


def test_j_mock_or_blocked_evidence_requires_human_review() -> None:
    model, result, validation, sensitivity, robustness, _ = _passing_chain()
    mock_red_team = RedTeamAnalyzer().compile(
        model=model,
        validation=validation,
        sensitivity=sensitivity,
        robustness=robustness,
        draft=RedTeamDraft(summary="Mock fixture has no finding."),
        reviewer_agent_run_id=uuid4(),
        review_is_mock=True,
    )
    blocked, _ = RobustnessAnalyzer(experiment_engine()).analyze(
        model=model,
        result=result,
        validation=validation,
        sensitivity=sensitivity,
        config=RobustnessConfig(method=RobustnessMethod.BOOTSTRAP),
    )

    assert (
        _verified_gate(validation, sensitivity, robustness, mock_red_team).status.value
        == "HUMAN_REVIEW"
    )
    assert (
        _verified_gate(validation, sensitivity, blocked, mock_red_team).status.value
        == "HUMAN_REVIEW"
    )
