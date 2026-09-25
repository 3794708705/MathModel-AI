import asyncio
from copy import deepcopy
from unittest.mock import Mock
from uuid import uuid4

import pytest

from mathmodel_ai.core.errors import QualityGateError
from mathmodel_ai.mathematical.digests import mathematical_model_digest
from mathmodel_ai.schemas.files import ArtifactKind, ArtifactRecord
from mathmodel_ai.schemas.independent_verification import (
    CsvObservationSpec,
    IndependentStatus,
    IndependentVerificationReport,
    MetricSpec,
    ReplayRecord,
    ReviewedValidationEvidence,
    ScenarioSpec,
    ValidationRequirementBinding,
    VerificationPlan,
    VerificationRequirements,
    VerifiedMetric,
)
from mathmodel_ai.schemas.mathematical import ExpressionKind, MathExpression
from mathmodel_ai.schemas.quality import QualityGateStatus
from mathmodel_ai.schemas.results import EvidenceChainReport
from mathmodel_ai.schemas.verification import ValidationCheckStatus, ValidationStatus
from mathmodel_ai.verification.causal_holdout import AuditedCausalEvidence, CausalHoldoutResult
from mathmodel_ai.verification.evaluator import (
    IndependentEvaluationError,
    IndependentExpressionEvaluator,
)
from mathmodel_ai.verification.metric_recompute import content_digest
from mathmodel_ai.verification.quality_gates import validation_quality_gate
from mathmodel_ai.verification.validation import IndependentValidator
from mathmodel_ai.verification.workflow import VerificationWorkflow
from tests.mathematical.helpers import constant, symbol
from tests.verification.helpers import result_bundle


def _reviewed_validation_evidence():
    model, result, solver_run, execution, _, evidence = result_bundle()
    common = {
        "unit": "dimensionless",
        "tolerance_provenance": "fixture numerical identity",
        "model_binding": result.model_digest,
    }
    baseline_spec = MetricSpec(
        metric_id="objective",
        key="objective",
        version="1",
        quantity="formal objective",
        calculation="independent objective AST",
        inputs=["x", "y"],
        **common,
    )
    scenario_metric = MetricSpec(
        metric_id="stress_objective",
        key="objective",
        version="1",
        quantity="stress objective",
        calculation="independent scenario objective AST",
        inputs=["x", "y"],
        **common,
    )
    scenario = ScenarioSpec(
        scenario_id="demand_stress",
        version="1",
        parameter_values={"demand": 11.0},
        metrics=[scenario_metric],
        baseline="formal baseline demand",
        perturbation="demand=11",
        reason="exercise a changed input",
        input_changes=["demand"],
        comparison_quantity="objective",
        acceptance_criterion="independent objective recomputation passes",
        criterion_provenance="fixture mathematical identity",
    )
    bindings = [
        ValidationRequirementBinding(
            requirement=requirement,
            metric_ids=["objective"],
            scenario_ids=["demand_stress"],
        )
        for requirement in model.validation_requirements
    ]
    policy = VerificationRequirements(
        version="2",
        review_status="REVIEWED",
        production_eligible=True,
        benchmark_id="BENCH-fixture",
        manifest_digest="f" * 64,
        model_digest=result.model_digest,
        metrics=[baseline_spec],
        scenarios=[scenario],
        validation_requirement_bindings=bindings,
        scientific_scope="Explicit fixture verification evidence bridge.",
    )
    plan = VerificationPlan(
        attempt_id=uuid4(),
        result_id=result.result_id,
        version="1",
        source_artifact_id=uuid4(),
        source_sha256="a" * 64,
        metrics=policy.metrics,
        scenarios=policy.scenarios,
        scientific_scope=policy.scientific_scope,
    )
    output_digest = "d" * 64
    replay_execution = execution.model_copy(update={"run_id": uuid4()})
    output = ArtifactRecord(
        project_id=model.project_id,
        problem_id=model.problem_id,
        execution_run_id=replay_execution.run_id,
        kind=ArtifactKind.SANDBOX_OUTPUT,
        name="result.json",
        mime_type="application/json",
        size_bytes=2,
        sha256=output_digest,
        storage_key="fixture/result.json",
    )
    replay_metric = VerifiedMetric(
        metric_id=scenario_metric.metric_id,
        key=scenario_metric.key,
        reported=33.0,
        verified=33.0,
        delta=0.0,
        status=IndependentStatus.PASS,
        source_digest=output_digest,
        calculator_digest="c" * 64,
    )
    replay = ReplayRecord(
        scenario_id=scenario.scenario_id,
        scenario_digest=content_digest(scenario),
        input_digest="e" * 64,
        input_parameters={"demand": 11.0},
        generator_version="fixture-v1",
        execution=replay_execution,
        artifacts=[output],
        output_artifact_id=output.artifact_id,
        output_digest=output.sha256,
        status=IndependentStatus.PASS,
        metrics=[replay_metric],
    )
    baseline_metric = VerifiedMetric(
        metric_id=baseline_spec.metric_id,
        key=baseline_spec.key,
        reported=30.0,
        verified=30.0,
        delta=0.0,
        status=IndependentStatus.PASS,
        source_digest=plan.source_sha256,
        calculator_digest="c" * 64,
    )
    independent_report = IndependentVerificationReport(
        attempt_id=plan.attempt_id,
        plan_id=plan.plan_id,
        report_id=plan.plan_id,
        plan_digest=content_digest(plan),
        result_id=result.result_id,
        status=IndependentStatus.PASS,
        metrics=[baseline_metric],
        replays=[replay],
        required_metrics=1,
        passed_metrics=1,
        required_scenarios=1,
        passed_scenarios=1,
        errors=[],
    )
    reviewed = ReviewedValidationEvidence(
        requirements=policy,
        plan=plan,
        report=independent_report,
    )
    return model, result, solver_run, evidence, reviewed


def test_independent_validator_recomputes_full_result_and_passes_gate() -> None:
    model, result, solver_run, _, _, evidence = result_bundle()

    report = IndependentValidator().validate(
        model=model,
        result=result,
        solver_run=solver_run,
        evidence=evidence,
    )
    gate = validation_quality_gate(report)

    assert report.status is ValidationStatus.PASS
    assert report.max_constraint_violation == pytest.approx(0)
    assert all(item.status is ValidationCheckStatus.PASS for item in report.variable_checks)
    assert all(item.status is ValidationCheckStatus.PASS for item in report.constraint_checks)
    assert report.metric_recalculations[0].recomputed_value == pytest.approx(30)
    assert gate.status.value == "PASS"


def test_reviewed_evidence_bridge_checks_exact_plan_report_and_requirements() -> None:
    model, result, solver_run, evidence, reviewed = _reviewed_validation_evidence()

    report = IndependentValidator().validate(
        model=model,
        result=result,
        solver_run=solver_run,
        evidence=evidence,
        reviewed_evidence=reviewed,
    )

    assert report.status is ValidationStatus.PASS
    assert any(item.metric_id == "independent:objective" for item in report.metric_recalculations)
    assert all(item.status is ValidationCheckStatus.PASS for item in report.requirement_checks)
    assert f"independent_report:{reviewed.report.report_id}" in report.evidence_refs
    assert validation_quality_gate(report).status.value == "PASS"

    source = CsvObservationSpec(
        source_csv_sha256="d" * 64,
        source_column="outcome",
        positive_value="1",
        negative_value="0",
    )
    changed_policy = reviewed.requirements.model_copy(update={"csv_observation": source})
    mismatched = reviewed.model_copy(update={"requirements": changed_policy})
    rejected = IndependentValidator().validate(
        model=model,
        result=result,
        solver_run=solver_run,
        evidence=evidence,
        reviewed_evidence=mismatched,
    )
    assert rejected.status is ValidationStatus.FAIL
    assert "VALIDATION_FAIL:REVIEWED_PLAN_POLICY_MISMATCH" in rejected.errors


@pytest.mark.parametrize(
    "tamper",
    [
        "wrong_result",
        "missing_binding",
        "reused_execution",
        "mock_or_failed_execution",
        "missing_output_artifact",
        "failed_scenario_metric",
        "tampered_counts",
    ],
)
def test_reviewed_evidence_bridge_fails_closed_on_tampering(tamper: str) -> None:
    model, result, solver_run, evidence, reviewed = _reviewed_validation_evidence()
    independent = reviewed.report
    if tamper == "wrong_result":
        independent = independent.model_copy(update={"result_id": uuid4()})
    elif tamper == "missing_binding":
        reviewed = reviewed.model_copy(
            update={
                "requirements": reviewed.requirements.model_copy(
                    update={
                        "validation_requirement_bindings": (
                            reviewed.requirements.validation_requirement_bindings[:-1]
                        )
                    }
                )
            }
        )
    elif tamper == "reused_execution":
        replay = independent.replays[0]
        execution = replay.execution.model_copy(update={"run_id": solver_run.execution_ref})
        artifact = replay.artifacts[0].model_copy(
            update={"execution_run_id": solver_run.execution_ref}
        )
        independent = independent.model_copy(
            update={
                "replays": [
                    replay.model_copy(update={"execution": execution, "artifacts": [artifact]})
                ]
            }
        )
    elif tamper == "mock_or_failed_execution":
        replay = independent.replays[0]
        execution = replay.execution.model_copy(
            update={"status": "FAILED", "exit_code": 1, "network_disabled": False}
        )
        independent = independent.model_copy(
            update={"replays": [replay.model_copy(update={"execution": execution})]}
        )
    elif tamper == "missing_output_artifact":
        replay = independent.replays[0]
        independent = independent.model_copy(
            update={"replays": [replay.model_copy(update={"artifacts": []})]}
        )
    elif tamper == "failed_scenario_metric":
        replay = independent.replays[0]
        metric = replay.metrics[0].model_copy(update={"status": IndependentStatus.FAIL})
        independent = independent.model_copy(
            update={"replays": [replay.model_copy(update={"metrics": [metric]})]}
        )
    else:
        independent = independent.model_copy(update={"passed_scenarios": 0})
    if tamper != "missing_binding":
        reviewed = reviewed.model_copy(update={"report": independent})

    report = IndependentValidator().validate(
        model=model,
        result=result,
        solver_run=solver_run,
        evidence=evidence,
        reviewed_evidence=reviewed,
    )

    assert report.status is not ValidationStatus.PASS
    assert validation_quality_gate(report).status.value == "RETRY"


def test_independent_validator_rejects_objective_and_variable_tampering() -> None:
    model, result, solver_run, _, _, evidence = result_bundle()
    broken_result = result.model_copy(update={"objective": 999.0})
    payload = deepcopy(solver_run.result.model_dump())
    payload["variable_values"] = {"x": 2.0, "y": 0.0}
    broken_solver = solver_run.model_copy(
        update={"result": solver_run.result.model_validate(payload)}
    )

    report = IndependentValidator().validate(
        model=model,
        result=broken_result,
        solver_run=broken_solver,
        evidence=EvidenceChainReport(
            **evidence.model_dump(exclude={"valid", "errors"}),
            valid=False,
            errors=[],
        ),
    )

    assert report.status is ValidationStatus.FAIL
    assert any(item.status is ValidationCheckStatus.FAIL for item in report.constraint_checks)
    assert any(item.status is ValidationCheckStatus.FAIL for item in report.metric_recalculations)


def test_independent_validator_marks_unknown_requirement_not_evaluable() -> None:
    model, result, solver_run, _, _, evidence = result_bundle()
    model = model.model_copy(
        update={"validation_requirements": [*model.validation_requirements, "manual visual review"]}
    )
    digest = mathematical_model_digest(model)
    result = result.model_copy(update={"model_digest": digest})
    solver_run = solver_run.model_copy(update={"model_digest": digest})

    report = IndependentValidator().validate(
        model=model,
        result=result,
        solver_run=solver_run,
        evidence=evidence,
    )

    assert report.status is ValidationStatus.NOT_EVALUABLE
    assert report.requirement_checks[-1].status is ValidationCheckStatus.UNCHECKED
    assert validation_quality_gate(report).status.value == "RETRY"


def test_audited_holdout_is_bound_but_does_not_satisfy_unrelated_requirements() -> None:
    model, result, solver_run, _, _, evidence = result_bundle()
    model = model.model_copy(
        update={
            "validation_requirements": [
                "Recalculate held-out match Brier score.",
                "Check calibration and match flow against official data.",
            ]
        }
    )
    digest = mathematical_model_digest(model)
    result = result.model_copy(update={"model_digest": digest})
    solver_run = solver_run.model_copy(update={"model_digest": digest})
    causal = AuditedCausalEvidence(
        formal_result_id=result.result_id,
        holdout_execution_id=uuid4(),
        source_sha256="a" * 64,
        trace_sha256="b" * 64,
        result=CausalHoldoutResult(
            heldout_groups=("match-2",),
            training_groups=("match-1",),
            predictions=(0.6,),
            observations=(1.0,),
            baseline_predictions=(0.5,),
            brier=0.16,
            baseline_brier=0.25,
        ),
    )
    validator = IndependentValidator()
    report = validator.validate(
        model=model,
        result=result,
        solver_run=solver_run,
        evidence=evidence,
        causal_evidence=causal,
    )
    assert report.status is ValidationStatus.NOT_EVALUABLE
    assert [item.metric_id for item in report.metric_recalculations][-2:] == [
        "causal_holdout:brier",
        "causal_holdout:baseline_brier",
    ]
    assert all(item.status is ValidationCheckStatus.UNCHECKED for item in report.requirement_checks)
    assert validation_quality_gate(report).status.value == "RETRY"
    assert (
        validator.audit_report(
            report=report,
            model=model,
            result=result,
            solver_run=solver_run,
            evidence=evidence,
            causal_evidence=causal,
        )
        == []
    )
    assert "VALIDATION_REPORT_MISMATCH:metric_recalculations" in validator.audit_report(
        report=report,
        model=model,
        result=result,
        solver_run=solver_run,
        evidence=evidence,
    )
    wrong = AuditedCausalEvidence(
        formal_result_id=uuid4(),
        holdout_execution_id=causal.holdout_execution_id,
        source_sha256=causal.source_sha256,
        trace_sha256=causal.trace_sha256,
        result=causal.result,
    )
    rejected = validator.validate(
        model=model,
        result=result,
        solver_run=solver_run,
        evidence=evidence,
        causal_evidence=wrong,
    )
    assert rejected.status is ValidationStatus.FAIL
    assert "VALIDATION_FAIL:CAUSAL_HOLDOUT_FORMAL_RESULT_MISMATCH" in rejected.errors


def test_causal_workflow_stops_after_formal_report_without_reviewed_policy() -> None:
    workflow = object.__new__(VerificationWorkflow)
    workflow.validate = Mock(  # type: ignore[method-assign]
        return_value=Mock(gate=Mock(status=QualityGateStatus.PASS))
    )
    with pytest.raises(
        QualityGateError, match="CAUSAL_HOLDOUT_REVIEWED_REQUIREMENT_POLICY_MISSING"
    ):
        asyncio.run(workflow.run(uuid4(), causal_auditor=Mock()))
    workflow.validate.assert_called_once()  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "requirement",
    [
        "Compute Jacobian eigenvalues and compare variable and fixed rules for stability.",
        "Estimate population bounds by bootstrap resampling.",
        "Verify every constraint under all out-of-sample scenarios.",
        "Recalculate objective metric and prove global optimality.",
        "Validate output predictions against independent empirical observations.",
        "Verify external literature evidence supports the parameter choices.",
        "recompute variable bounds and constraints; perform Monte Carlo analysis",
        "Sensitivity analysis of the objective under parameter changes.",
    ],
)
def test_scientific_requirements_cannot_pass_from_keyword_collisions(requirement: str) -> None:
    model, result, solver_run, _, _, evidence = result_bundle()
    model = model.model_copy(update={"validation_requirements": [requirement]})
    digest = mathematical_model_digest(model)
    result = result.model_copy(update={"model_digest": digest})
    solver_run = solver_run.model_copy(update={"model_digest": digest})
    validator = IndependentValidator()

    report = validator.validate(
        model=model, result=result, solver_run=solver_run, evidence=evidence
    )

    assert report.status is ValidationStatus.NOT_EVALUABLE
    check = report.requirement_checks[0]
    assert check.status is ValidationCheckStatus.UNCHECKED
    assert check.evidence_refs == []
    assert validation_quality_gate(report).status.value == "RETRY"
    # Re-auditing old keyword-based PASS evidence must also reject it.
    stale = report.model_copy(
        update={
            "status": ValidationStatus.PASS,
            "validator_version": "independent-validator-5.0.0",
            "requirement_checks": [check.model_copy(update={"status": ValidationCheckStatus.PASS})],
        }
    )
    errors = validator.audit_report(
        report=stale, model=model, result=result, solver_run=solver_run, evidence=evidence
    )
    assert "VALIDATION_REPORT_MISMATCH:requirement_checks" in errors
    assert "VALIDATION_REPORT_MISMATCH:validator_version" in errors


def test_combined_requirement_checks_bounds_even_when_constraints_pass() -> None:
    model, result, solver_run, _, _, evidence = result_bundle()
    model = model.model_copy(
        update={
            "decision_variables": [
                model.decision_variables[0].model_copy(update={"upper_bound": 5.0}),
                model.decision_variables[1],
            ],
            "validation_requirements": ["recompute variable bounds and constraints"],
        }
    )
    digest = mathematical_model_digest(model)
    report = IndependentValidator().validate(
        model=model,
        result=result.model_copy(update={"model_digest": digest}),
        solver_run=solver_run.model_copy(update={"model_digest": digest}),
        evidence=evidence,
    )

    assert all(item.status is ValidationCheckStatus.PASS for item in report.constraint_checks)
    assert report.requirement_checks[0].status is ValidationCheckStatus.FAIL
    assert model.decision_variables[0].variable_id in report.requirement_checks[0].evidence_refs
    assert model.constraints[0].constraint_id in report.requirement_checks[0].evidence_refs


def test_combined_requirement_does_not_silently_drop_empty_check_component() -> None:
    model, result, solver_run, _, _, evidence = result_bundle()
    model = model.model_copy(
        update={
            "constraints": [],
            "validation_requirements": ["recompute variable bounds and constraints"],
        }
    )
    digest = mathematical_model_digest(model)
    report = IndependentValidator().validate(
        model=model,
        result=result.model_copy(update={"model_digest": digest}),
        solver_run=solver_run.model_copy(update={"model_digest": digest}),
        evidence=evidence,
    )
    assert report.requirement_checks[0].status is ValidationCheckStatus.UNCHECKED
    assert report.status is ValidationStatus.NOT_EVALUABLE


def test_validation_report_audit_detects_persisted_semantic_tamper() -> None:
    model, result, solver_run, _, _, evidence = result_bundle()
    validator = IndependentValidator()
    report = validator.validate(
        model=model,
        result=result,
        solver_run=solver_run,
        evidence=evidence,
    )
    tampered = report.model_copy(update={"constraint_checks": []})

    errors = validator.audit_report(
        report=tampered,
        model=model,
        result=result,
        solver_run=solver_run,
        evidence=evidence,
    )

    assert "VALIDATION_REPORT_MISMATCH:constraint_checks" in errors


def test_independent_evaluator_rejects_missing_symbols_and_division_by_zero() -> None:
    evaluator = IndependentExpressionEvaluator()

    with pytest.raises(IndependentEvaluationError, match="no value"):
        evaluator.evaluate(symbol("missing"), {})
    with pytest.raises(IndependentEvaluationError, match="division by zero"):
        evaluator.evaluate(
            MathExpression(
                kind=ExpressionKind.DIVIDE,
                operands=[constant(1), constant(0)],
            ),
            {},
        )
