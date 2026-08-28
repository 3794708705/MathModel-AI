from copy import deepcopy

import pytest

from mathmodel_ai.mathematical.digests import mathematical_model_digest
from mathmodel_ai.schemas.mathematical import ExpressionKind, MathExpression
from mathmodel_ai.schemas.results import EvidenceChainReport
from mathmodel_ai.schemas.verification import ValidationCheckStatus, ValidationStatus
from mathmodel_ai.verification.evaluator import (
    IndependentEvaluationError,
    IndependentExpressionEvaluator,
)
from mathmodel_ai.verification.quality_gates import validation_quality_gate
from mathmodel_ai.verification.validation import IndependentValidator
from tests.mathematical.helpers import constant, symbol
from tests.verification.helpers import result_bundle


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
