from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import Mock
from uuid import uuid4

import pytest

from mathmodel_ai.core.errors import SolverUnavailableError
from mathmodel_ai.mathematical.algorithms import AlgorithmSelector
from mathmodel_ai.mathematical.digests import mathematical_model_digest
from mathmodel_ai.mathematical.evidence import EvidenceIntegrityVerifier
from mathmodel_ai.mathematical.strategy import ExecutionStrategySelector
from mathmodel_ai.schemas.execution import (
    ExecutionOrigin,
    ExecutionRecord,
    ExecutionStatus,
    SandboxLimits,
)
from mathmodel_ai.schemas.program import ExecutionStrategy
from mathmodel_ai.schemas.results import EvidenceErrorCode, ResultRecord
from mathmodel_ai.schemas.solver import (
    FeasibilityReport,
    ProblemSize,
    ProblemSizeClass,
    SolverFamily,
    SolverName,
    SolverOptions,
    SolverResult,
    SolverRouteDecision,
    SolverRun,
    SolverStatus,
)
from mathmodel_ai.solvers.programs import build_deterministic_program
from mathmodel_ai.solvers.router import SolverRouter
from tests.mathematical.helpers import add, constant, lp_model, multiply, symbol


def _evidence_chain() -> tuple[
    ResultRecord,
    SolverRun,
    ExecutionRecord,
    object,
    object,
]:
    model = lp_model()
    digest = mathematical_model_digest(model)
    options = SolverOptions(time_limit_seconds=20)
    program = build_deterministic_program(
        model,
        options,
        backend="scipy",
        target=SolverFamily.SCIPY_HIGHS,
        dependencies=["scipy"],
    )
    entrypoint = program.files[0]
    assert entrypoint.sha256 is not None
    execution_id = uuid4()
    execution = ExecutionRecord(
        run_id=execution_id,
        project_id=model.project_id,
        problem_id=model.problem_id,
        code_hash=entrypoint.sha256,
        executed_bundle_hash=program.code_hash,
        code_artifact_id=uuid4(),
        execution_origin=ExecutionOrigin.DETERMINISTIC_SOLVER_ADAPTER,
        model_digest=digest,
        generated_program_id=program.program_id,
        image="solver:test",
        image_id="sha256:test",
        end_time=datetime.now(UTC),
        runtime_seconds=0.01,
        status=ExecutionStatus.SUCCEEDED,
        exit_code=0,
        limits=SandboxLimits(
            cpu_cores=1,
            memory_mb=64,
            timeout_seconds=20,
            pids_limit=16,
            max_output_bytes=1024,
            max_artifacts=2,
            max_artifact_bytes=1024,
        ),
    )
    feasibility = FeasibilityReport(
        checked=True,
        max_constraint_violation=0,
        tolerance=1e-7,
        message="fixture solution is feasible",
    )
    solver_result = SolverResult(
        solver_name=SolverName.SCIPY,
        solver_version="fixture",
        status=SolverStatus.OPTIMAL,
        objective_value=30,
        variable_values={"x": 10, "y": 0},
        runtime_seconds=0.01,
        is_optimal=True,
        is_feasible=True,
        execution_record_id=execution_id,
        feasibility=feasibility,
    )
    result_id = uuid4()
    solver_run_id = uuid4()
    route = SolverRouteDecision(
        selected_solver=SolverName.SCIPY,
        selected_family=SolverFamily.SCIPY_HIGHS,
        attempted_families=[SolverFamily.SCIPY_HIGHS],
        problem_size=ProblemSize(
            classification=ProblemSizeClass.TINY,
            number_of_variables=2,
            number_of_integer_variables=0,
            number_of_binary_variables=0,
            number_of_constraints=1,
            number_of_nonzero_coefficients=4,
        ),
        runtime_budget_seconds=20,
        fallback_used=False,
        reason="fixture route",
    )
    solver_run = SolverRun(
        solver_run_id=solver_run_id,
        project_id=model.project_id,
        problem_id=model.problem_id,
        model_id=model.model_id,
        model_version=model.version,
        model_digest=digest,
        generated_program_id=program.program_id,
        execution_origin=ExecutionOrigin.DETERMINISTIC_SOLVER_ADAPTER,
        routing_decision_id=route.routing_decision_id,
        routing_decision=route,
        solver=SolverName.SCIPY,
        solver_version="fixture",
        options=options,
        end_time=datetime.now(UTC),
        status=SolverStatus.OPTIMAL,
        objective=30,
        runtime_seconds=0.01,
        result_ref=result_id,
        execution_ref=execution_id,
        result=solver_result,
    )
    result = ResultRecord(
        result_id=result_id,
        project_id=model.project_id,
        problem_id=model.problem_id,
        model_id=model.model_id,
        model_version=model.version,
        model_digest=digest,
        solver_run_id=solver_run_id,
        execution_record_id=execution_id,
        solver=SolverName.SCIPY,
        objective=30,
        key_outputs={"x": 10, "y": 0},
        status=SolverStatus.OPTIMAL,
        evidence_refs=[
            f"model:{model.model_id}:v{model.version}",
            f"solver_run:{solver_run_id}",
            f"execution:{execution_id}",
        ],
    )
    return result, solver_run, execution, model, program


def test_evidence_integrity_accepts_valid_chain_and_numeric_tolerance() -> None:
    result, solver_run, execution, model, program = _evidence_chain()
    verifier = EvidenceIntegrityVerifier(absolute_tolerance=1e-8, relative_tolerance=0)

    valid = verifier.verify(
        result=result.model_copy(update={"objective": 30 + 1e-9}),
        solver_run=solver_run,
        execution=execution,
        model=model,
        program=program,
    )
    invalid = verifier.verify(
        result=result.model_copy(update={"objective": 30.01}),
        solver_run=solver_run,
        execution=execution,
        model=model,
        program=program,
    )

    assert valid.valid is True
    assert invalid.valid is False
    assert EvidenceErrorCode.OBJECTIVE_MISMATCH in invalid.error_codes


def test_model_digest_ignores_identity_and_prose_but_changes_with_math_semantics() -> None:
    model = lp_model()
    same_math = model.model_copy(
        update={
            "model_id": uuid4(),
            "version": 2,
            "name": "renamed fixture",
            "description": "different audit prose",
        }
    )
    assert model.objective is not None
    changed_objective = model.objective.model_copy(
        update={
            "expression": add(
                multiply(constant(5), symbol("x")),
                multiply(constant(4), symbol("y")),
            )
        }
    )
    changed_math = model.model_copy(update={"objective": changed_objective})

    assert mathematical_model_digest(same_math) == mathematical_model_digest(model)
    assert mathematical_model_digest(changed_math) != mathematical_model_digest(model)


def test_evidence_integrity_reports_missing_chain_nodes() -> None:
    result, _, _, _, _ = _evidence_chain()

    report = EvidenceIntegrityVerifier().verify(
        result=result,
        solver_run=None,
        execution=None,
        model=None,
        program=None,
    )

    assert report.valid is False
    assert set(report.error_codes) >= {
        EvidenceErrorCode.MISSING_SOLVER_RUN,
        EvidenceErrorCode.MISSING_EXECUTION,
        EvidenceErrorCode.MISSING_MODEL,
    }


def test_evidence_integrity_detects_solver_identity_mock_and_bundle_tampering() -> None:
    result, solver_run, execution, model, program = _evidence_chain()
    verifier = EvidenceIntegrityVerifier()

    wrong_solver = verifier.verify(
        result=result.model_copy(update={"solver": SolverName.GUROBI}),
        solver_run=solver_run,
        execution=execution,
        model=model,
        program=program,
    )
    mock_execution = verifier.verify(
        result=result,
        solver_run=solver_run,
        execution=execution.model_copy(
            update={"is_mock": True, "status": ExecutionStatus.FAILED, "exit_code": 1}
        ),
        model=model,
        program=program,
    )
    wrong_bundle = verifier.verify(
        result=result,
        solver_run=solver_run,
        execution=execution.model_copy(update={"executed_bundle_hash": "f" * 64}),
        model=model,
        program=program,
    )

    assert EvidenceErrorCode.SOLVER_IDENTITY_MISMATCH in wrong_solver.error_codes
    assert EvidenceErrorCode.MOCK_EXECUTION in mock_execution.error_codes
    assert EvidenceErrorCode.EXECUTION_FAILED in mock_execution.error_codes
    assert EvidenceErrorCode.CODE_HASH_MISMATCH in wrong_bundle.error_codes


def test_execution_strategy_is_deterministic_first_and_honors_explicit_generated() -> None:
    model = lp_model()
    plan = AlgorithmSelector().select(model)
    options = SolverOptions()
    resolved = SolverOptions(time_limit_seconds=12)
    router = Mock(spec=SolverRouter)
    routed = Mock(options=resolved)
    router.route.return_value = routed
    router.resolve_options.return_value = resolved
    selector = ExecutionStrategySelector(router)

    automatic = selector.select(
        requested=ExecutionStrategy.AUTO,
        model=model,
        plan=plan,
        options=options,
    )
    generated = selector.select(
        requested=ExecutionStrategy.GENERATED,
        model=model,
        plan=plan,
        options=options,
    )

    assert automatic.decision.selected is ExecutionStrategy.DETERMINISTIC
    assert automatic.routed_solver is routed
    assert generated.decision.selected is ExecutionStrategy.GENERATED
    assert generated.routed_solver is None
    assert generated.options.time_limit_seconds == 12


def test_execution_strategy_auto_falls_back_but_explicit_deterministic_fails() -> None:
    model = lp_model()
    plan = AlgorithmSelector().select(model)
    resolved = SolverOptions(time_limit_seconds=5)
    router = Mock(spec=SolverRouter)
    router.route.side_effect = SolverUnavailableError("SOLVER_UNAVAILABLE: fixture")
    router.resolve_options.return_value = resolved
    selector = ExecutionStrategySelector(router)

    automatic = selector.select(
        requested=ExecutionStrategy.AUTO,
        model=model,
        plan=plan,
        options=SolverOptions(),
    )

    assert automatic.decision.selected is ExecutionStrategy.GENERATED
    with pytest.raises(SolverUnavailableError, match="SOLVER_UNAVAILABLE"):
        selector.select(
            requested=ExecutionStrategy.DETERMINISTIC,
            model=model,
            plan=plan,
            options=SolverOptions(),
        )
