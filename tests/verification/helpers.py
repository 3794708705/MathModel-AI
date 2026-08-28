from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

from mathmodel_ai.mathematical.algorithms import AlgorithmSelector
from mathmodel_ai.mathematical.digests import mathematical_model_digest
from mathmodel_ai.mathematical.evidence import EvidenceIntegrityVerifier
from mathmodel_ai.sandbox.executor import SandboxExecution
from mathmodel_ai.schemas.execution import (
    ExecutionOrigin,
    ExecutionRecord,
    ExecutionStatus,
    SandboxLimits,
)
from mathmodel_ai.schemas.mathematical import ParameterDefinition, ParameterSourceType
from mathmodel_ai.schemas.program import GeneratedProgramStatus
from mathmodel_ai.schemas.results import EvidenceChainReport, ResultRecord
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
from mathmodel_ai.solvers.base import SolverExecution
from mathmodel_ai.solvers.programs import build_deterministic_program
from mathmodel_ai.verification.experiments import ExperimentEngine
from mathmodel_ai.verification.validation import IndependentValidator
from tests.mathematical.helpers import lp_model, symbol


def phase5_model():
    model = lp_model()
    parameter = ParameterDefinition(
        parameter_id="PAR-demand",
        symbol="demand",
        description="minimum required allocation",
        value=10.0,
        source_type=ParameterSourceType.PROBLEM_FACT,
        source_ref="EVID-fact-1",
        confidence=1,
    )
    constraint = model.constraints[0].model_copy(update={"rhs": symbol("demand")})
    equations = [
        item.model_copy(update={"rhs": symbol("demand"), "parameter_refs": ["demand"]})
        if item.equation_id == constraint.equation_ref
        else item
        for item in model.equations
    ]
    return model.model_copy(
        update={
            "parameters": [parameter],
            "constraints": [constraint],
            "equations": equations,
            "validation_requirements": [
                "recompute variable bounds",
                "recompute every constraint",
                "recalculate objective metric",
                "verify evidence trace",
            ],
        }
    )


def result_bundle():
    model = phase5_model()
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
            number_of_nonzero_coefficients=3,
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
    evidence = EvidenceIntegrityVerifier().verify(
        result=result,
        solver_run=solver_run,
        execution=execution,
        model=model,
        program=program,
    )
    assert evidence.valid
    return model, result, solver_run, execution, program, evidence


class ScenarioSolver:
    def solve(self, model, options: SolverOptions) -> SolverExecution:
        del options
        demand = float(model.parameters[0].value)
        objective = 3 * demand
        program = build_deterministic_program(
            model,
            SolverOptions(),
            backend="scipy",
            target=SolverFamily.SCIPY_HIGHS,
            dependencies=["scipy"],
        ).model_copy(update={"status": GeneratedProgramStatus.EXECUTED})
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
            model_digest=mathematical_model_digest(model),
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
        result = SolverResult(
            solver_name=SolverName.SCIPY,
            status=SolverStatus.OPTIMAL,
            objective_value=objective,
            variable_values={"x": demand, "y": 0},
            runtime_seconds=0.01,
            is_optimal=True,
            is_feasible=True,
            execution_record_id=execution_id,
            feasibility=FeasibilityReport(
                checked=True,
                max_constraint_violation=0,
                tolerance=1e-7,
                message="scenario is feasible",
            ),
        )
        return SolverExecution(
            result=result,
            execution=SandboxExecution(record=execution, artifact_records=[]),
            program=program,
        )


class ScenarioRouter:
    def __init__(self) -> None:
        self.solver = ScenarioSolver()

    def route(self, model, plan, options: SolverOptions):
        del model, plan
        return SimpleNamespace(solver=self.solver, options=options)


def experiment_engine() -> ExperimentEngine:
    return ExperimentEngine(
        algorithm_selector=AlgorithmSelector(),
        solver_router=ScenarioRouter(),  # type: ignore[arg-type]
        validator=IndependentValidator(),
    )


def valid_report() -> tuple[object, object, object, EvidenceChainReport]:
    model, result, solver_run, _, _, evidence = result_bundle()
    report = IndependentValidator().validate(
        model=model,
        result=result,
        solver_run=solver_run,
        evidence=evidence,
    )
    return model, result, report, evidence
