from __future__ import annotations

from mathmodel_ai.schemas.mathematical import MathematicalModel
from mathmodel_ai.schemas.solver import (
    SolverFamily,
    SolverHealth,
    SolverName,
    SolverOptions,
    SolverResult,
    SolverStatus,
    SupportAssessment,
)
from mathmodel_ai.solvers.base import BaseSolver, SolverExecution
from mathmodel_ai.solvers.feasibility import check_feasibility
from mathmodel_ai.solvers.programs import build_deterministic_program
from mathmodel_ai.verification.scalar_response import evaluate_scalar_responses


class ScalarResponseSolver(BaseSolver):
    """A fixed-point evaluator for experiments, not an optimization solver."""

    name = SolverName.SCALAR_RESPONSE
    families = frozenset({SolverFamily.SCALAR_RESPONSE})
    capabilities = frozenset()

    def assess_support(self, model: MathematicalModel) -> SupportAssessment:
        try:
            evaluate_scalar_responses(
                model, {item.symbol: 0.0 for item in model.decision_variables}
            )
        except ValueError as exc:
            return SupportAssessment(supported=False, reasons=[str(exc)])
        return SupportAssessment(supported=True)

    def health_check(self) -> SolverHealth:
        image = self._sandbox.image_id()
        return SolverHealth(
            solver_name=self.name,
            available=image is not None,
            reason="versioned deterministic runtime is available"
            if image
            else "solver image is unavailable",
        )

    def solve(self, model: MathematicalModel, options: SolverOptions) -> SolverExecution:
        fixed = options.initial_point
        evaluate_scalar_responses(model, fixed)
        program = build_deterministic_program(
            model,
            options,
            backend="scalar_response",
            target=SolverFamily.SCALAR_RESPONSE,
            dependencies=[],
        )
        execution, raw, program = self._run_program(program)
        if raw is None:
            return SolverExecution(
                result=self.execution_failure_result(execution),
                execution=execution,
                program=program,
            )
        feasibility = check_feasibility(
            model, raw.variables, tolerance=options.feasibility_tolerance
        )
        feasible = bool(
            feasibility.checked
            and not feasibility.bound_violations
            and not feasibility.violated_constraints
        )
        return SolverExecution(
            result=SolverResult(
                solver_name=self.name,
                solver_version=raw.solver_version,
                status=SolverStatus.FEASIBLE if feasible else SolverStatus.MODEL_INVALID,
                objective_value=None,
                variable_values=raw.variables,
                runtime_seconds=raw.runtime_seconds,
                message=raw.message,
                raw_status=str(raw.native_status),
                is_feasible=feasible,
                execution_record_id=execution.record.run_id,
                feasibility=feasibility,
            ),
            execution=execution,
            program=program,
        )
