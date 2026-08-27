from __future__ import annotations

from mathmodel_ai.mathematical.expressions import (
    ExpressionError,
    linearize_expression,
    scalar_parameter_values,
)
from mathmodel_ai.schemas.mathematical import MathematicalModel, VariableDomain
from mathmodel_ai.schemas.model_selection import ModelFamily
from mathmodel_ai.schemas.solver import (
    SolverCapability,
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


def _is_integer(value: float, tolerance: float = 1e-9) -> bool:
    return abs(value - round(value)) <= tolerance


class ORToolsSolver(BaseSolver):
    name = SolverName.ORTOOLS
    families = frozenset({SolverFamily.ORTOOLS_CP_SAT})
    capabilities = frozenset(
        {
            SolverCapability.MILP,
            SolverCapability.INTEGER,
            SolverCapability.BINARY,
        }
    )

    def assess_support(self, model: MathematicalModel) -> SupportAssessment:
        reasons: list[str] = []
        if model.model_family not in {
            ModelFamily.INTEGER_PROGRAMMING,
            ModelFamily.MIXED_INTEGER_LINEAR_PROGRAMMING,
        }:
            reasons.append("CP-SAT accepts integer or all-integer MILP models only")
        if model.objective is None:
            reasons.append("CP-SAT requires an objective in the Phase 4 adapter")
        if any(item.index_sets for item in model.decision_variables):
            reasons.append("Phase 4 CP-SAT adapter requires flattened scalar variables")
        if any(
            item.domain
            not in {
                VariableDomain.INTEGER,
                VariableDomain.NONNEGATIVE_INTEGER,
                VariableDomain.BINARY,
            }
            for item in model.decision_variables
        ):
            reasons.append("CP-SAT cannot consume continuous decision variables")
        if any(
            item.lower_bound is None
            or item.upper_bound is None
            or not _is_integer(item.lower_bound)
            or not _is_integer(item.upper_bound)
            for item in model.decision_variables
        ):
            reasons.append("CP-SAT requires explicit finite integral bounds")
        if model.objective is not None:
            variables = {item.symbol for item in model.decision_variables}
            parameters = scalar_parameter_values([*model.parameters, *model.constants])
            expressions = [
                model.objective.expression,
                *(
                    expression
                    for constraint in model.constraints
                    for expression in (constraint.expression, constraint.rhs)
                ),
            ]
            try:
                forms = [
                    linearize_expression(
                        expression,
                        variable_symbols=variables,
                        parameter_values=parameters,
                    )
                    for expression in expressions
                ]
                if any(
                    not _is_integer(form.constant)
                    or any(not _is_integer(value) for value in form.coefficients.values())
                    for form in forms
                ):
                    reasons.append(
                        "CP-SAT coefficients are non-integral and no explicit scaling plan exists"
                    )
            except ExpressionError as exc:
                reasons.append(str(exc))
        return SupportAssessment(supported=not reasons, reasons=reasons)

    def health_check(self) -> SolverHealth:
        image = self._sandbox.image_id()
        module = image is not None and self._sandbox.probe_python_module("ortools")
        return SolverHealth(
            solver_name=self.name,
            available=bool(image and module),
            reason=(
                "versioned solver image contains OR-Tools"
                if image and module
                else "solver image or OR-Tools module is unavailable"
            ),
        )

    def solve(self, model: MathematicalModel, options: SolverOptions) -> SolverExecution:
        assessment = self.assess_support(model)
        if not assessment.supported:
            raise ValueError(f"MODEL_INVALID: {'; '.join(assessment.reasons)}")
        program = build_deterministic_program(
            model,
            options,
            backend="ortools",
            target=SolverFamily.ORTOOLS_CP_SAT,
            dependencies=["ortools"],
        )
        execution, raw, program = self._run_program(program)
        if raw is None:
            return SolverExecution(
                result=self.execution_failure_result(execution),
                execution=execution,
                program=program,
            )
        status_name = (raw.native_status_name or "UNKNOWN").upper()
        status = {
            "OPTIMAL": SolverStatus.OPTIMAL,
            "FEASIBLE": SolverStatus.FEASIBLE,
            "INFEASIBLE": SolverStatus.INFEASIBLE,
            "MODEL_INVALID": SolverStatus.MODEL_INVALID,
        }.get(status_name)
        if status is None:
            status = (
                SolverStatus.TIME_LIMIT
                if options.time_limit_seconds is not None
                else SolverStatus.UNKNOWN
            )
        feasibility = check_feasibility(
            model,
            raw.variables,
            tolerance=options.feasibility_tolerance,
        )
        feasible = bool(
            raw.variables
            and feasibility.checked
            and not feasibility.violated_constraints
            and not feasibility.bound_violations
        )
        result = SolverResult(
            solver_name=self.name,
            solver_version=raw.solver_version,
            status=status,
            objective_value=raw.objective,
            variable_values=raw.variables,
            runtime_seconds=raw.runtime_seconds,
            iterations=raw.iterations,
            nodes=raw.nodes,
            mip_gap=raw.mip_gap,
            message=raw.message,
            raw_status=status_name,
            is_optimal=status is SolverStatus.OPTIMAL,
            is_feasible=feasible,
            execution_record_id=execution.record.run_id,
            feasibility=feasibility,
        )
        return SolverExecution(result=result, execution=execution, program=program)
