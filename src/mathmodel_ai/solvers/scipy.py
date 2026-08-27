from __future__ import annotations

from mathmodel_ai.mathematical.expressions import (
    ExpressionError,
    linearize_expression,
    scalar_parameter_values,
)
from mathmodel_ai.schemas.mathematical import (
    ConvexityStatus,
    MathematicalModel,
    VariableDomain,
)
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
from mathmodel_ai.solvers.base import BaseSolver, RawSolverOutput, SolverExecution
from mathmodel_ai.solvers.feasibility import check_feasibility
from mathmodel_ai.solvers.programs import build_deterministic_program


def canonicalize_scipy_status(
    model: MathematicalModel,
    raw: RawSolverOutput,
) -> SolverStatus:
    if model.model_family is ModelFamily.NONLINEAR_PROGRAMMING:
        if raw.success:
            return (
                SolverStatus.OPTIMAL
                if model.algorithm_requirements.convexity is ConvexityStatus.CONVEX
                else SolverStatus.FEASIBLE
            )
        return {
            2: SolverStatus.MODEL_INVALID,
            4: SolverStatus.INFEASIBLE,
            9: SolverStatus.ITERATION_LIMIT,
        }.get(raw.native_status, SolverStatus.NUMERICAL_ERROR)
    if raw.native_status == 0:
        return SolverStatus.OPTIMAL
    if raw.native_status == 1:
        return (
            SolverStatus.TIME_LIMIT
            if "time limit" in raw.message.casefold()
            else SolverStatus.ITERATION_LIMIT
        )
    return {
        2: SolverStatus.INFEASIBLE,
        3: SolverStatus.UNBOUNDED,
        4: SolverStatus.NUMERICAL_ERROR,
    }.get(raw.native_status, SolverStatus.UNKNOWN)


class SciPySolver(BaseSolver):
    name = SolverName.SCIPY
    families = frozenset(
        {
            SolverFamily.SCIPY_HIGHS,
            SolverFamily.SCIPY_MILP,
            SolverFamily.SCIPY_MINIMIZE,
        }
    )
    capabilities = frozenset(
        {
            SolverCapability.LP,
            SolverCapability.MILP,
            SolverCapability.INTEGER,
            SolverCapability.NLP,
            SolverCapability.BINARY,
            SolverCapability.CONTINUOUS,
        }
    )

    def assess_support(self, model: MathematicalModel) -> SupportAssessment:
        reasons: list[str] = []
        if model.objective is None:
            reasons.append("an optimization objective is required")
        if any(item.index_sets for item in model.decision_variables):
            reasons.append("Phase 4 SciPy adapter requires flattened scalar variables")
        family = model.model_family
        supported_families = {
            ModelFamily.LINEAR_PROGRAMMING,
            ModelFamily.MIXED_INTEGER_LINEAR_PROGRAMMING,
            ModelFamily.INTEGER_PROGRAMMING,
            ModelFamily.NONLINEAR_PROGRAMMING,
        }
        if family not in supported_families:
            reasons.append(f"SciPy adapter does not support {family.value}")
        if family is ModelFamily.LINEAR_PROGRAMMING and any(
            item.domain not in {VariableDomain.CONTINUOUS, VariableDomain.NONNEGATIVE_CONTINUOUS}
            for item in model.decision_variables
        ):
            reasons.append("LP path cannot contain integer or binary domains")
        if family is ModelFamily.NONLINEAR_PROGRAMMING and any(
            item.domain not in {VariableDomain.CONTINUOUS, VariableDomain.NONNEGATIVE_CONTINUOUS}
            for item in model.decision_variables
        ):
            reasons.append("SciPy minimize path supports continuous variables only")
        if (
            family
            in {
                ModelFamily.LINEAR_PROGRAMMING,
                ModelFamily.MIXED_INTEGER_LINEAR_PROGRAMMING,
                ModelFamily.INTEGER_PROGRAMMING,
            }
            and model.objective is not None
        ):
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
                for expression in expressions:
                    linearize_expression(
                        expression,
                        variable_symbols=variables,
                        parameter_values=parameters,
                    )
            except ExpressionError as exc:
                reasons.append(str(exc))
        return SupportAssessment(supported=not reasons, reasons=reasons)

    def health_check(self) -> SolverHealth:
        image = self._sandbox.image_id()
        module = image is not None and self._sandbox.probe_python_module("scipy")
        return SolverHealth(
            solver_name=self.name,
            available=bool(image and module),
            reason=(
                "versioned solver image contains SciPy"
                if image and module
                else "solver image or SciPy module is unavailable"
            ),
        )

    def solve(self, model: MathematicalModel, options: SolverOptions) -> SolverExecution:
        assessment = self.assess_support(model)
        if not assessment.supported:
            raise ValueError(f"MODEL_INVALID: {'; '.join(assessment.reasons)}")
        target = {
            ModelFamily.LINEAR_PROGRAMMING: SolverFamily.SCIPY_HIGHS,
            ModelFamily.MIXED_INTEGER_LINEAR_PROGRAMMING: SolverFamily.SCIPY_MILP,
            ModelFamily.INTEGER_PROGRAMMING: SolverFamily.SCIPY_MILP,
            ModelFamily.NONLINEAR_PROGRAMMING: SolverFamily.SCIPY_MINIMIZE,
        }[model.model_family]
        program = build_deterministic_program(
            model,
            options,
            backend="scipy",
            target=target,
            dependencies=["numpy", "scipy"],
        )
        execution, raw, program = self._run_program(program)
        if raw is None:
            return SolverExecution(
                result=self.execution_failure_result(execution),
                execution=execution,
                program=program,
            )
        status = canonicalize_scipy_status(model, raw)
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
            raw_status=str(raw.native_status),
            is_optimal=status is SolverStatus.OPTIMAL,
            is_feasible=feasible,
            warnings=(
                ["successful nonlinear solve is a feasible local solution, not a global optimum"]
                if status is SolverStatus.FEASIBLE
                else []
            ),
            execution_record_id=execution.record.run_id,
            feasibility=feasibility,
        )
        return SolverExecution(result=result, execution=execution, program=program)
