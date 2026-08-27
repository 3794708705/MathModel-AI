from __future__ import annotations

from pathlib import Path

from mathmodel_ai.core.errors import SolverUnavailableError
from mathmodel_ai.files.storage import FileStore
from mathmodel_ai.mathematical.expressions import (
    ExpressionError,
    linearize_expression,
    scalar_parameter_values,
)
from mathmodel_ai.sandbox.executor import SandboxExecutor
from mathmodel_ai.schemas.mathematical import MathematicalModel
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


def canonicalize_gurobi_status(native_status: int) -> SolverStatus:
    return {
        2: SolverStatus.OPTIMAL,
        3: SolverStatus.INFEASIBLE,
        4: SolverStatus.INFEASIBLE_OR_UNBOUNDED,
        5: SolverStatus.UNBOUNDED,
        7: SolverStatus.ITERATION_LIMIT,
        9: SolverStatus.TIME_LIMIT,
        12: SolverStatus.NUMERICAL_ERROR,
        13: SolverStatus.FEASIBLE,
    }.get(native_status, SolverStatus.UNKNOWN)


class GurobiSolver(BaseSolver):
    name = SolverName.GUROBI
    families = frozenset({SolverFamily.GUROBI})
    capabilities = frozenset(
        {
            SolverCapability.LP,
            SolverCapability.MILP,
            SolverCapability.INTEGER,
            SolverCapability.BINARY,
            SolverCapability.CONTINUOUS,
        }
    )

    def __init__(
        self,
        *,
        sandbox: SandboxExecutor,
        store: FileStore,
        license_file: Path | None = None,
    ) -> None:
        super().__init__(sandbox=sandbox, store=store)
        self._license_file = license_file

    def assess_support(self, model: MathematicalModel) -> SupportAssessment:
        reasons: list[str] = []
        if model.model_family not in {
            ModelFamily.LINEAR_PROGRAMMING,
            ModelFamily.MIXED_INTEGER_LINEAR_PROGRAMMING,
            ModelFamily.INTEGER_PROGRAMMING,
        }:
            reasons.append("Phase 4 Gurobi adapter supports LP and linear integer models")
        if model.objective is None:
            reasons.append("Gurobi adapter requires an objective")
        if any(item.index_sets for item in model.decision_variables):
            reasons.append("Phase 4 Gurobi adapter requires flattened scalar variables")
        if model.objective is not None:
            variables = {item.symbol for item in model.decision_variables}
            parameters = scalar_parameter_values([*model.parameters, *model.constants])
            try:
                for expression in [
                    model.objective.expression,
                    *(
                        expression
                        for constraint in model.constraints
                        for expression in (constraint.expression, constraint.rhs)
                    ),
                ]:
                    linearize_expression(
                        expression,
                        variable_symbols=variables,
                        parameter_values=parameters,
                    )
            except ExpressionError as exc:
                reasons.append(str(exc))
        return SupportAssessment(supported=not reasons, reasons=reasons)

    def health_check(self) -> SolverHealth:
        license_exists = self._license_file is not None and self._license_file.is_file()
        image = self._sandbox.image_id()
        module = image is not None and self._sandbox.probe_python_module("gurobipy")
        available = bool(license_exists and image and module)
        return SolverHealth(
            solver_name=self.name,
            available=available,
            reason=(
                "Gurobi image, module, and runtime license file are available"
                if available
                else "Gurobi module/image/license unavailable; use a configured fallback"
            ),
        )

    def solve(self, model: MathematicalModel, options: SolverOptions) -> SolverExecution:
        assessment = self.assess_support(model)
        if not assessment.supported:
            raise ValueError(f"MODEL_INVALID: {'; '.join(assessment.reasons)}")
        if not self.health_check().available:
            raise SolverUnavailableError(
                "SOLVER_UNAVAILABLE: Gurobi runtime or license is unavailable"
            )
        program = build_deterministic_program(
            model,
            options,
            backend="gurobi",
            target=SolverFamily.GUROBI,
            dependencies=["gurobipy"],
        )
        execution, raw, program = self._run_program(
            program,
            secret_mounts=self.license_mount(self._license_file),
        )
        if raw is None:
            return SolverExecution(
                result=self.execution_failure_result(execution),
                execution=execution,
                program=program,
            )
        status = canonicalize_gurobi_status(raw.native_status)
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
            execution_record_id=execution.record.run_id,
            feasibility=feasibility,
        )
        return SolverExecution(result=result, execution=execution, program=program)
