from __future__ import annotations

import math
from dataclasses import dataclass

from mathmodel_ai.mathematical.algorithms import AlgorithmSelector
from mathmodel_ai.mathematical.digests import mathematical_model_digest
from mathmodel_ai.schemas.execution import ExecutionStatus
from mathmodel_ai.schemas.mathematical import MathematicalModel, ParameterDefinition
from mathmodel_ai.schemas.solver import SolverOptions, SolverStatus
from mathmodel_ai.schemas.verification import (
    ExperimentRun,
    ExperimentStatus,
    ParameterPerturbation,
)
from mathmodel_ai.solvers.base import SolverExecution
from mathmodel_ai.solvers.router import SolverRouter
from mathmodel_ai.solvers.scalar_response import ScalarResponseSolver
from mathmodel_ai.verification.experiment_integrity import ExperimentIntegrityVerifier
from mathmodel_ai.verification.scalar_response import evaluate_scalar_responses
from mathmodel_ai.verification.validation import IndependentValidator


@dataclass(frozen=True)
class ExperimentOutcome:
    record: ExperimentRun
    execution: SolverExecution | None


class ExperimentEngine:
    def __init__(
        self,
        *,
        algorithm_selector: AlgorithmSelector,
        solver_router: SolverRouter,
        validator: IndependentValidator,
        integrity_verifier: ExperimentIntegrityVerifier | None = None,
        response_solver: ScalarResponseSolver | None = None,
    ) -> None:
        self._algorithm_selector = algorithm_selector
        self._solver_router = solver_router
        self._validator = validator
        self._integrity_verifier = integrity_verifier or ExperimentIntegrityVerifier(validator)
        self._response_solver = response_solver

    def execute_response(
        self,
        *,
        model: MathematicalModel,
        perturbations: list[ParameterPerturbation],
        options: SolverOptions,
        experiment_type: str,
        fixed_decision_values: dict[str, float],
    ) -> ExperimentOutcome:
        scenario = self.perturb_model(model, perturbations)
        base_digest = mathematical_model_digest(model)
        scenario_digest = mathematical_model_digest(scenario)
        try:
            if self._response_solver is None:
                raise ValueError("scalar response runtime is not configured")
            expected = evaluate_scalar_responses(scenario, fixed_decision_values)
            execution = self._response_solver.solve(
                scenario,
                options.model_copy(update={"initial_point": fixed_decision_values}),
            )
        except Exception as exc:
            return ExperimentOutcome(
                record=ExperimentRun(
                    experiment_type=experiment_type,
                    base_model_digest=base_digest,
                    scenario_model_digest=scenario_digest,
                    perturbations=perturbations,
                    fixed_decision_values=fixed_decision_values,
                    status=ExperimentStatus.FAIL,
                    error=f"{type(exc).__name__}: {exc}",
                ),
                execution=None,
            )
        result = execution.result
        feasible, max_violation, failed_checks = self._validator.candidate_is_feasible(
            scenario, result.variable_values
        )
        outputs_match = all(
            math.isclose(
                result.variable_values.get(symbol, float("nan")),
                value,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            for symbol, value in expected.items()
        )
        execution_valid = (
            execution.execution.record.status is ExecutionStatus.SUCCEEDED
            and execution.execution.record.exit_code == 0
            and not execution.execution.record.is_mock
        )
        errors = []
        if not execution_valid:
            errors.append("response execution did not prove non-Mock success")
        if not feasible or not result.is_feasible:
            errors.append("independent scenario checks failed: " + ", ".join(failed_checks))
        if not outputs_match:
            errors.append("sandbox responses disagree with independent model equations")
        provisional = ExperimentRun(
            experiment_type=experiment_type,
            base_model_digest=base_digest,
            scenario_model_digest=scenario_digest,
            perturbations=perturbations,
            solver=result.solver_name,
            solver_status=result.status,
            execution_record_id=execution.execution.record.run_id,
            key_outputs=result.variable_values,
            fixed_decision_values=fixed_decision_values,
            feasible=feasible and result.is_feasible,
            max_constraint_violation=max_violation,
            status=ExperimentStatus.FAIL,
            error="; ".join(errors) or None,
        )
        integrity = self._integrity_verifier.audit(
            base_model=model, record=provisional, execution=execution
        )
        errors.extend(integrity.errors)
        passed = (
            execution_valid
            and feasible
            and result.is_feasible
            and outputs_match
            and integrity.valid
        )
        return ExperimentOutcome(
            record=ExperimentRun(
                **provisional.model_dump(exclude={"experiment_id", "status", "error"}),
                status=ExperimentStatus.PASS if passed else ExperimentStatus.FAIL,
                error="; ".join(errors) or None,
            ),
            execution=execution,
        )

    def execute(
        self,
        *,
        model: MathematicalModel,
        perturbations: list[ParameterPerturbation],
        options: SolverOptions,
        experiment_type: str,
        baseline_objective: float,
    ) -> ExperimentOutcome:
        scenario = self.perturb_model(model, perturbations)
        base_digest = mathematical_model_digest(model)
        scenario_digest = mathematical_model_digest(scenario)
        try:
            plan = self._algorithm_selector.select(scenario)
            routed = self._solver_router.route(scenario, plan, options)
            execution = routed.solver.solve(scenario, routed.options)
        except Exception as exc:
            return ExperimentOutcome(
                record=ExperimentRun(
                    experiment_type=experiment_type,
                    base_model_digest=base_digest,
                    scenario_model_digest=scenario_digest,
                    perturbations=perturbations,
                    status=ExperimentStatus.FAIL,
                    error=f"{type(exc).__name__}: {exc}",
                ),
                execution=None,
            )

        solver_result = execution.result
        feasible, max_violation, failed_checks = self._validator.candidate_is_feasible(
            scenario,
            solver_result.variable_values,
        )
        execution_valid = (
            execution.execution.record.status is ExecutionStatus.SUCCEEDED
            and execution.execution.record.exit_code == 0
            and not execution.execution.record.is_mock
        )
        terminal_feasible = solver_result.status in {
            SolverStatus.OPTIMAL,
            SolverStatus.FEASIBLE,
        }
        computationally_passed = (
            execution_valid
            and terminal_feasible
            and solver_result.is_feasible
            and feasible
            and solver_result.objective_value is not None
        )
        objective = solver_result.objective_value
        change = objective - baseline_objective if objective is not None else None
        relative = (
            change / abs(baseline_objective)
            if change is not None and baseline_objective != 0
            else None
        )
        errors: list[str] = []
        if not execution_valid:
            errors.append("experiment execution did not prove non-Mock success")
        if not terminal_feasible or not solver_result.is_feasible:
            errors.append(f"experiment solver status is {solver_result.status.value}")
        if not feasible:
            errors.append("independent scenario checks failed: " + ", ".join(failed_checks))
        if objective is None:
            errors.append("experiment objective is missing")
        warnings = list(solver_result.warnings)
        if baseline_objective == 0:
            warnings.append("baseline objective is zero; relative objective change is undefined")
        provisional = ExperimentRun(
            experiment_type=experiment_type,
            base_model_digest=base_digest,
            scenario_model_digest=scenario_digest,
            perturbations=perturbations,
            solver=solver_result.solver_name,
            solver_status=solver_result.status,
            execution_record_id=execution.execution.record.run_id,
            objective_value=objective,
            objective_change=change,
            objective_change_fraction=relative,
            key_outputs=dict(sorted(solver_result.variable_values.items())[:100]),
            feasible=feasible and solver_result.is_feasible,
            max_constraint_violation=max_violation,
            status=ExperimentStatus.FAIL,
            error="; ".join(errors) or None,
            warnings=warnings,
        )
        integrity = self._integrity_verifier.audit(
            base_model=model,
            record=provisional,
            execution=execution,
        )
        errors.extend(integrity.errors)
        passed = computationally_passed and integrity.valid
        return ExperimentOutcome(
            record=ExperimentRun(
                **provisional.model_dump(
                    exclude={"experiment_id", "status", "error"},
                ),
                status=ExperimentStatus.PASS if passed else ExperimentStatus.FAIL,
                error="; ".join(errors) or None,
            ),
            execution=execution,
        )

    @staticmethod
    def perturb_model(
        model: MathematicalModel,
        perturbations: list[ParameterPerturbation],
    ) -> MathematicalModel:
        updates = {item.parameter_id: item.perturbed_value for item in perturbations}
        known = {item.parameter_id for item in model.parameters}
        missing = set(updates) - known
        if missing:
            raise ValueError("unknown perturbation parameters: " + ", ".join(sorted(missing)))
        parameters = [
            item.model_copy(update={"value": updates[item.parameter_id]})
            if item.parameter_id in updates
            else item
            for item in model.parameters
        ]
        return model.model_copy(update={"parameters": parameters})

    @staticmethod
    def scalar_parameters(model: MathematicalModel) -> list[tuple[ParameterDefinition, float]]:
        result: list[tuple[ParameterDefinition, float]] = []
        for parameter in model.parameters:
            value = parameter.value
            if isinstance(value, bool):
                continue
            if isinstance(value, int | float) and math.isfinite(float(value)):
                result.append((parameter, float(value)))
        return result

    @staticmethod
    def select_scalar_parameters(
        model: MathematicalModel,
        requested: list[str],
        limit: int,
    ) -> list[tuple[ParameterDefinition, float]]:
        eligible = ExperimentEngine.scalar_parameters(model)
        if requested:
            requested_set = set(requested)
            selected = [item for item in eligible if item[0].symbol in requested_set]
            missing = requested_set - {item[0].symbol for item in selected}
            if missing:
                raise ValueError(
                    "requested scalar parameters are unavailable: " + ", ".join(sorted(missing))
                )
        else:
            selected = eligible
        return sorted(selected, key=lambda item: item[0].parameter_id)[:limit]

    @staticmethod
    def perturbation(
        parameter: ParameterDefinition,
        baseline: float,
        fraction: float,
        *,
        value: float | None = None,
    ) -> ParameterPerturbation:
        return ParameterPerturbation(
            parameter_id=parameter.parameter_id,
            symbol=parameter.symbol,
            source_ref=parameter.source_ref,
            baseline_value=baseline,
            fraction=fraction,
            perturbed_value=baseline * (1 + fraction) if value is None else value,
        )
