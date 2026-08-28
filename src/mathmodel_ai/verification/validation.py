from __future__ import annotations

import math

from mathmodel_ai.mathematical.digests import mathematical_model_digest
from mathmodel_ai.schemas.mathematical import (
    ConstraintRelation,
    MathematicalModel,
    ParameterDefinition,
    UnitCheckStatus,
    VariableDefinition,
    VariableDomain,
)
from mathmodel_ai.schemas.results import EvidenceChainReport, ResultRecord
from mathmodel_ai.schemas.solver import SolverRun
from mathmodel_ai.schemas.verification import (
    ConstraintValidationCheck,
    MetricRecalculation,
    ValidationCheckCategory,
    ValidationCheckStatus,
    ValidationReport,
    ValidationRequirementCheck,
    ValidationStatus,
    VariableValidationCheck,
)
from mathmodel_ai.verification.evaluator import (
    IndependentEvaluationError,
    IndependentExpressionEvaluator,
)


class IndependentValidator:
    version = "independent-validator-5.0.0"

    def __init__(
        self,
        *,
        absolute_tolerance: float = 1e-7,
        relative_tolerance: float = 1e-7,
    ) -> None:
        if absolute_tolerance <= 0 or relative_tolerance < 0:
            raise ValueError("validation tolerances must be positive/non-negative")
        self._absolute_tolerance = absolute_tolerance
        self._relative_tolerance = relative_tolerance
        self._evaluator = IndependentExpressionEvaluator()

    def validate(
        self,
        *,
        model: MathematicalModel,
        result: ResultRecord,
        solver_run: SolverRun,
        evidence: EvidenceChainReport,
    ) -> ValidationReport:
        errors: list[str] = []
        warnings: list[str] = []
        digest = mathematical_model_digest(model)
        if digest != result.model_digest or digest != solver_run.model_digest:
            errors.append("VALIDATION_FAIL:MODEL_DIGEST_MISMATCH")
        if not evidence.valid:
            errors.append("VALIDATION_FAIL:EVIDENCE_INVALID")
            errors.extend(f"VALIDATION_FAIL:EVIDENCE:{item.code.value}" for item in evidence.errors)

        values = self._numeric_environment(model, solver_run.result.variable_values)
        variable_checks = [
            self.check_variable(item, solver_run.result.variable_values)
            for item in model.decision_variables
        ]
        constraint_checks = [
            self.check_constraint(item, values)
            for item in [
                *model.constraints,
                *model.initial_conditions,
                *model.boundary_conditions,
            ]
        ]
        metrics = self._metric_checks(model, result, solver_run, values)
        requirement_checks = self._requirement_checks(
            model,
            evidence=evidence,
            variable_checks=variable_checks,
            constraint_checks=constraint_checks,
            metrics=metrics,
        )

        failures = [
            *(
                item.message
                for item in variable_checks
                if item.status is ValidationCheckStatus.FAIL
            ),
            *(
                item.message
                for item in constraint_checks
                if item.status is ValidationCheckStatus.FAIL
            ),
            *(item.message for item in metrics if item.status is ValidationCheckStatus.FAIL),
            *(
                item.message
                for item in requirement_checks
                if item.status is ValidationCheckStatus.FAIL
            ),
        ]
        unchecked = [
            *(
                item.message
                for item in variable_checks
                if item.status is ValidationCheckStatus.UNCHECKED
            ),
            *(
                item.message
                for item in constraint_checks
                if item.status is ValidationCheckStatus.UNCHECKED
            ),
            *(item.message for item in metrics if item.status is ValidationCheckStatus.UNCHECKED),
            *(
                item.message
                for item in requirement_checks
                if item.status is ValidationCheckStatus.UNCHECKED
            ),
        ]
        errors.extend(f"VALIDATION_FAIL:{message}" for message in failures)
        warnings.extend(f"VALIDATION_UNCHECKED:{message}" for message in unchecked)
        if any(
            equation.dimension_status is UnitCheckStatus.UNKNOWN for equation in model.equations
        ):
            warnings.append("VALIDATION_WARNING:UNIT_STATUS_UNKNOWN")

        status = (
            ValidationStatus.FAIL
            if errors
            else ValidationStatus.NOT_EVALUABLE
            if unchecked
            else ValidationStatus.PASS
        )
        violations = [item.violation for item in constraint_checks if item.violation is not None]
        return ValidationReport(
            project_id=model.project_id,
            problem_id=model.problem_id,
            model_id=model.model_id,
            model_version=model.version,
            model_digest=digest,
            result_id=result.result_id,
            solver_run_id=solver_run.solver_run_id,
            execution_record_id=solver_run.execution_ref,
            validator_version=self.version,
            status=status,
            evidence=evidence,
            variable_checks=variable_checks,
            constraint_checks=constraint_checks,
            metric_recalculations=metrics,
            requirement_checks=requirement_checks,
            max_constraint_violation=max(violations, default=None),
            errors=list(dict.fromkeys(errors)),
            warnings=list(dict.fromkeys(warnings)),
            evidence_refs=[
                f"model:{model.model_id}:v{model.version}",
                f"result:{result.result_id}",
                f"solver_run:{solver_run.solver_run_id}",
                f"execution:{solver_run.execution_ref}",
            ],
        )

    def audit_report(
        self,
        *,
        report: ValidationReport,
        model: MathematicalModel,
        result: ResultRecord,
        solver_run: SolverRun,
        evidence: EvidenceChainReport,
    ) -> list[str]:
        """Recompute a persisted report and compare every deterministic field."""
        recomputed = self.validate(
            model=model,
            result=result,
            solver_run=solver_run,
            evidence=evidence,
        )
        deterministic_fields = (
            "project_id",
            "problem_id",
            "model_id",
            "model_version",
            "model_digest",
            "result_id",
            "solver_run_id",
            "execution_record_id",
            "validator_version",
            "status",
            "evidence",
            "variable_checks",
            "constraint_checks",
            "metric_recalculations",
            "requirement_checks",
            "max_constraint_violation",
            "errors",
            "warnings",
            "evidence_refs",
        )
        errors = [
            f"VALIDATION_REPORT_MISMATCH:{field}"
            for field in deterministic_fields
            if getattr(report, field) != getattr(recomputed, field)
        ]
        return list(dict.fromkeys(errors))

    def candidate_is_feasible(
        self,
        model: MathematicalModel,
        variable_values: dict[str, float],
    ) -> tuple[bool, float | None, list[str]]:
        values = self._numeric_environment(model, variable_values)
        variable_checks = [
            self.check_variable(item, variable_values) for item in model.decision_variables
        ]
        constraint_checks = [
            self.check_constraint(item, values)
            for item in [
                *model.constraints,
                *model.initial_conditions,
                *model.boundary_conditions,
            ]
        ]
        failed = [
            *(
                item.variable_id
                for item in variable_checks
                if item.status is not ValidationCheckStatus.PASS
            ),
            *(
                item.constraint_id
                for item in constraint_checks
                if item.status is not ValidationCheckStatus.PASS
            ),
        ]
        violations = [item.violation for item in constraint_checks if item.violation is not None]
        return not failed, max(violations, default=None), failed

    def check_variable(
        self,
        variable: VariableDefinition,
        variable_values: dict[str, float],
    ) -> VariableValidationCheck:
        value = variable_values.get(variable.symbol)
        if value is None or not math.isfinite(value):
            return VariableValidationCheck(
                variable_id=variable.variable_id,
                symbol=variable.symbol,
                status=ValidationCheckStatus.FAIL,
                message=f"{variable.variable_id} has no finite solver value",
            )
        lower = variable.lower_bound
        upper = variable.upper_bound
        if variable.domain in {
            VariableDomain.NONNEGATIVE_CONTINUOUS,
            VariableDomain.NONNEGATIVE_INTEGER,
            VariableDomain.BINARY,
        }:
            lower = max(lower if lower is not None else 0.0, 0.0)
        if variable.domain is VariableDomain.BINARY:
            upper = min(upper if upper is not None else 1.0, 1.0)
        bound_violation = max(
            lower - value if lower is not None else 0.0,
            value - upper if upper is not None else 0.0,
            0.0,
        )
        integral_domains = {
            VariableDomain.INTEGER,
            VariableDomain.NONNEGATIVE_INTEGER,
            VariableDomain.BINARY,
        }
        integrality = abs(value - round(value)) if variable.domain in integral_domains else 0.0
        passed = max(bound_violation, integrality) <= self._absolute_tolerance
        return VariableValidationCheck(
            variable_id=variable.variable_id,
            symbol=variable.symbol,
            value=value,
            lower_bound=lower,
            upper_bound=upper,
            bound_violation=bound_violation,
            integrality_violation=integrality,
            status=ValidationCheckStatus.PASS if passed else ValidationCheckStatus.FAIL,
            message=(
                f"{variable.variable_id} independently satisfies domain and bounds"
                if passed
                else f"{variable.variable_id} independently violates domain or bounds"
            ),
        )

    def check_constraint(
        self,
        constraint: object,
        values: dict[str, float],
    ) -> ConstraintValidationCheck:
        from mathmodel_ai.schemas.mathematical import ConstraintDefinition

        typed = ConstraintDefinition.model_validate(constraint)
        try:
            left = self._evaluator.evaluate(typed.expression, values)
            right = self._evaluator.evaluate(typed.rhs, values)
        except IndependentEvaluationError as exc:
            return ConstraintValidationCheck(
                constraint_id=typed.constraint_id,
                relation=typed.relation.value,
                tolerance=self._absolute_tolerance,
                status=ValidationCheckStatus.UNCHECKED,
                message=f"{typed.constraint_id} could not be independently evaluated: {exc}",
            )
        if typed.relation is ConstraintRelation.LE:
            violation = max(left - right, 0.0)
        elif typed.relation is ConstraintRelation.GE:
            violation = max(right - left, 0.0)
        else:
            violation = abs(left - right)
        passed = violation <= self._absolute_tolerance
        return ConstraintValidationCheck(
            constraint_id=typed.constraint_id,
            relation=typed.relation.value,
            left_value=left,
            right_value=right,
            violation=violation,
            tolerance=self._absolute_tolerance,
            status=ValidationCheckStatus.PASS if passed else ValidationCheckStatus.FAIL,
            message=(
                f"{typed.constraint_id} independently satisfies tolerance"
                if passed
                else f"{typed.constraint_id} independently violates tolerance"
            ),
        )

    def _metric_checks(
        self,
        model: MathematicalModel,
        result: ResultRecord,
        solver_run: SolverRun,
        values: dict[str, float],
    ) -> list[MetricRecalculation]:
        if model.objective is None:
            return []
        try:
            recomputed = self._evaluator.evaluate(model.objective.expression, values)
        except IndependentEvaluationError as exc:
            return [
                MetricRecalculation(
                    metric_id=model.objective.objective_id,
                    category=ValidationCheckCategory.OBJECTIVE,
                    reported_value=result.objective,
                    absolute_tolerance=self._absolute_tolerance,
                    relative_tolerance=self._relative_tolerance,
                    status=ValidationCheckStatus.UNCHECKED,
                    message=f"objective could not be independently recalculated: {exc}",
                )
            ]
        checks = [
            self._numeric_check(
                metric_id=model.objective.objective_id,
                reported=result.objective,
                recomputed=recomputed,
                label="result objective",
            ),
            self._numeric_check(
                metric_id=f"{model.objective.objective_id}:solver_run",
                reported=solver_run.result.objective_value,
                recomputed=recomputed,
                label="solver-run objective",
            ),
        ]
        for key, reported in sorted(result.key_outputs.items()):
            checks.append(
                self._numeric_check(
                    metric_id=f"KEY_OUTPUT:{key}",
                    reported=reported,
                    recomputed=values.get(key),
                    label=f"key output {key}",
                    category=ValidationCheckCategory.OUTPUT,
                )
            )
        return checks

    def _numeric_check(
        self,
        *,
        metric_id: str,
        reported: float | None,
        recomputed: float | None,
        label: str,
        category: ValidationCheckCategory = ValidationCheckCategory.OBJECTIVE,
    ) -> MetricRecalculation:
        if reported is None or recomputed is None:
            return MetricRecalculation(
                metric_id=metric_id,
                category=category,
                reported_value=reported,
                recomputed_value=recomputed,
                absolute_tolerance=self._absolute_tolerance,
                relative_tolerance=self._relative_tolerance,
                status=ValidationCheckStatus.FAIL,
                message=f"{label} is missing",
            )
        absolute_error = abs(reported - recomputed)
        scale = max(abs(reported), abs(recomputed))
        relative_error = absolute_error / scale if scale else 0.0
        passed = absolute_error <= self._absolute_tolerance + self._relative_tolerance * scale
        return MetricRecalculation(
            metric_id=metric_id,
            category=category,
            reported_value=reported,
            recomputed_value=recomputed,
            absolute_error=absolute_error,
            relative_error=relative_error,
            absolute_tolerance=self._absolute_tolerance,
            relative_tolerance=self._relative_tolerance,
            status=ValidationCheckStatus.PASS if passed else ValidationCheckStatus.FAIL,
            message=(
                f"{label} independently matches"
                if passed
                else f"{label} differs from independent recalculation"
            ),
        )

    def _requirement_checks(
        self,
        model: MathematicalModel,
        *,
        evidence: EvidenceChainReport,
        variable_checks: list[VariableValidationCheck],
        constraint_checks: list[ConstraintValidationCheck],
        metrics: list[MetricRecalculation],
    ) -> list[ValidationRequirementCheck]:
        checks: list[ValidationRequirementCheck] = []
        for requirement in model.validation_requirements:
            normalized = requirement.casefold()
            relevant: list[ValidationCheckStatus]
            refs: list[str]
            if "constraint" in normalized:
                relevant = [item.status for item in constraint_checks]
                refs = [item.constraint_id for item in constraint_checks]
            elif any(token in normalized for token in ("bound", "domain", "variable")):
                relevant = [item.status for item in variable_checks]
                refs = [item.variable_id for item in variable_checks]
            elif any(token in normalized for token in ("objective", "metric", "output")):
                relevant = [item.status for item in metrics]
                refs = [item.metric_id for item in metrics]
            elif "evidence" in normalized or "trace" in normalized:
                relevant = [
                    ValidationCheckStatus.PASS if evidence.valid else ValidationCheckStatus.FAIL
                ]
                refs = [f"result:{evidence.result_id}"]
            else:
                relevant = [ValidationCheckStatus.UNCHECKED]
                refs = []
            status = (
                ValidationCheckStatus.FAIL
                if any(item is ValidationCheckStatus.FAIL for item in relevant)
                else ValidationCheckStatus.UNCHECKED
                if not relevant or any(item is ValidationCheckStatus.UNCHECKED for item in relevant)
                else ValidationCheckStatus.PASS
            )
            checks.append(
                ValidationRequirementCheck(
                    requirement=requirement,
                    status=status,
                    evidence_refs=refs,
                    message=f"validation requirement is {status.value.lower()}: {requirement}",
                )
            )
        return checks

    @staticmethod
    def _numeric_environment(
        model: MathematicalModel,
        variable_values: dict[str, float],
    ) -> dict[str, float]:
        values: dict[str, float] = {}
        for parameter in [*model.parameters, *model.constants]:
            scalar = IndependentValidator._scalar_parameter(parameter)
            if scalar is not None:
                values[parameter.symbol] = scalar
        values.update(variable_values)
        return values

    @staticmethod
    def _scalar_parameter(parameter: ParameterDefinition) -> float | None:
        value = parameter.value
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, int | float) and math.isfinite(float(value)):
            return float(value)
        return None
