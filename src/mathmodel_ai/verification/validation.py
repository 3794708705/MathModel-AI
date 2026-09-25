from __future__ import annotations

import math
from dataclasses import asdict

from mathmodel_ai.mathematical.digests import mathematical_model_digest
from mathmodel_ai.paper.hashing import sha256_json
from mathmodel_ai.schemas.benchmark import CausalScienceCheck
from mathmodel_ai.schemas.execution import ExecutionStatus
from mathmodel_ai.schemas.independent_verification import (
    IndependentStatus,
    ReviewedValidationEvidence,
    ValidationRequirementBinding,
)
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
from mathmodel_ai.verification.causal_holdout import (
    AuditedCausalEvidence,
    assess_binary_calibration,
)
from mathmodel_ai.verification.evaluator import (
    IndependentEvaluationError,
    IndependentExpressionEvaluator,
)

# Exact legacy contracts, not natural-language classifiers. Additional scientific
# obligations must retain UNCHECKED until an evaluator supplies their own evidence.
_REQUIREMENT_CONTRACTS = {
    "recompute variable bounds": (ValidationCheckCategory.VARIABLE,),
    "recompute every constraint": (ValidationCheckCategory.CONSTRAINT,),
    "recompute variable bounds and constraints": (
        ValidationCheckCategory.VARIABLE,
        ValidationCheckCategory.CONSTRAINT,
    ),
    "recalculate objective metric": (ValidationCheckCategory.OBJECTIVE,),
    "verify evidence trace": (ValidationCheckCategory.EVIDENCE,),
}


class IndependentValidator:
    version = "independent-validator-5.1.0"

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
        reviewed_evidence: ReviewedValidationEvidence | None = None,
        causal_evidence: AuditedCausalEvidence | None = None,
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
        reviewed_errors = self._reviewed_evidence_errors(
            model=model,
            result=result,
            solver_run=solver_run,
            reviewed_evidence=reviewed_evidence,
        )
        errors.extend(f"VALIDATION_FAIL:{item}" for item in reviewed_errors)
        causal_bound = (
            causal_evidence is None or causal_evidence.formal_result_id == result.result_id
        )
        if not causal_bound:
            errors.append("VALIDATION_FAIL:CAUSAL_HOLDOUT_FORMAL_RESULT_MISMATCH")
        calibration_valid = False
        if causal_evidence is not None and causal_evidence.calibration is not None:
            try:
                calibration_valid = causal_evidence.calibration == assess_binary_calibration(
                    causal_evidence.result
                )
            except ValueError:
                pass
            if not calibration_valid:
                errors.append("VALIDATION_FAIL:CAUSAL_CALIBRATION_RECOMPUTATION_MISMATCH")
        metrics = [
            *self._metric_checks(model, result, solver_run, values),
            *self._reviewed_metric_checks(reviewed_evidence),
            *self._causal_metric_checks(
                causal_evidence, bound=causal_bound, calibration_valid=calibration_valid
            ),
        ]
        requirement_checks = self._requirement_checks(
            model,
            evidence=evidence,
            variable_checks=variable_checks,
            constraint_checks=constraint_checks,
            metrics=metrics,
            reviewed_evidence=reviewed_evidence,
            reviewed_errors=reviewed_errors,
            causal_evidence=causal_evidence,
            causal_bound=causal_bound and evidence.valid,
            calibration_valid=calibration_valid,
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
                *(
                    [
                        f"reviewed_policy:{sha256_json(reviewed_evidence.requirements)}",
                        f"independent_plan:{reviewed_evidence.plan.plan_id}",
                        f"independent_report:{reviewed_evidence.report.report_id}",
                    ]
                    if reviewed_evidence is not None
                    else []
                ),
                *(
                    [
                        f"causal_holdout_execution:{causal_evidence.holdout_execution_id}",
                        f"causal_source_sha256:{causal_evidence.source_sha256}",
                        f"causal_trace_sha256:{causal_evidence.trace_sha256}",
                        *(
                            [
                                f"causal_science_policy_sha256:{causal_evidence.science_policy_sha256}"
                            ]
                            if causal_evidence.science_policy_sha256 is not None
                            else []
                        ),
                        *(
                            [
                                "causal_calibration_sha256:"
                                + sha256_json(asdict(causal_evidence.calibration))
                            ]
                            if causal_evidence.calibration is not None
                            else []
                        ),
                    ]
                    if causal_evidence is not None
                    else []
                ),
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
        reviewed_evidence: ReviewedValidationEvidence | None = None,
        causal_evidence: AuditedCausalEvidence | None = None,
    ) -> list[str]:
        """Recompute a persisted report and compare every deterministic field."""
        recomputed = self.validate(
            model=model,
            result=result,
            solver_run=solver_run,
            evidence=evidence,
            reviewed_evidence=reviewed_evidence,
            causal_evidence=causal_evidence,
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

    @staticmethod
    def _model_evidence_refs(model: MathematicalModel) -> set[str]:
        return {
            *(item.assumption_id for item in model.assumptions),
            *(item.ambiguity_id for item in model.interpretation_resolutions),
            *(item.set_id for item in model.sets),
            *(item.index_id for item in model.indices),
            *(
                item.variable_id
                for item in [
                    *model.decision_variables,
                    *model.state_variables,
                    *model.derived_variables,
                ]
            ),
            *(item.parameter_id for item in [*model.parameters, *model.constants]),
            *(
                item.constraint_id
                for item in [
                    *model.constraints,
                    *model.initial_conditions,
                    *model.boundary_conditions,
                ]
            ),
            *(item.equation_id for item in model.equations),
            *(item.output_id for item in model.expected_outputs),
            *model.source_evidence,
        }

    @classmethod
    def _reviewed_evidence_errors(
        cls,
        *,
        model: MathematicalModel,
        result: ResultRecord,
        solver_run: SolverRun,
        reviewed_evidence: ReviewedValidationEvidence | None,
    ) -> list[str]:
        if reviewed_evidence is None:
            return []
        policy = reviewed_evidence.requirements
        plan = reviewed_evidence.plan
        report = reviewed_evidence.report
        errors: list[str] = []
        model_digest = mathematical_model_digest(model)
        if policy.model_digest != model_digest:
            errors.append("REVIEWED_POLICY_MODEL_DIGEST_MISMATCH")
        if [item.requirement for item in policy.validation_requirement_bindings] != (
            model.validation_requirements
        ):
            errors.append("REVIEWED_REQUIREMENT_COVERAGE_MISMATCH")
        if any(
            not set(item.model_evidence_refs) <= cls._model_evidence_refs(model)
            for item in policy.validation_requirement_bindings
        ):
            errors.append("REVIEWED_REQUIREMENT_MODEL_EVIDENCE_MISMATCH")
        if (
            plan.result_id != result.result_id
            or report.result_id != result.result_id
            or report.result_id != plan.result_id
        ):
            errors.append("REVIEWED_EVIDENCE_RESULT_MISMATCH")
        if (
            report.plan_id != plan.plan_id
            or report.report_id != plan.plan_id
            or report.attempt_id != plan.attempt_id
            or report.plan_digest != sha256_json(plan)
        ):
            errors.append("REVIEWED_EVIDENCE_PLAN_MISMATCH")
        if (
            plan.metrics != policy.metrics
            or plan.scenarios != policy.scenarios
            or plan.scientific_scope != policy.scientific_scope
            or not policy.matches_observation(plan)
        ):
            errors.append("REVIEWED_PLAN_POLICY_MISMATCH")
        if report.status is not IndependentStatus.PASS or report.errors:
            errors.append("REVIEWED_INDEPENDENT_REPORT_NOT_PASS")

        required_metrics = [item for item in policy.metrics if item.required]
        if (
            report.required_metrics != len(required_metrics)
            or report.passed_metrics != len(required_metrics)
            or [item.metric_id for item in report.metrics]
            != [item.metric_id for item in policy.metrics]
        ):
            errors.append("REVIEWED_INDEPENDENT_METRIC_COVERAGE_MISMATCH")
        top_metrics = {item.metric_id: item for item in report.metrics}
        if any(
            (verified := top_metrics.get(spec.metric_id)) is None
            or verified.key is not spec.key
            or verified.status is not IndependentStatus.PASS
            or verified.source_digest != plan.source_sha256
            for spec in required_metrics
        ):
            errors.append("REVIEWED_INDEPENDENT_METRIC_FAILED_OR_TAMPERED")

        required_scenarios = [item for item in policy.scenarios if item.required]
        if (
            report.required_scenarios != len(required_scenarios)
            or report.passed_scenarios != len(required_scenarios)
            or [item.scenario_id for item in report.replays]
            != [item.scenario_id for item in policy.scenarios]
        ):
            errors.append("REVIEWED_SCENARIO_COVERAGE_MISMATCH")
        execution_ids = [item.execution.run_id for item in report.replays]
        if (
            len(execution_ids) != len(set(execution_ids))
            or solver_run.execution_ref in execution_ids
        ):
            errors.append("REVIEWED_SCENARIO_EXECUTION_REUSED")
        replay_by_id = {item.scenario_id: item for item in report.replays}
        for spec in required_scenarios:
            replay = replay_by_id.get(spec.scenario_id)
            if replay is None:
                errors.append(f"REVIEWED_SCENARIO_MISSING:{spec.scenario_id}")
                continue
            required_ids = [item.metric_id for item in spec.metrics if item.required]
            metric_by_id = {item.metric_id: item for item in replay.metrics}
            output_artifact = next(
                (
                    item
                    for item in replay.artifacts
                    if item.artifact_id == replay.output_artifact_id
                ),
                None,
            )
            if (
                replay.scenario_digest != sha256_json(spec)
                or replay.status is not IndependentStatus.PASS
                or replay.error is not None
                or replay.execution.status is not ExecutionStatus.SUCCEEDED
                or replay.execution.is_mock
                or not replay.execution.network_disabled
                or not replay.execution.non_root
                or not replay.execution.read_only_root
                or output_artifact is None
                or output_artifact.sha256 != replay.output_digest
                or output_artifact.execution_run_id != replay.execution.run_id
                or [item.metric_id for item in replay.metrics]
                != [item.metric_id for item in spec.metrics]
                or any(
                    (metric := metric_by_id.get(metric_id)) is None
                    or metric.status is not IndependentStatus.PASS
                    or metric.source_digest != replay.output_digest
                    for metric_id in required_ids
                )
            ):
                errors.append(f"REVIEWED_SCENARIO_FAILED_OR_TAMPERED:{spec.scenario_id}")
        return list(dict.fromkeys(errors))

    @staticmethod
    def _causal_metric_checks(
        evidence: AuditedCausalEvidence | None, *, bound: bool, calibration_valid: bool
    ) -> list[MetricRecalculation]:
        if evidence is None:
            return []
        checks = [
            MetricRecalculation(
                metric_id=f"causal_holdout:{name}",
                category=ValidationCheckCategory.OUTPUT,
                recomputed_value=value if bound else None,
                absolute_tolerance=0,
                relative_tolerance=0,
                status=ValidationCheckStatus.PASS if bound else ValidationCheckStatus.FAIL,
                message=(
                    f"independently replayed causal holdout {name}"
                    if bound
                    else "causal holdout is not bound to the formal result"
                ),
            )
            for name, value in (
                ("brier", evidence.result.brier),
                ("baseline_brier", evidence.result.baseline_brier),
            )
        ]
        if evidence.calibration is not None:
            checks.append(
                MetricRecalculation(
                    metric_id="causal_holdout:calibration_ece",
                    category=ValidationCheckCategory.OUTPUT,
                    recomputed_value=(
                        evidence.calibration.ece if bound and calibration_valid else None
                    ),
                    absolute_tolerance=0,
                    relative_tolerance=0,
                    status=(
                        ValidationCheckStatus.PASS
                        if bound and calibration_valid
                        else ValidationCheckStatus.FAIL
                    ),
                    message=(
                        "independently recomputed holdout reliability bins and ECE"
                        if bound and calibration_valid
                        else "holdout calibration could not be independently recomputed"
                    ),
                )
            )
        return checks

    @staticmethod
    def _reviewed_metric_checks(
        reviewed_evidence: ReviewedValidationEvidence | None,
    ) -> list[MetricRecalculation]:
        if reviewed_evidence is None:
            return []
        verified_by_id = {item.metric_id: item for item in reviewed_evidence.report.metrics}
        checks: list[MetricRecalculation] = []
        for spec in reviewed_evidence.requirements.metrics:
            verified = verified_by_id.get(spec.metric_id)
            passed = (
                verified is not None
                and verified.status is IndependentStatus.PASS
                and verified.verified is not None
            )
            checks.append(
                MetricRecalculation(
                    metric_id=f"independent:{spec.metric_id}",
                    category=ValidationCheckCategory.OUTPUT,
                    reported_value=verified.reported if verified is not None else None,
                    recomputed_value=verified.verified if verified is not None else None,
                    absolute_error=(
                        abs(verified.delta)
                        if verified is not None and verified.delta is not None
                        else None
                    ),
                    relative_error=None,
                    absolute_tolerance=spec.absolute_tolerance,
                    relative_tolerance=spec.relative_tolerance,
                    status=(ValidationCheckStatus.PASS if passed else ValidationCheckStatus.FAIL),
                    message=(
                        f"reviewed independent metric passed: {spec.metric_id}"
                        if passed
                        else f"reviewed independent metric failed: {spec.metric_id}"
                    ),
                )
            )
        return checks

    def _reviewed_requirement_check(
        self,
        model: MathematicalModel,
        binding: ValidationRequirementBinding,
        reviewed_evidence: ReviewedValidationEvidence,
        reviewed_errors: list[str],
    ) -> ValidationRequirementCheck:
        report = reviewed_evidence.report
        metric_statuses: dict[str, IndependentStatus] = {
            item.metric_id: item.status for item in report.metrics
        }
        for replay in report.replays:
            metric_statuses.update({item.metric_id: item.status for item in replay.metrics})
        scenario_statuses = {item.scenario_id: item.status for item in report.replays}
        known_model_refs = self._model_evidence_refs(model)
        refs = [
            *(f"independent_metric:{item}" for item in binding.metric_ids),
            *(f"independent_scenario:{item}" for item in binding.scenario_ids),
            *(f"model_evidence:{item}" for item in binding.model_evidence_refs),
        ]
        if binding.scope == "PAPER":
            refs.append("paper_gate:required")
        passed = (
            not reviewed_errors
            and all(
                metric_statuses.get(item) is IndependentStatus.PASS for item in binding.metric_ids
            )
            and all(
                scenario_statuses.get(item) is IndependentStatus.PASS
                for item in binding.scenario_ids
            )
            and set(binding.model_evidence_refs) <= known_model_refs
        )
        if binding.scope == "PAPER":
            message = (
                "paper-scoped model requirement is bound to immutable reviewed evidence; "
                "the Paper gate remains responsible for rendered-claim enforcement"
                if passed
                else "paper-scoped model requirement has invalid reviewed evidence"
            )
        else:
            message = (
                f"reviewed validation requirement is {'passed' if passed else 'failed'}: "
                f"{binding.requirement}"
            )
        return ValidationRequirementCheck(
            requirement=binding.requirement,
            scope=binding.scope,
            status=ValidationCheckStatus.PASS if passed else ValidationCheckStatus.FAIL,
            evidence_refs=refs,
            message=message,
        )

    def _requirement_checks(
        self,
        model: MathematicalModel,
        *,
        evidence: EvidenceChainReport,
        variable_checks: list[VariableValidationCheck],
        constraint_checks: list[ConstraintValidationCheck],
        metrics: list[MetricRecalculation],
        reviewed_evidence: ReviewedValidationEvidence | None,
        reviewed_errors: list[str],
        causal_evidence: AuditedCausalEvidence | None,
        causal_bound: bool,
        calibration_valid: bool,
    ) -> list[ValidationRequirementCheck]:
        objective_metrics = [
            item for item in metrics if item.category is ValidationCheckCategory.OBJECTIVE
        ]
        available = {
            ValidationCheckCategory.VARIABLE: (
                [item.status for item in variable_checks],
                [item.variable_id for item in variable_checks],
            ),
            ValidationCheckCategory.CONSTRAINT: (
                [item.status for item in constraint_checks],
                [item.constraint_id for item in constraint_checks],
            ),
            ValidationCheckCategory.OBJECTIVE: (
                [item.status for item in objective_metrics],
                [item.metric_id for item in objective_metrics],
            ),
            ValidationCheckCategory.EVIDENCE: (
                [ValidationCheckStatus.PASS if evidence.valid else ValidationCheckStatus.FAIL],
                [f"result:{evidence.result_id}"],
            ),
        }
        reviewed_bindings = (
            {
                item.requirement: item
                for item in reviewed_evidence.requirements.validation_requirement_bindings
            }
            if reviewed_evidence is not None
            else {}
        )
        checks: list[ValidationRequirementCheck] = []
        causal_checks = self._causal_requirement_checks(
            causal_evidence, bound=causal_bound, calibration_valid=calibration_valid
        )
        causal_names = {item.requirement for item in causal_checks}
        for requirement in model.validation_requirements:
            binding = reviewed_bindings.get(requirement)
            if binding is not None and reviewed_evidence is not None:
                checks.append(
                    self._reviewed_requirement_check(
                        model,
                        binding,
                        reviewed_evidence,
                        reviewed_errors,
                    )
                )
                continue
            if requirement in causal_names:
                continue  # The host check below covers this exact declared identifier.
            normalized = " ".join(requirement.casefold().split())
            relevant: list[ValidationCheckStatus] = []
            refs: list[str] = []
            for category in _REQUIREMENT_CONTRACTS.get(normalized, ()):
                statuses, evidence_refs = available[category]
                relevant.extend(statuses or [ValidationCheckStatus.UNCHECKED])
                refs.extend(evidence_refs)
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
        checks.extend(causal_checks)
        return checks

    @staticmethod
    def _causal_requirement_checks(
        evidence: AuditedCausalEvidence | None, *, bound: bool, calibration_valid: bool
    ) -> list[ValidationRequirementCheck]:
        if evidence is None:
            return []
        policy_valid = (
            evidence.science_policy_sha256 is not None
            and len(evidence.science_policy_sha256) == 64
            and all(character in "0123456789abcdef" for character in evidence.science_policy_sha256)
            and len(set(evidence.required_scientific_checks))
            == len(evidence.required_scientific_checks)
        )
        checks: list[ValidationRequirementCheck] = []
        for capability in evidence.required_scientific_checks:
            status = (
                ValidationCheckStatus.FAIL
                if not bound or not policy_valid
                else ValidationCheckStatus.FAIL
                if capability is CausalScienceCheck.CALIBRATION_ASSESSMENT
                and evidence.calibration is not None
                and not calibration_valid
                else ValidationCheckStatus.PASS
                if capability is CausalScienceCheck.HELDOUT_PREDICTION
                or (capability is CausalScienceCheck.CALIBRATION_ASSESSMENT and calibration_valid)
                else ValidationCheckStatus.UNCHECKED
            )
            checks.append(
                ValidationRequirementCheck(
                    requirement=f"causal_science:{capability.value}",
                    status=status,
                    evidence_refs=[
                        f"causal_holdout_execution:{evidence.holdout_execution_id}",
                        f"causal_trace_sha256:{evidence.trace_sha256}",
                        f"causal_science_policy_sha256:{evidence.science_policy_sha256}",
                        *(
                            [
                                "causal_calibration_sha256:"
                                + sha256_json(asdict(evidence.calibration))
                            ]
                            if capability is CausalScienceCheck.CALIBRATION_ASSESSMENT
                            and evidence.calibration is not None
                            else []
                        ),
                    ],
                    message=(
                        f"host scientific check is {status.value.lower()}: {capability.value}"
                    ),
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
