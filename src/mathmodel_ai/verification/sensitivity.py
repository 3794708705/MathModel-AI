from __future__ import annotations

from collections import defaultdict

from mathmodel_ai.schemas.independent_verification import ReviewedValidationEvidence
from mathmodel_ai.schemas.mathematical import MathematicalModel
from mathmodel_ai.schemas.results import ResultRecord
from mathmodel_ai.schemas.verification import (
    ExperimentReportStatus,
    ExperimentStatus,
    SensitivityConfig,
    SensitivityReport,
    ValidationReport,
    ValidationStatus,
)
from mathmodel_ai.verification.experiments import ExperimentEngine, ExperimentOutcome
from mathmodel_ai.verification.reviewed_response import reviewed_response_summary


class SensitivityAnalyzer:
    def __init__(self, engine: ExperimentEngine) -> None:
        self._engine = engine

    def analyze(
        self,
        *,
        model: MathematicalModel,
        result: ResultRecord,
        validation: ValidationReport,
        config: SensitivityConfig,
        reviewed_evidence: ReviewedValidationEvidence | None = None,
    ) -> tuple[SensitivityReport, list[ExperimentOutcome]]:
        if validation.status is not ValidationStatus.PASS:
            return self._blocked(model, result, validation, config, "validation did not pass"), []
        if result.objective is None:
            if reviewed_evidence is not None:
                try:
                    replay_ids, values = reviewed_response_summary(
                        reviewed_evidence,
                        model_digest=result.model_digest,
                        result_id=result.result_id,
                        parameter_only=True,
                    )
                except ValueError as exc:
                    return self._blocked(model, result, validation, config, str(exc)), []
                return SensitivityReport(
                    project_id=model.project_id,
                    problem_id=model.problem_id,
                    model_id=model.model_id,
                    model_version=model.version,
                    model_digest=result.model_digest,
                    result_id=result.result_id,
                    validation_id=validation.validation_id,
                    baseline_objective=None,
                    reviewed_report_id=reviewed_evidence.report.report_id,
                    reviewed_replay_ids=replay_ids,
                    reviewed_metric_values=values,
                    config=config,
                    successful_runs=0,
                    failed_runs=0,
                    status=ExperimentReportStatus.PASS,
                ), []
            return self._blocked(
                model, result, validation, config, "baseline objective is absent"
            ), []
        try:
            parameters = self._engine.select_scalar_parameters(
                model,
                config.parameter_symbols,
                config.max_parameters,
            )
        except ValueError as exc:
            return self._blocked(model, result, validation, config, str(exc)), []
        if not parameters:
            return self._blocked(
                model,
                result,
                validation,
                config,
                "no sourced scalar model parameters are eligible for perturbation",
            ), []

        scenarios = [
            (parameter, baseline, direction * fraction)
            for parameter, baseline in parameters
            for fraction in config.perturbation_fractions
            for direction in (-1.0, 1.0)
        ]
        if len(scenarios) > config.max_runs:
            scenarios = scenarios[: config.max_runs]
        outcomes = [
            self._engine.execute(
                model=model,
                perturbations=[self._engine.perturbation(parameter, baseline, fraction)],
                options=config.solver_options,
                experiment_type="SENSITIVITY",
                baseline_objective=result.objective,
            )
            for parameter, baseline, fraction in scenarios
        ]
        records = [item.record for item in outcomes]
        passed = [item for item in records if item.status is ExperimentStatus.PASS]
        failed = [item for item in records if item.status is ExperimentStatus.FAIL]
        objectives = [item.objective_value for item in passed if item.objective_value is not None]
        relative_changes = [
            abs(item.objective_change_fraction)
            for item in passed
            if item.objective_change_fraction is not None
        ]
        elasticities: dict[str, list[float]] = defaultdict(list)
        for item in passed:
            change = item.objective_change_fraction
            fraction = item.perturbations[0].fraction
            if change is not None and fraction:
                elasticities[item.perturbations[0].symbol].append(change / fraction)
        summarized = {
            symbol: sum(values) / len(values) for symbol, values in sorted(elasticities.items())
        }
        status = (
            ExperimentReportStatus.PASS
            if records and not failed
            else ExperimentReportStatus.PARTIAL
            if passed
            else ExperimentReportStatus.FAIL
        )
        warnings = []
        if len(scenarios) == config.max_runs and (
            len(parameters) * len(config.perturbation_fractions) * 2 > config.max_runs
        ):
            warnings.append("sensitivity scenarios were truncated by max_runs")
        return (
            SensitivityReport(
                project_id=model.project_id,
                problem_id=model.problem_id,
                model_id=model.model_id,
                model_version=model.version,
                model_digest=result.model_digest,
                result_id=result.result_id,
                validation_id=validation.validation_id,
                baseline_objective=result.objective,
                config=config,
                experiments=records,
                parameter_elasticities=summarized,
                objective_min=min(objectives) if objectives else None,
                objective_max=max(objectives) if objectives else None,
                maximum_absolute_relative_change=max(relative_changes, default=None),
                successful_runs=len(passed),
                failed_runs=len(failed),
                status=status,
                errors=[item.error for item in failed if item.error],
                warnings=warnings,
            ),
            outcomes,
        )

    @staticmethod
    def _blocked(
        model: MathematicalModel,
        result: ResultRecord,
        validation: ValidationReport,
        config: SensitivityConfig,
        reason: str,
    ) -> SensitivityReport:
        return SensitivityReport(
            project_id=model.project_id,
            problem_id=model.problem_id,
            model_id=model.model_id,
            model_version=model.version,
            model_digest=result.model_digest,
            result_id=result.result_id,
            validation_id=validation.validation_id,
            baseline_objective=result.objective,
            config=config,
            successful_runs=0,
            failed_runs=0,
            status=ExperimentReportStatus.BLOCKED,
            errors=[f"SENSITIVITY_BLOCKED:{reason}"],
        )
