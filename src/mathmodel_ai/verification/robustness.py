from __future__ import annotations

import random
import statistics

from mathmodel_ai.schemas.independent_verification import ReviewedValidationEvidence
from mathmodel_ai.schemas.mathematical import MathematicalModel, ObjectiveSense
from mathmodel_ai.schemas.results import ResultRecord
from mathmodel_ai.schemas.verification import (
    ExperimentReportStatus,
    ExperimentRun,
    ExperimentStatus,
    RobustnessConfig,
    RobustnessMethod,
    RobustnessReport,
    RobustnessSummary,
    SensitivityReport,
    ValidationReport,
    ValidationStatus,
)
from mathmodel_ai.verification.experiments import ExperimentEngine, ExperimentOutcome
from mathmodel_ai.verification.reviewed_response import reviewed_response_summary


class RobustnessAnalyzer:
    def __init__(self, engine: ExperimentEngine) -> None:
        self._engine = engine

    def analyze(
        self,
        *,
        model: MathematicalModel,
        result: ResultRecord,
        validation: ValidationReport,
        sensitivity: SensitivityReport,
        config: RobustnessConfig,
        reviewed_evidence: ReviewedValidationEvidence | None = None,
    ) -> tuple[RobustnessReport, list[ExperimentOutcome]]:
        if validation.status is not ValidationStatus.PASS:
            return self._blocked(
                model, result, validation, sensitivity, config, "validation did not pass"
            ), []
        if sensitivity.status not in {
            ExperimentReportStatus.PASS,
            ExperimentReportStatus.PARTIAL,
        }:
            return self._blocked(
                model, result, validation, sensitivity, config, "sensitivity produced no evidence"
            ), []
        if result.objective is None:
            if (
                reviewed_evidence is not None
                and sensitivity.reviewed_report_id == reviewed_evidence.report.report_id
            ):
                try:
                    replay_ids, values = reviewed_response_summary(
                        reviewed_evidence,
                        model_digest=result.model_digest,
                        result_id=result.result_id,
                        parameter_only=False,
                    )
                except ValueError as exc:
                    return self._blocked(
                        model, result, validation, sensitivity, config, str(exc)
                    ), []
                return RobustnessReport(
                    project_id=model.project_id,
                    problem_id=model.problem_id,
                    model_id=model.model_id,
                    model_version=model.version,
                    model_digest=result.model_digest,
                    result_id=result.result_id,
                    validation_id=validation.validation_id,
                    sensitivity_id=sensitivity.sensitivity_id,
                    baseline_objective=None,
                    reviewed_report_id=reviewed_evidence.report.report_id,
                    reviewed_replay_ids=replay_ids,
                    reviewed_metric_values=values,
                    method=config.method,
                    config=config,
                    summary=RobustnessSummary(requested_runs=0, successful_runs=0, failed_runs=0),
                    status=ExperimentReportStatus.PASS,
                ), []
            return self._blocked(
                model, result, validation, sensitivity, config, "baseline objective is absent"
            ), []
        if config.method is RobustnessMethod.BOOTSTRAP:
            return self._blocked(
                model,
                result,
                validation,
                sensitivity,
                config,
                "bootstrap requires an explicit data-resampling contract not present in the model",
            ), []
        try:
            parameters = self._engine.select_scalar_parameters(
                model,
                config.parameter_symbols,
                limit=20,
            )
        except ValueError as exc:
            return self._blocked(model, result, validation, sensitivity, config, str(exc)), []
        if not parameters:
            return self._blocked(
                model,
                result,
                validation,
                sensitivity,
                config,
                "no sourced scalar model parameters are eligible for robustness scenarios",
            ), []

        fractions = self._scenario_fractions(config, len(parameters))
        outcomes: list[ExperimentOutcome] = []
        for index, scenario in enumerate(fractions[: config.max_runs], start=1):
            perturbations = [
                self._engine.perturbation(parameter, baseline, fraction)
                for (parameter, baseline), fraction in zip(parameters, scenario, strict=True)
            ]
            outcomes.append(
                self._engine.execute(
                    model=model,
                    perturbations=perturbations,
                    options=config.solver_options,
                    experiment_type=f"ROBUSTNESS:{config.method.value}:{index}",
                    baseline_objective=result.objective,
                )
            )
        records = [item.record for item in outcomes]
        passed = [item for item in records if item.status is ExperimentStatus.PASS]
        failed = [item for item in records if item.status is ExperimentStatus.FAIL]
        objectives = [item.objective_value for item in passed if item.objective_value is not None]
        worst = self._worst(model, passed)
        summary = RobustnessSummary(
            requested_runs=len(records),
            successful_runs=len(passed),
            failed_runs=len(failed),
            feasibility_rate=(len(passed) / len(records) if records else None),
            objective_mean=statistics.fmean(objectives) if objectives else None,
            objective_std=statistics.pstdev(objectives) if objectives else None,
            objective_min=min(objectives) if objectives else None,
            objective_max=max(objectives) if objectives else None,
            objective_quantiles=self._quantiles(objectives),
            worst_case_experiment_id=worst.experiment_id if worst is not None else None,
        )
        status = (
            ExperimentReportStatus.PASS
            if records and not failed
            else ExperimentReportStatus.PARTIAL
            if passed
            else ExperimentReportStatus.FAIL
        )
        return (
            RobustnessReport(
                project_id=model.project_id,
                problem_id=model.problem_id,
                model_id=model.model_id,
                model_version=model.version,
                model_digest=result.model_digest,
                result_id=result.result_id,
                validation_id=validation.validation_id,
                sensitivity_id=sensitivity.sensitivity_id,
                baseline_objective=result.objective,
                method=config.method,
                config=config,
                experiments=records,
                summary=summary,
                status=status,
                errors=[item.error for item in failed if item.error],
                warnings=(
                    ["robustness scenarios were truncated by max_runs"]
                    if len(fractions) > config.max_runs
                    else []
                ),
            ),
            outcomes,
        )

    @staticmethod
    def _scenario_fractions(config: RobustnessConfig, parameter_count: int) -> list[list[float]]:
        if config.method in {
            RobustnessMethod.SCENARIO_ANALYSIS,
            RobustnessMethod.WORST_CASE,
        }:
            return [[fraction] * parameter_count for fraction in config.scenario_fractions]
        generator = random.Random(config.random_seed)
        scenarios: list[list[float]] = []
        for _ in range(config.sample_count):
            if config.method is RobustnessMethod.NOISE_PERTURBATION:
                scenarios.append(
                    [
                        generator.uniform(-config.noise_fraction, config.noise_fraction)
                        for _ in range(parameter_count)
                    ]
                )
            elif config.method is RobustnessMethod.MONTE_CARLO:
                scenarios.append(
                    [
                        max(
                            -1.0,
                            min(1.0, generator.gauss(0.0, config.noise_fraction)),
                        )
                        for _ in range(parameter_count)
                    ]
                )
        return scenarios

    @staticmethod
    def _worst(
        model: MathematicalModel,
        records: list[ExperimentRun],
    ) -> ExperimentRun | None:
        with_objective = [item for item in records if item.objective_value is not None]
        if not with_objective:
            return None
        minimize = model.objective is None or model.objective.sense is ObjectiveSense.MINIMIZE
        return (
            max(with_objective, key=lambda item: item.objective_value or 0.0)
            if minimize
            else min(with_objective, key=lambda item: item.objective_value or 0.0)
        )

    @staticmethod
    def _quantiles(values: list[float]) -> dict[str, float]:
        if not values:
            return {}
        ordered = sorted(values)
        result: dict[str, float] = {}
        for label, probability in (("p05", 0.05), ("p50", 0.5), ("p95", 0.95)):
            index = round((len(ordered) - 1) * probability)
            result[label] = ordered[index]
        return result

    @staticmethod
    def _blocked(
        model: MathematicalModel,
        result: ResultRecord,
        validation: ValidationReport,
        sensitivity: SensitivityReport,
        config: RobustnessConfig,
        reason: str,
    ) -> RobustnessReport:
        return RobustnessReport(
            project_id=model.project_id,
            problem_id=model.problem_id,
            model_id=model.model_id,
            model_version=model.version,
            model_digest=result.model_digest,
            result_id=result.result_id,
            validation_id=validation.validation_id,
            sensitivity_id=sensitivity.sensitivity_id,
            baseline_objective=result.objective,
            method=config.method,
            config=config,
            summary=RobustnessSummary(
                requested_runs=0,
                successful_runs=0,
                failed_runs=0,
            ),
            status=ExperimentReportStatus.BLOCKED,
            errors=[f"ROBUSTNESS_BLOCKED:{reason}"],
        )
