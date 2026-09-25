import math
import random
import statistics
from collections import defaultdict

from mathmodel_ai.mathematical.digests import mathematical_model_digest
from mathmodel_ai.schemas.independent_verification import ReviewedValidationEvidence
from mathmodel_ai.schemas.quality import QualityGateResult, QualityGateStatus
from mathmodel_ai.schemas.solver import SolverName
from mathmodel_ai.schemas.verification import (
    ExperimentReportStatus,
    ExperimentRun,
    ExperimentStatus,
    ModelRepairOutput,
    RedTeamReport,
    RedTeamSeverity,
    RobustnessReport,
    SensitivityReport,
    ValidationReport,
    ValidationStatus,
)
from mathmodel_ai.verification.reviewed_response import reviewed_response_summary


def validation_quality_gate(report: ValidationReport) -> QualityGateResult:
    checks = {
        "evidence_chain_valid": report.evidence.valid,
        "independent_status_pass": report.status is ValidationStatus.PASS,
        "all_variables_checked": bool(report.variable_checks)
        and all(item.status.value == "PASS" for item in report.variable_checks),
        "all_constraints_checked": all(
            item.status.value == "PASS" for item in report.constraint_checks
        ),
        "all_metrics_recalculated": bool(report.metric_recalculations)
        and all(item.status.value == "PASS" for item in report.metric_recalculations),
        "declared_requirements_checked": all(
            item.status.value == "PASS" for item in report.requirement_checks
        ),
    }
    errors = [f"VALIDATE_GATE_FAIL:{name}" for name, passed in checks.items() if not passed]
    errors.extend(report.errors)
    return QualityGateResult(
        gate="VALIDATE",
        status=QualityGateStatus.PASS if not errors else QualityGateStatus.RETRY,
        checks=checks,
        errors=list(dict.fromkeys(errors)),
        warnings=report.warnings,
    )


def sensitivity_quality_gate(
    report: SensitivityReport,
    reviewed_evidence: ReviewedValidationEvidence | None = None,
) -> QualityGateResult:
    if report.reviewed_report_id is not None:
        reviewed_valid = _reviewed_report_matches(report, reviewed_evidence, parameter_only=True)
        checks = {
            "independent_report_bound": reviewed_valid,
            "parameter_perturbations_present": bool(report.reviewed_replay_ids),
            "all_required_numeric_responses_verified": bool(report.reviewed_metric_values),
            "no_objective_fabricated": report.baseline_objective is None and not report.experiments,
            "report_pass": report.status is ExperimentReportStatus.PASS,
        }
        errors = [f"SENSITIVITY_GATE_FAIL:{name}" for name, passed in checks.items() if not passed]
        errors.extend(report.errors)
        return QualityGateResult(
            gate="SENSITIVITY",
            status=QualityGateStatus.PASS if not errors else QualityGateStatus.RETRY,
            checks=checks,
            errors=list(dict.fromkeys(errors)),
            warnings=report.warnings,
        )
    if report.baseline_responses:
        checks = {
            "experiments_present": bool(report.experiments),
            "all_experiments_executed": bool(report.experiments)
            and all(item.execution_record_id is not None for item in report.experiments),
            "all_experiments_pass": report.status is ExperimentReportStatus.PASS,
            "no_failed_runs": report.failed_runs == 0,
            "response_summary_recomputed": _response_summary_matches(
                report.baseline_responses, report.response_ranges, report.experiments
            ),
            "no_objective_fabricated": report.baseline_objective is None,
        }
        errors = [f"SENSITIVITY_GATE_FAIL:{name}" for name, passed in checks.items() if not passed]
        errors.extend(report.errors)
        return QualityGateResult(
            gate="SENSITIVITY",
            status=QualityGateStatus.PASS if not errors else QualityGateStatus.RETRY,
            checks=checks,
            errors=list(dict.fromkeys(errors)),
            warnings=report.warnings,
        )
    checks = {
        "experiments_present": bool(report.experiments),
        "all_experiments_executed": bool(report.experiments)
        and all(item.execution_record_id is not None for item in report.experiments),
        "all_experiments_pass": report.status is ExperimentReportStatus.PASS,
        "no_failed_runs": report.failed_runs == 0,
        "summary_computed": report.objective_min is not None and report.objective_max is not None,
    }
    errors = [f"SENSITIVITY_GATE_FAIL:{name}" for name, passed in checks.items() if not passed]
    errors.extend(report.errors)
    status = (
        QualityGateStatus.HUMAN_REVIEW
        if report.status is ExperimentReportStatus.BLOCKED
        else QualityGateStatus.PASS
        if not errors
        else QualityGateStatus.RETRY
    )
    return QualityGateResult(
        gate="SENSITIVITY",
        status=status,
        checks=checks,
        errors=list(dict.fromkeys(errors)),
        warnings=report.warnings,
    )


def robustness_quality_gate(
    report: RobustnessReport,
    reviewed_evidence: ReviewedValidationEvidence | None = None,
) -> QualityGateResult:
    if report.reviewed_report_id is not None:
        reviewed_valid = _reviewed_report_matches(report, reviewed_evidence, parameter_only=False)
        checks = {
            "independent_report_bound": reviewed_valid,
            "all_required_scenarios_covered": bool(report.reviewed_replay_ids),
            "all_required_numeric_responses_verified": bool(report.reviewed_metric_values),
            "no_objective_fabricated": report.baseline_objective is None and not report.experiments,
            "report_pass": report.status is ExperimentReportStatus.PASS,
            "method_explicit": report.method is report.config.method,
        }
        errors = [f"ROBUSTNESS_GATE_FAIL:{name}" for name, passed in checks.items() if not passed]
        errors.extend(report.errors)
        return QualityGateResult(
            gate="ROBUSTNESS",
            status=QualityGateStatus.PASS if not errors else QualityGateStatus.RETRY,
            checks=checks,
            errors=list(dict.fromkeys(errors)),
            warnings=report.warnings,
        )
    if report.baseline_responses:
        checks = {
            "method_explicit": report.method is report.config.method,
            "experiments_present": bool(report.experiments),
            "all_experiments_executed": bool(report.experiments)
            and all(item.execution_record_id is not None for item in report.experiments),
            "all_experiments_pass": report.status is ExperimentReportStatus.PASS,
            "full_feasibility_rate": report.summary.feasibility_rate == 1.0,
            "response_summary_recomputed": _response_summary_matches(
                report.baseline_responses, report.response_ranges, report.experiments
            ),
            "no_objective_fabricated": report.baseline_objective is None,
        }
        errors = [f"ROBUSTNESS_GATE_FAIL:{name}" for name, passed in checks.items() if not passed]
        errors.extend(report.errors)
        return QualityGateResult(
            gate="ROBUSTNESS",
            status=QualityGateStatus.PASS if not errors else QualityGateStatus.RETRY,
            checks=checks,
            errors=list(dict.fromkeys(errors)),
            warnings=report.warnings,
        )
    checks = {
        "method_explicit": report.method is report.config.method,
        "experiments_present": bool(report.experiments),
        "all_experiments_executed": bool(report.experiments)
        and all(item.execution_record_id is not None for item in report.experiments),
        "all_experiments_pass": report.status is ExperimentReportStatus.PASS,
        "full_feasibility_rate": report.summary.feasibility_rate == 1.0,
        "summary_computed": report.summary.objective_mean is not None,
    }
    errors = [f"ROBUSTNESS_GATE_FAIL:{name}" for name, passed in checks.items() if not passed]
    errors.extend(report.errors)
    status = (
        QualityGateStatus.HUMAN_REVIEW
        if report.status is ExperimentReportStatus.BLOCKED
        else QualityGateStatus.PASS
        if not errors
        else QualityGateStatus.RETRY
    )
    return QualityGateResult(
        gate="ROBUSTNESS",
        status=status,
        checks=checks,
        errors=list(dict.fromkeys(errors)),
        warnings=report.warnings,
    )


def red_team_quality_gate(report: RedTeamReport) -> QualityGateResult:
    unresolved = [item for item in report.findings if not item.resolved]
    computed_critical = sum(item.severity is RedTeamSeverity.CRITICAL for item in unresolved)
    computed_major = sum(item.severity is RedTeamSeverity.MAJOR for item in unresolved)
    computed_minor = sum(item.severity is RedTeamSeverity.MINOR for item in unresolved)
    checks = {
        "review_is_non_mock": not report.review_is_mock,
        "severity_counts_recomputed": (
            report.critical_count,
            report.major_count,
            report.minor_count,
        )
        == (computed_critical, computed_major, computed_minor),
        "no_unresolved_critical_findings": computed_critical == 0,
        "report_status_pass": report.status is ValidationStatus.PASS,
    }
    errors = [f"RED_TEAM_GATE_FAIL:{name}" for name, passed in checks.items() if not passed]
    errors.extend(
        f"RED_TEAM_CRITICAL:{item.finding_id}:{item.title}"
        for item in report.findings
        if item.severity is RedTeamSeverity.CRITICAL and not item.resolved
    )
    status = (
        QualityGateStatus.HUMAN_REVIEW
        if report.review_is_mock
        else QualityGateStatus.PASS
        if not errors
        else QualityGateStatus.RETRY
    )
    return QualityGateResult(
        gate="RED_TEAM",
        status=status,
        checks=checks,
        errors=errors,
        warnings=[
            f"{item.severity.value}:{item.finding_id}:{item.title}"
            for item in report.findings
            if item.severity is not RedTeamSeverity.CRITICAL
        ],
    )


def verified_result_quality_gate(
    validation: ValidationReport,
    sensitivity: SensitivityReport,
    robustness: RobustnessReport,
    red_team: RedTeamReport,
    *,
    validation_integrity_errors: list[str],
    sensitivity_integrity_errors: list[str],
    robustness_integrity_errors: list[str],
    reviewed_evidence: ReviewedValidationEvidence | None = None,
) -> QualityGateResult:
    unresolved_critical = sum(
        item.severity is RedTeamSeverity.CRITICAL and not item.resolved
        for item in red_team.findings
    )
    recomputed_counts = (
        unresolved_critical,
        sum(
            item.severity is RedTeamSeverity.MAJOR and not item.resolved
            for item in red_team.findings
        ),
        sum(
            item.severity is RedTeamSeverity.MINOR and not item.resolved
            for item in red_team.findings
        ),
    )
    model_keys = {
        (item.model_id, item.model_version, item.model_digest)
        for item in (validation, sensitivity, robustness, red_team)
    }
    result_ids = {
        validation.result_id,
        sensitivity.result_id,
        robustness.result_id,
        red_team.result_id,
    }
    reviewed = (
        sensitivity.reviewed_report_id is not None or robustness.reviewed_report_id is not None
    )
    reviewed_sensitivity = reviewed and _reviewed_report_matches(
        sensitivity, reviewed_evidence, parameter_only=True
    )
    reviewed_robustness = reviewed and _reviewed_report_matches(
        robustness, reviewed_evidence, parameter_only=False
    )
    response_mode = bool(
        sensitivity.baseline_responses
        and sensitivity.baseline_responses == robustness.baseline_responses
        and sensitivity.baseline_objective is None
        and robustness.baseline_objective is None
        and not reviewed
    )
    response_sensitivity_valid = response_mode and _response_summary_matches(
        sensitivity.baseline_responses, sensitivity.response_ranges, sensitivity.experiments
    )
    response_robustness_valid = response_mode and _response_summary_matches(
        robustness.baseline_responses, robustness.response_ranges, robustness.experiments
    )
    nonobjective_validation = not any(
        item.category.value == "OBJECTIVE" for item in validation.metric_recalculations
    )
    if response_mode:
        baseline_consistent = nonobjective_validation
    elif reviewed:
        baseline_consistent = bool(
            reviewed_sensitivity
            and reviewed_robustness
            and sensitivity.baseline_objective is None
            and robustness.baseline_objective is None
            and nonobjective_validation
        )
    else:
        baseline_consistent = _baseline_objective_matches(validation, sensitivity, robustness)
    sensitivity_experiments_pass = bool(sensitivity.experiments) and all(
        item.status is ExperimentStatus.PASS for item in sensitivity.experiments
    )
    robustness_experiments_pass = bool(robustness.experiments) and all(
        item.status is ExperimentStatus.PASS for item in robustness.experiments
    )
    checks = {
        "validation_pass": validation.status is ValidationStatus.PASS,
        "validation_evidence_valid": validation.evidence.valid,
        "validation_has_no_errors": not validation.errors,
        "validation_variables_pass": bool(validation.variable_checks)
        and all(item.status.value == "PASS" for item in validation.variable_checks),
        "validation_constraints_pass": all(
            item.status.value == "PASS" for item in validation.constraint_checks
        ),
        "validation_metrics_pass": bool(validation.metric_recalculations)
        and all(item.status.value == "PASS" for item in validation.metric_recalculations),
        "validation_requirements_pass": all(
            item.status.value == "PASS" for item in validation.requirement_checks
        ),
        "validation_report_integrity": not validation_integrity_errors,
        "baseline_objective_matches_validation": baseline_consistent,
        "sensitivity_pass": sensitivity.status is ExperimentReportStatus.PASS,
        "sensitivity_experiments_pass": reviewed_sensitivity
        if reviewed
        else sensitivity_experiments_pass,
        "sensitivity_design_matches_config": reviewed_sensitivity
        if reviewed
        else _sensitivity_design_matches(sensitivity),
        "sensitivity_deltas_recomputed": response_sensitivity_valid
        if response_mode
        else reviewed_sensitivity
        if reviewed
        else _objective_deltas_match(
            sensitivity.baseline_objective,
            sensitivity.experiments,
        ),
        "sensitivity_summary_recomputed": response_sensitivity_valid
        if response_mode
        else reviewed_sensitivity
        if reviewed
        else _sensitivity_summary_matches(sensitivity),
        "sensitivity_execution_integrity": not sensitivity_integrity_errors,
        "robustness_pass": robustness.status is ExperimentReportStatus.PASS,
        "robustness_experiments_pass": reviewed_robustness
        if reviewed
        else robustness_experiments_pass,
        "robustness_design_matches_config": reviewed_robustness
        if reviewed
        else _robustness_design_matches(robustness),
        "robustness_deltas_recomputed": response_robustness_valid
        if response_mode
        else reviewed_robustness
        if reviewed
        else _objective_deltas_match(
            robustness.baseline_objective,
            robustness.experiments,
        ),
        "robustness_summary_recomputed": response_robustness_valid
        if response_mode
        else reviewed_robustness
        if reviewed
        else _robustness_summary_matches(robustness),
        "robustness_execution_integrity": not robustness_integrity_errors,
        "red_team_non_mock": not red_team.review_is_mock,
        "red_team_pass": red_team.status is ValidationStatus.PASS,
        "red_team_counts_recomputed": recomputed_counts
        == (red_team.critical_count, red_team.major_count, red_team.minor_count),
        "no_recomputed_critical_findings": unresolved_critical == 0,
        "single_model_revision": len(model_keys) == 1,
        "single_formal_result": len(result_ids) == 1,
        "validation_chain_matches": sensitivity.validation_id == validation.validation_id,
        "robustness_validation_matches": robustness.validation_id == validation.validation_id,
        "sensitivity_chain_matches": robustness.sensitivity_id == sensitivity.sensitivity_id,
        "robustness_chain_matches": red_team.robustness_id == robustness.robustness_id,
        "red_team_validation_matches": red_team.validation_id == validation.validation_id,
        "red_team_sensitivity_matches": red_team.sensitivity_id == sensitivity.sensitivity_id,
    }
    errors = [f"VERIFIED_GATE_FAIL:{name}" for name, passed in checks.items() if not passed]
    errors.extend(f"VALIDATION_INTEGRITY:{item}" for item in validation_integrity_errors)
    errors.extend(f"SENSITIVITY_INTEGRITY:{item}" for item in sensitivity_integrity_errors)
    errors.extend(f"ROBUSTNESS_INTEGRITY:{item}" for item in robustness_integrity_errors)
    human_review = (
        validation.status in {ValidationStatus.NOT_EVALUABLE, ValidationStatus.INCONCLUSIVE}
        or sensitivity.status is ExperimentReportStatus.BLOCKED
        or robustness.status is ExperimentReportStatus.BLOCKED
        or red_team.review_is_mock
    )
    return QualityGateResult(
        gate="VERIFIED",
        subject_ref=f"result:{validation.result_id}",
        status=(
            QualityGateStatus.PASS
            if not errors
            else QualityGateStatus.HUMAN_REVIEW
            if human_review
            else QualityGateStatus.RETRY
        ),
        checks=checks,
        errors=list(dict.fromkeys(errors)),
    )


def _sensitivity_design_matches(report: SensitivityReport) -> bool:
    records = report.experiments
    if not records or any(len(item.perturbations) != 1 for item in records):
        return False
    observed = [
        (item.perturbations[0].parameter_id, item.perturbations[0].fraction) for item in records
    ]
    parameter_ids = list(dict.fromkeys(parameter_id for parameter_id, _ in observed))
    expected = [
        (parameter_id, direction * fraction)
        for parameter_id in parameter_ids
        for fraction in report.config.perturbation_fractions
        for direction in (-1.0, 1.0)
    ][: report.config.max_runs]
    if observed != expected:
        return False
    if report.config.parameter_symbols:
        observed_symbols = {item.perturbations[0].symbol for item in records}
        if observed_symbols != set(report.config.parameter_symbols):
            return False
    return True


def _reviewed_report_matches(
    report: SensitivityReport | RobustnessReport,
    evidence: ReviewedValidationEvidence | None,
    *,
    parameter_only: bool,
) -> bool:
    if evidence is None or report.reviewed_report_id != evidence.report.report_id:
        return False
    try:
        replay_ids, values = reviewed_response_summary(
            evidence,
            model_digest=report.model_digest,
            result_id=report.result_id,
            parameter_only=parameter_only,
        )
    except ValueError:
        return False
    return report.reviewed_replay_ids == replay_ids and report.reviewed_metric_values == values


def _robustness_design_matches(report: RobustnessReport) -> bool:
    records = report.experiments
    if not records:
        return False
    parameter_ids = [item.parameter_id for item in records[0].perturbations]
    if not parameter_ids or any(
        [item.parameter_id for item in record.perturbations] != parameter_ids for record in records
    ):
        return False
    config = report.config
    if config.method.value in {"SCENARIO_ANALYSIS", "WORST_CASE"}:
        expected = [
            [fraction] * len(parameter_ids)
            for fraction in config.scenario_fractions[: config.max_runs]
        ]
    else:
        generator = random.Random(config.random_seed)
        expected = []
        for _ in range(min(config.sample_count, config.max_runs)):
            if config.method.value == "NOISE_PERTURBATION":
                expected.append(
                    [
                        generator.uniform(-config.noise_fraction, config.noise_fraction)
                        for _ in parameter_ids
                    ]
                )
            elif config.method.value == "MONTE_CARLO":
                expected.append(
                    [
                        max(-1.0, min(1.0, generator.gauss(0.0, config.noise_fraction)))
                        for _ in parameter_ids
                    ]
                )
            else:
                return False
    observed = [[item.fraction for item in record.perturbations] for record in records]
    symbols_match = not config.parameter_symbols or {
        item.symbol for item in records[0].perturbations
    } == set(config.parameter_symbols)
    return (
        symbols_match
        and len(observed) == len(expected)
        and all(
            len(left) == len(right)
            and all(
                math.isclose(actual, configured, rel_tol=1e-12, abs_tol=1e-12)
                for actual, configured in zip(left, right, strict=True)
            )
            for left, right in zip(observed, expected, strict=True)
        )
    )


def _baseline_objective_matches(
    validation: ValidationReport,
    sensitivity: SensitivityReport,
    robustness: RobustnessReport,
) -> bool:
    objective_metrics = [
        item for item in validation.metric_recalculations if item.category.value == "OBJECTIVE"
    ]
    if not objective_metrics:
        return False
    expected = sensitivity.baseline_objective
    if expected is None or robustness.baseline_objective is None:
        return False
    if not math.isclose(expected, robustness.baseline_objective, rel_tol=1e-9, abs_tol=1e-9):
        return False
    return all(
        item.recomputed_value is not None
        and math.isclose(expected, item.recomputed_value, rel_tol=1e-9, abs_tol=1e-9)
        for item in objective_metrics
    )


def _objective_deltas_match(baseline: float | None, records: list[ExperimentRun]) -> bool:
    if baseline is None:
        return False
    for record in records:
        if record.objective_value is None or record.objective_change is None:
            return False
        expected_change = record.objective_value - baseline
        if not math.isclose(
            record.objective_change,
            expected_change,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            return False
        expected_relative = expected_change / abs(baseline) if baseline else None
        if expected_relative is None:
            if record.objective_change_fraction is not None:
                return False
        elif record.objective_change_fraction is None or not math.isclose(
            record.objective_change_fraction,
            expected_relative,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            return False
    return True


def _response_summary_matches(
    baseline: dict[str, float],
    ranges: dict[str, tuple[float, float]],
    records: list[ExperimentRun],
) -> bool:
    if not baseline or not records:
        return False
    if any(
        item.solver is not SolverName.SCALAR_RESPONSE
        or item.objective_value is not None
        or item.objective_change is not None
        or item.objective_change_fraction is not None
        or item.fixed_decision_values != records[0].fixed_decision_values
        or set(item.key_outputs) != set(baseline) | set(item.fixed_decision_values)
        for item in records
    ):
        return False
    expected = {
        symbol: (
            min(item.key_outputs[symbol] for item in records),
            max(item.key_outputs[symbol] for item in records),
        )
        for symbol in baseline
    }
    return ranges == expected


def _sensitivity_summary_matches(report: SensitivityReport) -> bool:
    passed = [item for item in report.experiments if item.status is ExperimentStatus.PASS]
    failed = [item for item in report.experiments if item.status is ExperimentStatus.FAIL]
    objectives = [item.objective_value for item in passed if item.objective_value is not None]
    relative = [
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
    expected_elasticities = {
        symbol: sum(values) / len(values) for symbol, values in sorted(elasticities.items())
    }
    return (
        report.successful_runs == len(passed)
        and report.failed_runs == len(failed)
        and report.objective_min == (min(objectives) if objectives else None)
        and report.objective_max == (max(objectives) if objectives else None)
        and report.maximum_absolute_relative_change == (max(relative) if relative else None)
        and report.parameter_elasticities == expected_elasticities
    )


def _robustness_summary_matches(report: RobustnessReport) -> bool:
    passed = [item for item in report.experiments if item.status is ExperimentStatus.PASS]
    failed = [item for item in report.experiments if item.status is ExperimentStatus.FAIL]
    objectives = [item.objective_value for item in passed if item.objective_value is not None]
    ordered = sorted(objectives)
    quantiles = (
        {
            label: ordered[round((len(ordered) - 1) * probability)]
            for label, probability in (("p05", 0.05), ("p50", 0.5), ("p95", 0.95))
        }
        if ordered
        else {}
    )
    summary = report.summary
    expected_rate = len(passed) / len(report.experiments) if report.experiments else None
    return (
        summary.requested_runs == len(report.experiments)
        and summary.successful_runs == len(passed)
        and summary.failed_runs == len(failed)
        and summary.feasibility_rate == expected_rate
        and summary.objective_mean == (statistics.fmean(objectives) if objectives else None)
        and summary.objective_std == (statistics.pstdev(objectives) if objectives else None)
        and summary.objective_min == (min(objectives) if objectives else None)
        and summary.objective_max == (max(objectives) if objectives else None)
        and summary.objective_quantiles == quantiles
    )


def repair_quality_gate(
    output: ModelRepairOutput,
    report: RedTeamReport,
    *,
    is_mock: bool,
) -> QualityGateResult:
    critical_ids = {
        item.finding_id
        for item in report.findings
        if item.severity is RedTeamSeverity.CRITICAL and not item.resolved
    }
    addressed = set(output.addressed_finding_ids)
    action_findings = {finding for action in output.actions for finding in action.finding_ids}
    checks = {
        "stable_model_identity": output.revised_model.model_id == report.model_id,
        "project_identity": output.revised_model.project_id == report.project_id,
        "problem_identity": output.revised_model.problem_id == report.problem_id,
        "version_incremented": output.revised_model.version == report.model_version + 1,
        "mathematical_content_changed": (
            mathematical_model_digest(output.revised_model) != report.model_digest
        ),
        "all_critical_findings_addressed": critical_ids <= addressed,
        "actions_reference_addressed_findings": addressed <= action_findings,
        "repair_is_non_mock": not is_mock,
    }
    errors = [f"MODEL_REPAIR_GATE_FAIL:{name}" for name, passed in checks.items() if not passed]
    return QualityGateResult(
        gate="MODEL_REPAIR",
        status=(
            QualityGateStatus.HUMAN_REVIEW
            if is_mock
            else QualityGateStatus.PASS
            if not errors
            else QualityGateStatus.RETRY
        ),
        checks=checks,
        errors=errors,
        warnings=output.remaining_risks,
    )
