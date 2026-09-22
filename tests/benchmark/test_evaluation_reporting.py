from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from mathmodel_ai.benchmark.evaluation import BenchmarkEvaluator
from mathmodel_ai.benchmark.reporting import BenchmarkReportBuilder
from mathmodel_ai.schemas.benchmark import (
    BenchmarkAttempt,
    BenchmarkBudget,
    BenchmarkCaseResult,
    BenchmarkCaseStatus,
    BenchmarkConfig,
    BenchmarkDimensionScore,
    BenchmarkFailure,
    BenchmarkHumanIntervention,
    BenchmarkMetric,
    BenchmarkMetricKind,
    BenchmarkPricing,
    BenchmarkRun,
    BenchmarkRunStatus,
    FailureCategory,
    FailureSeverity,
    HumanInterventionType,
    LiveValidationStatus,
    benchmark_attempt_digest,
    benchmark_failure_digest,
    benchmark_intervention_digest,
    benchmark_metric_digest,
    benchmark_result_digest,
    benchmark_run_digest,
)

ZERO = "0" * 64


def test_evaluator_recalculates_tampered_summary_fields_from_atomic_metrics() -> None:
    attempt = _attempt("BENCH-case-a", provider_live=True, literature_live=True)
    metrics = _passing_metrics(attempt.attempt_id)
    claimed = _claimed_result(attempt, score=3, status=BenchmarkCaseStatus.FAIL)
    claimed = claimed.model_copy(
        update={
            "human_intervention_count": 999,
            "provider_call_count": 999,
            "total_tokens": 999,
            "estimated_cost": 999,
        }
    )

    result = BenchmarkEvaluator().evaluate_attempt(attempt, metrics, [], [], claimed=claimed)

    assert result.status is BenchmarkCaseStatus.PASS
    assert result.score == 100
    assert result.human_intervention_count == 0
    assert result.provider_call_count == 6
    assert result.total_tokens == 1200
    assert result.estimated_cost == 1.25
    assert result.result_digest == benchmark_result_digest(result)


def test_p0_failure_overrides_perfect_score() -> None:
    attempt = _attempt("BENCH-case-a", provider_live=True, literature_live=True)
    failure = _failure(attempt.attempt_id, FailureSeverity.P0, "central result is wrong")

    result = BenchmarkEvaluator().evaluate_attempt(
        attempt,
        _passing_metrics(attempt.attempt_id),
        [failure],
        [],
        claimed=_claimed_result(attempt),
    )

    assert result.score == 100
    assert result.status is BenchmarkCaseStatus.FAIL
    assert "P0/P1 benchmark failure remains unresolved" in result.hard_failures


def test_non_live_provider_can_never_become_pass() -> None:
    attempt = _attempt("BENCH-case-a", provider_live=False, literature_live=True)

    result = BenchmarkEvaluator().evaluate_attempt(
        attempt,
        _passing_metrics(attempt.attempt_id),
        [],
        [],
        claimed=_claimed_result(attempt),
    )

    assert result.status is BenchmarkCaseStatus.FAIL
    assert "critical agent chain did not use a live provider" in result.hard_failures


def test_human_intervention_count_is_recomputed_and_cannot_be_hidden() -> None:
    attempt = _attempt("BENCH-case-a", provider_live=True, literature_live=True)
    intervention = BenchmarkHumanIntervention(
        attempt_id=attempt.attempt_id,
        intervention_type=HumanInterventionType.MODEL_CORRECTION,
        reason="human corrected the selected model",
        actor="benchmark reviewer",
        intervention_digest=ZERO,
    )
    intervention = intervention.model_copy(
        update={"intervention_digest": benchmark_intervention_digest(intervention)}
    )

    result = BenchmarkEvaluator().evaluate_attempt(
        attempt,
        _passing_metrics(attempt.attempt_id),
        [],
        [intervention],
        claimed=_claimed_result(attempt),
    )

    assert result.human_intervention_count == 1
    assert result.status is BenchmarkCaseStatus.PASS_WITH_WARNINGS


def test_acceptance_uses_latest_attempt_but_reports_every_formal_attempt() -> None:
    run = _run()
    attempts = [
        _attempt(
            "BENCH-case-a",
            run_id=run.run_id,
            number=1,
            status=BenchmarkCaseStatus.FAIL,
        ),
        _attempt("BENCH-case-a", run_id=run.run_id, number=2),
        _attempt("BENCH-case-b", run_id=run.run_id),
        _attempt("BENCH-case-c", run_id=run.run_id, status=BenchmarkCaseStatus.FAIL),
    ]
    all_metrics = [metric for item in attempts for metric in _passing_metrics(item.attempt_id)]
    claimed = [_claimed_result(item) for item in attempts]
    failures = [
        _failure(attempts[0].attempt_id, FailureSeverity.P1, "initial attempt failed"),
        _failure(attempts[3].attempt_id, FailureSeverity.P1, "case remains failed"),
    ]

    report = BenchmarkReportBuilder().build(
        run=run,
        attempts=attempts,
        claimed_results=claimed,
        metrics=all_metrics,
        failures=failures,
        interventions=[],
        literature_required={item.benchmark_id: True for item in attempts},
    )

    assert len(report.attempts) == 4
    assert {
        item.attempt_number for item in report.attempts if item.benchmark_id == "BENCH-case-a"
    } == {
        1,
        2,
    }
    assert report.acceptance.real_cases_attempted == 3
    # Legacy aggregate metrics alone no longer establish acceptance.
    assert report.acceptance.successful_cases == 0
    assert report.acceptance.live_provider_status is LiveValidationStatus.PASS
    assert report.acceptance.live_literature_status is LiveValidationStatus.PASS
    assert report.acceptance.status is BenchmarkRunStatus.NOT_READY
    assert all(
        any("independent metric/scenario" in reason for reason in item.hard_failures)
        for item in report.results
    )


def test_report_rejects_local_paths_and_credential_material() -> None:
    run = _run()
    attempt = _attempt("BENCH-case-a", run_id=run.run_id)
    failure = _failure(attempt.attempt_id, FailureSeverity.P2, r"C:\private\prompt.txt")

    with pytest.raises(ValueError, match="local-path or credential"):
        BenchmarkReportBuilder().build(
            run=run,
            attempts=[attempt],
            claimed_results=[_claimed_result(attempt)],
            metrics=_passing_metrics(attempt.attempt_id),
            failures=[failure],
            interventions=[],
        )


def test_report_keeps_an_active_run_non_terminal() -> None:
    terminal = _run()
    running = terminal.model_copy(
        update={
            "status": BenchmarkRunStatus.RUNNING,
            "live_provider_status": LiveValidationStatus.NOT_RUN,
            "live_literature_status": LiveValidationStatus.NOT_RUN,
            "finished_at": None,
        }
    )

    report = BenchmarkReportBuilder().build(
        run=running,
        attempts=[],
        claimed_results=[],
        metrics=[],
        failures=[],
        interventions=[],
    )

    assert report.run.status is BenchmarkRunStatus.RUNNING
    assert report.run.finished_at is None
    assert report.acceptance.status is BenchmarkRunStatus.NOT_READY


def _run() -> BenchmarkRun:
    run = BenchmarkRun(
        code_commit="a" * 40,
        source_tree_digest="f" * 64,
        working_tree_dirty=True,
        config=BenchmarkConfig(
            provider="openai",
            model="live-model",
            reasoning_tier="high",
            pricing=BenchmarkPricing(version="test"),
            budget=BenchmarkBudget(),
        ),
        case_manifest_digests={"set": "b" * 64},
        competition_profile_digests={"profile": "c" * 64},
        status=BenchmarkRunStatus.NOT_READY,
        live_provider_status=LiveValidationStatus.BLOCKED,
        live_literature_status=LiveValidationStatus.BLOCKED,
        finished_at=datetime.now(UTC),
        run_digest=ZERO,
    )
    return run.model_copy(update={"run_digest": benchmark_run_digest(run)})


def _attempt(
    benchmark_id: str,
    *,
    run_id: UUID | None = None,
    number: int = 1,
    status: BenchmarkCaseStatus = BenchmarkCaseStatus.PASS,
    provider_live: bool = True,
    literature_live: bool = True,
) -> BenchmarkAttempt:
    attempt = BenchmarkAttempt(
        run_id=run_id or uuid4(),
        benchmark_id=benchmark_id,
        manifest_digest="d" * 64,
        attempt_number=number,
        status=status,
        solve_input_digest="e" * 64,
        provider_is_live=provider_live,
        literature_is_live=literature_live,
        finished_at=datetime.now(UTC),
        attempt_digest=ZERO,
    )
    return attempt.model_copy(update={"attempt_digest": benchmark_attempt_digest(attempt)})


def _passing_metrics(attempt_id: UUID) -> list[BenchmarkMetric]:
    quality_names = {
        "problem_understanding_accuracy",
        "critical_constraint_recall",
        "subproblem_coverage",
        "data_extraction_accuracy",
        "model_appropriateness",
        "mathematical_validity",
        "solver_success",
        "validation_pass",
        "sensitivity_completion",
        "robustness_completion",
        "paper_factual_consistency",
        "citation_validity",
        "citation_support_accuracy",
        "competition_compliance",
        "submission_completeness",
        "central_model_valid",
        "live_provider_critical_agent_coverage",
    }
    zero_names = {
        "major_unanswered_subproblem_count",
        "fabricated_result_count",
        "unverified_central_result_count",
        "fabricated_critical_citation_count",
        "wrong_submission_artifact_count",
        "blocking_competition_violation_count",
        "secret_leak_count",
    }
    metrics = [_metric(attempt_id, name, 1) for name in quality_names]
    metrics.extend(_metric(attempt_id, name, 0) for name in zero_names)
    metrics.extend(
        [
            _metric(attempt_id, "live_literature_verified_reference_count", 1),
            _metric(attempt_id, "provider_calls", 6),
            _metric(attempt_id, "total_tokens", 1200),
            _metric(attempt_id, "estimated_cost", 1.25),
            _metric(attempt_id, "wall_time_seconds", 30),
        ]
    )
    return metrics


def _metric(attempt_id: UUID, name: str, value: float) -> BenchmarkMetric:
    metric = BenchmarkMetric(
        attempt_id=attempt_id,
        name=name,
        kind=BenchmarkMetricKind.QUALITY,
        value=value,
        unit="count" if value > 1 else "ratio",
        evidence_ref=f"evidence:{name}",
        deterministic=True,
        metric_digest=ZERO,
    )
    return metric.model_copy(update={"metric_digest": benchmark_metric_digest(metric)})


def _claimed_result(
    attempt: BenchmarkAttempt,
    *,
    score: float = 100,
    status: BenchmarkCaseStatus | None = None,
) -> BenchmarkCaseResult:
    result = BenchmarkCaseResult(
        attempt_id=attempt.attempt_id,
        benchmark_id=attempt.benchmark_id,
        status=status or attempt.status,
        dimensions=[
            BenchmarkDimensionScore(
                dimension="claimed",
                score=score,
                maximum=100,
                evidence_refs=["persisted summary"],
            )
        ],
        score=score,
        verified_result_id=uuid4(),
        paper_id=uuid4(),
        paper_version=1,
        submission_id=uuid4(),
        submission_status="FROZEN",
        human_intervention_count=0,
        provider_call_count=0,
        total_tokens=0,
        estimated_cost=0,
        wall_time_seconds=0,
        result_digest=ZERO,
    )
    return result.model_copy(update={"result_digest": benchmark_result_digest(result)})


def _failure(attempt_id: UUID, severity: FailureSeverity, cause: str) -> BenchmarkFailure:
    failure = BenchmarkFailure(
        attempt_id=attempt_id,
        stage="EVALUATION",
        category=FailureCategory.VALIDATION,
        severity=severity,
        root_cause=cause,
        evidence_refs=["evidence:failure"],
        reproducible=True,
        generic_issue=True,
        proposed_fix="fix the generic validator",
        failure_digest=ZERO,
    )
    return failure.model_copy(update={"failure_digest": benchmark_failure_digest(failure)})
