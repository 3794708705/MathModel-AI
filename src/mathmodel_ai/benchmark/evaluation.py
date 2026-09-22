from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from uuid import UUID

from mathmodel_ai.schemas.benchmark import (
    BenchmarkAcceptance,
    BenchmarkAttempt,
    BenchmarkCaseResult,
    BenchmarkCaseStatus,
    BenchmarkDimensionScore,
    BenchmarkFailure,
    BenchmarkHumanIntervention,
    BenchmarkMetric,
    BenchmarkRun,
    BenchmarkRunStatus,
    FailureSeverity,
    LiveValidationStatus,
    benchmark_result_digest,
)


@dataclass(frozen=True)
class ScoreDimension:
    name: str
    maximum: float
    metric_weights: tuple[tuple[str, float], ...]


SCORECARD = (
    ScoreDimension(
        "Problem Understanding",
        10,
        (("problem_understanding_accuracy", 0.5), ("critical_constraint_recall", 0.5)),
    ),
    ScoreDimension("Requirement Coverage", 10, (("subproblem_coverage", 1.0),)),
    ScoreDimension("Data Handling", 10, (("data_extraction_accuracy", 1.0),)),
    ScoreDimension("Model Appropriateness", 15, (("model_appropriateness", 1.0),)),
    ScoreDimension("Mathematical Correctness", 15, (("mathematical_validity", 1.0),)),
    ScoreDimension("Computation / Solver", 10, (("solver_success", 1.0),)),
    ScoreDimension(
        "Validation / Robustness",
        10,
        (
            ("validation_pass", 0.5),
            ("sensitivity_completion", 0.25),
            ("robustness_completion", 0.25),
        ),
    ),
    ScoreDimension("Paper Quality", 10, (("paper_factual_consistency", 1.0),)),
    ScoreDimension(
        "Citation Integrity",
        5,
        (("citation_validity", 0.5), ("citation_support_accuracy", 0.5)),
    ),
    ScoreDimension(
        "Competition / Submission",
        5,
        (("competition_compliance", 0.5), ("submission_completeness", 0.5)),
    ),
)

HARD_FAILURE_METRICS: dict[str, tuple[str, bool]] = {
    "major_unanswered_subproblem_count": ("major unanswered subproblem", False),
    "central_model_valid": ("central mathematical model is invalid", True),
    "fabricated_result_count": ("fabricated final result", False),
    "unverified_central_result_count": ("unverified central result", False),
    "fabricated_critical_citation_count": ("fabricated critical citation", False),
    "wrong_submission_artifact_count": ("wrong submission artifact", False),
    "blocking_competition_violation_count": ("blocking competition violation", False),
    "secret_leak_count": ("secret leak", False),
}

SUCCESS_STATUSES = {BenchmarkCaseStatus.PASS, BenchmarkCaseStatus.PASS_WITH_WARNINGS}
PRESERVED_TERMINAL_STATUSES = {
    BenchmarkCaseStatus.BLOCKED_ENVIRONMENT,
    BenchmarkCaseStatus.BLOCKED_RULE_POLICY,
    BenchmarkCaseStatus.BUDGET_EXCEEDED,
    BenchmarkCaseStatus.INVALID_BENCHMARK_SETUP,
    BenchmarkCaseStatus.CANCELLED_BY_HUMAN,
    BenchmarkCaseStatus.HUMAN_REVIEW,
}


class BenchmarkEvaluator:
    """Recompute scores and gates from atomic records, never persisted totals."""

    def evaluate_attempt(
        self,
        attempt: BenchmarkAttempt,
        metrics: list[BenchmarkMetric],
        failures: list[BenchmarkFailure],
        interventions: list[BenchmarkHumanIntervention],
        *,
        claimed: BenchmarkCaseResult | None = None,
        requires_literature: bool = True,
    ) -> BenchmarkCaseResult:
        self._same_attempt(attempt, metrics, failures, interventions)
        if claimed is not None and (
            claimed.attempt_id != attempt.attempt_id or claimed.benchmark_id != attempt.benchmark_id
        ):
            raise ValueError("persisted benchmark summary crosses case identity")
        values: dict[str, BenchmarkMetric] = {}
        duplicate_metrics: list[str] = []
        for item in metrics:
            if item.name in values:
                duplicate_metrics.append(item.name)
            values[item.name] = item

        hard_failures = [f"duplicate metric: {name}" for name in sorted(duplicate_metrics)]
        dimensions: list[BenchmarkDimensionScore] = []
        for dimension in SCORECARD:
            score = 0.0
            evidence: list[str] = []
            for metric_name, weight in dimension.metric_weights:
                metric = values.get(metric_name)
                if metric is None:
                    hard_failures.append(f"missing deterministic metric: {metric_name}")
                    evidence.append(f"MISSING:{metric_name}")
                    continue
                if not metric.deterministic:
                    hard_failures.append(f"non-deterministic score metric: {metric_name}")
                if not 0 <= metric.value <= 1:
                    hard_failures.append(f"score metric outside [0,1]: {metric_name}")
                    normalized = 0.0
                else:
                    normalized = metric.value
                score += normalized * weight * dimension.maximum
                evidence.append(metric.evidence_ref)
            dimensions.append(
                BenchmarkDimensionScore(
                    dimension=dimension.name,
                    score=round(score, 6),
                    maximum=dimension.maximum,
                    evidence_refs=evidence,
                )
            )

        for metric_name, (message, one_means_pass) in HARD_FAILURE_METRICS.items():
            metric = values.get(metric_name)
            if metric is None:
                hard_failures.append(f"missing hard-fail check: {metric_name}")
                continue
            failed = metric.value != 1 if one_means_pass else metric.value != 0
            if failed:
                hard_failures.append(message)

        if any(
            failure.severity in {FailureSeverity.P0, FailureSeverity.P1} for failure in failures
        ):
            hard_failures.append("P0/P1 benchmark failure remains unresolved")
        if not attempt.provider_is_live:
            hard_failures.append("critical agent chain did not use a live provider")
        if requires_literature and not attempt.literature_is_live:
            hard_failures.append("required literature chain was not live")

        score = round(sum(item.score for item in dimensions), 6)
        status = self._status(attempt.status, score, hard_failures, interventions)
        links = claimed or self._empty_claim(attempt)
        if status in SUCCESS_STATUSES:
            if links.verified_result_id is None:
                hard_failures.append("successful case lacks verified_result_id")
            if links.paper_id is None or links.paper_version is None:
                hard_failures.append("successful case lacks a final-ready paper")
            if links.submission_id is None or links.submission_status not in {
                "FROZEN",
                "TECHNICALLY_FROZEN",
            }:
                hard_failures.append("successful case lacks an integrity-checked package")
            status = self._status(attempt.status, score, hard_failures, interventions)

        result = BenchmarkCaseResult(
            result_id=links.result_id,
            attempt_id=attempt.attempt_id,
            benchmark_id=attempt.benchmark_id,
            status=status,
            dimensions=dimensions,
            score=score,
            hard_failures=sorted(set(hard_failures)),
            verified_result_id=links.verified_result_id,
            paper_id=links.paper_id,
            paper_version=links.paper_version,
            submission_id=links.submission_id,
            submission_status=links.submission_status,
            artifact_refs=list(links.artifact_refs),
            human_intervention_count=len(interventions),
            provider_call_count=self._count(values, "provider_calls"),
            total_tokens=self._count(values, "total_tokens"),
            estimated_cost=self._number(values, "estimated_cost"),
            wall_time_seconds=self._number(values, "wall_time_seconds"),
            result_digest="0" * 64,
            created_at=links.created_at,
        )
        return result.model_copy(update={"result_digest": benchmark_result_digest(result)})

    def acceptance(
        self,
        run: BenchmarkRun,
        attempts: list[BenchmarkAttempt],
        results: list[BenchmarkCaseResult],
        metrics: list[BenchmarkMetric],
        failures: list[BenchmarkFailure],
    ) -> BenchmarkAcceptance:
        official = [item for item in attempts if item.official]
        result_by_attempt = {item.attempt_id: item for item in results}
        latest: dict[str, BenchmarkAttempt] = {}
        for attempt in official:
            previous = latest.get(attempt.benchmark_id)
            if previous is None or attempt.attempt_number > previous.attempt_number:
                latest[attempt.benchmark_id] = attempt
        successful = sum(
            result_by_attempt.get(item.attempt_id) is not None
            and result_by_attempt[item.attempt_id].status in SUCCESS_STATUSES
            for item in latest.values()
        )
        provider_status = self._live_provider_status(attempts, metrics, results)
        literature_status = self._live_literature_status(attempts, metrics)
        reasons: list[str] = []
        if len(latest) < 3:
            reasons.append("fewer than three real competition cases were attempted")
        if successful < 2:
            reasons.append("fewer than two cases reached PASS or PASS_WITH_WARNINGS")
        profiles = len(run.competition_profile_digests)
        if profiles < 1:
            reasons.append("no VERIFIED real competition profile is bound to the run")
        if provider_status is not LiveValidationStatus.PASS:
            reasons.append("live provider critical chain did not pass")
        if literature_status is not LiveValidationStatus.PASS:
            reasons.append("no case completed a verified live literature lookup")
        p0_count = sum(item.severity is FailureSeverity.P0 for item in failures)
        if p0_count:
            reasons.append(f"{p0_count} P0 failure(s) are present and fully reported")
        if any(
            item.status in {BenchmarkCaseStatus.PENDING, BenchmarkCaseStatus.RUNNING}
            for item in attempts
        ):
            reasons.append("one or more formal attempts remain unfinished")
        if len(result_by_attempt) != len(attempts):
            reasons.append("one or more formal attempts lacks a deterministic case result")
        return BenchmarkAcceptance(
            status=(
                BenchmarkRunStatus.SELF_TEST_READY if not reasons else BenchmarkRunStatus.NOT_READY
            ),
            real_cases_attempted=len(latest),
            successful_cases=successful,
            verified_real_profiles=profiles,
            live_provider_status=provider_status,
            live_literature_status=literature_status,
            hidden_p0_count=0,
            reasons=reasons,
        )

    @staticmethod
    def _status(
        claimed: BenchmarkCaseStatus,
        score: float,
        hard_failures: list[str],
        interventions: list[BenchmarkHumanIntervention],
    ) -> BenchmarkCaseStatus:
        if claimed in PRESERVED_TERMINAL_STATUSES:
            return claimed
        if hard_failures or score < 60:
            return BenchmarkCaseStatus.FAIL
        if score < 75 or interventions:
            return BenchmarkCaseStatus.PASS_WITH_WARNINGS
        return BenchmarkCaseStatus.PASS

    @staticmethod
    def _same_attempt(
        attempt: BenchmarkAttempt,
        metrics: list[BenchmarkMetric],
        failures: list[BenchmarkFailure],
        interventions: list[BenchmarkHumanIntervention],
    ) -> None:
        identities = {
            *(item.attempt_id for item in metrics),
            *(item.attempt_id for item in failures),
            *(item.attempt_id for item in interventions),
        }
        if identities and identities != {attempt.attempt_id}:
            raise ValueError("benchmark atomic records cross attempt identity")

    @staticmethod
    def _number(values: dict[str, BenchmarkMetric], name: str) -> float:
        metric = values.get(name)
        return metric.value if metric is not None and metric.value >= 0 else 0

    @classmethod
    def _count(cls, values: dict[str, BenchmarkMetric], name: str) -> int:
        return int(cls._number(values, name))

    @staticmethod
    def _empty_claim(attempt: BenchmarkAttempt) -> BenchmarkCaseResult:
        return BenchmarkCaseResult(
            attempt_id=attempt.attempt_id,
            benchmark_id=attempt.benchmark_id,
            status=attempt.status,
            dimensions=[
                BenchmarkDimensionScore(
                    dimension="unscored",
                    score=0,
                    maximum=100,
                    evidence_refs=["benchmark evaluator"],
                )
            ],
            score=0,
            human_intervention_count=0,
            provider_call_count=0,
            total_tokens=0,
            estimated_cost=0,
            wall_time_seconds=0,
            result_digest="0" * 64,
        )

    @staticmethod
    def _live_provider_status(
        attempts: list[BenchmarkAttempt],
        metrics: list[BenchmarkMetric],
        results: list[BenchmarkCaseResult],
    ) -> LiveValidationStatus:
        result_by_attempt = {item.attempt_id: item for item in results}
        coverage = {
            item.attempt_id: item.value
            for item in metrics
            if item.name == "live_provider_critical_agent_coverage" and item.deterministic
        }
        if any(
            item.provider_is_live
            and coverage.get(item.attempt_id) == 1
            and result_by_attempt.get(item.attempt_id) is not None
            for item in attempts
        ):
            return LiveValidationStatus.PASS
        if any(item.provider_is_live for item in attempts):
            return LiveValidationStatus.FAIL
        return LiveValidationStatus.BLOCKED

    @staticmethod
    def _live_literature_status(
        attempts: list[BenchmarkAttempt], metrics: list[BenchmarkMetric]
    ) -> LiveValidationStatus:
        verified_counts: dict[UUID, float] = defaultdict(float)
        for metric in metrics:
            if metric.name == "live_literature_verified_reference_count" and metric.deterministic:
                verified_counts[metric.attempt_id] += metric.value
        if any(
            item.literature_is_live and verified_counts[item.attempt_id] >= 1 for item in attempts
        ):
            return LiveValidationStatus.PASS
        if any(item.literature_is_live for item in attempts):
            return LiveValidationStatus.FAIL
        return LiveValidationStatus.BLOCKED
