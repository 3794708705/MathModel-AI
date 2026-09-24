from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from uuid import UUID

from mathmodel_ai.benchmark.evaluation import SUCCESS_STATUSES, BenchmarkEvaluator
from mathmodel_ai.schemas.benchmark import (
    BenchmarkAttempt,
    BenchmarkCaseResult,
    BenchmarkCaseStatus,
    BenchmarkFailure,
    BenchmarkHumanIntervention,
    BenchmarkMetric,
    BenchmarkReport,
    BenchmarkRun,
    benchmark_report_digest,
    benchmark_result_digest,
)
from mathmodel_ai.schemas.independent_verification import (
    IndependentStatus,
    IndependentVerificationView,
)

_PRIVATE_PATH = re.compile(r"(?:[A-Za-z]:\\|/home/|/Users/|/var/run/)")
_SECRET_TEXT = re.compile(
    r"(?:sk-[A-Za-z0-9_-]{12,}|AIza[A-Za-z0-9_-]{12,}|(?:API_KEY|PASSWORD|SECRET)\s*=)",
    re.IGNORECASE,
)


class BenchmarkReportBuilder:
    def __init__(
        self,
        evaluator: BenchmarkEvaluator | None = None,
        *,
        independent_verifier: Callable[[UUID], IndependentVerificationView] | None = None,
    ) -> None:
        self._evaluator = evaluator or BenchmarkEvaluator()
        self._independent_verifier = independent_verifier

    def build(
        self,
        *,
        run: BenchmarkRun,
        attempts: list[BenchmarkAttempt],
        claimed_results: list[BenchmarkCaseResult],
        metrics: list[BenchmarkMetric],
        failures: list[BenchmarkFailure],
        interventions: list[BenchmarkHumanIntervention],
        literature_required: dict[str, bool] | None = None,
    ) -> BenchmarkReport:
        if any(item.run_id != run.run_id for item in attempts):
            raise ValueError("benchmark report attempts cross run identity")
        claimed = self._unique_results(claimed_results)
        by_metrics = self._group(metrics)
        by_failures = self._group(failures)
        by_interventions = self._group(interventions)
        results = [
            self._evaluator.evaluate_attempt(
                attempt,
                by_metrics[attempt.attempt_id],
                by_failures[attempt.attempt_id],
                by_interventions[attempt.attempt_id],
                claimed=claimed.get(attempt.attempt_id),
                requires_literature=(literature_required or {}).get(attempt.benchmark_id, True),
            )
            for attempt in attempts
        ]
        results = [self._independent_gate(result) for result in results]
        acceptance = self._evaluator.acceptance(run, attempts, results, metrics, failures)
        normalized_run = (
            run.model_copy(
                update={
                    "status": acceptance.status,
                    "live_provider_status": acceptance.live_provider_status,
                    "live_literature_status": acceptance.live_literature_status,
                }
            )
            if run.finished_at is not None
            else run
        )
        report = BenchmarkReport(
            run=normalized_run,
            attempts=attempts,
            results=results,
            metrics=metrics,
            failures=failures,
            human_interventions=interventions,
            acceptance=acceptance,
            report_digest="0" * 64,
        )
        report = report.model_copy(update={"report_digest": benchmark_report_digest(report)})
        assert_public_report(report)
        return report

    def _independent_gate(self, result: BenchmarkCaseResult) -> BenchmarkCaseResult:
        view = self._independent_verifier(result.attempt_id) if self._independent_verifier else None
        report = view.report if view is not None else None
        if (
            view is not None
            and report is not None
            and view.status is IndependentStatus.PASS
            and report.status is IndependentStatus.PASS
            and not report.errors
            and report.attempt_id == result.attempt_id
            and report.result_id == result.verified_result_id
            and report.required_metrics > 0
            and report.passed_metrics == report.required_metrics
            and len(report.metrics) >= report.required_metrics
            and report.required_scenarios > 0
            and report.passed_scenarios == report.required_scenarios
            and len(report.replays) >= report.required_scenarios
        ):
            return result
        updated = result.model_copy(
            update={
                "status": BenchmarkCaseStatus.FAIL
                if result.status in SUCCESS_STATUSES
                else result.status,
                "hard_failures": [
                    *result.hard_failures,
                    "independent metric/scenario evidence missing, invalid, "
                    "or bound to another result",
                ],
            }
        )
        return updated.model_copy(update={"result_digest": benchmark_result_digest(updated)})

    @staticmethod
    def _group[T: BenchmarkMetric | BenchmarkFailure | BenchmarkHumanIntervention](
        items: list[T],
    ) -> defaultdict[object, list[T]]:
        grouped: defaultdict[object, list[T]] = defaultdict(list)
        for item in items:
            grouped[item.attempt_id].append(item)
        return grouped

    @staticmethod
    def _unique_results(
        results: list[BenchmarkCaseResult],
    ) -> dict[object, BenchmarkCaseResult]:
        by_attempt: dict[object, BenchmarkCaseResult] = {}
        for result in results:
            if result.attempt_id in by_attempt:
                raise ValueError("multiple persisted summaries exist for one benchmark attempt")
            by_attempt[result.attempt_id] = result
        return by_attempt


class BenchmarkReportRenderer:
    def render_json(self, report: BenchmarkReport) -> str:
        assert_public_report(report)
        return (
            json.dumps(
                report.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            + "\n"
        )

    def render_markdown(self, report: BenchmarkReport) -> str:
        assert_public_report(report)
        result_by_attempt = {item.attempt_id: item for item in report.results}
        metric_by_attempt = {
            (item.attempt_id, item.name): item for item in report.metrics
        }
        lines = [
            "# MathModel AI Phase 8 Benchmark Report",
            "",
            f"Status: `{report.acceptance.status.value}`",
            "",
            "## Benchmark Environment",
            "",
            f"- Commit: `{report.run.code_commit}`",
            f"- Source tree digest: `{report.run.source_tree_digest}`",
            f"- Working tree dirty: `{report.run.working_tree_dirty}`",
            f"- Provider: `{report.run.config.provider}`",
            f"- Model: `{report.run.config.model}`",
            f"- Reasoning tier: `{report.run.config.reasoning_tier}`",
            f"- Configuration version: `{report.run.config.config_version}`",
            f"- Random seed: `{report.run.config.random_seed}`",
            "- Live Provider readiness (critical-agent coverage): "
            f"`{report.acceptance.live_provider_status.value}`",
            f"- Live Literature: `{report.acceptance.live_literature_status.value}`",
            "",
            "## Cases",
            "",
            "| Case | Attempt | Status | Score | Human interventions | Runtime (s) | Cost |",
            "|---|---:|---|---:|---:|---:|---:|",
        ]
        for attempt in report.attempts:
            result = result_by_attempt[attempt.attempt_id]
            cost_metric = metric_by_attempt.get((attempt.attempt_id, "estimated_cost"))
            cost_display = (
                "UNKNOWN"
                if cost_metric is not None
                and cost_metric.evidence_ref.startswith("NOT_EVALUATED:")
                else f"{result.estimated_cost:.6f}"
            )
            lines.append(
                f"| {attempt.benchmark_id} | {attempt.attempt_number} | "
                f"{result.status.value} | {result.score:.2f} | "
                f"{result.human_intervention_count} | {result.wall_time_seconds:.3f} | "
                f"{cost_display} |"
            )
        lines.extend(["", "## Per-case Evidence", ""])
        for attempt in report.attempts:
            result = result_by_attempt[attempt.attempt_id]
            lines.extend(
                [
                    f"### {attempt.benchmark_id} / attempt {attempt.attempt_number}",
                    "",
                    f"- Solve input digest: `{attempt.solve_input_digest}`",
                    f"- Provider connectivity/live activity: `{attempt.provider_is_live}`",
                    f"- Literature live: `{attempt.literature_is_live}`",
                    f"- Verified result: `{result.verified_result_id or 'NONE'}`",
                    f"- Paper: `{result.paper_id or 'NONE'}` / `{result.paper_version or 'NONE'}`",
                    f"- Submission: `{result.submission_status or 'NONE'}`",
                    "",
                    "Dimension scores:",
                    "",
                ]
            )
            lines.extend(
                f"- {item.dimension}: {item.score:.2f}/{item.maximum:.2f}"
                for item in result.dimensions
            )
            lines.extend([
                "", "Diagnostic metric labels `NOT_EVALUATED` and `UPSTREAM_BLOCKED` "
                "are not observed failures; their numeric zero is only a fail-closed "
                "readiness-scoring placeholder.", "", "Hard failures:", "",
            ])
            lines.extend(f"- {item}" for item in result.hard_failures)
            if not result.hard_failures:
                lines.append("- None")
            lines.append("")
        lines.extend(["## Failures", ""])
        for failure in report.failures:
            lines.append(
                f"- `{failure.severity.value}` `{failure.category.value}` at "
                f"`{failure.stage}`: {failure.root_cause}"
            )
        if not report.failures:
            lines.append("- None")
        lines.extend(["", "## Human Interventions", ""])
        for intervention in report.human_interventions:
            lines.append(
                f"- `{intervention.intervention_type.value}`: {intervention.reason} "
                f"(actor: {intervention.actor})"
            )
        if not report.human_interventions:
            lines.append("- None")
        lines.extend(["", "## Atomic Metrics", ""])
        for metric in report.metrics:
            lines.append(
                f"- `{metric.name}` = {metric.value:g} {metric.unit} "
                f"(evidence: `{metric.evidence_ref}`)"
            )
        if not report.metrics:
            lines.append("- None")
        lines.extend(["", "## Remaining Blockers", ""])
        lines.extend(f"- {reason}" for reason in report.acceptance.reasons)
        if not report.acceptance.reasons:
            lines.append("- None")
        lines.extend(["", f"Report digest: `{report.report_digest}`", ""])
        return "\n".join(lines)

    def write(self, report: BenchmarkReport, output_root: Path) -> tuple[Path, Path]:
        output_root.mkdir(parents=True, exist_ok=True)
        json_path = output_root / "benchmark-results.json"
        markdown_path = output_root / "benchmark-report.md"
        self._replace(json_path, self.render_json(report))
        self._replace(markdown_path, self.render_markdown(report))
        return json_path, markdown_path

    @staticmethod
    def _replace(path: Path, content: str) -> None:
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        temporary.write_text(content, encoding="utf-8", newline="\n")
        temporary.replace(path)


def assert_public_report(report: BenchmarkReport) -> None:
    payload = report.model_dump(mode="json")

    def inspect(value: object, key: str = "") -> None:
        normalized_key = key.casefold()
        if any(part in normalized_key for part in ("api_key", "password", "secret", "credential")):
            raise ValueError("benchmark report contains a forbidden secret-bearing field")
        if isinstance(value, dict):
            for child_key, child in value.items():
                inspect(child, str(child_key))
        elif isinstance(value, list):
            for child in value:
                inspect(child, key)
        elif isinstance(value, str) and (
            _PRIVATE_PATH.search(value) is not None or _SECRET_TEXT.search(value) is not None
        ):
            raise ValueError("benchmark report contains local-path or credential material")

    inspect(payload)
