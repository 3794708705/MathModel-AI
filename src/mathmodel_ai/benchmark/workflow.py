from __future__ import annotations

import logging
import os
import stat
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Protocol
from uuid import UUID, uuid4

from mathmodel_ai.benchmark.evaluation import BenchmarkEvaluator
from mathmodel_ai.benchmark.manifests import BenchmarkManifestRegistry, BlindSolveBundle
from mathmodel_ai.benchmark.reporting import BenchmarkReportBuilder, BenchmarkReportRenderer
from mathmodel_ai.benchmark.sources import (
    BenchmarkResourceCache,
    BenchmarkSourceError,
    BenchmarkStructureInspector,
)
from mathmodel_ai.core.types import ProviderName
from mathmodel_ai.paper.hashing import sha256_bytes, sha256_json
from mathmodel_ai.paper.literature import CitationMetadataVerifier, LiteratureSource
from mathmodel_ai.providers.factory import ProviderRegistry
from mathmodel_ai.providers.repository import ProviderModelRegistry
from mathmodel_ai.providers.security import endpoint_identity_confidence
from mathmodel_ai.schemas.benchmark import (
    BenchmarkAttempt,
    BenchmarkCaseManifest,
    BenchmarkCaseResult,
    BenchmarkCaseStatus,
    BenchmarkConfig,
    BenchmarkFailure,
    BenchmarkHumanIntervention,
    BenchmarkMetric,
    BenchmarkMetricKind,
    BenchmarkReport,
    BenchmarkRun,
    BenchmarkRunRequest,
    BenchmarkRunStatus,
    FailureCategory,
    FailureSeverity,
    benchmark_attempt_digest,
    benchmark_failure_digest,
    benchmark_metric_digest,
    benchmark_run_digest,
)
from mathmodel_ai.schemas.independent_verification import IndependentVerificationView
from mathmodel_ai.schemas.paper import (
    LiteratureNeedType,
    LiteratureSearchNeed,
    ReferenceMetadataOrigin,
    ReferenceMetadataStatus,
)
from mathmodel_ai.schemas.provider_registry import (
    CapabilitySource,
    CapabilityStatus,
    ModelCapability,
    ProbeAuthenticationStatus,
    ProviderHealthStatus,
    StructuredOutputStrategy,
)
from mathmodel_ai.schemas.submission import CompetitionProfile, ProfileVerificationStatus
from mathmodel_ai.submission.integrity import validate_competition_profile

ZERO_DIGEST = "0" * 64
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CaseExecutionOutcome:
    status: BenchmarkCaseStatus
    project_id: UUID
    provider_is_live: bool
    metrics: tuple[BenchmarkMetric, ...]
    failures: tuple[BenchmarkFailure, ...]
    interventions: tuple[BenchmarkHumanIntervention, ...]
    result: BenchmarkCaseResult


class LiveCaseExecutor(Protocol):
    async def execute(
        self,
        *,
        attempt_id: UUID,
        bundle: BlindSolveBundle,
        profile: CompetitionProfile,
        request: BenchmarkRunRequest,
    ) -> CaseExecutionOutcome: ...


@dataclass(frozen=True)
class BenchmarkWorkflowOutcome:
    report: BenchmarkReport
    json_path: Path
    markdown_path: Path


@dataclass(frozen=True)
class GitCodeIdentity:
    commit: str
    source_tree_digest: str
    working_tree_dirty: bool


class BenchmarkWorkflow:
    def __init__(
        self,
        *,
        repository: object,
        manifests: BenchmarkManifestRegistry,
        resource_cache: BenchmarkResourceCache,
        inspector: BenchmarkStructureInspector,
        providers: ProviderRegistry,
        literature_source: LiteratureSource,
        profile: CompetitionProfile,
        code_commit: str,
        source_tree_digest: str,
        working_tree_dirty: bool,
        output_root: Path,
        case_executor: LiveCaseExecutor | None = None,
        evaluator: BenchmarkEvaluator | None = None,
        provider_configurations: ProviderModelRegistry | None = None,
        independent_verifier: Callable[[UUID], IndependentVerificationView] | None = None,
        independent_runner: Callable[[UUID, UUID], None] | None = None,
    ) -> None:
        from mathmodel_ai.benchmark.repository import BenchmarkRepository

        if not isinstance(repository, BenchmarkRepository):
            raise TypeError("benchmark repository has an invalid implementation")
        if len(code_commit) != 40 or any(
            character not in "0123456789abcdef" for character in code_commit
        ):
            raise ValueError("benchmark code commit must be a full lowercase Git hash")
        if len(source_tree_digest) != 64 or any(
            character not in "0123456789abcdef" for character in source_tree_digest
        ):
            raise ValueError("benchmark source tree digest must be a lowercase SHA-256")
        if profile.verification_status is not ProfileVerificationStatus.VERIFIED:
            raise ValueError("Phase 8 requires a VERIFIED real competition profile")
        self._repository = repository
        self._manifests = manifests
        self._cache = resource_cache
        self._inspector = inspector
        self._providers = providers
        self._literature = literature_source
        self._profile = profile
        self._code_commit = code_commit
        self._source_tree_digest = source_tree_digest
        self._working_tree_dirty = working_tree_dirty
        self._output_root = output_root
        self._case_executor = case_executor
        self._evaluator = evaluator or BenchmarkEvaluator()
        self._provider_configurations = provider_configurations
        self._reports = BenchmarkReportBuilder(
            self._evaluator, independent_verifier=independent_verifier
        )
        self._independent_runner = independent_runner
        self._renderer = BenchmarkReportRenderer()

    async def run(self, request: BenchmarkRunRequest) -> BenchmarkWorkflowOutcome:
        request = request.model_copy(update={"config": self._registry_bound_config(request.config)})
        manifests = [self._manifests.get(case_id) for case_id in request.case_ids]
        profile_digest = validate_competition_profile(self._profile)
        run = BenchmarkRun(
            code_commit=self._code_commit,
            source_tree_digest=self._source_tree_digest,
            working_tree_dirty=self._working_tree_dirty,
            config=request.config,
            case_manifest_digests={
                item.benchmark_id: self._manifests.digest(item.benchmark_id) for item in manifests
            },
            competition_profile_digests={
                f"{self._profile.profile_id}:v{self._profile.version}": profile_digest
            },
            status=BenchmarkRunStatus.RUNNING,
            run_digest=ZERO_DIGEST,
        )
        run = run.model_copy(update={"run_digest": benchmark_run_digest(run)})
        self._repository.create_run(run)

        for manifest in manifests:
            await self._run_case(run, request, manifest)

        attempts = self._repository.list_attempts(run.run_id)
        metrics = self._repository.list_metrics(run.run_id)
        failures = self._repository.list_failures(run.run_id)
        results = self._repository.list_results(run.run_id)
        acceptance = self._reports.build(
            run=run,
            attempts=attempts,
            claimed_results=results,
            metrics=metrics,
            failures=failures,
            interventions=self._repository.list_interventions(run.run_id),
            literature_required={item.benchmark_id: item.requires_literature for item in manifests},
        ).acceptance
        finished = run.model_copy(
            update={
                "status": acceptance.status,
                "live_provider_status": acceptance.live_provider_status,
                "live_literature_status": acceptance.live_literature_status,
                "finished_at": datetime.now(UTC),
            }
        )
        self._repository.finish_run(finished)
        report = self._reports.build(
            run=self._repository.get_run(run.run_id),
            attempts=attempts,
            claimed_results=results,
            metrics=metrics,
            failures=failures,
            interventions=self._repository.list_interventions(run.run_id),
            literature_required={item.benchmark_id: item.requires_literature for item in manifests},
        )
        json_path, markdown_path = self._renderer.write(report, self._output_root / str(run.run_id))
        return BenchmarkWorkflowOutcome(
            report=report,
            json_path=json_path,
            markdown_path=markdown_path,
        )

    def report(self, run_id: UUID) -> BenchmarkReport:
        run = self._repository.get_run(run_id)
        manifests = {item.benchmark_id: item for item in self._manifests.list()}
        return self._reports.build(
            run=run,
            attempts=self._repository.list_attempts(run_id),
            claimed_results=self._repository.list_results(run_id),
            metrics=self._repository.list_metrics(run_id),
            failures=self._repository.list_failures(run_id),
            interventions=self._repository.list_interventions(run_id),
            literature_required={
                benchmark_id: item.requires_literature for benchmark_id, item in manifests.items()
            },
        )

    async def _run_case(
        self,
        run: BenchmarkRun,
        request: BenchmarkRunRequest,
        manifest: BenchmarkCaseManifest,
    ) -> None:
        case_started = monotonic()
        try:
            bundle = self._cache.materialize(manifest, self._manifests)
        except (BenchmarkSourceError, OSError, ValueError) as exc:
            await self._invalid_setup(run, manifest, case_started, exc)
            return

        attempt = self._running_attempt(run, manifest, bundle.solve_input_digest)
        self._repository.create_attempt(attempt)
        checks = self._inspector.inspect(bundle)
        extraction_accuracy = sum(item.passed for item in checks) / len(checks) if checks else 1.0
        prior_cost = self._estimated_cost(run.run_id)
        max_total_cost = request.config.budget.max_total_cost
        if max_total_cost is not None and prior_cost >= max_total_cost:
            self._record_budget_block(
                attempt,
                manifest,
                extraction_accuracy=extraction_accuracy,
                elapsed=monotonic() - case_started,
                reason=(
                    f"run estimated cost {prior_cost:.6f} reached max_total_cost "
                    f"{max_total_cost:.6f} before this case"
                ),
            )
            return
        literature_count, literature_failure = await self._live_literature(
            manifest, attempt.attempt_id
        )
        literature_live = literature_count > 0
        provider_available = self._provider_available(request.config)
        if provider_available and self._case_executor is not None:
            try:
                outcome = await self._case_executor.execute(
                    attempt_id=attempt.attempt_id,
                    bundle=bundle,
                    profile=self._profile,
                    request=request,
                )
            except Exception as exc:  # every formal attempt must reach a recorded terminal state
                self._pipeline_failure(
                    attempt,
                    manifest,
                    literature_count=literature_count,
                    literature_failure=literature_failure,
                    extraction_accuracy=extraction_accuracy,
                    case_started=case_started,
                    error=exc,
                )
                return
            self._persist_live_outcome(
                run,
                attempt,
                outcome,
                request=request,
                literature_count=literature_count,
                literature_failure=literature_failure,
                extraction_accuracy=extraction_accuracy,
                case_started=case_started,
            )
            return

        category = (
            FailureCategory.PROVIDER if not provider_available else FailureCategory.INFRASTRUCTURE
        )
        root_cause = (
            f"configured live provider {request.config.provider!r} is unavailable; "
            "no Mock fallback was used"
            if not provider_available
            else "live full-pipeline case executor is not configured"
        )
        failure = self._failure(
            attempt.attempt_id,
            stage="LIVE_PROVIDER",
            category=category,
            severity=FailureSeverity.P1,
            root_cause=root_cause,
            evidence_refs=[f"run:{run.run_id}", f"manifest:{attempt.manifest_digest}"],
            reproducible=True,
            generic_issue=False,
            proposed_fix="configure an approved live provider and rerun the formal attempt",
        )
        failures = [failure]
        if literature_failure is not None:
            failures.append(literature_failure)
        for item in failures:
            self._repository.add_failure(item)
        elapsed = monotonic() - case_started
        budget_reasons = self._budget_reasons(
            request,
            case_cost=0,
            prior_total_cost=prior_cost,
            elapsed=elapsed,
        )
        if budget_reasons:
            failures.append(
                self._failure(
                    attempt.attempt_id,
                    stage="BENCHMARK_BUDGET",
                    category=FailureCategory.INFRASTRUCTURE,
                    severity=FailureSeverity.P2,
                    root_cause="; ".join(budget_reasons),
                    evidence_refs=[f"attempt:{attempt.attempt_id}"],
                    reproducible=True,
                    generic_issue=False,
                    proposed_fix="raise the explicit benchmark budget or reduce the workload",
                )
            )
            self._repository.add_failure(failures[-1])
        status = (
            BenchmarkCaseStatus.BUDGET_EXCEEDED
            if budget_reasons
            else BenchmarkCaseStatus.BLOCKED_ENVIRONMENT
        )
        metrics = self._blocked_metrics(
            attempt.attempt_id,
            extraction_accuracy=extraction_accuracy,
            literature_count=literature_count,
            elapsed=elapsed,
        )
        for metric in metrics:
            self._repository.add_metric(metric)
        terminal = self._finish_attempt(
            attempt,
            status=status,
            provider_is_live=False,
            literature_is_live=literature_live,
        )
        self._repository.finish_attempt(terminal)
        result = self._evaluator.evaluate_attempt(
            terminal,
            metrics,
            failures,
            [],
            requires_literature=manifest.requires_literature,
        )
        self._repository.add_result(result)

    def _record_budget_block(
        self,
        attempt: BenchmarkAttempt,
        manifest: BenchmarkCaseManifest,
        *,
        extraction_accuracy: float,
        elapsed: float,
        reason: str,
    ) -> None:
        failure = self._failure(
            attempt.attempt_id,
            stage="BENCHMARK_BUDGET",
            category=FailureCategory.INFRASTRUCTURE,
            severity=FailureSeverity.P2,
            root_cause=reason,
            evidence_refs=[f"attempt:{attempt.attempt_id}"],
            reproducible=True,
            generic_issue=False,
            proposed_fix="raise the explicit benchmark budget or reduce the workload",
        )
        self._repository.add_failure(failure)
        metrics = self._blocked_metrics(
            attempt.attempt_id,
            extraction_accuracy=extraction_accuracy,
            literature_count=0,
            elapsed=elapsed,
            blocked_reason="NOT_EXECUTED:BUDGET_EXCEEDED",
        )
        for metric in metrics:
            self._repository.add_metric(metric)
        terminal = self._finish_attempt(
            attempt,
            status=BenchmarkCaseStatus.BUDGET_EXCEEDED,
            provider_is_live=False,
            literature_is_live=False,
        )
        self._repository.finish_attempt(terminal)
        result = self._evaluator.evaluate_attempt(
            terminal,
            metrics,
            [failure],
            [],
            requires_literature=manifest.requires_literature,
        )
        self._repository.add_result(result)

    def _estimated_cost(self, run_id: UUID) -> float:
        return sum(
            item.value
            for item in self._repository.list_metrics(run_id)
            if item.name == "estimated_cost" and item.value >= 0
        )

    @staticmethod
    def _budget_reasons(
        request: BenchmarkRunRequest,
        *,
        case_cost: float,
        prior_total_cost: float,
        elapsed: float,
    ) -> list[str]:
        budget = request.config.budget
        reasons: list[str] = []
        if budget.max_case_cost is not None and case_cost > budget.max_case_cost:
            reasons.append(
                f"case estimated cost {case_cost:.6f} exceeded max_case_cost "
                f"{budget.max_case_cost:.6f}"
            )
        if (
            budget.max_total_cost is not None
            and prior_total_cost + case_cost > budget.max_total_cost
        ):
            reasons.append(
                f"run estimated cost {prior_total_cost + case_cost:.6f} exceeded "
                f"max_total_cost {budget.max_total_cost:.6f}"
            )
        if budget.max_wall_time_seconds is not None and elapsed > budget.max_wall_time_seconds:
            reasons.append(
                f"case wall time {elapsed:.6f}s exceeded max_wall_time_seconds "
                f"{budget.max_wall_time_seconds:.6f}s"
            )
        return reasons

    def _pipeline_failure(
        self,
        attempt: BenchmarkAttempt,
        manifest: BenchmarkCaseManifest,
        *,
        literature_count: int,
        literature_failure: BenchmarkFailure | None,
        extraction_accuracy: float,
        case_started: float,
        error: Exception,
    ) -> None:
        failure = self._failure(
            attempt.attempt_id,
            stage="PHASE_1_7_PIPELINE",
            category=FailureCategory.INFRASTRUCTURE,
            severity=FailureSeverity.P1,
            root_cause=f"live pipeline stopped with {type(error).__name__}",
            evidence_refs=[f"attempt:{attempt.attempt_id}"],
            reproducible=False,
            generic_issue=True,
            proposed_fix=(
                "inspect persisted stage evidence, apply a generic fix, and append a rerun"
            ),
        )
        failures = [failure]
        if literature_failure is not None:
            failures.append(literature_failure)
        for item in failures:
            self._repository.add_failure(item)
        metrics = self._blocked_metrics(
            attempt.attempt_id,
            extraction_accuracy=extraction_accuracy,
            literature_count=literature_count,
            elapsed=monotonic() - case_started,
        )
        for metric in metrics:
            self._repository.add_metric(metric)
        terminal = self._finish_attempt(
            attempt,
            status=BenchmarkCaseStatus.FAIL,
            provider_is_live=False,
            literature_is_live=literature_count > 0,
        )
        self._repository.finish_attempt(terminal)
        result = self._evaluator.evaluate_attempt(
            terminal,
            metrics,
            failures,
            [],
            requires_literature=manifest.requires_literature,
        )
        self._repository.add_result(result)

    async def _invalid_setup(
        self,
        run: BenchmarkRun,
        manifest: BenchmarkCaseManifest,
        case_started: float,
        error: Exception,
    ) -> None:
        unavailable_digest = sha256_json(
            {
                "manifest_digest": self._manifests.digest(manifest.benchmark_id),
                "materialization": "FAILED",
                "error_type": type(error).__name__,
            }
        )
        attempt = self._running_attempt(run, manifest, unavailable_digest)
        self._repository.create_attempt(attempt)
        failure = self._failure(
            attempt.attempt_id,
            stage="BENCHMARK_SETUP",
            category=FailureCategory.INFRASTRUCTURE,
            severity=FailureSeverity.P1,
            root_cause=f"official solve input could not be verified ({type(error).__name__})",
            evidence_refs=[f"manifest:{attempt.manifest_digest}"],
            reproducible=True,
            generic_issue=False,
            proposed_fix="restore the exact official hash-pinned input and rerun",
        )
        self._repository.add_failure(failure)
        metrics = self._blocked_metrics(
            attempt.attempt_id,
            extraction_accuracy=0,
            literature_count=0,
            elapsed=monotonic() - case_started,
        )
        for metric in metrics:
            self._repository.add_metric(metric)
        terminal = self._finish_attempt(
            attempt,
            status=BenchmarkCaseStatus.INVALID_BENCHMARK_SETUP,
            provider_is_live=False,
            literature_is_live=False,
        )
        self._repository.finish_attempt(terminal)
        result = self._evaluator.evaluate_attempt(
            terminal,
            metrics,
            [failure],
            [],
            requires_literature=manifest.requires_literature,
        )
        self._repository.add_result(result)

    async def _live_literature(
        self, manifest: BenchmarkCaseManifest, attempt_id: UUID
    ) -> tuple[int, BenchmarkFailure | None]:
        if not manifest.requires_literature or not manifest.literature_queries:
            return 0, None
        try:
            if self._literature.name.casefold() in {"fixture", "mock"}:
                raise ValueError("fixture literature cannot satisfy live benchmark validation")
            records = await self._literature.search(
                LiteratureSearchNeed(
                    need_id=f"LITNEED-{manifest.benchmark_id}",
                    need_type=LiteratureNeedType.DOMAIN_CONTEXT,
                    query=manifest.literature_queries[0],
                    purpose="Phase 8 live literature metadata validation",
                ),
                uuid4(),
            )
            if not records:
                raise ValueError("live literature search returned no usable records")
            if records[0].metadata_origin is not ReferenceMetadataOrigin.RETRIEVED:
                raise ValueError("literature metadata was not retrieved from a live source")
            check = await CitationMetadataVerifier().verify(records[0], self._literature)
            if check.status is not ReferenceMetadataStatus.VERIFIED:
                raise ValueError(f"live metadata verification returned {check.status.value}")
            return 1, None
        except Exception as exc:  # external adapters expose heterogeneous network exceptions
            return 0, self._failure(
                attempt_id,
                stage="LIVE_LITERATURE",
                category=FailureCategory.LITERATURE,
                severity=FailureSeverity.P1,
                root_cause=f"live literature verification failed ({type(exc).__name__})",
                evidence_refs=[f"manifest:{self._manifests.digest(manifest.benchmark_id)}"],
                reproducible=False,
                generic_issue=False,
                proposed_fix="restore live literature connectivity and rerun without fixtures",
            )

    def _persist_live_outcome(
        self,
        run: BenchmarkRun,
        attempt: BenchmarkAttempt,
        outcome: CaseExecutionOutcome,
        *,
        request: BenchmarkRunRequest,
        literature_count: int,
        literature_failure: BenchmarkFailure | None,
        extraction_accuracy: float,
        case_started: float,
    ) -> None:
        provider_coverage, provider_evidence = self._repository.live_provider_coverage(
            outcome.project_id
        )
        provider_activity, activity_evidence = self._repository.live_provider_activity(
            outcome.project_id
        )
        if outcome.provider_is_live != provider_activity:
            raise ValueError("live case executor provider activity conflicts with agent runs")
        if outcome.status is not BenchmarkCaseStatus.FAIL and provider_coverage != 1:
            raise ValueError(
                "live case executor lacks six independently persisted non-Mock agent runs"
            )
        if outcome.result.attempt_id != attempt.attempt_id:
            raise ValueError("live case result crossed attempt identity")
        reserved_names = {
            "data_extraction_accuracy",
            "live_literature_verified_reference_count",
            "live_provider_critical_agent_coverage",
            "wall_time_seconds",
        }
        if any(item.name in reserved_names for item in outcome.metrics):
            raise ValueError("live case executor attempted to override benchmark-owned metrics")
        elapsed = monotonic() - case_started
        supplemental = [
            self._metric(
                attempt.attempt_id,
                "data_extraction_accuracy",
                BenchmarkMetricKind.QUALITY,
                extraction_accuracy,
                "ratio",
                f"solve-input:{attempt.solve_input_digest}",
            ),
            self._metric(
                attempt.attempt_id,
                "live_literature_verified_reference_count",
                BenchmarkMetricKind.COUNT,
                literature_count,
                "references",
                "crossref-independent-resolution",
            ),
            self._metric(
                attempt.attempt_id,
                "live_provider_critical_agent_coverage",
                BenchmarkMetricKind.QUALITY,
                provider_coverage,
                "ratio",
                ",".join(provider_evidence or activity_evidence),
            ),
            self._metric(
                attempt.attempt_id,
                "wall_time_seconds",
                BenchmarkMetricKind.RUNTIME_SECONDS,
                elapsed,
                "seconds",
                f"attempt:{attempt.attempt_id}",
            ),
        ]
        all_metrics = [*outcome.metrics, *supplemental]
        case_cost = next((item.value for item in all_metrics if item.name == "estimated_cost"), 0)
        budget_reasons = self._budget_reasons(
            request,
            case_cost=case_cost,
            prior_total_cost=self._estimated_cost(run.run_id),
            elapsed=elapsed,
        )
        for metric in all_metrics:
            self._repository.add_metric(metric)
        failures = list(outcome.failures)
        if literature_failure is not None:
            failures.append(literature_failure)
        if budget_reasons:
            failures.append(
                self._failure(
                    attempt.attempt_id,
                    stage="BENCHMARK_BUDGET",
                    category=FailureCategory.INFRASTRUCTURE,
                    severity=FailureSeverity.P2,
                    root_cause="; ".join(budget_reasons),
                    evidence_refs=[f"attempt:{attempt.attempt_id}"],
                    reproducible=True,
                    generic_issue=False,
                    proposed_fix="raise the explicit benchmark budget or reduce the workload",
                )
            )
        for failure in failures:
            self._repository.add_failure(failure)
        for intervention in outcome.interventions:
            self._repository.add_intervention(intervention)
        terminal = self._finish_attempt(
            attempt.model_copy(update={"project_id": outcome.project_id}),
            status=(BenchmarkCaseStatus.BUDGET_EXCEEDED if budget_reasons else outcome.status),
            provider_is_live=provider_activity,
            literature_is_live=literature_count > 0,
        )
        self._repository.finish_attempt(terminal)
        if outcome.result.verified_result_id is not None and self._independent_runner is not None:
            try:
                self._independent_runner(attempt.attempt_id, outcome.result.verified_result_id)
            except (OSError, ValueError) as exc:
                # The deterministic report gate will expose NOT_READY. Never erase
                # the already persisted live pipeline evidence or claim a replay.
                logger.warning(
                    "Independent benchmark verification did not complete",
                    extra={
                        "attempt_id": str(attempt.attempt_id),
                        "error_type": type(exc).__name__,
                    },
                )
        recalculated = self._evaluator.evaluate_attempt(
            terminal,
            all_metrics,
            failures,
            list(outcome.interventions),
            claimed=outcome.result,
            requires_literature=True,
        )
        self._repository.add_result(recalculated)

    def _registry_bound_config(self, config: BenchmarkConfig) -> BenchmarkConfig:
        if config.model_id is None:
            return config
        registry = self._provider_configurations
        if registry is None:
            raise ValueError("benchmark model_id requires the configured model registry")
        model = registry.get_model(config.model_id)
        endpoint = registry.get_provider(model.provider_id)
        expected = {
            "provider_id": endpoint.provider_id,
            "provider_config_digest": endpoint.config_digest,
            "model_config_digest": model.config_digest,
            "protocol": endpoint.protocol,
            "endpoint_trust": endpoint.trust_level,
            "model_identity_confidence": endpoint_identity_confidence(endpoint),
        }
        for field, value in expected.items():
            configured = getattr(config, field)
            if configured is not None and configured != value:
                raise ValueError(f"benchmark {field} conflicts with current registry identity")
        return config.model_copy(
            update={
                "provider": endpoint.provider_id,
                "model": model.remote_model,
                **expected,
            }
        )

    def _provider_available(self, config: BenchmarkConfig) -> bool:
        if config.model_id is not None:
            return self._registered_provider_available(config)
        try:
            provider = ProviderName(config.provider.casefold())
        except ValueError:
            return False
        return provider is not ProviderName.MOCK and provider in self._providers.available

    def _registered_provider_available(self, config: BenchmarkConfig) -> bool:
        registry = self._provider_configurations
        if registry is None or config.model_id is None:
            return False
        try:
            model = registry.get_model(config.model_id)
            endpoint = registry.get_provider(model.provider_id)
            current_probe = registry.latest_probe(model.model_id, current_only=True)
            credential_ready = registry.credential_configured(endpoint.provider_id)
        except Exception:
            return False
        if (
            endpoint.provider_id == ProviderName.MOCK.value
            or not endpoint.enabled
            or not model.enabled
            or not credential_ready
            or endpoint.health_status is not ProviderHealthStatus.READY
            or config.provider_id != endpoint.provider_id
            or config.provider_config_digest != endpoint.config_digest
            or config.model_config_digest != model.config_digest
            or current_probe is None
            or current_probe.authentication_status is not ProbeAuthenticationStatus.PASS
        ):
            return False
        for capability in (ModelCapability.TEXT, ModelCapability.STRUCTURED_OUTPUT):
            if current_probe is None:  # pragma: no cover - guarded above
                return False
            evidence = current_probe.capabilities.get(capability)
            if (
                evidence is not None
                and evidence.source is CapabilitySource.PROBED
                and evidence.status is CapabilityStatus.SUPPORTED
            ):
                continue
            if (
                capability is ModelCapability.STRUCTURED_OUTPUT
                and model.structured_output_strategy
                is StructuredOutputStrategy.PROMPT_JSON_FALLBACK
                and evidence is not None
                and evidence.source is CapabilitySource.PROBED
                and evidence.status is CapabilityStatus.PARTIAL
                and (text_evidence := current_probe.capabilities.get(ModelCapability.TEXT))
                is not None
                and text_evidence.source is CapabilitySource.PROBED
                and text_evidence.status is CapabilityStatus.SUPPORTED
            ):
                continue
            return False
        return True

    def _running_attempt(
        self,
        run: BenchmarkRun,
        manifest: BenchmarkCaseManifest,
        solve_input_digest: str,
    ) -> BenchmarkAttempt:
        attempt = BenchmarkAttempt(
            run_id=run.run_id,
            benchmark_id=manifest.benchmark_id,
            manifest_digest=self._manifests.digest(manifest.benchmark_id),
            attempt_number=1,
            status=BenchmarkCaseStatus.RUNNING,
            solve_input_digest=solve_input_digest,
            attempt_digest=ZERO_DIGEST,
        )
        return attempt.model_copy(update={"attempt_digest": benchmark_attempt_digest(attempt)})

    @staticmethod
    def _finish_attempt(
        attempt: BenchmarkAttempt,
        *,
        status: BenchmarkCaseStatus,
        provider_is_live: bool,
        literature_is_live: bool,
    ) -> BenchmarkAttempt:
        terminal = attempt.model_copy(
            update={
                "status": status,
                "provider_is_live": provider_is_live,
                "literature_is_live": literature_is_live,
                "finished_at": datetime.now(UTC),
                "attempt_digest": ZERO_DIGEST,
            }
        )
        return terminal.model_copy(update={"attempt_digest": benchmark_attempt_digest(terminal)})

    def _blocked_metrics(
        self,
        attempt_id: UUID,
        *,
        extraction_accuracy: float,
        literature_count: int,
        elapsed: float,
        blocked_reason: str = "NOT_EXECUTED:BLOCKED_PROVIDER",
    ) -> list[BenchmarkMetric]:
        quality = {
            "problem_understanding_accuracy": 0,
            "critical_constraint_recall": 0,
            "subproblem_coverage": 0,
            "data_extraction_accuracy": extraction_accuracy,
            "model_appropriateness": 0,
            "mathematical_validity": 0,
            "solver_success": 0,
            "validation_pass": 0,
            "sensitivity_completion": 0,
            "robustness_completion": 0,
            "red_team_usefulness": 0,
            "repair_success": 0,
            "citation_validity": 0,
            "citation_support_accuracy": 0,
            "paper_factual_consistency": 0,
            "competition_compliance": 0,
            "submission_completeness": 0,
            "central_model_valid": 0,
        }
        counts = {
            "major_unanswered_subproblem_count": 1,
            "fabricated_result_count": 0,
            "unverified_central_result_count": 1,
            "fabricated_critical_citation_count": 0,
            "wrong_submission_artifact_count": 0,
            "blocking_competition_violation_count": 0,
            "secret_leak_count": 0,
            "live_provider_critical_agent_coverage": 0,
            "live_literature_verified_reference_count": literature_count,
            "provider_calls": 0,
            "total_tokens": 0,
            "solver_calls": 0,
            "experiment_runs": 0,
            "repair_iterations": 0,
            "human_interventions": 0,
        }
        metrics = [
            self._metric(
                attempt_id,
                name,
                BenchmarkMetricKind.QUALITY,
                value,
                "ratio",
                blocked_reason,
            )
            for name, value in quality.items()
        ]
        metrics.extend(
            self._metric(
                attempt_id,
                name,
                BenchmarkMetricKind.COUNT,
                value,
                "count",
                blocked_reason,
            )
            for name, value in counts.items()
        )
        metrics.extend(
            [
                self._metric(
                    attempt_id,
                    "wall_time_seconds",
                    BenchmarkMetricKind.RUNTIME_SECONDS,
                    elapsed,
                    "seconds",
                    f"attempt:{attempt_id}",
                ),
                self._metric(
                    attempt_id,
                    "estimated_cost",
                    BenchmarkMetricKind.COST,
                    0,
                    "USD",
                    "provider-calls:0",
                ),
            ]
        )
        return metrics

    @staticmethod
    def _metric(
        attempt_id: UUID,
        name: str,
        kind: BenchmarkMetricKind,
        value: float,
        unit: str,
        evidence_ref: str,
    ) -> BenchmarkMetric:
        metric = BenchmarkMetric(
            attempt_id=attempt_id,
            name=name,
            kind=kind,
            value=value,
            unit=unit,
            evidence_ref=evidence_ref,
            deterministic=True,
            metric_digest=ZERO_DIGEST,
        )
        return metric.model_copy(update={"metric_digest": benchmark_metric_digest(metric)})

    @staticmethod
    def _failure(
        attempt_id: UUID,
        *,
        stage: str,
        category: FailureCategory,
        severity: FailureSeverity,
        root_cause: str,
        evidence_refs: list[str],
        reproducible: bool,
        generic_issue: bool,
        proposed_fix: str,
    ) -> BenchmarkFailure:
        failure = BenchmarkFailure(
            attempt_id=attempt_id,
            stage=stage,
            category=category,
            severity=severity,
            root_cause=root_cause,
            evidence_refs=evidence_refs,
            reproducible=reproducible,
            generic_issue=generic_issue,
            proposed_fix=proposed_fix,
            failure_digest=ZERO_DIGEST,
        )
        return failure.model_copy(update={"failure_digest": benchmark_failure_digest(failure)})


def resolve_git_identity(repository_root: Path) -> GitCodeIdentity:
    root = repository_root.resolve(strict=True)
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
        shell=False,
    )
    commit = completed.stdout.strip().casefold()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("Git did not return a full commit identity")
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD", "--"],
        cwd=root,
        check=True,
        capture_output=True,
        timeout=10,
        shell=False,
    ).stdout
    untracked_output = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
        timeout=10,
        shell=False,
    ).stdout
    untracked: list[dict[str, str]] = []
    for raw_path in sorted(item for item in untracked_output.split(b"\0") if item):
        relative = Path(os.fsdecode(raw_path))
        path = (root / relative).resolve(strict=True)
        if path == root or root not in path.parents:
            raise ValueError("untracked benchmark source escaped repository root")
        info = os.lstat(path)
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise ValueError("untracked benchmark source must be a regular file")
        untracked.append(
            {
                "path": relative.as_posix(),
                "sha256": sha256_bytes(path.read_bytes()),
            }
        )
    source_tree_digest = sha256_json(
        {
            "commit": commit,
            "tracked_diff_sha256": sha256_bytes(diff),
            "untracked": untracked,
        }
    )
    return GitCodeIdentity(
        commit=commit,
        source_tree_digest=source_tree_digest,
        working_tree_dirty=bool(diff or untracked),
    )


def resolve_git_commit(repository_root: Path) -> str:
    return resolve_git_identity(repository_root).commit
