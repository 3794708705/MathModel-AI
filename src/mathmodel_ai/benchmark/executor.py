from __future__ import annotations

from collections.abc import Callable, Sequence
from io import BytesIO
from uuid import UUID

from pypdf import PdfReader

from mathmodel_ai.benchmark.manifests import BlindSolveBundle
from mathmodel_ai.benchmark.repository import BenchmarkRepository
from mathmodel_ai.benchmark.workflow import CaseExecutionOutcome
from mathmodel_ai.core.errors import QualityGateError
from mathmodel_ai.data.workflow import DataExecutionWorkflow
from mathmodel_ai.mathematical.workflow import MathematicalWorkflow
from mathmodel_ai.paper.workflow import PaperWorkflow
from mathmodel_ai.reasoning.repository import ReasoningRepository
from mathmodel_ai.reasoning.workflow import ReasoningWorkflow
from mathmodel_ai.schemas.benchmark import (
    BenchmarkCaseResult,
    BenchmarkCaseStatus,
    BenchmarkDimensionScore,
    BenchmarkFailure,
    BenchmarkMetric,
    BenchmarkMetricKind,
    BenchmarkRunRequest,
    FailureCategory,
    FailureSeverity,
    benchmark_failure_digest,
    benchmark_metric_digest,
    benchmark_result_digest,
)
from mathmodel_ai.schemas.execution import ExecutionStatus
from mathmodel_ai.schemas.independent_verification import (
    IndependentStatus,
    IndependentVerificationReport,
    IndependentVerificationView,
    ReviewedValidationEvidence,
)
from mathmodel_ai.schemas.paper import (
    CompetitionProfile as PaperCompetitionProfile,
)
from mathmodel_ai.schemas.paper import (
    PaperQualityStatus,
    PaperSectionType,
    ReferenceMetadataStatus,
)
from mathmodel_ai.schemas.problem_state import ProblemState
from mathmodel_ai.schemas.quality import QualityGateStatus
from mathmodel_ai.schemas.submission import (
    CompetitionProfile,
    RequirementCoverageStatus,
    RuleResultStatus,
    SubmissionCheckStatus,
)
from mathmodel_ai.schemas.verification import ExperimentReportStatus, ValidationStatus
from mathmodel_ai.submission.workflow import FinalSubmissionWorkflow
from mathmodel_ai.verification.requirements import VerificationRequirementRegistry
from mathmodel_ai.verification.workflow import VerificationWorkflow

ZERO = "0" * 64


class PipelineBenchmarkExecutor:
    """Drive the existing Phase 1-7 workflows without case-specific modeling hints."""

    def __init__(
        self,
        *,
        reasoning_repository: ReasoningRepository,
        reasoning_workflow: ReasoningWorkflow,
        data_workflow: DataExecutionWorkflow,
        mathematical_workflow: MathematicalWorkflow,
        verification_workflow: VerificationWorkflow,
        paper_workflow: PaperWorkflow,
        final_workflow: FinalSubmissionWorkflow,
        benchmark_repository: BenchmarkRepository,
        reviewed_models: VerificationRequirementRegistry | None = None,
        independent_runner: Callable[[UUID, UUID], None] | None = None,
        independent_verifier: Callable[[UUID], IndependentVerificationView] | None = None,
    ) -> None:
        self._reasoning_repository = reasoning_repository
        self._reasoning = reasoning_workflow
        self._data = data_workflow
        self._mathematical = mathematical_workflow
        self._verification = verification_workflow
        self._paper = paper_workflow
        self._final = final_workflow
        self._benchmark_repository = benchmark_repository
        self._reviewed_models = reviewed_models
        self._independent_runner = independent_runner
        self._independent_verifier = independent_verifier

    async def execute(
        self,
        *,
        attempt_id: UUID,
        bundle: BlindSolveBundle,
        profile: CompetitionProfile,
        request: BenchmarkRunRequest,
    ) -> CaseExecutionOutcome:
        reviewed = (
            self._reviewed_models.reviewed_model(bundle.manifest.benchmark_id)
            if self._reviewed_models is not None
            else None
        )
        state = self._reasoning_repository.create_project_problem(
            name=f"Phase 8 {bundle.manifest.benchmark_id}",
            title=bundle.manifest.title,
            raw_problem=self._problem_text(bundle),
            competition=(
                f"{bundle.manifest.competition} {bundle.manifest.year} {bundle.manifest.problem_id}"
            ),
        )
        self._benchmark_repository.bind_attempt_project(attempt_id, state.project_id)
        try:
            await self._reasoning.analyze(
                state.project_id,
                subproblem_identity_contract=(
                    reviewed.subproblem_identity if reviewed is not None else None
                ),
            )
        except Exception as exc:
            return self._failed_pipeline(
                attempt_id,
                state.project_id,
                bundle.manifest.benchmark_id,
                request,
                stage="PROBLEM_UNDERSTANDING",
                category=FailureCategory.PROBLEM_UNDERSTANDING,
                error=exc,
            )
        try:
            for artifact in bundle.artifacts:
                self._data.ingest_file(
                    state.project_id,
                    BytesIO(artifact.content),
                    original_name=artifact.resource.local_filename,
                    declared_mime_type=artifact.resource.media_type,
                )
            current = self._reasoning_repository.load_current(state.project_id)
            if current.data_profiles:
                await self._data.analyze_data(state.project_id)
        except Exception as exc:
            return self._failed_pipeline(
                attempt_id,
                state.project_id,
                bundle.manifest.benchmark_id,
                request,
                stage="DATA",
                category=FailureCategory.DATA_EXTRACTION,
                error=exc,
            )
        try:
            await self._reasoning.explore(state.project_id)
            await self._reasoning.select(state.project_id)
        except Exception as exc:
            return self._failed_pipeline(
                attempt_id,
                state.project_id,
                bundle.manifest.benchmark_id,
                request,
                stage="MODEL_SELECTION",
                category=FailureCategory.MODEL_SELECTION,
                error=exc,
            )
        try:
            mathematical = (
                await self._mathematical.run_reviewed(
                    state.project_id,
                    template=reviewed.model,
                    expected_model_digest=reviewed.model_digest,
                    model_contract_digest=reviewed.model_contract_digest,
                    policy_digest=reviewed.policy_digest,
                    subproblem_identity_digest=reviewed.subproblem_identity_digest,
                )
                if reviewed is not None
                else await self._mathematical.run(state.project_id)
            )
            if (
                mathematical.model_stage.gate.status is not QualityGateStatus.PASS
                or mathematical.solve_stage.gate.status is not QualityGateStatus.PASS
            ):
                raise QualityGateError(
                    "mathematical workflow returned before MODEL and SOLVE gates passed"
                )
        except Exception as exc:
            return self._failed_pipeline(
                attempt_id,
                state.project_id,
                bundle.manifest.benchmark_id,
                request,
                stage="MATHEMATICAL_MODEL_SOLVE",
                category=FailureCategory.MATHEMATICAL_MODEL,
                error=exc,
            )
        independent_report: IndependentVerificationReport | None = None
        reviewed_validation_evidence: ReviewedValidationEvidence | None = None
        if reviewed is not None:
            try:
                if self._independent_runner is None or self._independent_verifier is None:
                    raise QualityGateError("REVIEWED_INDEPENDENT_VERIFIER_NOT_CONFIGURED")
                self._independent_runner(attempt_id, mathematical.solve_stage.result.result_id)
                independent_view = self._independent_verifier(attempt_id)
                independent_report = self._require_independent_pass(
                    independent_view,
                    result_id=mathematical.solve_stage.result.result_id,
                    required_metrics=sum(item.required for item in reviewed.requirements.metrics),
                    required_scenarios=sum(
                        item.required for item in reviewed.requirements.scenarios
                    ),
                )
                if independent_view.plan is None:
                    raise QualityGateError("INDEPENDENT_VERIFICATION_PLAN_NOT_AVAILABLE")
                reviewed_validation_evidence = ReviewedValidationEvidence(
                    requirements=reviewed.requirements,
                    plan=independent_view.plan,
                    report=independent_report,
                )
            except Exception as exc:
                return self._failed_pipeline(
                    attempt_id,
                    state.project_id,
                    bundle.manifest.benchmark_id,
                    request,
                    stage="INDEPENDENT_VERIFICATION",
                    category=FailureCategory.VALIDATION,
                    error=exc,
                )
        try:
            verification = await self._verification.run(
                state.project_id,
                reviewed_evidence=reviewed_validation_evidence,
            )
        except Exception as exc:
            return self._failed_pipeline(
                attempt_id,
                state.project_id,
                bundle.manifest.benchmark_id,
                request,
                stage="VERIFICATION",
                category=FailureCategory.VALIDATION,
                error=exc,
            )
        repair_iterations = 0
        experiment_runs = (
            (independent_report.required_scenarios if independent_report is not None else 0)
            + len(verification.sensitivity.report.experiments)
            + len(verification.robustness.report.experiments)
        )
        if verification.red_team.report.critical_count:
            try:
                repair = await self._verification.repair_until_clear(state.project_id)
            except Exception as exc:
                return self._failed_pipeline(
                    attempt_id,
                    state.project_id,
                    bundle.manifest.benchmark_id,
                    request,
                    stage="MODEL_REPAIR",
                    category=FailureCategory.REPAIR,
                    error=exc,
                )
            repair_iterations = len(repair.iterations)
            for iteration in repair.iterations:
                if iteration.verification is not None:
                    experiment_runs += len(
                        iteration.verification.sensitivity.report.experiments
                    ) + len(iteration.verification.robustness.report.experiments)
        verified_state = self._reasoning_repository.load_current(state.project_id)
        if reviewed is not None and (
            verified_state.mathematical_model is None
            or verified_state.mathematical_model.model_digest != reviewed.model_digest
        ):
            return self._failed_before_paper(
                attempt_id,
                state.project_id,
                bundle.manifest.benchmark_id,
                request,
                "MODEL_DIGEST_CHANGED; VERIFICATION_POLICY_STALE",
            )
        if verified_state.verified_result_id is None:
            return self._failed_before_paper(
                attempt_id,
                state.project_id,
                bundle.manifest.benchmark_id,
                request,
                "full verification did not produce verified_result_id",
            )
        if reviewed is not None and (
            independent_report is None
            or independent_report.result_id != verified_state.verified_result_id
        ):
            return self._failed_before_paper(
                attempt_id,
                state.project_id,
                bundle.manifest.benchmark_id,
                request,
                "verified_result_id is not the independently verified formal result",
            )
        try:
            paper = await self._paper.run(
                state.project_id,
                competition_profile=self._paper_profile(profile),
            )
        except Exception as exc:
            return self._failed_pipeline(
                attempt_id,
                state.project_id,
                bundle.manifest.benchmark_id,
                request,
                stage="PAPER",
                category=FailureCategory.PAPER,
                error=exc,
            )
        try:
            final = await self._final.run(
                state.project_id,
                profile_id=profile.profile_id,
                profile_version=profile.version,
                paper_id=paper.version.paper_id,
                paper_version=paper.version.version,
                freeze_on_pass=True,
            )
        except Exception as exc:
            return self._failed_pipeline(
                attempt_id,
                state.project_id,
                bundle.manifest.benchmark_id,
                request,
                stage="FINAL_JURY_SUBMISSION",
                category=FailureCategory.SUBMISSION,
                error=exc,
            )
        agent_rows = self._reasoning_repository.list_agent_runs(state.project_id)
        blocking_rules = sum(
            item.status
            in {
                RuleResultStatus.FAIL,
                RuleResultStatus.NOT_EVALUABLE,
                RuleResultStatus.HUMAN_REVIEW,
            }
            for item in final.rule_results
            if item.severity.value == "BLOCKING"
        )
        metrics = self._metrics(
            attempt_id=attempt_id,
            request=request,
            model_passed=mathematical.model_stage.gate.status is QualityGateStatus.PASS,
            solver_passed=mathematical.solve_stage.gate.status is QualityGateStatus.PASS,
            validation_passed=(verification.validation.report.status is ValidationStatus.PASS),
            sensitivity_passed=(
                verification.sensitivity.report.status is ExperimentReportStatus.PASS
            ),
            robustness_passed=(
                verification.robustness.report.status is ExperimentReportStatus.PASS
            ),
            red_team_passed=(
                verification.red_team.report.status is ValidationStatus.PASS
                and not verification.red_team.report.critical_count
            ),
            repair_iterations=repair_iterations,
            paper_ready=paper.version.status is PaperQualityStatus.READY_FOR_FINAL_JURY,
            references_verified=bool(paper.references)
            and all(
                item.metadata_status is ReferenceMetadataStatus.VERIFIED
                for item in paper.references
            ),
            coverage=[item.status for item in final.requirements],
            rules=[item.status for item in final.rule_results],
            submission_passed=final.check.status is SubmissionCheckStatus.PASS,
            verified_result_present=final.state.verified_result_id is not None,
            package_present=final.package is not None,
            blocking_rule_violations=blocking_rules,
            solver_calls=self._formal_solver_call_count(final.state),
            experiment_runs=experiment_runs,
            agent_rows=agent_rows,
        )
        failures = [
            self._failure(
                attempt_id,
                category=FailureCategory.VALIDATION,
                severity=FailureSeverity.P2,
                cause=(
                    "problem-understanding, model-appropriateness, and red-team usefulness "
                    "still require independent benchmark evaluation"
                ),
                proposed_fix="complete the independent Phase 8 human/LLM rubric review",
            )
        ]
        if blocking_rules or final.package is None:
            failures.append(
                self._failure(
                    attempt_id,
                    category=FailureCategory.COMPETITION_RULE,
                    severity=FailureSeverity.P1,
                    cause="one or more verified competition rules prevented package freeze",
                    proposed_fix="resolve the recorded rule result without weakening the profile",
                )
            )
        status = (
            BenchmarkCaseStatus.BLOCKED_RULE_POLICY
            if blocking_rules or final.package is None
            else BenchmarkCaseStatus.HUMAN_REVIEW
        )
        snapshot = final.package.snapshot if final.package is not None else None
        claimed = BenchmarkCaseResult(
            attempt_id=attempt_id,
            benchmark_id=bundle.manifest.benchmark_id,
            status=status,
            dimensions=[
                BenchmarkDimensionScore(
                    dimension="pending deterministic recomputation",
                    score=0,
                    maximum=100,
                    evidence_refs=[f"project:{state.project_id}"],
                )
            ],
            score=0,
            verified_result_id=final.state.verified_result_id,
            paper_id=paper.version.paper_id,
            paper_version=paper.version.version,
            submission_id=snapshot.submission_id if snapshot is not None else None,
            submission_status=snapshot.status.value if snapshot is not None else None,
            artifact_refs=[
                *(f"paper-artifact:{item.sha256}" for item in paper.compilation.artifacts),
                *([f"submission:{snapshot.snapshot_digest}"] if snapshot is not None else []),
            ],
            human_intervention_count=0,
            provider_call_count=0,
            total_tokens=0,
            estimated_cost=0,
            wall_time_seconds=0,
            result_digest=ZERO,
        )
        claimed = claimed.model_copy(update={"result_digest": benchmark_result_digest(claimed)})
        return CaseExecutionOutcome(
            status=status,
            project_id=state.project_id,
            provider_is_live=True,
            metrics=tuple(metrics),
            failures=tuple(failures),
            interventions=(),
            result=claimed,
        )

    @staticmethod
    def _require_independent_pass(
        view: IndependentVerificationView,
        *,
        result_id: UUID,
        required_metrics: int,
        required_scenarios: int,
    ) -> IndependentVerificationReport:
        report = view.report
        if (
            view.status is not IndependentStatus.PASS
            or report is None
            or report.status is not IndependentStatus.PASS
            or report.result_id != result_id
            or report.required_metrics != required_metrics
            or report.passed_metrics != required_metrics
            or report.required_scenarios != required_scenarios
            or report.passed_scenarios != required_scenarios
            or len(report.replays) != required_scenarios
            or report.errors
            or view.blockers
        ):
            raise QualityGateError("INDEPENDENT_VERIFICATION_REQUIREMENTS_NOT_SATISFIED")
        execution_ids = [item.execution.run_id for item in report.replays]
        if len(execution_ids) != len(set(execution_ids)):
            raise QualityGateError("INDEPENDENT_VERIFICATION_REUSED_SCENARIO_EXECUTION")
        return report

    def _failed_pipeline(
        self,
        attempt_id: UUID,
        project_id: UUID,
        benchmark_id: str,
        request: BenchmarkRunRequest,
        *,
        stage: str,
        category: FailureCategory,
        error: Exception,
    ) -> CaseExecutionOutcome:
        agent_rows = self._reasoning_repository.list_agent_runs(project_id)
        live_rows = [item for item in agent_rows if self._is_live_agent_row(item)]
        metrics = self._partial_metrics(
            attempt_id,
            request,
            live_rows,
            state=self._reasoning_repository.load_current(project_id),
        )
        provider_calls = self._usage(live_rows)[3]
        failure = self._failure(
            attempt_id,
            stage=stage,
            category=category,
            severity=FailureSeverity.P1,
            cause=f"live pipeline stopped with {type(error).__name__}",
            proposed_fix=(
                "inspect persisted stage evidence, apply a generic fix, and append a rerun"
            ),
        )
        result = BenchmarkCaseResult(
            attempt_id=attempt_id,
            benchmark_id=benchmark_id,
            status=BenchmarkCaseStatus.FAIL,
            dimensions=[
                BenchmarkDimensionScore(
                    dimension="failed Phase 1-7 stage",
                    score=0,
                    maximum=100,
                    evidence_refs=[f"project:{project_id}", f"stage:{stage}"],
                )
            ],
            score=0,
            human_intervention_count=0,
            provider_call_count=provider_calls,
            total_tokens=int(next(item.value for item in metrics if item.name == "total_tokens")),
            estimated_cost=next(item.value for item in metrics if item.name == "estimated_cost"),
            wall_time_seconds=0,
            result_digest=ZERO,
        )
        result = result.model_copy(update={"result_digest": benchmark_result_digest(result)})
        return CaseExecutionOutcome(
            status=BenchmarkCaseStatus.FAIL,
            project_id=project_id,
            provider_is_live=provider_calls > 0,
            metrics=tuple(metrics),
            failures=(failure,),
            interventions=(),
            result=result,
        )

    def _partial_metrics(
        self,
        attempt_id: UUID,
        request: BenchmarkRunRequest,
        agent_rows: Sequence[object],
        *,
        state: ProblemState,
    ) -> list[BenchmarkMetric]:
        quality_names = (
            "problem_understanding_accuracy",
            "critical_constraint_recall",
            "subproblem_coverage",
            "model_appropriateness",
            "mathematical_validity",
            "solver_success",
            "validation_pass",
            "sensitivity_completion",
            "robustness_completion",
            "red_team_usefulness",
            "repair_success",
            "citation_validity",
            "citation_support_accuracy",
            "paper_factual_consistency",
            "competition_compliance",
            "submission_completeness",
            "central_model_valid",
        )
        metrics = [self._metric(attempt_id, name, 0) for name in quality_names]
        metrics.extend(
            self._metric(attempt_id, name, value, kind=BenchmarkMetricKind.COUNT)
            for name, value in {
                "major_unanswered_subproblem_count": 1,
                "fabricated_result_count": 0,
                "unverified_central_result_count": 1,
                "fabricated_critical_citation_count": 0,
                "wrong_submission_artifact_count": 1,
                "blocking_competition_violation_count": 0,
                "secret_leak_count": 0,
                "solver_calls": self._formal_solver_call_count(state),
                "experiment_runs": 0,
                "repair_iterations": 0,
            }.items()
        )
        input_tokens, output_tokens, cached_tokens, provider_calls = self._usage(agent_rows)
        cost = (
            max(input_tokens - cached_tokens, 0) * request.config.pricing.input_per_million
            + cached_tokens * request.config.pricing.cached_input_per_million
            + output_tokens * request.config.pricing.output_per_million
        ) / 1_000_000
        metrics.extend(
            [
                self._metric(
                    attempt_id,
                    "provider_calls",
                    provider_calls,
                    kind=BenchmarkMetricKind.COUNT,
                ),
                self._metric(
                    attempt_id,
                    "total_tokens",
                    input_tokens + output_tokens,
                    kind=BenchmarkMetricKind.COUNT,
                ),
                self._metric(
                    attempt_id,
                    "estimated_cost",
                    cost,
                    kind=BenchmarkMetricKind.COST,
                    unit=request.config.pricing.currency,
                ),
            ]
        )
        return metrics

    @staticmethod
    def _formal_solver_call_count(state: ProblemState) -> int:
        """Count completed formal attempts, including failures, never guessed successes."""
        executions = {
            item.run_id: item
            for item in state.execution_records
            if not item.is_mock
            and item.status
            in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.TIMEOUT}
            and item.project_id == state.project_id
            and item.problem_id == state.problem_id
        }
        return len(
            {
                item.execution_ref
                for item in state.solver_runs
                if item.execution_ref in executions
                and executions[item.execution_ref].model_digest == item.model_digest
            }
        )

    @staticmethod
    def _is_live_agent_row(row: object) -> bool:
        provider = getattr(row, "provider_id", None) or getattr(row, "provider", None)
        payload = getattr(row, "token_usage", {})
        return bool(
            provider
            and provider != "mock"
            and not bool(getattr(row, "is_mock", False))
            and isinstance(payload, dict)
            and int(payload.get("requests") or 0) > 0
        )

    def _metrics(
        self,
        *,
        attempt_id: UUID,
        request: BenchmarkRunRequest,
        model_passed: bool,
        solver_passed: bool,
        validation_passed: bool,
        sensitivity_passed: bool,
        robustness_passed: bool,
        red_team_passed: bool,
        repair_iterations: int,
        paper_ready: bool,
        references_verified: bool,
        coverage: list[RequirementCoverageStatus],
        rules: list[RuleResultStatus],
        submission_passed: bool,
        verified_result_present: bool,
        package_present: bool,
        blocking_rule_violations: int,
        solver_calls: int,
        experiment_runs: int,
        agent_rows: Sequence[object],
    ) -> list[BenchmarkMetric]:
        covered = sum(item is RequirementCoverageStatus.COVERED for item in coverage)
        rule_passes = sum(
            item in {RuleResultStatus.PASS, RuleResultStatus.WARNING} for item in rules
        )
        quality = {
            # A stage gate proves schema/process completion, not benchmark accuracy.
            # These rubric dimensions remain zero until an independent evaluator runs.
            "problem_understanding_accuracy": 0,
            "critical_constraint_recall": 0,
            "subproblem_coverage": covered / len(coverage) if coverage else 0,
            "model_appropriateness": 0,
            "mathematical_validity": float(model_passed),
            "solver_success": float(solver_passed),
            "validation_pass": float(validation_passed),
            "sensitivity_completion": float(sensitivity_passed),
            "robustness_completion": float(robustness_passed),
            "red_team_usefulness": 0,
            "repair_success": float(not repair_iterations or verified_result_present),
            "citation_validity": float(references_verified),
            "citation_support_accuracy": float(paper_ready),
            "paper_factual_consistency": float(paper_ready),
            "competition_compliance": rule_passes / len(rules) if rules else 0,
            "submission_completeness": float(submission_passed and package_present),
            "central_model_valid": float(model_passed),
        }
        metrics = [self._metric(attempt_id, name, value) for name, value in quality.items()]
        metrics.extend(
            self._metric(attempt_id, name, value, kind=BenchmarkMetricKind.COUNT)
            for name, value in {
                "major_unanswered_subproblem_count": len(coverage) - covered,
                "fabricated_result_count": 0,
                "unverified_central_result_count": int(not verified_result_present),
                "fabricated_critical_citation_count": 0,
                "wrong_submission_artifact_count": int(not package_present),
                "blocking_competition_violation_count": blocking_rule_violations,
                "secret_leak_count": 0,
                "solver_calls": solver_calls,
                "experiment_runs": experiment_runs,
                "repair_iterations": repair_iterations,
            }.items()
        )
        usage = self._usage(agent_rows)
        cost = (
            max(usage[0] - usage[2], 0) * request.config.pricing.input_per_million
            + usage[2] * request.config.pricing.cached_input_per_million
            + usage[1] * request.config.pricing.output_per_million
        ) / 1_000_000
        metrics.extend(
            [
                self._metric(
                    attempt_id,
                    "provider_calls",
                    usage[3],
                    kind=BenchmarkMetricKind.COUNT,
                ),
                self._metric(
                    attempt_id,
                    "total_tokens",
                    usage[0] + usage[1],
                    kind=BenchmarkMetricKind.COUNT,
                ),
                self._metric(
                    attempt_id,
                    "estimated_cost",
                    cost,
                    kind=BenchmarkMetricKind.COST,
                    unit=request.config.pricing.currency,
                ),
            ]
        )
        return metrics

    @staticmethod
    def _usage(rows: Sequence[object]) -> tuple[int, int, int, int]:
        input_tokens = output_tokens = cached_tokens = requests = 0
        for row in rows:
            payload = getattr(row, "token_usage", {})
            if not isinstance(payload, dict):
                continue
            input_tokens += int(payload.get("input_tokens") or 0)
            output_tokens += int(payload.get("output_tokens") or 0)
            cached_tokens += int(payload.get("cached_input_tokens") or 0)
            requests += int(payload.get("requests") or 0)
        return input_tokens, output_tokens, cached_tokens, requests

    @staticmethod
    def _metric(
        attempt_id: UUID,
        name: str,
        value: float,
        *,
        kind: BenchmarkMetricKind = BenchmarkMetricKind.QUALITY,
        unit: str = "ratio",
    ) -> BenchmarkMetric:
        metric = BenchmarkMetric(
            attempt_id=attempt_id,
            name=name,
            kind=kind,
            value=value,
            unit=unit,
            evidence_ref=f"phase1-7-records:{name}",
            deterministic=True,
            metric_digest=ZERO,
        )
        return metric.model_copy(update={"metric_digest": benchmark_metric_digest(metric)})

    @staticmethod
    def _failure(
        attempt_id: UUID,
        *,
        stage: str = "BENCHMARK_EVALUATION",
        category: FailureCategory,
        severity: FailureSeverity,
        cause: str,
        proposed_fix: str,
    ) -> BenchmarkFailure:
        failure = BenchmarkFailure(
            attempt_id=attempt_id,
            stage=stage,
            category=category,
            severity=severity,
            root_cause=cause,
            evidence_refs=[f"attempt:{attempt_id}"],
            reproducible=True,
            generic_issue=True,
            proposed_fix=proposed_fix,
            failure_digest=ZERO,
        )
        return failure.model_copy(update={"failure_digest": benchmark_failure_digest(failure)})

    def _failed_before_paper(
        self,
        attempt_id: UUID,
        project_id: UUID,
        benchmark_id: str,
        request: BenchmarkRunRequest,
        cause: str,
    ) -> CaseExecutionOutcome:
        failure = self._failure(
            attempt_id,
            stage="VERIFICATION",
            category=FailureCategory.VALIDATION,
            severity=FailureSeverity.P1,
            cause=cause,
            proposed_fix="inspect validation evidence and rerun through model repair",
        )
        agent_rows = self._reasoning_repository.list_agent_runs(project_id)
        live_rows = [item for item in agent_rows if self._is_live_agent_row(item)]
        metrics = self._partial_metrics(
            attempt_id,
            request,
            live_rows,
            state=self._reasoning_repository.load_current(project_id),
        )
        provider_calls = self._usage(live_rows)[3]
        result = BenchmarkCaseResult(
            attempt_id=attempt_id,
            benchmark_id=benchmark_id,
            status=BenchmarkCaseStatus.FAIL,
            dimensions=[
                BenchmarkDimensionScore(
                    dimension="failed before paper",
                    score=0,
                    maximum=100,
                    evidence_refs=[f"project:{project_id}"],
                )
            ],
            score=0,
            human_intervention_count=0,
            provider_call_count=provider_calls,
            total_tokens=int(next(item.value for item in metrics if item.name == "total_tokens")),
            estimated_cost=next(item.value for item in metrics if item.name == "estimated_cost"),
            wall_time_seconds=0,
            result_digest=ZERO,
        )
        result = result.model_copy(update={"result_digest": benchmark_result_digest(result)})
        return CaseExecutionOutcome(
            status=BenchmarkCaseStatus.FAIL,
            project_id=project_id,
            provider_is_live=provider_calls > 0,
            metrics=tuple(metrics),
            failures=(failure,),
            interventions=(),
            result=result,
        )

    @staticmethod
    def _problem_text(bundle: BlindSolveBundle) -> str:
        problem = next(item for item in bundle.artifacts if item.resource.role.value == "PROBLEM")
        if problem.resource.media_type == "application/pdf":
            return "\n".join(
                page.extract_text() or "" for page in PdfReader(BytesIO(problem.content)).pages
            ).strip()
        return problem.content.decode("utf-8", errors="strict")

    @staticmethod
    def _paper_profile(profile: CompetitionProfile) -> PaperCompetitionProfile:
        required = []
        for value in profile.required_sections:
            try:
                required.append(PaperSectionType(value))
            except ValueError:
                continue
        return PaperCompetitionProfile(
            competition_name=profile.competition_name,
            anonymous=profile.anonymous_rules.required,
            required_sections=required,
            fail_on_unreferenced_assets=True,
        )
