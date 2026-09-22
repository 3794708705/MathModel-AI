from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from uuid import UUID

from mathmodel_ai.benchmark.repository import BenchmarkRepository
from mathmodel_ai.core.errors import MathModelError
from mathmodel_ai.data.repository import DataRepository
from mathmodel_ai.files.storage import FileStore
from mathmodel_ai.mathematical.digests import mathematical_model_digest
from mathmodel_ai.mathematical.repository import MathematicalRepository, ResultContext
from mathmodel_ai.schemas.independent_verification import (
    IndependentStatus,
    IndependentVerificationReport,
    IndependentVerificationView,
    MetricBatch,
    ObservationData,
    RawMetricOutput,
    ReplayRecord,
    VerificationPlan,
    VerificationRequirements,
)
from mathmodel_ai.verification.independent_repository import IndependentVerificationRepository
from mathmodel_ai.verification.metric_recompute import content_digest, recompute
from mathmodel_ai.verification.replay_integrity import audit_replay
from mathmodel_ai.verification.scenario_replay import ScenarioReplayer, parse_raw_output

logger = logging.getLogger(__name__)


class IndependentVerificationService:
    def __init__(
        self,
        *,
        repository: IndependentVerificationRepository,
        benchmarks: BenchmarkRepository,
        mathematics: MathematicalRepository,
        data: DataRepository,
        store: FileStore,
        replayer: ScenarioReplayer,
        requirements: Callable[[str], VerificationRequirements | None],
    ):
        self.repository = repository
        self._benchmarks = benchmarks
        self._mathematics = mathematics
        self._data = data
        self._store = store
        self._replayer = replayer
        self._requirements = requirements

    def policy(self, attempt_id: UUID) -> VerificationRequirements:
        attempt = self._benchmarks.get_attempt(attempt_id)
        policy = self._requirements(attempt.benchmark_id)
        if policy is None:
            raise ValueError("MISSING_REVIEWED_METRIC_AND_SCENARIO_DEFINITIONS")
        if policy.manifest_digest != attempt.manifest_digest:
            raise ValueError("VERIFICATION_POLICY_MANIFEST_MISMATCH")
        return policy

    def register(self, plan: VerificationPlan) -> None:
        """Administrator/library boundary. No public endpoint may lower this policy."""
        policy = self.policy(plan.attempt_id)
        self._check_policy(plan, policy)
        self._inputs(plan)
        self.repository.register(plan, content_digest(policy))
        logger.info(
            "Independent verification plan registered", extra={"plan_id": str(plan.plan_id)}
        )

    def prepare_and_run(self, attempt_id: UUID, result_id: UUID) -> None:
        """Bind a reviewed policy to exact evidence and run every declared obligation."""
        attempt = self._benchmarks.get_attempt(attempt_id)
        policy = self._requirements(attempt.benchmark_id)
        if policy is None:
            return
        if policy.manifest_digest != attempt.manifest_digest:
            raise ValueError("VERIFICATION_POLICY_MANIFEST_MISMATCH")
        if attempt.project_id is None:
            raise ValueError("ATTEMPT_HAS_NO_FORMAL_RESULT_PROJECT")
        context = self._mathematics.get_result_context(attempt.project_id, result_id)
        candidates = [
            item
            for item in self._data.list_artifacts(attempt.project_id)
            if item.execution_run_id == context.execution.run_id and item.name == "result.json"
        ]
        if len(candidates) != 1:
            raise ValueError("FORMAL_RAW_OUTPUT_ARTIFACT_IS_NOT_UNIQUE")
        observation = None
        if policy.observation_sha256 is not None:
            matches = [
                item
                for item in self._data.list_files(attempt.project_id)
                if item.sha256 == policy.observation_sha256
            ]
            if len(matches) != 1:
                raise ValueError("REVIEWED_OBSERVATION_FILE_IS_NOT_UNIQUE")
            observation = matches[0]
        plan = VerificationPlan(
            attempt_id=attempt_id,
            result_id=result_id,
            version="1",
            source_artifact_id=candidates[0].artifact_id,
            source_sha256=candidates[0].sha256,
            observation_file_id=observation.file_id if observation else None,
            observation_sha256=observation.sha256 if observation else None,
            metrics=policy.metrics,
            scenarios=policy.scenarios,
            scientific_scope=policy.scientific_scope,
        )
        self.register(plan)
        self.recompute(attempt_id)
        for scenario in policy.scenarios:
            self.replay(attempt_id, scenario.scenario_id)

    @staticmethod
    def _check_policy(plan: VerificationPlan, policy: VerificationRequirements) -> None:
        if (
            plan.metrics != policy.metrics
            or plan.scenarios != policy.scenarios
            or plan.scientific_scope != policy.scientific_scope
        ):
            raise ValueError("PLAN_DOES_NOT_COVER_EXACT_REVIEWED_REQUIREMENTS")

    def _plan(self, attempt_id: UUID) -> VerificationPlan:
        policy = self.policy(attempt_id)
        plan, policy_digest = self.repository.get(attempt_id)
        self._check_policy(plan, policy)
        if policy_digest != content_digest(policy):
            raise ValueError("STALE_VERIFICATION_POLICY")
        return plan

    def _inputs(
        self, plan: VerificationPlan
    ) -> tuple[ResultContext, RawMetricOutput, list[float] | None]:
        attempt = self._benchmarks.get_attempt(plan.attempt_id)
        if attempt.project_id is None:
            raise ValueError("ATTEMPT_HAS_NO_FORMAL_RESULT_PROJECT")
        context = self._mathematics.get_result_context(attempt.project_id, plan.result_id)
        policy = self.policy(plan.attempt_id)
        if mathematical_model_digest(context.model) != policy.model_digest:
            raise ValueError("VERIFICATION_POLICY_MODEL_MISMATCH")
        if policy.problem_sha256 is not None:
            problem_files = [
                item
                for item in self._data.list_files(attempt.project_id)
                if item.sha256 == policy.problem_sha256
            ]
            if len(problem_files) != 1:
                raise ValueError("REVIEWED_PROBLEM_FILE_IS_NOT_UNIQUE")
        if not context.evidence.valid or context.execution.is_mock:
            raise ValueError("FORMAL_RESULT_EVIDENCE_INVALID_OR_MOCK")
        artifacts = {a.artifact_id: a for a in self._data.list_artifacts(attempt.project_id)}
        artifact = artifacts.get(plan.source_artifact_id)
        if (
            artifact is None
            or artifact.sha256 != plan.source_sha256
            or artifact.execution_run_id != context.execution.run_id
            or not any(
                a.artifact_id == artifact.artifact_id and a.sha256 == artifact.sha256
                for a in context.execution.artifacts
            )
        ):
            raise ValueError("RAW_OUTPUT_NOT_BOUND_TO_FORMAL_EXECUTION")
        for item in (artifact, artifacts.get(context.execution.code_artifact_id)):
            if item is None:
                raise ValueError("FORMAL_CODE_ARTIFACT_MISSING")
            raw_bytes = self._store.read_bytes(item.storage_key, max_bytes=16 * 1024 * 1024)
            if (
                len(raw_bytes) != item.size_bytes
                or hashlib.sha256(raw_bytes).hexdigest() != item.sha256
            ):
                raise ValueError("FORMAL_ARTIFACT_HASH_MISMATCH")
            if (
                item.artifact_id == context.execution.code_artifact_id
                and item.sha256 != context.execution.code_hash
            ):
                raise ValueError("FORMAL_EXECUTED_CODE_HASH_MISMATCH")
        raw = parse_raw_output(
            self._store.read_bytes(artifact.storage_key, max_bytes=16 * 1024 * 1024)
        )
        if any(raw.variables.get(k) != v for k, v in context.result.key_outputs.items()):
            raise ValueError("RAW_OUTPUT_AND_FORMAL_RESULT_DISAGREE")
        observations = None
        if plan.observation_file_id:
            file = self._data.get_file(attempt.project_id, plan.observation_file_id)
            data = self._store.read_bytes(file.storage_key, max_bytes=16 * 1024 * 1024)
            if (
                file.sha256 != plan.observation_sha256
                or hashlib.sha256(data).hexdigest() != plan.observation_sha256
            ):
                raise ValueError("OBSERVATION_FILE_HASH_MISMATCH")
            observations = ObservationData.model_validate_json(data).observations
        return context, raw, observations

    def recompute(self, attempt_id: UUID) -> IndependentVerificationView:
        plan = self._plan(attempt_id)
        context, raw, observations = self._inputs(plan)
        job = self.repository.claim(plan.plan_id, "baseline")
        if job is not None:
            metrics = [
                recompute(
                    spec,
                    raw,
                    source_digest=plan.source_sha256,
                    model=context.model,
                    observations=observations,
                )
                for spec in plan.metrics
            ]
            self.repository.finish(
                job, {"batch": MetricBatch(metrics=metrics).model_dump(mode="json")}
            )
            logger.info("Independent metric job completed", extra={"job_id": str(job)})
        return self.view(attempt_id)

    def replay(self, attempt_id: UUID, scenario_id: str) -> IndependentVerificationView:
        plan = self._plan(attempt_id)
        context, _, observations = self._inputs(plan)
        spec = next((s for s in plan.scenarios if s.scenario_id == scenario_id), None)
        if spec is None:
            raise ValueError("UNKNOWN_REQUIRED_SCENARIO")
        job = self.repository.claim(plan.plan_id, f"scenario:{scenario_id}")
        if job is not None:
            try:
                replay = self._replayer.execute(context.model, spec, observations=observations)
                self.repository.finish(
                    job, {"replay": replay.model_dump(mode="json")}, replay=replay
                )
            except (MathModelError, ValueError, OSError) as exc:
                # Log only category/identity; untrusted filenames or payloads may contain secrets.
                self.repository.finish(job, {"error": f"REPLAY_FAILED:{type(exc).__name__}"})
            logger.info("Independent replay job completed", extra={"job_id": str(job)})
        return self.view(attempt_id)

    def view(self, attempt_id: UUID) -> IndependentVerificationView:
        self._benchmarks.get_attempt(attempt_id)
        try:
            plan = self._plan(attempt_id)
            report = self._report(plan)
            return IndependentVerificationView(
                attempt_id=attempt_id,
                status=report.status,
                plan_id=plan.plan_id,
                plan=plan,
                scenario_ids=[s.scenario_id for s in plan.scenarios],
                report=report,
                blockers=report.errors,
            )
        except (MathModelError, ValueError, OSError) as exc:
            blocker = (
                str(exc)
                if isinstance(exc, ValueError) and str(exc).isupper()
                else type(exc).__name__
            )
            return IndependentVerificationView(
                attempt_id=attempt_id, status=IndependentStatus.NOT_READY, blockers=[blocker]
            )

    def _report(self, plan: VerificationPlan) -> IndependentVerificationReport:
        context, raw, observations = self._inputs(plan)
        jobs = self.repository.jobs(plan.plan_id)
        metrics = [
            recompute(
                spec,
                raw,
                source_digest=plan.source_sha256,
                model=context.model,
                observations=observations,
            )
            for spec in plan.metrics
        ]
        errors = []
        baseline = jobs.get("baseline")
        if baseline is None:
            errors.append("BASELINE_RECOMPUTATION_MISSING_OR_RUNNING")
        elif MetricBatch.model_validate(baseline.get("batch")).metrics != metrics:
            errors.append("BASELINE_METRICS_STALE_OR_TAMPERED")
        replays = []
        used_executions = {context.execution.run_id}
        passed_scenarios = 0
        persisted_executions = {
            e.run_id: e for e in self._data.list_executions(context.model.project_id)
        }
        persisted_artifacts = {
            a.artifact_id: a for a in self._data.list_artifacts(context.model.project_id)
        }
        for spec in plan.scenarios:
            payload = jobs.get(f"scenario:{spec.scenario_id}")
            if payload is None or "replay" not in payload:
                if spec.required:
                    errors.append(f"SCENARIO_MISSING_RUNNING_OR_FAILED:{spec.scenario_id}")
                continue
            replay = ReplayRecord.model_validate(payload["replay"])
            issues = audit_replay(context.model, spec, replay, self._store)
            if persisted_executions.get(replay.execution.run_id) != replay.execution or any(
                persisted_artifacts.get(a.artifact_id) is None
                or persisted_artifacts[a.artifact_id].model_dump(exclude={"created_at"})
                != a.model_dump(exclude={"created_at"})
                for a in replay.artifacts
            ):
                issues.append("REPLAY_PERSISTED_EVIDENCE_MISMATCH")
            if replay.execution.run_id in used_executions:
                issues.append("REPLAY_REUSED_EXECUTION")
            used_executions.add(replay.execution.run_id)
            outputs = next(
                (a for a in replay.artifacts if a.artifact_id == replay.output_artifact_id), None
            )
            verified = []
            if outputs and not issues:
                scenario_model = ScenarioReplayer.scenario_model(
                    context.model, ScenarioReplayer.parameters(context.model, spec)
                )
                raw_replay = parse_raw_output(
                    self._store.read_bytes(outputs.storage_key, max_bytes=16 * 1024 * 1024)
                )
                verified = [
                    recompute(
                        m,
                        raw_replay,
                        source_digest=outputs.sha256,
                        model=scenario_model,
                        observations=observations,
                    )
                    for m in spec.metrics
                ]
            if replay.metrics != verified:
                issues.append("REPLAY_METRICS_STALE_OR_TAMPERED")
            required_ids = {m.metric_id for m in spec.metrics if m.required}
            valid = (
                not issues
                and len(verified) == len(spec.metrics)
                and all(
                    m.status is IndependentStatus.PASS
                    for m in verified
                    if m.metric_id in required_ids
                )
            )
            if spec.required:
                passed_scenarios += int(valid)
                if not valid:
                    errors.extend([f"SCENARIO_FAILED:{spec.scenario_id}", *issues])
            replays.append(
                replay.model_copy(
                    update={
                        "metrics": verified,
                        "status": IndependentStatus.PASS if valid else IndependentStatus.FAIL,
                    }
                )
            )
        required_ids = {s.metric_id for s in plan.metrics if s.required}
        passed = sum(
            m.status is IndependentStatus.PASS for m in metrics if m.metric_id in required_ids
        )
        if passed != len(required_ids):
            errors.append("REQUIRED_INDEPENDENT_METRIC_FAILED_OR_INVALID")
        status = IndependentStatus.PASS if not errors else IndependentStatus.NOT_READY
        return IndependentVerificationReport(
            report_id=plan.plan_id,
            attempt_id=plan.attempt_id,
            plan_id=plan.plan_id,
            plan_digest=content_digest(plan),
            result_id=plan.result_id,
            status=status,
            metrics=metrics,
            replays=replays,
            required_metrics=len(required_ids),
            passed_metrics=passed,
            required_scenarios=sum(s.required for s in plan.scenarios),
            passed_scenarios=passed_scenarios,
            errors=list(dict.fromkeys(errors)),
            created_at=context.execution.end_time,
        )
