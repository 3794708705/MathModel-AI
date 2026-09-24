from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from mathmodel_ai.core.errors import ResourceNotFoundError
from mathmodel_ai.db.models import (
    AgentRunRecord,
    BenchmarkAttemptRecordModel,
    BenchmarkCaseResultRecordModel,
    BenchmarkFailureRecordModel,
    BenchmarkHumanInterventionRecordModel,
    BenchmarkMetricRecordModel,
    BenchmarkRunRecordModel,
    FinalJuryReportRecord,
    PaperVersionRecord,
    Project,
)
from mathmodel_ai.db.session import session_scope
from mathmodel_ai.schemas.benchmark import (
    BenchmarkAttempt,
    BenchmarkCaseResult,
    BenchmarkFailure,
    BenchmarkHumanIntervention,
    BenchmarkMetric,
    BenchmarkRun,
    BenchmarkRunStatus,
    benchmark_attempt_digest,
    benchmark_failure_digest,
    benchmark_intervention_digest,
    benchmark_metric_digest,
    benchmark_result_digest,
    benchmark_run_digest,
)


class BenchmarkIntegrityError(ValueError):
    """Persisted benchmark history conflicts with its immutable digests or identities."""


class BenchmarkRepository:
    REQUIRED_LIVE_AGENT_NAMES = frozenset(
        {
            "problem_agent",
            "model_explorer",
            "model_jury",
            "math_modeler",
            "paper_agent",
            "final_jury_agent",
        }
    )
    REVIEWED_MODEL_LIVE_AGENT_NAMES = frozenset(
        {
            "problem_agent",
            "model_explorer",
            "model_jury",
            "code_agent",
            "paper_agent",
            "final_jury_agent",
        }
    )

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def create_run(self, run: BenchmarkRun) -> None:
        self._validate_run(run)
        if run.status is not BenchmarkRunStatus.RUNNING:
            raise BenchmarkIntegrityError("new benchmark run must start in RUNNING state")
        with session_scope(self._session_factory) as session:
            if session.get(BenchmarkRunRecordModel, run.run_id) is not None:
                raise BenchmarkIntegrityError("benchmark run identity already exists")
            session.add(self._run_row(run))

    def finish_run(self, run: BenchmarkRun) -> None:
        self._validate_run(run)
        if run.status not in {
            BenchmarkRunStatus.SELF_TEST_READY,
            BenchmarkRunStatus.NOT_READY,
            BenchmarkRunStatus.CANCELLED_BY_HUMAN,
        }:
            raise BenchmarkIntegrityError("benchmark run can only finish in a terminal state")
        with session_scope(self._session_factory) as session:
            row = session.get(BenchmarkRunRecordModel, run.run_id, with_for_update=True)
            if row is None:
                raise ResourceNotFoundError("benchmark run was not found")
            current = self._validated_run(row)
            if current.status is not BenchmarkRunStatus.RUNNING:
                raise BenchmarkIntegrityError("terminal benchmark runs are immutable")
            if benchmark_run_digest(current) != benchmark_run_digest(run):
                raise BenchmarkIntegrityError("benchmark run configuration changed while running")
            self._assign_run(row, run)

    def create_attempt(self, attempt: BenchmarkAttempt) -> None:
        self._validate_attempt(attempt)
        if attempt.status.value != "RUNNING":
            raise BenchmarkIntegrityError("new benchmark attempt must start in RUNNING state")
        with session_scope(self._session_factory) as session:
            run = session.get(BenchmarkRunRecordModel, attempt.run_id)
            if run is None:
                raise ResourceNotFoundError("parent benchmark run was not found")
            if self._validated_run(run).status is not BenchmarkRunStatus.RUNNING:
                raise BenchmarkIntegrityError("cannot append an attempt to a terminal run")
            if session.get(BenchmarkAttemptRecordModel, attempt.attempt_id) is not None:
                raise BenchmarkIntegrityError("benchmark attempt identity already exists")
            session.add(self._attempt_row(attempt))

    def bind_attempt_project(self, attempt_id: UUID, project_id: UUID) -> BenchmarkAttempt:
        """Persist project provenance as soon as a live case creates its project."""
        with session_scope(self._session_factory) as session:
            row = session.get(BenchmarkAttemptRecordModel, attempt_id, with_for_update=True)
            if row is None:
                raise ResourceNotFoundError("benchmark attempt was not found")
            if session.get(Project, project_id) is None:
                raise ResourceNotFoundError("benchmark project was not found")
            current = self._validated_attempt(row)
            if current.status.value != "RUNNING":
                raise BenchmarkIntegrityError("terminal benchmark attempts are immutable")
            if current.project_id is not None:
                if current.project_id == project_id:
                    return current
                raise BenchmarkIntegrityError("benchmark attempt is already bound to a project")
            bound = current.model_copy(update={"project_id": project_id})
            bound = bound.model_copy(update={"attempt_digest": benchmark_attempt_digest(bound)})
            self._assign_attempt(row, bound)
            return bound

    def finish_attempt(self, attempt: BenchmarkAttempt) -> None:
        self._validate_attempt(attempt)
        if attempt.status.value in {"PENDING", "RUNNING"}:
            raise BenchmarkIntegrityError("benchmark attempt did not reach a terminal state")
        with session_scope(self._session_factory) as session:
            row = session.get(BenchmarkAttemptRecordModel, attempt.attempt_id, with_for_update=True)
            if row is None:
                raise ResourceNotFoundError("benchmark attempt was not found")
            current = self._validated_attempt(row)
            if current.status.value != "RUNNING":
                raise BenchmarkIntegrityError("terminal benchmark attempts are immutable")
            immutable = (
                "run_id",
                "benchmark_id",
                "manifest_digest",
                "attempt_number",
                "official",
                "solve_input_digest",
                "started_at",
            )
            if any(getattr(current, field) != getattr(attempt, field) for field in immutable):
                raise BenchmarkIntegrityError("benchmark attempt identity changed while running")
            self._assign_attempt(row, attempt)

    def add_metric(self, metric: BenchmarkMetric) -> None:
        if benchmark_metric_digest(metric) != metric.metric_digest:
            raise BenchmarkIntegrityError("benchmark metric digest is invalid")
        with session_scope(self._session_factory) as session:
            self._require_attempt(session, metric.attempt_id)
            session.add(
                BenchmarkMetricRecordModel(
                    id=metric.metric_id,
                    attempt_id=metric.attempt_id,
                    name=metric.name,
                    kind=metric.kind.value,
                    value=metric.value,
                    metric_digest=metric.metric_digest,
                    metric_json=metric.model_dump(mode="json"),
                    recorded_at=metric.recorded_at,
                )
            )

    def add_failure(self, failure: BenchmarkFailure) -> None:
        if benchmark_failure_digest(failure) != failure.failure_digest:
            raise BenchmarkIntegrityError("benchmark failure digest is invalid")
        with session_scope(self._session_factory) as session:
            self._require_attempt(session, failure.attempt_id)
            session.add(
                BenchmarkFailureRecordModel(
                    id=failure.failure_id,
                    attempt_id=failure.attempt_id,
                    category=failure.category.value,
                    severity=failure.severity.value,
                    generic_issue=failure.generic_issue,
                    failure_digest=failure.failure_digest,
                    failure_json=failure.model_dump(mode="json"),
                    created_at=failure.created_at,
                )
            )

    def add_intervention(self, intervention: BenchmarkHumanIntervention) -> None:
        if benchmark_intervention_digest(intervention) != intervention.intervention_digest:
            raise BenchmarkIntegrityError("benchmark intervention digest is invalid")
        with session_scope(self._session_factory) as session:
            self._require_attempt(session, intervention.attempt_id)
            session.add(
                BenchmarkHumanInterventionRecordModel(
                    id=intervention.intervention_id,
                    attempt_id=intervention.attempt_id,
                    intervention_type=intervention.intervention_type.value,
                    intervention_digest=intervention.intervention_digest,
                    intervention_json=intervention.model_dump(mode="json"),
                    created_at=intervention.created_at,
                )
            )

    def add_result(self, result: BenchmarkCaseResult) -> None:
        if benchmark_result_digest(result) != result.result_digest:
            raise BenchmarkIntegrityError("benchmark result digest is invalid")
        with session_scope(self._session_factory) as session:
            attempt_row = self._require_attempt(session, result.attempt_id)
            attempt = self._validated_attempt(attempt_row)
            if attempt.status.value in {"PENDING", "RUNNING"}:
                raise BenchmarkIntegrityError("cannot score an unfinished benchmark attempt")
            if result.benchmark_id != attempt.benchmark_id or result.status is not attempt.status:
                raise BenchmarkIntegrityError("benchmark result does not match its attempt")
            session.add(
                BenchmarkCaseResultRecordModel(
                    id=result.result_id,
                    attempt_id=result.attempt_id,
                    benchmark_id=result.benchmark_id,
                    status=result.status.value,
                    score=result.score,
                    result_digest=result.result_digest,
                    result_json=result.model_dump(mode="json"),
                    created_at=result.created_at,
                )
            )

    def get_run(self, run_id: UUID) -> BenchmarkRun:
        with session_scope(self._session_factory) as session:
            row = session.get(BenchmarkRunRecordModel, run_id)
            if row is None:
                raise ResourceNotFoundError("benchmark run was not found")
            return self._validated_run(row)

    def list_runs(self, *, limit: int = 50) -> list[BenchmarkRun]:
        if limit < 1 or limit > 100:
            raise ValueError("benchmark history limit must be between 1 and 100")
        with session_scope(self._session_factory) as session:
            rows = session.scalars(
                select(BenchmarkRunRecordModel)
                .order_by(BenchmarkRunRecordModel.started_at.desc())
                .limit(limit)
            )
            return [self._validated_run(row) for row in rows]

    def get_attempt(self, attempt_id: UUID) -> BenchmarkAttempt:
        with session_scope(self._session_factory) as session:
            return self._validated_attempt(self._require_attempt(session, attempt_id))

    def list_attempts(self, run_id: UUID) -> list[BenchmarkAttempt]:
        with session_scope(self._session_factory) as session:
            rows = session.scalars(
                select(BenchmarkAttemptRecordModel)
                .where(BenchmarkAttemptRecordModel.run_id == run_id)
                .order_by(
                    BenchmarkAttemptRecordModel.benchmark_id,
                    BenchmarkAttemptRecordModel.attempt_number,
                    BenchmarkAttemptRecordModel.started_at,
                )
            )
            return [self._validated_attempt(row) for row in rows]

    def list_metrics(self, run_id: UUID) -> list[BenchmarkMetric]:
        with session_scope(self._session_factory) as session:
            rows = session.scalars(
                select(BenchmarkMetricRecordModel)
                .join(
                    BenchmarkAttemptRecordModel,
                    BenchmarkAttemptRecordModel.id == BenchmarkMetricRecordModel.attempt_id,
                )
                .where(BenchmarkAttemptRecordModel.run_id == run_id)
                .order_by(BenchmarkMetricRecordModel.recorded_at, BenchmarkMetricRecordModel.name)
            )
            metrics = [self._validated_metric(row) for row in rows]
            for metric in metrics:
                if metric.name == "live_provider_critical_agent_coverage":
                    actual, _ = self._provider_coverage_for_attempt(session, metric.attempt_id)
                    if metric.value != actual:
                        raise BenchmarkIntegrityError(
                            "persisted live-provider coverage conflicts with agent runs"
                        )
            return metrics

    def live_provider_coverage(self, project_id: UUID) -> tuple[float, list[str]]:
        with session_scope(self._session_factory) as session:
            return self._provider_coverage_for_project(session, project_id)

    def live_provider_activity(self, project_id: UUID) -> tuple[bool, list[str]]:
        with session_scope(self._session_factory) as session:
            rows = session.scalars(
                select(AgentRunRecord).where(AgentRunRecord.project_id == project_id)
            )
            accepted = [
                row
                for row in rows
                if not row.is_mock
                and row.provider is not None
                and row.provider != "mock"
                and isinstance(row.token_usage, dict)
                and int(row.token_usage.get("requests") or 0) > 0
            ]
            return bool(accepted), [f"agent-run:{row.id}" for row in accepted]

    def list_failures(self, run_id: UUID) -> list[BenchmarkFailure]:
        with session_scope(self._session_factory) as session:
            rows = session.scalars(
                select(BenchmarkFailureRecordModel)
                .join(
                    BenchmarkAttemptRecordModel,
                    BenchmarkAttemptRecordModel.id == BenchmarkFailureRecordModel.attempt_id,
                )
                .where(BenchmarkAttemptRecordModel.run_id == run_id)
                .order_by(BenchmarkFailureRecordModel.created_at)
            )
            return [self._validated_failure(row) for row in rows]

    def list_interventions(self, run_id: UUID) -> list[BenchmarkHumanIntervention]:
        with session_scope(self._session_factory) as session:
            rows = session.scalars(
                select(BenchmarkHumanInterventionRecordModel)
                .join(
                    BenchmarkAttemptRecordModel,
                    BenchmarkAttemptRecordModel.id
                    == BenchmarkHumanInterventionRecordModel.attempt_id,
                )
                .where(BenchmarkAttemptRecordModel.run_id == run_id)
                .order_by(BenchmarkHumanInterventionRecordModel.created_at)
            )
            return [self._validated_intervention(row) for row in rows]

    def list_results(self, run_id: UUID) -> list[BenchmarkCaseResult]:
        with session_scope(self._session_factory) as session:
            rows = session.scalars(
                select(BenchmarkCaseResultRecordModel)
                .join(
                    BenchmarkAttemptRecordModel,
                    BenchmarkAttemptRecordModel.id == BenchmarkCaseResultRecordModel.attempt_id,
                )
                .where(BenchmarkAttemptRecordModel.run_id == run_id)
                .order_by(BenchmarkCaseResultRecordModel.created_at)
            )
            return [self._validated_result(row) for row in rows]

    @staticmethod
    def _require_attempt(session: Session, attempt_id: UUID) -> BenchmarkAttemptRecordModel:
        row = session.get(BenchmarkAttemptRecordModel, attempt_id)
        if row is None:
            raise ResourceNotFoundError("benchmark attempt was not found")
        return row

    @classmethod
    def _provider_coverage_for_attempt(
        cls, session: Session, attempt_id: UUID
    ) -> tuple[float, list[str]]:
        attempt = cls._require_attempt(session, attempt_id)
        if attempt.project_id is None:
            return 0, []
        return cls._provider_coverage_for_project(session, attempt.project_id)

    @classmethod
    def _provider_coverage_for_project(
        cls, session: Session, project_id: UUID
    ) -> tuple[float, list[str]]:
        rows = list(
            session.scalars(select(AgentRunRecord).where(AgentRunRecord.project_id == project_id))
        )
        reviewed_binding = any(
            row.agent_name == "reviewed_model_binder"
            and row.status == "SUCCEEDED"
            and not row.is_mock
            and row.provider is None
            and row.provider_id is None
            and int((row.token_usage or {}).get("requests") or 0) == 0
            for row in rows
        )
        required_names = (
            cls.REVIEWED_MODEL_LIVE_AGENT_NAMES
            if reviewed_binding
            else cls.REQUIRED_LIVE_AGENT_NAMES
        )
        bound_paper_runs = set(
            session.scalars(
                select(PaperVersionRecord.paper_agent_run_id).where(
                    PaperVersionRecord.project_id == project_id,
                    PaperVersionRecord.paper_agent_run_id.is_not(None),
                )
            )
        )
        bound_jury_runs = set(
            session.scalars(
                select(FinalJuryReportRecord.agent_run_id).where(
                    FinalJuryReportRecord.project_id == project_id,
                    FinalJuryReportRecord.agent_run_id.is_not(None),
                )
            )
        )
        accepted = {
            row.agent_name: row
            for row in rows
            if row.agent_name in required_names
            and row.status == "SUCCEEDED"
            and not row.is_mock
            and row.provider is not None
            and row.provider != "mock"
            and (row.agent_name != "paper_agent" or row.id in bound_paper_runs)
            and (row.agent_name != "final_jury_agent" or row.id in bound_jury_runs)
        }
        coverage = len(accepted) / len(required_names)
        return coverage, [f"agent-run:{accepted[name].id}" for name in sorted(accepted)]

    @staticmethod
    def _validate_run(run: BenchmarkRun) -> None:
        if benchmark_run_digest(run) != run.run_digest:
            raise BenchmarkIntegrityError("benchmark run digest is invalid")

    @staticmethod
    def _validate_attempt(attempt: BenchmarkAttempt) -> None:
        if benchmark_attempt_digest(attempt) != attempt.attempt_digest:
            raise BenchmarkIntegrityError("benchmark attempt digest is invalid")

    @staticmethod
    def _run_row(run: BenchmarkRun) -> BenchmarkRunRecordModel:
        return BenchmarkRunRecordModel(
            id=run.run_id,
            code_commit=run.code_commit,
            source_tree_digest=run.source_tree_digest,
            working_tree_dirty=run.working_tree_dirty,
            status=run.status.value,
            provider=run.config.provider,
            model=run.config.model,
            provider_id=run.config.provider_id,
            provider_config_digest=run.config.provider_config_digest,
            model_id=run.config.model_id,
            model_config_digest=run.config.model_config_digest,
            protocol=run.config.protocol,
            endpoint_trust=run.config.endpoint_trust,
            model_identity_confidence=run.config.model_identity_confidence,
            live_provider_status=run.live_provider_status.value,
            live_literature_status=run.live_literature_status.value,
            run_digest=run.run_digest,
            run_json=run.model_dump(mode="json"),
            started_at=run.started_at,
            finished_at=run.finished_at,
        )

    @staticmethod
    def _assign_run(row: BenchmarkRunRecordModel, run: BenchmarkRun) -> None:
        row.status = run.status.value
        row.live_provider_status = run.live_provider_status.value
        row.live_literature_status = run.live_literature_status.value
        row.run_digest = run.run_digest
        row.run_json = run.model_dump(mode="json")
        row.finished_at = run.finished_at

    @staticmethod
    def _attempt_row(attempt: BenchmarkAttempt) -> BenchmarkAttemptRecordModel:
        return BenchmarkAttemptRecordModel(
            id=attempt.attempt_id,
            run_id=attempt.run_id,
            benchmark_id=attempt.benchmark_id,
            attempt_number=attempt.attempt_number,
            status=attempt.status.value,
            project_id=attempt.project_id,
            official=attempt.official,
            solve_input_digest=attempt.solve_input_digest,
            attempt_digest=attempt.attempt_digest,
            attempt_json=attempt.model_dump(mode="json"),
            started_at=attempt.started_at,
            finished_at=attempt.finished_at,
        )

    @staticmethod
    def _assign_attempt(row: BenchmarkAttemptRecordModel, attempt: BenchmarkAttempt) -> None:
        row.status = attempt.status.value
        row.project_id = attempt.project_id
        row.solve_input_digest = attempt.solve_input_digest
        row.attempt_digest = attempt.attempt_digest
        row.attempt_json = attempt.model_dump(mode="json")
        row.finished_at = attempt.finished_at

    @staticmethod
    def _validated_run(row: BenchmarkRunRecordModel) -> BenchmarkRun:
        run = BenchmarkRun.model_validate(row.run_json)
        if (
            row.id != run.run_id
            or row.code_commit != run.code_commit
            or row.source_tree_digest != run.source_tree_digest
            or row.working_tree_dirty != run.working_tree_dirty
            or row.status != run.status.value
            or row.live_provider_status != run.live_provider_status.value
            or row.live_literature_status != run.live_literature_status.value
            or row.run_digest != run.run_digest
            or benchmark_run_digest(run) != run.run_digest
        ):
            raise BenchmarkIntegrityError("persisted benchmark run was modified")
        return run

    @staticmethod
    def _validated_attempt(row: BenchmarkAttemptRecordModel) -> BenchmarkAttempt:
        attempt = BenchmarkAttempt.model_validate(row.attempt_json)
        if (
            row.id != attempt.attempt_id
            or row.run_id != attempt.run_id
            or row.status != attempt.status.value
            or row.solve_input_digest != attempt.solve_input_digest
            or row.attempt_digest != attempt.attempt_digest
            or benchmark_attempt_digest(attempt) != attempt.attempt_digest
        ):
            raise BenchmarkIntegrityError("persisted benchmark attempt was modified")
        return attempt

    @staticmethod
    def _validated_metric(row: BenchmarkMetricRecordModel) -> BenchmarkMetric:
        metric = BenchmarkMetric.model_validate(row.metric_json)
        if (
            row.id != metric.metric_id
            or row.attempt_id != metric.attempt_id
            or row.name != metric.name
            or row.value != metric.value
            or row.metric_digest != metric.metric_digest
            or benchmark_metric_digest(metric) != metric.metric_digest
        ):
            raise BenchmarkIntegrityError("persisted benchmark metric was modified")
        return metric

    @staticmethod
    def _validated_failure(row: BenchmarkFailureRecordModel) -> BenchmarkFailure:
        failure = BenchmarkFailure.model_validate(row.failure_json)
        if (
            row.id != failure.failure_id
            or row.attempt_id != failure.attempt_id
            or row.severity != failure.severity.value
            or row.failure_digest != failure.failure_digest
            or benchmark_failure_digest(failure) != failure.failure_digest
        ):
            raise BenchmarkIntegrityError("persisted benchmark failure was modified")
        return failure

    @staticmethod
    def _validated_intervention(
        row: BenchmarkHumanInterventionRecordModel,
    ) -> BenchmarkHumanIntervention:
        intervention = BenchmarkHumanIntervention.model_validate(row.intervention_json)
        if (
            row.id != intervention.intervention_id
            or row.attempt_id != intervention.attempt_id
            or row.intervention_digest != intervention.intervention_digest
            or benchmark_intervention_digest(intervention) != intervention.intervention_digest
        ):
            raise BenchmarkIntegrityError("persisted benchmark intervention was modified")
        return intervention

    @staticmethod
    def _validated_result(row: BenchmarkCaseResultRecordModel) -> BenchmarkCaseResult:
        result = BenchmarkCaseResult.model_validate(row.result_json)
        if (
            row.id != result.result_id
            or row.attempt_id != result.attempt_id
            or row.status != result.status.value
            or row.score != result.score
            or row.result_digest != result.result_digest
            or benchmark_result_digest(result) != result.result_digest
        ):
            raise BenchmarkIntegrityError("persisted benchmark result was modified")
        return result
