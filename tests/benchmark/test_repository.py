from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy import Engine, create_engine, update
from sqlalchemy.orm import Session, sessionmaker

from mathmodel_ai.agents.base import AgentRunResult, AgentRunStatus
from mathmodel_ai.benchmark.repository import BenchmarkIntegrityError, BenchmarkRepository
from mathmodel_ai.db.base import Base
from mathmodel_ai.db.models import BenchmarkCaseResultRecordModel
from mathmodel_ai.providers.schemas import ModelUsage
from mathmodel_ai.reasoning.repository import ReasoningRepository
from mathmodel_ai.schemas.benchmark import (
    BenchmarkAttempt,
    BenchmarkCaseResult,
    BenchmarkCaseStatus,
    BenchmarkConfig,
    BenchmarkDimensionScore,
    BenchmarkMetric,
    BenchmarkMetricKind,
    BenchmarkPricing,
    BenchmarkRun,
    BenchmarkRunStatus,
    benchmark_attempt_digest,
    benchmark_metric_digest,
    benchmark_result_digest,
    benchmark_run_digest,
)

ZERO = "0" * 64


@pytest.fixture
def persistence() -> tuple[BenchmarkRepository, Engine]:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    return BenchmarkRepository(sessionmaker(engine, expire_on_commit=False)), engine


def test_terminal_attempt_is_immutable_and_all_retries_remain_visible(
    persistence: tuple[BenchmarkRepository, Engine],
) -> None:
    repository, _ = persistence
    run = _running_run()
    repository.create_run(run)
    first = _running_attempt(run, 1)
    second = _running_attempt(run, 2)
    repository.create_attempt(first)
    repository.finish_attempt(_terminal(first, BenchmarkCaseStatus.FAIL))
    repository.create_attempt(second)
    terminal_second = _terminal(second, BenchmarkCaseStatus.BLOCKED_ENVIRONMENT)
    repository.finish_attempt(terminal_second)

    with pytest.raises(BenchmarkIntegrityError, match="immutable"):
        repository.finish_attempt(terminal_second)

    attempts = repository.list_attempts(run.run_id)
    assert [item.attempt_number for item in attempts] == [1, 2]
    assert [item.status for item in attempts] == [
        BenchmarkCaseStatus.FAIL,
        BenchmarkCaseStatus.BLOCKED_ENVIRONMENT,
    ]


def test_running_attempt_binds_project_once_before_pipeline_completion(
    persistence: tuple[BenchmarkRepository, Engine],
) -> None:
    repository, engine = persistence
    run = _running_run()
    repository.create_run(run)
    attempt = _running_attempt(run, 1)
    repository.create_attempt(attempt)
    reasoning = ReasoningRepository(sessionmaker(engine, expire_on_commit=False))
    project = reasoning.create_project_problem(
        name="crash-safe benchmark",
        title="Test",
        raw_problem="Test problem",
    )

    bound = repository.bind_attempt_project(attempt.attempt_id, project.project_id)

    assert bound.project_id == project.project_id
    assert repository.list_attempts(run.run_id)[0].project_id == project.project_id
    assert repository.bind_attempt_project(attempt.attempt_id, project.project_id) == bound

    other = reasoning.create_project_problem(
        name="unrelated benchmark",
        title="Other",
        raw_problem="Other problem",
    )
    with pytest.raises(BenchmarkIntegrityError, match="already bound"):
        repository.bind_attempt_project(attempt.attempt_id, other.project_id)

    repository.finish_attempt(_terminal(bound, BenchmarkCaseStatus.FAIL))
    with pytest.raises(BenchmarkIntegrityError, match="immutable"):
        repository.bind_attempt_project(attempt.attempt_id, project.project_id)


def test_repository_detects_persisted_result_score_tamper(
    persistence: tuple[BenchmarkRepository, Engine],
) -> None:
    repository, engine = persistence
    run = _running_run()
    repository.create_run(run)
    attempt = _running_attempt(run, 1)
    repository.create_attempt(attempt)
    terminal = _terminal(attempt, BenchmarkCaseStatus.FAIL)
    repository.finish_attempt(terminal)
    result = _result(terminal)
    repository.add_result(result)

    with Session(engine) as session:
        session.execute(
            update(BenchmarkCaseResultRecordModel)
            .where(BenchmarkCaseResultRecordModel.id == result.result_id)
            .values(score=99)
        )
        session.commit()

    with pytest.raises(BenchmarkIntegrityError, match="result was modified"):
        repository.list_results(run.run_id)


def test_run_configuration_cannot_change_at_finish(
    persistence: tuple[BenchmarkRepository, Engine],
) -> None:
    repository, _ = persistence
    run = _running_run()
    repository.create_run(run)
    changed_config = run.config.model_copy(update={"model": "different-model"})
    terminal = run.model_copy(
        update={
            "config": changed_config,
            "status": BenchmarkRunStatus.NOT_READY,
            "finished_at": datetime.now(UTC),
            "run_digest": ZERO,
        }
    )
    terminal = terminal.model_copy(update={"run_digest": benchmark_run_digest(terminal)})

    with pytest.raises(BenchmarkIntegrityError, match="configuration changed"):
        repository.finish_run(terminal)


def test_live_provider_coverage_cannot_be_self_reported_without_agent_runs(
    persistence: tuple[BenchmarkRepository, Engine],
) -> None:
    repository, _ = persistence
    run = _running_run()
    repository.create_run(run)
    attempt = _running_attempt(run, 1)
    repository.create_attempt(attempt)
    metric = BenchmarkMetric(
        attempt_id=attempt.attempt_id,
        name="live_provider_critical_agent_coverage",
        kind=BenchmarkMetricKind.QUALITY,
        value=1,
        unit="ratio",
        evidence_ref="self-reported",
        deterministic=True,
        metric_digest=ZERO,
    )
    metric = metric.model_copy(update={"metric_digest": benchmark_metric_digest(metric)})
    repository.add_metric(metric)
    repository.finish_attempt(_terminal(attempt, BenchmarkCaseStatus.FAIL))

    with pytest.raises(BenchmarkIntegrityError, match="conflicts with agent runs"):
        repository.list_metrics(run.run_id)


def test_failed_live_agent_activity_is_distinct_from_full_critical_coverage(
    persistence: tuple[BenchmarkRepository, Engine],
) -> None:
    repository, engine = persistence
    reasoning = ReasoningRepository(sessionmaker(engine, expire_on_commit=False))
    state = reasoning.create_project_problem(
        name="partial live benchmark",
        title="Test",
        raw_problem="Test problem",
    )
    now = datetime.now(UTC)
    reasoning.record_run(
        state.project_id,
        state.problem_id,
        AgentRunResult(
            agent_name="model_explorer",
            status=AgentRunStatus.HUMAN_REVIEW,
            attempts=2,
            input_state_version=state.version,
            provider="deepseek",
            provider_id="deepseek",
            model="deepseek-v4-flash",
            model_id="deepseek-main",
            token_usage=ModelUsage(
                input_tokens=100,
                output_tokens=200,
                total_tokens=300,
                requests=2,
            ),
            latency_ms=1,
            started_at=now,
            ended_at=now,
        ),
    )

    active, evidence = repository.live_provider_activity(state.project_id)
    coverage, _ = repository.live_provider_coverage(state.project_id)

    assert active is True
    assert len(evidence) == 1
    assert coverage == 0


def test_reviewed_model_binding_requires_live_code_agent_instead_of_math_modeler(
    persistence: tuple[BenchmarkRepository, Engine],
) -> None:
    repository, engine = persistence
    reasoning = ReasoningRepository(sessionmaker(engine, expire_on_commit=False))
    state = reasoning.create_project_problem(
        name="reviewed benchmark",
        title="Reviewed model path",
        raw_problem="Use a reviewed mathematical contract.",
    )
    now = datetime.now(UTC)
    reasoning.record_run(
        state.project_id,
        state.problem_id,
        AgentRunResult(
            agent_name="reviewed_model_binder",
            status=AgentRunStatus.SUCCEEDED,
            attempts=0,
            input_state_version=state.version,
            latency_ms=0,
            started_at=now,
            ended_at=now,
        ),
    )
    for name in BenchmarkRepository.REVIEWED_MODEL_LIVE_AGENT_NAMES:
        reasoning.record_run(
            state.project_id,
            state.problem_id,
            AgentRunResult(
                agent_name=name,
                status=AgentRunStatus.SUCCEEDED,
                attempts=1,
                input_state_version=state.version,
                provider="deepseek",
                provider_id="deepseek",
                model="deepseek-chat",
                model_id="deepseek-main",
                token_usage=ModelUsage(
                    input_tokens=10,
                    output_tokens=20,
                    total_tokens=30,
                    requests=1,
                ),
                latency_ms=1,
                started_at=now,
                ended_at=now,
            ),
        )

    active, activity_evidence = repository.live_provider_activity(state.project_id)
    coverage, coverage_evidence = repository.live_provider_coverage(state.project_id)

    assert active is True
    assert coverage == 1
    assert len(activity_evidence) == len(BenchmarkRepository.REVIEWED_MODEL_LIVE_AGENT_NAMES)
    assert len(coverage_evidence) == len(BenchmarkRepository.REVIEWED_MODEL_LIVE_AGENT_NAMES)


def _running_run() -> BenchmarkRun:
    run = BenchmarkRun(
        code_commit="a" * 40,
        source_tree_digest="f" * 64,
        working_tree_dirty=True,
        config=BenchmarkConfig(
            provider="openai",
            model="live-model",
            reasoning_tier="high",
            pricing=BenchmarkPricing(version="test"),
        ),
        case_manifest_digests={"BENCH-case": "b" * 64},
        competition_profile_digests={"profile": "c" * 64},
        status=BenchmarkRunStatus.RUNNING,
        run_digest=ZERO,
    )
    return run.model_copy(update={"run_digest": benchmark_run_digest(run)})


def _running_attempt(run: BenchmarkRun, number: int) -> BenchmarkAttempt:
    attempt = BenchmarkAttempt(
        run_id=run.run_id,
        benchmark_id="BENCH-case",
        manifest_digest="b" * 64,
        attempt_number=number,
        status=BenchmarkCaseStatus.RUNNING,
        solve_input_digest="d" * 64,
        attempt_digest=ZERO,
    )
    return attempt.model_copy(update={"attempt_digest": benchmark_attempt_digest(attempt)})


def _terminal(attempt: BenchmarkAttempt, status: BenchmarkCaseStatus) -> BenchmarkAttempt:
    terminal = attempt.model_copy(
        update={
            "status": status,
            "finished_at": datetime.now(UTC),
            "attempt_digest": ZERO,
        }
    )
    return terminal.model_copy(update={"attempt_digest": benchmark_attempt_digest(terminal)})


def _result(attempt: BenchmarkAttempt) -> BenchmarkCaseResult:
    result = BenchmarkCaseResult(
        attempt_id=attempt.attempt_id,
        benchmark_id=attempt.benchmark_id,
        status=attempt.status,
        dimensions=[
            BenchmarkDimensionScore(
                dimension="test",
                score=0,
                maximum=100,
                evidence_refs=["test"],
            )
        ],
        score=0,
        hard_failures=["test failure"],
        human_intervention_count=0,
        provider_call_count=0,
        total_tokens=0,
        estimated_cost=0,
        wall_time_seconds=0,
        result_digest=ZERO,
        result_id=uuid4(),
    )
    return result.model_copy(update={"result_digest": benchmark_result_digest(result)})
