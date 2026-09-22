from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import httpx
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from mathmodel_ai.agents.base import AgentRunResult, AgentRunStatus
from mathmodel_ai.benchmark.executor import PipelineBenchmarkExecutor
from mathmodel_ai.benchmark.manifests import BenchmarkManifestRegistry
from mathmodel_ai.benchmark.profiles import comap_mcm_2024_profile
from mathmodel_ai.benchmark.repository import BenchmarkRepository
from mathmodel_ai.benchmark.sources import BenchmarkResourceCache, BenchmarkStructureInspector
from mathmodel_ai.benchmark.workflow import (
    BenchmarkWorkflow,
    CaseExecutionOutcome,
    resolve_git_identity,
)
from mathmodel_ai.core.types import Environment, ProviderName
from mathmodel_ai.db.base import Base
from mathmodel_ai.paper.hashing import sha256_bytes
from mathmodel_ai.paper.literature import FixtureLiteratureSource
from mathmodel_ai.providers.factory import ProviderRegistry
from mathmodel_ai.providers.repository import ProviderModelRegistry
from mathmodel_ai.providers.schemas import ModelUsage
from mathmodel_ai.providers.secrets import EnvironmentSecretResolver
from mathmodel_ai.providers.security import EndpointSecurityPolicy
from mathmodel_ai.reasoning.repository import ReasoningRepository
from mathmodel_ai.schemas.benchmark import (
    BenchmarkBudget,
    BenchmarkCaseManifest,
    BenchmarkCaseResult,
    BenchmarkCaseStatus,
    BenchmarkConfig,
    BenchmarkDeadlineMode,
    BenchmarkDimensionScore,
    BenchmarkPhase,
    BenchmarkPricing,
    BenchmarkResource,
    BenchmarkResourceRole,
    BenchmarkRunRequest,
    BenchmarkRunStatus,
    FailureCategory,
    FailureSeverity,
    GroundTruthPolicy,
    ModelingCategory,
    benchmark_result_digest,
)
from mathmodel_ai.schemas.provider_registry import (
    CapabilityEvidence,
    CapabilityProbeResult,
    CapabilitySource,
    CapabilityStatus,
    EndpointTrustLevel,
    ModelCapability,
    ModelProfile,
    ProbeAuthenticationStatus,
    ProviderEndpoint,
    ProviderProtocol,
    QualityTier,
    StructuredOutputStrategy,
)


class _ProviderMarker:
    name = ProviderName.OPENAI


class _FailingCaseExecutor:
    async def execute(self, **_kwargs: object) -> object:
        raise RuntimeError("synthetic transient provider failure")


class _PartialLiveFailureExecutor:
    def __init__(self) -> None:
        self.session_factory: object | None = None

    async def execute(
        self,
        *,
        attempt_id: UUID,
        request: BenchmarkRunRequest,
        **_kwargs: object,
    ) -> CaseExecutionOutcome:
        assert self.session_factory is not None
        reasoning = ReasoningRepository(self.session_factory)  # type: ignore[arg-type]
        state = reasoning.create_project_problem(
            name="partial live case",
            title="Partial live case",
            raw_problem="Official benchmark problem",
        )
        now = datetime.now(UTC)
        run = AgentRunResult(
            agent_name="problem_agent",
            status=AgentRunStatus.SUCCEEDED,
            attempts=1,
            input_state_version=state.version,
            provider="deepseek",
            provider_id="deepseek",
            model="deepseek-v4-flash",
            model_id="deepseek-main",
            token_usage=ModelUsage(
                input_tokens=100,
                output_tokens=200,
                total_tokens=300,
                requests=1,
            ),
            latency_ms=1,
            started_at=now,
            ended_at=now,
        )
        reasoning.record_run(state.project_id, state.problem_id, run)
        pipeline = object.__new__(PipelineBenchmarkExecutor)
        metrics = pipeline._partial_metrics(
            attempt_id,
            request,
            reasoning.list_agent_runs(state.project_id),
            state=state,
        )
        result = BenchmarkCaseResult(
            attempt_id=attempt_id,
            benchmark_id="BENCH-test",
            status=BenchmarkCaseStatus.FAIL,
            dimensions=[
                BenchmarkDimensionScore(
                    dimension="partial",
                    score=0,
                    maximum=100,
                    evidence_refs=[f"project:{state.project_id}"],
                )
            ],
            score=0,
            human_intervention_count=0,
            provider_call_count=1,
            total_tokens=300,
            estimated_cost=0,
            wall_time_seconds=0,
            result_digest="0" * 64,
        )
        result = result.model_copy(update={"result_digest": benchmark_result_digest(result)})
        return CaseExecutionOutcome(
            status=BenchmarkCaseStatus.FAIL,
            project_id=state.project_id,
            provider_is_live=True,
            metrics=tuple(metrics),
            failures=(),
            interventions=(),
            result=result,
        )


@pytest.mark.asyncio
async def test_pipeline_exception_is_terminal_and_rerun_preserves_first_history(
    tmp_path: Path,
) -> None:
    workflow, repository, cache = _workflow(
        tmp_path,
        providers=ProviderRegistry([_ProviderMarker()]),  # type: ignore[list-item]
        case_executor=_FailingCaseExecutor(),
    )
    request = _request()
    try:
        first = await workflow.run(request)
        second = await workflow.run(request)
    finally:
        cache.close()

    assert first.report.run.run_id != second.report.run.run_id
    assert first.report.run.status is BenchmarkRunStatus.NOT_READY
    assert first.report.run.config.deadline_modes == list(BenchmarkDeadlineMode)
    assert [item.status for item in first.report.attempts] == [BenchmarkCaseStatus.FAIL]
    assert all(item.finished_at is not None for item in first.report.attempts)
    assert all(not item.provider_is_live for item in first.report.attempts)
    assert any(item.severity is FailureSeverity.P1 for item in first.report.failures)
    assert any(item.stage == "PHASE_1_7_PIPELINE" for item in first.report.failures)
    assert repository.get_run(first.report.run.run_id).status is BenchmarkRunStatus.NOT_READY
    assert len(repository.list_attempts(first.report.run.run_id)) == 1
    assert len(repository.list_attempts(second.report.run.run_id)) == 1


@pytest.mark.asyncio
async def test_partial_live_failure_is_project_bound_and_provider_failed_not_blocked(
    tmp_path: Path,
) -> None:
    executor = _PartialLiveFailureExecutor()
    workflow, repository, cache = _workflow(
        tmp_path,
        providers=ProviderRegistry([_ProviderMarker()]),  # type: ignore[list-item]
        case_executor=executor,
    )
    executor.session_factory = repository._session_factory  # type: ignore[attr-defined]
    try:
        outcome = await workflow.run(_request())
    finally:
        cache.close()

    attempt = outcome.report.attempts[0]
    values = {item.name: item.value for item in outcome.report.metrics}

    assert attempt.status is BenchmarkCaseStatus.FAIL
    assert attempt.project_id is not None
    assert attempt.provider_is_live is True
    assert outcome.report.run.live_provider_status.value == "FAIL"
    assert values["provider_calls"] == 1
    assert values["total_tokens"] == 300
    assert values["live_provider_critical_agent_coverage"] == pytest.approx(1 / 6)


@pytest.mark.asyncio
async def test_wall_budget_exceeded_cannot_be_reported_as_provider_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = iter((100.0, 101.0))
    monkeypatch.setattr("mathmodel_ai.benchmark.workflow.monotonic", lambda: next(clock))
    workflow, _, cache = _workflow(
        tmp_path,
        providers=ProviderRegistry([]),
        case_executor=None,
    )
    request = _request(
        budget=BenchmarkBudget(max_wall_time_seconds=1e-12),
    )
    try:
        outcome = await workflow.run(request)
    finally:
        cache.close()

    assert [item.status for item in outcome.report.attempts] == [
        BenchmarkCaseStatus.BUDGET_EXCEEDED
    ]
    assert [item.status for item in outcome.report.results] == [BenchmarkCaseStatus.BUDGET_EXCEEDED]
    assert not outcome.report.attempts[0].provider_is_live
    assert any(item.stage == "BENCHMARK_BUDGET" for item in outcome.report.failures)


def test_budget_policy_covers_case_total_and_wall_limits() -> None:
    request = _request(
        budget=BenchmarkBudget(
            max_case_cost=1,
            max_total_cost=2,
            max_wall_time_seconds=3,
        )
    )

    reasons = BenchmarkWorkflow._budget_reasons(
        request,
        case_cost=1.1,
        prior_total_cost=1,
        elapsed=3.1,
    )

    assert len(reasons) == 3
    assert "max_case_cost" in reasons[0]
    assert "max_total_cost" in reasons[1]
    assert "max_wall_time_seconds" in reasons[2]


@pytest.mark.asyncio
async def test_registered_custom_model_is_bound_into_benchmark_history(
    tmp_path: Path,
) -> None:
    registry_engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(registry_engine)
    registry_factory = sessionmaker(registry_engine, expire_on_commit=False)
    registry = ProviderModelRegistry(
        registry_factory,
        secrets=EnvironmentSecretResolver({"BENCH_CUSTOM_KEY": "configured"}),
        security_policy=EndpointSecurityPolicy(
            environment=Environment.TEST,
            resolver=lambda _host, _port: ["93.184.216.34"],
        ),
    )
    endpoint = registry.create_provider(
        ProviderEndpoint(
            provider_id="benchmark-custom",
            display_name="Benchmark Custom",
            protocol=ProviderProtocol.OPENAI_CHAT_COMPLETIONS,
            base_url="https://benchmark-custom.example.test/v1",
            credential_ref="env:BENCH_CUSTOM_KEY",
            trust_level=EndpointTrustLevel.USER_MANAGED_PROXY,
        )
    )
    model = registry.create_model(
        ModelProfile(
            model_id="benchmark-model",
            provider_id=endpoint.provider_id,
            display_name="Benchmark Model",
            remote_model="remote-benchmark-model",
            quality_tier=QualityTier.FLAGSHIP_MAX,
            declared_capabilities={
                capability: CapabilityEvidence(
                    status=CapabilityStatus.SUPPORTED,
                    source=CapabilitySource.USER_DECLARED,
                )
                for capability in (
                    ModelCapability.TEXT,
                    ModelCapability.STRUCTURED_OUTPUT,
                )
            },
            structured_output_strategy=StructuredOutputStrategy.PROMPT_JSON_FALLBACK,
        )
    )
    registry.record_probe(
        CapabilityProbeResult(
            provider_id=endpoint.provider_id,
            model_id=model.model_id,
            provider_config_digest=endpoint.config_digest,
            model_config_digest=model.config_digest,
            capabilities={
                ModelCapability.TEXT: CapabilityEvidence(
                    status=CapabilityStatus.SUPPORTED,
                    source=CapabilitySource.PROBED,
                ),
                ModelCapability.STRUCTURED_OUTPUT: CapabilityEvidence(
                    status=CapabilityStatus.PARTIAL,
                    source=CapabilitySource.PROBED,
                ),
            },
            authentication_status=ProbeAuthenticationStatus.PASS,
            latency_ms=1,
        )
    )
    workflow, _, cache = _workflow(
        tmp_path,
        providers=ProviderRegistry([]),
        case_executor=None,
        provider_configurations=registry,
    )
    request = _request().model_copy(
        update={
            "config": _request().config.model_copy(
                update={
                    "provider": "placeholder",
                    "model": "placeholder",
                    "model_id": model.model_id,
                }
            )
        }
    )
    try:
        outcome = await workflow.run(request)
    finally:
        cache.close()
    config = outcome.report.run.config
    assert config.provider_id == endpoint.provider_id
    assert config.provider_config_digest == endpoint.config_digest
    assert config.model_id == model.model_id
    assert config.model_config_digest == model.config_digest
    assert config.model == model.remote_model
    assert config.endpoint_trust == EndpointTrustLevel.USER_MANAGED_PROXY.value
    assert config.model_identity_confidence == "NOT_INDEPENDENTLY_VERIFIED"
    registry.record_probe(
        CapabilityProbeResult(
            provider_id=endpoint.provider_id,
            model_id=model.model_id,
            provider_config_digest=endpoint.config_digest,
            model_config_digest=model.config_digest,
            capabilities={
                ModelCapability.TEXT: CapabilityEvidence(
                    status=CapabilityStatus.SUPPORTED,
                    source=CapabilitySource.PROBED,
                ),
                ModelCapability.STRUCTURED_OUTPUT: CapabilityEvidence(
                    status=CapabilityStatus.PARTIAL,
                    source=CapabilitySource.PROBED,
                ),
            },
            authentication_status=ProbeAuthenticationStatus.AUTH_FAILED,
            latency_ms=1,
        )
    )
    assert workflow._provider_available(config) is False
    assert any(
        failure.category is FailureCategory.INFRASTRUCTURE and "case executor" in failure.root_cause
        for failure in outcome.report.failures
    )


def test_git_identity_binds_tracked_diff_and_untracked_files(tmp_path: Path) -> None:
    import subprocess

    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init"], cwd=repository, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.name", "Phase 8 Test"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "phase8@example.invalid"],
        cwd=repository,
        check=True,
    )
    tracked = repository / "tracked.txt"
    tracked.write_text("baseline\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repository, check=True)
    subprocess.run(
        ["git", "commit", "-m", "baseline"],
        cwd=repository,
        check=True,
        capture_output=True,
    )

    clean = resolve_git_identity(repository)
    tracked.write_text("changed\n", encoding="utf-8")
    dirty_tracked = resolve_git_identity(repository)
    tracked.write_text("baseline\n", encoding="utf-8")
    (repository / "untracked.txt").write_text("untracked\n", encoding="utf-8")
    dirty_untracked = resolve_git_identity(repository)

    assert not clean.working_tree_dirty
    assert dirty_tracked.working_tree_dirty
    assert dirty_untracked.working_tree_dirty
    assert (
        len(
            {
                clean.source_tree_digest,
                dirty_tracked.source_tree_digest,
                dirty_untracked.source_tree_digest,
            }
        )
        == 3
    )


def _workflow(
    tmp_path: Path,
    *,
    providers: ProviderRegistry,
    case_executor: object | None,
    provider_configurations: ProviderModelRegistry | None = None,
) -> tuple[BenchmarkWorkflow, BenchmarkRepository, BenchmarkResourceCache]:
    content = b"official benchmark problem"
    manifest = _manifest(content)
    manifest_root = tmp_path / "manifests"
    case_root = manifest_root / "case-001"
    case_root.mkdir(parents=True, exist_ok=True)
    (case_root / "manifest.json").write_text(
        json.dumps(manifest.model_dump(mode="json")), encoding="utf-8"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=content, request=request)

    cache = BenchmarkResourceCache(
        tmp_path / "cache",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    repository = BenchmarkRepository(sessionmaker(engine, expire_on_commit=False))
    workflow = BenchmarkWorkflow(
        repository=repository,
        manifests=BenchmarkManifestRegistry(manifest_root),
        resource_cache=cache,
        inspector=BenchmarkStructureInspector(),
        providers=providers,
        literature_source=FixtureLiteratureSource(records=()),
        profile=comap_mcm_2024_profile(),
        code_commit="a" * 40,
        source_tree_digest="b" * 64,
        working_tree_dirty=True,
        output_root=tmp_path / "runs",
        case_executor=case_executor,  # type: ignore[arg-type]
        provider_configurations=provider_configurations,
    )
    return workflow, repository, cache


def _request(*, budget: BenchmarkBudget | None = None) -> BenchmarkRunRequest:
    return BenchmarkRunRequest(
        case_ids=["BENCH-test"],
        config=BenchmarkConfig(
            provider="openai",
            model="live-model",
            reasoning_tier="high",
            deadline_modes=list(BenchmarkDeadlineMode),
            pricing=BenchmarkPricing(version="test"),
            budget=budget or BenchmarkBudget(),
        ),
    )


def _manifest(content: bytes) -> BenchmarkCaseManifest:
    return BenchmarkCaseManifest(
        benchmark_id="BENCH-test",
        competition="Official Test Competition",
        year=2024,
        problem_id="A",
        title="Test problem",
        modeling_category=ModelingCategory.MULTI_STAGE,
        difficulty="MULTI_STAGE",
        resources=[
            BenchmarkResource(
                resource_id="RESOURCE-problem",
                phase=BenchmarkPhase.SOLVE,
                role=BenchmarkResourceRole.PROBLEM,
                source_url="https://www.contest.comap.com/problem.txt",
                sha256=sha256_bytes(content),
                media_type="text/plain",
                local_filename="problem.txt",
                expected_size_bytes=len(content),
                distribution_notes="generated test input",
            )
        ],
        requires_external_data=False,
        requires_literature=False,
        requires_solver=True,
        ground_truth_policy=GroundTruthPolicy(required_outputs=["answer every task"]),
        license_or_distribution_notes="generated test input",
    )
