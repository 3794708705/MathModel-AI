from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from uuid import uuid4

import pytest

from mathmodel_ai.benchmark.executor import PipelineBenchmarkExecutor
from mathmodel_ai.benchmark.manifests import BlindSolveArtifact, BlindSolveBundle
from mathmodel_ai.benchmark.profiles import comap_mcm_2024_profile
from mathmodel_ai.core.errors import QualityGateError
from mathmodel_ai.paper.hashing import sha256_bytes
from mathmodel_ai.schemas.benchmark import (
    BenchmarkCaseManifest,
    BenchmarkCaseStatus,
    BenchmarkConfig,
    BenchmarkPhase,
    BenchmarkPricing,
    BenchmarkResource,
    BenchmarkResourceRole,
    BenchmarkRunRequest,
    FailureCategory,
    GroundTruthPolicy,
    ModelingCategory,
)
from mathmodel_ai.schemas.execution import ExecutionStatus
from mathmodel_ai.schemas.independent_verification import (
    IndependentStatus,
    IndependentVerificationReport,
    IndependentVerificationView,
)
from mathmodel_ai.schemas.problem_state import ProblemState
from mathmodel_ai.schemas.quality import QualityGateStatus
from mathmodel_ai.schemas.solver import SolverRunRef, SolverStatus
from mathmodel_ai.schemas.submission import RequirementCoverageStatus, RuleResultStatus
from tests.verification.helpers import result_bundle


@dataclass
class _UsageRow:
    token_usage: object
    provider_id: str | None = "deepseek"
    provider: str | None = "deepseek"
    is_mock: bool = False


@dataclass
class _ReasoningRows:
    rows: list[_UsageRow]
    state: ProblemState = field(
        default_factory=lambda: ProblemState(project_id=uuid4(), title="test", raw_problem="test")
    )

    def list_agent_runs(self, _project_id: object) -> list[_UsageRow]:
        return self.rows

    def load_current(self, _project_id: object) -> ProblemState:
        return self.state


def test_reviewed_independent_gate_requires_exact_counts_and_unique_executions() -> None:
    result_id = uuid4()
    replay_a = SimpleNamespace(execution=SimpleNamespace(run_id=uuid4()))
    replay_b = SimpleNamespace(execution=SimpleNamespace(run_id=uuid4()))
    report = IndependentVerificationReport.model_construct(
        status=IndependentStatus.PASS,
        result_id=result_id,
        required_metrics=1,
        passed_metrics=1,
        required_scenarios=2,
        passed_scenarios=2,
        replays=[replay_a, replay_b],
        errors=[],
    )
    view = IndependentVerificationView.model_construct(
        status=IndependentStatus.PASS,
        report=report,
        blockers=[],
    )

    accepted = PipelineBenchmarkExecutor._require_independent_pass(
        view,
        result_id=result_id,
        required_metrics=1,
        required_scenarios=2,
    )
    assert accepted is report

    duplicate = report.model_copy(update={"replays": [replay_a, replay_a]})
    duplicate_view = view.model_copy(update={"report": duplicate})
    with pytest.raises(QualityGateError, match="REUSED_SCENARIO_EXECUTION"):
        PipelineBenchmarkExecutor._require_independent_pass(
            duplicate_view,
            result_id=result_id,
            required_metrics=1,
            required_scenarios=2,
        )


def test_pipeline_metrics_use_real_execution_counts_and_do_not_fake_human_rubric() -> None:
    executor = object.__new__(PipelineBenchmarkExecutor)
    attempt_id = uuid4()
    request = BenchmarkRunRequest(
        case_ids=["BENCH-test"],
        config=BenchmarkConfig(
            provider="openai",
            model="live-model",
            reasoning_tier="high",
            pricing=BenchmarkPricing(
                version="test",
                input_per_million=2,
                cached_input_per_million=1,
                output_per_million=4,
            ),
        ),
    )

    metrics = executor._metrics(
        attempt_id=attempt_id,
        request=request,
        model_passed=True,
        solver_passed=True,
        validation_passed=True,
        sensitivity_passed=True,
        robustness_passed=True,
        red_team_passed=True,
        repair_iterations=1,
        paper_ready=True,
        references_verified=True,
        coverage=[RequirementCoverageStatus.COVERED, RequirementCoverageStatus.MISSING],
        rules=[RuleResultStatus.FAIL, RuleResultStatus.PASS],
        submission_passed=True,
        verified_result_present=True,
        package_present=True,
        blocking_rule_violations=0,
        solver_calls=4,
        experiment_runs=18,
        agent_rows=[
            _UsageRow(
                token_usage={
                    "input_tokens": 1_000,
                    "cached_input_tokens": 500,
                    "output_tokens": 500,
                    "requests": 6,
                }
            ),
            _UsageRow(token_usage="invalid"),
        ],
    )
    values = {item.name: item.value for item in metrics}

    assert values["problem_understanding_accuracy"] == 0
    assert values["critical_constraint_recall"] == 0
    assert values["model_appropriateness"] == 0
    assert values["red_team_usefulness"] == 0
    assert values["subproblem_coverage"] == 0.5
    assert values["blocking_competition_violation_count"] == 0
    assert values["solver_calls"] == 4
    assert values["experiment_runs"] == 18
    assert values["provider_calls"] == 6
    assert values["total_tokens"] == 1_500
    assert values["estimated_cost"] == 0.0035


def test_problem_text_and_paper_profile_are_derived_without_model_hints() -> None:
    content = b"Solve every subproblem without hidden evaluation material."
    resource = BenchmarkResource(
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
    manifest = BenchmarkCaseManifest(
        benchmark_id="BENCH-test",
        competition="Official Test Competition",
        year=2024,
        problem_id="A",
        title="Test problem",
        modeling_category=ModelingCategory.MULTI_STAGE,
        difficulty="MULTI_STAGE",
        resources=[resource],
        requires_external_data=False,
        requires_literature=False,
        requires_solver=True,
        ground_truth_policy=GroundTruthPolicy(required_outputs=["answer every task"]),
        license_or_distribution_notes="generated test input",
    )
    bundle = BlindSolveBundle(
        manifest=manifest,
        artifacts=(BlindSolveArtifact(resource=resource, content=content),),
        solve_input_digest="a" * 64,
    )

    assert PipelineBenchmarkExecutor._problem_text(bundle) == content.decode()
    paper_profile = PipelineBenchmarkExecutor._paper_profile(comap_mcm_2024_profile())
    assert paper_profile.competition_name == "COMAP Mathematical Contest in Modeling"
    assert paper_profile.anonymous
    assert {item.value for item in paper_profile.required_sections} == {
        "ABSTRACT",
        "REFERENCES",
    }


def test_partial_live_pipeline_failure_retains_usage_and_project_identity() -> None:
    executor = object.__new__(PipelineBenchmarkExecutor)
    executor._reasoning_repository = _ReasoningRows(  # type: ignore[attr-defined]
        [
            _UsageRow(
                token_usage={
                    "input_tokens": 100,
                    "output_tokens": 200,
                    "cached_input_tokens": 20,
                    "requests": 2,
                }
            )
        ]
    )
    attempt_id = uuid4()
    project_id = uuid4()
    request = BenchmarkRunRequest(
        case_ids=["BENCH-test"],
        config=BenchmarkConfig(
            provider="deepseek",
            model="deepseek-v4-flash",
            reasoning_tier="xhigh",
            pricing=BenchmarkPricing(
                version="test",
                input_per_million=2,
                cached_input_per_million=1,
                output_per_million=4,
            ),
        ),
    )

    outcome = executor._failed_pipeline(
        attempt_id,
        project_id,
        "BENCH-test",
        request,
        stage="MODEL_SELECTION",
        category=FailureCategory.MODEL_SELECTION,
        error=RuntimeError("synthetic"),
    )
    values = {item.name: item.value for item in outcome.metrics}

    assert outcome.status is BenchmarkCaseStatus.FAIL
    assert outcome.project_id == project_id
    assert outcome.provider_is_live is True
    assert values["provider_calls"] == 2
    assert values["total_tokens"] == 300
    assert values["estimated_cost"] == 0.00098
    assert outcome.failures[0].stage == "MODEL_SELECTION"


def test_paper_failure_metrics_preserve_completed_science_and_unknown_cost() -> None:
    executor = object.__new__(PipelineBenchmarkExecutor)
    state = SimpleNamespace(
        project_id=uuid4(), problem_id=uuid4(), execution_records=[], solver_runs=[],
        quality_gates=[
            SimpleNamespace(gate=name, status=QualityGateStatus.PASS, errors=[])
            for name in ("SOLVE", "VALIDATE", "SENSITIVITY", "ROBUSTNESS", "VERIFIED")
        ],
    )
    request = BenchmarkRunRequest(
        case_ids=["BENCH-test"],
        config=BenchmarkConfig(
            provider="local", model="test", reasoning_tier="low",
            pricing=BenchmarkPricing(
                version="unknown", input_per_million=0,
                cached_input_per_million=0, output_per_million=0,
            ),
        ),
    )
    metrics = executor._partial_metrics(
        uuid4(), request, [_UsageRow(token_usage={
            "input_tokens": 100, "output_tokens": 10, "requests": 1,
        })], state=state,
    )
    by_name = {item.name: item for item in metrics}
    for name in (
        "mathematical_validity", "solver_success", "validation_pass",
        "sensitivity_completion", "robustness_completion", "central_model_valid",
    ):
        assert by_name[name].value == 1
        assert by_name[name].evidence_ref.startswith("PASS:")
    assert by_name["unverified_central_result_count"].value == 0
    assert by_name["wrong_submission_artifact_count"].value == 0
    assert by_name["wrong_submission_artifact_count"].evidence_ref.startswith(
        "UPSTREAM_BLOCKED:"
    )
    assert by_name["estimated_cost"].evidence_ref.startswith("NOT_EVALUATED:")
    assert by_name["problem_understanding_accuracy"].evidence_ref.startswith(
        "NOT_EVALUATED:"
    )


def test_partial_failure_preserves_every_real_formal_solve_attempt() -> None:
    model, result, solver_run, execution, _, _ = result_bundle()
    ref = SolverRunRef(
        solver_run_id=solver_run.solver_run_id,
        model_id=model.model_id,
        model_version=model.version,
        model_digest=solver_run.model_digest,
        solver=solver_run.solver,
        status=SolverStatus.EXECUTION_ERROR,
        execution_ref=execution.run_id,
        result_ref=result.result_id,
    )
    second_execution = execution.model_copy(update={"run_id": uuid4()})
    second = ref.model_copy(
        update={
            "solver_run_id": uuid4(),
            "execution_ref": second_execution.run_id,
            "status": SolverStatus.FEASIBLE,
            "result_ref": uuid4(),
        }
    )
    mock_execution = execution.model_copy(
        update={"run_id": uuid4(), "is_mock": True, "status": ExecutionStatus.REJECTED}
    )
    mock_ref = ref.model_copy(
        update={"solver_run_id": uuid4(), "execution_ref": mock_execution.run_id}
    )
    orphan = ref.model_copy(update={"solver_run_id": uuid4(), "execution_ref": uuid4()})
    repeated_execution = second.model_copy(update={"solver_run_id": uuid4()})
    rejected_execution = execution.model_copy(
        update={"run_id": uuid4(), "status": ExecutionStatus.REJECTED}
    )
    rejected_ref = ref.model_copy(
        update={"solver_run_id": uuid4(), "execution_ref": rejected_execution.run_id}
    )
    state = ProblemState(
        project_id=model.project_id,
        problem_id=model.problem_id,
        title="synthetic execution-count audit",
        raw_problem="test",
        solver_runs=[ref, second, repeated_execution, mock_ref, orphan, rejected_ref],
        execution_records=[execution, second_execution, mock_execution, rejected_execution],
    )
    executor = object.__new__(PipelineBenchmarkExecutor)
    executor._reasoning_repository = _ReasoningRows([], state)  # type: ignore[attr-defined]
    request = BenchmarkRunRequest(
        case_ids=["BENCH-test"],
        config=BenchmarkConfig(
            provider="deepseek",
            model="test",
            reasoning_tier="high",
            pricing=BenchmarkPricing(
                version="synthetic-test", input_per_million=0, output_per_million=0
            ),
        ),
    )

    for outcome in (
        executor._failed_pipeline(
            uuid4(),
            model.project_id,
            "BENCH-test",
            request,
            stage="VERIFICATION",
            category=FailureCategory.VALIDATION,
            error=RuntimeError("synthetic verification failure"),
        ),
        executor._failed_before_paper(
            uuid4(), model.project_id, "BENCH-test", request, "synthetic missing verified result"
        ),
    ):
        values = {item.name: item.value for item in outcome.metrics}
        assert values["solver_calls"] == 2
        assert values["validation_pass"] == 0
        assert outcome.status is BenchmarkCaseStatus.FAIL
    assert PipelineBenchmarkExecutor._formal_solver_call_count(state) == 2
