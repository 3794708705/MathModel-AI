import json

import pytest

from mathmodel_ai.benchmark.reporting import BenchmarkReportBuilder
from mathmodel_ai.schemas.benchmark import BenchmarkCaseStatus
from mathmodel_ai.schemas.independent_verification import (
    IndependentVerificationReport,
    IndependentVerificationView,
    VerificationRequirements,
)
from mathmodel_ai.verification.metric_recompute import content_digest
from mathmodel_ai.verification.requirements import VerificationRequirementRegistry
from mathmodel_ai.verification.scenario_replay import parse_raw_output
from tests.benchmark.test_evaluation_reporting import _attempt, _claimed_result
from tests.verification.test_independent_repository import repository as repository_fixture

repository = repository_fixture


def test_reviewed_policy_registry_missing_duplicate_and_unsafe(tmp_path, repository):
    _, plan, _ = repository
    registry = VerificationRequirementRegistry(tmp_path)
    assert registry.get("BENCH-case") is None
    directory = tmp_path / "case-a"
    directory.mkdir()
    digest = "b" * 64
    metrics = [
        item.model_copy(
            update={
                "quantity": "synthetic fixture metric",
                "calculation": "closed registry recomputation",
                "inputs": ["fixture"],
                "unit": "dimensionless",
                "tolerance_provenance": "synthetic fixture tolerance",
                "model_binding": digest,
            }
        )
        for item in plan.metrics
    ]
    scenarios = [
        item.model_copy(
            update={
                "metrics": metrics,
                "baseline": "synthetic baseline",
                "perturbation": "no changed inputs",
                "reason": "registry fixture",
                "input_changes": [],
                "comparison_quantity": "synthetic output",
                "acceptance_criterion": "required metrics pass",
                "criterion_provenance": "fixture contract",
            }
        )
        for item in plan.scenarios
    ]
    policy = VerificationRequirements(
        version="2",
        review_status="REVIEWED",
        production_eligible=True,
        benchmark_id="BENCH-case",
        manifest_digest="a" * 64,
        model_digest=digest,
        metrics=metrics,
        scenarios=scenarios,
        scientific_scope=plan.scientific_scope,
    )
    path = directory / "independent-verification.json"
    path.write_text(policy.model_dump_json(), encoding="utf-8")
    assert registry.get("BENCH-case") == policy
    duplicate = tmp_path / "case-b"
    duplicate.mkdir()
    other = duplicate / path.name
    other.write_text(policy.model_dump_json(), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        registry.get("BENCH-case")
    other.write_bytes(b"x" * (1024 * 1024 + 1))
    with pytest.raises(ValueError, match="invalid"):
        registry.get("BENCH-case")


def test_independent_gate_is_and_not_override_and_exact_result(repository):
    _, plan, _ = repository
    attempt = _attempt("BENCH-case")
    result = _claimed_result(attempt)
    report = IndependentVerificationReport(
        attempt_id=attempt.attempt_id,
        plan_id=plan.plan_id,
        plan_digest=content_digest(plan),
        result_id=result.verified_result_id,
        status="PASS",
        metrics=[],
        replays=[],
        required_metrics=1,
        passed_metrics=1,
        required_scenarios=1,
        passed_scenarios=1,
        errors=[],
    )
    # This injected callback tests gate composition only, never evidence truth.
    view = IndependentVerificationView(attempt_id=attempt.attempt_id, status="PASS", report=report)
    builder = BenchmarkReportBuilder(independent_verifier=lambda _: view)
    assert builder._independent_gate(result).status == "FAIL"  # counters without evidence
    failed = result.model_copy(
        update={"status": BenchmarkCaseStatus.FAIL, "hard_failures": ["critical finding"]}
    )
    assert "critical finding" in builder._independent_gate(failed).hard_failures
    assert builder._independent_gate(failed).status == "FAIL"
    switched = result.model_copy(update={"verified_result_id": plan.result_id})
    assert builder._independent_gate(switched).status == "FAIL"
    assert BenchmarkReportBuilder()._independent_gate(result).status == "FAIL"


@pytest.mark.parametrize("payload", [[], 1, None, "invalid"])
def test_raw_metric_artifact_requires_an_object(payload):
    with pytest.raises(ValueError, match="JSON object"):
        parse_raw_output(json.dumps(payload).encode())
