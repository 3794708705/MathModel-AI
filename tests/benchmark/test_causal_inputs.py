import csv
import hashlib
import io
import json
from pathlib import Path

import pytest

from mathmodel_ai.benchmark.causal_inputs import split_causal_csv
from mathmodel_ai.benchmark.executor import PipelineBenchmarkExecutor
from mathmodel_ai.benchmark.manifests import BenchmarkManifestRegistry
from mathmodel_ai.schemas.benchmark import (
    BenchmarkCaseManifest,
    BenchmarkPhase,
    BenchmarkResource,
    BenchmarkResourceRole,
    CausalScienceCheck,
    GroundTruthPolicy,
    ModelingCategory,
)
from mathmodel_ai.verification.causal_binary import CausalBinarySpec


def _source() -> bytes:
    return (
        b"match,server,winner,after\n"
        b"A,1,1,100\nA,2,2,100\n"
        b"B,1,2,100\nB,2,1,100\n"
        b"C,1,1,100\nC,2,1,100\n"
        b"D,1,2,100\nD,2,2,100\n"
    )


def _spec(source: bytes) -> CausalBinarySpec:
    return CausalBinarySpec(
        source_sha256=hashlib.sha256(source).hexdigest(),
        group_column="match",
        condition_column="server",
        outcome_column="winner",
        positive_value="1",
        negative_value="2",
    )


def _fixture(tmp_path: Path) -> tuple[BenchmarkManifestRegistry, Path]:
    source = _source()
    problem = b"problem"
    root = tmp_path / "manifests" / "case-001"
    root.mkdir(parents=True)
    manifest = BenchmarkCaseManifest(
        benchmark_id="BENCH-CAUSAL-TEST",
        competition="test",
        year=2024,
        problem_id="TEST",
        title="causal test",
        modeling_category=ModelingCategory.DATA_PREDICTION,
        difficulty="MODERATE",
        resources=[
            BenchmarkResource(
                resource_id="RESOURCE-problem",
                phase=BenchmarkPhase.SOLVE,
                role=BenchmarkResourceRole.PROBLEM,
                source_url="https://www.contest.comap.com/problem.pdf",
                sha256=hashlib.sha256(problem).hexdigest(),
                media_type="application/pdf",
                local_filename="problem.pdf",
                expected_size_bytes=len(problem),
                distribution_notes="test",
            ),
            BenchmarkResource(
                resource_id="RESOURCE-data",
                phase=BenchmarkPhase.SOLVE,
                role=BenchmarkResourceRole.ATTACHMENT,
                source_url="https://www.contest.comap.com/data.csv",
                sha256=hashlib.sha256(source).hexdigest(),
                media_type="text/csv",
                local_filename="data.csv",
                expected_size_bytes=len(source),
                distribution_notes="test",
            ),
        ],
        requires_external_data=False,
        requires_literature=False,
        requires_solver=True,
        ground_truth_policy=GroundTruthPolicy(required_outputs=["prediction"]),
        license_or_distribution_notes="test",
    )
    (root / "manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")
    policy = {
        "version": "1",
        "resource_id": "RESOURCE-data",
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "group_column": "match",
        "condition_column": "server",
        "outcome_column": "winner",
        "positive_value": "1",
        "negative_value": "2",
        "fraction": 0.25,
        "salt": "v1",
    }
    (root / "causal-holdout-v1.json").write_text(json.dumps(policy), encoding="utf-8")
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "problem.pdf").write_bytes(problem)
    (cache / "data.csv").write_bytes(source)
    return BenchmarkManifestRegistry(root.parent), cache


def test_group_split_does_not_use_labels_or_post_outcome_columns() -> None:
    source = _source()
    original = split_causal_csv(source, _spec(source), fraction=0.25, salt="v1")
    altered = source.replace(b"A,1,1,100", b"A,1,2,999")
    changed = split_causal_csv(altered, _spec(altered), fraction=0.25, salt="v1")
    assert original.heldout_groups == changed.heldout_groups
    assert original.training_groups == changed.training_groups
    assert original.training_rows + original.heldout_rows == 8
    rows = list(csv.DictReader(io.StringIO(original.training_csv.decode())))
    assert {row["match"] for row in rows} == set(original.training_groups)
    assert not ({row["match"] for row in rows} & set(original.heldout_groups))
    assert hashlib.sha256(original.training_csv).hexdigest() == original.training_sha256


def test_registry_keeps_official_bytes_but_exposes_only_training_csv(tmp_path: Path) -> None:
    registry, cache = _fixture(tmp_path)
    bundle = registry.load_blind_solve_bundle("BENCH-CAUSAL-TEST", cache)
    assert bundle.causal_split is not None
    assert bundle.causal_policy is not None
    full = next(item for item in bundle.artifacts if item.resource.resource_id == "RESOURCE-data")
    visible = next(
        item for item in bundle.solver_artifacts if item.resource.resource_id == "RESOURCE-data"
    )
    assert full.content == _source()
    assert visible.content == bundle.causal_split.training_csv
    assert visible.content != full.content
    assert b"problem" in next(
        item.content
        for item in bundle.solver_artifacts
        if item.resource.resource_id == "RESOURCE-problem"
    )
    assert bundle.causal_policy.source_sha256 == bundle.causal_split.source_sha256
    guidance = PipelineBenchmarkExecutor._causal_user_guidance(bundle)
    assert len(guidance) == 1
    assert "causal_predictor.py" in guidance[0]
    assert "only training groups" in guidance[0]


def test_registry_rejects_stale_source_binding(tmp_path: Path) -> None:
    registry, cache = _fixture(tmp_path)
    path = tmp_path / "manifests" / "case-001" / "causal-holdout-v1.json"
    policy = json.loads(path.read_text(encoding="utf-8"))
    policy["source_sha256"] = "0" * 64
    path.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(ValueError, match="CAUSAL_HOLDOUT_RESOURCE_IDENTITY_MISMATCH"):
        registry.load_blind_solve_bundle("BENCH-CAUSAL-TEST", cache)


def test_solve_input_digest_binds_holdout_policy(tmp_path: Path) -> None:
    registry, cache = _fixture(tmp_path)
    before = registry.load_blind_solve_bundle("BENCH-CAUSAL-TEST", cache)
    path = tmp_path / "manifests" / "case-001" / "causal-holdout-v1.json"
    policy = json.loads(path.read_text(encoding="utf-8"))
    policy["salt"] = "v2"
    path.write_text(json.dumps(policy), encoding="utf-8")
    after = registry.load_blind_solve_bundle("BENCH-CAUSAL-TEST", cache)
    assert before.solve_input_digest != after.solve_input_digest
    assert before.causal_split is not None and after.causal_split is not None
    assert before.causal_split.policy_sha256 != after.causal_split.policy_sha256


def test_solve_input_digest_binds_scientific_obligations(tmp_path: Path) -> None:
    registry, cache = _fixture(tmp_path)
    before = registry.load_blind_solve_bundle("BENCH-CAUSAL-TEST", cache)
    path = tmp_path / "manifests" / "case-001" / "causal-holdout-v1.json"
    policy = json.loads(path.read_text(encoding="utf-8"))
    policy["required_scientific_checks"] = [
        CausalScienceCheck.HELDOUT_PREDICTION,
        CausalScienceCheck.MATCH_FLOW,
    ]
    policy["problem_sha256"] = hashlib.sha256(b"problem").hexdigest()
    path.write_text(json.dumps(policy), encoding="utf-8")
    after = registry.load_blind_solve_bundle("BENCH-CAUSAL-TEST", cache)
    assert before.solve_input_digest != after.solve_input_digest
    assert before.causal_split is not None and after.causal_split is not None
    assert before.causal_split.policy_sha256 == after.causal_split.policy_sha256
    assert after.causal_policy is not None
    assert after.causal_policy.required_scientific_checks == [
        CausalScienceCheck.HELDOUT_PREDICTION,
        CausalScienceCheck.MATCH_FLOW,
    ]


@pytest.mark.parametrize(
    "checks",
    [["match_flow"], ["heldout_prediction", "heldout_prediction"]],
)
def test_causal_science_policy_rejects_incomplete_or_duplicate_checks(
    tmp_path: Path, checks: list[str]
) -> None:
    registry, cache = _fixture(tmp_path)
    path = tmp_path / "manifests" / "case-001" / "causal-holdout-v1.json"
    policy = json.loads(path.read_text(encoding="utf-8"))
    policy["required_scientific_checks"] = checks
    policy["problem_sha256"] = hashlib.sha256(b"problem").hexdigest()
    path.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(ValueError, match="causal scientific checks"):
        registry.load_blind_solve_bundle("BENCH-CAUSAL-TEST", cache)


def test_scientific_obligations_reject_a_different_problem_source(tmp_path: Path) -> None:
    registry, cache = _fixture(tmp_path)
    path = tmp_path / "manifests" / "case-001" / "causal-holdout-v1.json"
    policy = json.loads(path.read_text(encoding="utf-8"))
    policy["required_scientific_checks"] = ["heldout_prediction"]
    policy["problem_sha256"] = "0" * 64
    path.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(ValueError, match="CAUSAL_SCIENCE_PROBLEM_SOURCE_MISMATCH"):
        registry.load_blind_solve_bundle("BENCH-CAUSAL-TEST", cache)


def test_official_c_cache_is_split_without_new_benchmark_run() -> None:
    root = Path(__file__).resolve().parents[2]
    cache = root / "var/benchmarks/cache/BENCH-MCM2024-C"
    if not (cache / "Wimbledon_featured_matches.csv").is_file():
        pytest.skip("official runtime cache is unavailable")
    bundle = BenchmarkManifestRegistry(root / "benchmarks").load_blind_solve_bundle(
        "BENCH-MCM2024-C", cache
    )
    assert bundle.causal_split is not None
    assert bundle.causal_policy is not None
    assert bundle.causal_policy.required_scientific_checks == list(CausalScienceCheck)
    assert (
        "causal_science:randomness_test"
        in PipelineBenchmarkExecutor._causal_user_guidance(bundle)[0]
    )
    assert "groupby, set sort=False" in PipelineBenchmarkExecutor._causal_user_guidance(bundle)[0]
    assert len(bundle.causal_split.heldout_groups) == 6
    assert len(bundle.causal_split.training_groups) == 25
    assert bundle.causal_split.training_rows + bundle.causal_split.heldout_rows == 7284
    visible = next(
        item
        for item in bundle.solver_artifacts
        if item.resource.resource_id == "RESOURCE-C-WIMBLEDON"
    )
    visible_groups = {
        row["match_id"] for row in csv.DictReader(io.StringIO(visible.content.decode()))
    }
    assert visible_groups == set(bundle.causal_split.training_groups)
    assert visible_groups.isdisjoint(bundle.causal_split.heldout_groups)
