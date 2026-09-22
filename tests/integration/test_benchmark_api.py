from __future__ import annotations

import json
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from mathmodel_ai.benchmark.manifests import BenchmarkManifestRegistry
from mathmodel_ai.benchmark.profiles import comap_mcm_2024_profile
from mathmodel_ai.benchmark.sources import BenchmarkResourceCache, BenchmarkStructureInspector
from mathmodel_ai.benchmark.workflow import BenchmarkWorkflow
from mathmodel_ai.core.config import Settings
from mathmodel_ai.db.base import Base
from mathmodel_ai.main import create_app
from mathmodel_ai.paper.hashing import sha256_bytes
from mathmodel_ai.paper.literature import FixtureLiteratureSource
from mathmodel_ai.schemas.benchmark import (
    BenchmarkCaseManifest,
    BenchmarkCaseStatus,
    BenchmarkPhase,
    BenchmarkResource,
    BenchmarkResourceRole,
    GroundTruthPolicy,
    ModelingCategory,
)


def test_benchmark_api_preserves_all_cases_and_never_promotes_fixture_services(
    tmp_path: Path,
) -> None:
    bodies = {f"/case-{index}.txt": f"official problem {index}".encode() for index in range(1, 4)}
    manifest_root = tmp_path / "manifests"
    for index in range(1, 4):
        manifest = _manifest(index, bodies[f"/case-{index}.txt"])
        case_root = manifest_root / f"case-{index:03d}"
        case_root.mkdir(parents=True)
        (case_root / "manifest.json").write_text(
            json.dumps(manifest.model_dump(mode="json")), encoding="utf-8"
        )
    settings = Settings(
        environment="test",
        database_url="sqlite+pysqlite:///:memory:",
        benchmark_repository_root=tmp_path,
        benchmark_manifest_root=manifest_root,
        benchmark_cache_root=tmp_path / "cache",
        benchmark_artifact_root=tmp_path / "runs",
        benchmark_code_commit="a" * 40,
    )
    app = create_app(settings)
    Base.metadata.create_all(app.state.engine)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=bodies[request.url.path], request=request)

    cache = BenchmarkResourceCache(
        tmp_path / "cache",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    app.state.benchmark_resource_cache.close()
    app.state.benchmark_resource_cache = cache
    app.state.benchmark_workflow = BenchmarkWorkflow(
        repository=app.state.benchmark_repository,
        manifests=BenchmarkManifestRegistry(manifest_root),
        resource_cache=cache,
        inspector=BenchmarkStructureInspector(),
        providers=app.state.providers,
        literature_source=FixtureLiteratureSource(records=()),
        profile=comap_mcm_2024_profile(),
        code_commit="a" * 40,
        source_tree_digest="f" * 64,
        working_tree_dirty=True,
        output_root=tmp_path / "runs",
    )
    payload = {
        "case_ids": ["BENCH-case-1", "BENCH-case-2", "BENCH-case-3"],
        "config": {
            "provider": "openai",
            "model": "gpt-live",
            "reasoning_tier": "high",
            "pricing": {"version": "test"},
        },
    }

    with TestClient(app) as client:
        response = client.post("/api/v1/benchmarks/runs", json=payload)
        assert response.status_code == 200, response.text
        report = response.json()
        run_id = report["run"]["run_id"]
        assert report["acceptance"]["status"] == "NOT_READY"
        assert report["acceptance"]["real_cases_attempted"] == 3
        assert report["acceptance"]["live_provider_status"] == "BLOCKED"
        assert report["acceptance"]["live_literature_status"] == "BLOCKED"
        assert len(report["attempts"]) == 3
        assert len(report["results"]) == 3
        assert {item["status"] for item in report["results"]} == {
            BenchmarkCaseStatus.BLOCKED_ENVIRONMENT.value
        }
        assert all(not item["provider_is_live"] for item in report["attempts"])
        assert all(not item["literature_is_live"] for item in report["attempts"])

        assert client.get(f"/api/v1/benchmarks/runs/{run_id}").status_code == 200
        cases = client.get(f"/api/v1/benchmarks/runs/{run_id}/cases")
        assert cases.status_code == 200
        assert len(cases.json()) == 3
        rebuilt = client.get(f"/api/v1/benchmarks/runs/{run_id}/report")
        assert rebuilt.status_code == 200
        assert rebuilt.json()["report_digest"] == report["report_digest"]

    assert list((tmp_path / "runs" / run_id).glob("benchmark-*"))


def _manifest(index: int, content: bytes) -> BenchmarkCaseManifest:
    return BenchmarkCaseManifest(
        benchmark_id=f"BENCH-case-{index}",
        competition="Official Test Competition",
        year=2024,
        problem_id=str(index),
        title=f"Case {index}",
        modeling_category=(
            ModelingCategory.DATA_PREDICTION
            if index == 1
            else ModelingCategory.OPTIMIZATION_DECISION
        ),
        difficulty="MODERATE" if index == 1 else "DIFFICULT",
        resources=[
            BenchmarkResource(
                resource_id=f"RESOURCE-case-{index}",
                phase=BenchmarkPhase.SOLVE,
                role=BenchmarkResourceRole.PROBLEM,
                source_url=f"https://www.contest.comap.com/case-{index}.txt",
                sha256=sha256_bytes(content),
                media_type="text/plain",
                local_filename=f"case-{index}.txt",
                expected_size_bytes=len(content),
                distribution_notes="generated integration fixture",
            )
        ],
        requires_external_data=False,
        requires_literature=True,
        requires_solver=True,
        literature_queries=["live benchmark query"],
        ground_truth_policy=GroundTruthPolicy(required_outputs=["answer every task"]),
        license_or_distribution_notes="generated integration fixture",
    )
