from uuid import UUID

from fastapi import APIRouter, Query, Request

from mathmodel_ai.benchmark.manifests import BenchmarkManifestRegistry
from mathmodel_ai.benchmark.repository import BenchmarkRepository
from mathmodel_ai.benchmark.workflow import BenchmarkWorkflow
from mathmodel_ai.schemas.benchmark import (
    BenchmarkCaseResult,
    BenchmarkCaseSummary,
    BenchmarkReport,
    BenchmarkRun,
    BenchmarkRunRequest,
)

router = APIRouter(prefix="/api/v1/benchmarks", tags=["benchmarks"])


def _services(request: Request) -> tuple[BenchmarkRepository, BenchmarkWorkflow]:
    repository: BenchmarkRepository = request.app.state.benchmark_repository
    workflow: BenchmarkWorkflow = request.app.state.benchmark_workflow
    return repository, workflow


@router.post("/runs", response_model=BenchmarkReport)
async def create_benchmark_run(payload: BenchmarkRunRequest, request: Request) -> BenchmarkReport:
    _, workflow = _services(request)
    return (await workflow.run(payload)).report


@router.get("/runs", response_model=list[BenchmarkRun])
def list_benchmark_runs(
    request: Request,
    limit: int = Query(default=50, ge=1, le=100),
) -> list[BenchmarkRun]:
    repository, _ = _services(request)
    return repository.list_runs(limit=limit)


@router.get("/cases", response_model=list[BenchmarkCaseSummary])
def list_benchmark_cases(request: Request) -> list[BenchmarkCaseSummary]:
    manifests: BenchmarkManifestRegistry = request.app.state.benchmark_manifest_registry
    return [
        BenchmarkCaseSummary(
            benchmark_id=manifest.benchmark_id,
            title=manifest.title,
            competition=manifest.competition,
            year=manifest.year,
            modeling_category=manifest.modeling_category,
            difficulty=manifest.difficulty,
            requires_literature=manifest.requires_literature,
            requires_solver=manifest.requires_solver,
        )
        for manifest in manifests.list()
    ]


@router.get("/runs/{run_id}", response_model=BenchmarkRun)
def get_benchmark_run(run_id: UUID, request: Request) -> BenchmarkRun:
    repository, _ = _services(request)
    return repository.get_run(run_id)


@router.get("/runs/{run_id}/cases", response_model=list[BenchmarkCaseResult])
def get_benchmark_cases(run_id: UUID, request: Request) -> list[BenchmarkCaseResult]:
    repository, workflow = _services(request)
    repository.get_run(run_id)
    return workflow.report(run_id).results


@router.get("/runs/{run_id}/report", response_model=BenchmarkReport)
def get_benchmark_report(run_id: UUID, request: Request) -> BenchmarkReport:
    _, workflow = _services(request)
    return workflow.report(run_id)
