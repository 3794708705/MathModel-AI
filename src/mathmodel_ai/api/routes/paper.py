from uuid import UUID

from fastapi import APIRouter, Request

from mathmodel_ai.api.schemas import (
    LiteratureSearchResponse,
    PaperRunRequest,
    PaperRunResponse,
)
from mathmodel_ai.paper.repository import PaperRepository
from mathmodel_ai.paper.workflow import PaperWorkflow
from mathmodel_ai.schemas.paper import (
    PaperArtifact,
    PaperCompileRecord,
    PaperQualityReport,
    PaperVersion,
    ReferenceRecord,
)

router = APIRouter(prefix="/api/v1/projects", tags=["evidence-paper"])


def _services(request: Request) -> tuple[PaperRepository, PaperWorkflow]:
    repository: PaperRepository = request.app.state.paper_repository
    workflow: PaperWorkflow = request.app.state.paper_workflow
    return repository, workflow


@router.post("/{project_id}/literature/search", response_model=LiteratureSearchResponse)
async def search_literature(project_id: UUID, request: Request) -> LiteratureSearchResponse:
    _, workflow = _services(request)
    references, checks, plan = await workflow.search_literature(project_id)
    return LiteratureSearchResponse(references=references, metadata_checks=checks, plan=plan)


@router.get("/{project_id}/literature", response_model=list[ReferenceRecord])
def list_literature(project_id: UUID, request: Request) -> list[ReferenceRecord]:
    repository, _ = _services(request)
    return repository.list_references(project_id)


async def _run(project_id: UUID, payload: PaperRunRequest, request: Request) -> PaperRunResponse:
    repository, workflow = _services(request)
    outcome = await workflow.run(project_id, competition_profile=payload.competition_profile)
    return PaperRunResponse(
        paper=outcome.version,
        quality=outcome.quality,
        compile_record=outcome.compilation.record,
        artifacts=repository.list_artifacts(project_id, outcome.version.paper_id),
    )


@router.post("/{project_id}/paper/run", response_model=PaperRunResponse)
async def run_paper(
    project_id: UUID, payload: PaperRunRequest, request: Request
) -> PaperRunResponse:
    return await _run(project_id, payload, request)


@router.post("/{project_id}/paper/build", response_model=PaperRunResponse)
async def build_paper(
    project_id: UUID, payload: PaperRunRequest, request: Request
) -> PaperRunResponse:
    return await _run(project_id, payload, request)


@router.get("/{project_id}/paper", response_model=PaperVersion)
def get_paper(project_id: UUID, request: Request) -> PaperVersion:
    repository, _ = _services(request)
    return repository.get_version(project_id)


@router.post("/{project_id}/paper/validate", response_model=PaperQualityReport)
def validate_paper(project_id: UUID, request: Request) -> PaperQualityReport:
    repository, _ = _services(request)
    return repository.get_quality(project_id)


@router.post("/{project_id}/paper/render", response_model=PaperCompileRecord)
def render_paper(project_id: UUID, request: Request) -> PaperCompileRecord:
    repository, _ = _services(request)
    return repository.get_compile(project_id)


@router.get("/{project_id}/paper/artifacts", response_model=list[PaperArtifact])
def list_paper_artifacts(project_id: UUID, request: Request) -> list[PaperArtifact]:
    repository, _ = _services(request)
    paper = repository.get_version(project_id)
    return repository.list_artifacts(project_id, paper.paper_id)
