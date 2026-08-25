from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, File, Request, UploadFile

from mathmodel_ai.api.schemas import (
    AgentRunSummary,
    DataAnalyzeRequest,
    DataAnalyzeResponse,
    ExecutionRequest,
    ExecutionResponse,
    FileIngestResponse,
)
from mathmodel_ai.core.errors import QualityGateError
from mathmodel_ai.data.repository import DataRepository
from mathmodel_ai.data.workflow import DataExecutionWorkflow
from mathmodel_ai.reasoning.repository import ReasoningRepository
from mathmodel_ai.schemas.data import DataProfile, DatasetRecord, DataUnderstanding
from mathmodel_ai.schemas.execution import ExecutionRecord
from mathmodel_ai.schemas.files import ArtifactRecord, RegisteredFile

router = APIRouter(prefix="/api/v1/projects", tags=["data-execution"])


def _services(request: Request) -> tuple[DataRepository, DataExecutionWorkflow]:
    repository: DataRepository = request.app.state.data_repository
    workflow: DataExecutionWorkflow = request.app.state.data_execution_workflow
    return repository, workflow


@router.post("/{project_id}/files", response_model=FileIngestResponse, status_code=201)
def upload_file(
    project_id: UUID,
    request: Request,
    upload: Annotated[UploadFile, File()],
) -> FileIngestResponse:
    _, workflow = _services(request)
    outcome = workflow.ingest_file(
        project_id,
        upload.file,
        original_name=upload.filename or "",
        declared_mime_type=upload.content_type,
    )
    bundle = outcome.processed.profile_bundle
    return FileIngestResponse(
        parsed_file=outcome.processed.parsed_file,
        datasets=outcome.processed.datasets,
        data_profiles=bundle.profiles if bundle is not None else [],
        relationships=bundle.cross_dataset_relationships if bundle is not None else [],
        artifacts=outcome.processed.artifacts,
        state_version=outcome.state.version,
        data_stage=outcome.state.data_stage,
        gate=outcome.gate,
    )


@router.get("/{project_id}/files", response_model=list[RegisteredFile])
def list_files(project_id: UUID, request: Request) -> list[RegisteredFile]:
    repository, _ = _services(request)
    return repository.list_files(project_id)


@router.get("/{project_id}/datasets", response_model=list[DatasetRecord])
def list_datasets(project_id: UUID, request: Request) -> list[DatasetRecord]:
    repository, _ = _services(request)
    return repository.list_datasets(project_id)


@router.get("/{project_id}/data-profiles", response_model=list[DataProfile])
def list_profiles(project_id: UUID, request: Request) -> list[DataProfile]:
    repository, _ = _services(request)
    return repository.list_profiles(project_id)


@router.get("/{project_id}/artifacts", response_model=list[ArtifactRecord])
def list_artifacts(project_id: UUID, request: Request) -> list[ArtifactRecord]:
    repository, _ = _services(request)
    return repository.list_artifacts(project_id)


@router.post("/{project_id}/data/analyze", response_model=DataAnalyzeResponse)
async def analyze_data(
    project_id: UUID,
    payload: DataAnalyzeRequest,
    request: Request,
) -> DataAnalyzeResponse:
    _, workflow = _services(request)
    outcome = await workflow.analyze_data(
        project_id,
        media_file_ids=payload.media_file_ids,
        user_guidance=payload.user_guidance,
    )
    return DataAnalyzeResponse(
        output=outcome.output,
        state_version=outcome.state.version,
        data_stage=outcome.state.data_stage,
        gate=outcome.gate,
        agent_run=AgentRunSummary.from_run(outcome.run),
        is_mock=outcome.run.is_mock,
    )


@router.get("/{project_id}/data-understanding", response_model=DataUnderstanding)
def get_data_understanding(project_id: UUID, request: Request) -> DataUnderstanding:
    repository: ReasoningRepository = request.app.state.reasoning_repository
    state = repository.load_current(project_id)
    if state.data_understanding is None:
        raise QualityGateError("data understanding has not been completed")
    return state.data_understanding


@router.post("/{project_id}/executions", response_model=ExecutionResponse)
def execute_code(
    project_id: UUID,
    payload: ExecutionRequest,
    request: Request,
) -> ExecutionResponse:
    _, workflow = _services(request)
    outcome = workflow.execute_code(
        project_id,
        payload.code,
        input_file_ids=payload.input_file_ids,
    )
    return ExecutionResponse(
        execution=outcome.execution.record,
        artifacts=outcome.execution.artifact_records,
        state_version=outcome.state.version,
        data_stage=outcome.state.data_stage,
        gate=outcome.gate,
    )


@router.get("/{project_id}/executions", response_model=list[ExecutionRecord])
def list_executions(project_id: UUID, request: Request) -> list[ExecutionRecord]:
    repository, _ = _services(request)
    return repository.list_executions(project_id)
