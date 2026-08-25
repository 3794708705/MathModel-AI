from uuid import UUID

from fastapi import APIRouter, Request

from mathmodel_ai.api.schemas import (
    AgentRunSummary,
    ModelExplorationStageResponse,
    ModelSelectionStageResponse,
    NotesRequest,
    ProblemAnalysisStageResponse,
    ProjectProblemCreateRequest,
    ReasoningRunRequest,
    ReasoningRunResponse,
)
from mathmodel_ai.core.errors import QualityGateError
from mathmodel_ai.reasoning.repository import ReasoningRepository
from mathmodel_ai.reasoning.workflow import ReasoningWorkflow
from mathmodel_ai.schemas.model_selection import ModelCandidate, ModelSelection
from mathmodel_ai.schemas.problem_analysis import ProblemAnalysis
from mathmodel_ai.schemas.problem_state import ProblemState

router = APIRouter(prefix="/api/v1/projects", tags=["reasoning"])


def _services(request: Request) -> tuple[ReasoningRepository, ReasoningWorkflow]:
    repository: ReasoningRepository = request.app.state.reasoning_repository
    workflow: ReasoningWorkflow = request.app.state.reasoning_workflow
    return repository, workflow


@router.post("", response_model=ProblemState, status_code=201)
def create_project(payload: ProjectProblemCreateRequest, request: Request) -> ProblemState:
    repository, _ = _services(request)
    return repository.create_project_problem(**payload.model_dump())


@router.post("/{project_id}/problem/analyze", response_model=ProblemAnalysisStageResponse)
async def analyze_problem(
    project_id: UUID, payload: NotesRequest, request: Request
) -> ProblemAnalysisStageResponse:
    _, workflow = _services(request)
    outcome = await workflow.analyze(project_id, user_notes=payload.notes)
    summary = AgentRunSummary.from_run(outcome.run)
    return ProblemAnalysisStageResponse(
        output=outcome.output,
        state_version=outcome.state.version,
        gate=outcome.gate,
        agent_run=summary,
        is_mock=outcome.run.is_mock,
    )


@router.post("/{project_id}/models/explore", response_model=ModelExplorationStageResponse)
async def explore_models(
    project_id: UUID, payload: NotesRequest, request: Request
) -> ModelExplorationStageResponse:
    _, workflow = _services(request)
    outcome = await workflow.explore(project_id, user_guidance=payload.notes)
    summary = AgentRunSummary.from_run(outcome.run)
    return ModelExplorationStageResponse(
        output=outcome.output,
        state_version=outcome.state.version,
        gate=outcome.gate,
        agent_run=summary,
        is_mock=outcome.run.is_mock,
    )


@router.post("/{project_id}/models/select", response_model=ModelSelectionStageResponse)
async def select_model(
    project_id: UUID, payload: NotesRequest, request: Request
) -> ModelSelectionStageResponse:
    _, workflow = _services(request)
    outcome = await workflow.select(project_id, jury_notes=payload.notes)
    summary = AgentRunSummary.from_run(outcome.run)
    return ModelSelectionStageResponse(
        output=outcome.output,
        state_version=outcome.state.version,
        gate=outcome.gate,
        agent_run=summary,
        is_mock=outcome.run.is_mock,
    )


@router.post("/{project_id}/reasoning/run", response_model=ReasoningRunResponse)
async def run_reasoning(
    project_id: UUID, payload: ReasoningRunRequest, request: Request
) -> ReasoningRunResponse:
    _, workflow = _services(request)
    outcome = await workflow.run(
        project_id,
        user_notes=payload.user_notes,
        user_guidance=payload.user_guidance,
        jury_notes=payload.jury_notes,
    )
    return ReasoningRunResponse(
        state=outcome.state,
        agent_runs=[AgentRunSummary.from_run(run) for run in outcome.runs],
        is_mock=outcome.is_mock,
    )


@router.get("/{project_id}/problem-analysis", response_model=ProblemAnalysis)
def get_problem_analysis(project_id: UUID, request: Request) -> ProblemAnalysis:
    repository, _ = _services(request)
    analysis = repository.load_current(project_id).problem_analysis
    if analysis is None:
        raise QualityGateError("problem analysis has not been completed")
    return analysis


@router.get("/{project_id}/models", response_model=list[ModelCandidate])
def get_models(project_id: UUID, request: Request) -> list[ModelCandidate]:
    repository, _ = _services(request)
    return repository.load_current(project_id).candidate_models


@router.get("/{project_id}/model-selection", response_model=ModelSelection)
def get_model_selection(project_id: UUID, request: Request) -> ModelSelection:
    repository, _ = _services(request)
    selection = repository.load_current(project_id).model_selection
    if selection is None:
        raise QualityGateError("model selection has not been completed")
    return selection
