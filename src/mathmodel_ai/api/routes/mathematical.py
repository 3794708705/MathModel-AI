from uuid import UUID

from fastapi import APIRouter, Request

from mathmodel_ai.api.schemas import (
    AgentRunSummary,
    MathematicalRunRequest,
    MathematicalRunResponse,
    ModelBuildRequest,
    ModelBuildResponse,
    SolveRequest,
    SolveResponse,
)
from mathmodel_ai.mathematical.repository import MathematicalRepository
from mathmodel_ai.mathematical.workflow import MathematicalWorkflow
from mathmodel_ai.schemas.mathematical import MathematicalModel
from mathmodel_ai.schemas.program import GeneratedProgram
from mathmodel_ai.schemas.results import EvidenceChainReport, ResultRecord
from mathmodel_ai.schemas.solver import SolverRun

router = APIRouter(prefix="/api/v1/projects", tags=["mathematical-solver"])


def _services(request: Request) -> tuple[MathematicalRepository, MathematicalWorkflow]:
    repository: MathematicalRepository = request.app.state.mathematical_repository
    workflow: MathematicalWorkflow = request.app.state.mathematical_workflow
    return repository, workflow


@router.post("/{project_id}/model/build", response_model=ModelBuildResponse)
async def build_model(
    project_id: UUID,
    payload: ModelBuildRequest,
    request: Request,
) -> ModelBuildResponse:
    _, workflow = _services(request)
    outcome = await workflow.build_model(project_id, user_guidance=payload.user_guidance)
    return ModelBuildResponse(
        mathematical_model=outcome.model,
        algorithm_plan=outcome.plan,
        state_version=outcome.state.version,
        gate=outcome.gate,
        agent_run=AgentRunSummary.from_run(outcome.run),
        is_mock=outcome.run.is_mock,
    )


@router.get("/{project_id}/mathematical-model", response_model=MathematicalModel)
def get_mathematical_model(project_id: UUID, request: Request) -> MathematicalModel:
    repository, _ = _services(request)
    return repository.get_model(project_id)


@router.post("/{project_id}/solve", response_model=SolveResponse)
async def solve_model(
    project_id: UUID,
    payload: SolveRequest,
    request: Request,
) -> SolveResponse:
    _, workflow = _services(request)
    outcome = await workflow.solve(
        project_id,
        options=payload.options,
        execution_strategy=payload.execution_strategy,
        user_guidance=payload.user_guidance,
    )
    return SolveResponse(
        result=outcome.result,
        solver_run=outcome.solver_run,
        generated_program=outcome.execution.program,
        execution=outcome.execution.execution.record,
        route=outcome.route,
        execution_strategy=outcome.strategy,
        code_agent_run=(
            AgentRunSummary.from_run(outcome.code_agent_run)
            if outcome.code_agent_run is not None
            else None
        ),
        state_version=outcome.state.version,
        gate=outcome.gate,
    )


@router.post("/{project_id}/mathematical/run", response_model=MathematicalRunResponse)
async def run_mathematical_workflow(
    project_id: UUID,
    payload: MathematicalRunRequest,
    request: Request,
) -> MathematicalRunResponse:
    _, workflow = _services(request)
    outcome = await workflow.run(
        project_id,
        user_guidance=payload.user_guidance,
        options=payload.options,
        execution_strategy=payload.execution_strategy,
    )
    return MathematicalRunResponse(
        model_stage=ModelBuildResponse(
            mathematical_model=outcome.model_stage.model,
            algorithm_plan=outcome.model_stage.plan,
            state_version=outcome.model_stage.state.version,
            gate=outcome.model_stage.gate,
            agent_run=AgentRunSummary.from_run(outcome.model_stage.run),
            is_mock=outcome.model_stage.run.is_mock,
        ),
        solve_stage=SolveResponse(
            result=outcome.solve_stage.result,
            solver_run=outcome.solve_stage.solver_run,
            generated_program=outcome.solve_stage.execution.program,
            execution=outcome.solve_stage.execution.execution.record,
            route=outcome.solve_stage.route,
            execution_strategy=outcome.solve_stage.strategy,
            code_agent_run=(
                AgentRunSummary.from_run(outcome.solve_stage.code_agent_run)
                if outcome.solve_stage.code_agent_run is not None
                else None
            ),
            state_version=outcome.solve_stage.state.version,
            gate=outcome.solve_stage.gate,
        ),
    )


@router.get("/{project_id}/results", response_model=list[ResultRecord])
def list_results(project_id: UUID, request: Request) -> list[ResultRecord]:
    repository, _ = _services(request)
    return repository.list_results(project_id)


@router.get("/{project_id}/solver-runs", response_model=list[SolverRun])
def list_solver_runs(project_id: UUID, request: Request) -> list[SolverRun]:
    repository, _ = _services(request)
    return repository.list_solver_runs(project_id)


@router.get("/{project_id}/generated-programs", response_model=list[GeneratedProgram])
def list_programs(project_id: UUID, request: Request) -> list[GeneratedProgram]:
    repository, _ = _services(request)
    return repository.list_programs(project_id)


@router.get(
    "/{project_id}/results/{result_id}/evidence",
    response_model=EvidenceChainReport,
)
def verify_result_evidence(
    project_id: UUID,
    result_id: UUID,
    request: Request,
) -> EvidenceChainReport:
    repository, _ = _services(request)
    return repository.verify_result_evidence(project_id, result_id)
