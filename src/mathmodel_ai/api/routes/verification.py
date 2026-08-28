from uuid import UUID

from fastapi import APIRouter, Request

from mathmodel_ai.api.schemas import (
    AgentRunSummary,
    ModelRepairRunRequest,
    ModelRepairRunResponse,
    RedTeamRunRequest,
    RedTeamRunResponse,
    RepairLoopRunRequest,
    RepairLoopRunResponse,
    RobustnessRunRequest,
    RobustnessRunResponse,
    SensitivityRunRequest,
    SensitivityRunResponse,
    ValidationRunRequest,
    ValidationRunResponse,
    VerificationRunRequest,
    VerificationRunResponse,
)
from mathmodel_ai.schemas.verification import (
    RedTeamReport,
    RepairCycleRecord,
    RobustnessReport,
    SensitivityReport,
    ValidationReport,
)
from mathmodel_ai.verification.repository import VerificationRepository
from mathmodel_ai.verification.workflow import VerificationWorkflow

router = APIRouter(prefix="/api/v1/projects", tags=["verification-repair"])


def _services(request: Request) -> tuple[VerificationRepository, VerificationWorkflow]:
    repository: VerificationRepository = request.app.state.verification_repository
    workflow: VerificationWorkflow = request.app.state.verification_workflow
    return repository, workflow


@router.post("/{project_id}/verification/validate", response_model=ValidationRunResponse)
def validate_result(
    project_id: UUID,
    payload: ValidationRunRequest,
    request: Request,
) -> ValidationRunResponse:
    _, workflow = _services(request)
    outcome = workflow.validate(project_id, result_id=payload.result_id)
    return ValidationRunResponse(
        report=outcome.report,
        state_version=outcome.state.version,
        gate=outcome.gate,
    )


@router.post("/{project_id}/verification/sensitivity", response_model=SensitivityRunResponse)
def run_sensitivity(
    project_id: UUID,
    payload: SensitivityRunRequest,
    request: Request,
) -> SensitivityRunResponse:
    _, workflow = _services(request)
    outcome = workflow.sensitivity(project_id, config=payload.config)
    return SensitivityRunResponse(
        report=outcome.report,
        state_version=outcome.state.version,
        gate=outcome.gate,
    )


@router.post("/{project_id}/verification/robustness", response_model=RobustnessRunResponse)
def run_robustness(
    project_id: UUID,
    payload: RobustnessRunRequest,
    request: Request,
) -> RobustnessRunResponse:
    _, workflow = _services(request)
    outcome = workflow.robustness(project_id, config=payload.config)
    return RobustnessRunResponse(
        report=outcome.report,
        state_version=outcome.state.version,
        gate=outcome.gate,
    )


@router.post("/{project_id}/verification/red-team", response_model=RedTeamRunResponse)
async def run_red_team(
    project_id: UUID,
    payload: RedTeamRunRequest,
    request: Request,
) -> RedTeamRunResponse:
    _, workflow = _services(request)
    outcome = await workflow.red_team(project_id, user_guidance=payload.user_guidance)
    return RedTeamRunResponse(
        report=outcome.report,
        state_version=outcome.state.version,
        gate=outcome.gate,
        verified_gate=outcome.verified_gate,
        verified_result_id=outcome.state.verified_result_id,
        agent_run=AgentRunSummary.from_run(outcome.run),
    )


@router.post("/{project_id}/model/repair", response_model=ModelRepairRunResponse)
async def repair_model(
    project_id: UUID,
    payload: ModelRepairRunRequest,
    request: Request,
) -> ModelRepairRunResponse:
    _, workflow = _services(request)
    outcome = await workflow.repair(project_id, user_guidance=payload.user_guidance)
    return ModelRepairRunResponse(
        output=outcome.output,
        cycle=outcome.cycle,
        state_version=outcome.state.version,
        model_gate=outcome.model_gate,
        repair_gate=outcome.repair_gate,
        agent_run=AgentRunSummary.from_run(outcome.run),
    )


@router.post("/{project_id}/model/repair-loop", response_model=RepairLoopRunResponse)
async def repair_model_until_clear(
    project_id: UUID,
    payload: RepairLoopRunRequest,
    request: Request,
) -> RepairLoopRunResponse:
    _, workflow = _services(request)
    outcome = await workflow.repair_until_clear(
        project_id,
        sensitivity_config=payload.sensitivity,
        robustness_config=payload.robustness,
        user_guidance=payload.user_guidance,
    )
    return RepairLoopRunResponse(
        state=outcome.state,
        cycles=[item.repair.cycle for item in outcome.iterations],
        final_red_team=outcome.final_red_team,
        resolved=outcome.resolved,
        exhausted=outcome.exhausted,
    )


@router.post("/{project_id}/verification/run", response_model=VerificationRunResponse)
async def run_verification(
    project_id: UUID,
    payload: VerificationRunRequest,
    request: Request,
) -> VerificationRunResponse:
    _, workflow = _services(request)
    outcome = await workflow.run(
        project_id,
        sensitivity_config=payload.sensitivity,
        robustness_config=payload.robustness,
        user_guidance=payload.user_guidance,
    )
    return VerificationRunResponse(
        validation=ValidationRunResponse(
            report=outcome.validation.report,
            state_version=outcome.validation.state.version,
            gate=outcome.validation.gate,
        ),
        sensitivity=SensitivityRunResponse(
            report=outcome.sensitivity.report,
            state_version=outcome.sensitivity.state.version,
            gate=outcome.sensitivity.gate,
        ),
        robustness=RobustnessRunResponse(
            report=outcome.robustness.report,
            state_version=outcome.robustness.state.version,
            gate=outcome.robustness.gate,
        ),
        red_team=RedTeamRunResponse(
            report=outcome.red_team.report,
            state_version=outcome.red_team.state.version,
            gate=outcome.red_team.gate,
            verified_gate=outcome.red_team.verified_gate,
            verified_result_id=outcome.red_team.state.verified_result_id,
            agent_run=AgentRunSummary.from_run(outcome.red_team.run),
        ),
    )


@router.get("/{project_id}/validation-runs/latest", response_model=ValidationReport)
def get_validation(project_id: UUID, request: Request) -> ValidationReport:
    repository, _ = _services(request)
    return repository.get_validation(project_id)


@router.get("/{project_id}/sensitivity-runs/latest", response_model=SensitivityReport)
def get_sensitivity(project_id: UUID, request: Request) -> SensitivityReport:
    repository, _ = _services(request)
    return repository.get_sensitivity(project_id)


@router.get("/{project_id}/robustness-runs/latest", response_model=RobustnessReport)
def get_robustness(project_id: UUID, request: Request) -> RobustnessReport:
    repository, _ = _services(request)
    return repository.get_robustness(project_id)


@router.get("/{project_id}/red-team-reports/latest", response_model=RedTeamReport)
def get_red_team(project_id: UUID, request: Request) -> RedTeamReport:
    repository, _ = _services(request)
    return repository.get_red_team(project_id)


@router.get("/{project_id}/repair-cycles", response_model=list[RepairCycleRecord])
def list_repairs(project_id: UUID, request: Request) -> list[RepairCycleRecord]:
    repository, _ = _services(request)
    return repository.list_repairs(project_id)
