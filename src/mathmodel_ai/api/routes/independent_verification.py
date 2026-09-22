from uuid import UUID

from fastapi import APIRouter, HTTPException, Request

from mathmodel_ai.schemas.independent_verification import IndependentVerificationView
from mathmodel_ai.verification.independent_service import IndependentVerificationService

router = APIRouter(prefix="/api/v1/benchmarks/attempts", tags=["independent-verification"])


def _service(request: Request) -> IndependentVerificationService:
    service: IndependentVerificationService = request.app.state.independent_verification
    return service


@router.get("/{attempt_id}/verification", response_model=IndependentVerificationView)
def get_verification(attempt_id: UUID, request: Request) -> IndependentVerificationView:
    return _service(request).view(attempt_id)


@router.post("/{attempt_id}/verification/recompute", response_model=IndependentVerificationView)
def recompute_metrics(attempt_id: UUID, request: Request) -> IndependentVerificationView:
    try:
        return _service(request).recompute(attempt_id)
    except ValueError as exc:
        raise HTTPException(
            409,
            "Independent verification contract or evidence is invalid; "
            "inspect verification status.",
        ) from exc


@router.post(
    "/{attempt_id}/verification/scenarios/{scenario_id}/replay",
    response_model=IndependentVerificationView,
)
def replay_scenario(
    attempt_id: UUID, scenario_id: str, request: Request
) -> IndependentVerificationView:
    try:
        return _service(request).replay(attempt_id, scenario_id)
    except ValueError as exc:
        raise HTTPException(
            409, "Independent replay contract or evidence is invalid; inspect verification status."
        ) from exc
