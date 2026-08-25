from fastapi import APIRouter, HTTPException, Request, status
from sqlalchemy import Engine

from mathmodel_ai.api.schemas import HealthResponse, ReadinessResponse
from mathmodel_ai.db.session import database_is_ready

router = APIRouter(tags=["health"])


@router.get("/health/live", response_model=HealthResponse)
def liveness() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/health/ready", response_model=ReadinessResponse)
def readiness(request: Request) -> ReadinessResponse:
    engine: Engine = request.app.state.engine
    if not database_is_ready(engine):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="database is unavailable",
        )
    return ReadinessResponse(status="ready", database="ready")
