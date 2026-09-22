from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from mathmodel_ai.core.errors import ResourceNotFoundError
from mathmodel_ai.data.repository import DataRepository
from mathmodel_ai.db.models import (
    IndependentVerificationJobRecord as Job,
)
from mathmodel_ai.db.models import (
    IndependentVerificationPlanRecord as Plan,
)
from mathmodel_ai.db.session import session_scope
from mathmodel_ai.schemas.independent_verification import ReplayRecord, VerificationPlan
from mathmodel_ai.verification.metric_recompute import content_digest


class IndependentVerificationRepository:
    def __init__(self, factory: sessionmaker[Session]):
        self._factory = factory

    def register(self, plan: VerificationPlan, requirements_digest: str) -> None:
        with session_scope(self._factory) as session:
            existing = session.scalar(select(Plan).where(Plan.attempt_id == plan.attempt_id))
            if existing is not None:
                if existing.plan_digest != content_digest(plan):
                    raise ValueError("plan is immutable; use a new benchmark attempt")
                return
            session.add(
                Plan(
                    id=plan.plan_id,
                    attempt_id=plan.attempt_id,
                    result_id=plan.result_id,
                    plan_digest=content_digest(plan),
                    requirements_digest=requirements_digest,
                    plan_json=plan.model_dump(mode="json"),
                )
            )

    def get(self, attempt_id: UUID) -> tuple[VerificationPlan, str]:
        with session_scope(self._factory) as session:
            row = session.scalar(select(Plan).where(Plan.attempt_id == attempt_id))
            if row is None:
                raise ResourceNotFoundError("independent verification plan is not configured")
            plan = VerificationPlan.model_validate(row.plan_json)
            if (
                row.id != plan.plan_id
                or row.attempt_id != plan.attempt_id
                or row.result_id != plan.result_id
                or row.plan_digest != content_digest(plan)
            ):
                raise ValueError("independent plan integrity failure")
            return plan, row.requirements_digest

    def claim(self, plan_id: UUID, operation: str) -> UUID | None:
        job_id = uuid4()
        try:
            with session_scope(self._factory) as session:
                session.add(
                    Job(
                        id=job_id,
                        plan_id=plan_id,
                        operation=operation,
                        status="RUNNING",
                        started_at=datetime.now(UTC),
                    )
                )
        except IntegrityError:
            # Verify the conflict was the idempotency key, not a missing foreign key.
            with session_scope(self._factory) as session:
                if (
                    session.scalar(
                        select(Job.id).where(Job.plan_id == plan_id, Job.operation == operation)
                    )
                    is None
                ):
                    raise
            return None
        return job_id

    def finish(
        self, job_id: UUID, payload: dict[str, Any], *, replay: ReplayRecord | None = None
    ) -> None:
        with session_scope(self._factory) as session:
            row = session.scalar(select(Job).where(Job.id == job_id).with_for_update())
            if row is None or row.status != "RUNNING":
                raise ValueError("job terminal outcome is immutable")
            if replay is not None:
                session.add(DataRepository._execution_model(replay.execution))
                session.flush()
                session.add_all(DataRepository._artifact_model(a) for a in replay.artifacts)
                row.execution_id = replay.execution.run_id
            row.status = "COMPLETE"
            row.payload_json = payload
            row.payload_digest = content_digest(payload)
            row.finished_at = datetime.now(UTC)

    def jobs(self, plan_id: UUID) -> dict[str, dict[str, Any] | None]:
        with session_scope(self._factory) as session:
            rows = session.scalars(select(Job).where(Job.plan_id == plan_id))
            outputs: dict[str, dict[str, Any] | None] = {}
            for row in rows:
                if row.status == "RUNNING":
                    if row.payload_json is not None or row.finished_at is not None:
                        raise ValueError("running job contains a terminal outcome")
                    outputs[row.operation] = None
                elif (
                    row.status == "COMPLETE"
                    and row.payload_json is not None
                    and row.payload_digest == content_digest(row.payload_json)
                    and row.finished_at is not None
                ):
                    replay_json = row.payload_json.get("replay")
                    if replay_json and str(row.execution_id) != replay_json["execution"]["run_id"]:
                        raise ValueError("job execution identity mismatch")
                    outputs[row.operation] = row.payload_json
                else:
                    raise ValueError("independent job integrity failure")
            return outputs
