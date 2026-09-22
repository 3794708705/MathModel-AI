from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from mathmodel_ai.agents.base import AgentRunResult
from mathmodel_ai.core.errors import ResourceNotFoundError
from mathmodel_ai.db.models import (
    AgentRunRecord,
    ModelDecisionRecord,
    Problem,
    ProblemStateRecord,
    Project,
)
from mathmodel_ai.db.session import session_scope
from mathmodel_ai.schemas.model_selection import ModelDecisionEvidence
from mathmodel_ai.schemas.problem_state import ProblemState, ProjectSummary


class ReasoningRepository:
    """Short-transaction persistence for versioned shared state and audit evidence."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def create_project_problem(
        self,
        *,
        name: str,
        title: str,
        raw_problem: str,
        competition: str | None = None,
        deadline: datetime | None = None,
    ) -> ProblemState:
        project_id = uuid4()
        problem_id = uuid4()
        state = ProblemState(
            project_id=project_id,
            problem_id=problem_id,
            title=title,
            raw_problem=raw_problem,
            competition=competition,
            deadline=deadline,
        )
        with session_scope(self._session_factory) as session:
            session.add_all(
                [
                    Project(id=project_id, name=name),
                    Problem(
                        id=problem_id,
                        project_id=project_id,
                        title=title,
                        raw_problem=raw_problem,
                        competition=competition,
                        deadline=deadline,
                    ),
                    self._state_record(state, revision=0, is_current=True),
                ]
            )
        return state

    def load_current(self, project_id: UUID) -> ProblemState:
        with session_scope(self._session_factory) as session:
            record = self._current_record(session, project_id)
            return ProblemState.model_validate(record.state_json)

    def list_current_projects(self) -> list[ProjectSummary]:
        with session_scope(self._session_factory) as session:
            rows = session.execute(
                select(Project.name, ProblemStateRecord.state_json)
                .join(Problem, Problem.project_id == Project.id)
                .join(ProblemStateRecord, ProblemStateRecord.problem_id == Problem.id)
                .where(ProblemStateRecord.is_current.is_(True))
                .order_by(Project.created_at.desc(), Project.id)
            )
            summaries: list[ProjectSummary] = []
            for name, state_json in rows:
                state = ProblemState.model_validate(state_json)
                summaries.append(
                    ProjectSummary(
                        project_id=state.project_id,
                        name=name,
                        title=state.title,
                        current_stage=state.current_stage,
                        status=state.status,
                        version=state.version,
                        created_at=state.created_at,
                        updated_at=state.updated_at,
                    )
                )
            return summaries

    def save_revision[OutputT: BaseModel](
        self,
        *,
        project_id: UUID,
        state: ProblemState,
        run: AgentRunResult[OutputT],
        decision: ModelDecisionEvidence | None = None,
    ) -> AgentRunResult[OutputT]:
        with session_scope(self._session_factory) as session:
            current = self._current_record(session, project_id, for_update=True)
            if current.revision + 1 != state.version:
                raise ValueError(
                    f"stale state update: current={current.revision}, requested={state.version}"
                )
            current.is_current = False
            session.flush()
            session.add(self._state_record(state, revision=state.version, is_current=True))
            stored_run = run.model_copy(update={"output_state_version": state.version})
            session.add(self._agent_record(project_id, state.problem_id, stored_run))
            if decision is not None:
                session.add(
                    ModelDecisionRecord(
                        project_id=project_id,
                        problem_id=state.problem_id,
                        state_version=state.version,
                        agent_run_id=stored_run.run_id,
                        selected_model_id=decision.selected_model_id,
                        backup_model_id=decision.backup_model_id,
                        weights_version=decision.weights.version,
                        decision_json=decision.model_dump(mode="json"),
                    )
                )
        return stored_run

    def record_run[OutputT: BaseModel](
        self, project_id: UUID, problem_id: UUID, run: AgentRunResult[OutputT]
    ) -> None:
        with session_scope(self._session_factory) as session:
            session.add(self._agent_record(project_id, problem_id, run))

    def list_state_revisions(self, project_id: UUID) -> list[ProblemState]:
        with session_scope(self._session_factory) as session:
            records = list(
                session.scalars(
                    select(ProblemStateRecord)
                    .join(Problem, Problem.id == ProblemStateRecord.problem_id)
                    .where(Problem.project_id == project_id)
                    .order_by(ProblemStateRecord.revision)
                )
            )
            if not records:
                raise ResourceNotFoundError(f"project {project_id} was not found")
            return [ProblemState.model_validate(record.state_json) for record in records]

    def list_agent_runs(self, project_id: UUID) -> list[AgentRunRecord]:
        with session_scope(self._session_factory) as session:
            return list(
                session.scalars(
                    select(AgentRunRecord)
                    .where(AgentRunRecord.project_id == project_id)
                    .order_by(AgentRunRecord.started_at)
                )
            )

    @staticmethod
    def _current_record(
        session: Session, project_id: UUID, *, for_update: bool = False
    ) -> ProblemStateRecord:
        statement = (
            select(ProblemStateRecord)
            .join(Problem, Problem.id == ProblemStateRecord.problem_id)
            .where(Problem.project_id == project_id, ProblemStateRecord.is_current.is_(True))
        )
        if for_update:
            statement = statement.with_for_update()
        record = session.scalar(statement)
        if record is None:
            raise ResourceNotFoundError(f"project {project_id} was not found")
        return record

    @staticmethod
    def _state_record(
        state: ProblemState, *, revision: int, is_current: bool
    ) -> ProblemStateRecord:
        return ProblemStateRecord(
            problem_id=state.problem_id,
            revision=revision,
            schema_version=state.schema_version,
            current_stage=state.current_stage.value,
            status=state.status.value,
            is_current=is_current,
            state_json=state.model_dump(mode="json"),
            updated_by=state.updated_by,
            update_reason=state.update_reason,
        )

    @staticmethod
    def _agent_record[OutputT: BaseModel](
        project_id: UUID,
        problem_id: UUID,
        run: AgentRunResult[OutputT],
    ) -> AgentRunRecord:
        return AgentRunRecord(
            id=run.run_id,
            project_id=project_id,
            problem_id=problem_id,
            agent_name=run.agent_name,
            input_state_version=run.input_state_version,
            output_state_version=run.output_state_version,
            provider=run.provider,
            model=run.model,
            provider_id=run.provider_id,
            provider_config_digest=run.provider_config_digest,
            model_id=run.model_id,
            remote_model=run.remote_model,
            model_config_digest=run.model_config_digest,
            protocol=run.protocol,
            structured_output_mode=run.structured_output_mode,
            reasoning_requested=run.reasoning_requested,
            reasoning_effective=run.reasoning_effective,
            endpoint_trust=run.endpoint_trust,
            reasoning_level=run.reasoning,
            prompt_version=run.prompt_version,
            started_at=run.started_at,
            ended_at=run.ended_at,
            latency_ms=run.latency_ms,
            token_usage=run.token_usage.model_dump(mode="json"),
            status=run.status.value,
            retry_count=max(run.attempts - 1, 0),
            error="\n".join(run.errors) or None,
            is_mock=run.is_mock,
            route_json=[route.model_dump(mode="json") for route in run.routes],
        )
