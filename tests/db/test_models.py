from uuid import uuid4

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from mathmodel_ai.db.base import Base
from mathmodel_ai.db.models import Problem, ProblemStateRecord, Project
from mathmodel_ai.db.session import create_session_factory, session_scope
from mathmodel_ai.schemas.problem_state import ProblemState


def test_phase_one_models_persist_validated_state() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    project = Project(id=uuid4(), name="Competition entry")
    problem = Problem(
        id=uuid4(),
        project_id=project.id,
        title="Network optimization",
        raw_problem="Find the minimum-cost network.",
    )
    state = ProblemState(
        problem_id=problem.id,
        project_id=project.id,
        title=problem.title,
        raw_problem=problem.raw_problem,
    )
    record = ProblemStateRecord(
        problem_id=problem.id,
        revision=1,
        schema_version=state.schema_version,
        current_stage=state.current_stage.value,
        status=state.status.value,
        state_json=state.model_dump(mode="json"),
    )
    with Session(engine) as session:
        session.add_all([project, problem, record])
        session.commit()
        stored = session.scalar(select(ProblemStateRecord))
        assert stored is not None
        restored = ProblemState.model_validate(stored.state_json)
        assert restored.problem_id == problem.id

        duplicate_current = ProblemStateRecord(
            problem_id=problem.id,
            revision=2,
            schema_version=1,
            current_stage="INGEST",
            status="PENDING",
            state_json=state.model_dump(mode="json"),
        )
        session.add(duplicate_current)
        with pytest.raises(IntegrityError):
            session.commit()
        session.rollback()
    engine.dispose()


def test_session_scope_commits_and_rolls_back() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = create_session_factory(engine)
    with session_scope(factory) as session:
        session.add(Project(id=uuid4(), name="Committed"))
    with Session(engine) as session:
        assert session.scalar(select(Project.name)) == "Committed"

    with pytest.raises(RuntimeError, match="rollback"):
        with session_scope(factory) as session:
            session.add(Project(id=uuid4(), name="Rolled back"))
            raise RuntimeError("rollback")
    with Session(engine) as session:
        names = list(session.scalars(select(Project.name)))
        assert names == ["Committed"]
    engine.dispose()
