from uuid import uuid4

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from mathmodel_ai.db.base import Base
from mathmodel_ai.db.models import (
    MathematicalModelRecord,
    Problem,
    ProblemStateRecord,
    Project,
)
from mathmodel_ai.db.session import create_session_factory, session_scope
from mathmodel_ai.mathematical.digests import mathematical_model_digest
from mathmodel_ai.mathematical.repository import MathematicalRepository
from mathmodel_ai.schemas.problem_state import ProblemState
from tests.mathematical.helpers import lp_model


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


def test_mathematical_model_identity_versions_without_overwriting() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = create_session_factory(engine)
    project_id = uuid4()
    problem_id = uuid4()
    model_id = uuid4()
    model = lp_model(
        project_id=project_id,
        problem_id=problem_id,
        model_id=model_id,
    )
    with session_scope(factory) as session:
        session.add(Project(id=project_id, name="Versioned model project"))
        session.add(
            Problem(
                id=problem_id,
                project_id=project_id,
                title="Versioned LP",
                raw_problem="Retain every accepted mathematical model version.",
            )
        )
        session.add(
            MathematicalModelRecord(
                id=uuid4(),
                project_id=project_id,
                problem_id=problem_id,
                model_id=model_id,
                version=1,
                model_digest=mathematical_model_digest(model),
                state_version=4,
                selected_model_id=model.source_selected_model_id,
                status=model.status.value,
                model_json=model.model_dump(mode="json"),
            )
        )

    assigned_model_id, assigned_version = MathematicalRepository(factory).next_model_identity(
        project_id
    )

    assert assigned_model_id == model_id
    assert assigned_version == 2
    with Session(engine) as session:
        stored = list(session.scalars(select(MathematicalModelRecord)))
        assert len(stored) == 1
        assert stored[0].version == 1
    engine.dispose()
