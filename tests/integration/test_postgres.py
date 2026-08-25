import os
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, inspect, select
from sqlalchemy.orm import Session

from mathmodel_ai.db.models import Problem, ProblemStateRecord, Project
from mathmodel_ai.schemas.problem_state import ProblemState

DATABASE_URL = os.getenv("MM_TEST_DATABASE_URL")
pytestmark = [
    pytest.mark.integration,
    pytest.mark.postgres,
    pytest.mark.skipif(DATABASE_URL is None, reason="MM_TEST_DATABASE_URL is not configured"),
]


def test_migrated_postgres_persists_problem_state_jsonb() -> None:
    assert DATABASE_URL is not None
    engine = create_engine(DATABASE_URL)
    inspector = inspect(engine)
    assert {
        "projects",
        "problems",
        "problem_states",
        "agent_runs",
        "model_decisions",
    } <= set(inspector.get_table_names())
    assert {
        "input_state_version",
        "output_state_version",
        "prompt_version",
        "token_usage",
        "is_mock",
    } <= {column["name"] for column in inspector.get_columns("agent_runs")}
    assert {"updated_by", "update_reason"} <= {
        column["name"] for column in inspector.get_columns("problem_states")
    }

    connection = engine.connect()
    transaction = connection.begin()
    try:
        with Session(bind=connection) as session:
            project = Project(id=uuid4(), name="PostgreSQL integration")
            problem = Problem(
                id=uuid4(),
                project_id=project.id,
                title="Traceability test",
                raw_problem="Verify persisted state.",
            )
            state = ProblemState(
                problem_id=problem.id,
                project_id=project.id,
                title=problem.title,
                raw_problem=problem.raw_problem,
            )
            session.add_all(
                [
                    project,
                    problem,
                    ProblemStateRecord(
                        problem_id=problem.id,
                        revision=1,
                        schema_version=state.schema_version,
                        current_stage=state.current_stage.value,
                        status=state.status.value,
                        state_json=state.model_dump(mode="json"),
                    ),
                ]
            )
            session.flush()
            stored = session.scalar(select(ProblemStateRecord))
            assert stored is not None
            assert ProblemState.model_validate(stored.state_json) == state
    finally:
        transaction.rollback()
        connection.close()
        engine.dispose()
