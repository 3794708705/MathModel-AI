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
        "files",
        "artifacts",
        "datasets",
        "data_profiles",
        "execution_records",
        "mathematical_models",
        "generated_programs",
        "solver_runs",
        "results",
        "validation_runs",
        "sensitivity_runs",
        "robustness_runs",
        "verification_experiments",
        "red_team_reports",
        "repair_cycles",
        "paper_versions",
        "evidence_records",
        "claims",
        "claim_evidence_links",
        "literature_searches",
        "literature_references",
        "citation_support_checks",
        "paper_sections",
        "figures",
        "tables",
        "document_registry",
        "paper_artifacts",
        "competition_profiles",
        "competition_rules",
        "requirement_coverage",
        "final_jury_reports",
        "jury_findings",
        "submission_checks",
        "submission_snapshots",
        "submission_artifacts",
        "submission_manifests",
        "correction_plans",
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
    assert {"model_id", "version", "model_json"} <= {
        column["name"] for column in inspector.get_columns("mathematical_models")
    }
    assert {"model_version", "execution_record_id", "result_json"} <= {
        column["name"] for column in inspector.get_columns("solver_runs")
    }
    result_foreign_keys = {
        tuple(item["constrained_columns"]) for item in inspector.get_foreign_keys("results")
    }
    assert {
        ("mathematical_model_record_id",),
        ("solver_run_id",),
        ("execution_record_id",),
    } <= result_foreign_keys
    validation_foreign_keys = {
        tuple(item["constrained_columns"]) for item in inspector.get_foreign_keys("validation_runs")
    }
    assert {
        ("mathematical_model_record_id",),
        ("result_id",),
        ("solver_run_id",),
        ("execution_record_id",),
    } <= validation_foreign_keys
    repair_foreign_keys = {
        tuple(item["constrained_columns"]) for item in inspector.get_foreign_keys("repair_cycles")
    }
    assert {
        ("source_model_record_id",),
        ("target_model_record_id",),
        ("red_team_report_id",),
        ("agent_run_id",),
    } <= repair_foreign_keys
    paper_foreign_keys = {
        tuple(item["constrained_columns"]) for item in inspector.get_foreign_keys("paper_versions")
    }
    assert {
        ("project_id",),
        ("problem_id",),
        ("verified_result_id",),
    } <= paper_foreign_keys
    evidence_foreign_keys = {
        tuple(item["constrained_columns"])
        for item in inspector.get_foreign_keys("evidence_records")
    }
    assert {
        ("project_id",),
        ("problem_id",),
        ("paper_version_record_id",),
    } <= evidence_foreign_keys
    claim_link_foreign_keys = {
        tuple(item["constrained_columns"])
        for item in inspector.get_foreign_keys("claim_evidence_links")
    }
    assert {
        ("project_id",),
        ("claim_record_id",),
        ("evidence_record_id",),
    } <= claim_link_foreign_keys
    snapshot_foreign_keys = {
        tuple(item["constrained_columns"])
        for item in inspector.get_foreign_keys("submission_snapshots")
    }
    assert {
        ("project_id",),
        ("paper_version_record_id",),
        ("verified_result_id",),
        ("profile_record_id",),
        ("submission_check_id",),
    } <= snapshot_foreign_keys
    artifact_foreign_keys = {
        tuple(item["constrained_columns"])
        for item in inspector.get_foreign_keys("submission_artifacts")
    }
    assert {("project_id",), ("submission_id",)} <= artifact_foreign_keys

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
