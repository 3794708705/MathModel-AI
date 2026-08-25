from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from mathmodel_ai.schemas.problem_state import (
    EvidenceKind,
    ProblemState,
    TraceableItem,
)

REQUIRED_FIELDS = {
    "problem_id",
    "project_id",
    "title",
    "raw_problem",
    "competition",
    "deadline",
    "remaining_hours",
    "files",
    "background",
    "objectives",
    "subproblems",
    "facts",
    "data_sources",
    "assumptions",
    "ambiguities",
    "constraints",
    "candidate_models",
    "selected_model",
    "backup_model",
    "variables",
    "parameters",
    "units",
    "equations",
    "objective",
    "model_constraints",
    "algorithm",
    "code_files",
    "execution_records",
    "results",
    "validation_results",
    "sensitivity_results",
    "robustness_results",
    "literature",
    "citations",
    "red_team_reports",
    "revisions",
    "figures",
    "tables",
    "paper_state",
    "submission_state",
    "current_stage",
    "status",
}


@pytest.mark.schema
def test_problem_state_contains_contract_fields_and_round_trips() -> None:
    assert REQUIRED_FIELDS <= ProblemState.model_fields.keys()
    state = ProblemState(
        project_id=uuid4(),
        title="Routing problem",
        raw_problem="Minimize total travel distance.",
        deadline=datetime(2026, 8, 26, tzinfo=UTC),
        facts=[
            TraceableItem(
                item_id="fact-1",
                kind=EvidenceKind.FACT,
                statement="There are ten destinations.",
            )
        ],
    )
    assert ProblemState.model_validate_json(state.model_dump_json()) == state


@pytest.mark.schema
def test_problem_state_rejects_assumption_in_fact_collection() -> None:
    with pytest.raises(ValidationError, match="facts entries must have kind=FACT"):
        ProblemState(
            project_id=uuid4(),
            title="Invalid state",
            raw_problem="Text",
            facts=[
                TraceableItem(
                    item_id="assumption-1",
                    kind=EvidenceKind.ASSUMPTION,
                    statement="Demand is stationary.",
                )
            ],
        )


@pytest.mark.schema
def test_problem_state_rejects_naive_deadline() -> None:
    with pytest.raises(ValidationError, match="timezone"):
        ProblemState(
            project_id=uuid4(),
            title="Invalid deadline",
            raw_problem="Text",
            deadline=datetime(2026, 8, 26),
        )
