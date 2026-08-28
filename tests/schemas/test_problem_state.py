from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from mathmodel_ai.schemas.problem_state import (
    EvidenceKind,
    ProblemState,
    TraceableItem,
    VerificationStatus,
)
from mathmodel_ai.schemas.quality import QualityGateResult, QualityGateStatus
from mathmodel_ai.schemas.results import ResultRecordRef
from mathmodel_ai.schemas.solver import SolverStatus

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
    "mathematical_model",
    "algorithm_plan",
    "code_files",
    "generated_programs",
    "execution_records",
    "solver_runs",
    "result_records",
    "verified_result_id",
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
    assert state.schema_version == 5
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


@pytest.mark.schema
def test_verified_result_pointer_requires_exact_formal_result_and_gate() -> None:
    result_id = uuid4()
    model_id = uuid4()
    trace = TraceableItem(
        item_id=f"RESULT-{result_id}",
        kind=EvidenceKind.RESULT,
        statement="accepted formal result",
        verification_status=VerificationStatus.VERIFIED,
    )
    result_ref = ResultRecordRef(
        result_id=result_id,
        model_id=model_id,
        model_version=1,
        model_digest="a" * 64,
        solver_run_id=uuid4(),
        status=SolverStatus.OPTIMAL,
    )
    later_result_id = uuid4()
    later_ref = result_ref.model_copy(update={"result_id": later_result_id})
    later_trace = TraceableItem(
        item_id=f"RESULT-{later_result_id}",
        kind=EvidenceKind.RESULT,
        statement="later but unverified formal result",
    )

    with pytest.raises(ValidationError, match="require verified_result_id"):
        ProblemState(
            project_id=uuid4(),
            title="missing pointer",
            raw_problem="fixture",
            result_records=[result_ref],
            results=[trace],
        )

    accepted = ProblemState(
        project_id=uuid4(),
        title="accepted pointer",
        raw_problem="fixture",
        result_records=[result_ref, later_ref],
        verified_result_id=result_id,
        results=[trace, later_trace],
        quality_gates=[
            QualityGateResult(
                gate="VERIFIED",
                status=QualityGateStatus.PASS,
                subject_ref=f"result:{result_id}",
                checks={"complete_phase5_chain": True},
            )
        ],
    )

    assert accepted.verified_result_id == result_id
