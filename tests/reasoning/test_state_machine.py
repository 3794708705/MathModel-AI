import pytest

from mathmodel_ai.core.errors import InvalidStateTransitionError
from mathmodel_ai.reasoning.state_machine import ensure_transition
from mathmodel_ai.schemas.problem_state import WorkflowStage


def test_phase_two_transitions_are_ordered() -> None:
    ensure_transition(WorkflowStage.INGEST, WorkflowStage.UNDERSTAND)
    ensure_transition(WorkflowStage.UNDERSTAND, WorkflowStage.EXPLORE)
    ensure_transition(WorkflowStage.EXPLORE, WorkflowStage.SELECT)
    ensure_transition(WorkflowStage.SELECT, WorkflowStage.MODEL)
    ensure_transition(WorkflowStage.MODEL, WorkflowStage.SOLVE)


def test_ingest_cannot_skip_directly_to_select() -> None:
    with pytest.raises(InvalidStateTransitionError, match="INGEST -> SELECT"):
        ensure_transition(WorkflowStage.INGEST, WorkflowStage.SELECT)
