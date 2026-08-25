from mathmodel_ai.core.errors import InvalidStateTransitionError
from mathmodel_ai.schemas.problem_state import WorkflowStage

_PHASE_TWO_TRANSITIONS: dict[WorkflowStage, frozenset[WorkflowStage]] = {
    WorkflowStage.INGEST: frozenset({WorkflowStage.UNDERSTAND}),
    WorkflowStage.UNDERSTAND: frozenset({WorkflowStage.EXPLORE}),
    WorkflowStage.EXPLORE: frozenset({WorkflowStage.SELECT}),
}


def ensure_transition(current: WorkflowStage, target: WorkflowStage) -> None:
    if target not in _PHASE_TWO_TRANSITIONS.get(current, frozenset()):
        raise InvalidStateTransitionError(
            f"invalid Phase 2 transition: {current.value} -> {target.value}"
        )
