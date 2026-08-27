from mathmodel_ai.core.errors import InvalidStateTransitionError
from mathmodel_ai.schemas.problem_state import WorkflowStage

_WORKFLOW_TRANSITIONS: dict[WorkflowStage, frozenset[WorkflowStage]] = {
    WorkflowStage.INGEST: frozenset({WorkflowStage.UNDERSTAND}),
    WorkflowStage.UNDERSTAND: frozenset({WorkflowStage.EXPLORE}),
    WorkflowStage.EXPLORE: frozenset({WorkflowStage.SELECT}),
    WorkflowStage.SELECT: frozenset({WorkflowStage.MODEL}),
    WorkflowStage.MODEL: frozenset({WorkflowStage.SOLVE}),
}


def ensure_transition(current: WorkflowStage, target: WorkflowStage) -> None:
    if target not in _WORKFLOW_TRANSITIONS.get(current, frozenset()):
        raise InvalidStateTransitionError(
            f"invalid workflow transition: {current.value} -> {target.value}"
        )
