from mathmodel_ai.core.errors import InvalidStateTransitionError
from mathmodel_ai.schemas.problem_state import WorkflowStage

_WORKFLOW_TRANSITIONS: dict[WorkflowStage, frozenset[WorkflowStage]] = {
    WorkflowStage.INGEST: frozenset({WorkflowStage.UNDERSTAND}),
    WorkflowStage.UNDERSTAND: frozenset({WorkflowStage.EXPLORE}),
    WorkflowStage.EXPLORE: frozenset({WorkflowStage.SELECT}),
    WorkflowStage.SELECT: frozenset({WorkflowStage.MODEL}),
    WorkflowStage.MODEL: frozenset({WorkflowStage.SOLVE}),
    WorkflowStage.SOLVE: frozenset({WorkflowStage.VALIDATE}),
    WorkflowStage.VALIDATE: frozenset({WorkflowStage.SENSITIVITY}),
    WorkflowStage.SENSITIVITY: frozenset({WorkflowStage.ROBUSTNESS}),
    WorkflowStage.ROBUSTNESS: frozenset({WorkflowStage.RED_TEAM}),
    WorkflowStage.RED_TEAM: frozenset({WorkflowStage.MODEL_REPAIR, WorkflowStage.PAPER}),
    WorkflowStage.MODEL_REPAIR: frozenset({WorkflowStage.SOLVE}),
    WorkflowStage.PAPER: frozenset({WorkflowStage.PAPER, WorkflowStage.FINAL_JURY}),
    WorkflowStage.FINAL_JURY: frozenset(
        {WorkflowStage.FINAL_JURY, WorkflowStage.SUBMISSION, WorkflowStage.PAPER}
    ),
    WorkflowStage.SUBMISSION: frozenset(
        {WorkflowStage.SUBMISSION, WorkflowStage.FINAL, WorkflowStage.PAPER}
    ),
    WorkflowStage.FINAL: frozenset(
        {WorkflowStage.FINAL, WorkflowStage.SUBMISSION, WorkflowStage.PAPER}
    ),
}


def ensure_transition(current: WorkflowStage, target: WorkflowStage) -> None:
    if target not in _WORKFLOW_TRANSITIONS.get(current, frozenset()):
        raise InvalidStateTransitionError(
            f"invalid workflow transition: {current.value} -> {target.value}"
        )
