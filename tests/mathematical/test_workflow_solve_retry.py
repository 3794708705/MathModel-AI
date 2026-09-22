from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from mathmodel_ai.core.errors import QualityGateError
from mathmodel_ai.mathematical.workflow import MathematicalWorkflow
from mathmodel_ai.schemas.program import ExecutionStrategy, ExecutionStrategyDecision
from mathmodel_ai.schemas.quality import QualityGateResult, QualityGateStatus


def _solve_outcome(status: QualityGateStatus, *, message: str) -> SimpleNamespace:
    return SimpleNamespace(
        gate=QualityGateResult(
            gate="SOLVE",
            status=status,
            checks={"canonical_result": status is QualityGateStatus.PASS},
            errors=(
                [] if status is QualityGateStatus.PASS else ["SOLVE_GATE_FAIL:EXECUTION_ERROR"]
            ),
        ),
        strategy=ExecutionStrategyDecision(
            requested=ExecutionStrategy.AUTO,
            selected=ExecutionStrategy.GENERATED,
            reason="fixture generated execution",
        ),
        execution=SimpleNamespace(result=SimpleNamespace(message=message)),
    )


@pytest.mark.asyncio
async def test_run_retries_generated_solve_with_exact_persisted_failure_feedback() -> None:
    workflow = object.__new__(MathematicalWorkflow)
    workflow._max_generated_solve_attempts = 2
    workflow.build_model = AsyncMock(return_value=SimpleNamespace())
    first = _solve_outcome(
        QualityGateStatus.RETRY,
        message="generated result.json failed schema validation: status: invalid enum",
    )
    second = _solve_outcome(QualityGateStatus.PASS, message="accepted")
    workflow.solve = AsyncMock(side_effect=[first, second])

    outcome = await MathematicalWorkflow.run(
        workflow,
        uuid4(),
        user_guidance=["preserve the fixed model"],
    )

    assert outcome.solve_stage is second
    assert workflow.solve.await_count == 2
    retry_guidance = workflow.solve.await_args_list[1].kwargs["user_guidance"]
    assert retry_guidance[0] == "preserve the fixed model"
    assert any("AUTOMATED_SOLVE_RETRY_FEEDBACK" in item for item in retry_guidance)
    assert any("status: invalid enum" in item for item in retry_guidance)


@pytest.mark.asyncio
async def test_run_stops_before_verification_when_generated_solve_retries_exhausted() -> None:
    workflow = object.__new__(MathematicalWorkflow)
    workflow._max_generated_solve_attempts = 2
    workflow.build_model = AsyncMock(return_value=SimpleNamespace())
    failed = _solve_outcome(
        QualityGateStatus.RETRY,
        message="generated result.json failed schema validation: status: invalid enum",
    )
    workflow.solve = AsyncMock(side_effect=[failed, failed])

    with pytest.raises(
        QualityGateError,
        match=r"SOLVE quality gate rejected 2 persisted attempt\(s\)",
    ):
        await MathematicalWorkflow.run(workflow, uuid4())

    assert workflow.solve.await_count == 2
