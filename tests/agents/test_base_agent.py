from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest
from pydantic import BaseModel, ValidationError

from mathmodel_ai.agents import AgentRunResult, AgentRunStatus, BaseAgent
from mathmodel_ai.core.config import Settings
from mathmodel_ai.core.types import ProviderName
from mathmodel_ai.providers.base import BaseModelProvider
from mathmodel_ai.providers.factory import ProviderRegistry
from mathmodel_ai.providers.mock import MockProvider
from mathmodel_ai.providers.schemas import GenerationRequest, ModelMessage
from mathmodel_ai.routing import (
    EscalationLevel,
    ModelRouter,
    RouteDecision,
    TaskProfile,
    TaskType,
)
from mathmodel_ai.schemas.problem_state import ProblemState


class AgentInput(BaseModel):
    prompt: str


class AgentOutput(BaseModel):
    value: int


class ExampleAgent(BaseAgent[AgentInput, AgentOutput]):
    name = "example"
    role = "test contract"
    capabilities = frozenset({"structured_generation"})
    input_schema = AgentInput
    output_schema = AgentOutput

    async def execute(
        self,
        input_data: AgentInput,
        state: ProblemState,
        provider: BaseModelProvider,
        route: RouteDecision,
    ) -> AgentOutput | dict[str, Any]:
        request = GenerationRequest(
            model=route.selected_model or "missing",
            messages=[ModelMessage(role="user", content=input_data.prompt)],
        )
        return (await provider.structured_generate(request, AgentOutput)).parsed


def state() -> ProblemState:
    return ProblemState(project_id=uuid4(), title="Test", raw_problem="Test problem")


def test_agent_run_reasoning_level_respects_persistence_contract() -> None:
    now = datetime.now(UTC)

    with pytest.raises(ValidationError, match="String should have at most 32 characters"):
        AgentRunResult[AgentOutput](
            agent_name="deterministic_test",
            status=AgentRunStatus.SUCCEEDED,
            attempts=0,
            input_state_version=0,
            reasoning="x" * 33,
            latency_ms=0,
            started_at=now,
            ended_at=now,
        )


@pytest.mark.asyncio
async def test_agent_retries_validation_failure_and_marks_mock() -> None:
    mock = MockProvider(['{"value":"invalid"}', '{"value":2}'])
    providers = ProviderRegistry([mock])
    router = ModelRouter(Settings(), available_providers={ProviderName.MOCK})
    agent = ExampleAgent(router=router, providers=providers, max_retries=2)
    result = await agent.run(
        AgentInput(prompt="answer"),
        state(),
        TaskProfile(task_type=TaskType.DOCUMENTATION),
    )
    assert result.status is AgentRunStatus.SUCCEEDED
    assert result.attempts == 2
    assert result.output == AgentOutput(value=2)
    assert result.is_mock is True
    assert len(result.errors) == 1
    assert [item.level for item in result.routes] == [
        EscalationLevel.FAST,
        EscalationLevel.FAST,
    ]
    assert result.token_usage.requests == 2


@pytest.mark.asyncio
async def test_agent_reports_failure_without_fabricating_output() -> None:
    mock = MockProvider(["invalid-json"])
    agent = ExampleAgent(
        router=ModelRouter(Settings(), available_providers={ProviderName.MOCK}),
        providers=ProviderRegistry([mock]),
        max_retries=0,
    )
    result = await agent.run(
        AgentInput(prompt="answer"),
        state(),
        TaskProfile(task_type=TaskType.DOCUMENTATION),
    )
    assert result.status is AgentRunStatus.ESCALATED
    assert result.output is None
    assert result.attempts == 1


@pytest.mark.asyncio
async def test_agent_escalates_only_after_one_same_tier_retry() -> None:
    mock = MockProvider(["invalid-json", "invalid-json", "invalid-json"])
    agent = ExampleAgent(
        router=ModelRouter(Settings(), available_providers={ProviderName.MOCK}),
        providers=ProviderRegistry([mock]),
        max_retries=2,
    )

    result = await agent.run(
        AgentInput(prompt="answer"),
        state(),
        TaskProfile(task_type=TaskType.DOCUMENTATION),
    )

    assert result.status is AgentRunStatus.ESCALATED
    assert result.attempts == 3
    assert [item.level for item in result.routes] == [
        EscalationLevel.FAST,
        EscalationLevel.FAST,
        EscalationLevel.BALANCED,
    ]
    assert result.token_usage.requests == 3
