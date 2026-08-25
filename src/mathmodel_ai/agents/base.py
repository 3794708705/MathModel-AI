import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from mathmodel_ai.providers.base import BaseModelProvider
from mathmodel_ai.providers.factory import ProviderRegistry
from mathmodel_ai.providers.schemas import ModelResponse, ModelUsage
from mathmodel_ai.routing.router import ModelRouter
from mathmodel_ai.routing.schemas import RouteAction, RouteDecision, TaskProfile
from mathmodel_ai.schemas.problem_state import ProblemState


class AgentRunStatus(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    RETRY = "RETRY"
    ESCALATED = "ESCALATED"
    HUMAN_REVIEW = "HUMAN_REVIEW"


class AgentRunResult[OutputT: BaseModel](BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    run_id: UUID = Field(default_factory=uuid4)
    agent_name: str
    status: AgentRunStatus
    output: OutputT | None = None
    attempts: int = Field(ge=0)
    routes: list[RouteDecision] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    is_mock: bool = False
    input_state_version: int = Field(ge=0)
    output_state_version: int | None = Field(default=None, ge=1)
    provider: str | None = None
    model: str | None = None
    reasoning: str | None = None
    prompt_version: str | None = None
    token_usage: ModelUsage = Field(default_factory=ModelUsage)
    latency_ms: int = Field(ge=0)
    started_at: datetime
    ended_at: datetime


@dataclass(frozen=True)
class AgentExecution[OutputT: BaseModel]:
    output: OutputT | dict[str, Any]
    response: ModelResponse
    prompt_version: str


class BaseAgent[InputT: BaseModel, OutputT: BaseModel](ABC):
    name: str
    role: str
    capabilities: frozenset[str]
    input_schema: type[InputT]
    output_schema: type[OutputT]

    def __init__(
        self,
        *,
        router: ModelRouter,
        providers: ProviderRegistry,
        max_retries: int = 2,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self._router = router
        self._providers = providers
        self._max_retries = max_retries
        self._logger = logging.getLogger(f"mathmodel_ai.agent.{self.name}")

    @abstractmethod
    async def execute(
        self,
        input_data: InputT,
        state: ProblemState,
        provider: BaseModelProvider,
        route: RouteDecision,
    ) -> OutputT | dict[str, Any] | AgentExecution[OutputT]:
        raise NotImplementedError

    def validate_output(self, output: OutputT | dict[str, Any]) -> OutputT:
        return self.output_schema.model_validate(output)

    async def retry(
        self,
        input_data: InputT,
        state: ProblemState,
        profile: TaskProfile,
    ) -> AgentRunResult[OutputT]:
        return await self.run(input_data, state, profile)

    async def run(
        self,
        input_data: InputT,
        state: ProblemState,
        profile: TaskProfile,
    ) -> AgentRunResult[OutputT]:
        started_at = datetime.now(UTC)
        run_id = uuid4()
        errors: list[str] = []
        routes: list[RouteDecision] = []

        validated_input = self.input_schema.model_validate(input_data)
        for attempt in range(1, self._max_retries + 2):
            retry_profile = profile.model_copy(
                update={"retry_count": profile.retry_count + attempt - 1}
            )
            route = self._router.route(retry_profile)
            routes.append(route)
            if route.action in {RouteAction.MULTI_MODEL_REVIEW, RouteAction.HUMAN_REVIEW}:
                status = (
                    AgentRunStatus.HUMAN_REVIEW
                    if route.action is RouteAction.HUMAN_REVIEW
                    else AgentRunStatus.ESCALATED
                )
                return AgentRunResult(
                    run_id=run_id,
                    agent_name=self.name,
                    status=status,
                    attempts=attempt - 1,
                    routes=routes,
                    errors=errors,
                    input_state_version=state.version,
                    latency_ms=int((datetime.now(UTC) - started_at).total_seconds() * 1000),
                    started_at=started_at,
                    ended_at=datetime.now(UTC),
                )
            if route.selected_provider is None:
                errors.append("route did not select a provider")
                continue

            try:
                provider = self._providers.get(route.selected_provider)
                execution = await self.execute(validated_input, state, provider, route)
                if isinstance(execution, AgentExecution):
                    raw_output = execution.output
                    response = execution.response
                    prompt_version = execution.prompt_version
                else:
                    raw_output = execution
                    response = None
                    prompt_version = None
                output = self.validate_output(raw_output)
            except Exception as exc:
                safe_error = f"{type(exc).__name__}: {exc}"
                errors.append(safe_error)
                self._logger.warning(
                    "agent attempt failed",
                    extra={"run_id": str(run_id), "agent": self.name, "attempt": attempt},
                )
                continue

            return AgentRunResult(
                run_id=run_id,
                agent_name=self.name,
                status=AgentRunStatus.SUCCEEDED,
                output=output,
                attempts=attempt,
                routes=routes,
                errors=errors,
                is_mock=route.selected_provider.value == "mock",
                input_state_version=state.version,
                provider=route.selected_provider.value,
                model=route.selected_model,
                reasoning=(
                    route.selected_reasoning.value if route.selected_reasoning is not None else None
                ),
                prompt_version=prompt_version,
                token_usage=response.usage if response is not None else ModelUsage(),
                latency_ms=int((datetime.now(UTC) - started_at).total_seconds() * 1000),
                started_at=started_at,
                ended_at=datetime.now(UTC),
            )

        return AgentRunResult(
            run_id=run_id,
            agent_name=self.name,
            status=AgentRunStatus.ESCALATED,
            attempts=self._max_retries + 1,
            routes=routes,
            errors=errors,
            input_state_version=state.version,
            latency_ms=int((datetime.now(UTC) - started_at).total_seconds() * 1000),
            started_at=started_at,
            ended_at=datetime.now(UTC),
        )
