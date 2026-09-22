import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from mathmodel_ai.providers.base import BaseModelProvider
from mathmodel_ai.providers.factory import ProviderRegistry
from mathmodel_ai.providers.schemas import ModelResponse, ModelUsage
from mathmodel_ai.providers.security import safe_error
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
    remote_model: str | None = None
    remote_model_reported: str | None = None
    provider_id: str | None = None
    provider_config_digest: str | None = None
    model_id: str | None = None
    model_config_digest: str | None = None
    protocol: str | None = None
    structured_output_mode: str | None = None
    reasoning_requested: str | None = None
    reasoning_effective: str | None = None
    endpoint_trust: str | None = None
    reasoning: str | None = Field(default=None, max_length=32)
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

    def prepare_attempt_input(
        self,
        input_data: InputT,
        state: ProblemState,
        previous_errors: tuple[str, ...],
    ) -> InputT:
        """Return typed input for an attempt; subclasses may add safe retry feedback."""

        return input_data

    def validate_output_for_state(self, output: OutputT, state: ProblemState) -> OutputT:
        """Apply deterministic state-aware checks before an attempt is accepted."""

        return output

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
        executed_attempts = 0
        accumulated_usage = ModelUsage()
        last_execution_route: RouteDecision | None = None
        last_execution_was_mock = False

        validated_input = self.input_schema.model_validate(input_data)
        for attempt in range(1, self._max_retries + 2):
            retry_profile = profile.model_copy(
                # A malformed or schema-invalid response should receive one
                # same-tier retry before routing escalates to a stronger model.
                update={"retry_count": profile.retry_count + max(attempt - 2, 0)}
            )
            route = self._router.route(retry_profile, agent_name=self.name)
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
                    attempts=executed_attempts,
                    routes=routes,
                    errors=errors,
                    is_mock=last_execution_was_mock,
                    input_state_version=state.version,
                    token_usage=accumulated_usage,
                    latency_ms=int((datetime.now(UTC) - started_at).total_seconds() * 1000),
                    started_at=started_at,
                    ended_at=datetime.now(UTC),
                    **self._route_trace(last_execution_route),
                )
            if route.selected_provider is None:
                errors.append("route did not select a provider")
                continue

            provider: BaseModelProvider | None = None
            usage_before = ModelUsage()
            response: ModelResponse | None = None
            prompt_version: str | None = None
            try:
                if route.task_profile_digest != retry_profile.requirements_digest:
                    raise ValueError("route TaskProfile digest differs from execution requirements")
                attempt_input = self.input_schema.model_validate(
                    self.prepare_attempt_input(validated_input, state, tuple(errors))
                )
                provider = self._providers.get(
                    route.selected_provider,
                    model_id=route.selected_model_id,
                    expected_provider_config_digest=route.provider_config_digest,
                    expected_model_config_digest=route.model_config_digest,
                    expected_probe_id=route.capability_probe_id,
                    expected_probe_digest=route.capability_probe_digest,
                )
                usage_before = provider.get_usage()
                executed_attempts += 1
                last_execution_route = route
                last_execution_was_mock = route.selected_provider == "mock"
                execution = await self.execute(attempt_input, state, provider, route)
                if isinstance(execution, AgentExecution):
                    raw_output = execution.output
                    response = execution.response
                    prompt_version = execution.prompt_version
                else:
                    raw_output = execution
                    response = None
                    prompt_version = None
                output = self.validate_output(raw_output)
                output = self.validate_output_for_state(output, state)
                if response is not None:
                    self._validate_response_trace(route, response)
            except Exception as exc:
                if response is not None:
                    accumulated_usage = accumulated_usage.plus(response.usage)
                    last_execution_was_mock = response.is_mock
                elif provider is not None:
                    accumulated_usage = accumulated_usage.plus(
                        self._usage_delta(usage_before, provider.get_usage())
                    )
                errors.append(
                    "provider or agent output failed validation"
                    if isinstance(exc, ValidationError)
                    else safe_error(exc)
                )
                self._logger.warning(
                    "agent attempt failed",
                    extra={"run_id": str(run_id), "agent": self.name, "attempt": attempt},
                )
                continue

            if response is not None:
                accumulated_usage = accumulated_usage.plus(response.usage)
            elif provider is not None:
                accumulated_usage = accumulated_usage.plus(
                    self._usage_delta(usage_before, provider.get_usage())
                )
            return AgentRunResult(
                run_id=run_id,
                agent_name=self.name,
                status=AgentRunStatus.SUCCEEDED,
                output=output,
                attempts=executed_attempts,
                routes=routes,
                errors=errors,
                is_mock=(
                    response.is_mock if response is not None else route.selected_provider == "mock"
                ),
                input_state_version=state.version,
                provider=response.provider if response is not None else route.selected_provider,
                model=response.model if response is not None else route.selected_model,
                remote_model=route.selected_model,
                remote_model_reported=(response.model if response is not None else None),
                provider_id=route.selected_provider,
                provider_config_digest=(
                    response.provider_config_digest
                    if response is not None
                    else route.provider_config_digest
                ),
                model_id=response.model_id if response is not None else route.selected_model_id,
                model_config_digest=(
                    response.model_config_digest
                    if response is not None
                    else route.model_config_digest
                ),
                protocol=response.protocol if response is not None else route.protocol,
                structured_output_mode=(
                    response.structured_output_mode
                    if response is not None
                    else (
                        route.structured_output_mode.value
                        if route.structured_output_mode is not None
                        else None
                    )
                ),
                reasoning_requested=(
                    route.selected_reasoning.value if route.selected_reasoning is not None else None
                ),
                reasoning_effective=(
                    response.reasoning_effective
                    if response is not None
                    else route.reasoning_effective
                ),
                endpoint_trust=(
                    response.endpoint_trust
                    if response is not None
                    else (route.endpoint_trust.value if route.endpoint_trust is not None else None)
                ),
                reasoning=(
                    route.selected_reasoning.value if route.selected_reasoning is not None else None
                ),
                prompt_version=prompt_version,
                token_usage=accumulated_usage,
                latency_ms=int((datetime.now(UTC) - started_at).total_seconds() * 1000),
                started_at=started_at,
                ended_at=datetime.now(UTC),
            )

        return AgentRunResult(
            run_id=run_id,
            agent_name=self.name,
            status=AgentRunStatus.ESCALATED,
            attempts=executed_attempts,
            routes=routes,
            errors=errors,
            is_mock=last_execution_was_mock,
            input_state_version=state.version,
            token_usage=accumulated_usage,
            latency_ms=int((datetime.now(UTC) - started_at).total_seconds() * 1000),
            started_at=started_at,
            ended_at=datetime.now(UTC),
            **self._route_trace(last_execution_route),
        )

    @staticmethod
    def _route_trace(route: RouteDecision | None) -> dict[str, object]:
        if route is None:
            return {}
        return {
            "provider": route.selected_provider,
            "model": route.selected_model,
            "remote_model": route.selected_model,
            "provider_id": route.selected_provider,
            "provider_config_digest": route.provider_config_digest,
            "model_id": route.selected_model_id,
            "model_config_digest": route.model_config_digest,
            "protocol": route.protocol,
            "structured_output_mode": (
                route.structured_output_mode.value
                if route.structured_output_mode is not None
                else None
            ),
            "reasoning_requested": (
                route.selected_reasoning.value if route.selected_reasoning is not None else None
            ),
            "reasoning_effective": route.reasoning_effective,
            "endpoint_trust": (
                route.endpoint_trust.value if route.endpoint_trust is not None else None
            ),
            "reasoning": (
                route.selected_reasoning.value if route.selected_reasoning is not None else None
            ),
        }

    @staticmethod
    def _usage_delta(before: ModelUsage, after: ModelUsage) -> ModelUsage:
        def difference(previous: int | None, current: int | None) -> int | None:
            if current is None:
                return None
            return max(current - (previous or 0), 0)

        return ModelUsage(
            input_tokens=difference(before.input_tokens, after.input_tokens),
            output_tokens=difference(before.output_tokens, after.output_tokens),
            cached_input_tokens=difference(before.cached_input_tokens, after.cached_input_tokens),
            reasoning_tokens=difference(before.reasoning_tokens, after.reasoning_tokens),
            total_tokens=difference(before.total_tokens, after.total_tokens),
            requests=max(after.requests - before.requests, 0),
        )

    @staticmethod
    def _validate_response_trace(route: RouteDecision, response: ModelResponse) -> None:
        if route.selected_provider is not None and response.provider != route.selected_provider:
            raise ValueError("provider response identity differs from route decision")
        if route.selected_model_id is not None and response.model_id != route.selected_model_id:
            raise ValueError("provider response model_id differs from route decision")
        if route.protocol is not None and response.protocol != route.protocol:
            raise ValueError("provider response protocol differs from route decision")
        if (
            route.endpoint_trust is not None
            and response.endpoint_trust != route.endpoint_trust.value
        ):
            raise ValueError("provider response endpoint trust differs from route decision")
        if (
            route.structured_output_mode is not None
            and response.structured_output_mode != route.structured_output_mode.value
        ):
            raise ValueError("provider response structured mode differs from route decision")
        if response.is_mock and route.selected_provider != "mock":
            raise ValueError("real provider route returned a Mock response")
        if (
            route.provider_config_digest is not None
            and response.provider_config_digest != route.provider_config_digest
        ):
            raise ValueError("provider response config digest differs from route decision")
        if (
            route.model_config_digest is not None
            and response.model_config_digest != route.model_config_digest
        ):
            raise ValueError("provider response model digest differs from route decision")
