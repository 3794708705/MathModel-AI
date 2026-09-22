from collections.abc import AsyncIterator, Callable

from pydantic import BaseModel

from mathmodel_ai.core.errors import ProviderError
from mathmodel_ai.core.types import ReasoningEffort
from mathmodel_ai.providers.base import BaseModelProvider
from mathmodel_ai.providers.schemas import (
    GenerationRequest,
    ModelResponse,
    ModelUsage,
    StructuredModelResponse,
)
from mathmodel_ai.schemas.provider_registry import (
    ModelCapability,
    ModelProfile,
    ProviderEndpoint,
    ToolCallingStrategy,
)


class BoundModelProvider(BaseModelProvider):
    """Bind a native protocol adapter to stable registry identity and trace metadata."""

    def __init__(
        self,
        provider: BaseModelProvider,
        *,
        endpoint: ProviderEndpoint,
        model_profile: ModelProfile,
    ) -> None:
        self._provider = provider
        self.endpoint = endpoint
        self.model_profile = model_profile
        self.name = endpoint.provider_id

    def _request(self, request: GenerationRequest) -> GenerationRequest:
        updates: dict[str, object] = {"model": self.model_profile.remote_model}
        if not self.model_profile.supports(ModelCapability.REASONING_CONTROL, allow_partial=True):
            updates["reasoning_effort"] = None
        elif request.reasoning_effort is not None:
            mapped = self.model_profile.reasoning_mapping.get(
                request.reasoning_effort.value.upper()
            ) or self.model_profile.reasoning_mapping.get(request.reasoning_effort.value)
            if mapped is None:
                updates["reasoning_effort"] = None
            else:
                try:
                    updates["reasoning_effort"] = ReasoningEffort(mapped.casefold())
                except ValueError as exc:
                    raise ProviderError(
                        "native provider reasoning mapping is not a normalized effort"
                    ) from exc
        if request.tools and (
            self.model_profile.tool_calling_strategy is not ToolCallingStrategy.NATIVE_TOOLS
            or not self.model_profile.supports(ModelCapability.TOOLS, allow_partial=True)
        ):
            raise ProviderError("configured model does not support native tools")
        return request.model_copy(update=updates)

    def _response(
        self,
        response: ModelResponse,
        *,
        request: GenerationRequest,
    ) -> ModelResponse:
        return response.model_copy(
            update={
                "provider": self.endpoint.provider_id,
                "provider_config_digest": self.endpoint.config_digest,
                "model_id": self.model_profile.model_id,
                "model_config_digest": self.model_profile.config_digest,
                "protocol": self.endpoint.protocol.value,
                "endpoint_trust": self.endpoint.trust_level.value,
                "reasoning_effective": (
                    request.reasoning_effort.value if request.reasoning_effort is not None else None
                ),
            }
        )

    async def generate(self, request: GenerationRequest) -> ModelResponse:
        prepared = self._request(request)
        return self._response(await self._provider.generate(prepared), request=prepared)

    async def structured_generate[T: BaseModel](
        self, request: GenerationRequest, response_model: type[T]
    ) -> StructuredModelResponse[T]:
        prepared = self._request(request)
        result = await self._provider.structured_generate(prepared, response_model)
        response = self._response(result.response, request=prepared).model_copy(
            update={"structured_output_mode": self.model_profile.structured_output_strategy.value}
        )
        return StructuredModelResponse(parsed=result.parsed, response=response)

    def stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        return self._provider.stream(self._request(request))

    def get_usage(self) -> ModelUsage:
        return self._provider.get_usage()

    async def aclose(self) -> None:
        await self._provider.aclose()


class ExecutionGuardedProvider(BaseModelProvider):
    """Revalidate an exact routing snapshot immediately before every network call."""

    def __init__(self, provider: BaseModelProvider, guard: Callable[[], None]) -> None:
        self._provider = provider
        self._guard = guard
        self.name = provider.name

    async def generate(self, request: GenerationRequest) -> ModelResponse:
        self._guard()
        return await self._provider.generate(request)

    async def structured_generate[T: BaseModel](
        self, request: GenerationRequest, response_model: type[T]
    ) -> StructuredModelResponse[T]:
        self._guard()
        return await self._provider.structured_generate(request, response_model)

    def stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        async def iterator() -> AsyncIterator[str]:
            self._guard()
            async for chunk in self._provider.stream(request):
                yield chunk

        return iterator()

    def get_usage(self) -> ModelUsage:
        return self._provider.get_usage()

    async def aclose(self) -> None:
        await self._provider.aclose()
