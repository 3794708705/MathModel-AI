from collections import deque
from collections.abc import AsyncIterator, Iterable

from pydantic import BaseModel

from mathmodel_ai.core.types import ProviderName
from mathmodel_ai.providers.base import BaseModelProvider
from mathmodel_ai.providers.schemas import (
    GenerationRequest,
    ModelResponse,
    ModelUsage,
    StructuredModelResponse,
)


class MockProvider(BaseModelProvider):
    name = ProviderName.MOCK

    def __init__(self, responses: Iterable[str] = ()) -> None:
        self._responses = deque(responses)
        self._usage = ModelUsage()

    def queue(self, response: str) -> None:
        self._responses.append(response)

    async def generate(self, request: GenerationRequest) -> ModelResponse:
        if not self._responses:
            raise RuntimeError("MockProvider has no queued response")
        content = self._responses.popleft()
        usage = ModelUsage(requests=1)
        self._usage = self._usage.plus(usage)
        return ModelResponse(
            content=content,
            provider=self.name,
            model=request.model,
            usage=usage,
            is_mock=True,
        )

    async def structured_generate[T: BaseModel](
        self, request: GenerationRequest, response_model: type[T]
    ) -> StructuredModelResponse[T]:
        response = await self.generate(request)
        return StructuredModelResponse(
            parsed=response_model.model_validate_json(response.content), response=response
        )

    def stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        async def iterator() -> AsyncIterator[str]:
            response = await self.generate(request)
            yield response.content

        return iterator()

    def get_usage(self) -> ModelUsage:
        return self._usage.model_copy(deep=True)
