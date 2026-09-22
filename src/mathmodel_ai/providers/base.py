from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from pydantic import BaseModel

from mathmodel_ai.providers.schemas import (
    GenerationRequest,
    ModelResponse,
    ModelUsage,
    StructuredModelResponse,
)


class BaseModelProvider(ABC):
    name: str

    @abstractmethod
    async def generate(self, request: GenerationRequest) -> ModelResponse:
        raise NotImplementedError

    @abstractmethod
    async def structured_generate[T: BaseModel](
        self, request: GenerationRequest, response_model: type[T]
    ) -> StructuredModelResponse[T]:
        raise NotImplementedError

    @abstractmethod
    def stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        raise NotImplementedError

    @abstractmethod
    def get_usage(self) -> ModelUsage:
        raise NotImplementedError

    async def aclose(self) -> None:
        """Release owned network resources; providers without resources need no action."""
        return None
