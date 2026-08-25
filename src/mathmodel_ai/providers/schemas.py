from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from mathmodel_ai.core.types import ProviderName, ReasoningEffort


class ModelMessage(BaseModel):
    role: Literal["system", "developer", "user", "assistant"]
    content: str = Field(min_length=1)


class GenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    messages: list[ModelMessage] = Field(min_length=1)
    max_output_tokens: int = Field(default=4096, ge=1, le=128_000)
    temperature: float | None = Field(default=None, ge=0, le=2)
    reasoning_effort: ReasoningEffort | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class ModelUsage(BaseModel):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cached_input_tokens: int = Field(default=0, ge=0)
    requests: int = Field(default=0, ge=0)

    def plus(self, other: "ModelUsage") -> "ModelUsage":
        return ModelUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
            requests=self.requests + other.requests,
        )


class ModelResponse(BaseModel):
    content: str
    provider: ProviderName
    model: str
    usage: ModelUsage
    response_id: str | None = None
    finish_reason: str | None = None
    is_mock: bool = False


class StructuredModelResponse[T: BaseModel](BaseModel):
    parsed: T
    response: ModelResponse


JsonObject = dict[str, Any]
