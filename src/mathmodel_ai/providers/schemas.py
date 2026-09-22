import base64
import binascii
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from mathmodel_ai.core.types import ReasoningEffort

MAX_REPORTED_TOKENS = 1_000_000_000
MAX_REPORTED_REQUESTS = 1_000_000


class MediaPart(BaseModel):
    mime_type: str = Field(pattern=r"^(image/(png|jpeg|webp)|application/pdf)$")
    data_base64: str = Field(min_length=1)
    source_id: str = Field(min_length=1)

    @field_validator("data_base64")
    @classmethod
    def media_must_be_valid_base64(cls, value: str) -> str:
        try:
            base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("media payload is not valid base64") from exc
        return value


class ModelMessage(BaseModel):
    role: Literal["system", "developer", "user", "assistant"]
    content: str = ""
    media: list[MediaPart] = Field(default_factory=list)

    @model_validator(mode="after")
    def text_or_media_is_required(self) -> "ModelMessage":
        if not self.content and not self.media:
            raise ValueError("a model message requires text or media")
        if self.role in {"system", "developer", "assistant"} and self.media:
            raise ValueError("media is supported only on user messages")
        return self


class GenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    messages: list[ModelMessage] = Field(min_length=1)
    max_output_tokens: int = Field(default=32_768, ge=1, le=128_000)
    temperature: float | None = Field(default=None, ge=0, le=2)
    reasoning_effort: ReasoningEffort | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
    tools: list["ToolDefinition"] = Field(default_factory=list, max_length=32)


class ToolFunction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    description: str = Field(min_length=1, max_length=500)
    parameters: dict[str, Any]


class ToolDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["function"] = "function"
    function: ToolFunction


class ModelUsage(BaseModel):
    input_tokens: int | None = Field(default=None, ge=0, le=MAX_REPORTED_TOKENS, strict=True)
    output_tokens: int | None = Field(default=None, ge=0, le=MAX_REPORTED_TOKENS, strict=True)
    cached_input_tokens: int | None = Field(default=None, ge=0, le=MAX_REPORTED_TOKENS, strict=True)
    reasoning_tokens: int | None = Field(default=None, ge=0, le=MAX_REPORTED_TOKENS, strict=True)
    total_tokens: int | None = Field(default=None, ge=0, le=MAX_REPORTED_TOKENS, strict=True)
    requests: int = Field(default=0, ge=0, le=MAX_REPORTED_REQUESTS, strict=True)

    def plus(self, other: "ModelUsage") -> "ModelUsage":
        return ModelUsage(
            input_tokens=_optional_sum(self.input_tokens, other.input_tokens),
            output_tokens=_optional_sum(self.output_tokens, other.output_tokens),
            cached_input_tokens=_optional_sum(self.cached_input_tokens, other.cached_input_tokens),
            reasoning_tokens=_optional_sum(self.reasoning_tokens, other.reasoning_tokens),
            total_tokens=_optional_sum(self.total_tokens, other.total_tokens),
            requests=self.requests + other.requests,
        )


class ModelResponse(BaseModel):
    content: str
    provider: str = Field(min_length=1, max_length=100)
    model: str = Field(min_length=1, max_length=500)
    usage: ModelUsage
    provider_config_digest: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    model_id: str | None = None
    model_config_digest: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    protocol: str | None = None
    endpoint_trust: str | None = None
    structured_output_mode: str | None = None
    reasoning_effective: str | None = None
    tool_call_count: int = Field(default=0, ge=0)
    tool_call_names: list[str] = Field(default_factory=list, max_length=32)
    response_id: str | None = Field(default=None, max_length=500)
    finish_reason: str | None = Field(default=None, max_length=200)
    is_mock: bool = False

    @field_validator("tool_call_names")
    @classmethod
    def valid_tool_call_names(cls, value: list[str]) -> list[str]:
        import re

        if any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,99}", item) for item in value):
            raise ValueError("provider returned an invalid tool name")
        return value


class StructuredModelResponse[T: BaseModel](BaseModel):
    parsed: T
    response: ModelResponse


JsonObject = dict[str, Any]


def _optional_sum(left: int | None, right: int | None) -> int | None:
    if left is None and right is None:
        return None
    return (left or 0) + (right or 0)


GenerationRequest.model_rebuild()
