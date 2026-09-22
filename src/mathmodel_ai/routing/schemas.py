import hashlib
import json
from enum import IntEnum, StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from mathmodel_ai.core.types import ProviderName, ReasoningEffort
from mathmodel_ai.schemas.provider_registry import (
    EndpointTrustLevel,
    ModelCapability,
    StructuredOutputStrategy,
)


class TaskType(StrEnum):
    ARCHITECTURE_REDESIGN = "architecture_redesign"
    MATHEMATICAL_MODELING = "mathematical_modeling"
    MODEL_REPAIR = "model_repair"
    SANDBOX_SECURITY = "sandbox_security"
    E2E_ROOT_CAUSE = "e2e_root_cause"
    FINAL_ACCEPTANCE = "final_acceptance"
    PROBLEM_UNDERSTANDING = "problem_understanding"
    MODEL_EXPLORATION = "model_exploration"
    MODEL_JURY = "model_jury"
    DATA_UNDERSTANDING = "data_understanding"
    DATABASE_ARCHITECTURE = "database_architecture"
    CODE_GENERATION = "code_generation"
    VALIDATION = "validation"
    RED_TEAM = "red_team"
    PAPER_IR = "paper_ir"
    CITATION_VERIFICATION = "citation_verification"
    API = "api"
    CRUD = "crud"
    PARSER = "parser"
    PROVIDER_ADAPTER = "provider_adapter"
    TESTING = "testing"
    TABLE = "table"
    LATEX = "latex"
    REFACTOR = "refactor"
    DOCUMENTATION = "documentation"
    FORMATTING = "formatting"
    CLEANUP = "cleanup"
    OTHER = "other"


class EscalationLevel(IntEnum):
    FAST = 1
    BALANCED = 2
    FLAGSHIP_HIGH = 3
    FLAGSHIP_XHIGH = 4
    FLAGSHIP_MAX = 5
    MULTI_MODEL_REVIEW = 6
    HUMAN_REVIEW = 7


class RouteAction(StrEnum):
    EXECUTE = "EXECUTE"
    MULTI_MODEL_REVIEW = "MULTI_MODEL_REVIEW"
    HUMAN_REVIEW = "HUMAN_REVIEW"


class TaskProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_type: TaskType
    complexity: int = Field(default=1, ge=0, le=5)
    reasoning_requirement: int = Field(default=1, ge=0, le=5)
    maximum_reasoning_effort: ReasoningEffort | None = None
    math_requirement: int = Field(default=0, ge=0, le=5)
    coding_requirement: int = Field(default=0, ge=0, le=5)
    multimodal_requirement: int = Field(default=0, ge=0, le=5)
    long_context_requirement: int = Field(default=0, ge=0, le=5)
    review_requirement: int = Field(default=0, ge=0, le=5)
    security_risk: int = Field(default=0, ge=0, le=5)
    blast_radius: int = Field(default=0, ge=0, le=5)
    cost_sensitivity: int = Field(default=3, ge=0, le=5)
    deadline_pressure: int = Field(default=0, ge=0, le=5)
    retry_count: int = Field(default=0, ge=0)
    minimum_level: EscalationLevel | None = None
    required_capabilities: set[ModelCapability] = Field(default_factory=set)
    requires_json_schema: bool = False
    allows_prompt_json_fallback: bool = True
    requires_native_tools: bool = False
    requires_reasoning_control: bool = False
    minimum_context_tokens: int | None = Field(default=None, ge=1)
    preferred_model_id: str | None = Field(default=None, pattern=r"^[a-z0-9][a-z0-9._-]{1,99}$")
    allowed_endpoint_trust: set[EndpointTrustLevel] | None = None
    allow_model_fallback: bool = True

    @property
    def requirements_digest(self) -> str:
        payload = _canonical(self.model_dump(mode="json"))
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class RouteDecision(BaseModel):
    level: EscalationLevel
    action: RouteAction
    recommended_provider: ProviderName | str | None = None
    recommended_model: str | None = None
    recommended_reasoning: ReasoningEffort | None = None
    selected_provider: ProviderName | str | None = None
    selected_model: str | None = None
    selected_model_id: str | None = None
    provider_config_digest: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    model_config_digest: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    capability_probe_id: UUID | None = None
    capability_probe_digest: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    task_profile_digest: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    protocol: str | None = None
    endpoint_trust: EndpointTrustLevel | None = None
    structured_output_mode: StructuredOutputStrategy | None = None
    selected_reasoning: ReasoningEffort | None = None
    reasoning_effective: str | None = None
    fallback_used: bool = False
    fallback_reason: str | None = None
    rejected_models: dict[str, list[str]] = Field(default_factory=dict)
    legacy_config_used: bool = False
    reason: str = Field(min_length=1)


def _canonical(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _canonical(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        return sorted((_canonical(item) for item in value), key=lambda item: repr(item))
    return value
