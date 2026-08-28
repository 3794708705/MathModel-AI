from enum import IntEnum, StrEnum

from pydantic import BaseModel, ConfigDict, Field

from mathmodel_ai.core.types import ProviderName, ReasoningEffort


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


class RouteDecision(BaseModel):
    level: EscalationLevel
    action: RouteAction
    recommended_provider: ProviderName | None = None
    recommended_model: str | None = None
    recommended_reasoning: ReasoningEffort | None = None
    selected_provider: ProviderName | None = None
    selected_model: str | None = None
    selected_reasoning: ReasoningEffort | None = None
    fallback_used: bool = False
    reason: str = Field(min_length=1)
