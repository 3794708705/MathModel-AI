from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mathmodel_ai.agents.base import AgentRunResult
from mathmodel_ai.core.types import Environment, ProviderName
from mathmodel_ai.schemas.data import (
    CrossDatasetRelationship,
    DataProfile,
    DatasetRecord,
    DataUnderstanding,
)
from mathmodel_ai.schemas.execution import ExecutionRecord
from mathmodel_ai.schemas.files import ArtifactRecord, ParsedFile
from mathmodel_ai.schemas.model_selection import ModelExploration, ModelSelection
from mathmodel_ai.schemas.problem_analysis import ProblemAnalysis
from mathmodel_ai.schemas.problem_state import ProblemState
from mathmodel_ai.schemas.quality import DataWorkflowStage, QualityGateResult


class HealthResponse(BaseModel):
    status: Literal["ok"]


class ReadinessResponse(BaseModel):
    status: Literal["ready"]
    database: Literal["ready"]


class SystemInfoResponse(BaseModel):
    name: str
    version: str
    environment: Environment
    default_provider: ProviderName
    configured_providers: list[ProviderName]


class ProjectProblemCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=255)
    title: str = Field(min_length=1, max_length=500)
    raw_problem: str = Field(min_length=20)
    competition: str | None = Field(default=None, max_length=255)
    deadline: datetime | None = None

    @field_validator("deadline")
    @classmethod
    def deadline_must_be_timezone_aware(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("deadline must include a timezone")
        return value


class NotesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    notes: list[str] = Field(default_factory=list)


class ReasoningRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_notes: list[str] = Field(default_factory=list)
    user_guidance: list[str] = Field(default_factory=list)
    jury_notes: list[str] = Field(default_factory=list)


class AgentRunSummary(BaseModel):
    run_id: UUID
    agent_name: str
    status: str
    input_state_version: int
    output_state_version: int | None
    provider: str | None
    model: str | None
    reasoning: str | None
    prompt_version: str | None
    latency_ms: int
    token_usage: dict[str, int]
    attempts: int
    errors: list[str]
    is_mock: bool

    @classmethod
    def from_run(cls, run: AgentRunResult[Any]) -> "AgentRunSummary":
        return cls(
            **run.model_dump(
                include={
                    "run_id",
                    "agent_name",
                    "status",
                    "input_state_version",
                    "output_state_version",
                    "provider",
                    "model",
                    "reasoning",
                    "prompt_version",
                    "latency_ms",
                    "attempts",
                    "errors",
                    "is_mock",
                },
                mode="json",
            ),
            token_usage=run.token_usage.model_dump(),
        )


class ProblemAnalysisStageResponse(BaseModel):
    output: ProblemAnalysis
    state_version: int
    gate: QualityGateResult
    agent_run: AgentRunSummary
    is_mock: bool


class ModelExplorationStageResponse(BaseModel):
    output: ModelExploration
    state_version: int
    gate: QualityGateResult
    agent_run: AgentRunSummary
    is_mock: bool


class ModelSelectionStageResponse(BaseModel):
    output: ModelSelection
    state_version: int
    gate: QualityGateResult
    agent_run: AgentRunSummary
    is_mock: bool


class ReasoningRunResponse(BaseModel):
    state: ProblemState
    agent_runs: list[AgentRunSummary]
    is_mock: bool


class FileIngestResponse(BaseModel):
    parsed_file: ParsedFile
    datasets: list[DatasetRecord]
    data_profiles: list[DataProfile]
    relationships: list[CrossDatasetRelationship]
    artifacts: list[ArtifactRecord]
    state_version: int
    data_stage: DataWorkflowStage
    gate: QualityGateResult


class DataAnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    media_file_ids: list[UUID] = Field(default_factory=list)
    user_guidance: list[str] = Field(default_factory=list)


class DataAnalyzeResponse(BaseModel):
    output: DataUnderstanding
    state_version: int
    data_stage: DataWorkflowStage
    gate: QualityGateResult
    agent_run: AgentRunSummary
    is_mock: bool


class ExecutionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(min_length=1, max_length=2 * 1024 * 1024)
    input_file_ids: list[UUID] = Field(default_factory=list)


class ExecutionResponse(BaseModel):
    execution: ExecutionRecord
    artifacts: list[ArtifactRecord]
    state_version: int
    data_stage: DataWorkflowStage
    gate: QualityGateResult
