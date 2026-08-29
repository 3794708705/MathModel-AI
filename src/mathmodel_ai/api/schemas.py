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
from mathmodel_ai.schemas.mathematical import MathematicalModel
from mathmodel_ai.schemas.model_selection import ModelExploration, ModelSelection
from mathmodel_ai.schemas.paper import (
    CitationMetadataCheck,
    CompetitionProfile,
    LiteraturePlan,
    PaperArtifact,
    PaperCompileRecord,
    PaperQualityReport,
    PaperVersion,
    ReferenceRecord,
)
from mathmodel_ai.schemas.problem_analysis import ProblemAnalysis
from mathmodel_ai.schemas.problem_state import ProblemState
from mathmodel_ai.schemas.program import (
    ExecutionStrategy,
    ExecutionStrategyDecision,
    GeneratedProgram,
)
from mathmodel_ai.schemas.quality import DataWorkflowStage, QualityGateResult
from mathmodel_ai.schemas.results import ResultRecord
from mathmodel_ai.schemas.solver import (
    AlgorithmPlan,
    SolverOptions,
    SolverRouteDecision,
    SolverRun,
)
from mathmodel_ai.schemas.verification import (
    ModelRepairOutput,
    RedTeamReport,
    RepairCycleRecord,
    RobustnessConfig,
    RobustnessReport,
    SensitivityConfig,
    SensitivityReport,
    ValidationReport,
)


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


class ModelBuildRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_guidance: list[str] = Field(default_factory=list)


class ModelBuildResponse(BaseModel):
    mathematical_model: MathematicalModel
    algorithm_plan: AlgorithmPlan
    state_version: int
    gate: QualityGateResult
    agent_run: AgentRunSummary
    is_mock: bool


class SolveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    options: SolverOptions = Field(default_factory=SolverOptions)
    execution_strategy: ExecutionStrategy = ExecutionStrategy.AUTO
    user_guidance: list[str] = Field(default_factory=list)


class SolveResponse(BaseModel):
    result: ResultRecord
    solver_run: SolverRun
    generated_program: GeneratedProgram
    execution: ExecutionRecord
    route: SolverRouteDecision
    execution_strategy: ExecutionStrategyDecision
    code_agent_run: AgentRunSummary | None = None
    state_version: int
    gate: QualityGateResult


class MathematicalRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_guidance: list[str] = Field(default_factory=list)
    options: SolverOptions = Field(default_factory=SolverOptions)
    execution_strategy: ExecutionStrategy = ExecutionStrategy.AUTO


class MathematicalRunResponse(BaseModel):
    model_stage: ModelBuildResponse
    solve_stage: SolveResponse


class ValidationRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    result_id: UUID | None = None


class ValidationRunResponse(BaseModel):
    report: ValidationReport
    state_version: int
    gate: QualityGateResult


class SensitivityRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config: SensitivityConfig = Field(default_factory=SensitivityConfig)


class SensitivityRunResponse(BaseModel):
    report: SensitivityReport
    state_version: int
    gate: QualityGateResult


class RobustnessRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config: RobustnessConfig = Field(default_factory=RobustnessConfig)


class RobustnessRunResponse(BaseModel):
    report: RobustnessReport
    state_version: int
    gate: QualityGateResult


class RedTeamRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_guidance: list[str] = Field(default_factory=list)


class RedTeamRunResponse(BaseModel):
    report: RedTeamReport
    state_version: int
    gate: QualityGateResult
    verified_gate: QualityGateResult
    verified_result_id: UUID | None
    agent_run: AgentRunSummary


class ModelRepairRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_guidance: list[str] = Field(default_factory=list)


class ModelRepairRunResponse(BaseModel):
    output: ModelRepairOutput
    cycle: RepairCycleRecord
    state_version: int
    model_gate: QualityGateResult
    repair_gate: QualityGateResult
    agent_run: AgentRunSummary


class RepairLoopRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sensitivity: SensitivityConfig = Field(default_factory=SensitivityConfig)
    robustness: RobustnessConfig = Field(default_factory=RobustnessConfig)
    user_guidance: list[str] = Field(default_factory=list)


class RepairLoopRunResponse(BaseModel):
    state: ProblemState
    cycles: list[RepairCycleRecord]
    final_red_team: RedTeamReport
    resolved: bool
    exhausted: bool


class VerificationRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sensitivity: SensitivityConfig = Field(default_factory=SensitivityConfig)
    robustness: RobustnessConfig = Field(default_factory=RobustnessConfig)
    user_guidance: list[str] = Field(default_factory=list)


class VerificationRunResponse(BaseModel):
    validation: ValidationRunResponse
    sensitivity: SensitivityRunResponse
    robustness: RobustnessRunResponse
    red_team: RedTeamRunResponse


class LiteratureSearchResponse(BaseModel):
    references: list[ReferenceRecord]
    metadata_checks: list[CitationMetadataCheck]
    plan: LiteraturePlan


class PaperRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    competition_profile: CompetitionProfile = Field(default_factory=CompetitionProfile)


class PaperRunResponse(BaseModel):
    paper: PaperVersion
    quality: PaperQualityReport
    compile_record: PaperCompileRecord
    artifacts: list[PaperArtifact]
