from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mathmodel_ai.schemas.problem_analysis import DataRequirement, ProblemAnalysis


class ModelFamily(StrEnum):
    LINEAR_PROGRAMMING = "linear_programming"
    INTEGER_PROGRAMMING = "integer_programming"
    MILP = "mixed_integer_linear_programming"
    NONLINEAR_PROGRAMMING = "nonlinear_programming"
    MULTI_OBJECTIVE = "multi_objective_optimization"
    REGRESSION = "regression"
    STATISTICAL = "statistical_model"
    TIME_SERIES = "time_series"
    GREY_MODEL = "grey_model"
    GRAPH = "graph_network"
    VEHICLE_ROUTING = "vehicle_routing"
    MARKOV = "markov"
    DIFFERENTIAL_EQUATION = "differential_equation"
    DYNAMIC_SYSTEM = "dynamic_system"
    QUEUEING = "queueing"
    MONTE_CARLO = "monte_carlo"
    DISCRETE_EVENT = "discrete_event_simulation"
    CLUSTERING = "clustering"
    TREE_ENSEMBLE = "tree_ensemble"
    NEURAL_NETWORK = "neural_network"
    ROBUST_OPTIMIZATION = "robust_optimization"
    STOCHASTIC_OPTIMIZATION = "stochastic_optimization"
    MODEL_CHAIN = "model_chain"
    OTHER = "other"


class ComplexityLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class CapabilityAssessment(BaseModel):
    score: float = Field(ge=0, le=10)
    rationale: str = Field(min_length=1)


class ModelIOContract(BaseModel):
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    source_or_destination: str = Field(min_length=1)


class ModelChainStage(BaseModel):
    order: int = Field(ge=1)
    name: str = Field(min_length=1)
    family: ModelFamily
    mathematical_method: str = Field(min_length=1)
    inputs: list[ModelIOContract] = Field(default_factory=list)
    outputs: list[ModelIOContract] = Field(min_length=1)


class ModelCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(pattern=r"^CAND-[A-Za-z0-9_-]+$")
    name: str = Field(min_length=1)
    normalized_name: str = Field(min_length=1)
    family: ModelFamily
    target_subproblems: list[str] = Field(min_length=1)
    description: str = Field(min_length=1)
    fit_reason: str = Field(min_length=1)
    mathematical_core: str = Field(min_length=1)
    required_inputs: list[ModelIOContract] = Field(min_length=1)
    expected_outputs: list[ModelIOContract] = Field(min_length=1)
    assumptions_required: list[str] = Field(default_factory=list)
    data_requirements: list[DataRequirement] = Field(default_factory=list)
    advantages: list[str] = Field(min_length=1)
    disadvantages: list[str] = Field(min_length=1)
    explainability: CapabilityAssessment
    mathematical_rigor: CapabilityAssessment
    computational_complexity: ComplexityLevel
    implementation_complexity: ComplexityLevel
    validation_potential: CapabilityAssessment
    robustness_potential: CapabilityAssessment
    innovation_potential: CapabilityAssessment
    competition_feasibility: CapabilityAssessment
    risks: list[str] = Field(default_factory=list)
    model_chain_position: list[ModelChainStage] = Field(min_length=1)
    references_needed: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)


class ModelExploration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidates: list[ModelCandidate] = Field(min_length=2, max_length=5)
    fewer_than_three_reason: str | None = None
    exploration_summary: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_candidate_count_and_ids(self) -> "ModelExploration":
        ids = [candidate.candidate_id for candidate in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate ids must be unique")
        if len(self.candidates) < 3 and not self.fewer_than_three_reason:
            raise ValueError("fewer than three candidates requires a reason")
        return self


class ModelExplorerInput(BaseModel):
    analysis: ProblemAnalysis
    user_guidance: list[str] = Field(default_factory=list)


class HardFailure(StrEnum):
    DATA_NOT_AVAILABLE = "DATA_NOT_AVAILABLE"
    MATHEMATICALLY_INVALID = "MATHEMATICALLY_INVALID"
    VIOLATES_PROBLEM_REQUIREMENT = "VIOLATES_PROBLEM_REQUIREMENT"
    IMPOSSIBLE_WITHIN_DEADLINE = "IMPOSSIBLE_WITHIN_DEADLINE"
    NOT_VALIDATABLE = "NOT_VALIDATABLE"
    CRITICAL_ASSUMPTION_UNSUPPORTED = "CRITICAL_ASSUMPTION_UNSUPPORTED"


class CandidateJuryAssessment(BaseModel):
    candidate_id: str = Field(min_length=1)
    problem_fit: float = Field(ge=0, le=10)
    data_fit: float = Field(ge=0, le=10)
    mathematical_validity: float = Field(ge=0, le=10)
    explainability: float = Field(ge=0, le=10)
    validation_potential: float = Field(ge=0, le=10)
    innovation_potential: float = Field(ge=0, le=10)
    competition_feasibility: float = Field(ge=0, le=10)
    computational_cost: float = Field(
        ge=0,
        le=10,
        description="10 means low cost and strong feasibility; 0 means prohibitive cost",
    )
    hard_failures: list[HardFailure] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    rationale: str = Field(min_length=1)


class JuryAssessment(BaseModel):
    assessments: list[CandidateJuryAssessment] = Field(min_length=2)
    critical_risks: list[str] = Field(default_factory=list)
    overall_rationale: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def assessment_ids_must_be_unique(self) -> "JuryAssessment":
        ids = [assessment.candidate_id for assessment in self.assessments]
        if len(ids) != len(set(ids)):
            raise ValueError("jury assessment candidate ids must be unique")
        return self


class ModelJuryWeights(BaseModel):
    problem_fit: float = Field(default=25, ge=0)
    data_fit: float = Field(default=15, ge=0)
    mathematical_validity: float = Field(default=15, ge=0)
    explainability: float = Field(default=10, ge=0)
    validation_potential: float = Field(default=10, ge=0)
    innovation_potential: float = Field(default=10, ge=0)
    competition_feasibility: float = Field(default=10, ge=0)
    computational_cost: float = Field(default=5, ge=0)
    version: str = "default-v1"

    @model_validator(mode="after")
    def weights_must_sum_to_one_hundred(self) -> "ModelJuryWeights":
        total = sum(
            (
                self.problem_fit,
                self.data_fit,
                self.mathematical_validity,
                self.explainability,
                self.validation_potential,
                self.innovation_potential,
                self.competition_feasibility,
                self.computational_cost,
            )
        )
        if abs(total - 100) > 1e-9:
            raise ValueError("model jury weights must sum to 100")
        return self


class ModelScore(BaseModel):
    candidate_id: str
    problem_fit: float = Field(ge=0, le=10)
    data_fit: float = Field(ge=0, le=10)
    mathematical_validity: float = Field(ge=0, le=10)
    explainability: float = Field(ge=0, le=10)
    validation_potential: float = Field(ge=0, le=10)
    innovation_potential: float = Field(ge=0, le=10)
    competition_feasibility: float = Field(ge=0, le=10)
    computational_cost: float = Field(ge=0, le=10)
    weighted_total: float = Field(ge=0, le=100)
    hard_failures: list[HardFailure] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    rationale: str
    eligible: bool


class RejectedModel(BaseModel):
    candidate_id: str
    reason: str
    hard_failures: list[HardFailure] = Field(default_factory=list)


class ModelSelection(BaseModel):
    selected_model_id: str
    backup_model_id: str
    ranking: list[str] = Field(min_length=2)
    score_matrix: list[ModelScore] = Field(min_length=2)
    decision_reason: str = Field(min_length=1)
    rejected_models: list[RejectedModel] = Field(default_factory=list)
    critical_risks: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    weights_version: str

    @model_validator(mode="after")
    def selected_and_backup_must_differ(self) -> "ModelSelection":
        if self.selected_model_id == self.backup_model_id:
            raise ValueError("selected and backup models must differ")
        return self


class ModelJuryInput(BaseModel):
    analysis: ProblemAnalysis
    candidates: list[ModelCandidate] = Field(min_length=2)
    jury_notes: list[str] = Field(default_factory=list)


class ModelDecisionEvidence(BaseModel):
    decision_id: str = Field(pattern=r"^DECISION-[A-Za-z0-9_-]+$")
    candidate_scores: list[ModelScore]
    weights: ModelJuryWeights
    jury_rationale: str
    selected_model_id: str
    backup_model_id: str
    agent_run_id: UUID
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
