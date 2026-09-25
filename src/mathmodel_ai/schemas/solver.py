from __future__ import annotations

import math
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mathmodel_ai.schemas.execution import ExecutionOrigin

SeriesKey = Annotated[str, Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,79}$")]


class SolverName(StrEnum):
    SCIPY = "SCIPY"
    GUROBI = "GUROBI"
    ORTOOLS = "ORTOOLS"
    SCALAR_RESPONSE = "SCALAR_RESPONSE"


class SolverFamily(StrEnum):
    SCIPY_HIGHS = "SCIPY_HIGHS"
    SCIPY_MILP = "SCIPY_MILP"
    SCIPY_MINIMIZE = "SCIPY_MINIMIZE"
    GUROBI = "GUROBI"
    ORTOOLS_CP_SAT = "ORTOOLS_CP_SAT"
    SCALAR_RESPONSE = "SCALAR_RESPONSE"


class SolverCapability(StrEnum):
    LP = "LP"
    MILP = "MILP"
    INTEGER = "INTEGER"
    NLP = "NLP"
    BINARY = "BINARY"
    CONTINUOUS = "CONTINUOUS"
    MULTIOBJECTIVE = "MULTIOBJECTIVE"


class ProblemSizeClass(StrEnum):
    TINY = "TINY"
    SMALL = "SMALL"
    MEDIUM = "MEDIUM"
    LARGE = "LARGE"


class SolverStatus(StrEnum):
    OPTIMAL = "OPTIMAL"
    FEASIBLE = "FEASIBLE"
    INFEASIBLE = "INFEASIBLE"
    UNBOUNDED = "UNBOUNDED"
    INFEASIBLE_OR_UNBOUNDED = "INFEASIBLE_OR_UNBOUNDED"
    TIME_LIMIT = "TIME_LIMIT"
    ITERATION_LIMIT = "ITERATION_LIMIT"
    NUMERICAL_ERROR = "NUMERICAL_ERROR"
    MODEL_INVALID = "MODEL_INVALID"
    SOLVER_UNAVAILABLE = "SOLVER_UNAVAILABLE"
    EXECUTION_ERROR = "EXECUTION_ERROR"
    UNKNOWN = "UNKNOWN"


class AlgorithmFamily(StrEnum):
    LINEAR_OPTIMIZATION = "LINEAR_OPTIMIZATION"
    MIXED_INTEGER_OPTIMIZATION = "MIXED_INTEGER_OPTIMIZATION"
    INTEGER_CONSTRAINT_PROGRAMMING = "INTEGER_CONSTRAINT_PROGRAMMING"
    NONLINEAR_LOCAL_OPTIMIZATION = "NONLINEAR_LOCAL_OPTIMIZATION"
    LEAST_SQUARES = "LEAST_SQUARES"
    STATISTICAL_ESTIMATION = "STATISTICAL_ESTIMATION"
    GRAPH_ALGORITHM = "GRAPH_ALGORITHM"
    SIMULATION = "SIMULATION"
    UNSUPPORTED = "UNSUPPORTED"


class AlgorithmPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    algorithm_family: AlgorithmFamily
    recommended_solver_family: SolverFamily | None = None
    reason: str = Field(min_length=1)
    requirements: list[str] = Field(default_factory=list)
    alternatives: list[SolverFamily] = Field(default_factory=list)
    complexity_notes: list[str] = Field(default_factory=list)
    numerical_risks: list[str] = Field(default_factory=list)
    deterministic: bool = True


class SolverOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    time_limit_seconds: float | None = Field(default=None, gt=0, le=3600)
    iteration_limit: int | None = Field(default=None, ge=1)
    mip_gap: float | None = Field(default=None, ge=0, le=1)
    feasibility_tolerance: float = Field(default=1e-7, gt=0, le=0.1)
    initial_point: dict[str, float] = Field(default_factory=dict)
    deadline_pressure: int = Field(default=0, ge=0, le=5)


class SolverHealth(BaseModel):
    solver_name: SolverName
    available: bool
    version: str | None = None
    reason: str = Field(min_length=1)


class SupportAssessment(BaseModel):
    supported: bool
    reasons: list[str] = Field(default_factory=list)


class ProblemSize(BaseModel):
    classification: ProblemSizeClass
    number_of_variables: int = Field(ge=0)
    number_of_integer_variables: int = Field(ge=0)
    number_of_binary_variables: int = Field(ge=0)
    number_of_constraints: int = Field(ge=0)
    number_of_nonzero_coefficients: int | None = Field(default=None, ge=0)


class ProblemSizeThresholds(BaseModel):
    tiny_variables: int = Field(default=10, ge=1)
    tiny_constraints: int = Field(default=10, ge=1)
    tiny_nonzeros: int = Field(default=100, ge=1)
    small_variables: int = Field(default=100, ge=1)
    small_constraints: int = Field(default=100, ge=1)
    small_nonzeros: int = Field(default=2_000, ge=1)
    medium_variables: int = Field(default=1_000, ge=1)
    medium_constraints: int = Field(default=1_000, ge=1)
    medium_nonzeros: int = Field(default=100_000, ge=1)

    @model_validator(mode="after")
    def thresholds_increase(self) -> ProblemSizeThresholds:
        for prefix in ("variables", "constraints", "nonzeros"):
            if not (
                getattr(self, f"tiny_{prefix}")
                < getattr(self, f"small_{prefix}")
                < getattr(self, f"medium_{prefix}")
            ):
                raise ValueError(f"{prefix} size thresholds must strictly increase")
        return self


class SolverScore(BaseModel):
    family: SolverFamily
    solver: SolverName
    score: float
    reasons: list[str] = Field(default_factory=list)


class SolverHardRejection(BaseModel):
    family: SolverFamily
    reason: str = Field(min_length=1)


class SolverRouteDecision(BaseModel):
    routing_decision_id: UUID = Field(default_factory=uuid4)
    selected_solver: SolverName
    selected_family: SolverFamily
    attempted_families: list[SolverFamily]
    alternatives: list[SolverFamily] = Field(default_factory=list)
    scores: list[SolverScore] = Field(default_factory=list)
    hard_rejections: list[SolverHardRejection] = Field(default_factory=list)
    problem_size: ProblemSize
    runtime_budget_seconds: float | None = Field(default=None, gt=0)
    fallback_used: bool
    reason: str = Field(min_length=1)


class FeasibilityReport(BaseModel):
    checked: bool
    max_constraint_violation: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    violated_constraints: list[str] = Field(default_factory=list)
    bound_violations: list[str] = Field(default_factory=list)
    tolerance: float = Field(gt=0)
    message: str = Field(min_length=1)


class SolverResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    solver_name: SolverName
    solver_version: str | None = None
    status: SolverStatus
    objective_value: float | None = Field(default=None, allow_inf_nan=False)
    variable_values: dict[str, float] = Field(default_factory=dict)
    runtime_seconds: float = Field(ge=0)
    iterations: int | None = Field(default=None, ge=0)
    nodes: int | None = Field(default=None, ge=0)
    mip_gap: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    message: str = ""
    raw_status: str = ""
    is_optimal: bool = False
    is_feasible: bool = False
    warnings: list[str] = Field(default_factory=list)
    execution_record_id: UUID
    feasibility: FeasibilityReport | None = None

    @model_validator(mode="after")
    def status_flags_must_be_truthful(self) -> SolverResult:
        if self.status is SolverStatus.OPTIMAL and not (self.is_optimal and self.is_feasible):
            raise ValueError("OPTIMAL requires is_optimal=true and is_feasible=true")
        if self.is_optimal and self.status is not SolverStatus.OPTIMAL:
            raise ValueError("is_optimal may only accompany OPTIMAL status")
        if self.status is SolverStatus.FEASIBLE and not self.is_feasible:
            raise ValueError("FEASIBLE requires is_feasible=true")
        if self.objective_value is not None and not math.isfinite(self.objective_value):
            raise ValueError("objective_value must be finite")
        if any(not math.isfinite(value) for value in self.variable_values.values()):
            raise ValueError("variable values must be finite")
        return self


class GeneratedResultPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    solver_name: SolverName
    solver_version: str | None = None
    model_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    status: SolverStatus
    objective: float | None = Field(default=None, allow_inf_nan=False)
    variable_values: dict[str, float] = Field(default_factory=dict)
    predictions: list[float] = Field(default_factory=list, max_length=100000)
    series: dict[SeriesKey, list[float]] = Field(default_factory=dict, max_length=100)
    metrics: dict[str, int | float | str | bool | None] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    message: str = ""
    raw_status: str = ""
    is_optimal: bool = False
    is_feasible: bool = False

    @model_validator(mode="after")
    def status_semantics_are_truthful(self) -> GeneratedResultPayload:
        if self.status is SolverStatus.OPTIMAL and not (self.is_optimal and self.is_feasible):
            raise ValueError("generated OPTIMAL result must be optimal and feasible")
        if self.is_optimal and self.status is not SolverStatus.OPTIMAL:
            raise ValueError("generated is_optimal requires OPTIMAL status")
        if self.status is SolverStatus.FEASIBLE and not self.is_feasible:
            raise ValueError("generated FEASIBLE result requires is_feasible=true")
        if any(not math.isfinite(value) for value in self.predictions):
            raise ValueError("generated predictions must be finite")
        if any(
            len(values) > 100000 or any(not math.isfinite(value) for value in values)
            for values in self.series.values()
        ):
            raise ValueError("generated series must be bounded and finite")
        return self


class SolverRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    solver_run_id: UUID = Field(default_factory=uuid4)
    project_id: UUID
    problem_id: UUID
    model_id: UUID
    model_version: int = Field(ge=1)
    model_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    generated_program_id: UUID | None = None
    execution_origin: ExecutionOrigin
    routing_decision_id: UUID
    routing_decision: SolverRouteDecision
    solver: SolverName
    solver_version: str | None = None
    options: SolverOptions
    start_time: datetime = Field(default_factory=lambda: datetime.now(UTC))
    end_time: datetime
    status: SolverStatus
    objective: float | None = Field(default=None, allow_inf_nan=False)
    runtime_seconds: float = Field(ge=0)
    result_ref: UUID
    execution_ref: UUID
    error: str | None = None
    result: SolverResult

    @model_validator(mode="after")
    def result_links_match(self) -> SolverRun:
        if self.result.execution_record_id != self.execution_ref:
            raise ValueError("solver run execution_ref must match SolverResult")
        if self.result.status is not self.status:
            raise ValueError("solver run status must match SolverResult")
        if self.routing_decision.routing_decision_id != self.routing_decision_id:
            raise ValueError("solver run routing decision reference must match payload")
        if self.result.solver_name is not self.solver:
            raise ValueError("solver run solver must match SolverResult")
        return self


class SolverRunRef(BaseModel):
    solver_run_id: UUID
    model_id: UUID
    model_version: int = Field(ge=1)
    model_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    solver: SolverName
    status: SolverStatus
    execution_ref: UUID
    result_ref: UUID
