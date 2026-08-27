from __future__ import annotations

import math
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mathmodel_ai.schemas.data import DataProfile, DataUnderstanding
from mathmodel_ai.schemas.model_selection import ModelCandidate, ModelFamily
from mathmodel_ai.schemas.problem_analysis import ProblemAnalysis

type ParameterScalar = int | float | str | bool
type ParameterValue = ParameterScalar | list[ParameterScalar] | dict[str, ParameterScalar]


class ExpressionKind(StrEnum):
    CONSTANT = "CONSTANT"
    SYMBOL = "SYMBOL"
    ADD = "ADD"
    SUBTRACT = "SUBTRACT"
    MULTIPLY = "MULTIPLY"
    DIVIDE = "DIVIDE"
    POWER = "POWER"
    NEGATE = "NEGATE"


class MathExpression(BaseModel):
    """Small safe expression tree; it is deliberately not a general CAS."""

    model_config = ConfigDict(extra="forbid")

    kind: ExpressionKind
    value: float | None = Field(default=None, allow_inf_nan=False)
    symbol: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z][A-Za-z0-9_]*(?:\[[A-Za-z0-9_,]+\])?$",
    )
    operands: list[MathExpression] = Field(default_factory=list, max_length=1000)

    @model_validator(mode="after")
    def validate_shape(self) -> MathExpression:
        if self.kind is ExpressionKind.CONSTANT:
            if self.value is None or self.symbol is not None or self.operands:
                raise ValueError("CONSTANT requires only a finite value")
        elif self.kind is ExpressionKind.SYMBOL:
            if self.symbol is None or self.value is not None or self.operands:
                raise ValueError("SYMBOL requires only a symbol")
        elif self.kind is ExpressionKind.NEGATE:
            if self.value is not None or self.symbol is not None or len(self.operands) != 1:
                raise ValueError("NEGATE requires exactly one operand")
        elif self.kind in {
            ExpressionKind.SUBTRACT,
            ExpressionKind.DIVIDE,
            ExpressionKind.POWER,
        }:
            if self.value is not None or self.symbol is not None or len(self.operands) != 2:
                raise ValueError(f"{self.kind.value} requires exactly two operands")
        elif self.kind in {ExpressionKind.ADD, ExpressionKind.MULTIPLY}:
            if self.value is not None or self.symbol is not None or len(self.operands) < 2:
                raise ValueError(f"{self.kind.value} requires at least two operands")
        return self

    @classmethod
    def constant(cls, value: float) -> MathExpression:
        return cls(kind=ExpressionKind.CONSTANT, value=value)

    @classmethod
    def symbol_ref(cls, symbol: str) -> MathExpression:
        return cls(kind=ExpressionKind.SYMBOL, symbol=symbol)


class BaseDimension(StrEnum):
    MASS = "mass"
    LENGTH = "length"
    TIME = "time"
    CURRENCY = "currency"
    COUNT = "count"
    ENERGY = "energy"
    POWER = "power"
    DIMENSIONLESS = "dimensionless"


class UnitCheckStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"


class UnitExpression(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dimensions: dict[BaseDimension, int] = Field(default_factory=dict)
    scale: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    display: str = Field(default="1", min_length=1)
    unknown_units: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_dimensions(self) -> UnitExpression:
        if self.dimensions.get(BaseDimension.DIMENSIONLESS, 0) != 0:
            raise ValueError("dimensionless cannot carry a non-zero exponent")
        if len(self.unknown_units) != len(set(self.unknown_units)):
            raise ValueError("unknown unit tokens must be unique")
        return self

    @property
    def is_known(self) -> bool:
        return not self.unknown_units


class VariableDomain(StrEnum):
    CONTINUOUS = "CONTINUOUS"
    INTEGER = "INTEGER"
    BINARY = "BINARY"
    NONNEGATIVE_CONTINUOUS = "NONNEGATIVE_CONTINUOUS"
    NONNEGATIVE_INTEGER = "NONNEGATIVE_INTEGER"


class VariableRole(StrEnum):
    DECISION = "DECISION"
    STATE = "STATE"
    DERIVED = "DERIVED"


class ParameterSourceType(StrEnum):
    PROBLEM_FACT = "PROBLEM_FACT"
    DATA = "DATA"
    ASSUMPTION = "ASSUMPTION"
    DERIVATION = "DERIVATION"
    EXTERNAL = "EXTERNAL"
    ESTIMATED = "ESTIMATED"


class ObjectiveSense(StrEnum):
    MINIMIZE = "MINIMIZE"
    MAXIMIZE = "MAXIMIZE"


class ConstraintRelation(StrEnum):
    LE = "LE"
    EQ = "EQ"
    GE = "GE"


class ConvexityStatus(StrEnum):
    CONVEX = "CONVEX"
    NONCONVEX = "NONCONVEX"
    UNKNOWN = "UNKNOWN"


class MathematicalModelStatus(StrEnum):
    DRAFT = "DRAFT"
    READY = "READY"
    MODEL_GATE_FAIL = "MODEL_GATE_FAIL"
    SOLVED = "SOLVED"


class InterpretationResolutionStatus(StrEnum):
    RESOLVED_FROM_EVIDENCE = "RESOLVED_FROM_EVIDENCE"
    HUMAN_CONFIRMED = "HUMAN_CONFIRMED"
    ASSUMPTION_REQUIRED = "ASSUMPTION_REQUIRED"
    UNRESOLVED = "UNRESOLVED"


class SetDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    set_id: str = Field(pattern=r"^SET-[A-Za-z0-9_-]+$")
    symbol: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]*$")
    description: str = Field(min_length=1)
    values: list[ParameterScalar] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)


class IndexDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index_id: str = Field(pattern=r"^IDX-[A-Za-z0-9_-]+$")
    symbol: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]*$")
    description: str = Field(min_length=1)
    set_ref: str = Field(pattern=r"^SET-[A-Za-z0-9_-]+$")


class VariableDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    variable_id: str = Field(pattern=r"^VAR-[A-Za-z0-9_-]+$")
    symbol: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]*$")
    description: str = Field(min_length=1)
    index_sets: list[str] = Field(default_factory=list)
    domain: VariableDomain
    lower_bound: float | None = Field(default=None, allow_inf_nan=False)
    upper_bound: float | None = Field(default=None, allow_inf_nan=False)
    unit: UnitExpression | None = None
    role: VariableRole
    source_refs: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_bounds_and_role(self) -> VariableDefinition:
        if (
            self.lower_bound is not None
            and self.upper_bound is not None
            and self.lower_bound > self.upper_bound
        ):
            raise ValueError("variable lower_bound cannot exceed upper_bound")
        if self.domain is VariableDomain.BINARY:
            if self.lower_bound is not None and self.lower_bound < 0:
                raise ValueError("binary lower_bound cannot be below zero")
            if self.upper_bound is not None and self.upper_bound > 1:
                raise ValueError("binary upper_bound cannot exceed one")
        if (
            self.domain
            in {
                VariableDomain.NONNEGATIVE_CONTINUOUS,
                VariableDomain.NONNEGATIVE_INTEGER,
            }
            and self.lower_bound is not None
            and self.lower_bound < 0
        ):
            raise ValueError("nonnegative domain cannot have a negative lower bound")
        return self


class DataBinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    binding_id: str = Field(pattern=r"^BIND-[A-Za-z0-9_-]+$")
    dataset_id: UUID
    column: str = Field(min_length=1)
    selector: str | None = None
    transform: str | None = None
    artifact_ref: str | None = None


class ParameterDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parameter_id: str = Field(pattern=r"^PAR-[A-Za-z0-9_-]+$")
    symbol: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]*$")
    description: str = Field(min_length=1)
    value: ParameterValue | None = None
    data_binding: DataBinding | None = None
    unit: UnitExpression | None = None
    source_type: ParameterSourceType
    source_ref: str = Field(min_length=1)
    is_estimated: bool = False
    estimation_method: str | None = None
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_source_and_value(self) -> ParameterDefinition:
        if self.value is None and self.data_binding is None:
            raise ValueError("parameter requires a value or data_binding")
        if self.is_estimated and not self.estimation_method:
            raise ValueError("estimated parameter requires estimation_method")
        if self.source_type is ParameterSourceType.ESTIMATED and not self.is_estimated:
            raise ValueError("ESTIMATED source requires is_estimated=true")
        if isinstance(self.value, float) and not math.isfinite(self.value):
            raise ValueError("parameter scalar must be finite")
        return self


class ConstantDefinition(ParameterDefinition):
    parameter_id: str = Field(pattern=r"^CONST-[A-Za-z0-9_-]+$")


class ObjectiveDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    objective_id: str = Field(pattern=r"^OBJ-[A-Za-z0-9_-]+$")
    sense: ObjectiveSense
    expression: MathExpression
    description: str = Field(min_length=1)
    unit: UnitExpression | None = None
    source_refs: list[str] = Field(min_length=1)
    derivation: str = Field(min_length=1)
    equation_ref: str = Field(pattern=r"^EQ-[A-Za-z0-9_-]+$")


class ConstraintDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    constraint_id: str = Field(pattern=r"^CON-[A-Za-z0-9_-]+$")
    name: str = Field(min_length=1)
    expression: MathExpression
    relation: ConstraintRelation
    rhs: MathExpression
    normalized_expression: str = Field(min_length=1)
    description: str = Field(min_length=1)
    source_refs: list[str] = Field(min_length=1)
    assumption_refs: list[str] = Field(default_factory=list)
    unit: UnitExpression | None = None
    index_scope: list[str] = Field(default_factory=list)
    is_hard: bool = True
    equation_ref: str = Field(pattern=r"^EQ-[A-Za-z0-9_-]+$")


class EquationDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    equation_id: str = Field(pattern=r"^EQ-[A-Za-z0-9_-]+$")
    latex: str = Field(min_length=1)
    normalized_expression: str = Field(min_length=1)
    lhs: MathExpression
    rhs: MathExpression
    meaning: str = Field(min_length=1)
    source_refs: list[str] = Field(min_length=1)
    derivation: str = Field(min_length=1)
    symbol_refs: list[str] = Field(default_factory=list)
    parameter_refs: list[str] = Field(default_factory=list)
    dependency_refs: list[str] = Field(default_factory=list)
    unit_lhs: UnitExpression | None = None
    unit_rhs: UnitExpression | None = None
    dimension_status: UnitCheckStatus = UnitCheckStatus.UNKNOWN


class ModelAssumption(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assumption_id: str = Field(pattern=r"^(EVID-|ASSUMP-)[A-Za-z0-9_-]+$")
    statement: str = Field(min_length=1)
    source_refs: list[str] = Field(min_length=1)
    critical: bool = False
    supported: bool = False
    support_reason: str | None = None

    @model_validator(mode="after")
    def supported_requires_reason(self) -> ModelAssumption:
        if self.supported and not self.support_reason:
            raise ValueError("supported assumption requires support_reason")
        return self


class InterpretationResolution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ambiguity_id: str = Field(pattern=r"^AMB-[A-Za-z0-9_-]+$")
    interpretation_id: str | None = None
    status: InterpretationResolutionStatus
    reason: str = Field(min_length=1)
    source_refs: list[str] = Field(default_factory=list)
    critical: bool = True


class AlgorithmRequirements(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exact_solution_required: bool = False
    deterministic_required: bool = True
    convexity: ConvexityStatus = ConvexityStatus.UNKNOWN
    supports_local_solution: bool = False
    numerical_tolerance: float = Field(default=1e-7, gt=0, le=0.1)
    notes: list[str] = Field(default_factory=list)


class SolverRequirements(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requires_continuous: bool = False
    requires_integer: bool = False
    requires_binary: bool = False
    requires_nonlinear: bool = False
    requires_multiobjective: bool = False
    required_capabilities: list[str] = Field(default_factory=list)
    allow_commercial_solver: bool = True
    maximum_runtime_seconds: float | None = Field(default=None, gt=0, le=3600)
    preferred_solver_families: list[str] = Field(default_factory=list)


class ExpectedOutput(BaseModel):
    output_id: str = Field(pattern=r"^OUT-[A-Za-z0-9_-]+$")
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    unit: UnitExpression | None = None
    source_refs: list[str] = Field(default_factory=list)


class MathematicalModelContent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    target_subproblems: list[str] = Field(min_length=1)
    model_family: ModelFamily
    assumptions: list[ModelAssumption] = Field(default_factory=list)
    interpretation_resolutions: list[InterpretationResolution] = Field(default_factory=list)
    sets: list[SetDefinition] = Field(default_factory=list)
    indices: list[IndexDefinition] = Field(default_factory=list)
    decision_variables: list[VariableDefinition] = Field(default_factory=list)
    state_variables: list[VariableDefinition] = Field(default_factory=list)
    derived_variables: list[VariableDefinition] = Field(default_factory=list)
    parameters: list[ParameterDefinition] = Field(default_factory=list)
    constants: list[ConstantDefinition] = Field(default_factory=list)
    objective: ObjectiveDefinition | None = None
    constraints: list[ConstraintDefinition] = Field(default_factory=list)
    equations: list[EquationDefinition] = Field(default_factory=list)
    initial_conditions: list[ConstraintDefinition] = Field(default_factory=list)
    boundary_conditions: list[ConstraintDefinition] = Field(default_factory=list)
    units: dict[str, UnitExpression] = Field(default_factory=dict)
    algorithm_requirements: AlgorithmRequirements = Field(default_factory=AlgorithmRequirements)
    solver_requirements: SolverRequirements = Field(default_factory=SolverRequirements)
    data_bindings: list[DataBinding] = Field(default_factory=list)
    source_evidence: list[str] = Field(min_length=1)
    expected_outputs: list[ExpectedOutput] = Field(min_length=1)
    validation_requirements: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_identifiers_and_roles(self) -> MathematicalModelContent:
        collections = {
            "set": [item.set_id for item in self.sets],
            "index": [item.index_id for item in self.indices],
            "variable": [
                item.variable_id
                for item in [
                    *self.decision_variables,
                    *self.state_variables,
                    *self.derived_variables,
                ]
            ],
            "parameter": [item.parameter_id for item in [*self.parameters, *self.constants]],
            "constraint": [
                item.constraint_id
                for item in [
                    *self.constraints,
                    *self.initial_conditions,
                    *self.boundary_conditions,
                ]
            ],
            "equation": [item.equation_id for item in self.equations],
            "output": [item.output_id for item in self.expected_outputs],
        }
        for kind, identifiers in collections.items():
            if len(identifiers) != len(set(identifiers)):
                raise ValueError(f"{kind} identifiers must be unique")
        expected_roles = (
            (self.decision_variables, VariableRole.DECISION),
            (self.state_variables, VariableRole.STATE),
            (self.derived_variables, VariableRole.DERIVED),
        )
        for variables, role in expected_roles:
            if any(item.role is not role for item in variables):
                raise ValueError(f"{role.value} variables must use role={role.value}")
        set_ids = {item.set_id for item in self.sets}
        if any(item.set_ref not in set_ids for item in self.indices):
            raise ValueError("index set_ref must reference a declared set")
        known_index_sets = set_ids
        if any(
            not set(item.index_sets) <= known_index_sets
            for item in [
                *self.decision_variables,
                *self.state_variables,
                *self.derived_variables,
            ]
        ):
            raise ValueError("variable index_sets must reference declared sets")
        return self


class MathematicalModelDraft(MathematicalModelContent):
    """LLM-authored content before deterministic identity/version binding."""


class MathematicalModel(MathematicalModelContent):
    model_id: UUID
    project_id: UUID
    problem_id: UUID
    version: int = Field(ge=1)
    source_selected_model_id: str = Field(pattern=r"^CAND-[A-Za-z0-9_-]+$")
    status: MathematicalModelStatus = MathematicalModelStatus.DRAFT


class MathematicalModelRef(BaseModel):
    record_id: UUID
    model_id: UUID
    version: int = Field(ge=1)
    model_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_selected_model_id: str
    status: MathematicalModelStatus


class MathModelerInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assigned_model_id: UUID
    assigned_version: int = Field(ge=1)
    selected_model: ModelCandidate
    problem_analysis: ProblemAnalysis
    data_understanding: DataUnderstanding | None = None
    data_profiles: list[DataProfile] = Field(default_factory=list)
    user_guidance: list[str] = Field(default_factory=list)
