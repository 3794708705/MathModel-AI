from __future__ import annotations

import math
from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from mathmodel_ai.schemas.mathematical import MathematicalModel, MathematicalModelDraft
from mathmodel_ai.schemas.results import EvidenceChainReport
from mathmodel_ai.schemas.solver import SolverName, SolverOptions, SolverStatus


class ValidationStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_EVALUABLE = "NOT_EVALUABLE"
    INCONCLUSIVE = "INCONCLUSIVE"


class ValidationCheckStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNCHECKED = "UNCHECKED"


class ValidationCheckCategory(StrEnum):
    EVIDENCE = "EVIDENCE"
    VARIABLE = "VARIABLE"
    BOUND = "BOUND"
    DOMAIN = "DOMAIN"
    CONSTRAINT = "CONSTRAINT"
    OBJECTIVE = "OBJECTIVE"
    OUTPUT = "OUTPUT"
    REQUIREMENT = "REQUIREMENT"


class VariableValidationCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    variable_id: str = Field(min_length=1)
    symbol: str = Field(min_length=1)
    value: float | None = Field(default=None, allow_inf_nan=False)
    lower_bound: float | None = Field(default=None, allow_inf_nan=False)
    upper_bound: float | None = Field(default=None, allow_inf_nan=False)
    bound_violation: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    integrality_violation: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    status: ValidationCheckStatus
    message: str = Field(min_length=1)


class ConstraintValidationCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    constraint_id: str = Field(min_length=1)
    relation: str = Field(min_length=1)
    left_value: float | None = Field(default=None, allow_inf_nan=False)
    right_value: float | None = Field(default=None, allow_inf_nan=False)
    violation: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    tolerance: float = Field(gt=0)
    status: ValidationCheckStatus
    message: str = Field(min_length=1)


class MetricRecalculation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric_id: str = Field(min_length=1)
    category: ValidationCheckCategory
    reported_value: float | None = Field(default=None, allow_inf_nan=False)
    recomputed_value: float | None = Field(default=None, allow_inf_nan=False)
    absolute_error: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    relative_error: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    absolute_tolerance: float = Field(ge=0)
    relative_tolerance: float = Field(ge=0)
    status: ValidationCheckStatus
    message: str = Field(min_length=1)


class ValidationRequirementCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requirement: str = Field(min_length=1)
    scope: Literal["RESULT", "PAPER"] = "RESULT"
    status: ValidationCheckStatus
    evidence_refs: list[str] = Field(default_factory=list)
    message: str = Field(min_length=1)


class ValidationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    validation_id: UUID = Field(default_factory=uuid4)
    project_id: UUID
    problem_id: UUID
    model_id: UUID
    model_version: int = Field(ge=1)
    model_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    result_id: UUID
    solver_run_id: UUID
    execution_record_id: UUID
    validator_version: str = Field(min_length=1)
    status: ValidationStatus
    evidence: EvidenceChainReport
    variable_checks: list[VariableValidationCheck] = Field(default_factory=list)
    constraint_checks: list[ConstraintValidationCheck] = Field(default_factory=list)
    metric_recalculations: list[MetricRecalculation] = Field(default_factory=list)
    requirement_checks: list[ValidationRequirementCheck] = Field(default_factory=list)
    max_constraint_violation: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(min_length=4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def passing_report_is_complete(self) -> ValidationReport:
        checks: list[ValidationCheckStatus] = [
            *(item.status for item in self.variable_checks),
            *(item.status for item in self.constraint_checks),
            *(item.status for item in self.metric_recalculations),
            *(item.status for item in self.requirement_checks),
        ]
        if self.status is ValidationStatus.PASS:
            if not self.evidence.valid or self.errors:
                raise ValueError("PASS validation requires valid evidence and no errors")
            if any(item is not ValidationCheckStatus.PASS for item in checks):
                raise ValueError("PASS validation requires every declared check to pass")
        return self


class ValidationReportRef(BaseModel):
    validation_id: UUID
    model_id: UUID
    model_version: int = Field(ge=1)
    model_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    result_id: UUID
    status: ValidationStatus


class ExperimentStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIPPED = "SKIPPED"


class ExperimentReportStatus(StrEnum):
    PASS = "PASS"
    PARTIAL = "PARTIAL"
    FAIL = "FAIL"
    BLOCKED = "BLOCKED"


class ParameterPerturbation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parameter_id: str = Field(min_length=1)
    symbol: str = Field(min_length=1)
    source_ref: str = Field(min_length=1)
    baseline_value: float = Field(allow_inf_nan=False)
    fraction: float = Field(ge=-1, le=1, allow_inf_nan=False)
    perturbed_value: float = Field(allow_inf_nan=False)


class ExperimentRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: UUID = Field(default_factory=uuid4)
    experiment_type: str = Field(min_length=1)
    base_model_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    scenario_model_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    perturbations: list[ParameterPerturbation] = Field(min_length=1)
    solver: SolverName | None = None
    solver_status: SolverStatus | None = None
    execution_record_id: UUID | None = None
    objective_value: float | None = Field(default=None, allow_inf_nan=False)
    objective_change: float | None = Field(default=None, allow_inf_nan=False)
    objective_change_fraction: float | None = Field(default=None, allow_inf_nan=False)
    key_outputs: dict[str, float] = Field(default_factory=dict)
    fixed_decision_values: dict[str, float] = Field(default_factory=dict)
    feasible: bool | None = None
    max_constraint_violation: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    status: ExperimentStatus
    error: str | None = None
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def executed_pass_has_provenance(self) -> ExperimentRun:
        if self.status is ExperimentStatus.PASS:
            if (
                self.execution_record_id is None
                or self.solver is None
                or self.solver_status is None
            ):
                raise ValueError("PASS experiment requires solver and execution provenance")
            if self.feasible is not True:
                raise ValueError("PASS experiment requires an independently feasible result")
        return self


class SensitivityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    perturbation_fractions: list[float] = Field(default_factory=lambda: [0.05, 0.10, 0.20])
    parameter_symbols: list[str] = Field(default_factory=list)
    max_parameters: int = Field(default=5, ge=1, le=20)
    max_runs: int = Field(default=30, ge=2, le=200)
    solver_options: SolverOptions = Field(default_factory=SolverOptions)

    @field_validator("perturbation_fractions")
    @classmethod
    def fractions_are_unique_positive_and_ordered(cls, value: list[float]) -> list[float]:
        if not value or any(not math.isfinite(item) or item <= 0 or item > 1 for item in value):
            raise ValueError("sensitivity fractions must be finite values in (0, 1]")
        if value != sorted(set(value)):
            raise ValueError("sensitivity fractions must be unique and ascending")
        return value


class SensitivityReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sensitivity_id: UUID = Field(default_factory=uuid4)
    project_id: UUID
    problem_id: UUID
    model_id: UUID
    model_version: int = Field(ge=1)
    model_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    result_id: UUID
    validation_id: UUID
    baseline_objective: float | None = Field(default=None, allow_inf_nan=False)
    reviewed_report_id: UUID | None = None
    reviewed_replay_ids: dict[str, UUID] = Field(default_factory=dict)
    reviewed_metric_values: dict[str, float] = Field(default_factory=dict)
    baseline_responses: dict[str, float] = Field(default_factory=dict)
    response_ranges: dict[str, tuple[float, float]] = Field(default_factory=dict)
    config: SensitivityConfig
    experiments: list[ExperimentRun] = Field(default_factory=list)
    parameter_elasticities: dict[str, float] = Field(default_factory=dict)
    objective_min: float | None = Field(default=None, allow_inf_nan=False)
    objective_max: float | None = Field(default=None, allow_inf_nan=False)
    maximum_absolute_relative_change: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    successful_runs: int = Field(ge=0)
    failed_runs: int = Field(ge=0)
    status: ExperimentReportStatus
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def run_counts_match(self) -> SensitivityReport:
        successful = sum(item.status is ExperimentStatus.PASS for item in self.experiments)
        failed = sum(item.status is ExperimentStatus.FAIL for item in self.experiments)
        if (self.successful_runs, self.failed_runs) != (successful, failed):
            raise ValueError("sensitivity run counts must match experiments")
        if (
            self.status is ExperimentReportStatus.PASS
            and not self.reviewed_report_id
            and (not self.experiments or failed or successful != len(self.experiments))
        ):
            raise ValueError("PASS sensitivity requires every experiment to pass")
        if self.reviewed_report_id is not None and (
            self.experiments or self.baseline_objective is not None
        ):
            raise ValueError(
                "reviewed response evidence cannot masquerade as objective experiments"
            )
        return self


class SensitivityReportRef(BaseModel):
    sensitivity_id: UUID
    model_id: UUID
    model_version: int = Field(ge=1)
    model_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    validation_id: UUID
    status: ExperimentReportStatus


class RobustnessMethod(StrEnum):
    SCENARIO_ANALYSIS = "SCENARIO_ANALYSIS"
    WORST_CASE = "WORST_CASE"
    NOISE_PERTURBATION = "NOISE_PERTURBATION"
    MONTE_CARLO = "MONTE_CARLO"
    BOOTSTRAP = "BOOTSTRAP"


class RobustnessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: RobustnessMethod = RobustnessMethod.SCENARIO_ANALYSIS
    parameter_symbols: list[str] = Field(default_factory=list)
    scenario_fractions: list[float] = Field(default_factory=lambda: [-0.10, 0.10])
    sample_count: int = Field(default=20, ge=2, le=200)
    noise_fraction: float = Field(default=0.10, gt=0, le=1)
    random_seed: int = Field(default=20260828, ge=0, le=2**32 - 1)
    max_runs: int = Field(default=50, ge=2, le=200)
    solver_options: SolverOptions = Field(default_factory=SolverOptions)

    @field_validator("scenario_fractions")
    @classmethod
    def scenarios_are_finite_and_unique(cls, value: list[float]) -> list[float]:
        if not value or any(not math.isfinite(item) or item < -1 or item > 1 for item in value):
            raise ValueError("scenario fractions must be finite values in [-1, 1]")
        if len(value) != len(set(value)):
            raise ValueError("scenario fractions must be unique")
        return value


class RobustnessSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requested_runs: int = Field(ge=0)
    successful_runs: int = Field(ge=0)
    failed_runs: int = Field(ge=0)
    feasibility_rate: float | None = Field(default=None, ge=0, le=1)
    objective_mean: float | None = Field(default=None, allow_inf_nan=False)
    objective_std: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    objective_min: float | None = Field(default=None, allow_inf_nan=False)
    objective_max: float | None = Field(default=None, allow_inf_nan=False)
    objective_quantiles: dict[str, float] = Field(default_factory=dict)
    worst_case_experiment_id: UUID | None = None


class RobustnessReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    robustness_id: UUID = Field(default_factory=uuid4)
    project_id: UUID
    problem_id: UUID
    model_id: UUID
    model_version: int = Field(ge=1)
    model_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    result_id: UUID
    validation_id: UUID
    sensitivity_id: UUID
    baseline_objective: float | None = Field(default=None, allow_inf_nan=False)
    reviewed_report_id: UUID | None = None
    reviewed_replay_ids: dict[str, UUID] = Field(default_factory=dict)
    reviewed_metric_values: dict[str, float] = Field(default_factory=dict)
    baseline_responses: dict[str, float] = Field(default_factory=dict)
    response_ranges: dict[str, tuple[float, float]] = Field(default_factory=dict)
    method: RobustnessMethod
    config: RobustnessConfig
    experiments: list[ExperimentRun] = Field(default_factory=list)
    summary: RobustnessSummary
    status: ExperimentReportStatus
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def method_and_counts_match(self) -> RobustnessReport:
        if self.method is not self.config.method:
            raise ValueError("robustness method must match config")
        if self.summary.requested_runs != len(self.experiments):
            raise ValueError("robustness requested_runs must match experiments")
        if self.summary.successful_runs != sum(
            item.status is ExperimentStatus.PASS for item in self.experiments
        ):
            raise ValueError("robustness successful_runs must match experiments")
        if self.summary.failed_runs != sum(
            item.status is ExperimentStatus.FAIL for item in self.experiments
        ):
            raise ValueError("robustness failed_runs must match experiments")
        if (
            self.status is ExperimentReportStatus.PASS
            and not self.reviewed_report_id
            and (
                not self.experiments
                or self.summary.failed_runs
                or self.summary.successful_runs != len(self.experiments)
            )
        ):
            raise ValueError("PASS robustness requires every experiment to pass")
        if self.reviewed_report_id is not None and (
            self.experiments or self.baseline_objective is not None
        ):
            raise ValueError(
                "reviewed response evidence cannot masquerade as objective experiments"
            )
        return self


class RobustnessReportRef(BaseModel):
    robustness_id: UUID
    model_id: UUID
    model_version: int = Field(ge=1)
    model_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    sensitivity_id: UUID
    method: RobustnessMethod
    status: ExperimentReportStatus


class RedTeamSeverity(StrEnum):
    CRITICAL = "CRITICAL"
    MAJOR = "MAJOR"
    MINOR = "MINOR"


class RedTeamCategory(StrEnum):
    PROBLEM_INTERPRETATION = "PROBLEM_INTERPRETATION"
    ASSUMPTION = "ASSUMPTION"
    DATA = "DATA"
    MODEL_STRUCTURE = "MODEL_STRUCTURE"
    OBJECTIVE = "OBJECTIVE"
    CONSTRAINT = "CONSTRAINT"
    ALGORITHM = "ALGORITHM"
    PARAMETER = "PARAMETER"
    EXTREME_CASE = "EXTREME_CASE"
    OVERFITTING = "OVERFITTING"
    DATA_LEAKAGE = "DATA_LEAKAGE"
    SENSITIVITY = "SENSITIVITY"
    ROBUSTNESS = "ROBUSTNESS"
    RESULT_INTERPRETATION = "RESULT_INTERPRETATION"


class RedTeamFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    finding_id: str = Field(pattern=r"^RTF-[A-Za-z0-9_-]+$")
    severity: RedTeamSeverity
    category: RedTeamCategory
    title: str = Field(min_length=1)
    attack: str = Field(min_length=1)
    evidence_refs: list[str] = Field(min_length=1)
    affected_refs: list[str] = Field(default_factory=list)
    recommendation: str = Field(min_length=1)
    deterministic: bool = False
    resolved: bool = False


class RedTeamDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    findings: list[RedTeamFinding] = Field(default_factory=list)
    summary: str = Field(min_length=1)
    residual_risks: list[str] = Field(default_factory=list)


class RedTeamInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mathematical_model: MathematicalModel
    validation: ValidationReport
    sensitivity: SensitivityReport
    robustness: RobustnessReport
    user_guidance: list[str] = Field(default_factory=list)


class RedTeamReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    report_id: UUID = Field(default_factory=uuid4)
    project_id: UUID
    problem_id: UUID
    model_id: UUID
    model_version: int = Field(ge=1)
    model_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    result_id: UUID
    validation_id: UUID
    sensitivity_id: UUID
    robustness_id: UUID
    findings: list[RedTeamFinding] = Field(default_factory=list)
    critical_count: int = Field(ge=0)
    major_count: int = Field(ge=0)
    minor_count: int = Field(ge=0)
    summary: str = Field(min_length=1)
    residual_risks: list[str] = Field(default_factory=list)
    reviewer_agent_run_id: UUID
    review_is_mock: bool = False
    status: ValidationStatus
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def severity_counts_and_status_match(self) -> RedTeamReport:
        counts = {
            RedTeamSeverity.CRITICAL: self.critical_count,
            RedTeamSeverity.MAJOR: self.major_count,
            RedTeamSeverity.MINOR: self.minor_count,
        }
        for severity, expected in counts.items():
            if expected != sum(
                item.severity is severity and not item.resolved for item in self.findings
            ):
                raise ValueError("red-team severity counts must match unresolved findings")
        if self.status is ValidationStatus.PASS and self.critical_count:
            raise ValueError("red-team PASS cannot contain unresolved critical findings")
        if self.status is ValidationStatus.PASS and self.review_is_mock:
            raise ValueError("Mock red-team review cannot pass")
        return self


class RedTeamReportRef(BaseModel):
    report_id: UUID
    model_id: UUID
    model_version: int = Field(ge=1)
    model_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    robustness_id: UUID
    critical_count: int = Field(ge=0)
    status: ValidationStatus
    review_is_mock: bool


class RepairTargetType(StrEnum):
    ASSUMPTION = "ASSUMPTION"
    VARIABLE = "VARIABLE"
    PARAMETER = "PARAMETER"
    OBJECTIVE = "OBJECTIVE"
    CONSTRAINT = "CONSTRAINT"
    EQUATION = "EQUATION"
    ALGORITHM_REQUIREMENT = "ALGORITHM_REQUIREMENT"
    SOLVER_REQUIREMENT = "SOLVER_REQUIREMENT"
    LIMITATION = "LIMITATION"


class RepairAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_id: str = Field(pattern=r"^REPAIR-[A-Za-z0-9_-]+$")
    finding_ids: list[str] = Field(min_length=1)
    target_type: RepairTargetType
    target_ref: str = Field(min_length=1)
    description: str = Field(min_length=1)
    rationale: str = Field(min_length=1)


class ModelRepairDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    revised_model: MathematicalModelDraft
    actions: list[RepairAction] = Field(min_length=1)
    addressed_finding_ids: list[str] = Field(min_length=1)
    remaining_risks: list[str] = Field(default_factory=list)


class ModelRepairInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assigned_version: int = Field(ge=2)
    repair_cycle: int = Field(ge=1, le=3)
    current_model: MathematicalModel
    red_team_report: RedTeamReport
    validation: ValidationReport
    sensitivity: SensitivityReport
    robustness: RobustnessReport
    user_guidance: list[str] = Field(default_factory=list)


class ModelRepairOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    revised_model: MathematicalModel
    actions: list[RepairAction] = Field(min_length=1)
    addressed_finding_ids: list[str] = Field(min_length=1)
    remaining_risks: list[str] = Field(default_factory=list)


class RepairCycleStatus(StrEnum):
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    RESOLVED = "RESOLVED"
    HUMAN_REVIEW = "HUMAN_REVIEW"


class RepairCycleRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repair_id: UUID = Field(default_factory=uuid4)
    project_id: UUID
    problem_id: UUID
    stable_model_id: UUID
    source_model_version: int = Field(ge=1)
    source_model_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    target_model_version: int | None = Field(default=None, ge=2)
    target_model_digest: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    red_team_report_id: UUID
    repair_cycle: int = Field(ge=1, le=3)
    actions: list[RepairAction] = Field(default_factory=list)
    addressed_finding_ids: list[str] = Field(default_factory=list)
    remaining_risks: list[str] = Field(default_factory=list)
    agent_run_id: UUID
    is_mock: bool
    status: RepairCycleStatus
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def accepted_cycle_has_real_next_version(self) -> RepairCycleRecord:
        target_fields_present = (
            self.target_model_version is not None,
            self.target_model_digest is not None,
        )
        if len(set(target_fields_present)) != 1:
            raise ValueError("repair target version and digest must be present together")
        if self.target_model_version is not None and (
            self.target_model_version != self.source_model_version + 1
        ):
            raise ValueError("repair target must be the next model version")
        if self.status is RepairCycleStatus.ACCEPTED:
            if not all(target_fields_present):
                raise ValueError("accepted repair requires a target model revision")
            if self.is_mock:
                raise ValueError("Mock repair cannot be accepted")
            if self.target_model_digest == self.source_model_digest:
                raise ValueError("accepted repair must change mathematical content")
        return self


class RepairCycleRef(BaseModel):
    repair_id: UUID
    stable_model_id: UUID
    source_model_version: int = Field(ge=1)
    target_model_version: int | None = Field(default=None, ge=2)
    red_team_report_id: UUID
    repair_cycle: int = Field(ge=1, le=3)
    status: RepairCycleStatus
