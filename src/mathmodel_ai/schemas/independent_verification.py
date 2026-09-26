from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mathmodel_ai.schemas.execution import ExecutionRecord
from mathmodel_ai.schemas.files import ArtifactRecord
from mathmodel_ai.schemas.program import GeneratedProgram
from mathmodel_ai.schemas.solver import SolverOptions, SolverStatus

Number = Annotated[float, Field(strict=True, allow_inf_nan=False)]
Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
Key = Annotated[str, Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,79}$")]


class FrozenContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class MetricKey(StrEnum):
    MAE = "mae"
    RMSE = "rmse"
    R2 = "r2"
    MAX_ERROR = "max_error"
    BRIER = "brier"
    LOG_LOSS = "log_loss"
    MEAN = "mean"
    MINIMUM = "minimum"
    MAXIMUM = "maximum"
    FINAL_VALUE = "final_value"
    OBJECTIVE = "objective"
    CONSTRAINT_MAX_VIOLATION = "constraint_max_violation"
    FEASIBLE = "feasible"
    MIP_GAP = "mip_gap"
    ALGEBRAIC_SCALAR = "algebraic_scalar"


class IndependentStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    INVALID = "INVALID"
    NOT_READY = "NOT_READY"
    RUNNING = "RUNNING"


class MetricSpec(FrozenContract):
    metric_id: Key
    key: MetricKey
    version: Literal["1"]
    required: bool = True
    series_key: Key | None = None
    value_symbol: Key | None = None
    reported_key: Key | None = None
    absolute_tolerance: Number = Field(default=1e-9, ge=0, le=0.01)
    relative_tolerance: Number = Field(default=1e-7, ge=0, le=0.01)
    lower_threshold: Number | None = None
    upper_threshold: Number | None = None
    exact: bool = False
    quantity: str | None = Field(default=None, min_length=1, max_length=500)
    calculation: str | None = Field(default=None, min_length=1, max_length=1000)
    inputs: list[Key] = Field(default_factory=list, max_length=100)
    unit: str | None = Field(default=None, min_length=1, max_length=100)
    tolerance_provenance: str | None = Field(default=None, min_length=1, max_length=1000)
    threshold_provenance: str | None = Field(default=None, min_length=1, max_length=1000)
    model_binding: Digest | None = None

    @model_validator(mode="after")
    def thresholds_are_ordered(self) -> "MetricSpec":
        if self.lower_threshold is not None and self.upper_threshold is not None:
            if self.lower_threshold > self.upper_threshold:
                raise ValueError("metric thresholds are inverted")
        if (
            self.key
            in {MetricKey.MEAN, MetricKey.MINIMUM, MetricKey.MAXIMUM, MetricKey.FINAL_VALUE}
            and self.series_key is None
        ):
            raise ValueError("series metric requires an explicit series_key")
        if self.key is MetricKey.ALGEBRAIC_SCALAR and self.value_symbol is None:
            raise ValueError("algebraic scalar metric requires an explicit value_symbol")
        if self.key is not MetricKey.ALGEBRAIC_SCALAR and self.value_symbol is not None:
            raise ValueError("value_symbol is only valid for an algebraic scalar metric")
        if self.series_key is not None and self.key not in {
            MetricKey.MEAN,
            MetricKey.MINIMUM,
            MetricKey.MAXIMUM,
            MetricKey.FINAL_VALUE,
        }:
            raise ValueError("series_key is only valid for a series metric")
        if (
            self.lower_threshold is not None or self.upper_threshold is not None
        ) and self.threshold_provenance is None:
            raise ValueError("metric threshold requires explicit provenance")
        return self


class RawMetricOutput(FrozenContract):
    """Raw numeric output only; reported aggregates are never calculator inputs."""

    variables: dict[Key, Number] = Field(default_factory=dict, max_length=10000)
    predictions: list[Number] = Field(default_factory=list, max_length=100000)
    series: dict[Key, Annotated[list[Number], Field(max_length=100000)]] = Field(
        default_factory=dict, max_length=100
    )
    reported: dict[Key, Number] = Field(default_factory=dict, max_length=100)
    best_bound: Number | None = None


class ObservationData(FrozenContract):
    observations: list[Number] = Field(min_length=1, max_length=100000)
    source_csv_sha256: Digest | None = None
    source_column: str | None = Field(default=None, min_length=1, max_length=255)
    positive_value: str | None = Field(default=None, min_length=1, max_length=255)
    negative_value: str | None = Field(default=None, min_length=1, max_length=255)

    @model_validator(mode="after")
    def binary_csv_provenance_is_complete(self) -> "ObservationData":
        fields = (
            self.source_csv_sha256,
            self.source_column,
            self.positive_value,
            self.negative_value,
        )
        if any(item is not None for item in fields):
            if any(item is None for item in fields):
                raise ValueError("binary CSV observation provenance must be complete")
            if self.positive_value == self.negative_value:
                raise ValueError("binary CSV observation labels must differ")
            if any(value not in (0.0, 1.0) for value in self.observations):
                raise ValueError("binary CSV observations must contain only zero or one")
        return self


class CsvObservationSpec(FrozenContract):
    """Reviewed binary target derived directly from an exact registered CSV."""

    source_csv_sha256: Digest
    source_column: str = Field(min_length=1, max_length=255)
    positive_value: str = Field(min_length=1, max_length=255)
    negative_value: str = Field(min_length=1, max_length=255)

    @model_validator(mode="after")
    def distinct_codes(self) -> "CsvObservationSpec":
        if self.positive_value == self.negative_value:
            raise ValueError("binary CSV observation labels must differ")
        return self


class DynamicReplaySpec(FrozenContract):
    """Explicit ODE binding; never infer derivative meaning from symbol spelling."""

    equation_by_state: dict[Key, Key] = Field(min_length=1, max_length=100)
    initial_state: dict[Key, Number] = Field(min_length=1, max_length=100)
    start: Number = 0.0
    stop: Number = Field(gt=0)
    samples: int = Field(default=101, ge=2, le=10001)
    rtol: Number = Field(default=1e-8, gt=0, le=1e-3)
    atol: Number = Field(default=1e-10, gt=0, le=1e-3)

    @model_validator(mode="after")
    def explicit_state_and_time(self) -> "DynamicReplaySpec":
        if set(self.equation_by_state) != set(self.initial_state) or self.stop <= self.start:
            raise ValueError("dynamic replay requires exact state bindings and ordered time")
        return self


class ScenarioSpec(FrozenContract):
    scenario_id: Key
    version: Literal["1"]
    required: bool = True
    parameter_values: dict[Key, Number] = Field(default_factory=dict, max_length=100)
    decision_values: dict[Key, Number] = Field(default_factory=dict, max_length=100)
    noise_fraction: Number = Field(default=0.0, ge=0, le=1)
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)
    timeout_seconds: Number = Field(default=30.0, gt=0, le=120)
    dynamic: DynamicReplaySpec | None = None
    metrics: list[MetricSpec] = Field(min_length=1, max_length=100)
    baseline: str | None = Field(default=None, min_length=1, max_length=1000)
    perturbation: str | None = Field(default=None, min_length=1, max_length=1000)
    reason: str | None = Field(default=None, min_length=1, max_length=1000)
    input_changes: list[Key] = Field(default_factory=list, max_length=100)
    comparison_quantity: str | None = Field(default=None, min_length=1, max_length=1000)
    acceptance_criterion: str | None = Field(default=None, min_length=1, max_length=1000)
    criterion_provenance: str | None = Field(default=None, min_length=1, max_length=1000)

    @model_validator(mode="after")
    def stochastic_seed_required(self) -> "ScenarioSpec":
        if self.noise_fraction and (self.seed is None or not self.parameter_values):
            raise ValueError("stochastic input perturbation requires seed and named parameters")
        _unique_metrics(self.metrics)
        return self


def _unique_metrics(metrics: list[MetricSpec]) -> None:
    if len({m.metric_id for m in metrics}) != len(metrics):
        raise ValueError("metric identities must be unique")
    if not any(m.required for m in metrics):
        raise ValueError("at least one required metric must be specified")


class VerificationPlan(FrozenContract):
    plan_id: UUID = Field(default_factory=uuid4)
    attempt_id: UUID
    result_id: UUID
    version: Literal["1"]
    source_artifact_id: UUID
    source_sha256: Digest
    observation_file_id: UUID | None = None
    observation_sha256: Digest | None = None
    csv_observation: CsvObservationSpec | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    metrics: list[MetricSpec] = Field(min_length=1, max_length=100)
    scenarios: list[ScenarioSpec] = Field(min_length=1, max_length=30)
    scientific_scope: str = Field(min_length=10, max_length=2000)

    @model_validator(mode="after")
    def mandatory_scope(self) -> "VerificationPlan":
        _unique_metrics(self.metrics)
        if len({s.scenario_id for s in self.scenarios}) != len(self.scenarios):
            raise ValueError("scenario identities must be unique")
        if not any(s.required for s in self.scenarios):
            raise ValueError("at least one required replay scenario is necessary")
        if (self.observation_file_id is None) != (self.observation_sha256 is None):
            raise ValueError("observations require both original file identity and digest")
        if self.csv_observation is not None and (
            self.observation_sha256 != self.csv_observation.source_csv_sha256
            or self.observation_file_id is None
        ):
            raise ValueError("CSV observation must bind its exact registered source file")
        return self


class VerifiedMetric(FrozenContract):
    metric_id: Key
    key: MetricKey
    version: Literal["1"] = "1"
    reported: Number | None = None
    verified: Number | None = None
    delta: Number | None = None
    status: IndependentStatus
    source_digest: Digest
    calculator_digest: Digest
    error: str | None = None


class ValidationRequirementBinding(FrozenContract):
    """Reviewed, exact evidence mapping for one MathematicalModel obligation."""

    requirement: str = Field(min_length=1, max_length=2000)
    scope: Literal["RESULT", "PAPER"] = "RESULT"
    metric_ids: list[Key] = Field(default_factory=list, max_length=100)
    scenario_ids: list[Key] = Field(default_factory=list, max_length=30)
    model_evidence_refs: list[str] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def requires_explicit_evidence(self) -> "ValidationRequirementBinding":
        if not self.metric_ids and not self.scenario_ids and not self.model_evidence_refs:
            raise ValueError("validation requirement binding requires explicit evidence")
        if len(self.metric_ids) != len(set(self.metric_ids)):
            raise ValueError("validation requirement metric bindings must be unique")
        if len(self.scenario_ids) != len(set(self.scenario_ids)):
            raise ValueError("validation requirement scenario bindings must be unique")
        if len(self.model_evidence_refs) != len(set(self.model_evidence_refs)):
            raise ValueError("validation requirement model evidence bindings must be unique")
        if self.scope == "PAPER" and (self.metric_ids or self.scenario_ids):
            raise ValueError("paper-scoped requirements must bind model evidence, not results")
        return self


class VerificationRequirements(FrozenContract):
    """Evaluation-only policy supplied by reviewed case definitions, never an Agent."""

    version: Literal["2"]
    review_status: Literal["REVIEWED"]
    production_eligible: Literal[True]
    benchmark_id: str = Field(pattern=r"^BENCH-[A-Za-z0-9_-]+$")
    manifest_digest: Digest
    problem_sha256: Digest | None = None
    model_digest: Digest
    model_contract_digest: Digest | None = None
    red_team_report_digest: Digest | None = None
    model_jury_report_digest: Digest | None = None
    observation_sha256: Digest | None = None
    csv_observation: CsvObservationSpec | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    metrics: list[MetricSpec] = Field(min_length=1, max_length=100)
    scenarios: list[ScenarioSpec] = Field(min_length=1, max_length=30)
    validation_requirement_bindings: list[ValidationRequirementBinding] = Field(
        default_factory=list, max_length=100
    )
    scientific_scope: str = Field(min_length=10, max_length=2000)
    unresolved_obligations: list[str] = Field(default_factory=list, max_length=0)

    @model_validator(mode="after")
    def mandatory_requirements(self) -> "VerificationRequirements":
        if self.csv_observation is not None and self.observation_sha256 is not None:
            raise ValueError("choose one reviewed observation source")
        _unique_metrics(self.metrics)
        if len({s.scenario_id for s in self.scenarios}) != len(self.scenarios):
            raise ValueError("duplicate scenario requirement")
        if not any(s.required for s in self.scenarios):
            raise ValueError("required scenario is missing")
        if len({item.requirement for item in self.validation_requirement_bindings}) != len(
            self.validation_requirement_bindings
        ):
            raise ValueError("duplicate validation requirement binding")
        top_metric_ids = {item.metric_id for item in self.metrics}
        scenario_ids = {item.scenario_id for item in self.scenarios}
        scenario_metric_ids = {
            item.metric_id for scenario in self.scenarios for item in scenario.metrics
        }
        all_metric_ids = [
            *(item.metric_id for item in self.metrics),
            *(item.metric_id for scenario in self.scenarios for item in scenario.metrics),
        ]
        bound_metric_ids = {
            metric_id
            for binding in self.validation_requirement_bindings
            for metric_id in binding.metric_ids
        }
        if any(all_metric_ids.count(metric_id) != 1 for metric_id in bound_metric_ids):
            raise ValueError("bound reviewed metric identity must resolve exactly once")
        if any(
            not set(binding.metric_ids) <= top_metric_ids | scenario_metric_ids
            or not set(binding.scenario_ids) <= scenario_ids
            for binding in self.validation_requirement_bindings
        ):
            raise ValueError("validation requirement binding references undeclared evidence")
        for metric in [*self.metrics, *(m for s in self.scenarios for m in s.metrics)]:
            if (
                metric.quantity is None
                or metric.calculation is None
                or not metric.inputs
                or metric.unit is None
                or metric.tolerance_provenance is None
                or metric.model_binding != self.model_digest
            ):
                raise ValueError("reviewed metric metadata or model binding is incomplete")
        for scenario in self.scenarios:
            if (
                scenario.baseline is None
                or scenario.perturbation is None
                or scenario.reason is None
                or scenario.comparison_quantity is None
                or scenario.acceptance_criterion is None
                or scenario.criterion_provenance is None
            ):
                raise ValueError("reviewed scenario metadata is incomplete")
            changed = set(scenario.parameter_values) | set(scenario.decision_values)
            if set(scenario.input_changes) != changed:
                raise ValueError("scenario input_changes must exactly name changed inputs")
        return self

    def matches_observation(self, plan: VerificationPlan) -> bool:
        """Require the same reviewed target source at every evidence handoff."""
        expected_digest = (
            self.csv_observation.source_csv_sha256
            if self.csv_observation is not None
            else self.observation_sha256
        )
        return (
            plan.csv_observation == self.csv_observation
            and plan.observation_sha256 == expected_digest
            and (plan.observation_file_id is None) == (expected_digest is None)
        )


class MetricBatch(FrozenContract):
    metrics: list[VerifiedMetric]


class IndependentVerificationView(FrozenContract):
    attempt_id: UUID
    status: IndependentStatus
    plan_id: UUID | None = None
    plan: VerificationPlan | None = None
    scenario_ids: list[Key] = Field(default_factory=list)
    report: "IndependentVerificationReport | None" = None
    blockers: list[str] = Field(default_factory=list)


class ReplayRecord(FrozenContract):
    replay_id: UUID = Field(default_factory=uuid4)
    scenario_id: Key
    scenario_digest: Digest
    input_digest: Digest
    input_parameters: dict[Key, Number]
    generator_version: str
    execution: ExecutionRecord
    program: GeneratedProgram | None = None
    solver_options: SolverOptions | None = None
    solver_status: SolverStatus | None = None
    artifacts: list[ArtifactRecord]
    output_artifact_id: UUID | None = None
    output_digest: Digest | None = None
    status: IndependentStatus
    metrics: list[VerifiedMetric] = Field(default_factory=list)
    error: str | None = None


class IndependentVerificationReport(FrozenContract):
    report_id: UUID = Field(default_factory=uuid4)
    attempt_id: UUID
    plan_id: UUID
    plan_digest: Digest
    result_id: UUID
    status: IndependentStatus
    metrics: list[VerifiedMetric]
    replays: list[ReplayRecord]
    required_metrics: int
    passed_metrics: int
    required_scenarios: int
    passed_scenarios: int
    errors: list[str]
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ReviewedValidationEvidence(FrozenContract):
    """Complete immutable input needed to audit reviewed evidence inside Phase 5."""

    requirements: VerificationRequirements
    plan: VerificationPlan
    report: IndependentVerificationReport
