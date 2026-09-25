from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from math import isfinite
from pathlib import PurePosixPath
from typing import Literal
from urllib.parse import urlparse
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from mathmodel_ai.paper.hashing import sha256_json
from mathmodel_ai.schemas.provider_registry import (
    EndpointTrustLevel,
    ModelIdentityConfidence,
    ProviderProtocol,
)

SHA256_PATTERN = r"^[a-f0-9]{64}$"
COMMIT_PATTERN = r"^[a-f0-9]{40}$"


class BenchmarkRunStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SELF_TEST_READY = "SELF_TEST_READY"
    NOT_READY = "NOT_READY"
    CANCELLED_BY_HUMAN = "CANCELLED_BY_HUMAN"


class BenchmarkCaseStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    PASS = "PASS"
    PASS_WITH_WARNINGS = "PASS_WITH_WARNINGS"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    FAIL = "FAIL"
    BLOCKED_ENVIRONMENT = "BLOCKED_ENVIRONMENT"
    BLOCKED_RULE_POLICY = "BLOCKED_RULE_POLICY"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    INVALID_BENCHMARK_SETUP = "INVALID_BENCHMARK_SETUP"
    CANCELLED_BY_HUMAN = "CANCELLED_BY_HUMAN"


class LiveValidationStatus(StrEnum):
    NOT_RUN = "NOT_RUN"
    PASS = "PASS"
    FAIL = "FAIL"
    BLOCKED = "BLOCKED"


class BenchmarkPhase(StrEnum):
    SOLVE = "SOLVE"
    EVALUATION = "EVALUATION"


class BenchmarkResourceRole(StrEnum):
    PROBLEM = "PROBLEM"
    ATTACHMENT = "ATTACHMENT"
    RULES = "RULES"
    EXTERNAL_DATA = "EXTERNAL_DATA"
    REFERENCE_SOLUTION = "REFERENCE_SOLUTION"
    JUDGE_COMMENTARY = "JUDGE_COMMENTARY"
    HUMAN_NOTES = "HUMAN_NOTES"


class ModelingCategory(StrEnum):
    DATA_PREDICTION = "DATA_PREDICTION"
    OPTIMIZATION_DECISION = "OPTIMIZATION_DECISION"
    SIMULATION_EVALUATION = "SIMULATION_EVALUATION"
    MULTI_STAGE = "MULTI_STAGE"


class FailureCategory(StrEnum):
    PROBLEM_UNDERSTANDING = "PROBLEM_UNDERSTANDING"
    DATA_EXTRACTION = "DATA_EXTRACTION"
    MODEL_SELECTION = "MODEL_SELECTION"
    MATHEMATICAL_MODEL = "MATHEMATICAL_MODEL"
    CODE_GENERATION = "CODE_GENERATION"
    SOLVER = "SOLVER"
    VALIDATION = "VALIDATION"
    SENSITIVITY = "SENSITIVITY"
    ROBUSTNESS = "ROBUSTNESS"
    RED_TEAM = "RED_TEAM"
    REPAIR = "REPAIR"
    LITERATURE = "LITERATURE"
    CITATION = "CITATION"
    PAPER = "PAPER"
    COMPETITION_RULE = "COMPETITION_RULE"
    SUBMISSION = "SUBMISSION"
    INFRASTRUCTURE = "INFRASTRUCTURE"
    PROVIDER = "PROVIDER"


class FailureSeverity(StrEnum):
    P0 = "P0"
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"


class HumanInterventionType(StrEnum):
    CREDENTIAL_SETUP = "CREDENTIAL_SETUP"
    RULE_VERIFICATION = "RULE_VERIFICATION"
    DATA_CORRECTION = "DATA_CORRECTION"
    MODEL_CORRECTION = "MODEL_CORRECTION"
    CITATION_REVIEW = "CITATION_REVIEW"
    PAPER_EDIT = "PAPER_EDIT"
    SUBMISSION_REVIEW = "SUBMISSION_REVIEW"
    ENVIRONMENT_RECOVERY = "ENVIRONMENT_RECOVERY"


class BenchmarkMetricKind(StrEnum):
    QUALITY = "QUALITY"
    RUNTIME_SECONDS = "RUNTIME_SECONDS"
    TOKEN_COUNT = "TOKEN_COUNT"
    COST = "COST"
    COUNT = "COUNT"
    BOOLEAN = "BOOLEAN"


class BenchmarkDeadlineMode(StrEnum):
    NORMAL = "NORMAL"
    PRIORITY = "PRIORITY"
    FINALIZATION = "FINALIZATION"
    MODEL_FREEZE = "MODEL_FREEZE"
    SUBMISSION_MODE = "SUBMISSION_MODE"


class BenchmarkResource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resource_id: str = Field(pattern=r"^RESOURCE-[A-Za-z0-9_-]+$")
    phase: BenchmarkPhase
    role: BenchmarkResourceRole
    source_url: str = Field(min_length=1)
    sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    media_type: str = Field(min_length=1)
    local_filename: str = Field(min_length=1)
    expected_size_bytes: int | None = Field(default=None, ge=1)
    distribution_notes: str = Field(min_length=1)

    @field_validator("source_url")
    @classmethod
    def source_is_https(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("benchmark sources must use credential-free HTTPS URLs")
        return value

    @field_validator("local_filename")
    @classmethod
    def filename_is_bounded(cls, value: str) -> str:
        normalized = value.replace("\\", "/")
        path = PurePosixPath(normalized)
        if path.is_absolute() or len(path.parts) != 1 or path.name in {"", ".", ".."}:
            raise ValueError("benchmark resource filenames must be plain bounded names")
        if ":" in normalized or any(ord(character) < 32 for character in normalized):
            raise ValueError("benchmark resource filename contains forbidden characters")
        return normalized


class ExpectedStructuralFact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fact_id: str = Field(pattern=r"^FACT-[A-Za-z0-9_-]+$")
    resource_id: str = Field(pattern=r"^RESOURCE-[A-Za-z0-9_-]+$")
    field: str = Field(min_length=1)
    expected: str | int | float | bool
    unit: str | None = None
    source_location: str = Field(min_length=1)


class CausalHoldoutPolicy(BaseModel):
    """Benchmark-owned input split; never supplied by a solve agent."""

    model_config = ConfigDict(extra="forbid")

    version: Literal["1"]
    resource_id: str = Field(pattern=r"^RESOURCE-[A-Za-z0-9_-]+$")
    source_sha256: str = Field(pattern=SHA256_PATTERN)
    group_column: str = Field(min_length=1)
    condition_column: str = Field(min_length=1)
    outcome_column: str = Field(min_length=1)
    positive_value: str = Field(min_length=1)
    negative_value: str = Field(min_length=1)
    history_window: int = Field(default=8, ge=1, le=1000)
    fraction: float = Field(gt=0, lt=1)
    salt: str = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def columns_and_codes_are_distinct(self) -> CausalHoldoutPolicy:
        if len({self.group_column, self.condition_column, self.outcome_column}) != 3:
            raise ValueError("causal holdout columns must be distinct")
        if self.positive_value == self.negative_value:
            raise ValueError("causal holdout outcome codes must differ")
        return self


class GroundTruthPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hard_facts: list[str] = Field(default_factory=list)
    required_outputs: list[str] = Field(min_length=1)
    known_constraints: list[str] = Field(default_factory=list)
    reference_ranges: list[str] = Field(default_factory=list)
    reasonable_model_families: list[str] = Field(default_factory=list)
    known_invalid_approaches: list[str] = Field(default_factory=list)
    historical_quality_signals: list[str] = Field(default_factory=list)


class BenchmarkCaseManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manifest_version: int = Field(default=1, ge=1)
    benchmark_id: str = Field(pattern=r"^BENCH-[A-Za-z0-9_-]+$")
    competition: str = Field(min_length=1)
    year: int = Field(ge=1900, le=3000)
    problem_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    modeling_category: ModelingCategory
    difficulty: str = Field(pattern=r"^(MODERATE|DIFFICULT|MULTI_STAGE)$")
    resources: list[BenchmarkResource] = Field(min_length=1)
    expected_structural_facts: list[ExpectedStructuralFact] = Field(default_factory=list)
    requires_external_data: bool
    requires_literature: bool
    requires_solver: bool
    literature_queries: list[str] = Field(default_factory=list)
    ground_truth_policy: GroundTruthPolicy
    license_or_distribution_notes: str = Field(min_length=1)

    @model_validator(mode="after")
    def resources_are_isolated_and_complete(self) -> BenchmarkCaseManifest:
        ids = [item.resource_id for item in self.resources]
        if len(ids) != len(set(ids)):
            raise ValueError("benchmark resource identifiers must be unique")
        solve_resources = [item for item in self.resources if item.phase is BenchmarkPhase.SOLVE]
        if not any(item.role is BenchmarkResourceRole.PROBLEM for item in solve_resources):
            raise ValueError("solve resources must include one official problem")
        forbidden = {
            BenchmarkResourceRole.REFERENCE_SOLUTION,
            BenchmarkResourceRole.JUDGE_COMMENTARY,
            BenchmarkResourceRole.HUMAN_NOTES,
        }
        if any(item.role in forbidden for item in solve_resources):
            raise ValueError("evaluation material cannot be exposed to blind solve")
        if self.requires_literature and not self.literature_queries:
            raise ValueError("literature-required cases must declare live search queries")
        known_ids = set(ids)
        if any(item.resource_id not in known_ids for item in self.expected_structural_facts):
            raise ValueError("structural fact references an unknown resource")
        return self


class BenchmarkPricing(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str = Field(min_length=1)
    currency: str = Field(default="USD", min_length=3, max_length=3)
    input_per_million: float = Field(default=0, ge=0)
    cached_input_per_million: float = Field(default=0, ge=0)
    output_per_million: float = Field(default=0, ge=0)
    reasoning_per_million: float = Field(default=0, ge=0)


class BenchmarkBudget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_case_cost: float | None = Field(default=None, gt=0)
    max_total_cost: float | None = Field(default=None, gt=0)
    max_wall_time_seconds: float | None = Field(default=None, gt=0)


class BenchmarkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config_version: str = Field(default="8.0.0", min_length=1)
    provider: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{1,99}$")
    model: str = Field(min_length=1)
    provider_id: str | None = Field(default=None, pattern=r"^[a-z0-9][a-z0-9._-]{1,99}$")
    provider_config_digest: str | None = Field(default=None, pattern=SHA256_PATTERN)
    model_id: str | None = Field(default=None, pattern=r"^[a-z0-9][a-z0-9._-]{1,99}$")
    model_config_digest: str | None = Field(default=None, pattern=SHA256_PATTERN)
    protocol: ProviderProtocol | None = None
    endpoint_trust: EndpointTrustLevel | None = None
    model_identity_confidence: ModelIdentityConfidence | None = None
    reasoning_tier: str = Field(min_length=1)
    temperature: float = Field(default=0, ge=0, le=2)
    random_seed: int = Field(default=20240801, ge=0)
    allow_live_literature: bool = True
    enable_independent_evaluator: bool = True
    deadline_modes: list[BenchmarkDeadlineMode] = Field(
        default_factory=lambda: [BenchmarkDeadlineMode.NORMAL]
    )
    pricing: BenchmarkPricing
    budget: BenchmarkBudget = Field(default_factory=BenchmarkBudget)


class BenchmarkRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: UUID = Field(default_factory=uuid4)
    code_commit: str = Field(pattern=COMMIT_PATTERN)
    source_tree_digest: str = Field(pattern=SHA256_PATTERN)
    working_tree_dirty: bool
    config: BenchmarkConfig
    case_manifest_digests: dict[str, str] = Field(min_length=1)
    competition_profile_digests: dict[str, str] = Field(default_factory=dict)
    status: BenchmarkRunStatus = BenchmarkRunStatus.PENDING
    live_provider_status: LiveValidationStatus = LiveValidationStatus.NOT_RUN
    live_literature_status: LiveValidationStatus = LiveValidationStatus.NOT_RUN
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    run_digest: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def completion_is_consistent(self) -> BenchmarkRun:
        terminal = {
            BenchmarkRunStatus.SELF_TEST_READY,
            BenchmarkRunStatus.NOT_READY,
            BenchmarkRunStatus.CANCELLED_BY_HUMAN,
        }
        if (self.status in terminal) != (self.finished_at is not None):
            raise ValueError("terminal benchmark runs require exactly one finish timestamp")
        return self

    @field_validator("case_manifest_digests", "competition_profile_digests")
    @classmethod
    def digest_maps_are_valid(cls, values: dict[str, str]) -> dict[str, str]:
        if any(not key or re_fullmatch_sha256(value) is False for key, value in values.items()):
            raise ValueError("benchmark identity maps require non-empty keys and SHA-256 values")
        return values


class BenchmarkAttempt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attempt_id: UUID = Field(default_factory=uuid4)
    run_id: UUID
    benchmark_id: str = Field(pattern=r"^BENCH-[A-Za-z0-9_-]+$")
    manifest_digest: str = Field(pattern=SHA256_PATTERN)
    attempt_number: int = Field(ge=1)
    status: BenchmarkCaseStatus = BenchmarkCaseStatus.PENDING
    project_id: UUID | None = None
    official: bool = True
    solve_input_digest: str = Field(pattern=SHA256_PATTERN)
    provider_is_live: bool = False
    literature_is_live: bool = False
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    attempt_digest: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def completion_is_consistent(self) -> BenchmarkAttempt:
        if self.status in {BenchmarkCaseStatus.PENDING, BenchmarkCaseStatus.RUNNING}:
            if self.finished_at is not None:
                raise ValueError("unfinished attempt cannot have a finish timestamp")
        elif self.finished_at is None:
            raise ValueError("terminal attempt requires a finish timestamp")
        return self


class BenchmarkMetric(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric_id: UUID = Field(default_factory=uuid4)
    attempt_id: UUID
    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    kind: BenchmarkMetricKind
    value: float
    unit: str = Field(min_length=1)
    evidence_ref: str = Field(min_length=1)
    deterministic: bool
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metric_digest: str = Field(pattern=SHA256_PATTERN)

    @field_validator("value")
    @classmethod
    def value_is_finite(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("benchmark metrics must be finite")
        return value


class BenchmarkFailure(BaseModel):
    model_config = ConfigDict(extra="forbid")

    failure_id: UUID = Field(default_factory=uuid4)
    attempt_id: UUID
    stage: str = Field(min_length=1)
    category: FailureCategory
    severity: FailureSeverity
    root_cause: str = Field(min_length=1)
    evidence_refs: list[str] = Field(min_length=1)
    reproducible: bool
    generic_issue: bool
    proposed_fix: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    failure_digest: str = Field(pattern=SHA256_PATTERN)


class BenchmarkHumanIntervention(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intervention_id: UUID = Field(default_factory=uuid4)
    attempt_id: UUID
    intervention_type: HumanInterventionType
    reason: str = Field(min_length=1)
    actor: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    intervention_digest: str = Field(pattern=SHA256_PATTERN)


class BenchmarkDimensionScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dimension: str = Field(min_length=1)
    score: float = Field(ge=0)
    maximum: float = Field(gt=0)
    evidence_refs: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def score_is_bounded(self) -> BenchmarkDimensionScore:
        if self.score > self.maximum:
            raise ValueError("dimension score exceeds its configured maximum")
        return self


class BenchmarkCaseResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    result_id: UUID = Field(default_factory=uuid4)
    attempt_id: UUID
    benchmark_id: str = Field(pattern=r"^BENCH-[A-Za-z0-9_-]+$")
    status: BenchmarkCaseStatus
    dimensions: list[BenchmarkDimensionScore] = Field(min_length=1)
    score: float = Field(ge=0, le=100)
    hard_failures: list[str] = Field(default_factory=list)
    verified_result_id: UUID | None = None
    paper_id: UUID | None = None
    paper_version: int | None = Field(default=None, ge=1)
    submission_id: UUID | None = None
    submission_status: str | None = None
    artifact_refs: list[str] = Field(default_factory=list)
    human_intervention_count: int = Field(ge=0)
    provider_call_count: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    estimated_cost: float = Field(ge=0)
    wall_time_seconds: float = Field(ge=0)
    result_digest: str = Field(pattern=SHA256_PATTERN)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class BenchmarkAcceptance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: BenchmarkRunStatus
    real_cases_attempted: int = Field(ge=0)
    successful_cases: int = Field(ge=0)
    verified_real_profiles: int = Field(ge=0)
    live_provider_status: LiveValidationStatus
    live_literature_status: LiveValidationStatus
    hidden_p0_count: int = Field(ge=0)
    reasons: list[str] = Field(default_factory=list)


class BenchmarkReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run: BenchmarkRun
    attempts: list[BenchmarkAttempt]
    results: list[BenchmarkCaseResult]
    metrics: list[BenchmarkMetric]
    failures: list[BenchmarkFailure]
    human_interventions: list[BenchmarkHumanIntervention]
    acceptance: BenchmarkAcceptance
    report_digest: str = Field(pattern=SHA256_PATTERN)


class BenchmarkRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_ids: list[str] = Field(min_length=1)
    config: BenchmarkConfig

    @field_validator("case_ids")
    @classmethod
    def case_ids_are_unique(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("benchmark case identifiers must be unique")
        return values


class BenchmarkCaseSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    benchmark_id: str = Field(pattern=r"^BENCH-[A-Za-z0-9_-]+$")
    title: str = Field(min_length=1)
    competition: str = Field(min_length=1)
    year: int = Field(ge=1900, le=3000)
    modeling_category: ModelingCategory
    difficulty: str
    requires_literature: bool
    requires_solver: bool


def benchmark_manifest_digest(manifest: BenchmarkCaseManifest) -> str:
    return sha256_json(manifest)


def benchmark_run_digest(run: BenchmarkRun) -> str:
    payload = run.model_dump(
        mode="json",
        exclude={
            "run_digest",
            "status",
            "live_provider_status",
            "live_literature_status",
            "finished_at",
        },
    )
    config = payload["config"]
    if isinstance(config, dict):
        for field in (
            "provider_id",
            "provider_config_digest",
            "model_id",
            "model_config_digest",
            "protocol",
            "endpoint_trust",
            "model_identity_confidence",
        ):
            if config.get(field) is None:
                config.pop(field, None)
    return sha256_json(payload)


def benchmark_attempt_digest(attempt: BenchmarkAttempt) -> str:
    return sha256_json(attempt.model_dump(mode="json", exclude={"attempt_digest"}))


def benchmark_metric_digest(metric: BenchmarkMetric) -> str:
    return sha256_json(metric.model_dump(mode="json", exclude={"metric_digest"}))


def benchmark_failure_digest(failure: BenchmarkFailure) -> str:
    return sha256_json(failure.model_dump(mode="json", exclude={"failure_digest"}))


def benchmark_intervention_digest(intervention: BenchmarkHumanIntervention) -> str:
    return sha256_json(intervention.model_dump(mode="json", exclude={"intervention_digest"}))


def benchmark_result_digest(result: BenchmarkCaseResult) -> str:
    return sha256_json(result.model_dump(mode="json", exclude={"result_digest"}))


def benchmark_report_digest(report: BenchmarkReport) -> str:
    return sha256_json(report.model_dump(mode="json", exclude={"report_digest"}))


def re_fullmatch_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)
