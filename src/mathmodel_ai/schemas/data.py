from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mathmodel_ai.schemas.files import MultimodalAsset
from mathmodel_ai.schemas.problem_analysis import ProblemAnalysis


class DataSemanticType(StrEnum):
    INTEGER = "integer"
    CONTINUOUS = "continuous"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    CATEGORICAL = "categorical"
    IDENTIFIER = "identifier"
    TEXT = "text"
    UNKNOWN = "unknown"


class DataQualitySeverity(StrEnum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class DatasetLayer(StrEnum):
    RAW = "raw"
    PARSED = "parsed"
    CLEANED = "cleaned"
    PROCESSED = "processed"


class DatasetRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: UUID = Field(default_factory=uuid4)
    project_id: UUID
    problem_id: UUID
    source_file_id: UUID
    name: str = Field(min_length=1, max_length=255)
    sheet_name: str | None = Field(default=None, max_length=255)
    layer: DatasetLayer = DatasetLayer.PARSED
    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)
    columns: list[str]
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class NumericStatistics(BaseModel):
    count: int = Field(ge=0)
    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None
    median: float | None = None
    standard_deviation: float | None = None
    q1: float | None = None
    q3: float | None = None


class ValueCount(BaseModel):
    value: str
    count: int = Field(ge=1)


class OutlierSummary(BaseModel):
    method: str = "IQR_1.5"
    lower_bound: float | None = None
    upper_bound: float | None = None
    count: int = Field(ge=0)
    rate: float = Field(ge=0, le=1)


class ColumnProfile(BaseModel):
    name: str = Field(min_length=1)
    source_name: str = Field(min_length=1)
    physical_dtype: str = Field(min_length=1)
    semantic_type: DataSemanticType
    inferred_unit: str | None = None
    missing_count: int = Field(ge=0)
    missing_rate: float = Field(ge=0, le=1)
    unique_count: int = Field(ge=0)
    unique_rate: float = Field(ge=0, le=1)
    sample_values: list[Any] = Field(default_factory=list, max_length=10)
    numeric_statistics: NumericStatistics | None = None
    top_values: list[ValueCount] = Field(default_factory=list, max_length=10)
    outliers: OutlierSummary | None = None


class CorrelationRecord(BaseModel):
    left_column: str
    right_column: str
    pearson: float = Field(ge=-1, le=1)
    pair_count: int = Field(ge=2)


class QualityIssue(BaseModel):
    issue_id: str = Field(min_length=1)
    severity: DataQualitySeverity
    category: str = Field(min_length=1)
    message: str = Field(min_length=1)
    column: str | None = None
    evidence: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class FeatureRole(StrEnum):
    FEATURE = "feature"
    TARGET = "target"
    IDENTIFIER = "identifier"
    TIME = "time"
    SPATIAL = "spatial"


class FeatureCandidate(BaseModel):
    column: str
    role: FeatureRole
    confidence: float = Field(ge=0, le=1)
    reason: str = Field(min_length=1)


class DataProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile_id: UUID = Field(default_factory=uuid4)
    dataset_id: UUID
    source_file_id: UUID
    dataset_name: str
    sheet_name: str | None = None
    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)
    columns: list[ColumnProfile]
    duplicate_row_count: int = Field(ge=0)
    duplicate_row_rate: float = Field(ge=0, le=1)
    correlations: list[CorrelationRecord] = Field(default_factory=list)
    feature_candidates: list[FeatureCandidate] = Field(default_factory=list)
    quality_score: float = Field(ge=0, le=100)
    quality_issues: list[QualityIssue] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    deterministic: bool = True


class CrossDatasetRelationship(BaseModel):
    relationship_id: str = Field(min_length=1)
    left_dataset_id: UUID
    left_column: str
    right_dataset_id: UUID
    right_column: str
    relationship_type: str = Field(min_length=1)
    overlap_count: int = Field(ge=0)
    overlap_ratio: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    evidence: str = Field(min_length=1)


class DataProfileBundle(BaseModel):
    profiles: list[DataProfile] = Field(min_length=1)
    cross_dataset_relationships: list[CrossDatasetRelationship] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    deterministic: bool = True


class SemanticColumnRole(BaseModel):
    dataset_id: UUID
    column: str
    semantic_role: str = Field(min_length=1)
    interpreted_unit: str | None = None
    evidence: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


class DatasetUnderstanding(BaseModel):
    dataset_id: UUID
    purpose: str = Field(min_length=1)
    key_columns: list[SemanticColumnRole] = Field(default_factory=list)
    potential_features: list[str] = Field(default_factory=list)
    potential_targets: list[str] = Field(default_factory=list)
    data_quality_risks: list[str] = Field(default_factory=list)
    recommended_preprocessing: list[str] = Field(default_factory=list)


class DataUnderstanding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    datasets: list[DatasetUnderstanding] = Field(default_factory=list)
    interpreted_relationships: list[str] = Field(default_factory=list)
    problem_data_alignment: list[str] = Field(default_factory=list)
    quality_priorities: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    user_clarifications: list[str] = Field(default_factory=list)
    multimodal_observations: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def dataset_ids_are_unique(self) -> "DataUnderstanding":
        ids = [item.dataset_id for item in self.datasets]
        if len(ids) != len(set(ids)):
            raise ValueError("data understanding dataset ids must be unique")
        return self


class DataAgentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_problem: str = Field(min_length=1)
    problem_analysis: ProblemAnalysis | None = None
    profiles: list[DataProfile] = Field(default_factory=list)
    cross_dataset_relationships: list[CrossDatasetRelationship] = Field(default_factory=list)
    media_assets: list[MultimodalAsset] = Field(default_factory=list)
    user_guidance: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def requires_profile_or_media(self) -> "DataAgentInput":
        if not self.profiles and not self.media_assets:
            raise ValueError("data understanding requires a profile or media attachment")
        return self
