from __future__ import annotations

import math
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SHA256_PATTERN = r"^[a-f0-9]{64}$"
DOCUMENT_ID_PATTERN = r"^(SEC|EQ|FIG|TAB|REF|CLAIM)-[A-Za-z0-9_-]+$"


class EvidenceType(StrEnum):
    PROBLEM_FACT = "PROBLEM_FACT"
    DATA_FACT = "DATA_FACT"
    ASSUMPTION = "ASSUMPTION"
    DERIVATION = "DERIVATION"
    MODEL = "MODEL"
    EQUATION = "EQUATION"
    RESULT = "RESULT"
    VALIDATION = "VALIDATION"
    SENSITIVITY = "SENSITIVITY"
    ROBUSTNESS = "ROBUSTNESS"
    RED_TEAM = "RED_TEAM"
    REPAIR = "REPAIR"
    LITERATURE = "LITERATURE"
    FIGURE_DATA = "FIGURE_DATA"
    TABLE_DATA = "TABLE_DATA"


class EvidenceVerificationStatus(StrEnum):
    VERIFIED = "VERIFIED"
    UNVERIFIED = "UNVERIFIED"
    REJECTED = "REJECTED"
    MOCK = "MOCK"


class EvidenceProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_refs: list[str] = Field(min_length=1)
    source_hash: str = Field(pattern=SHA256_PATTERN)
    model_id: UUID | None = None
    model_version: int | None = Field(default=None, ge=1)
    model_digest: str | None = Field(default=None, pattern=SHA256_PATTERN)
    result_id: UUID | None = None
    solver_run_id: UUID | None = None
    execution_record_id: UUID | None = None
    validation_id: UUID | None = None
    sensitivity_id: UUID | None = None
    robustness_id: UUID | None = None
    red_team_report_id: UUID | None = None
    is_mock: bool = False


class EvidenceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: UUID = Field(default_factory=uuid4)
    project_id: UUID
    problem_id: UUID
    evidence_type: EvidenceType
    source_type: str = Field(min_length=1, max_length=64)
    source_id: str = Field(min_length=1, max_length=255)
    source_version: str = Field(min_length=1, max_length=100)
    content_summary: str = Field(min_length=1)
    structured_payload: dict[str, Any]
    verified: bool
    verification_status: EvidenceVerificationStatus
    provenance: EvidenceProvenance
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def verified_status_is_truthful(self) -> EvidenceRecord:
        if self.verified != (self.verification_status is EvidenceVerificationStatus.VERIFIED):
            raise ValueError("verified must exactly match VERIFIED status")
        if self.verified and self.provenance.is_mock:
            raise ValueError("Mock evidence cannot be verified")
        if self.verification_status is EvidenceVerificationStatus.MOCK and not (
            self.provenance.is_mock
        ):
            raise ValueError("MOCK status requires Mock provenance")
        return self


class ClaimType(StrEnum):
    FACTUAL = "FACTUAL"
    NUMERIC = "NUMERIC"
    MODEL = "MODEL"
    ASSUMPTION = "ASSUMPTION"
    DERIVATION = "DERIVATION"
    RESULT = "RESULT"
    COMPARISON = "COMPARISON"
    SENSITIVITY = "SENSITIVITY"
    ROBUSTNESS = "ROBUSTNESS"
    LITERATURE = "LITERATURE"
    CONCLUSION = "CONCLUSION"


class ClaimImportance(StrEnum):
    CRITICAL = "CRITICAL"
    MAJOR = "MAJOR"
    MINOR = "MINOR"


class ClaimVerificationStatus(StrEnum):
    PENDING = "PENDING"
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    HUMAN_REVIEW = "HUMAN_REVIEW"


class NumericClaimValue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: float = Field(allow_inf_nan=False)
    unit: str | None = None
    metric_name: str = Field(min_length=1)
    source_field: str = Field(min_length=1)
    tolerance: float = Field(default=1e-9, ge=0, le=0.1)


class ComparisonDirection(StrEnum):
    INCREASE = "INCREASE"
    DECREASE = "DECREASE"
    CHANGE = "CHANGE"


class ComparisonClaimValue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    baseline_value: float = Field(allow_inf_nan=False)
    verified_value: float = Field(allow_inf_nan=False)
    percentage_change: float = Field(allow_inf_nan=False)
    direction: ComparisonDirection
    unit: str | None = None
    baseline_source_field: str = Field(min_length=1)
    verified_source_field: str = Field(min_length=1)
    tolerance: float = Field(default=1e-9, ge=0, le=0.1)

    @model_validator(mode="after")
    def percentage_is_deterministic(self) -> ComparisonClaimValue:
        if self.baseline_value == 0:
            raise ValueError("comparison baseline cannot be zero")
        expected = (self.verified_value - self.baseline_value) / abs(self.baseline_value) * 100
        if not math.isclose(
            self.percentage_change,
            expected,
            rel_tol=self.tolerance,
            abs_tol=self.tolerance,
        ):
            raise ValueError("percentage_change must be deterministically calculated")
        expected_direction = (
            ComparisonDirection.INCREASE
            if expected > self.tolerance
            else ComparisonDirection.DECREASE
            if expected < -self.tolerance
            else ComparisonDirection.CHANGE
        )
        if self.direction is not expected_direction:
            raise ValueError("comparison direction does not match calculated change")
        return self


ClaimStructuredValue = NumericClaimValue | ComparisonClaimValue | dict[str, Any]


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def parse_typed_structured_value(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        claim_type = value.get("claim_type")
        structured = value.get("structured_value")
        if claim_type == ClaimType.NUMERIC and structured is not None:
            return {**value, "structured_value": NumericClaimValue.model_validate(structured)}
        if claim_type == ClaimType.COMPARISON and structured is not None:
            return {**value, "structured_value": ComparisonClaimValue.model_validate(structured)}
        return value

    claim_id: str = Field(pattern=r"^CLAIM-[A-Za-z0-9_-]+$")
    project_id: UUID
    paper_id: UUID
    paper_version: int = Field(ge=1)
    claim_type: ClaimType
    text: str = Field(min_length=1)
    structured_value: ClaimStructuredValue | None = None
    section_id: str = Field(pattern=r"^SEC-[A-Za-z0-9_-]+$")
    evidence_refs: list[UUID] = Field(default_factory=list)
    citation_refs: list[str] = Field(default_factory=list)
    verification_status: ClaimVerificationStatus = ClaimVerificationStatus.PENDING
    importance: ClaimImportance
    generated_by: str = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def structured_type_matches_claim(self) -> Claim:
        if self.claim_type is ClaimType.NUMERIC and not isinstance(
            self.structured_value, NumericClaimValue
        ):
            raise ValueError("NUMERIC claim requires NumericClaimValue")
        if self.claim_type is ClaimType.COMPARISON and not isinstance(
            self.structured_value, ComparisonClaimValue
        ):
            raise ValueError("COMPARISON claim requires ComparisonClaimValue")
        return self


class ClaimEvidenceSupport(StrEnum):
    DIRECT_SUPPORT = "DIRECT_SUPPORT"
    PARTIAL_SUPPORT = "PARTIAL_SUPPORT"
    CONTEXT_ONLY = "CONTEXT_ONLY"
    CONTRADICTS = "CONTRADICTS"


class ClaimEvidenceLink(BaseModel):
    model_config = ConfigDict(extra="forbid")

    link_id: UUID = Field(default_factory=uuid4)
    project_id: UUID
    claim_id: str = Field(pattern=r"^CLAIM-[A-Za-z0-9_-]+$")
    evidence_id: UUID
    support: ClaimEvidenceSupport
    source_field: str | None = None
    rationale: str = Field(min_length=1)


class LiteratureNeedType(StrEnum):
    THEORY = "THEORY"
    ALGORITHM = "ALGORITHM"
    DOMAIN_CONTEXT = "DOMAIN_CONTEXT"
    PARAMETER_SOURCE = "PARAMETER_SOURCE"
    BENCHMARK = "BENCHMARK"


class LiteratureSearchNeed(BaseModel):
    model_config = ConfigDict(extra="forbid")

    need_id: str = Field(pattern=r"^LITNEED-[A-Za-z0-9_-]+$")
    need_type: LiteratureNeedType
    query: str = Field(min_length=3)
    purpose: str = Field(min_length=1)
    target_claim_types: list[ClaimType] = Field(default_factory=list)
    required: bool = True


class LiteraturePlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    needs: list[LiteratureSearchNeed] = Field(min_length=1, max_length=20)
    selection_strategy: str = Field(min_length=1)


class LiteratureAgentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    problem_title: str = Field(min_length=1)
    problem_summary: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    model_family: str = Field(min_length=1)
    evidence_summaries: list[str] = Field(default_factory=list)
    requested_uses: list[LiteratureNeedType] = Field(default_factory=list)


class ReferenceSource(StrEnum):
    OPENALEX = "OPENALEX"
    CROSSREF = "CROSSREF"
    SEMANTIC_SCHOLAR = "SEMANTIC_SCHOLAR"
    ARXIV = "ARXIV"
    MANUAL = "MANUAL"
    FIXTURE = "FIXTURE"


class ReferenceMetadataOrigin(StrEnum):
    RETRIEVED = "RETRIEVED"
    MANUAL_VERIFIED = "MANUAL_VERIFIED"


class ReferenceMetadataStatus(StrEnum):
    VERIFIED = "VERIFIED"
    PARTIAL = "PARTIAL"
    NOT_FOUND = "NOT_FOUND"
    CONFLICT = "CONFLICT"
    PENDING = "PENDING"


class CitationSupportStatus(StrEnum):
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    NOT_SUPPORTED = "NOT_SUPPORTED"
    INSUFFICIENT_TEXT = "INSUFFICIENT_TEXT"
    CONTRADICTED = "CONTRADICTED"
    PENDING = "PENDING"


class ReferenceAccessStatus(StrEnum):
    AVAILABLE = "AVAILABLE"
    METADATA_ONLY = "METADATA_ONLY"
    UNAVAILABLE = "UNAVAILABLE"


class ReferenceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reference_id: str = Field(pattern=r"^REF-[A-Za-z0-9_-]+$")
    project_id: UUID
    title: str = Field(min_length=1)
    authors: list[str] = Field(min_length=1)
    year: int = Field(ge=1000, le=3000)
    venue: str = Field(min_length=1)
    doi: str | None = Field(default=None, max_length=255)
    url: str | None = Field(default=None, max_length=2048)
    abstract: str | None = None
    trusted_excerpt: str | None = None
    source: ReferenceSource
    source_id: str = Field(min_length=1)
    metadata_origin: ReferenceMetadataOrigin
    retrieved_at: datetime
    metadata_status: ReferenceMetadataStatus = ReferenceMetadataStatus.PENDING
    access_status: ReferenceAccessStatus
    raw_metadata_hash: str = Field(pattern=SHA256_PATTERN)
    is_mock: bool = False

    @field_validator("doi")
    @classmethod
    def normalize_doi(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().lower()
        for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :].strip()
        if not normalized.startswith("10.") or "/" not in normalized:
            raise ValueError("DOI must use a normalized 10.x/... identifier")
        return normalized


class CitationMetadataCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    check_id: UUID = Field(default_factory=uuid4)
    project_id: UUID
    reference_id: str = Field(pattern=r"^REF-[A-Za-z0-9_-]+$")
    reference_digest: str = Field(pattern=SHA256_PATTERN)
    status: ReferenceMetadataStatus
    field_matches: dict[str, bool]
    errors: list[str] = Field(default_factory=list)
    checked_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class CitationSupportCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    check_id: UUID = Field(default_factory=uuid4)
    project_id: UUID
    claim_id: str = Field(pattern=r"^CLAIM-[A-Za-z0-9_-]+$")
    reference_id: str = Field(pattern=r"^REF-[A-Za-z0-9_-]+$")
    claim_digest: str = Field(pattern=SHA256_PATTERN)
    reference_digest: str = Field(pattern=SHA256_PATTERN)
    status: CitationSupportStatus
    trusted_text_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)
    supporting_excerpt: str | None = None
    rationale: str = Field(min_length=1)
    reviewer_is_mock: bool = False
    checked_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def supported_requires_non_mock_text(self) -> CitationSupportCheck:
        if self.status in {
            CitationSupportStatus.SUPPORTED,
            CitationSupportStatus.PARTIALLY_SUPPORTED,
        } and (self.reviewer_is_mock or self.trusted_text_hash is None):
            raise ValueError("supported citation requires non-Mock trusted text")
        return self


class CitationSupportDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: CitationSupportStatus
    supporting_excerpt: str | None = None
    rationale: str = Field(min_length=1)


class CitationAgentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim: Claim
    reference: ReferenceRecord
    trusted_text: str = Field(min_length=1)


class DocumentObjectType(StrEnum):
    SECTION = "SECTION"
    EQUATION = "EQUATION"
    FIGURE = "FIGURE"
    TABLE = "TABLE"
    REFERENCE = "REFERENCE"
    CLAIM = "CLAIM"


class DocumentRegistryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    object_id: str = Field(pattern=DOCUMENT_ID_PATTERN)
    object_type: DocumentObjectType
    source_id: str = Field(min_length=1)
    order: int = Field(ge=1)
    content_hash: str = Field(pattern=SHA256_PATTERN)


class PaperBlockType(StrEnum):
    PARAGRAPH = "PARAGRAPH"
    CLAIM = "CLAIM"
    EQUATION = "EQUATION"
    FIGURE = "FIGURE"
    TABLE = "TABLE"
    LIST = "LIST"
    SUBSECTION = "SUBSECTION"


class AbstractRole(StrEnum):
    PROBLEM = "PROBLEM"
    METHOD = "METHOD"
    KEY_RESULT = "KEY_RESULT"
    CONCLUSION = "CONCLUSION"


class PaperBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    block_id: str = Field(pattern=r"^BLOCK-[A-Za-z0-9_-]+$")
    block_type: PaperBlockType
    text: str | None = None
    items: list[str] = Field(default_factory=list)
    claim_ref: str | None = Field(default=None, pattern=r"^CLAIM-[A-Za-z0-9_-]+$")
    equation_ref: str | None = Field(default=None, pattern=r"^EQ-[A-Za-z0-9_-]+$")
    figure_ref: str | None = Field(default=None, pattern=r"^FIG-[A-Za-z0-9_-]+$")
    table_ref: str | None = Field(default=None, pattern=r"^TAB-[A-Za-z0-9_-]+$")
    citation_refs: list[str] = Field(default_factory=list)
    abstract_role: AbstractRole | None = None

    @model_validator(mode="after")
    def block_has_required_payload(self) -> PaperBlock:
        required = {
            PaperBlockType.CLAIM: self.claim_ref,
            PaperBlockType.EQUATION: self.equation_ref,
            PaperBlockType.FIGURE: self.figure_ref,
            PaperBlockType.TABLE: self.table_ref,
        }
        if self.block_type in required and required[self.block_type] is None:
            raise ValueError(f"{self.block_type.value} block requires its registry reference")
        if self.block_type in {PaperBlockType.PARAGRAPH, PaperBlockType.SUBSECTION} and not (
            self.text
        ):
            raise ValueError(f"{self.block_type.value} block requires text")
        if self.block_type is PaperBlockType.LIST and not self.items:
            raise ValueError("LIST block requires items")
        return self


class PaperSectionType(StrEnum):
    ABSTRACT = "ABSTRACT"
    PROBLEM_RESTATEMENT = "PROBLEM_RESTATEMENT"
    PROBLEM_ANALYSIS = "PROBLEM_ANALYSIS"
    ASSUMPTIONS = "ASSUMPTIONS"
    SYMBOLS = "SYMBOLS"
    DATA = "DATA"
    MODEL = "MODEL"
    SOLUTION = "SOLUTION"
    RESULTS = "RESULTS"
    VALIDATION = "VALIDATION"
    SENSITIVITY = "SENSITIVITY"
    ROBUSTNESS = "ROBUSTNESS"
    EVALUATION = "EVALUATION"
    CONCLUSION = "CONCLUSION"
    REFERENCES = "REFERENCES"
    APPENDIX = "APPENDIX"
    CUSTOM = "CUSTOM"


class PaperSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    section_id: str = Field(pattern=r"^SEC-[A-Za-z0-9_-]+$")
    title: str = Field(min_length=1)
    section_type: PaperSectionType
    blocks: list[PaperBlock]
    claim_refs: list[str] = Field(default_factory=list)
    equation_refs: list[str] = Field(default_factory=list)
    figure_refs: list[str] = Field(default_factory=list)
    table_refs: list[str] = Field(default_factory=list)
    citation_refs: list[str] = Field(default_factory=list)
    subproblem_refs: list[str] = Field(default_factory=list)
    order: int = Field(ge=1)


class ProblemRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subproblem_id: str = Field(pattern=r"^Q[A-Za-z0-9_-]+$")
    required_outputs: list[str] = Field(min_length=1)


class SubproblemCoverageRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subproblem_id: str = Field(pattern=r"^Q[A-Za-z0-9_-]+$")
    required_outputs: list[str] = Field(min_length=1)
    section_ids: list[str] = Field(min_length=1)
    claim_refs: list[str] = Field(min_length=1)
    output_claim_refs: dict[str, list[str]] = Field(default_factory=dict)


class CompetitionProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    competition_name: str = "Generic Mathematical Modeling Competition"
    document_class: str = "article"
    paper_size: str = "a4paper"
    margin: str = "1in"
    anonymous: bool = True
    reference_style: str = "plain"
    required_sections: list[PaperSectionType] = Field(default_factory=list)
    fail_on_unreferenced_assets: bool = False

    @field_validator("document_class", "paper_size", "margin", "reference_style")
    @classmethod
    def latex_options_are_safe(cls, value: str) -> str:
        if not value or any(
            character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.-"
            for character in value
        ):
            raise ValueError("competition profile contains unsafe LaTeX option")
        return value


class EvidenceSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    snapshot_id: UUID = Field(default_factory=uuid4)
    verified_model_id: UUID
    verified_model_version: int = Field(ge=1)
    verified_model_digest: str = Field(pattern=SHA256_PATTERN)
    verified_result_id: UUID
    validation_id: UUID
    sensitivity_id: UUID
    robustness_id: UUID
    red_team_report_id: UUID
    citation_reference_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[UUID] = Field(min_length=1)
    snapshot_hash: str = Field(pattern=SHA256_PATTERN)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class PaperQualityStatus(StrEnum):
    DRAFT = "DRAFT"
    VALIDATING = "VALIDATING"
    FAILED = "FAILED"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    READY_FOR_FINAL_JURY = "READY_FOR_FINAL_JURY"


class PaperIR(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paper_id: UUID = Field(default_factory=uuid4)
    version: int = Field(ge=1)
    title: str = Field(min_length=1)
    abstract: list[PaperBlock] = Field(min_length=1)
    keywords: list[str] = Field(min_length=1, max_length=10)
    sections: list[PaperSection] = Field(min_length=1)
    bibliography: list[str] = Field(default_factory=list)
    appendices: list[PaperSection] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    competition_profile: CompetitionProfile = Field(default_factory=CompetitionProfile)
    document_registry: list[DocumentRegistryEntry] = Field(default_factory=list)
    subproblem_coverage: list[SubproblemCoverageRecord] = Field(default_factory=list)
    evidence_snapshot: EvidenceSnapshot
    status: PaperQualityStatus = PaperQualityStatus.DRAFT

    @model_validator(mode="after")
    def section_identity_and_order_are_unique(self) -> PaperIR:
        all_sections = [*self.sections, *self.appendices]
        ids = [item.section_id for item in all_sections]
        orders = [item.order for item in all_sections]
        if len(ids) != len(set(ids)):
            raise ValueError("paper section identifiers must be unique")
        if len(orders) != len(set(orders)):
            raise ValueError("paper section order values must be unique")
        claim_ids = [item.claim_id for item in self.claims]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("paper claim identifiers must be unique")
        if any(
            (item.paper_id, item.paper_version) != (self.paper_id, self.version)
            for item in self.claims
        ):
            raise ValueError("PaperIR claims must belong to its exact paper version")
        coverage_ids = [item.subproblem_id for item in self.subproblem_coverage]
        if len(coverage_ids) != len(set(coverage_ids)):
            raise ValueError("paper subproblem coverage identifiers must be unique")
        known_claims = set(claim_ids)
        referenced_claims = {
            reference for section in all_sections for reference in section.claim_refs
        } | {
            block.claim_ref
            for block in [
                *self.abstract,
                *(block for section in all_sections for block in section.blocks),
            ]
            if block.claim_ref is not None
        }
        if not referenced_claims <= known_claims:
            raise ValueError("PaperIR contains an unknown claim reference")
        return self


class PaperAgentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: UUID
    assigned_paper_id: UUID
    assigned_version: int = Field(ge=1)
    title: str = Field(min_length=1)
    evidence: list[EvidenceRecord] = Field(min_length=1)
    evidence_snapshot: EvidenceSnapshot
    equation_ids: list[str] = Field(default_factory=list)
    symbol_definitions: dict[str, str] = Field(default_factory=dict)
    figure_ids: list[str] = Field(default_factory=list)
    table_ids: list[str] = Field(default_factory=list)
    reference_ids: list[str] = Field(default_factory=list)
    required_subproblems: list[ProblemRequirement] = Field(default_factory=list)
    competition_profile: CompetitionProfile = Field(default_factory=CompetitionProfile)
    repair_feedback: list[str] = Field(default_factory=list, max_length=3)


class PaperFactualAuditInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paper: PaperIR
    claims: list[Claim]
    evidence: list[EvidenceRecord]


class PaperFactualAuditDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passed: bool
    findings: list[str] = Field(default_factory=list)
    reviewed_claim_ids: list[str] = Field(default_factory=list)


class FigureType(StrEnum):
    TREND = "TREND"
    SCATTER = "SCATTER"
    BOXPLOT = "BOXPLOT"
    HEATMAP = "HEATMAP"
    CORRELATION = "CORRELATION"
    PREDICTED_VS_ACTUAL = "PREDICTED_VS_ACTUAL"
    RESIDUAL = "RESIDUAL"
    OPTIMIZATION_COMPARISON = "OPTIMIZATION_COMPARISON"
    SENSITIVITY_CURVE = "SENSITIVITY_CURVE"
    ROBUSTNESS_DISTRIBUTION = "ROBUSTNESS_DISTRIBUTION"
    SCENARIO_COMPARISON = "SCENARIO_COMPARISON"
    NETWORK_DIAGRAM = "NETWORK_DIAGRAM"


class RegistryGenerationStatus(StrEnum):
    GENERATED = "GENERATED"
    VERIFIED = "VERIFIED"
    FAILED = "FAILED"


class FigureRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    figure_id: str = Field(pattern=r"^FIG-[A-Za-z0-9_-]+$")
    project_id: UUID
    paper_id: UUID
    paper_version: int = Field(ge=1)
    title: str = Field(min_length=1)
    caption: str = Field(min_length=1)
    figure_type: FigureType
    source_evidence_refs: list[UUID] = Field(min_length=1)
    source_evidence_types: list[EvidenceType] = Field(min_length=1)
    source_binding: dict[str, Any] = Field(default_factory=dict)
    data_payload: dict[str, Any]
    data_hash: str = Field(pattern=SHA256_PATTERN)
    code_hash: str = Field(pattern=SHA256_PATTERN)
    image_hash: str = Field(pattern=SHA256_PATTERN)
    data_artifact_id: UUID
    code_artifact_id: UUID
    image_artifact_id: UUID
    generation_status: RegistryGenerationStatus


class TableRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    table_id: str = Field(pattern=r"^TAB-[A-Za-z0-9_-]+$")
    project_id: UUID
    paper_id: UUID
    paper_version: int = Field(ge=1)
    title: str = Field(min_length=1)
    caption: str = Field(min_length=1)
    columns: list[str] = Field(min_length=1)
    rows: list[list[str | int | float | bool | None]]
    source_evidence_refs: list[UUID] = Field(min_length=1)
    source_evidence_types: list[EvidenceType] = Field(min_length=1)
    source_binding: dict[str, Any] = Field(default_factory=dict)
    data_hash: str = Field(pattern=SHA256_PATTERN)
    data_artifact_id: UUID
    generation_status: RegistryGenerationStatus

    @model_validator(mode="after")
    def row_widths_match_columns(self) -> TableRecord:
        if any(len(row) != len(self.columns) for row in self.rows):
            raise ValueError("table row width must match columns")
        return self


class PaperArtifactKind(StrEnum):
    TEX = "TEX"
    BIB = "BIB"
    PDF = "PDF"
    MANIFEST = "MANIFEST"
    FIGURE_DATA = "FIGURE_DATA"
    FIGURE_CODE = "FIGURE_CODE"
    FIGURE_IMAGE = "FIGURE_IMAGE"
    TABLE_DATA = "TABLE_DATA"


class PaperArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_id: UUID = Field(default_factory=uuid4)
    project_id: UUID
    problem_id: UUID
    paper_id: UUID
    paper_version: int = Field(ge=1)
    kind: PaperArtifactKind
    name: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    mime_type: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=SHA256_PATTERN)
    storage_key: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("storage_key")
    @classmethod
    def storage_key_is_bounded(cls, value: str) -> str:
        normalized = value.replace("\\", "/")
        if normalized.startswith("/") or "../" in f"/{normalized}/" or ":" in normalized:
            raise ValueError("paper artifact storage key must be a bounded relative key")
        return value


class PaperCompileStatus(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    UNAVAILABLE = "UNAVAILABLE"
    REJECTED = "REJECTED"


class PaperCompileRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    compile_id: UUID = Field(default_factory=uuid4)
    paper_id: UUID
    paper_version: int = Field(ge=1)
    compiler: str = Field(min_length=1)
    compiler_version: str | None = None
    container_image: str = Field(min_length=1)
    container_image_id: str | None = None
    status: PaperCompileStatus
    exit_code: int | None = None
    runtime_seconds: float = Field(ge=0)
    stdout: str = ""
    stderr: str = ""
    warnings: list[str] = Field(default_factory=list)
    fatal_warnings: list[str] = Field(default_factory=list)
    page_count: int | None = Field(default=None, ge=1)
    tex_hash: str = Field(pattern=SHA256_PATTERN)
    bib_hash: str = Field(pattern=SHA256_PATTERN)
    pdf_artifact_id: UUID | None = None
    error: str | None = None
    network_disabled: bool = True
    shell_escape_disabled: bool = True
    non_root: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def success_requires_safe_pdf(self) -> PaperCompileRecord:
        if self.status is PaperCompileStatus.SUCCEEDED and (
            self.exit_code != 0
            or self.pdf_artifact_id is None
            or not self.network_disabled
            or not self.shell_escape_disabled
            or not self.non_root
            or self.page_count is None
            or self.fatal_warnings
        ):
            raise ValueError("successful compile requires a safe PDF artifact")
        return self


class PaperManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paper_id: UUID
    paper_version: int = Field(ge=1)
    evidence_snapshot_hash: str = Field(pattern=SHA256_PATTERN)
    verified_model_version: int = Field(ge=1)
    verified_model_digest: str = Field(pattern=SHA256_PATTERN)
    verified_result_id: UUID
    reference_ids: list[str]
    figure_ids: list[str]
    table_ids: list[str]
    paper_ir_hash: str = Field(pattern=SHA256_PATTERN)
    claim_set_hash: str = Field(pattern=SHA256_PATTERN)
    document_registry_hash: str = Field(pattern=SHA256_PATTERN)
    reference_set_hash: str = Field(pattern=SHA256_PATTERN)
    figure_set_hash: str = Field(pattern=SHA256_PATTERN)
    table_set_hash: str = Field(pattern=SHA256_PATTERN)
    tex_artifact_id: UUID
    bib_artifact_id: UUID
    pdf_artifact_id: UUID
    tex_hash: str = Field(pattern=SHA256_PATTERN)
    bib_hash: str = Field(pattern=SHA256_PATTERN)
    pdf_hash: str = Field(pattern=SHA256_PATTERN)
    manifest_hash: str = Field(pattern=SHA256_PATTERN)


class PaperValidationSeverity(StrEnum):
    ERROR = "ERROR"
    WARNING = "WARNING"


class PaperValidationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    severity: PaperValidationSeverity
    object_ref: str | None = None


class PaperQualityReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paper_id: UUID
    paper_version: int = Field(ge=1)
    status: PaperQualityStatus
    checks: dict[str, bool]
    issues: list[PaperValidationIssue] = Field(default_factory=list)
    claim_count: int = Field(ge=0)
    supported_claim_count: int = Field(ge=0)
    unsupported_claim_count: int = Field(ge=0)
    reference_count: int = Field(ge=0)
    verified_reference_count: int = Field(ge=0)
    figure_count: int = Field(ge=0)
    table_count: int = Field(ge=0)


class PaperVersion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paper_id: UUID = Field(default_factory=uuid4)
    project_id: UUID
    problem_id: UUID
    version: int = Field(ge=1)
    parent_version: int | None = Field(default=None, ge=1)
    revision_reason: str = Field(min_length=1)
    evidence_snapshot: EvidenceSnapshot
    paper_ir: PaperIR
    status: PaperQualityStatus
    manifest_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)
    paper_agent_run_id: UUID | None = None
    paper_agent_is_mock: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def identity_and_version_match_ir(self) -> PaperVersion:
        if (self.paper_id, self.version) != (self.paper_ir.paper_id, self.paper_ir.version):
            raise ValueError("PaperVersion identity/version must match PaperIR")
        if self.evidence_snapshot != self.paper_ir.evidence_snapshot:
            raise ValueError("PaperVersion and PaperIR evidence snapshots must match")
        if self.parent_version is not None and self.parent_version >= self.version:
            raise ValueError("paper parent_version must precede version")
        if self.status is not self.paper_ir.status:
            raise ValueError("PaperVersion status must match PaperIR status")
        if self.status is PaperQualityStatus.READY_FOR_FINAL_JURY and self.manifest_hash is None:
            raise ValueError("ready PaperVersion requires a manifest hash")
        return self


class PaperVersionRef(BaseModel):
    paper_id: UUID
    version: int = Field(ge=1)
    verified_result_id: UUID
    evidence_snapshot_hash: str = Field(pattern=SHA256_PATTERN)
    status: PaperQualityStatus
    manifest_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)
