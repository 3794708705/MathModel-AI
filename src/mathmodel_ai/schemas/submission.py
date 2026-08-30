from __future__ import annotations

import unicodedata
from datetime import UTC, datetime
from enum import StrEnum
from math import isfinite
from pathlib import PurePosixPath
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from mathmodel_ai.schemas.paper import ClaimType, PaperIR, PaperQualityStatus

SHA256_PATTERN = r"^[a-f0-9]{64}$"


class ProfileVerificationStatus(StrEnum):
    TEST_FIXTURE = "TEST_FIXTURE"
    VERIFIED = "VERIFIED"
    PARTIALLY_VERIFIED = "PARTIALLY_VERIFIED"
    UNVERIFIED = "UNVERIFIED"
    UNKNOWN = "UNKNOWN"
    CONFLICT = "CONFLICT"


class RuleType(StrEnum):
    PAGE_LIMIT = "PAGE_LIMIT"
    FILE_FORMAT = "FILE_FORMAT"
    FILE_COUNT = "FILE_COUNT"
    FILE_SIZE = "FILE_SIZE"
    FILENAME = "FILENAME"
    ANONYMITY = "ANONYMITY"
    REQUIRED_SECTION = "REQUIRED_SECTION"
    PROHIBITED_CONTENT = "PROHIBITED_CONTENT"
    REFERENCE_STYLE = "REFERENCE_STYLE"
    APPENDIX = "APPENDIX"
    CODE_SUBMISSION = "CODE_SUBMISSION"
    DATA_SUBMISSION = "DATA_SUBMISSION"
    AI_DISCLOSURE = "AI_DISCLOSURE"
    LANGUAGE = "LANGUAGE"
    DEADLINE = "DEADLINE"
    CUSTOM = "CUSTOM"


class RuleSeverity(StrEnum):
    BLOCKING = "BLOCKING"
    MAJOR = "MAJOR"
    MINOR = "MINOR"
    INFO = "INFO"


class RuleResultStatus(StrEnum):
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"
    NOT_EVALUABLE = "NOT_EVALUABLE"
    HUMAN_REVIEW = "HUMAN_REVIEW"


class PageScope(StrEnum):
    TOTAL = "TOTAL"
    MAIN_TEXT = "MAIN_TEXT"
    EXCLUDE_APPENDIX = "EXCLUDE_APPENDIX"
    EXCLUDE_REFERENCES = "EXCLUDE_REFERENCES"


class InclusionPolicy(StrEnum):
    REQUIRED = "REQUIRED"
    ALLOWED = "ALLOWED"
    PROHIBITED = "PROHIBITED"


class AIDisclosurePolicy(StrEnum):
    REQUIRED = "REQUIRED"
    PROHIBITED = "PROHIBITED"
    ALLOWED_WITH_CONDITIONS = "ALLOWED_WITH_CONDITIONS"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    UNKNOWN = "UNKNOWN"


class DeadlineStatus(StrEnum):
    OPEN = "OPEN"
    NEAR_DEADLINE = "NEAR_DEADLINE"
    EXPIRED = "EXPIRED"
    UNKNOWN = "UNKNOWN"


class DeadlineMode(StrEnum):
    NORMAL = "NORMAL"
    PRIORITY = "PRIORITY"
    FINALIZATION = "FINALIZATION"
    MODEL_FREEZE = "MODEL_FREEZE"
    SUBMISSION_MODE = "SUBMISSION_MODE"
    UNKNOWN = "UNKNOWN"


class CompetitionRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rule_id: str = Field(pattern=r"^RULE-[A-Za-z0-9_-]+$")
    description: str = Field(min_length=1)
    rule_type: RuleType
    severity: RuleSeverity
    parameters: dict[str, Any] = Field(default_factory=dict)
    source_ref: str = Field(min_length=1)
    source_location: str = Field(min_length=1)
    verification_status: ProfileVerificationStatus


class PageRules(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_pages: int | None = Field(default=None, ge=1, le=10_000)
    scope: PageScope = PageScope.TOTAL


class FontRules(BaseModel):
    model_config = ConfigDict(extra="forbid")

    required_fonts: list[str] = Field(default_factory=list)
    minimum_size_pt: float | None = Field(default=None, gt=0)


class MarginRules(BaseModel):
    model_config = ConfigDict(extra="forbid")

    minimum_inches: float | None = Field(default=None, ge=0)


class AnonymousRules(BaseModel):
    model_config = ConfigDict(extra="forbid")

    required: bool = False
    prohibited_terms: list[str] = Field(default_factory=list)
    scan_pdf_metadata: bool = True


class ReferenceRules(BaseModel):
    model_config = ConfigDict(extra="forbid")

    style: str | None = None
    require_verified_critical_references: bool = True
    reject_duplicate_doi: bool = True


class AppendixRules(BaseModel):
    model_config = ConfigDict(extra="forbid")

    allowed: bool = True
    included_in_page_limit: bool | None = None


class FileRules(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_file_count: int | None = Field(default=None, ge=1, le=100_000)
    max_file_size_bytes: int | None = Field(default=None, ge=1, le=10 * 1024**4)
    max_package_size_bytes: int | None = Field(default=None, ge=1, le=10 * 1024**4)
    allowed_mime_types: list[str] = Field(default_factory=list)


class NamingRules(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paper_filename: str | None = None
    prefix: str | None = None
    suffix: str | None = None
    allowed_pattern: str = r"^[A-Za-z0-9_.-]+$"
    case_sensitive: bool = True

    @field_validator("allowed_pattern")
    @classmethod
    def filename_pattern_is_bounded(cls, value: str) -> str:
        if value != r"^[A-Za-z0-9_.-]+$":
            raise ValueError("custom filename regular expressions are not supported")
        return value


class SubmissionContentRules(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy: InclusionPolicy = InclusionPolicy.ALLOWED
    readme_required: bool = False
    entrypoint_required: bool = False


class AIDisclosureRules(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy: AIDisclosurePolicy = AIDisclosurePolicy.UNKNOWN
    required_text: str | None = None


class CompetitionProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile_id: UUID = Field(default_factory=uuid4)
    name: str = Field(min_length=1)
    version: int = Field(ge=1)
    competition_name: str = Field(min_length=1)
    competition_year: int | None = Field(default=None, ge=1900, le=3000)
    language: str = Field(min_length=1)
    required_sections: list[str] = Field(default_factory=list)
    optional_sections: list[str] = Field(default_factory=list)
    page_rules: PageRules = Field(default_factory=PageRules)
    font_rules: FontRules = Field(default_factory=FontRules)
    margin_rules: MarginRules = Field(default_factory=MarginRules)
    anonymous_rules: AnonymousRules = Field(default_factory=AnonymousRules)
    reference_rules: ReferenceRules = Field(default_factory=ReferenceRules)
    appendix_rules: AppendixRules = Field(default_factory=AppendixRules)
    file_rules: FileRules = Field(default_factory=FileRules)
    naming_rules: NamingRules = Field(default_factory=NamingRules)
    allowed_submission_files: list[str] = Field(default_factory=list)
    prohibited_submission_files: list[str] = Field(default_factory=list)
    code_submission_rules: SubmissionContentRules = Field(default_factory=SubmissionContentRules)
    data_submission_rules: SubmissionContentRules = Field(default_factory=SubmissionContentRules)
    ai_disclosure_rules: AIDisclosureRules = Field(default_factory=AIDisclosureRules)
    deadline: datetime | None = None
    timezone: str | None = None
    special_rules: dict[str, Any] = Field(default_factory=dict)
    source_refs: list[str] = Field(min_length=1)
    verification_status: ProfileVerificationStatus
    rules: list[CompetitionRule] = Field(min_length=1)
    minimum_jury_score: float = Field(default=75, ge=0, le=100)
    jury_weights: dict[str, float] = Field(default_factory=dict)
    require_independent_reviewer: bool = False

    @field_validator("deadline")
    @classmethod
    def deadline_is_aware(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("competition deadline must include a timezone")
        return value

    @field_validator("allowed_submission_files", "prohibited_submission_files")
    @classmethod
    def package_patterns_are_bounded(cls, values: list[str]) -> list[str]:
        for value in values:
            normalized = value.replace("\\", "/")
            path = PurePosixPath(normalized)
            if (
                not normalized
                or path.is_absolute()
                or ".." in path.parts
                or ":" in normalized
                or any(ord(character) < 32 for character in normalized)
            ):
                raise ValueError("submission file patterns must be bounded relative paths")
        return values

    @model_validator(mode="after")
    def unique_rules_and_weights(self) -> CompetitionProfile:
        rule_ids = [item.rule_id for item in self.rules]
        if len(rule_ids) != len(set(rule_ids)):
            raise ValueError("competition rule identifiers must be unique")
        if self.jury_weights and abs(sum(self.jury_weights.values()) - 100) > 1e-9:
            raise ValueError("jury weights must sum to 100")
        page_limits = {
            int(item.parameters["max_pages"])
            for item in self.rules
            if item.rule_type is RuleType.PAGE_LIMIT and "max_pages" in item.parameters
        }
        if self.page_rules.max_pages is not None:
            page_limits.add(self.page_rules.max_pages)
        if len(page_limits) > 1:
            raise ValueError("competition profile contains conflicting page limits")
        for rule in self.rules:
            if not _json_values_are_finite(rule.parameters):
                raise ValueError("competition rule parameters must be finite JSON values")
            if rule.rule_type is RuleType.PAGE_LIMIT and "max_pages" in rule.parameters:
                value = rule.parameters["max_pages"]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or not 1 <= value <= 10_000
                ):
                    raise ValueError("page limit rule must use a positive bounded integer")
                try:
                    PageScope(str(rule.parameters.get("scope", self.page_rules.scope.value)))
                except ValueError as exc:
                    raise ValueError("page limit rule has an invalid scope") from exc
            if rule.rule_type is RuleType.FILE_COUNT and "maximum" in rule.parameters:
                maximum = rule.parameters["maximum"]
                if (
                    isinstance(maximum, bool)
                    or not isinstance(maximum, int)
                    or not 1 <= maximum <= 100_000
                ):
                    raise ValueError("file count rule must use a positive bounded integer")
            if rule.rule_type is RuleType.FILE_SIZE and "maximum" in rule.parameters:
                maximum = rule.parameters["maximum"]
                if (
                    isinstance(maximum, bool)
                    or not isinstance(maximum, int)
                    or not 1 <= maximum <= 10 * 1024**4
                ):
                    raise ValueError("file size rule must use a positive bounded integer")
        if not _json_values_are_finite(self.special_rules):
            raise ValueError("special rules must contain finite JSON values")
        return self


def _json_values_are_finite(value: Any) -> bool:
    if isinstance(value, float):
        return isfinite(value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _json_values_are_finite(item) for key, item in value.items()
        )
    if isinstance(value, list):
        return all(_json_values_are_finite(item) for item in value)
    return value is None or isinstance(value, (str, int, bool))


class RuleResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    result_id: UUID = Field(default_factory=uuid4)
    rule_id: str = Field(pattern=r"^RULE-[A-Za-z0-9_-]+$")
    rule_type: RuleType
    severity: RuleSeverity
    status: RuleResultStatus
    message: str = Field(min_length=1)
    evidence_refs: list[str] = Field(default_factory=list)
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SubmissionArtifactRole(StrEnum):
    PAPER_PDF = "PAPER_PDF"
    PAPER_TEX = "PAPER_TEX"
    REFERENCES = "REFERENCES"
    FIGURE = "FIGURE"
    TABLE = "TABLE"
    CODE = "CODE"
    DATA = "DATA"
    README = "README"
    MANIFEST = "MANIFEST"
    PACKAGE = "PACKAGE"


class SubmissionArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_id: UUID = Field(default_factory=uuid4)
    project_id: UUID
    role: SubmissionArtifactRole
    relative_path: str = Field(min_length=1)
    mime_type: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=SHA256_PATTERN)
    storage_key: str = Field(min_length=1)
    source_artifact_id: UUID | None = None
    paper_id: UUID | None = None
    paper_version: int | None = Field(default=None, ge=1)

    @field_validator("relative_path", "storage_key")
    @classmethod
    def paths_are_relative_and_bounded(cls, value: str) -> str:
        normalized = value.replace("\\", "/")
        path = PurePosixPath(normalized)
        if (
            path.is_absolute()
            or not path.parts
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError("submission paths must be bounded relative paths")
        if ":" in normalized or any(ord(character) < 32 for character in normalized):
            raise ValueError("submission path contains forbidden characters")
        return normalized


class SubmissionCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: UUID = Field(default_factory=uuid4)
    project_id: UUID
    problem_id: UUID
    competition_profile_id: UUID
    competition_profile_version: int = Field(ge=1)
    competition_profile_digest: str = Field(pattern=SHA256_PATTERN)
    paper_id: UUID
    paper_version: int = Field(ge=1)
    paper_status: PaperQualityStatus
    paper_manifest_hash: str = Field(pattern=SHA256_PATTERN)
    paper_ir_hash: str = Field(pattern=SHA256_PATTERN)
    verified_model_id: UUID
    verified_model_version: int = Field(ge=1)
    verified_model_digest: str = Field(pattern=SHA256_PATTERN)
    verified_result_id: UUID
    phase5_verified: bool
    page_count: int = Field(ge=1)
    section_types: list[str]
    paper_text: str
    paper_metadata: dict[str, str] = Field(default_factory=dict)
    artifacts: list[SubmissionArtifact] = Field(min_length=1)


class RequirementCoverageStatus(StrEnum):
    COVERED = "COVERED"
    PARTIAL = "PARTIAL"
    MISSING = "MISSING"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class SubmissionRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requirement_id: str = Field(pattern=r"^REQ-[A-Za-z0-9_-]+$")
    source_text: str = Field(min_length=1)
    normalized_requirement: str = Field(min_length=1)
    subproblem_id: str = Field(pattern=r"^Q[A-Za-z0-9_-]+$")
    required_output: str = Field(min_length=1)
    allowed_claim_types: list[ClaimType] = Field(min_length=1)
    required: bool = True
    evidence_refs: list[UUID] = Field(default_factory=list)
    paper_refs: list[str] = Field(default_factory=list)
    artifact_refs: list[UUID] = Field(default_factory=list)
    status: RequirementCoverageStatus = RequirementCoverageStatus.MISSING


class RequirementCoverage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    coverage_id: UUID = Field(default_factory=uuid4)
    requirement_id: str
    requirement_digest: str = Field(pattern=SHA256_PATTERN)
    subproblem_id: str
    required: bool = True
    status: RequirementCoverageStatus
    evidence_refs: list[UUID] = Field(default_factory=list)
    paper_refs: list[str] = Field(default_factory=list)
    artifact_refs: list[UUID] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class JurySeverity(StrEnum):
    CRITICAL = "CRITICAL"
    MAJOR = "MAJOR"
    MINOR = "MINOR"
    INFO = "INFO"


class JuryDecision(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    RECHECK_REQUIRED = "RECHECK_REQUIRED"


class JuryFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    finding_id: str = Field(pattern=r"^JURY-[A-Za-z0-9_-]+$")
    category: str = Field(min_length=1)
    severity: JurySeverity
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    evidence_refs: list[str] = Field(default_factory=list)
    paper_refs: list[str] = Field(default_factory=list)
    rule_refs: list[str] = Field(default_factory=list)
    impact: str = Field(min_length=1)
    recommendation: str = Field(min_length=1)
    auto_fixable: bool = False
    confidence: float = Field(ge=0, le=1)
    resolved: bool = False


class JuryDimensionScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dimension: str = Field(min_length=1)
    score: float = Field(ge=0)
    maximum: float = Field(gt=0)
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def score_does_not_exceed_maximum(self) -> JuryDimensionScore:
        if self.score > self.maximum:
            raise ValueError("jury dimension score exceeds its maximum")
        return self


class FinalJuryDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dimensions: list[JuryDimensionScore] = Field(min_length=1)
    findings: list[JuryFinding] = Field(default_factory=list)
    summary: str = Field(min_length=1)


class FinalJuryInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile: CompetitionProfile
    paper: PaperIR
    candidate: SubmissionCandidate
    requirements: list[RequirementCoverage]
    rule_results: list[RuleResult]


class FinalJuryReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    report_id: UUID = Field(default_factory=uuid4)
    project_id: UUID
    paper_id: UUID
    paper_version: int = Field(ge=1)
    paper_manifest_hash: str = Field(pattern=SHA256_PATTERN)
    paper_ir_hash: str = Field(pattern=SHA256_PATTERN)
    verified_model_id: UUID
    verified_model_version: int = Field(ge=1)
    verified_model_digest: str = Field(pattern=SHA256_PATTERN)
    verified_result_id: UUID
    competition_profile_id: UUID
    competition_profile_version: int = Field(ge=1)
    competition_profile_digest: str = Field(pattern=SHA256_PATTERN)
    candidate_digest: str = Field(pattern=SHA256_PATTERN)
    requirement_snapshot_digest: str = Field(pattern=SHA256_PATTERN)
    rule_result_snapshot_digest: str = Field(pattern=SHA256_PATTERN)
    artifact_set_digest: str = Field(pattern=SHA256_PATTERN)
    dimensions: list[JuryDimensionScore] = Field(min_length=1)
    findings: list[JuryFinding] = Field(default_factory=list)
    claimed_score: float = Field(ge=0, le=100)
    decision: JuryDecision
    summary: str = Field(min_length=1)
    reviewer_is_mock: bool = False
    independent_reviewer_used: bool = False
    agent_run_id: UUID | None = None
    report_digest: str = Field(pattern=SHA256_PATTERN)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SubmissionCheckCategory(StrEnum):
    PROBLEM_COVERAGE = "PROBLEM_COVERAGE"
    PAPER = "PAPER"
    MODEL = "MODEL"
    RESULT = "RESULT"
    CITATION = "CITATION"
    FIGURE = "FIGURE"
    TABLE = "TABLE"
    CODE = "CODE"
    DATA = "DATA"
    FILES = "FILES"
    FORMAT = "FORMAT"
    ANONYMITY = "ANONYMITY"
    SECURITY = "SECURITY"
    MANIFEST = "MANIFEST"
    DEADLINE = "DEADLINE"


class SubmissionCheckIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(min_length=1)
    category: SubmissionCheckCategory
    severity: RuleSeverity
    message: str = Field(min_length=1)
    object_ref: str | None = None


class SubmissionCheckStatus(StrEnum):
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"
    HUMAN_REVIEW = "HUMAN_REVIEW"


class SubmissionCheckResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    check_id: UUID = Field(default_factory=uuid4)
    project_id: UUID
    paper_id: UUID
    paper_version: int = Field(ge=1)
    verified_result_id: UUID
    jury_report_id: UUID
    competition_profile_digest: str = Field(pattern=SHA256_PATTERN)
    candidate_digest: str = Field(pattern=SHA256_PATTERN)
    requirement_snapshot_digest: str = Field(pattern=SHA256_PATTERN)
    rule_result_snapshot_digest: str = Field(pattern=SHA256_PATTERN)
    jury_report_digest: str = Field(pattern=SHA256_PATTERN)
    status: SubmissionCheckStatus
    issues: list[SubmissionCheckIssue] = Field(default_factory=list)
    rule_results: list[RuleResult]
    requirement_coverage: list[RequirementCoverage]
    claimed_failed_rule_count: int = Field(ge=0)
    claimed_missing_requirement_count: int = Field(ge=0)
    check_digest: str = Field(pattern=SHA256_PATTERN)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SubmissionStatus(StrEnum):
    DRAFT = "DRAFT"
    CHECKING = "CHECKING"
    FAILED = "FAILED"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    READY = "READY"
    FROZEN = "FROZEN"
    DIRTY = "DIRTY"


class CorrectionScope(StrEnum):
    FORMAT_ONLY = "FORMAT_ONLY"
    PAPER_ONLY = "PAPER_ONLY"
    CITATION = "CITATION"
    ARTIFACT = "ARTIFACT"
    CODE = "CODE"
    MODEL = "MODEL"
    DATA = "DATA"


class CorrectionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    correction_id: UUID = Field(default_factory=uuid4)
    project_id: UUID
    finding_refs: list[str] = Field(min_length=1)
    scope: CorrectionScope
    affected_components: list[str] = Field(min_length=1)
    requires_model_change: bool
    requires_result_change: bool
    requires_paper_change: bool
    required_revalidation: list[str] = Field(min_length=1)
    risk: RuleSeverity
    priority: int = Field(ge=1, le=5)


class InvalidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    invalidated_stages: list[str]
    jury_recheck_required: bool
    submission_dirty: bool
    jury_status: JuryDecision = JuryDecision.RECHECK_REQUIRED
    submission_status: SubmissionStatus = SubmissionStatus.DIRTY


class SubmissionManifestFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    role: SubmissionArtifactRole
    sha256: str = Field(pattern=SHA256_PATTERN)
    size_bytes: int = Field(ge=0)
    mime_type: str = Field(min_length=1)
    source_artifact_id: UUID

    @field_validator("path")
    @classmethod
    def manifest_path_is_bounded(cls, value: str) -> str:
        normalized = value.replace("\\", "/")
        path = PurePosixPath(normalized)
        if (
            path.is_absolute()
            or not path.parts
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError("manifest path must be relative and traversal-free")
        return normalized


class SubmissionManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    submission_id: UUID
    project_id: UUID
    competition_profile_id: UUID
    competition_profile_version: int = Field(ge=1)
    competition_profile_digest: str = Field(pattern=SHA256_PATTERN)
    requirement_snapshot_digest: str = Field(pattern=SHA256_PATTERN)
    jury_report_id: UUID
    jury_report_digest: str = Field(pattern=SHA256_PATTERN)
    submission_check_id: UUID
    submission_check_digest: str = Field(pattern=SHA256_PATTERN)
    candidate_digest: str = Field(pattern=SHA256_PATTERN)
    artifact_set_digest: str = Field(pattern=SHA256_PATTERN)
    paper_id: UUID
    paper_version: int = Field(ge=1)
    paper_manifest_hash: str = Field(pattern=SHA256_PATTERN)
    paper_ir_hash: str = Field(pattern=SHA256_PATTERN)
    verified_model_id: UUID
    verified_model_version: int = Field(ge=1)
    verified_model_digest: str = Field(pattern=SHA256_PATTERN)
    verified_result_id: UUID
    files: list[SubmissionManifestFile] = Field(min_length=1)
    package_hash: str = Field(pattern=SHA256_PATTERN)
    manifest_hash: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def file_identity_is_unambiguous(self) -> SubmissionManifest:
        paths = [
            unicodedata.normalize("NFC", item.path.replace("\\", "/")).casefold()
            for item in self.files
        ]
        source_ids = [item.source_artifact_id for item in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("manifest file paths contain a case or Unicode collision")
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("manifest source artifact identifiers must be unique")
        return self


class SubmissionSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    submission_id: UUID = Field(default_factory=uuid4)
    project_id: UUID
    competition_profile_id: UUID
    competition_profile_version: int = Field(ge=1)
    competition_profile_digest: str = Field(pattern=SHA256_PATTERN)
    requirement_snapshot_digest: str = Field(pattern=SHA256_PATTERN)
    jury_report_id: UUID
    jury_report_digest: str = Field(pattern=SHA256_PATTERN)
    jury_reviewer_is_mock: bool
    submission_check_id: UUID
    submission_check_digest: str = Field(pattern=SHA256_PATTERN)
    candidate_digest: str = Field(pattern=SHA256_PATTERN)
    artifact_set_digest: str = Field(pattern=SHA256_PATTERN)
    verified_model_id: UUID
    verified_model_version: int = Field(ge=1)
    verified_model_digest: str = Field(pattern=SHA256_PATTERN)
    verified_result_id: UUID
    paper_id: UUID
    paper_version: int = Field(ge=1)
    paper_manifest_hash: str = Field(pattern=SHA256_PATTERN)
    paper_ir_hash: str = Field(pattern=SHA256_PATTERN)
    artifact_ids: list[UUID] = Field(min_length=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    status: SubmissionStatus
    manifest_hash: str = Field(pattern=SHA256_PATTERN)
    package_hash: str = Field(pattern=SHA256_PATTERN)
    snapshot_digest: str = Field(pattern=SHA256_PATTERN)

    @field_validator("artifact_ids")
    @classmethod
    def artifact_ids_are_unique(cls, values: list[UUID]) -> list[UUID]:
        if len(values) != len(set(values)):
            raise ValueError("frozen artifact identifiers must be unique")
        return values


class SubmissionSummaryRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    submission_id: UUID
    paper_id: UUID
    paper_version: int = Field(ge=1)
    verified_result_id: UUID
    competition_profile_id: UUID
    competition_profile_version: int = Field(ge=1)
    status: SubmissionStatus
    manifest_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)
    package_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)


class ReproductionStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIPPED_BY_PROFILE = "SKIPPED_BY_PROFILE"
    SKIPPED_BY_BUDGET = "SKIPPED_BY_BUDGET"
    NOT_APPLICABLE = "NOT_APPLICABLE"
