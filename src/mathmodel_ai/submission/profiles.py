from __future__ import annotations

from uuid import UUID

from mathmodel_ai.schemas.submission import (
    AIDisclosurePolicy,
    AIDisclosureRules,
    AnonymousRules,
    CompetitionProfile,
    CompetitionRule,
    FileRules,
    InclusionPolicy,
    NamingRules,
    PageRules,
    ProfileVerificationStatus,
    RuleSeverity,
    RuleType,
    SubmissionContentRules,
)
from mathmodel_ai.submission.integrity import validate_competition_profile

DEFAULT_JURY_WEIGHTS = {
    "problem_understanding": 10.0,
    "subproblem_coverage": 10.0,
    "model_quality": 15.0,
    "mathematical_rigor": 10.0,
    "data_computation": 10.0,
    "validation_robustness": 10.0,
    "results_interpretation": 10.0,
    "innovation": 5.0,
    "paper_quality": 10.0,
    "competition_compliance": 5.0,
    "submission_completeness": 5.0,
}


def generic_modeling_test_profile() -> CompetitionProfile:
    """Return a deterministic fixture, never an asserted real competition profile."""

    source = "fixture://generic-modeling/v1"
    status = ProfileVerificationStatus.TEST_FIXTURE
    return CompetitionProfile(
        profile_id=UUID("00000000-0000-0000-0000-000000007001"),
        name="GENERIC_MODELING_TEST_PROFILE",
        version=1,
        competition_name="Generic Modeling Test Competition",
        competition_year=2026,
        language="en",
        required_sections=["RESULTS"],
        page_rules=PageRules(max_pages=20),
        anonymous_rules=AnonymousRules(required=True),
        file_rules=FileRules(
            max_file_count=100,
            max_file_size_bytes=50 * 1024 * 1024,
            max_package_size_bytes=100 * 1024 * 1024,
        ),
        naming_rules=NamingRules(paper_filename="paper.pdf"),
        allowed_submission_files=[
            "paper.pdf",
            "paper.tex",
            "references.bib",
            "figures/*",
            "tables/*",
            "code/*",
            "data/*",
            "README.md",
        ],
        prohibited_submission_files=[
            ".env",
            ".git/*",
            "tests/*",
            "__pycache__/*",
            ".pytest_cache/*",
        ],
        code_submission_rules=SubmissionContentRules(policy=InclusionPolicy.ALLOWED),
        data_submission_rules=SubmissionContentRules(policy=InclusionPolicy.ALLOWED),
        ai_disclosure_rules=AIDisclosureRules(policy=AIDisclosurePolicy.NOT_APPLICABLE),
        source_refs=[source],
        verification_status=status,
        jury_weights=DEFAULT_JURY_WEIGHTS,
        rules=[
            CompetitionRule(
                rule_id="RULE-page-limit",
                description="The submitted PDF must not exceed 20 total pages.",
                rule_type=RuleType.PAGE_LIMIT,
                severity=RuleSeverity.BLOCKING,
                parameters={"max_pages": 20, "scope": "TOTAL"},
                source_ref=source,
                source_location="fixture.page_rules",
                verification_status=status,
            ),
            CompetitionRule(
                rule_id="RULE-paper-name",
                description="The formal paper filename is paper.pdf.",
                rule_type=RuleType.FILENAME,
                severity=RuleSeverity.BLOCKING,
                parameters={"role": "PAPER_PDF", "exact": "paper.pdf"},
                source_ref=source,
                source_location="fixture.naming_rules",
                verification_status=status,
            ),
            CompetitionRule(
                rule_id="RULE-paper-format",
                description="The formal paper must be a genuine PDF.",
                rule_type=RuleType.FILE_FORMAT,
                severity=RuleSeverity.BLOCKING,
                parameters={"role": "PAPER_PDF", "mime_type": "application/pdf"},
                source_ref=source,
                source_location="fixture.file_rules",
                verification_status=status,
            ),
            CompetitionRule(
                rule_id="RULE-anonymous",
                description="The package must not reveal participant identity or local paths.",
                rule_type=RuleType.ANONYMITY,
                severity=RuleSeverity.BLOCKING,
                parameters={"required": True},
                source_ref=source,
                source_location="fixture.anonymous_rules",
                verification_status=status,
            ),
            CompetitionRule(
                rule_id="RULE-results-section",
                description="A results section is required.",
                rule_type=RuleType.REQUIRED_SECTION,
                severity=RuleSeverity.BLOCKING,
                parameters={"sections": ["RESULTS"]},
                source_ref=source,
                source_location="fixture.required_sections",
                verification_status=status,
            ),
            CompetitionRule(
                rule_id="RULE-code-policy",
                description="Code is permitted but not required by the fixture.",
                rule_type=RuleType.CODE_SUBMISSION,
                severity=RuleSeverity.BLOCKING,
                parameters={"policy": "ALLOWED"},
                source_ref=source,
                source_location="fixture.code_submission_rules",
                verification_status=status,
            ),
            CompetitionRule(
                rule_id="RULE-data-policy",
                description="Data is permitted but not required by the fixture.",
                rule_type=RuleType.DATA_SUBMISSION,
                severity=RuleSeverity.BLOCKING,
                parameters={"policy": "ALLOWED"},
                source_ref=source,
                source_location="fixture.data_submission_rules",
                verification_status=status,
            ),
            CompetitionRule(
                rule_id="RULE-file-count",
                description="The package may contain at most 100 formal files.",
                rule_type=RuleType.FILE_COUNT,
                severity=RuleSeverity.BLOCKING,
                parameters={"maximum": 100},
                source_ref=source,
                source_location="fixture.file_rules",
                verification_status=status,
            ),
        ],
    )


class CompetitionProfileRegistry:
    def __init__(self, profiles: list[CompetitionProfile] | None = None) -> None:
        self._profiles: dict[tuple[UUID, int], CompetitionProfile] = {}
        self._digests: dict[tuple[UUID, int], str] = {}
        for profile in profiles or []:
            self.register(profile)

    def register(self, profile: CompetitionProfile) -> None:
        snapshot = CompetitionProfile.model_validate(profile.model_dump(mode="json"))
        digest = validate_competition_profile(snapshot)
        key = (profile.profile_id, profile.version)
        existing = self._profiles.get(key)
        if existing is not None and (existing != snapshot or self._digests[key] != digest):
            raise ValueError("immutable competition profile version has conflicting content")
        self._profiles[key] = snapshot
        self._digests[key] = digest

    def get(self, profile_id: UUID, version: int) -> CompetitionProfile:
        try:
            key = (profile_id, version)
            profile = self._profiles[key]
        except KeyError as exc:
            raise KeyError("competition profile version was not found") from exc
        if validate_competition_profile(profile) != self._digests[key]:
            raise ValueError("COMPETITION_PROFILE_INTEGRITY_ERROR: registry snapshot changed")
        return CompetitionProfile.model_validate(profile.model_dump(mode="json"))

    def list(self) -> list[CompetitionProfile]:
        return sorted(
            (self.get(profile_id, version) for profile_id, version in self._profiles),
            key=lambda item: (item.name, item.version),
        )
