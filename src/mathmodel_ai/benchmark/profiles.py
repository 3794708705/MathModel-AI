from __future__ import annotations

from uuid import NAMESPACE_URL, uuid5

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
    PageScope,
    ProfileVerificationStatus,
    RuleSeverity,
    RuleType,
    SubmissionContentRules,
)

PROBLEM_A_URL = (
    "https://www.contest.comap.com/undergraduate/contests/mcm/contests/2024/"
    "problems/2024_MCM_Problem_A.pdf"
)
TIPS_URL = (
    "https://www.contest.comap.com/undergraduate/contests/mcm/contests/2024/"
    "problems/MCM-ICM_Tips.pdf"
)
PROBLEM_A_REF = (
    f"{PROBLEM_A_URL}#sha256=3c69f8f3b56c9aeebc6ff45ab70b4e97f68ca7bd779923edab16ac61e36ce75a"
)
TIPS_REF = f"{TIPS_URL}#sha256=6ae9f44b7fcf2b09e4f1ec64b39fd906e64836987f7451677a0b2952c73f08b5"


def comap_mcm_2024_profile() -> CompetitionProfile:
    """Verified source profile; unsupported historical semantics remain fail-closed rules."""

    verified = ProfileVerificationStatus.VERIFIED
    blocking = RuleSeverity.BLOCKING
    return CompetitionProfile(
        profile_id=uuid5(NAMESPACE_URL, "mathmodel-ai:competition-profile:comap-mcm-2024"),
        name="COMAP MCM 2024 official benchmark profile",
        version=1,
        competition_name="COMAP Mathematical Contest in Modeling",
        competition_year=2024,
        language="English",
        required_sections=["ABSTRACT", "REFERENCES"],
        page_rules=PageRules(max_pages=25, scope=PageScope.TOTAL),
        anonymous_rules=AnonymousRules(required=True),
        file_rules=FileRules(
            max_file_count=1,
            max_file_size_bytes=17 * 1024 * 1024,
            max_package_size_bytes=17 * 1024 * 1024,
            allowed_mime_types=["application/pdf"],
        ),
        naming_rules=NamingRules(suffix=".pdf"),
        allowed_submission_files=["*.pdf"],
        prohibited_submission_files=["*.py", "*.csv", "*.xlsx", "*.zip"],
        code_submission_rules=SubmissionContentRules(policy=InclusionPolicy.PROHIBITED),
        data_submission_rules=SubmissionContentRules(policy=InclusionPolicy.PROHIBITED),
        ai_disclosure_rules=AIDisclosureRules(
            policy=AIDisclosurePolicy.ALLOWED_WITH_CONDITIONS,
            required_text="Report on Use of AI",
        ),
        special_rules={
            "historical_profile": True,
            "technical_benchmark": True,
            "ai_report_excluded_from_official_page_limit": True,
            "conservative_total_pdf_limit": 25,
            "reproduction_required": False,
        },
        source_refs=[PROBLEM_A_REF, TIPS_REF],
        verification_status=verified,
        minimum_jury_score=75,
        require_independent_reviewer=True,
        rules=[
            CompetitionRule(
                rule_id="RULE-COMAP24-PAGES",
                description=(
                    "Solution PDF is limited to 25 pages; benchmark enforcement is "
                    "conservatively total-PDF."
                ),
                rule_type=RuleType.PAGE_LIMIT,
                severity=blocking,
                parameters={"max_pages": 25, "scope": "TOTAL"},
                source_ref=PROBLEM_A_REF,
                source_location="2024 MCM Problem A, page 2",
                verification_status=verified,
            ),
            CompetitionRule(
                rule_id="RULE-COMAP24-PDF",
                description="Submit one Adobe PDF solution.",
                rule_type=RuleType.FILE_FORMAT,
                severity=blocking,
                parameters={"role": "PAPER_PDF", "mime_type": "application/pdf"},
                source_ref=TIPS_REF,
                source_location="MCM-ICM Tips, pages 5-6",
                verification_status=verified,
            ),
            CompetitionRule(
                rule_id="RULE-COMAP24-ONE-FILE",
                description="Only one solution PDF is accepted.",
                rule_type=RuleType.FILE_COUNT,
                severity=blocking,
                parameters={"maximum": 1},
                source_ref=TIPS_REF,
                source_location="MCM-ICM Tips, page 6",
                verification_status=verified,
            ),
            CompetitionRule(
                rule_id="RULE-COMAP24-SIZE",
                description="The PDF attachment must be less than 17 MB.",
                rule_type=RuleType.FILE_SIZE,
                severity=blocking,
                parameters={"maximum": 17 * 1024 * 1024},
                source_ref=TIPS_REF,
                source_location="MCM-ICM Tips, page 6",
                verification_status=verified,
            ),
            CompetitionRule(
                rule_id="RULE-COMAP24-FILENAME",
                description="The filename is the team control number with a .pdf suffix.",
                rule_type=RuleType.FILENAME,
                severity=blocking,
                parameters={"role": "PAPER_PDF", "suffix": ".pdf"},
                source_ref=TIPS_REF,
                source_location="MCM-ICM Tips, page 6",
                verification_status=verified,
            ),
            CompetitionRule(
                rule_id="RULE-COMAP24-ANONYMOUS",
                description="Student, advisor, and institution identities are prohibited.",
                rule_type=RuleType.ANONYMITY,
                severity=blocking,
                parameters={"required": True},
                source_ref=TIPS_REF,
                source_location="MCM-ICM Tips, page 5",
                verification_status=verified,
            ),
            CompetitionRule(
                rule_id="RULE-COMAP24-SECTIONS",
                description="The solution includes a summary and references.",
                rule_type=RuleType.REQUIRED_SECTION,
                severity=blocking,
                parameters={"sections": ["ABSTRACT", "REFERENCES"]},
                source_ref=PROBLEM_A_REF,
                source_location="2024 MCM Problem A, page 2",
                verification_status=verified,
            ),
            CompetitionRule(
                rule_id="RULE-COMAP24-NO-CODE",
                description="Programs and non-paper support files are not submitted.",
                rule_type=RuleType.CODE_SUBMISSION,
                severity=blocking,
                parameters={"policy": "PROHIBITED"},
                source_ref=TIPS_REF,
                source_location="MCM-ICM Tips, page 6",
                verification_status=verified,
            ),
            CompetitionRule(
                rule_id="RULE-COMAP24-NO-DATA",
                description="Data files are not submitted separately.",
                rule_type=RuleType.DATA_SUBMISSION,
                severity=blocking,
                parameters={"policy": "PROHIBITED"},
                source_ref=TIPS_REF,
                source_location="MCM-ICM Tips, page 6",
                verification_status=verified,
            ),
            CompetitionRule(
                rule_id="RULE-COMAP24-AI",
                description=(
                    "Generative-AI use is permitted only with the required disclosure report."
                ),
                rule_type=RuleType.AI_DISCLOSURE,
                severity=blocking,
                parameters={"required": True},
                source_ref=PROBLEM_A_REF,
                source_location="2024 MCM Problem A, pages 3-4",
                verification_status=verified,
            ),
            CompetitionRule(
                rule_id="RULE-COMAP24-CONTROL-NUMBER",
                description=(
                    "The registered team control number must appear in the filename and page "
                    "headers."
                ),
                rule_type=RuleType.CUSTOM,
                severity=blocking,
                parameters={"handler": "REGISTERED_CONTROL_NUMBER"},
                source_ref=TIPS_REF,
                source_location="MCM-ICM Tips, pages 5-6",
                verification_status=verified,
            ),
        ],
    )
