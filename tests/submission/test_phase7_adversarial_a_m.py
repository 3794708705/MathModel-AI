from __future__ import annotations

from uuid import UUID, uuid4

from mathmodel_ai.schemas.submission import (
    AIDisclosurePolicy,
    InclusionPolicy,
    JuryDecision,
    ProfileVerificationStatus,
    RequirementCoverage,
    RequirementCoverageStatus,
    RuleResultStatus,
    RuleType,
    SubmissionArtifactRole,
    SubmissionCheckStatus,
)
from mathmodel_ai.submission.checks import SubmissionCheck
from mathmodel_ai.submission.jury import FinalJuryGate
from mathmodel_ai.submission.requirements import RequirementCoverageValidator, RequirementRegistry
from mathmodel_ai.submission.rules import RuleEngine
from tests.submission.helpers import (
    candidate_fixture,
    covered_paper,
    jury_draft,
    subproblem,
)


def _coverage(status: RequirementCoverageStatus = RequirementCoverageStatus.COVERED):
    return [
        RequirementCoverage(
            requirement_id="REQ-Q1-1",
            requirement_digest="a" * 64,
            subproblem_id="Q1",
            status=status,
            evidence_refs=[UUID(int=0x7101)] if status is RequirementCoverageStatus.COVERED else [],
            paper_refs=["SEC-results"] if status is RequirementCoverageStatus.COVERED else [],
            artifact_refs=[UUID(int=0x7102)] if status is RequirementCoverageStatus.COVERED else [],
        )
    ]


def _jury(profile, candidate, rules, *, innovation: float = 5, critical: bool = False):
    return FinalJuryGate().build_report(
        draft=jury_draft(innovation=innovation, critical=critical),
        profile=profile,
        candidate=candidate,
        requirements=_coverage(),
        rule_results=rules,
        reviewer_is_mock=True,
        agent_run_id=uuid4(),
    )


def test_a_missing_subproblem_fails_final_jury(tmp_path) -> None:
    _, profile, candidate, payloads = candidate_fixture(tmp_path)
    requirements = RequirementRegistry.from_subproblems(
        [subproblem("Q1", "one"), subproblem("Q2", "two"), subproblem("Q3", "three")]
    )
    coverage = RequirementCoverageValidator().validate(
        requirements,
        covered_paper(("Q1", "one"), ("Q2", "two")),
        candidate.artifacts,
    )
    assert [item.status for item in coverage] == [
        RequirementCoverageStatus.COVERED,
        RequirementCoverageStatus.COVERED,
        RequirementCoverageStatus.MISSING,
    ]
    rules = RuleEngine(_).evaluate(profile, candidate, artifact_bytes=payloads)
    report = FinalJuryGate().build_report(
        draft=jury_draft(),
        profile=profile,
        candidate=candidate,
        requirements=coverage,
        rule_results=rules,
        reviewer_is_mock=True,
        agent_run_id=uuid4(),
    )
    assert report.decision is JuryDecision.FAIL


def test_b_fake_section_without_supported_claim_is_not_covered(tmp_path) -> None:
    _, _, candidate, _ = candidate_fixture(tmp_path)
    paper = covered_paper(("Q3", "evaluation"))
    unsupported = paper.claims[0].model_copy(update={"evidence_refs": []})
    paper = paper.model_copy(update={"claims": [unsupported]})
    requirement = RequirementRegistry.from_subproblems([subproblem("Q3", "evaluation")])
    coverage = RequirementCoverageValidator().validate(requirement, paper, candidate.artifacts)
    assert coverage[0].status is RequirementCoverageStatus.PARTIAL
    assert "coverage has no supported evidence-linked claim" in coverage[0].reasons


def test_c_page_limit_violation_fails(tmp_path) -> None:
    store, profile, candidate, payloads = candidate_fixture(tmp_path, pages=3)
    page_rule = next(item for item in profile.rules if item.rule_type is RuleType.PAGE_LIMIT)
    page_rule = page_rule.model_copy(update={"parameters": {"max_pages": 2, "scope": "TOTAL"}})
    profile = profile.model_copy(
        update={
            "page_rules": profile.page_rules.model_copy(update={"max_pages": 2}),
            "rules": [page_rule],
        }
    )
    result = RuleEngine(store).evaluate(profile, candidate, artifact_bytes=payloads)
    assert result[0].status is RuleResultStatus.FAIL


def test_d_wrong_filename_fails(tmp_path) -> None:
    store, profile, candidate, payloads = candidate_fixture(tmp_path)
    rule = next(item for item in profile.rules if item.rule_type is RuleType.FILENAME)
    rule = rule.model_copy(update={"parameters": {"role": "PAPER_PDF", "exact": "result.pdf"}})
    profile = profile.model_copy(update={"rules": [rule]})
    assert (
        RuleEngine(store).evaluate(profile, candidate, artifact_bytes=payloads)[0].status
        is RuleResultStatus.FAIL
    )


def test_e_anonymity_leak_fails(tmp_path) -> None:
    store, profile, candidate, payloads = candidate_fixture(
        tmp_path,
        paper_text=(
            "Author: Alice, school: Example University, alice@example.org C:\\Users\\alice\\work"
        ),
    )
    rule = next(item for item in profile.rules if item.rule_type is RuleType.ANONYMITY)
    profile = profile.model_copy(update={"rules": [rule]})
    assert (
        RuleEngine(store).evaluate(profile, candidate, artifact_bytes=payloads)[0].status
        is RuleResultStatus.FAIL
    )


def test_f_secret_leak_is_blocking(tmp_path) -> None:
    store, profile, candidate, payloads = candidate_fixture(
        tmp_path,
        extra_files=[
            (
                SubmissionArtifactRole.README,
                "README.md",
                b"api_key=sk-testOnlyCredential123456789",
                "text/markdown",
            )
        ],
    )
    rules = RuleEngine(store).evaluate(profile, candidate, artifact_bytes=payloads)
    jury = _jury(profile, candidate, rules)
    check = SubmissionCheck().run(
        profile=profile,
        candidate=candidate,
        requirements=_coverage(),
        rules=rules,
        jury=jury,
        artifact_bytes=payloads,
    )
    assert check.status is SubmissionCheckStatus.FAIL
    assert "SECRET_LEAK" in {item.code for item in check.issues}


def test_g_pdf_name_with_zip_bytes_fails_mime_rule(tmp_path) -> None:
    store, profile, candidate, payloads = candidate_fixture(tmp_path)
    pdf = candidate.artifacts[0]
    payloads[pdf.artifact_id] = b"PK\x03\x04not-a-pdf"
    rule = next(item for item in profile.rules if item.rule_type is RuleType.FILE_FORMAT)
    profile = profile.model_copy(update={"rules": [rule]})
    assert (
        RuleEngine(store).evaluate(profile, candidate, artifact_bytes=payloads)[0].status
        is RuleResultStatus.FAIL
    )


def test_h_required_code_missing_fails(tmp_path) -> None:
    store, profile, candidate, payloads = candidate_fixture(tmp_path)
    rule = next(item for item in profile.rules if item.rule_type is RuleType.CODE_SUBMISSION)
    rule = rule.model_copy(update={"parameters": {"policy": InclusionPolicy.REQUIRED.value}})
    profile = profile.model_copy(update={"rules": [rule]})
    assert (
        RuleEngine(store).evaluate(profile, candidate, artifact_bytes=payloads)[0].status
        is RuleResultStatus.FAIL
    )


def test_i_prohibited_code_fails(tmp_path) -> None:
    store, profile, candidate, payloads = candidate_fixture(
        tmp_path,
        extra_files=[(SubmissionArtifactRole.CODE, "code/main.py", b"print(1)\n", "text/x-python")],
    )
    rule = next(item for item in profile.rules if item.rule_type is RuleType.CODE_SUBMISSION)
    rule = rule.model_copy(update={"parameters": {"policy": InclusionPolicy.PROHIBITED.value}})
    profile = profile.model_copy(update={"rules": [rule]})
    assert (
        RuleEngine(store).evaluate(profile, candidate, artifact_bytes=payloads)[0].status
        is RuleResultStatus.FAIL
    )


def test_j_unverified_blocking_rule_requires_human_review(tmp_path) -> None:
    store, profile, candidate, payloads = candidate_fixture(tmp_path)
    rule = next(item for item in profile.rules if item.rule_type is RuleType.ANONYMITY)
    rule = rule.model_copy(update={"verification_status": ProfileVerificationStatus.UNVERIFIED})
    profile = profile.model_copy(update={"rules": [rule]})
    rules = RuleEngine(store).evaluate(profile, candidate, artifact_bytes=payloads)
    jury = _jury(profile, candidate, rules)
    check = SubmissionCheck().run(
        profile=profile,
        candidate=candidate,
        requirements=_coverage(),
        rules=rules,
        jury=jury,
        artifact_bytes=payloads,
    )
    assert rules[0].status is RuleResultStatus.HUMAN_REVIEW
    assert jury.decision is JuryDecision.HUMAN_REVIEW
    assert check.status is SubmissionCheckStatus.HUMAN_REVIEW


def test_k_critical_finding_overrides_score_100(tmp_path) -> None:
    store, profile, candidate, payloads = candidate_fixture(tmp_path)
    rules = RuleEngine(store).evaluate(profile, candidate, artifact_bytes=payloads)
    assert _jury(profile, candidate, rules, critical=True).decision is JuryDecision.FAIL


def test_l_missing_requirement_overrides_high_score(tmp_path) -> None:
    store, profile, candidate, payloads = candidate_fixture(tmp_path)
    rules = RuleEngine(store).evaluate(profile, candidate, artifact_bytes=payloads)
    report = FinalJuryGate().build_report(
        draft=jury_draft(),
        profile=profile,
        candidate=candidate,
        requirements=_coverage(RequirementCoverageStatus.MISSING),
        rule_results=rules,
        reviewer_is_mock=True,
        agent_run_id=uuid4(),
    )
    assert report.claimed_score == 100
    assert report.decision is JuryDecision.FAIL


def test_m_low_innovation_is_not_a_blocking_failure(tmp_path) -> None:
    store, profile, candidate, payloads = candidate_fixture(tmp_path)
    rules = RuleEngine(store).evaluate(profile, candidate, artifact_bytes=payloads)
    report = _jury(profile, candidate, rules, innovation=0)
    assert report.claimed_score == 95
    assert report.decision is JuryDecision.PASS


def test_page_limit_recomputes_count_from_pdf_bytes(tmp_path) -> None:
    store, profile, candidate, payloads = candidate_fixture(tmp_path, pages=3)
    candidate = candidate.model_copy(update={"page_count": 1})
    rule = next(item for item in profile.rules if item.rule_type is RuleType.PAGE_LIMIT)
    profile = profile.model_copy(update={"rules": [rule]})
    result = RuleEngine(store).evaluate(profile, candidate, artifact_bytes=payloads)[0]
    assert result.status is RuleResultStatus.FAIL
    assert "disagrees" in result.message


def test_code_readme_and_entrypoint_requirements_are_enforced(tmp_path) -> None:
    store, profile, candidate, payloads = candidate_fixture(
        tmp_path,
        extra_files=[
            (SubmissionArtifactRole.CODE, "code/other.py", b"print(1)\n", "text/x-python")
        ],
    )
    content_rules = profile.code_submission_rules.model_copy(
        update={"readme_required": True, "entrypoint_required": True}
    )
    rule = next(item for item in profile.rules if item.rule_type is RuleType.CODE_SUBMISSION)
    profile = profile.model_copy(update={"code_submission_rules": content_rules, "rules": [rule]})
    result = RuleEngine(store).evaluate(profile, candidate, artifact_bytes=payloads)[0]
    assert result.status is RuleResultStatus.FAIL
    assert "README" in result.message


def test_ai_prohibition_cannot_silently_pass(tmp_path) -> None:
    store, profile, candidate, payloads = candidate_fixture(tmp_path)
    rule = profile.rules[0].model_copy(
        update={
            "rule_id": "RULE-ai-disclosure",
            "rule_type": RuleType.AI_DISCLOSURE,
            "parameters": {},
        }
    )
    ai_rules = profile.ai_disclosure_rules.model_copy(
        update={"policy": AIDisclosurePolicy.PROHIBITED}
    )
    profile = profile.model_copy(update={"ai_disclosure_rules": ai_rules, "rules": [rule]})
    result = RuleEngine(store).evaluate(profile, candidate, artifact_bytes=payloads)[0]
    assert result.status is RuleResultStatus.FAIL
