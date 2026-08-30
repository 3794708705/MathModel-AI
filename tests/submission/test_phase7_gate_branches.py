from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest

from mathmodel_ai.schemas.paper import PaperQualityStatus
from mathmodel_ai.schemas.submission import (
    AIDisclosurePolicy,
    CompetitionRule,
    InclusionPolicy,
    JuryDecision,
    PageScope,
    ProfileVerificationStatus,
    RequirementCoverage,
    RequirementCoverageStatus,
    RuleResult,
    RuleResultStatus,
    RuleSeverity,
    RuleType,
    SubmissionArtifactRole,
    SubmissionCheckCategory,
    SubmissionCheckIssue,
    SubmissionCheckStatus,
)
from mathmodel_ai.submission.checks import SubmissionCheck
from mathmodel_ai.submission.jury import FinalJuryGate
from mathmodel_ai.submission.rules import (
    DeadlineEvaluator,
    RuleEngine,
    SubmissionSecurityScanner,
    detect_mime,
)
from tests.submission.helpers import candidate_fixture, jury_draft


def _rule(
    profile,
    rule_type: RuleType,
    *,
    parameters: dict | None = None,
    severity: RuleSeverity = RuleSeverity.BLOCKING,
) -> CompetitionRule:
    return profile.rules[0].model_copy(
        update={
            "rule_id": f"RULE-{rule_type.value.lower()}",
            "rule_type": rule_type,
            "severity": severity,
            "parameters": parameters or {},
            "verification_status": ProfileVerificationStatus.TEST_FIXTURE,
        }
    )


def _coverage(
    status: RequirementCoverageStatus = RequirementCoverageStatus.COVERED,
) -> list[RequirementCoverage]:
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


def _report(profile, candidate, rules, coverage=None, **draft_options):
    return FinalJuryGate().build_report(
        draft=jury_draft(**draft_options),
        profile=profile,
        candidate=candidate,
        requirements=coverage or _coverage(),
        rule_results=rules,
        reviewer_is_mock=True,
        agent_run_id=uuid4(),
    )


def test_deadline_modes_and_unknown_expired_states() -> None:
    now = datetime.now(UTC)
    assert DeadlineEvaluator.evaluate(None) == ("UNKNOWN", "UNKNOWN")
    with pytest.raises(ValueError, match="timezone-aware"):
        DeadlineEvaluator.evaluate(datetime(2026, 1, 1))
    assert DeadlineEvaluator.evaluate(now - timedelta(seconds=1), now)[0] == "EXPIRED"
    expected = [
        (0.5, "SUBMISSION_MODE"),
        (2, "MODEL_FREEZE"),
        (4, "FINALIZATION"),
        (8, "PRIORITY"),
        (24, "NORMAL"),
    ]
    for hours, mode in expected:
        assert DeadlineEvaluator.evaluate(now + timedelta(hours=hours), now)[1] == mode


def test_security_scanner_recomputes_duplicate_hidden_missing_and_size_findings(tmp_path) -> None:
    _, _, candidate, payloads = candidate_fixture(
        tmp_path,
        paper_text="C:\\Users\\alice\\input.csv token=secretValue123",
    )
    original = candidate.artifacts[0]
    duplicate = original.model_copy(
        update={
            "artifact_id": uuid4(),
            "relative_path": original.relative_path,
            "size_bytes": original.size_bytes + 1,
        }
    )
    hidden = original.model_copy(
        update={
            "artifact_id": uuid4(),
            "relative_path": ".env",
        }
    )
    candidate = candidate.model_copy(update={"artifacts": [original, duplicate, hidden]})
    findings = SubmissionSecurityScanner().scan(candidate, payloads)
    codes = {item[0] for item in findings}
    assert {
        "ABSOLUTE_PATH_LEAK",
        "SECRET_LEAK",
        "DUPLICATE_PATH",
        "FORBIDDEN_FILE",
        "MISSING_ARTIFACT_BYTES",
    } <= codes

    payloads[duplicate.artifact_id] = payloads[original.artifact_id]
    codes = {item[0] for item in SubmissionSecurityScanner().scan(candidate, payloads)}
    assert "ARTIFACT_SIZE_MISMATCH" in codes


def test_rule_engine_rejects_profile_mismatch_and_handles_unimplemented_severity(tmp_path) -> None:
    store, profile, candidate, _ = candidate_fixture(tmp_path)
    with pytest.raises(ValueError, match="different competition profile"):
        RuleEngine(store).evaluate(
            profile,
            candidate.model_copy(update={"competition_profile_version": 2}),
        )
    blocking = _rule(profile, RuleType.CUSTOM)
    minor = blocking.model_copy(
        update={"rule_id": "RULE-custom-minor", "severity": RuleSeverity.MINOR}
    )
    profile = profile.model_copy(update={"rules": [blocking, minor]})
    statuses = [item.status for item in RuleEngine(store).evaluate(profile, candidate)]
    assert statuses == [RuleResultStatus.HUMAN_REVIEW, RuleResultStatus.NOT_EVALUABLE]


@pytest.mark.parametrize(
    ("parameters", "expected"),
    [
        ({"max_pages": 20, "scope": PageScope.MAIN_TEXT.value}, RuleResultStatus.HUMAN_REVIEW),
        ({"max_pages": 0, "scope": PageScope.TOTAL.value}, RuleResultStatus.NOT_EVALUABLE),
    ],
)
def test_page_rule_honestly_handles_unsupported_scope_and_missing_limit(
    tmp_path, parameters, expected
) -> None:
    store, profile, candidate, payloads = candidate_fixture(tmp_path)
    profile = profile.model_copy(
        update={"rules": [_rule(profile, RuleType.PAGE_LIMIT, parameters=parameters)]}
    )
    assert (
        RuleEngine(store).evaluate(profile, candidate, artifact_bytes=payloads)[0].status
        is expected
    )


def test_page_and_file_format_reject_missing_or_corrupt_pdf(tmp_path) -> None:
    store, profile, candidate, payloads = candidate_fixture(tmp_path)
    no_pdf = candidate.model_copy(update={"artifacts": []})
    page = _rule(profile, RuleType.PAGE_LIMIT, parameters={"max_pages": 20, "scope": "TOTAL"})
    profile = profile.model_copy(update={"rules": [page]})
    assert (
        RuleEngine(store).evaluate(profile, no_pdf, artifact_bytes={})[0].status
        is RuleResultStatus.FAIL
    )

    pdf = candidate.artifacts[0]
    payloads[pdf.artifact_id] = b"%PDF-corrupt"
    assert (
        RuleEngine(store).evaluate(profile, candidate, artifact_bytes=payloads)[0].status
        is RuleResultStatus.FAIL
    )
    file_format = _rule(
        profile,
        RuleType.FILE_FORMAT,
        parameters={"role": "CODE", "mime_type": "text/x-python"},
    )
    profile = profile.model_copy(update={"rules": [file_format]})
    assert (
        RuleEngine(store).evaluate(profile, candidate, artifact_bytes=payloads)[0].status
        is RuleResultStatus.FAIL
    )


def test_file_count_size_filename_and_content_rules_cover_failure_branches(tmp_path) -> None:
    store, profile, candidate, payloads = candidate_fixture(tmp_path)
    rules = [
        _rule(profile, RuleType.FILE_COUNT, parameters={"maximum": 0}),
        _rule(profile, RuleType.FILE_SIZE, parameters={"maximum": 1}),
        _rule(
            profile,
            RuleType.FILENAME,
            parameters={"role": "PAPER_PDF", "prefix": "result-"},
        ),
        _rule(
            profile,
            RuleType.REQUIRED_SECTION,
            parameters={"sections": ["CONCLUSION"]},
        ),
        _rule(
            profile,
            RuleType.PROHIBITED_CONTENT,
            parameters={"terms": ["verified results"]},
        ),
    ]
    profile = profile.model_copy(update={"rules": rules})
    assert all(
        item.status is RuleResultStatus.FAIL
        for item in RuleEngine(store).evaluate(profile, candidate, artifact_bytes=payloads)
    )


def test_content_entrypoint_ai_and_deadline_rule_branches(tmp_path) -> None:
    store, profile, candidate, payloads = candidate_fixture(
        tmp_path,
        extra_files=[
            (SubmissionArtifactRole.CODE, "code/other.py", b"print(1)\n", "text/x-python"),
            (SubmissionArtifactRole.README, "README.md", b"reproduce\n", "text/markdown"),
        ],
    )
    code_policy = profile.code_submission_rules.model_copy(update={"entrypoint_required": True})
    code_rule = _rule(
        profile, RuleType.CODE_SUBMISSION, parameters={"policy": InclusionPolicy.ALLOWED.value}
    )
    profile = profile.model_copy(
        update={"code_submission_rules": code_policy, "rules": [code_rule]}
    )
    result = RuleEngine(store).evaluate(profile, candidate, artifact_bytes=payloads)[0]
    assert result.status is RuleResultStatus.FAIL
    assert "entrypoint" in result.message

    ai_rule = _rule(profile, RuleType.AI_DISCLOSURE)
    unknown = profile.ai_disclosure_rules.model_copy(update={"policy": AIDisclosurePolicy.UNKNOWN})
    profile = profile.model_copy(update={"ai_disclosure_rules": unknown, "rules": [ai_rule]})
    assert (
        RuleEngine(store).evaluate(profile, candidate, artifact_bytes=payloads)[0].status
        is RuleResultStatus.HUMAN_REVIEW
    )

    required = unknown.model_copy(
        update={"policy": AIDisclosurePolicy.REQUIRED, "required_text": "AI was used"}
    )
    profile = profile.model_copy(update={"ai_disclosure_rules": required})
    assert (
        RuleEngine(store).evaluate(profile, candidate, artifact_bytes=payloads)[0].status
        is RuleResultStatus.FAIL
    )
    disclosed = candidate.model_copy(update={"paper_text": "AI was used under the rules."})
    assert (
        RuleEngine(store).evaluate(profile, disclosed, artifact_bytes=payloads)[0].status
        is RuleResultStatus.PASS
    )

    deadline_rule = _rule(profile, RuleType.DEADLINE)
    profile = profile.model_copy(update={"deadline": None, "rules": [deadline_rule]})
    assert (
        RuleEngine(store).evaluate(profile, candidate, artifact_bytes=payloads)[0].status
        is RuleResultStatus.HUMAN_REVIEW
    )
    now = datetime.now(UTC)
    profile = profile.model_copy(update={"deadline": now - timedelta(seconds=1)})
    assert (
        RuleEngine(store).evaluate(profile, candidate, artifact_bytes=payloads, now=now)[0].status
        is RuleResultStatus.FAIL
    )


def test_mime_detection_covers_supported_signatures_and_binary() -> None:
    assert detect_mime(b"PK\x03\x04") == "application/zip"
    assert detect_mime(b"\x89PNG\r\n\x1a\n") == "image/png"
    assert detect_mime(b"\xff\xd8\xff") == "image/jpeg"
    assert detect_mime(b"plain text") == "text/plain"
    assert detect_mime(b"\x00binary") is None


def test_submission_check_recomputes_issue_severity_counts_and_identity(tmp_path) -> None:
    _, profile, candidate, _payloads = candidate_fixture(tmp_path)
    blocking_fail = RuleResult(
        rule_id="RULE-block",
        rule_type=RuleType.CUSTOM,
        severity=RuleSeverity.BLOCKING,
        status=RuleResultStatus.FAIL,
        message="blocking",
    )
    minor_fail = blocking_fail.model_copy(
        update={
            "rule_id": "RULE-minor",
            "severity": RuleSeverity.MINOR,
            "message": "minor",
        }
    )
    failed_candidate = candidate.model_copy(
        update={
            "phase5_verified": False,
            "paper_status": PaperQualityStatus.FAILED,
            "paper_manifest_hash": "0" * 64,
        }
    )
    jury = _report(profile, failed_candidate, [blocking_fail, minor_fail])
    check = SubmissionCheck().run(
        profile=profile,
        candidate=failed_candidate,
        requirements=_coverage(RequirementCoverageStatus.PARTIAL),
        rules=[blocking_fail, minor_fail],
        jury=jury,
        artifact_bytes={},
    )
    codes = {item.code for item in check.issues}
    assert {
        "RESULT_NOT_VERIFIED",
        "PAPER_NOT_READY",
        "PAPER_MANIFEST_INVALID",
        "BLOCKING_RULE_FAILED",
        "COMPETITION_RULE_FAILED",
        "REQUIREMENT_NOT_COVERED",
        "FINAL_JURY_NOT_PASS",
        "ARTIFACT_INTEGRITY_FAILED",
    } <= codes
    assert check.status is SubmissionCheckStatus.FAIL
    assert (
        SubmissionCheck().verify(
            check.model_copy(update={"project_id": uuid4()}),
            profile=profile,
            candidate=failed_candidate,
            requirements=_coverage(RequirementCoverageStatus.PARTIAL),
            rules=[blocking_fail, minor_fail],
            jury=jury,
            artifact_bytes={},
        )
        is SubmissionCheckStatus.FAIL
    )


def test_nonblocking_rule_failure_produces_warning_not_ready(tmp_path) -> None:
    _, profile, candidate, payloads = candidate_fixture(tmp_path)
    warning_rule = RuleResult(
        rule_id="RULE-minor",
        rule_type=RuleType.CUSTOM,
        severity=RuleSeverity.MINOR,
        status=RuleResultStatus.FAIL,
        message="minor compliance issue",
    )
    jury = _report(profile, candidate, [warning_rule])
    check = SubmissionCheck().run(
        profile=profile,
        candidate=candidate,
        requirements=_coverage(),
        rules=[warning_rule],
        jury=jury,
        artifact_bytes=payloads,
    )
    assert jury.decision is JuryDecision.PASS
    assert check.status is SubmissionCheckStatus.WARNING


def test_submission_check_rejects_recomputed_count_issue_and_status_tampering(tmp_path) -> None:
    _, profile, candidate, payloads = candidate_fixture(tmp_path)
    jury = _report(profile, candidate, [])
    checker = SubmissionCheck()
    check = checker.run(
        profile=profile,
        candidate=candidate,
        requirements=_coverage(),
        rules=[],
        jury=jury,
        artifact_bytes=payloads,
    )
    injected = SubmissionCheckIssue(
        code="TAMPERED",
        category=SubmissionCheckCategory.FILES,
        severity=RuleSeverity.MAJOR,
        message="persisted issue list was changed",
    )
    variants = [
        check.model_copy(update={"claimed_failed_rule_count": 1}),
        check.model_copy(update={"issues": [injected]}),
        check.model_copy(update={"status": SubmissionCheckStatus.WARNING}),
    ]
    for tampered in variants:
        assert (
            checker.verify(
                tampered,
                profile=profile,
                candidate=candidate,
                requirements=_coverage(),
                rules=[],
                jury=jury,
                artifact_bytes=payloads,
            )
            is SubmissionCheckStatus.FAIL
        )


def test_submission_check_recomputes_forbidden_and_outside_allowlist(tmp_path) -> None:
    _, profile, candidate, payloads = candidate_fixture(tmp_path)
    hidden = candidate.artifacts[0].model_copy(update={"relative_path": ".env"})
    candidate = candidate.model_copy(update={"artifacts": [hidden]})
    jury = _report(profile, candidate, [])
    check = SubmissionCheck().run(
        profile=profile,
        candidate=candidate,
        requirements=_coverage(),
        rules=[],
        jury=jury,
        artifact_bytes=payloads,
    )
    assert {item.code for item in check.issues} >= {"FILE_NOT_ALLOWED", "FORBIDDEN_FILE"}
