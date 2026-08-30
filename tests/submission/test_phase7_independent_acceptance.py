from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from io import BytesIO
from threading import Barrier
from types import SimpleNamespace
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
from pydantic import ValidationError
from pypdf import PdfWriter

from mathmodel_ai.paper.hashing import sha256_bytes
from mathmodel_ai.schemas.submission import (
    CompetitionProfile,
    DeadlineMode,
    DeadlineStatus,
    JuryDecision,
    JuryFinding,
    JurySeverity,
    ProfileVerificationStatus,
    RequirementCoverageStatus,
    RuleResultStatus,
    RuleSeverity,
    RuleType,
    SubmissionArtifactRole,
    SubmissionCheckStatus,
    SubmissionManifest,
    SubmissionStatus,
)
from mathmodel_ai.submission.checks import SubmissionCheck
from mathmodel_ai.submission.corrections import InvalidationEngine
from mathmodel_ai.submission.freeze import SubmissionFreeze
from mathmodel_ai.submission.integrity import (
    CompetitionProfileIntegrityError,
    competition_profile_digest,
    submission_snapshot_digest,
    validate_competition_profile,
)
from mathmodel_ai.submission.jury import FinalJuryGate
from mathmodel_ai.submission.package import (
    SubmissionIntegrityVerifier,
    SubmissionPackageBuilder,
)
from mathmodel_ai.submission.profiles import (
    CompetitionProfileRegistry,
    generic_modeling_test_profile,
)
from mathmodel_ai.submission.requirements import (
    RequirementCoverageValidator,
    RequirementRegistry,
)
from mathmodel_ai.submission.rules import (
    DeadlineEvaluator,
    RuleEngine,
    SubmissionSecurityScanner,
)
from tests.submission.helpers import (
    candidate_fixture,
    covered_paper,
    jury_draft,
    subproblem,
)


def _accepted_chain(tmp_path, *, profile: CompetitionProfile | None = None):
    store, actual_profile, candidate, payloads = candidate_fixture(tmp_path, profile=profile)
    paper = covered_paper(("Q1", "answer"))
    requirements = RequirementRegistry.from_subproblems([subproblem("Q1", "answer")])
    coverage = RequirementCoverageValidator().validate(requirements, paper, candidate.artifacts)
    rules = RuleEngine(store).evaluate(actual_profile, candidate, artifact_bytes=payloads)
    jury = FinalJuryGate().build_report(
        draft=jury_draft(),
        profile=actual_profile,
        candidate=candidate,
        requirements=coverage,
        rule_results=rules,
        reviewer_is_mock=True,
        agent_run_id=uuid4(),
    )
    check = SubmissionCheck().run(
        profile=actual_profile,
        candidate=candidate,
        requirements=coverage,
        rules=rules,
        jury=jury,
        artifact_bytes=payloads,
    )
    assert jury.decision is JuryDecision.PASS
    assert check.status is SubmissionCheckStatus.PASS
    package = SubmissionFreeze(SubmissionPackageBuilder(store), RuleEngine(store)).freeze(
        profile=actual_profile,
        candidate=candidate,
        paper=paper,
        submission_requirements=requirements,
        requirements=coverage,
        rules=rules,
        jury=jury,
        check=check,
        artifact_bytes=payloads,
    )
    return (
        store,
        actual_profile,
        candidate,
        payloads,
        paper,
        requirements,
        coverage,
        rules,
        jury,
        check,
        package,
    )


def _pdf_with_hidden_content() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.add_metadata(
        {
            "/Author": "Student Name",
            "/Creator": "file:///C:/Users/zeon/project",
        }
    )
    writer.add_attachment("hidden.txt", b"Authorization: Bearer abcdefghijklmnop")
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def _replace_pdf(store, candidate, data: bytes):
    old = candidate.artifacts[0]
    store.resolve(old.storage_key).write_bytes(data)
    artifact = old.model_copy(update={"size_bytes": len(data), "sha256": sha256_bytes(data)})
    return candidate.model_copy(update={"artifacts": [artifact]}), {artifact.artifact_id: data}


def test_profile_versions_are_exact_snapshots_and_registry_returns_copies() -> None:
    profile_v1 = generic_modeling_test_profile()
    page_rule = next(item for item in profile_v1.rules if item.rule_type is RuleType.PAGE_LIMIT)
    profile_v2 = profile_v1.model_copy(
        update={
            "version": 2,
            "page_rules": profile_v1.page_rules.model_copy(update={"max_pages": 25}),
            "rules": [
                item.model_copy(update={"parameters": {"max_pages": 25, "scope": "TOTAL"}})
                if item.rule_id == page_rule.rule_id
                else item
                for item in profile_v1.rules
            ],
        }
    )
    registry = CompetitionProfileRegistry([profile_v1, profile_v2])
    returned = registry.get(profile_v1.profile_id, 1)
    returned.page_rules.max_pages = 200
    assert registry.get(profile_v1.profile_id, 1).page_rules.max_pages == 20
    assert registry.get(profile_v1.profile_id, 2).page_rules.max_pages == 25


def test_fixture_profile_cannot_be_promoted_without_real_provenance() -> None:
    profile = generic_modeling_test_profile()
    promoted = profile.model_copy(
        update={
            "verification_status": ProfileVerificationStatus.VERIFIED,
            "rules": [
                item.model_copy(update={"verification_status": ProfileVerificationStatus.VERIFIED})
                for item in profile.rules
            ],
        }
    )
    with pytest.raises(CompetitionProfileIntegrityError, match="fixture provenance"):
        validate_competition_profile(promoted)


@pytest.mark.parametrize("attack", ["negative_pages", "nan_rule", "unsafe_regex", "path"])
def test_malicious_profile_values_are_schema_rejected(attack) -> None:
    payload = generic_modeling_test_profile().model_dump(mode="python")
    if attack == "negative_pages":
        payload["page_rules"]["max_pages"] = -1
    elif attack == "nan_rule":
        payload["rules"][0]["parameters"]["max_pages"] = float("nan")
    elif attack == "unsafe_regex":
        payload["naming_rules"]["allowed_pattern"] = r"^(a+)+$"
    else:
        payload["allowed_submission_files"].append("../secret")
    with pytest.raises(ValidationError):
        CompetitionProfile.model_validate(payload)


def test_profile_rule_provenance_and_stale_candidate_digest_fail_closed(tmp_path) -> None:
    profile = generic_modeling_test_profile()
    broken_rule = profile.rules[0].model_copy(update={"source_ref": "fixture://missing"})
    broken = profile.model_copy(update={"rules": [broken_rule, *profile.rules[1:]]})
    with pytest.raises(CompetitionProfileIntegrityError, match="profile provenance"):
        validate_competition_profile(broken)

    store, _, candidate, _ = candidate_fixture(tmp_path)
    changed = profile.model_copy(update={"minimum_jury_score": 74})
    assert candidate.competition_profile_digest != competition_profile_digest(changed)
    with pytest.raises(ValueError, match="profile digest is stale"):
        SubmissionPackageBuilder(store).build(profile=changed, candidate=candidate)


def test_irrelevant_claim_cannot_cover_risk_requirement(tmp_path) -> None:
    _, _, candidate, _ = candidate_fixture(tmp_path)
    requirements = RequirementRegistry.from_subproblems(
        [subproblem("Q1", "evaluate final solution risk")]
    )
    paper = covered_paper(("Q1", "evaluate final solution risk"))
    coverage = RequirementCoverageValidator().validate(requirements, paper, candidate.artifacts)
    assert coverage[0].status is RequirementCoverageStatus.PARTIAL
    assert "supported evidence-linked claim" in coverage[0].reasons[-1]


def test_required_not_applicable_cannot_pass_final_jury(tmp_path) -> None:
    store, profile, candidate, payloads = candidate_fixture(tmp_path)
    paper = covered_paper(("Q1", "answer"))
    requirements = RequirementRegistry.from_subproblems([subproblem("Q1", "answer")])
    coverage = RequirementCoverageValidator().validate(requirements, paper, candidate.artifacts)
    tampered = [coverage[0].model_copy(update={"status": RequirementCoverageStatus.NOT_APPLICABLE})]
    rules = RuleEngine(store).evaluate(profile, candidate, artifact_bytes=payloads)
    report = FinalJuryGate().build_report(
        draft=jury_draft(),
        profile=profile,
        candidate=candidate,
        requirements=tampered,
        rule_results=rules,
        reviewer_is_mock=True,
        agent_run_id=uuid4(),
    )
    assert report.decision is JuryDecision.FAIL


def test_jury_is_invalid_after_paper_content_or_finding_change(tmp_path) -> None:
    (
        _,
        profile,
        candidate,
        _,
        _,
        _,
        coverage,
        rules,
        jury,
        _,
        _,
    ) = _accepted_chain(tmp_path)
    changed_candidate = candidate.model_copy(update={"paper_text": "Changed PaperIR content"})
    assert (
        FinalJuryGate().verify_report(
            jury,
            profile=profile,
            candidate=changed_candidate,
            requirements=coverage,
            rule_results=rules,
        )
        is JuryDecision.FAIL
    )
    minor = JuryFinding(
        finding_id="JURY-minor",
        category="writing",
        severity=JurySeverity.MINOR,
        title="Minor wording issue",
        description="A nonblocking wording issue remains.",
        impact="Presentation only.",
        recommendation="Polish wording.",
        confidence=1,
    )
    report_with_finding = FinalJuryGate().build_report(
        draft=jury_draft().model_copy(update={"findings": [minor]}),
        profile=profile,
        candidate=candidate,
        requirements=coverage,
        rule_results=rules,
        reviewer_is_mock=True,
        agent_run_id=uuid4(),
    )
    assert report_with_finding.decision is JuryDecision.PASS
    deleted_finding = report_with_finding.model_copy(update={"findings": []})
    assert (
        FinalJuryGate().verify_report(
            deleted_finding,
            profile=profile,
            candidate=candidate,
            requirements=coverage,
            rule_results=rules,
        )
        is JuryDecision.FAIL
    )


def test_freeze_recomputes_requirement_coverage_from_paper(tmp_path) -> None:
    (
        store,
        profile,
        candidate,
        payloads,
        paper,
        requirements,
        coverage,
        rules,
        jury,
        check,
        _,
    ) = _accepted_chain(tmp_path)
    paper_without_mapping = paper.model_copy(update={"subproblem_coverage": []})
    with pytest.raises(ValueError, match="coverage changed before freeze"):
        SubmissionFreeze(SubmissionPackageBuilder(store), RuleEngine(store)).freeze(
            profile=profile,
            candidate=candidate,
            paper=paper_without_mapping,
            submission_requirements=requirements,
            requirements=coverage,
            rules=rules,
            jury=jury,
            check=check,
            artifact_bytes=payloads,
        )


def test_pdf_metadata_attachment_unicode_paths_and_secrets_are_scanned(tmp_path) -> None:
    store, _, candidate, _ = candidate_fixture(tmp_path / "pdf")
    candidate, payloads = _replace_pdf(store, candidate, _pdf_with_hidden_content())
    scanner = SubmissionSecurityScanner()
    codes = {item[0] for item in scanner.scan(candidate, payloads)}
    assert {"ABSOLUTE_PATH_LEAK", "SECRET_LEAK", "EMBEDDED_ATTACHMENT"} <= codes
    findings = scanner.anonymity_findings(candidate, payloads, ["Student Name"])
    assert {"Student Name", "identity label", "local filesystem path"} <= set(findings)

    _, _, obfuscated, obfuscated_payloads = candidate_fixture(
        tmp_path / "unicode",
        paper_text=(
            "Contact z\u200beon\uff20example.com; "
            "\uff23\uff1a\uff0f\uff35\uff53\uff45\uff52\uff53\uff0fzeon\uff0fwork; "
            "Authorization: Bearer abcdefghijklmnop"
        ),
    )
    codes = {item[0] for item in scanner.scan(obfuscated, obfuscated_payloads)}
    assert {"ABSOLUTE_PATH_LEAK", "SECRET_LEAK"} <= codes
    assert "email address" in scanner.anonymity_findings(obfuscated, obfuscated_payloads, [])
    _, _, url_candidate, url_payloads = candidate_fixture(
        tmp_path / "url", paper_text="Source: https://example.invalid/fixture"
    )
    assert "ABSOLUTE_PATH_LEAK" not in {
        item[0] for item in scanner.scan(url_candidate, url_payloads)
    }


def test_internal_fixture_and_mock_indicator_cannot_enter_package(tmp_path) -> None:
    _, _, candidate, payloads = candidate_fixture(
        tmp_path,
        extra_files=[
            (
                SubmissionArtifactRole.DATA,
                "data/mock_result.json",
                b'{"is_mock": true}',
                "application/json",
            )
        ],
    )
    findings = SubmissionSecurityScanner().scan(candidate, payloads)
    assert ("FORBIDDEN_FILE", "data/mock_result.json") in findings
    assert ("MOCK_INDICATOR", "data/mock_result.json") in findings


@pytest.mark.parametrize(
    "secret",
    [
        "DATABASE_URL=postgresql://user:secret@db.internal/model",
        "password=hunter2",
        "-----BEGIN PRIVATE KEY-----",
    ],
)
def test_common_secret_variants_are_blocking(tmp_path, secret) -> None:
    _, _, candidate, payloads = candidate_fixture(tmp_path, paper_text=secret)
    assert "SECRET_LEAK" in {
        item[0] for item in SubmissionSecurityScanner().scan(candidate, payloads)
    }


def test_package_hash_and_physical_zip_are_deterministic(tmp_path) -> None:
    (
        store,
        profile,
        candidate,
        payloads,
        paper,
        requirements,
        coverage,
        rules,
        jury,
        check,
        first,
    ) = _accepted_chain(tmp_path)
    freeze = SubmissionFreeze(SubmissionPackageBuilder(store), RuleEngine(store))
    second = freeze.freeze(
        profile=profile,
        candidate=candidate,
        paper=paper,
        submission_requirements=requirements,
        requirements=coverage,
        rules=rules,
        jury=jury,
        check=check,
        artifact_bytes=payloads,
    )
    repeated = freeze.freeze(
        profile=profile,
        candidate=candidate,
        paper=paper,
        submission_requirements=requirements,
        requirements=coverage,
        rules=rules,
        jury=jury,
        check=check,
        artifact_bytes=payloads,
    )
    assert repeated is second
    assert first.snapshot.package_hash == second.snapshot.package_hash
    assert store.read_bytes(first.package_artifact.storage_key) == store.read_bytes(
        second.package_artifact.storage_key
    )

    concurrent_freeze = SubmissionFreeze(SubmissionPackageBuilder(store), RuleEngine(store))
    barrier = Barrier(2)

    def run_freeze():
        barrier.wait()
        return concurrent_freeze.freeze(
            profile=profile,
            candidate=candidate,
            paper=paper,
            submission_requirements=requirements,
            requirements=coverage,
            rules=rules,
            jury=jury,
            check=check,
            artifact_bytes=payloads,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        concurrent = list(executor.map(lambda _index: run_freeze(), range(2)))
    assert concurrent[0] is concurrent[1]


def test_snapshot_status_and_manifest_file_identity_are_digest_protected(tmp_path) -> None:
    store, *_, package = _accepted_chain(tmp_path)
    changed = package.snapshot.model_copy(update={"status": SubmissionStatus.DRAFT})
    changed = changed.model_copy(update={"snapshot_digest": submission_snapshot_digest(changed)})
    assert (
        SubmissionIntegrityVerifier(store).verify(
            package.__class__(
                profile=package.profile,
                snapshot=changed,
                manifest=package.manifest,
                manifest_artifact=package.manifest_artifact,
                package_artifact=package.package_artifact,
                protected_artifacts=package.protected_artifacts,
            )
        )
        is SubmissionStatus.DIRTY
    )
    payload = package.manifest.model_dump(mode="json")
    payload["files"].append(payload["files"][0])
    with pytest.raises(ValidationError, match="manifest file paths"):
        SubmissionManifest.model_validate(payload)


def test_archive_nested_case_unicode_and_traversal_entries_fail() -> None:
    variants = [
        [("paper.pdf", b"one"), ("Paper.pdf", b"two")],
        [("caf\u00e9.txt", b"one"), ("cafe\u0301.txt", b"two")],
        [("nested.zip", b"PK\x03\x04nested")],
        [("../evil.txt", b"bad")],
        [("C:\\absolute.txt", b"bad")],
    ]
    for entries in variants:
        output = BytesIO()
        with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
            for name, data in entries:
                archive.writestr(name, data)
        with pytest.raises(ValueError):
            SubmissionIntegrityVerifier._verify_zip(output.getvalue())


def test_windows_reparse_point_source_fails_closed(tmp_path, monkeypatch) -> None:
    store, profile, candidate, payloads = candidate_fixture(tmp_path)
    fake_path = SimpleNamespace(
        is_symlink=lambda: False,
        lstat=lambda: SimpleNamespace(st_file_attributes=0x400),
    )
    monkeypatch.setattr(store, "resolve", lambda _key: fake_path)
    with pytest.raises(ValueError, match="reparse points"):
        SubmissionPackageBuilder(store)._validate_sources(
            profile, candidate, candidate.artifacts, payloads
        )


def test_deadline_is_rechecked_when_frozen_package_is_reopened(tmp_path, monkeypatch) -> None:
    profile = generic_modeling_test_profile()
    deadline_rule = profile.rules[0].model_copy(
        update={
            "rule_id": "RULE-deadline",
            "rule_type": RuleType.DEADLINE,
            "parameters": {},
        }
    )
    profile = profile.model_copy(
        update={
            "deadline": datetime.now(UTC) + timedelta(hours=1),
            "timezone": "UTC",
            "rules": [*profile.rules, deadline_rule],
        }
    )
    store, *_, package = _accepted_chain(tmp_path, profile=profile)
    monkeypatch.setattr(
        DeadlineEvaluator,
        "evaluate",
        staticmethod(
            lambda _deadline, _now=None: (
                DeadlineStatus.EXPIRED,
                DeadlineMode.SUBMISSION_MODE,
            )
        ),
    )
    assert SubmissionIntegrityVerifier(store).verify(package) is SubmissionStatus.DIRTY


def test_reproduction_required_profile_is_fail_closed_without_real_handler(tmp_path) -> None:
    profile = generic_modeling_test_profile().model_copy(
        update={"special_rules": {"reproduction_required": True}}
    )
    with pytest.raises(CompetitionProfileIntegrityError, match="reproduction"):
        validate_competition_profile(profile)

    custom = profile.rules[0].model_copy(
        update={
            "rule_id": "RULE-reproduction",
            "rule_type": RuleType.CUSTOM,
            "severity": RuleSeverity.BLOCKING,
            "parameters": {"handler": "REPRODUCTION"},
        }
    )
    profile = profile.model_copy(update={"rules": [*profile.rules, custom]})
    store, profile, candidate, payloads = candidate_fixture(tmp_path, profile=profile)
    result = next(
        item
        for item in RuleEngine(store).evaluate(profile, candidate, artifact_bytes=payloads)
        if item.rule_id == custom.rule_id
    )
    assert result.status is RuleResultStatus.HUMAN_REVIEW


def test_format_label_cannot_hide_equation_change() -> None:
    from mathmodel_ai.schemas.submission import CorrectionPlan, CorrectionScope

    plan = CorrectionPlan(
        project_id=uuid4(),
        finding_refs=["JURY-equation"],
        scope=CorrectionScope.FORMAT_ONLY,
        affected_components=["equation"],
        requires_model_change=False,
        requires_result_change=False,
        requires_paper_change=True,
        required_revalidation=["PAPER", "FINAL_JURY", "SUBMISSION"],
        risk=RuleSeverity.BLOCKING,
        priority=1,
    )
    with pytest.raises(ValueError, match="FORMAT_ONLY classification"):
        InvalidationEngine().evaluate(plan)
