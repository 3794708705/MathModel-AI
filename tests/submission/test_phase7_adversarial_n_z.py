from __future__ import annotations

import stat
from datetime import UTC, datetime, timedelta
from io import BytesIO
from types import SimpleNamespace
from uuid import UUID, uuid4
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import pytest

from mathmodel_ai.paper.hashing import sha256_bytes
from mathmodel_ai.schemas.paper import PaperQualityStatus, PaperVersion, PaperVersionRef
from mathmodel_ai.schemas.submission import (
    CorrectionPlan,
    CorrectionScope,
    JuryDecision,
    RequirementCoverage,
    RequirementCoverageStatus,
    RuleResultStatus,
    RuleType,
    SubmissionArtifactRole,
    SubmissionCheckStatus,
    SubmissionStatus,
)
from mathmodel_ai.submission.checks import SubmissionCheck
from mathmodel_ai.submission.corrections import CorrectionWorkflow, InvalidationEngine
from mathmodel_ai.submission.jury import FinalJuryGate
from mathmodel_ai.submission.package import (
    SubmissionIntegrityVerifier,
    SubmissionPackageBuilder,
)
from mathmodel_ai.submission.rules import RuleEngine
from mathmodel_ai.submission.workflow import FinalSubmissionError, FinalSubmissionWorkflow
from tests.submission.helpers import candidate_fixture, covered_paper, jury_draft


def _coverage() -> list[RequirementCoverage]:
    return [
        RequirementCoverage(
            requirement_id="REQ-Q1-1",
            requirement_digest="a" * 64,
            subproblem_id="Q1",
            status=RequirementCoverageStatus.COVERED,
            evidence_refs=[UUID(int=0x7101)],
            paper_refs=["SEC-results"],
            artifact_refs=[UUID(int=0x7102)],
        )
    ]


def _accepted_chain(tmp_path, *, extra_files=None):
    store, profile, candidate, payloads = candidate_fixture(tmp_path, extra_files=extra_files)
    rules = RuleEngine(store).evaluate(profile, candidate, artifact_bytes=payloads)
    jury = FinalJuryGate().build_report(
        draft=jury_draft(),
        profile=profile,
        candidate=candidate,
        requirements=_coverage(),
        rule_results=rules,
        reviewer_is_mock=True,
        agent_run_id=uuid4(),
    )
    check = SubmissionCheck().run(
        profile=profile,
        candidate=candidate,
        requirements=_coverage(),
        rules=rules,
        jury=jury,
        artifact_bytes=payloads,
    )
    assert jury.decision is JuryDecision.PASS
    assert check.status is SubmissionCheckStatus.PASS
    package = SubmissionPackageBuilder(store).build(
        profile=profile,
        candidate=candidate,
        requirements=_coverage(),
        jury=jury,
        check=check,
    )
    return store, profile, candidate, payloads, rules, jury, check, package


def test_n_freeze_success_creates_snapshot_manifest_and_package(tmp_path) -> None:
    _, _, _, _, _, _, _, package = _accepted_chain(tmp_path)
    assert package.snapshot.status is SubmissionStatus.FROZEN
    assert package.snapshot.manifest_hash == package.manifest.manifest_hash
    assert package.snapshot.package_hash == package.manifest.package_hash


@pytest.mark.parametrize(
    ("case", "role", "path", "data", "mime_type"),
    [
        ("O", SubmissionArtifactRole.PAPER_PDF, "paper.pdf", None, "application/pdf"),
        (
            "P",
            SubmissionArtifactRole.FIGURE,
            "figures/chart.png",
            b"\x89PNG\r\n\x1a\nchart",
            "image/png",
        ),
        ("Q", SubmissionArtifactRole.CODE, "code/main.py", b"print(1)\n", "text/x-python"),
        ("R", SubmissionArtifactRole.DATA, "data/input.csv", b"x\n1\n", "text/csv"),
    ],
)
def test_o_to_r_protected_artifact_tamper_marks_dirty(
    tmp_path, case, role, path, data, mime_type
) -> None:
    extras = [] if case == "O" else [(role, path, data, mime_type)]
    store, _, _, _, _, _, _, package = _accepted_chain(tmp_path, extra_files=extras)
    target = next(item for item in package.protected_artifacts if item.role is role)
    store.resolve(target.storage_key).write_bytes(b"tampered-after-freeze")
    assert SubmissionIntegrityVerifier(store).verify(package) is SubmissionStatus.DIRTY


def test_s_manifest_tamper_marks_dirty(tmp_path) -> None:
    store, _, _, _, _, _, _, package = _accepted_chain(tmp_path)
    store.resolve(package.manifest_artifact.storage_key).write_bytes(b"{}")
    assert SubmissionIntegrityVerifier(store).verify(package) is SubmissionStatus.DIRTY


def test_t_package_swap_marks_dirty(tmp_path) -> None:
    store, _, _, _, _, _, _, package = _accepted_chain(tmp_path / "one")
    other_store, _, _, _, _, _, _, other = _accepted_chain(tmp_path / "two")
    swapped = other_store.read_bytes(other.package_artifact.storage_key)
    store.resolve(package.package_artifact.storage_key).write_bytes(swapped)
    assert SubmissionIntegrityVerifier(store).verify(package) is SubmissionStatus.DIRTY


def test_u_forbidden_extra_file_rejected(tmp_path) -> None:
    store, profile, candidate, _ = candidate_fixture(
        tmp_path,
        extra_files=[(SubmissionArtifactRole.DATA, ".env", b"SAFE_TEST_VALUE=1\n", "text/plain")],
    )
    with pytest.raises(ValueError, match=r"not allowed|security scan"):
        SubmissionPackageBuilder(store).build(profile=profile, candidate=candidate)


def test_v_absolute_local_path_leak_rejected(tmp_path) -> None:
    store, profile, candidate, _ = candidate_fixture(
        tmp_path,
        extra_files=[
            (
                SubmissionArtifactRole.CODE,
                "code/main.py",
                b"SOURCE = r'C:\\Users\\alice\\private\\input.csv'\n",
                "text/x-python",
            )
        ],
    )
    with pytest.raises(ValueError, match="ABSOLUTE_PATH_LEAK"):
        SubmissionPackageBuilder(store).build(profile=profile, candidate=candidate)


def test_w_package_roundtrip_reopens_and_verifies(tmp_path) -> None:
    store, _, _, _, _, _, _, package = _accepted_chain(tmp_path)
    assert SubmissionIntegrityVerifier(store).verify(package) is SubmissionStatus.FROZEN
    with ZipFile(BytesIO(store.read_bytes(package.package_artifact.storage_key))) as archive:
        assert "paper.pdf" in archive.namelist()
        assert "submission_manifest.json" in archive.namelist()


def test_x_freeze_uses_explicit_approved_paper_version_not_latest() -> None:
    project_id = uuid4()
    problem_id = uuid4()
    result_id = uuid4()
    base = covered_paper(("Q1", "answer"))
    snapshot = base.evidence_snapshot.model_copy(update={"verified_result_id": result_id})
    claims = [item.model_copy(update={"paper_version": 2}) for item in base.claims]
    approved_ir = base.model_copy(
        update={
            "version": 2,
            "claims": claims,
            "evidence_snapshot": snapshot,
            "status": PaperQualityStatus.READY_FOR_FINAL_JURY,
        }
    )
    approved = PaperVersion(
        paper_id=approved_ir.paper_id,
        project_id=project_id,
        problem_id=problem_id,
        version=2,
        parent_version=1,
        revision_reason="approved historical version",
        evidence_snapshot=snapshot,
        paper_ir=approved_ir,
        status=PaperQualityStatus.READY_FOR_FINAL_JURY,
        manifest_hash="a" * 64,
    )
    approved_ref = PaperVersionRef(
        paper_id=approved.paper_id,
        version=2,
        verified_result_id=result_id,
        evidence_snapshot_hash=snapshot.snapshot_hash,
        status=PaperQualityStatus.READY_FOR_FINAL_JURY,
        manifest_hash="a" * 64,
    )
    latest_ref = approved_ref.model_copy(update={"version": 3})
    wrong_state = SimpleNamespace(verified_result_id=result_id, paper_versions=[latest_ref])
    with pytest.raises(FinalSubmissionError, match="explicit paper version"):
        FinalSubmissionWorkflow._validate_explicit_paper(wrong_state, approved)  # type: ignore[arg-type]
    approved_state = SimpleNamespace(verified_result_id=result_id, paper_versions=[approved_ref])
    FinalSubmissionWorkflow._validate_explicit_paper(approved_state, approved)  # type: ignore[arg-type]


def test_y_modification_requires_jury_recheck_and_marks_submission_dirty() -> None:
    plan = CorrectionPlan(
        project_id=uuid4(),
        finding_refs=["JURY-critical-claim"],
        scope=CorrectionScope.PAPER_ONLY,
        affected_components=["CLAIM-critical"],
        requires_model_change=False,
        requires_result_change=False,
        requires_paper_change=True,
        required_revalidation=["PAPER", "FINAL_JURY", "SUBMISSION"],
        risk="BLOCKING",
        priority=1,
    )
    invalidation = InvalidationEngine().evaluate(plan)
    assert invalidation.jury_status is JuryDecision.RECHECK_REQUIRED
    assert invalidation.submission_status is SubmissionStatus.DIRTY
    assert not CorrectionWorkflow().validate_auto_fix(plan)


def test_z_clean_final_submission_reaches_frozen(tmp_path) -> None:
    store, _, _, _, rules, jury, check, package = _accepted_chain(tmp_path)
    assert all(item.status is RuleResultStatus.PASS for item in rules)
    assert jury.decision is JuryDecision.PASS
    assert check.status is SubmissionCheckStatus.PASS
    assert SubmissionIntegrityVerifier(store).verify(package) is SubmissionStatus.FROZEN


def test_persisted_jury_score_and_submission_counts_cannot_bypass_gates(tmp_path) -> None:
    _, profile, candidate, payloads, rules, jury, check, _ = _accepted_chain(tmp_path)
    score_tampered = jury.model_copy(
        update={"claimed_score": 100 if jury.claimed_score < 100 else 0}
    )
    assert (
        FinalJuryGate().verify_report(
            score_tampered,
            profile=profile,
            candidate=candidate,
            requirements=_coverage(),
            rule_results=rules,
        )
        is JuryDecision.FAIL
    )
    count_tampered = check.model_copy(update={"claimed_missing_requirement_count": 1})
    assert (
        SubmissionCheck().verify(
            count_tampered,
            profile=profile,
            candidate=candidate,
            requirements=_coverage(),
            rules=rules,
            jury=jury,
            artifact_bytes=payloads,
        )
        is SubmissionCheckStatus.FAIL
    )


def test_deadline_mode_never_skips_anonymity_or_secret_checks(tmp_path) -> None:
    store, profile, candidate, payloads = candidate_fixture(
        tmp_path, paper_text="Author: Alice api_key=sk-testOnlyCredential123456789"
    )
    deadline_rule = next(item for item in profile.rules if item.rule_type is RuleType.PAGE_LIMIT)
    deadline_rule = deadline_rule.model_copy(
        update={
            "rule_id": "RULE-deadline",
            "rule_type": RuleType.DEADLINE,
            "parameters": {},
        }
    )
    anonymity_rule = next(item for item in profile.rules if item.rule_type is RuleType.ANONYMITY)
    now = datetime.now(UTC)
    profile = profile.model_copy(
        update={"deadline": now + timedelta(minutes=5), "rules": [deadline_rule, anonymity_rule]}
    )
    results = RuleEngine(store).evaluate(profile, candidate, artifact_bytes=payloads, now=now)
    assert results[0].status is RuleResultStatus.PASS
    assert results[1].status is RuleResultStatus.FAIL


def test_zip_slip_symlink_and_compression_bomb_are_rejected() -> None:
    def archive_bytes(name: str, payload: bytes, *, symlink: bool = False) -> bytes:
        output = BytesIO()
        with ZipFile(output, "w", ZIP_DEFLATED) as archive:
            info = ZipInfo(name)
            info.compress_type = ZIP_DEFLATED
            info.external_attr = (stat.S_IFLNK | 0o777) << 16 if symlink else 0o100644 << 16
            archive.writestr(info, payload)
        return output.getvalue()

    with pytest.raises(ValueError, match="unsafe ZIP entry"):
        SubmissionIntegrityVerifier._verify_zip(archive_bytes("../escape", b"x"))
    with pytest.raises(ValueError, match="symlinks"):
        SubmissionIntegrityVerifier._verify_zip(archive_bytes("link", b"target", symlink=True))
    with pytest.raises(ValueError, match="compression ratio"):
        SubmissionIntegrityVerifier._verify_zip(archive_bytes("bomb.txt", b"0" * 100_000))


def test_package_artifact_hash_itself_is_checked(tmp_path) -> None:
    store, _, _, _, _, _, _, package = _accepted_chain(tmp_path)
    original = store.read_bytes(package.package_artifact.storage_key)
    store.resolve(package.package_artifact.storage_key).write_bytes(original + b"x")
    assert SubmissionIntegrityVerifier(store).verify(package) is SubmissionStatus.DIRTY
    assert sha256_bytes(original) == package.package_artifact.sha256
