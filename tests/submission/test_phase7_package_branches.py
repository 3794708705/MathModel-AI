from __future__ import annotations

from dataclasses import replace
from io import BytesIO
from types import SimpleNamespace
from uuid import UUID, uuid4
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from mathmodel_ai.core.errors import StorageError
from mathmodel_ai.paper.hashing import canonical_json_bytes, sha256_bytes, sha256_json
from mathmodel_ai.schemas.submission import (
    JuryDecision,
    RequirementCoverage,
    RequirementCoverageStatus,
    SubmissionCheckStatus,
    SubmissionStatus,
)
from mathmodel_ai.submission.checks import SubmissionCheck
from mathmodel_ai.submission.integrity import competition_profile_digest
from mathmodel_ai.submission.jury import FinalJuryGate
from mathmodel_ai.submission.package import (
    SubmissionArtifactRegistry,
    SubmissionIntegrityVerifier,
    SubmissionPackage,
    SubmissionPackageBuilder,
)
from mathmodel_ai.submission.rules import RuleEngine
from tests.submission.helpers import candidate_fixture, jury_draft, pdf_bytes


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


def _approvals(store, profile, candidate):
    payloads = {
        item.artifact_id: store.read_bytes(item.storage_key) for item in candidate.artifacts
    }
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
    return jury, check


def _package(tmp_path) -> tuple:
    store, profile, candidate, _ = candidate_fixture(tmp_path)
    jury, check = _approvals(store, profile, candidate)
    package = SubmissionPackageBuilder(store).build(
        profile=profile,
        candidate=candidate,
        requirements=_coverage(),
        jury=jury,
        check=check,
    )
    return store, profile, candidate, package


def _overwrite_artifact(store, artifact, data: bytes):
    store.resolve(artifact.storage_key).write_bytes(data)
    return artifact.model_copy(update={"size_bytes": len(data), "sha256": sha256_bytes(data)})


def _rehash_manifest(manifest):
    payload = manifest.model_dump(mode="json", exclude={"manifest_hash"})
    return manifest.model_copy(update={"manifest_hash": sha256_json(payload)})


def _with_embedded(
    store,
    package: SubmissionPackage,
    *,
    manifest=None,
    payloads=None,
    extra_entries: list[tuple[str, bytes]] | None = None,
) -> SubmissionPackage:
    actual_manifest = _rehash_manifest(manifest or package.manifest)
    raw_manifest = canonical_json_bytes(actual_manifest)
    manifest_artifact = _overwrite_artifact(store, package.manifest_artifact, raw_manifest)
    actual_payloads = payloads or {
        item.artifact_id: store.read_bytes(item.storage_key) for item in package.protected_artifacts
    }
    raw_zip = SubmissionPackageBuilder._build_zip(
        package.protected_artifacts, actual_payloads, raw_manifest
    )
    if extra_entries:
        source = BytesIO(raw_zip)
        output = BytesIO()
        with ZipFile(source) as current, ZipFile(output, "w", ZIP_DEFLATED) as changed:
            for info in current.infolist():
                changed.writestr(info.filename, current.read(info.filename))
            for name, data in extra_entries:
                changed.writestr(name, data)
        raw_zip = output.getvalue()
    package_artifact = _overwrite_artifact(store, package.package_artifact, raw_zip)
    snapshot = package.snapshot.model_copy(
        update={
            "manifest_hash": actual_manifest.manifest_hash,
            "package_hash": actual_manifest.package_hash,
        }
    )
    return replace(
        package,
        snapshot=snapshot,
        manifest=actual_manifest,
        manifest_artifact=manifest_artifact,
        package_artifact=package_artifact,
    )


def test_artifact_registry_rejects_conflicting_path_and_sorts(tmp_path) -> None:
    _, _, candidate, _ = candidate_fixture(tmp_path)
    original = candidate.artifacts[0]
    conflict = original.model_copy(update={"artifact_id": uuid4()})
    registry = SubmissionArtifactRegistry([original])
    registry.register(original)
    with pytest.raises(ValueError, match="already bound"):
        registry.register(conflict)
    assert registry.values() == [original]


def test_builder_rejects_source_hash_mime_allowlist_and_size_tampering(tmp_path) -> None:
    store, profile, candidate, _ = candidate_fixture(tmp_path / "hash")
    store.resolve(candidate.artifacts[0].storage_key).write_bytes(b"changed")
    with pytest.raises(ValueError, match="immutable record"):
        SubmissionPackageBuilder(store).build(profile=profile, candidate=candidate)

    store, profile, candidate, _ = candidate_fixture(tmp_path / "mime")
    wrong_mime = candidate.artifacts[0].model_copy(update={"mime_type": "text/plain"})
    with pytest.raises(ValueError, match="MIME does not match"):
        SubmissionPackageBuilder(store).build(
            profile=profile,
            candidate=candidate.model_copy(update={"artifacts": [wrong_mime]}),
        )

    store, profile, candidate, _ = candidate_fixture(tmp_path / "allowlist")
    rules = profile.file_rules.model_copy(update={"allowed_mime_types": ["text/plain"]})
    profile = profile.model_copy(update={"file_rules": rules})
    candidate = candidate.model_copy(
        update={"competition_profile_digest": competition_profile_digest(profile)}
    )
    with pytest.raises(ValueError, match="MIME is not allowed"):
        SubmissionPackageBuilder(store).build(profile=profile, candidate=candidate)

    store, profile, candidate, _ = candidate_fixture(tmp_path / "size")
    rules = profile.file_rules.model_copy(update={"max_package_size_bytes": 1})
    profile = profile.model_copy(update={"file_rules": rules})
    candidate = candidate.model_copy(
        update={"competition_profile_digest": competition_profile_digest(profile)}
    )
    with pytest.raises(ValueError, match="configured size limit"):
        SubmissionPackageBuilder(store).build(profile=profile, candidate=candidate)


def test_builder_normalizes_unsafe_resolve_and_roundtrip_failure(tmp_path, monkeypatch) -> None:
    store, profile, candidate, payloads = candidate_fixture(tmp_path / "resolve")
    builder = SubmissionPackageBuilder(store)
    monkeypatch.setattr(
        store,
        "resolve",
        lambda _key: (_ for _ in ()).throw(StorageError("unsafe")),
    )
    with pytest.raises(ValueError, match="missing or unsafe"):
        builder._validate_sources(profile, candidate, candidate.artifacts, payloads)

    store, profile, candidate, _ = candidate_fixture(tmp_path / "roundtrip")
    jury, check = _approvals(store, profile, candidate)
    monkeypatch.setattr(
        SubmissionIntegrityVerifier,
        "verify",
        lambda self, package: SubmissionStatus.DIRTY,
    )
    with pytest.raises(ValueError, match="roundtrip verification"):
        SubmissionPackageBuilder(store).build(
            profile=profile,
            candidate=candidate,
            requirements=_coverage(),
            jury=jury,
            check=check,
        )


def test_builder_rejects_symlink_source_and_zero_page_pdf(tmp_path, monkeypatch) -> None:
    store, profile, candidate, payloads = candidate_fixture(tmp_path / "symlink")
    monkeypatch.setattr(
        store,
        "resolve",
        lambda _key: SimpleNamespace(is_symlink=lambda: True),
    )
    with pytest.raises(ValueError, match="may not be symlinks"):
        SubmissionPackageBuilder(store)._validate_sources(
            profile, candidate, candidate.artifacts, payloads
        )

    store, profile, candidate, _ = candidate_fixture(tmp_path / "zero-pages")
    zero_page_pdf = pdf_bytes(0)
    artifact = _overwrite_artifact(store, candidate.artifacts[0], zero_page_pdf)
    candidate = candidate.model_copy(update={"artifacts": [artifact]})
    payloads = {artifact.artifact_id: zero_page_pdf}
    rules = RuleEngine(store).evaluate(profile, candidate, artifact_bytes=payloads)
    page_result = next(item for item in rules if item.rule_type.value == "PAGE_LIMIT")
    assert page_result.status.value == "FAIL"
    jury = FinalJuryGate().build_report(
        draft=jury_draft(),
        profile=profile,
        candidate=candidate,
        requirements=_coverage(),
        rule_results=rules,
        reviewer_is_mock=True,
        agent_run_id=uuid4(),
    )
    assert jury.decision is JuryDecision.FAIL


def test_verifier_rejects_snapshot_manifest_and_package_hash_mismatch(tmp_path) -> None:
    store, _, _, package = _package(tmp_path)
    verifier = SubmissionIntegrityVerifier(store)
    assert (
        verifier.verify(
            replace(
                package,
                snapshot=package.snapshot.model_copy(update={"project_id": uuid4()}),
            )
        )
        is SubmissionStatus.DIRTY
    )

    altered_manifest = package.manifest.model_copy(
        update={"verified_model_version": package.manifest.verified_model_version + 1}
    )
    assert verifier.verify(replace(package, manifest=altered_manifest)) is SubmissionStatus.DIRTY

    fake_hash = "0" * 64
    package_hash_changed = _rehash_manifest(
        package.manifest.model_copy(update={"package_hash": fake_hash})
    )
    snapshot = package.snapshot.model_copy(
        update={
            "package_hash": fake_hash,
            "manifest_hash": package_hash_changed.manifest_hash,
        }
    )
    assert (
        verifier.verify(replace(package, manifest=package_hash_changed, snapshot=snapshot))
        is SubmissionStatus.DIRTY
    )


def test_verifier_rejects_missing_or_mismatched_source_bindings(tmp_path) -> None:
    store, _, _, package = _package(tmp_path)
    manifest_file = package.manifest.files[0]
    missing_source = manifest_file.model_copy(update={"source_artifact_id": uuid4()})
    manifest = _rehash_manifest(package.manifest.model_copy(update={"files": [missing_source]}))
    snapshot = package.snapshot.model_copy(update={"manifest_hash": manifest.manifest_hash})
    assert (
        SubmissionIntegrityVerifier(store).verify(
            replace(package, manifest=manifest, snapshot=snapshot)
        )
        is SubmissionStatus.DIRTY
    )

    broken_source = package.protected_artifacts[0].model_copy(
        update={"storage_key": "missing/object.pdf"}
    )
    assert (
        SubmissionIntegrityVerifier(store).verify(
            replace(package, protected_artifacts=[broken_source])
        )
        is SubmissionStatus.DIRTY
    )


def test_verifier_rejects_missing_generated_files_and_malformed_zip(tmp_path) -> None:
    store, _, _, package = _package(tmp_path)
    missing_manifest = package.manifest_artifact.model_copy(
        update={"storage_key": "missing/manifest.json"}
    )
    assert (
        SubmissionIntegrityVerifier(store).verify(
            replace(package, manifest_artifact=missing_manifest)
        )
        is SubmissionStatus.DIRTY
    )

    bad_zip = b"not a zip"
    package_artifact = _overwrite_artifact(store, package.package_artifact, bad_zip)
    assert (
        SubmissionIntegrityVerifier(store).verify(
            replace(package, package_artifact=package_artifact)
        )
        is SubmissionStatus.DIRTY
    )


def test_verifier_rejects_embedded_manifest_and_path_set_tampering(tmp_path) -> None:
    store, _, _, package = _package(tmp_path)
    changed = package.manifest.model_copy(
        update={"verified_model_version": package.manifest.verified_model_version + 1}
    )
    changed = _with_embedded(store, package, manifest=changed)
    changed = replace(changed, snapshot=package.snapshot, manifest=package.manifest)
    assert SubmissionIntegrityVerifier(store).verify(changed) is SubmissionStatus.DIRTY

    store, _, _, package = _package(tmp_path / "extra")
    extra = _with_embedded(store, package, extra_entries=[("unexpected.txt", b"unexpected")])
    assert SubmissionIntegrityVerifier(store).verify(extra) is SubmissionStatus.DIRTY


def test_verifier_recomputes_embedded_hash_mime_and_pdf_validity(tmp_path) -> None:
    store, _, _, package = _package(tmp_path / "bytes")
    raw_manifest = canonical_json_bytes(package.manifest)
    payloads = {
        item.artifact_id: store.read_bytes(item.storage_key) for item in package.protected_artifacts
    }
    payloads[package.protected_artifacts[0].artifact_id] = b"changed-pdf"
    changed = _with_embedded(store, package, payloads=payloads)
    assert SubmissionIntegrityVerifier(store).verify(changed) is SubmissionStatus.DIRTY
    assert raw_manifest == canonical_json_bytes(package.manifest)

    store, _, _, package = _package(tmp_path / "mime")
    file_record = package.manifest.files[0].model_copy(update={"mime_type": "text/plain"})
    manifest = package.manifest.model_copy(update={"files": [file_record]})
    changed = _with_embedded(store, package, manifest=manifest)
    assert SubmissionIntegrityVerifier(store).verify(changed) is SubmissionStatus.DIRTY

    store, _, _, package = _package(tmp_path / "pdf")
    corrupt = b"%PDF-corrupt"
    source = _overwrite_artifact(store, package.protected_artifacts[0], corrupt)
    file_record = package.manifest.files[0].model_copy(
        update={"sha256": source.sha256, "size_bytes": source.size_bytes}
    )
    entries = [
        {
            "path": file_record.path,
            "sha256": file_record.sha256,
            "size_bytes": file_record.size_bytes,
        }
    ]
    manifest = package.manifest.model_copy(
        update={"files": [file_record], "package_hash": sha256_json(entries)}
    )
    changed = replace(package, protected_artifacts=[source])
    changed = _with_embedded(
        store,
        changed,
        manifest=manifest,
        payloads={source.artifact_id: corrupt},
    )
    assert SubmissionIntegrityVerifier(store).verify(changed) is SubmissionStatus.DIRTY


def test_zip_limits_duplicate_and_windows_path_are_rejected(monkeypatch) -> None:
    def archive(entries: list[tuple[str, bytes]]) -> bytes:
        output = BytesIO()
        with ZipFile(output, "w", ZIP_DEFLATED) as zipped:
            for name, data in entries:
                zipped.writestr(name, data)
        return output.getvalue()

    one = archive([("one.txt", b"1")])
    monkeypatch.setattr(SubmissionIntegrityVerifier, "MAX_ZIP_ENTRIES", 0)
    with pytest.raises(ValueError, match="too many entries"):
        SubmissionIntegrityVerifier._verify_zip(one)
    monkeypatch.setattr(SubmissionIntegrityVerifier, "MAX_ZIP_ENTRIES", 10_000)
    monkeypatch.setattr(SubmissionIntegrityVerifier, "MAX_UNCOMPRESSED_BYTES", 0)
    with pytest.raises(ValueError, match="uncompressed size"):
        SubmissionIntegrityVerifier._verify_zip(one)
    monkeypatch.setattr(SubmissionIntegrityVerifier, "MAX_UNCOMPRESSED_BYTES", 250 * 1024 * 1024)
    with pytest.warns(UserWarning, match="Duplicate name"):
        duplicate = archive([("same.txt", b"1"), ("same.txt", b"2")])
    with pytest.raises(ValueError, match="duplicate ZIP entry"):
        SubmissionIntegrityVerifier._verify_zip(duplicate)
    with pytest.raises(ValueError, match="unsafe ZIP entry"):
        SubmissionIntegrityVerifier._verify_zip(archive([("C:/secret", b"x")]))
