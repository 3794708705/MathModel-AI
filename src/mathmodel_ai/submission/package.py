from __future__ import annotations

import stat
import unicodedata
from dataclasses import dataclass
from io import BytesIO
from pathlib import PurePosixPath
from uuid import UUID, uuid4
from zipfile import ZIP_DEFLATED, BadZipFile, ZipFile, ZipInfo

from pydantic import ValidationError
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from mathmodel_ai.core.errors import StorageError
from mathmodel_ai.files.storage import FileStore
from mathmodel_ai.paper.hashing import canonical_json_bytes, sha256_bytes, sha256_json
from mathmodel_ai.schemas.submission import (
    CompetitionProfile,
    DeadlineStatus,
    FinalJuryReport,
    JuryDecision,
    RequirementCoverage,
    RuleSeverity,
    RuleType,
    SubmissionArtifact,
    SubmissionArtifactRole,
    SubmissionCandidate,
    SubmissionCheckResult,
    SubmissionCheckStatus,
    SubmissionManifest,
    SubmissionManifestFile,
    SubmissionSnapshot,
    SubmissionStatus,
)
from mathmodel_ai.submission.checks import SubmissionCheck
from mathmodel_ai.submission.integrity import (
    artifact_set_digest,
    final_jury_report_digest,
    manifest_file_payload,
    requirement_snapshot_digest,
    rule_result_snapshot_digest,
    submission_candidate_digest,
    submission_check_digest,
    submission_snapshot_digest,
    validate_competition_profile,
)
from mathmodel_ai.submission.jury import FinalJuryGate
from mathmodel_ai.submission.rules import (
    DeadlineEvaluator,
    RuleEngine,
    SubmissionSecurityScanner,
    is_allowed_path,
    mime_matches,
)


@dataclass(frozen=True)
class SubmissionPackage:
    profile: CompetitionProfile
    snapshot: SubmissionSnapshot
    manifest: SubmissionManifest
    manifest_artifact: SubmissionArtifact
    package_artifact: SubmissionArtifact
    protected_artifacts: list[SubmissionArtifact]


class SubmissionArtifactRegistry:
    def __init__(self, artifacts: list[SubmissionArtifact] | None = None) -> None:
        self._items: dict[str, SubmissionArtifact] = {}
        self._canonical_paths: set[str] = set()
        for artifact in artifacts or []:
            self.register(artifact)

    def register(self, artifact: SubmissionArtifact) -> None:
        canonical = _canonical_path(artifact.relative_path)
        if canonical in self._canonical_paths and artifact.relative_path not in self._items:
            raise ValueError("submission contains a case or Unicode filename collision")
        existing = self._items.get(artifact.relative_path)
        if existing is not None and existing != artifact:
            raise ValueError("submission path is already bound to another artifact")
        self._items[artifact.relative_path] = artifact
        self._canonical_paths.add(canonical)

    def values(self) -> list[SubmissionArtifact]:
        return [self._items[key] for key in sorted(self._items)]


class SubmissionPackageBuilder:
    def __init__(self, store: FileStore) -> None:
        self._store = store
        self._security = SubmissionSecurityScanner()

    def build(
        self,
        *,
        profile: CompetitionProfile,
        candidate: SubmissionCandidate,
        requirements: list[RequirementCoverage] | None = None,
        jury: FinalJuryReport | None = None,
        check: SubmissionCheckResult | None = None,
    ) -> SubmissionPackage:
        profile_digest = validate_competition_profile(profile)
        if candidate.competition_profile_digest != profile_digest:
            raise ValueError("candidate competition profile digest is stale")
        registry = SubmissionArtifactRegistry(candidate.artifacts)
        protected = registry.values()
        payloads = {
            artifact.artifact_id: self._store.read_bytes(artifact.storage_key)
            for artifact in protected
        }
        self._validate_sources(profile, candidate, protected, payloads)
        if requirements is None or jury is None or check is None:
            raise ValueError("submission package requires approved jury and check context")
        recomputed_rules = RuleEngine(self._store).evaluate(
            profile, candidate, artifact_bytes=payloads
        )
        if (
            jury.decision is not JuryDecision.PASS
            or check.status is not SubmissionCheckStatus.PASS
            or rule_result_snapshot_digest(recomputed_rules)
            != rule_result_snapshot_digest(check.rule_results)
            or FinalJuryGate().verify_report(
                jury,
                profile=profile,
                candidate=candidate,
                requirements=requirements,
                rule_results=recomputed_rules,
            )
            is not JuryDecision.PASS
            or SubmissionCheck().verify(
                check,
                profile=profile,
                candidate=candidate,
                requirements=requirements,
                rules=recomputed_rules,
                jury=jury,
                artifact_bytes=payloads,
            )
            is not SubmissionCheckStatus.PASS
        ):
            raise ValueError("submission package approval context failed deterministic recheck")
        submission_id = candidate.candidate_id
        files = [
            SubmissionManifestFile(
                path=artifact.relative_path,
                role=artifact.role,
                sha256=artifact.sha256,
                size_bytes=artifact.size_bytes,
                mime_type=artifact.mime_type,
                source_artifact_id=artifact.artifact_id,
            )
            for artifact in protected
        ]
        files.sort(key=lambda item: item.path)
        package_hash = sha256_json([manifest_file_payload(item) for item in files])
        requirements_digest = requirement_snapshot_digest(requirements)
        jury_digest = final_jury_report_digest(jury)
        check_digest = submission_check_digest(check)
        candidate_digest = submission_candidate_digest(candidate)
        artifacts_digest = artifact_set_digest(protected)
        manifest_payload = {
            "submission_id": submission_id,
            "project_id": candidate.project_id,
            "competition_profile_id": profile.profile_id,
            "competition_profile_version": profile.version,
            "competition_profile_digest": profile_digest,
            "requirement_snapshot_digest": requirements_digest,
            "jury_report_id": jury.report_id,
            "jury_report_digest": jury_digest,
            "submission_check_id": check.check_id,
            "submission_check_digest": check_digest,
            "candidate_digest": candidate_digest,
            "artifact_set_digest": artifacts_digest,
            "paper_id": candidate.paper_id,
            "paper_version": candidate.paper_version,
            "paper_manifest_hash": candidate.paper_manifest_hash,
            "paper_ir_hash": candidate.paper_ir_hash,
            "verified_model_id": candidate.verified_model_id,
            "verified_model_version": candidate.verified_model_version,
            "verified_model_digest": candidate.verified_model_digest,
            "verified_result_id": candidate.verified_result_id,
            "files": [item.model_dump(mode="json") for item in files],
            "package_hash": package_hash,
        }
        manifest = SubmissionManifest(
            **manifest_payload,
            manifest_hash=sha256_json(manifest_payload),
        )
        manifest_bytes = canonical_json_bytes(manifest)
        manifest_artifact = self._store_submission_artifact(
            candidate=candidate,
            role=SubmissionArtifactRole.MANIFEST,
            filename="submission_manifest.json",
            mime_type="application/json",
            data=manifest_bytes,
        )
        zip_bytes = self._build_zip(protected, payloads, manifest_bytes)
        maximum = profile.file_rules.max_package_size_bytes
        if maximum is not None and len(zip_bytes) > maximum:
            raise ValueError("physical submission package exceeds configured size limit")
        package_artifact = self._store_submission_artifact(
            candidate=candidate,
            role=SubmissionArtifactRole.PACKAGE,
            filename="submission.zip",
            mime_type="application/zip",
            data=zip_bytes,
        )
        snapshot = SubmissionSnapshot(
            submission_id=submission_id,
            project_id=candidate.project_id,
            competition_profile_id=profile.profile_id,
            competition_profile_version=profile.version,
            competition_profile_digest=profile_digest,
            requirement_snapshot_digest=requirements_digest,
            jury_report_id=jury.report_id,
            jury_report_digest=jury_digest,
            jury_reviewer_is_mock=jury.reviewer_is_mock,
            submission_check_id=check.check_id,
            submission_check_digest=check_digest,
            candidate_digest=candidate_digest,
            artifact_set_digest=artifacts_digest,
            verified_model_id=candidate.verified_model_id,
            verified_model_version=candidate.verified_model_version,
            verified_model_digest=candidate.verified_model_digest,
            verified_result_id=candidate.verified_result_id,
            paper_id=candidate.paper_id,
            paper_version=candidate.paper_version,
            paper_manifest_hash=candidate.paper_manifest_hash,
            paper_ir_hash=candidate.paper_ir_hash,
            artifact_ids=[
                *(item.artifact_id for item in protected),
                manifest_artifact.artifact_id,
                package_artifact.artifact_id,
            ],
            status=SubmissionStatus.FROZEN,
            manifest_hash=manifest.manifest_hash,
            package_hash=package_hash,
            snapshot_digest="0" * 64,
        )
        snapshot = snapshot.model_copy(
            update={"snapshot_digest": submission_snapshot_digest(snapshot)}
        )
        package = SubmissionPackage(
            profile=profile,
            snapshot=snapshot,
            manifest=manifest,
            manifest_artifact=manifest_artifact,
            package_artifact=package_artifact,
            protected_artifacts=protected,
        )
        if SubmissionIntegrityVerifier(self._store).verify(package) is not SubmissionStatus.FROZEN:
            raise ValueError("newly built submission package failed roundtrip verification")
        return package

    def verify(self, package: SubmissionPackage) -> SubmissionStatus:
        return SubmissionIntegrityVerifier(self._store).verify(package)

    def _validate_sources(
        self,
        profile: CompetitionProfile,
        candidate: SubmissionCandidate,
        protected: list[SubmissionArtifact],
        payloads: dict[UUID, bytes],
    ) -> None:
        for artifact in protected:
            data = payloads[artifact.artifact_id]
            if len(data) != artifact.size_bytes or sha256_bytes(data) != artifact.sha256:
                raise ValueError("source artifact bytes do not match their immutable record")
            if not mime_matches(artifact.mime_type, data):
                raise ValueError(
                    f"source artifact MIME does not match its bytes: {artifact.relative_path}"
                )
            if (
                profile.file_rules.allowed_mime_types
                and artifact.mime_type not in profile.file_rules.allowed_mime_types
            ):
                raise ValueError(
                    f"source artifact MIME is not allowed by profile: {artifact.relative_path}"
                )
            if not is_allowed_path(artifact.relative_path, profile):
                raise ValueError(f"submission path is not allowed: {artifact.relative_path}")
            try:
                resolved = self._store.resolve(artifact.storage_key)
            except StorageError as exc:
                raise ValueError("submission source is missing or unsafe") from exc
            if resolved.is_symlink():
                raise ValueError("submission sources may not be symlinks")
            try:
                attributes = int(getattr(resolved.lstat(), "st_file_attributes", 0))
            except OSError as exc:
                raise ValueError("submission source metadata cannot be verified") from exc
            if attributes & int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)):
                raise ValueError("submission sources may not be reparse points")
        findings = self._security.scan(candidate, payloads)
        if findings:
            raise ValueError(f"submission security scan failed: {findings[0][0]}")
        maximum = profile.file_rules.max_package_size_bytes
        if maximum is not None and sum(len(item) for item in payloads.values()) > maximum:
            raise ValueError("submission package exceeds configured size limit")

    def _store_submission_artifact(
        self,
        *,
        candidate: SubmissionCandidate,
        role: SubmissionArtifactRole,
        filename: str,
        mime_type: str,
        data: bytes,
    ) -> SubmissionArtifact:
        artifact_id = uuid4()
        stored = self._store.store_artifact(
            data,
            project_id=candidate.project_id,
            artifact_id=artifact_id,
            filename=filename,
        )
        return SubmissionArtifact(
            artifact_id=artifact_id,
            project_id=candidate.project_id,
            role=role,
            relative_path=filename,
            mime_type=mime_type,
            size_bytes=stored.size_bytes,
            sha256=stored.sha256,
            storage_key=stored.storage_key,
            paper_id=candidate.paper_id,
            paper_version=candidate.paper_version,
        )

    @staticmethod
    def _build_zip(
        artifacts: list[SubmissionArtifact],
        payloads: dict[UUID, bytes],
        manifest_bytes: bytes,
    ) -> bytes:
        output = BytesIO()
        with ZipFile(output, mode="w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
            entries = [(item.relative_path, payloads[item.artifact_id]) for item in artifacts]
            entries.append(("submission_manifest.json", manifest_bytes))
            for name, data in sorted(entries):
                info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, data)
        return output.getvalue()


class SubmissionIntegrityVerifier:
    MAX_ZIP_ENTRIES = 10_000
    MAX_UNCOMPRESSED_BYTES = 250 * 1024 * 1024
    MAX_COMPRESSION_RATIO = 100

    def __init__(self, store: FileStore) -> None:
        self._store = store
        self._security = SubmissionSecurityScanner()

    def verify(self, package: SubmissionPackage) -> SubmissionStatus:
        snapshot = package.snapshot
        manifest = package.manifest
        try:
            profile_digest = validate_competition_profile(package.profile)
        except ValueError:
            return SubmissionStatus.DIRTY
        if any(
            rule.rule_type is RuleType.DEADLINE and rule.severity is RuleSeverity.BLOCKING
            for rule in package.profile.rules
        ):
            deadline_status, _ = DeadlineEvaluator.evaluate(package.profile.deadline)
            if deadline_status in {DeadlineStatus.EXPIRED, DeadlineStatus.UNKNOWN}:
                return SubmissionStatus.DIRTY
        if submission_snapshot_digest(snapshot) != snapshot.snapshot_digest:
            return SubmissionStatus.DIRTY
        if snapshot.status is not SubmissionStatus.FROZEN:
            return SubmissionStatus.DIRTY
        if package.profile.require_independent_reviewer and snapshot.jury_reviewer_is_mock:
            return SubmissionStatus.DIRTY
        if (
            snapshot.submission_id != manifest.submission_id
            or snapshot.project_id != manifest.project_id
            or snapshot.paper_id != manifest.paper_id
            or snapshot.paper_version != manifest.paper_version
            or snapshot.verified_result_id != manifest.verified_result_id
            or snapshot.verified_model_id != manifest.verified_model_id
            or snapshot.verified_model_version != manifest.verified_model_version
            or snapshot.verified_model_digest != manifest.verified_model_digest
            or snapshot.paper_manifest_hash != manifest.paper_manifest_hash
            or snapshot.paper_ir_hash != manifest.paper_ir_hash
            or snapshot.competition_profile_id != manifest.competition_profile_id
            or snapshot.competition_profile_version != manifest.competition_profile_version
            or snapshot.manifest_hash != manifest.manifest_hash
            or snapshot.package_hash != manifest.package_hash
            or snapshot.competition_profile_digest != profile_digest
            or manifest.competition_profile_digest != profile_digest
            or snapshot.requirement_snapshot_digest != manifest.requirement_snapshot_digest
            or snapshot.jury_report_id != manifest.jury_report_id
            or snapshot.jury_report_digest != manifest.jury_report_digest
            or snapshot.submission_check_id != manifest.submission_check_id
            or snapshot.submission_check_digest != manifest.submission_check_digest
            or snapshot.candidate_digest != manifest.candidate_digest
            or snapshot.artifact_set_digest != manifest.artifact_set_digest
        ):
            return SubmissionStatus.DIRTY
        payload = manifest.model_dump(mode="json", exclude={"manifest_hash"})
        if sha256_json(payload) != manifest.manifest_hash:
            return SubmissionStatus.DIRTY
        entries = [manifest_file_payload(item) for item in manifest.files]
        if sha256_json(entries) != manifest.package_hash:
            return SubmissionStatus.DIRTY
        source_by_id = {item.artifact_id: item for item in package.protected_artifacts}
        if set(source_by_id) != {item.source_artifact_id for item in manifest.files}:
            return SubmissionStatus.DIRTY
        expected_artifact_ids = {
            *source_by_id,
            package.manifest_artifact.artifact_id,
            package.package_artifact.artifact_id,
        }
        if set(snapshot.artifact_ids) != expected_artifact_ids:
            return SubmissionStatus.DIRTY
        if artifact_set_digest(package.protected_artifacts) != snapshot.artifact_set_digest:
            return SubmissionStatus.DIRTY
        for item in manifest.files:
            source = source_by_id.get(item.source_artifact_id)
            if source is None or source.relative_path != item.path:
                return SubmissionStatus.DIRTY
            try:
                data = self._store.read_bytes(source.storage_key)
            except (StorageError, OSError, ValueError):
                return SubmissionStatus.DIRTY
            if len(data) != item.size_bytes or sha256_bytes(data) != item.sha256:
                return SubmissionStatus.DIRTY
        try:
            raw_manifest = self._store.read_bytes(package.manifest_artifact.storage_key)
            raw_zip = self._store.read_bytes(package.package_artifact.storage_key)
        except (StorageError, OSError, ValueError):
            return SubmissionStatus.DIRTY
        if (
            sha256_bytes(raw_manifest) != package.manifest_artifact.sha256
            or len(raw_manifest) != package.manifest_artifact.size_bytes
            or sha256_bytes(raw_zip) != package.package_artifact.sha256
            or len(raw_zip) != package.package_artifact.size_bytes
        ):
            return SubmissionStatus.DIRTY
        maximum = package.profile.file_rules.max_package_size_bytes
        if maximum is not None and len(raw_zip) > maximum:
            return SubmissionStatus.DIRTY
        try:
            embedded = self._verify_zip(raw_zip)
            parsed = SubmissionManifest.model_validate_json(
                embedded.pop("submission_manifest.json")
            )
        except (BadZipFile, KeyError, OSError, RuntimeError, ValueError, ValidationError):
            return SubmissionStatus.DIRTY
        if parsed != manifest or canonical_json_bytes(parsed) != raw_manifest:
            return SubmissionStatus.DIRTY
        expected_paths = {item.path for item in manifest.files}
        if set(embedded) != expected_paths:
            return SubmissionStatus.DIRTY
        packaged_payloads = {
            item.source_artifact_id: embedded[item.path] for item in manifest.files
        }
        if self._security.scan_artifacts(package.protected_artifacts, packaged_payloads):
            return SubmissionStatus.DIRTY
        if package.profile.anonymous_rules.required and self._security.anonymity_artifact_findings(
            package.protected_artifacts,
            packaged_payloads,
            package.profile.anonymous_rules.prohibited_terms,
        ):
            return SubmissionStatus.DIRTY
        if not self._profile_package_valid(package.profile, manifest, embedded):
            return SubmissionStatus.DIRTY
        for item in manifest.files:
            data = embedded[item.path]
            if len(data) != item.size_bytes or sha256_bytes(data) != item.sha256:
                return SubmissionStatus.DIRTY
            if not mime_matches(item.mime_type, data):
                return SubmissionStatus.DIRTY
            if item.role is SubmissionArtifactRole.PAPER_PDF:
                try:
                    if len(PdfReader(BytesIO(data), strict=True).pages) < 1:
                        return SubmissionStatus.DIRTY
                except (PdfReadError, ValueError, TypeError, KeyError):
                    return SubmissionStatus.DIRTY
        return SubmissionStatus.FROZEN

    @staticmethod
    def _profile_package_valid(
        profile: CompetitionProfile,
        manifest: SubmissionManifest,
        embedded: dict[str, bytes],
    ) -> bool:
        if len(manifest.files) > (profile.file_rules.max_file_count or len(manifest.files)):
            return False
        if any(not is_allowed_path(item.path, profile) for item in manifest.files):
            return False
        maximum_file_size = profile.file_rules.max_file_size_bytes
        if maximum_file_size is not None and any(
            len(embedded[item.path]) > maximum_file_size for item in manifest.files
        ):
            return False
        paper_files = [
            item for item in manifest.files if item.role is SubmissionArtifactRole.PAPER_PDF
        ]
        if len(paper_files) != 1:
            return False
        paper = paper_files[0]
        if profile.naming_rules.paper_filename and (
            _filename_key(paper.path, profile.naming_rules.case_sensitive)
            != _filename_key(
                profile.naming_rules.paper_filename,
                profile.naming_rules.case_sensitive,
            )
        ):
            return False
        try:
            page_count = len(PdfReader(BytesIO(embedded[paper.path]), strict=True).pages)
        except (PdfReadError, ValueError, TypeError, KeyError):
            return False
        total_limits = [
            int(rule.parameters["max_pages"])
            for rule in profile.rules
            if rule.rule_type.value == "PAGE_LIMIT"
            and str(rule.parameters.get("scope", profile.page_rules.scope.value)) == "TOTAL"
            and "max_pages" in rule.parameters
        ]
        if profile.page_rules.max_pages is not None and profile.page_rules.scope.value == "TOTAL":
            total_limits.append(profile.page_rules.max_pages)
        if total_limits and page_count > min(total_limits):
            return False
        return True

    @staticmethod
    def _verify_zip(data: bytes) -> dict[str, bytes]:
        extracted: dict[str, bytes] = {}
        canonical_paths: set[str] = set()
        with ZipFile(BytesIO(data)) as archive:
            infos = archive.infolist()
            if len(infos) > SubmissionIntegrityVerifier.MAX_ZIP_ENTRIES:
                raise ValueError("ZIP contains too many entries")
            total_size = sum(info.file_size for info in infos)
            if total_size > SubmissionIntegrityVerifier.MAX_UNCOMPRESSED_BYTES:
                raise ValueError("ZIP exceeds the uncompressed size limit")
            for info in infos:
                if (
                    info.file_size > 0
                    and info.file_size
                    > max(info.compress_size, 1) * SubmissionIntegrityVerifier.MAX_COMPRESSION_RATIO
                ):
                    raise ValueError("ZIP compression ratio exceeds the safety limit")
                path = info.filename.replace("\\", "/")
                canonical = _canonical_path(path)
                if canonical in canonical_paths:
                    raise ValueError("duplicate ZIP entry")
                if path.startswith("/") or ":" in path or ".." in path.split("/"):
                    raise ValueError("unsafe ZIP entry")
                mode = info.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise ValueError("ZIP symlinks are forbidden")
                payload = archive.read(info)
                if payload.startswith(b"PK\x03\x04"):
                    raise ValueError("nested archives are forbidden")
                extracted[path] = payload
                canonical_paths.add(canonical)
        return extracted


def _canonical_path(path: str) -> str:
    return unicodedata.normalize("NFC", path.replace("\\", "/")).casefold()


def _filename_key(path: str, case_sensitive: bool) -> str:
    name = PurePosixPath(path).name
    normalized = unicodedata.normalize("NFC", name)
    return normalized if case_sensitive else normalized.casefold()
