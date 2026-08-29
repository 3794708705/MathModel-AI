from __future__ import annotations

from uuid import uuid4

from mathmodel_ai.files.storage import FileStore
from mathmodel_ai.paper.hashing import canonical_json_bytes, sha256_json
from mathmodel_ai.schemas.paper import (
    FigureRecord,
    PaperArtifact,
    PaperArtifactKind,
    PaperManifest,
    PaperVersion,
    ReferenceRecord,
    TableRecord,
)


class PaperBundleBuilder:
    def __init__(self, store: FileStore) -> None:
        self._store = store

    def build_manifest(
        self,
        *,
        version: PaperVersion,
        references: list[ReferenceRecord],
        figures: list[FigureRecord],
        tables: list[TableRecord],
        artifacts: list[PaperArtifact],
    ) -> tuple[PaperManifest, PaperArtifact]:
        singleton_kinds = {
            PaperArtifactKind.TEX,
            PaperArtifactKind.BIB,
            PaperArtifactKind.PDF,
        }
        by_kind: dict[PaperArtifactKind, PaperArtifact] = {}
        for artifact in artifacts:
            if artifact.kind in singleton_kinds and artifact.kind in by_kind:
                raise ValueError(f"duplicate {artifact.kind.value} artifact")
            by_kind[artifact.kind] = artifact
        required = {PaperArtifactKind.TEX, PaperArtifactKind.BIB, PaperArtifactKind.PDF}
        if not required <= by_kind.keys():
            raise ValueError("manifest requires real TeX, BibTeX, and PDF artifacts")
        payload = {
            "paper_id": version.paper_id,
            "paper_version": version.version,
            "evidence_snapshot_hash": version.evidence_snapshot.snapshot_hash,
            "verified_model_version": version.evidence_snapshot.verified_model_version,
            "verified_model_digest": version.evidence_snapshot.verified_model_digest,
            "verified_result_id": version.evidence_snapshot.verified_result_id,
            "reference_ids": [item.reference_id for item in references],
            "figure_ids": [item.figure_id for item in figures],
            "table_ids": [item.table_id for item in tables],
            "paper_ir_hash": sha256_json(
                version.paper_ir.model_dump(mode="json", exclude={"status"})
            ),
            "claim_set_hash": sha256_json(
                [item.model_dump(mode="json") for item in version.paper_ir.claims]
            ),
            "document_registry_hash": sha256_json(
                [item.model_dump(mode="json") for item in version.paper_ir.document_registry]
            ),
            "reference_set_hash": sha256_json(
                [item.model_dump(mode="json") for item in references]
            ),
            "figure_set_hash": sha256_json([item.model_dump(mode="json") for item in figures]),
            "table_set_hash": sha256_json([item.model_dump(mode="json") for item in tables]),
            "tex_artifact_id": by_kind[PaperArtifactKind.TEX].artifact_id,
            "bib_artifact_id": by_kind[PaperArtifactKind.BIB].artifact_id,
            "pdf_artifact_id": by_kind[PaperArtifactKind.PDF].artifact_id,
            "tex_hash": by_kind[PaperArtifactKind.TEX].sha256,
            "bib_hash": by_kind[PaperArtifactKind.BIB].sha256,
            "pdf_hash": by_kind[PaperArtifactKind.PDF].sha256,
        }
        manifest_hash = sha256_json(payload)
        manifest = PaperManifest(**payload, manifest_hash=manifest_hash)
        data = canonical_json_bytes(manifest)
        artifact_id = uuid4()
        stored = self._store.store_artifact(
            data,
            project_id=version.project_id,
            artifact_id=artifact_id,
            filename="manifest.json",
        )
        artifact = PaperArtifact(
            artifact_id=artifact_id,
            project_id=version.project_id,
            problem_id=version.problem_id,
            paper_id=version.paper_id,
            paper_version=version.version,
            kind=PaperArtifactKind.MANIFEST,
            name="manifest.json",
            mime_type="application/json",
            size_bytes=stored.size_bytes,
            sha256=stored.sha256,
            storage_key=stored.storage_key,
        )
        return manifest, artifact

    @staticmethod
    def verify_manifest(manifest: PaperManifest) -> bool:
        payload = manifest.model_dump(mode="json", exclude={"manifest_hash"})
        return sha256_json(payload) == manifest.manifest_hash
