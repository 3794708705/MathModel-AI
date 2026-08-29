from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import UUID, uuid4

import pymupdf
import pytest
from pydantic import ValidationError

from mathmodel_ai.files.storage import LocalFileStore
from mathmodel_ai.mathematical.registry import EquationRegistry
from mathmodel_ai.paper.bundle import PaperBundleBuilder
from mathmodel_ai.paper.compiler import PaperCompilation
from mathmodel_ai.paper.hashing import sha256_bytes
from mathmodel_ai.paper.integrity import ArtifactIntegrityValidator
from mathmodel_ai.paper.registry import CitationRegistry, FigureRegistry, TableRegistry
from mathmodel_ai.paper.rendering import LaTeXRenderer, RenderedPaper
from mathmodel_ai.schemas.paper import (
    PaperArtifact,
    PaperArtifactKind,
    PaperCompileRecord,
    PaperCompileStatus,
    PaperManifest,
    PaperQualityStatus,
    PaperVersion,
)
from tests.mathematical.helpers import lp_model
from tests.paper.helpers import numeric_claim, paper_ir, result_evidence


@dataclass(frozen=True)
class _Bundle:
    store: LocalFileStore
    version: PaperVersion
    rendered: RenderedPaper
    compilation: PaperCompilation
    manifest: PaperManifest
    artifacts: list[PaperArtifact]


def _pdf(text: str) -> bytes:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    data = document.tobytes()
    document.close()
    return data


def _artifact(
    store: LocalFileStore,
    data: bytes,
    *,
    project_id: UUID,
    problem_id: UUID,
    paper_id: UUID,
    version: int,
    kind: PaperArtifactKind,
    name: str,
    mime_type: str,
) -> PaperArtifact:
    artifact_id = uuid4()
    stored = store.store_artifact(
        data,
        project_id=project_id,
        artifact_id=artifact_id,
        filename=name,
    )
    return PaperArtifact(
        artifact_id=artifact_id,
        project_id=project_id,
        problem_id=problem_id,
        paper_id=paper_id,
        paper_version=version,
        kind=kind,
        name=name,
        mime_type=mime_type,
        size_bytes=stored.size_bytes,
        sha256=stored.sha256,
        storage_key=stored.storage_key,
    )


def _bundle(
    root: Path,
    *,
    version_number: int = 1,
    paper_id: UUID | None = None,
    objective: float = 30,
    project_id: UUID | None = None,
    problem_id: UUID | None = None,
) -> _Bundle:
    store = LocalFileStore(root)
    evidence = result_evidence(
        objective=objective,
        project_id=project_id,
        problem_id=problem_id,
    )
    claim = numeric_claim(
        evidence,
        value=objective,
        paper_id=paper_id,
        text=f"The verified objective is {objective:g} kg.",
    ).model_copy(update={"paper_version": version_number})
    paper = paper_ir(evidence, claim).model_copy(
        update={"version": version_number, "status": PaperQualityStatus.VALIDATING}
    )
    rendered = LaTeXRenderer().render(
        paper=paper,
        claims=[claim],
        equations=EquationRegistry.from_model(lp_model()),
        figures=FigureRegistry([]),
        tables=TableRegistry([]),
        citations=CitationRegistry([]),
    )
    tex = _artifact(
        store,
        rendered.tex.encode(),
        project_id=evidence.project_id,
        problem_id=evidence.problem_id,
        paper_id=paper.paper_id,
        version=version_number,
        kind=PaperArtifactKind.TEX,
        name="paper.tex",
        mime_type="application/x-tex",
    )
    bib = _artifact(
        store,
        rendered.bibliography.encode(),
        project_id=evidence.project_id,
        problem_id=evidence.problem_id,
        paper_id=paper.paper_id,
        version=version_number,
        kind=PaperArtifactKind.BIB,
        name="references.bib",
        mime_type="application/x-bibtex",
    )
    pdf = _artifact(
        store,
        _pdf(claim.text),
        project_id=evidence.project_id,
        problem_id=evidence.problem_id,
        paper_id=paper.paper_id,
        version=version_number,
        kind=PaperArtifactKind.PDF,
        name="paper.pdf",
        mime_type="application/pdf",
    )
    compile_record = PaperCompileRecord(
        paper_id=paper.paper_id,
        paper_version=version_number,
        compiler="fixture-pdf",
        compiler_version="1",
        container_image="fixture@sha256:bound",
        container_image_id="sha256:bound",
        status=PaperCompileStatus.SUCCEEDED,
        exit_code=0,
        runtime_seconds=0.01,
        page_count=1,
        tex_hash=tex.sha256,
        bib_hash=bib.sha256,
        pdf_artifact_id=pdf.artifact_id,
    )
    compilation = PaperCompilation(record=compile_record, artifacts=(tex, bib, pdf))
    version = PaperVersion(
        paper_id=paper.paper_id,
        project_id=evidence.project_id,
        problem_id=evidence.problem_id,
        version=version_number,
        parent_version=version_number - 1 if version_number > 1 else None,
        revision_reason="artifact acceptance fixture",
        evidence_snapshot=paper.evidence_snapshot,
        paper_ir=paper,
        status=PaperQualityStatus.VALIDATING,
    )
    manifest, manifest_artifact = PaperBundleBuilder(store).build_manifest(
        version=version,
        references=[],
        figures=[],
        tables=[],
        artifacts=[tex, bib, pdf],
    )
    return _Bundle(
        store=store,
        version=version,
        rendered=rendered,
        compilation=compilation,
        manifest=manifest,
        artifacts=[tex, bib, pdf, manifest_artifact],
    )


def _validate(bundle: _Bundle):
    return ArtifactIntegrityValidator(bundle.store).validate(
        version=bundle.version,
        rendered=bundle.rendered,
        compilation=bundle.compilation,
        manifest=bundle.manifest,
        artifacts=bundle.artifacts,
        references=[],
        figures=[],
        tables=[],
    )


def test_d_clean_bundle_passes_then_pdf_artifact_swap_fails(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "d")
    assert _validate(bundle) == []
    pdf = next(item for item in bundle.artifacts if item.kind is PaperArtifactKind.PDF)
    bundle.store.resolve(pdf.storage_key).write_bytes(_pdf("The verified objective is 31 kg."))
    assert "DOCUMENT_INTEGRITY_ERROR" in {item.code for item in _validate(bundle)}


def test_j_bibliography_and_t_manifest_tampering_fail(tmp_path: Path) -> None:
    bib_bundle = _bundle(tmp_path / "j")
    bib = next(item for item in bib_bundle.artifacts if item.kind is PaperArtifactKind.BIB)
    bib_bundle.store.resolve(bib.storage_key).write_text(
        "@article{REF-fake, year={2099}}", encoding="utf-8"
    )
    assert "DOCUMENT_INTEGRITY_ERROR" in {item.code for item in _validate(bib_bundle)}

    manifest_bundle = _bundle(tmp_path / "t")
    manifest = next(
        item for item in manifest_bundle.artifacts if item.kind is PaperArtifactKind.MANIFEST
    )
    manifest_bundle.store.resolve(manifest.storage_key).write_text(
        '{"pdf_hash":"' + "0" * 64 + '"}', encoding="utf-8"
    )
    assert "DOCUMENT_INTEGRITY_ERROR" in {item.code for item in _validate(manifest_bundle)}


def test_e_u_paper_versions_and_snapshots_remain_immutable(tmp_path: Path) -> None:
    paper_id = uuid4()
    first = _bundle(tmp_path / "versions", paper_id=paper_id)
    first_version_json = first.version.model_dump_json()
    first_bytes = {
        item.artifact_id: first.store.read_bytes(item.storage_key) for item in first.artifacts
    }

    second = _bundle(
        tmp_path / "versions",
        version_number=2,
        paper_id=paper_id,
        objective=31,
        project_id=first.version.project_id,
        problem_id=first.version.problem_id,
    )
    assert first.version.model_dump_json() == first_version_json
    assert (
        first.version.evidence_snapshot.verified_result_id
        != second.version.evidence_snapshot.verified_result_id
    )
    assert all(
        first.store.read_bytes(item.storage_key) == first_bytes[item.artifact_id]
        for item in first.artifacts
    )
    assert _validate(first) == []
    assert _validate(second) == []


def test_artifact_path_traversal_is_rejected(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "safe")
    template = bundle.artifacts[0]
    with pytest.raises(ValidationError, match="name"):
        PaperArtifact.model_validate(
            {**template.model_dump(mode="json"), "name": "../../outside.png"}
        )
    with pytest.raises(ValidationError, match="storage key"):
        PaperArtifact.model_validate(
            {**template.model_dump(mode="json"), "storage_key": "../../secret.tex"}
        )
    assert sha256_bytes(bundle.store.read_bytes(template.storage_key)) == template.sha256
