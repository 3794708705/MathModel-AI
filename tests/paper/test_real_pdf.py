from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from mathmodel_ai.files.storage import LocalFileStore
from mathmodel_ai.mathematical.registry import EquationRegistry
from mathmodel_ai.paper.compiler import PDFCompiler
from mathmodel_ai.paper.registry import CitationRegistry, FigureRegistry, TableRegistry
from mathmodel_ai.paper.rendering import LaTeXRenderer
from mathmodel_ai.schemas.paper import PaperArtifactKind, PaperCompileStatus
from tests.mathematical.helpers import lp_model
from tests.paper.helpers import numeric_claim, paper_ir, result_evidence

IMAGE = "mathmodel-ai-paper:phase6"


def _image_available() -> bool:
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", IMAGE],
            capture_output=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _image_available(), reason=f"Docker image {IMAGE} is unavailable"),
]


def test_o_real_latex_compilation_produces_valid_pdf(tmp_path: Path) -> None:
    evidence = result_evidence()
    claim = numeric_claim(evidence)
    paper = paper_ir(evidence, claim)
    rendered = LaTeXRenderer().render(
        paper=paper,
        claims=[claim],
        equations=EquationRegistry.from_model(lp_model()),
        figures=FigureRegistry([]),
        tables=TableRegistry([]),
        citations=CitationRegistry([]),
    )
    store = LocalFileStore(tmp_path / "store")

    compilation = PDFCompiler(store=store, image=IMAGE).compile(
        rendered,
        project_id=evidence.project_id,
        problem_id=evidence.problem_id,
        paper_id=paper.paper_id,
        paper_version=paper.version,
    )

    assert compilation.record.status is PaperCompileStatus.SUCCEEDED
    assert compilation.record.exit_code == 0
    pdf_artifact = next(
        item for item in compilation.artifacts if item.kind is PaperArtifactKind.PDF
    )
    pdf = store.read_bytes(pdf_artifact.storage_key)
    assert pdf.startswith(b"%PDF-")
    assert len(pdf) > 1_000
    assert compilation.record.pdf_artifact_id == pdf_artifact.artifact_id
    assert compilation.record.network_disabled
    assert compilation.record.non_root
