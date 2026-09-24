from __future__ import annotations

import json
import subprocess
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from pypdf import PdfWriter

from mathmodel_ai.files.storage import LocalFileStore
from mathmodel_ai.mathematical.registry import EquationRegistry
from mathmodel_ai.paper.assets import FigureAgent, TableAgent
from mathmodel_ai.paper.compiler import PDFCompiler
from mathmodel_ai.paper.hashing import canonical_json_bytes, sha256_bytes
from mathmodel_ai.paper.literature import CrossrefLiteratureSource
from mathmodel_ai.paper.registry import (
    CitationRegistry,
    DocumentRegistry,
    FigureRegistry,
    TableRegistry,
)
from mathmodel_ai.paper.rendering import (
    BibTeXRenderer,
    LaTeXRenderer,
    LatexRenderError,
    LaTeXSafetyValidator,
    escape_latex,
)
from mathmodel_ai.paper.validators import CrossReferenceValidator, SymbolConsistencyValidator
from mathmodel_ai.schemas.paper import (
    DocumentObjectType,
    DocumentRegistryEntry,
    FigureType,
    LiteratureNeedType,
    LiteratureSearchNeed,
    PaperBlock,
    PaperBlockType,
    PaperCompileStatus,
    PaperValidationSeverity,
    ReferenceMetadataStatus,
    RegistryGenerationStatus,
)
from tests.mathematical.helpers import lp_model
from tests.paper.helpers import numeric_claim, paper_ir, reference_record, result_evidence


def _assets(tmp_path: Path):
    evidence = result_evidence()
    paper_id = uuid4()
    store = LocalFileStore(tmp_path / "storage")
    figure, figure_artifacts = FigureAgent(store).generate(
        project_id=evidence.project_id,
        problem_id=evidence.problem_id,
        paper_id=paper_id,
        paper_version=1,
        figure_id="FIG-001",
        title="Sensitivity",
        caption="Verified sensitivity response.",
        figure_type=FigureType.SENSITIVITY_CURVE,
        data_payload={"x": [-0.1, 0.1], "y": [27.0, 33.0]},
        source_evidence_refs=[evidence.evidence_id],
        evidence=[evidence],
    )
    table, table_artifact = TableAgent(store).generate(
        project_id=evidence.project_id,
        problem_id=evidence.problem_id,
        paper_id=paper_id,
        paper_version=1,
        table_id="TAB-001",
        title="Result",
        caption="Verified result.",
        columns=["Metric", "Value"],
        rows=[["objective", 30]],
        source_evidence_refs=[evidence.evidence_id],
        evidence=[evidence],
    )
    return evidence, store, figure, figure_artifacts, table, table_artifact


def test_i_j_l_cross_reference_missing_and_unreferenced_assets(tmp_path: Path) -> None:
    evidence, _, figure, _, table, _ = _assets(tmp_path)
    claim = numeric_claim(evidence)
    paper = paper_ir(
        evidence,
        claim,
        equation_refs=["EQ-999"],
        figure_refs=["FIG-404"],
        table_refs=["TAB-404"],
        citation_refs=["REF-missing"],
    )
    issues = CrossReferenceValidator().validate(
        paper,
        equations=EquationRegistry.from_model(lp_model()),
        figures=FigureRegistry([figure]),
        tables=TableRegistry([table]),
        citations=CitationRegistry([]),
    )
    codes = {item.code for item in issues}
    assert {"MISSING_EQUATION", "MISSING_FIGURE", "MISSING_TABLE", "MISSING_REFERENCE"} <= codes
    unreferenced = [item for item in issues if item.code == "UNREFERENCED_FIGURE"]
    assert len(unreferenced) == 1
    assert unreferenced[0].severity is PaperValidationSeverity.WARNING


def test_k_symbol_definition_conflict_fails() -> None:
    model = lp_model()
    issues = SymbolConsistencyValidator().validate(model, {"x": ("capacity", "kg")})
    assert [item.code for item in issues] == ["SYMBOL_CONFLICT"]


def test_m_n_table_and_figure_tamper_are_detected(tmp_path: Path) -> None:
    _, store, figure, figure_artifacts, table, table_artifact = _assets(tmp_path)
    by_id = {item.artifact_id: item for item in figure_artifacts}
    figure_errors = FigureRegistry([figure]).verify(
        figure.figure_id,
        data_bytes=canonical_json_bytes({"x": [-0.1, 0.1], "y": [99, 99]}),
        code_bytes=store.read_bytes(by_id[figure.code_artifact_id].storage_key),
        image_bytes=store.read_bytes(by_id[figure.image_artifact_id].storage_key),
    )
    assert any("FIGURE_DATA" in item for item in figure_errors)
    table_errors = TableRegistry([table]).verify(
        table.table_id,
        data_bytes=canonical_json_bytes({"columns": table.columns, "rows": [["objective", 99]]}),
    )
    assert table_errors == ["DOCUMENT_INTEGRITY_ERROR:TABLE_DATA"]
    assert table_artifact.sha256 == table.data_hash


def test_document_and_citation_registries_reject_duplicates() -> None:
    registry = DocumentRegistry()
    registry.add(DocumentObjectType.SECTION, source_id="source", content={"x": 1})
    with pytest.raises(ValueError, match="already registered"):
        registry.add(DocumentObjectType.SECTION, source_id="source", content={"x": 2})
    reference = reference_record(uuid4())
    with pytest.raises(ValueError, match="duplicate reference"):
        CitationRegistry([reference, reference])


def test_registry_identity_and_lookup_contracts_are_strict() -> None:
    registry = DocumentRegistry()
    section = registry.add(DocumentObjectType.SECTION, source_id="source-a", content={"x": 1})
    assert registry.get(section.object_id) == section
    assert registry.require(section.object_id) == section
    with pytest.raises(KeyError):
        registry.require("SEC-missing")
    with pytest.raises(ValueError, match="duplicate document object ID"):
        registry.register(section)
    with pytest.raises(ValueError, match="already registered"):
        registry.add(
            DocumentObjectType.SECTION,
            source_id="source-a",
            object_id="SEC-other",
            content={"x": 2},
        )
    with pytest.raises(ValueError, match="prefix"):
        registry.register(
            DocumentRegistryEntry(
                object_id="FIG-wrong-prefix",
                object_type=DocumentObjectType.TABLE,
                source_id="source-table",
                order=1,
                content_hash="a" * 64,
            )
        )

    reference = reference_record(uuid4())
    second_id = reference.model_copy(update={"reference_id": "REF-second"})
    with pytest.raises(ValueError, match="duplicate DOI"):
        CitationRegistry([reference, second_id])
    second_source = reference.model_copy(
        update={"reference_id": "REF-second", "doi": "10.0000/second"}
    )
    with pytest.raises(ValueError, match="duplicate literature source"):
        CitationRegistry([reference, second_source])
    citations = CitationRegistry([reference])
    assert citations.get(reference.reference_id) == reference
    assert citations.get("REF-missing") is None


def test_figure_and_table_registries_report_missing_unverified_and_hash_mismatch(
    tmp_path: Path,
) -> None:
    _, store, figure, figure_artifacts, table, table_artifact = _assets(tmp_path)
    with pytest.raises(ValueError, match="duplicate figure"):
        FigureRegistry([figure, figure])
    with pytest.raises(ValueError, match="duplicate table"):
        TableRegistry([table, table])
    assert FigureRegistry([]).verify(
        "FIG-missing", data_bytes=b"", code_bytes=b"", image_bytes=b""
    ) == ["MISSING_FIGURE"]
    assert TableRegistry([]).verify("TAB-missing", data_bytes=b"") == ["MISSING_TABLE"]

    by_id = {item.artifact_id: item for item in figure_artifacts}
    unverified_figure = figure.model_copy(
        update={"generation_status": RegistryGenerationStatus.GENERATED}
    )
    errors = FigureRegistry([unverified_figure]).verify(
        figure.figure_id,
        data_bytes=store.read_bytes(by_id[figure.data_artifact_id].storage_key),
        code_bytes=b"tampered code",
        image_bytes=b"tampered image",
    )
    assert {
        "DOCUMENT_INTEGRITY_ERROR:FIGURE_CODE",
        "DOCUMENT_INTEGRITY_ERROR:FIGURE_IMAGE",
        "UNVERIFIED_EVIDENCE:FIGURE",
    } <= set(errors)
    unverified_table = table.model_copy(
        update={"generation_status": RegistryGenerationStatus.GENERATED}
    )
    assert "UNVERIFIED_EVIDENCE:TABLE" in TableRegistry([unverified_table]).verify(
        table.table_id,
        data_bytes=store.read_bytes(table_artifact.storage_key),
    )


def test_document_registry_hash_and_citation_bibliography_are_cross_checked() -> None:
    evidence = result_evidence()
    reference = reference_record(evidence.project_id)
    claim = numeric_claim(evidence).model_copy(update={"citation_refs": [reference.reference_id]})
    paper = paper_ir(evidence, claim).model_copy(update={"claims": [claim]})
    registry = DocumentRegistry.from_model(lp_model())
    section = paper.sections[0]
    registry.add(
        DocumentObjectType.SECTION,
        source_id=section.section_id,
        object_id=section.section_id,
        content=section.model_dump(mode="json"),
    )
    claim_entry = registry.add(
        DocumentObjectType.CLAIM,
        source_id=claim.claim_id,
        object_id=claim.claim_id,
        content=claim.model_dump(mode="json"),
    )
    registry.add(
        DocumentObjectType.REFERENCE,
        source_id=reference.reference_id,
        object_id=reference.reference_id,
        content=reference.model_dump(mode="json"),
    )
    registry.add(
        DocumentObjectType.SECTION,
        source_id="SEC-orphan",
        object_id="SEC-orphan",
        content={"orphan": True},
    )
    entries = [
        item.model_copy(update={"content_hash": "f" * 64})
        if item.object_id == claim_entry.object_id
        else item
        for item in registry.entries()
    ]
    paper = paper.model_copy(update={"document_registry": entries})

    issues = CrossReferenceValidator().validate(
        paper,
        equations=EquationRegistry.from_model(lp_model()),
        figures=FigureRegistry([]),
        tables=TableRegistry([]),
        citations=CitationRegistry([reference]),
    )
    codes = {item.code for item in issues}
    assert "DOCUMENT_INTEGRITY_ERROR" in codes
    assert "MISSING_REFERENCE" in codes

    duplicate_registry = paper.model_copy(
        update={"document_registry": [*paper.document_registry, paper.document_registry[0]]}
    )
    duplicate_issues = CrossReferenceValidator().validate(
        duplicate_registry,
        equations=EquationRegistry.from_model(lp_model()),
        figures=FigureRegistry([]),
        tables=TableRegistry([]),
        citations=CitationRegistry([reference]),
    )
    assert any("duplicate document registry ID" in item.message for item in duplicate_issues)


def test_p_broken_citation_and_latex_injection_are_rejected(tmp_path: Path) -> None:
    evidence, _, figure, _, table, _ = _assets(tmp_path)
    claim = numeric_claim(evidence)
    paper = paper_ir(
        evidence,
        claim,
        blocks=[
            PaperBlock(
                block_id="BLOCK-broken-ref",
                block_type=PaperBlockType.CLAIM,
                claim_ref=claim.claim_id,
                citation_refs=["REF-missing"],
            )
        ],
    )
    with pytest.raises(LatexRenderError, match=r"unsafe citation|unknown"):
        LaTeXRenderer().render(
            paper=paper,
            claims=[claim],
            equations=EquationRegistry.from_model(lp_model()),
            figures=FigureRegistry([figure]),
            tables=TableRegistry([table]),
            citations=CitationRegistry([]),
        )
    with pytest.raises(LatexRenderError, match="unsafe"):
        LaTeXSafetyValidator().validate_equation(r"\input{/etc/passwd}")
    assert escape_latex(r"x_1 & 50% \write18") == (r"x\_1 \& 50\% \textbackslash{}write18")


def test_renderer_produces_tex_and_verified_bibtex(tmp_path: Path) -> None:
    evidence, _, figure, _, table, _ = _assets(tmp_path)
    reference = reference_record(evidence.project_id)
    claim = numeric_claim(evidence).model_copy(update={"citation_refs": [reference.reference_id]})
    paper = paper_ir(
        evidence,
        claim,
        blocks=[
            PaperBlock(
                block_id="BLOCK-claim",
                block_type=PaperBlockType.CLAIM,
                claim_ref=claim.claim_id,
            ),
            PaperBlock(
                block_id="BLOCK-equation",
                block_type=PaperBlockType.EQUATION,
                equation_ref="EQ-objective",
            ),
            PaperBlock(
                block_id="BLOCK-figure",
                block_type=PaperBlockType.FIGURE,
                figure_ref=figure.figure_id,
            ),
            PaperBlock(
                block_id="BLOCK-table",
                block_type=PaperBlockType.TABLE,
                table_ref=table.table_id,
            ),
        ],
        equation_refs=["EQ-objective"],
        figure_refs=[figure.figure_id],
        table_refs=[table.table_id],
    ).model_copy(update={"bibliography": [reference.reference_id], "claims": [claim]})
    rendered = LaTeXRenderer().render(
        paper=paper,
        claims=[claim],
        equations=EquationRegistry.from_model(lp_model()),
        figures=FigureRegistry([figure]),
        tables=TableRegistry([table]),
        citations=CitationRegistry([reference]),
    )
    assert r"\graphicspath{{figures/}}" in rendered.tex
    assert r"\includegraphics" in rendered.tex
    assert r"\begin{equation}" in rendered.tex
    assert r"\begin{tabularx}{\linewidth}" in rendered.tex
    assert f"@article{{{reference.reference_id}" in rendered.bibliography
    assert BibTeXRenderer().render(CitationRegistry([reference])) == rendered.bibliography

    unverified = reference.model_copy(update={"metadata_status": ReferenceMetadataStatus.PARTIAL})
    with pytest.raises(LatexRenderError, match="unverified"):
        BibTeXRenderer().render(CitationRegistry([unverified]))


def test_renderer_breaks_long_tables_across_pages_and_renders_repeats_as_refs(
    tmp_path: Path,
) -> None:
    evidence, _, _, _, table, _ = _assets(tmp_path)
    claim = numeric_claim(evidence)
    long_table = table.model_copy(
        update={"rows": [[f"symbol_{index}", "A descriptive meaning"] for index in range(20)]}
    )
    blocks = [
        PaperBlock(
            block_id=f"BLOCK-table-{index}",
            block_type=PaperBlockType.TABLE,
            table_ref=table.table_id,
        )
        for index in range(2)
    ]
    paper = paper_ir(evidence, claim, blocks=blocks, table_refs=[table.table_id])
    rendered = LaTeXRenderer().render(
        paper=paper,
        claims=[claim],
        equations=EquationRegistry.from_model(lp_model()),
        figures=FigureRegistry([]),
        tables=TableRegistry([long_table]),
        citations=CitationRegistry([]),
    )
    assert rendered.tex.count(r"\begin{longtable}") == 1
    assert rendered.tex.count(r"\caption{Verified result.}") == 1
    assert rendered.tex.count(r"Table~\ref{TAB-001}") == 1
    assert r"\endfirsthead" in rendered.tex
    assert r"\endhead" in rendered.tex


def test_renderer_handles_structured_prose_lists_and_nonanonymous_profile() -> None:
    evidence = result_evidence()
    claim = numeric_claim(evidence)
    reference = reference_record(evidence.project_id).model_copy(update={"doi": None, "url": None})
    blocks = [
        PaperBlock(
            block_id="BLOCK-paragraph",
            block_type=PaperBlockType.PARAGRAPH,
            text="Structured prose & evidence.",
            citation_refs=[reference.reference_id],
        ),
        PaperBlock(
            block_id="BLOCK-subsection",
            block_type=PaperBlockType.SUBSECTION,
            text="Deterministic checks",
        ),
        PaperBlock(
            block_id="BLOCK-list",
            block_type=PaperBlockType.LIST,
            items=["First invariant", "Second invariant"],
        ),
    ]
    profile = paper_ir(evidence, claim).competition_profile.model_copy(update={"anonymous": False})
    paper = paper_ir(evidence, claim, blocks=blocks, profile=profile).model_copy(
        update={"bibliography": [reference.reference_id]}
    )
    rendered = LaTeXRenderer().render(
        paper=paper,
        claims=[claim],
        equations=EquationRegistry.from_model(lp_model()),
        figures=FigureRegistry([]),
        tables=TableRegistry([]),
        citations=CitationRegistry([reference]),
    )
    assert r"\author{MathModel AI}" in rendered.tex
    assert r"Structured prose \& evidence." in rendered.tex
    assert r"\subsection*{Deterministic checks}" in rendered.tex
    assert r"\begin{itemize}" in rendered.tex
    assert "doi =" not in rendered.bibliography
    assert "url =" not in rendered.bibliography


def test_renderer_rejects_duplicate_claims_unknown_objects_and_unsafe_paths() -> None:
    evidence = result_evidence()
    claim = numeric_claim(evidence)
    renderer = LaTeXRenderer()
    base = paper_ir(evidence, claim)
    registries = {
        "equations": EquationRegistry.from_model(lp_model()),
        "figures": FigureRegistry([]),
        "tables": TableRegistry([]),
        "citations": CitationRegistry([]),
    }
    with pytest.raises(LatexRenderError, match="duplicate claim"):
        renderer.render(paper=base, claims=[claim, claim], **registries)

    unknown_blocks = [
        PaperBlock(
            block_id="BLOCK-unknown-claim",
            block_type=PaperBlockType.CLAIM,
            claim_ref="CLAIM-missing",
        ),
        PaperBlock(
            block_id="BLOCK-unknown-equation",
            block_type=PaperBlockType.EQUATION,
            equation_ref="EQ-missing",
        ),
        PaperBlock(
            block_id="BLOCK-unknown-figure",
            block_type=PaperBlockType.FIGURE,
            figure_ref="FIG-missing",
        ),
        PaperBlock(
            block_id="BLOCK-unknown-table",
            block_type=PaperBlockType.TABLE,
            table_ref="TAB-missing",
        ),
    ]
    for block, message in zip(
        unknown_blocks,
        ("unknown claim", "unknown equation", "unknown figure", "unknown table"),
        strict=True,
    ):
        section = base.sections[0].model_copy(update={"blocks": [block]})
        invalid = base.model_copy(update={"sections": [section]})
        with pytest.raises(LatexRenderError, match=message):
            renderer.render(paper=invalid, claims=[claim], **registries)

    with pytest.raises(LatexRenderError, match="path traversal"):
        LaTeXSafetyValidator().validate_equation(r"x = ../secret")
    with pytest.raises(LatexRenderError, match="unsafe"):
        LaTeXSafetyValidator().validate_rendered(r"\write18{touch pwned}")
    with pytest.raises(LatexRenderError, match="unsafe citation"):
        renderer._citations(["REF-invalid/path"])


class _CompilerRunner:
    def __call__(self, command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        del kwargs
        if command[1:3] == ["image", "inspect"]:
            return subprocess.CompletedProcess(command, 0, b"sha256:paper-image\n", b"")
        if command[1] == "rm":
            return subprocess.CompletedProcess(command, 0, b"", b"")
        output_mount = next(
            item for item in command if item.startswith("type=bind") and "target=/output" in item
        )
        source = output_mount.split("source=", 1)[1].split(",target=", 1)[0]
        writer = PdfWriter()
        writer.add_blank_page(width=612, height=792)
        with Path(source, "paper.pdf").open("wb") as handle:
            writer.write(handle)
        return subprocess.CompletedProcess(command, 0, b"compiled", b"")


def test_o_pdf_compiler_records_real_artifact_contract_with_safe_command(tmp_path: Path) -> None:
    evidence, store, figure, figure_artifacts, table, _ = _assets(tmp_path)
    claim = numeric_claim(evidence)
    paper = paper_ir(evidence, claim)
    rendered = LaTeXRenderer().render(
        paper=paper,
        claims=[claim],
        equations=EquationRegistry.from_model(lp_model()),
        figures=FigureRegistry([figure]),
        tables=TableRegistry([table]),
        citations=CitationRegistry([]),
    )
    by_id = {item.artifact_id: item for item in figure_artifacts}
    compilation = PDFCompiler(store=store, runner=_CompilerRunner()).compile(
        rendered,
        project_id=evidence.project_id,
        problem_id=evidence.problem_id,
        paper_id=paper.paper_id,
        paper_version=1,
        figure_images={
            "FIG-001.png": store.read_bytes(by_id[figure.image_artifact_id].storage_key)
        },
    )
    assert compilation.record.status is PaperCompileStatus.SUCCEEDED
    assert compilation.record.pdf_artifact_id is not None
    assert {item.name for item in compilation.artifacts} == {
        "paper.tex",
        "references.bib",
        "paper.pdf",
    }
    assert compilation.record.network_disabled
    assert compilation.record.non_root


@pytest.mark.asyncio
async def test_crossref_adapter_uses_retrieved_metadata_without_invention() -> None:
    response = {
        "message": {
            "items": [
                {
                    "title": ["Verified Work"],
                    "author": [{"given": "Ada", "family": "Lovelace"}],
                    "issued": {"date-parts": [[1843]]},
                    "container-title": ["Scientific Notes"],
                    "DOI": "10.1000/verified",
                    "URL": "https://doi.org/10.1000/verified",
                    "abstract": "<jats:p>Verified abstract.</jats:p>",
                }
            ]
        }
    }

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, json=response)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    source = CrossrefLiteratureSource(client=client)
    records = await source.search(
        LiteratureSearchNeed(
            need_id="LITNEED-lp",
            need_type=LiteratureNeedType.THEORY,
            query="linear programming",
            purpose="theory",
        ),
        uuid4(),
    )
    await client.aclose()
    assert len(records) == 1
    assert records[0].doi == "10.1000/verified"
    assert records[0].authors == ["Ada Lovelace"]
    assert records[0].raw_metadata_hash == sha256_bytes(
        json.dumps(response["message"]["items"][0], sort_keys=True, separators=(",", ":")).encode()
    )


@pytest.mark.asyncio
async def test_crossref_adapter_rejects_malformed_or_missing_records() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/works/missing"):
            return httpx.Response(404)
        return httpx.Response(200, json={"message": {"items": "not-a-list"}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    source = CrossrefLiteratureSource(client=client, mailto="audit@example.invalid")
    records = await source.search(
        LiteratureSearchNeed(
            need_id="LITNEED-malformed",
            need_type=LiteratureNeedType.THEORY,
            query="linear programming",
            purpose="adapter hardening",
        ),
        uuid4(),
    )
    assert records == []
    assert await source.resolve("missing", uuid4()) is None
    assert CrossrefLiteratureSource._record({"title": ["Incomplete"]}, uuid4()) is None
    await source.aclose()


@pytest.mark.asyncio
async def test_crossref_adapter_retries_bounded_rate_limit(monkeypatch) -> None:
    calls = 0
    delays = []

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": "3"})
        return httpx.Response(200, json={"message": {"items": []}})

    async def record_delay(seconds: float) -> None:
        delays.append(seconds)

    monkeypatch.setattr("mathmodel_ai.paper.literature.asyncio.sleep", record_delay)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    source = CrossrefLiteratureSource(client=client)
    records = await source.search(
        LiteratureSearchNeed(
            need_id="LITNEED-rate", need_type=LiteratureNeedType.THEORY,
            query="population dynamics", purpose="rate limit regression",
        ),
        uuid4(),
    )
    assert records == []
    assert calls == 2
    assert delays == [3.0]
    await client.aclose()


def test_crossref_books_use_retrieved_publisher_and_published_year() -> None:
    project_id = uuid4()
    record = CrossrefLiteratureSource._record(
        {
            "title": ["Linear Programming and Extensions"],
            "author": [{"given": "George", "family": "Dantzig"}],
            "published": {"date-parts": [[1963]]},
            "publisher": "Princeton University Press",
            "DOI": "10.1000/book",
            "URL": "https://doi.org/10.1000/book",
        },
        project_id,
    )

    assert record is not None
    assert record.year == 1963
    assert record.venue == "Princeton University Press"
