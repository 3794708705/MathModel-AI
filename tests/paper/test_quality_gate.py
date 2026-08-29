from __future__ import annotations

from mathmodel_ai.mathematical.registry import EquationRegistry
from mathmodel_ai.paper.claims import ClaimEvidenceGraph
from mathmodel_ai.paper.hashing import sha256_text
from mathmodel_ai.paper.literature import claim_digest, reference_digest
from mathmodel_ai.paper.registry import (
    CitationRegistry,
    DocumentRegistry,
    FigureRegistry,
    TableRegistry,
)
from mathmodel_ai.paper.validators import PaperQualityGate
from mathmodel_ai.schemas.paper import (
    CitationMetadataCheck,
    CitationSupportCheck,
    CitationSupportStatus,
    DocumentObjectType,
    PaperBlock,
    PaperBlockType,
    PaperQualityStatus,
    ReferenceMetadataStatus,
)
from tests.mathematical.helpers import lp_model
from tests.paper.helpers import (
    direct_link,
    numeric_claim,
    paper_ir,
    reference_record,
    result_evidence,
)


def _quality_fixture():
    evidence = result_evidence()
    reference = reference_record(evidence.project_id)
    claim = numeric_claim(evidence).model_copy(update={"citation_refs": [reference.reference_id]})
    paper = paper_ir(evidence, claim).model_copy(
        update={"bibliography": [reference.reference_id], "claims": [claim]}
    )
    model = lp_model()
    registry = DocumentRegistry.from_model(model)
    registry.add(
        DocumentObjectType.REFERENCE,
        source_id=reference.reference_id,
        object_id=reference.reference_id,
        content=reference.model_dump(mode="json"),
    )
    for section in paper.sections:
        registry.add(
            DocumentObjectType.SECTION,
            source_id=section.section_id,
            object_id=section.section_id,
            content=section.model_dump(mode="json"),
        )
    registry.add(
        DocumentObjectType.CLAIM,
        source_id=claim.claim_id,
        object_id=claim.claim_id,
        content=claim.model_dump(mode="json"),
    )
    paper = paper.model_copy(update={"document_registry": registry.entries()})
    graph = ClaimEvidenceGraph([evidence])
    graph.add_claim(claim)
    graph.add_link(direct_link(claim, evidence))
    metadata = CitationMetadataCheck(
        project_id=evidence.project_id,
        reference_id=reference.reference_id,
        reference_digest=reference_digest(reference),
        status=ReferenceMetadataStatus.VERIFIED,
        field_matches={
            "title": True,
            "authors": True,
            "year": True,
            "venue": True,
            "doi": True,
        },
    )
    trusted = reference.trusted_excerpt
    assert trusted is not None
    support = CitationSupportCheck(
        project_id=evidence.project_id,
        claim_id=claim.claim_id,
        reference_id=reference.reference_id,
        claim_digest=claim_digest(claim),
        reference_digest=reference_digest(reference),
        status=CitationSupportStatus.SUPPORTED,
        trusted_text_hash=sha256_text(trusted),
        supporting_excerpt=trusted,
        rationale="fixture source directly supports the method context",
        reviewer_is_mock=False,
    )
    return paper, graph, model, reference, metadata, support


def _evaluate(*, mock_review: bool = False, compile_succeeded: bool = True):
    paper, graph, model, reference, metadata, support = _quality_fixture()
    return PaperQualityGate().evaluate(
        paper=paper,
        graph=graph,
        model=model,
        equations=EquationRegistry.from_model(model),
        figures=FigureRegistry([]),
        tables=TableRegistry([]),
        citations=CitationRegistry([reference]),
        metadata_checks=[metadata],
        support_checks=[support],
        compile_succeeded=compile_succeeded,
        independent_review_passed=True,
        independent_review_is_mock=mock_review,
    )


def test_clean_paper_requires_deterministic_gates_and_independent_review() -> None:
    report = _evaluate()
    assert report.status is PaperQualityStatus.READY_FOR_FINAL_JURY
    assert report.unsupported_claim_count == 0
    assert report.reference_count == report.verified_reference_count == 1


def test_pdf_compile_failure_prevents_final_jury_readiness() -> None:
    report = _evaluate(compile_succeeded=False)
    assert report.status is PaperQualityStatus.FAILED
    assert "PDF_COMPILE_ERROR" in {item.code for item in report.issues}


def test_mock_independent_audit_requires_human_review() -> None:
    report = _evaluate(mock_review=True)
    assert report.status is PaperQualityStatus.HUMAN_REVIEW
    assert "INDEPENDENT_REVIEW_REQUIRED" in {item.code for item in report.issues}


def test_hallucinated_prose_number_and_mock_citation_review_fail_gate() -> None:
    paper, graph, model, reference, metadata, support = _quality_fixture()
    paragraph = PaperBlock(
        block_id="BLOCK-hallucinated-number",
        block_type=PaperBlockType.PARAGRAPH,
        text="The method improves performance by 25%.",
    )
    section = paper.sections[0].model_copy(
        update={"blocks": [*paper.sections[0].blocks, paragraph]}
    )
    paper = paper.model_copy(update={"sections": [section]})
    mock_support = support.model_copy(update={"reviewer_is_mock": True})
    report = PaperQualityGate().evaluate(
        paper=paper,
        graph=graph,
        model=model,
        equations=EquationRegistry.from_model(model),
        figures=FigureRegistry([]),
        tables=TableRegistry([]),
        citations=CitationRegistry([reference]),
        metadata_checks=[metadata],
        support_checks=[mock_support],
        compile_succeeded=True,
        independent_review_passed=True,
        independent_review_is_mock=False,
    )
    assert report.status is PaperQualityStatus.HUMAN_REVIEW
    assert {"UNSUPPORTED_CLAIM", "MOCK_EVIDENCE"} <= {item.code for item in report.issues}
