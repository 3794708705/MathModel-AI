from __future__ import annotations

from uuid import uuid4

import pytest

from mathmodel_ai.files.storage import LocalFileStore
from mathmodel_ai.mathematical.digests import mathematical_model_digest
from mathmodel_ai.mathematical.registry import EquationRegistry
from mathmodel_ai.paper.assets import FigureAgent, TableAgent
from mathmodel_ai.paper.claims import (
    ClaimEvidenceGraph,
    NumericClaimValidator,
    PaperNumericFormattingPolicy,
)
from mathmodel_ai.paper.compiler import PDFCompiler
from mathmodel_ai.paper.hashing import sha256_json, sha256_text
from mathmodel_ai.paper.integrity import (
    AssetSemanticIntegrityValidator,
    CitationFreshnessValidator,
    EvidenceSnapshotValidator,
    RenderedContentValidator,
    SubproblemCoverageValidator,
)
from mathmodel_ai.paper.literature import (
    CitationMetadataVerifier,
    CitationSupportVerifier,
    FixtureLiteratureSource,
    reference_digest,
)
from mathmodel_ai.paper.registry import CitationRegistry, FigureRegistry, TableRegistry
from mathmodel_ai.paper.rendering import (
    LaTeXRenderer,
    LatexRenderError,
    LaTeXSafetyValidator,
    RenderedPaper,
    escape_latex,
)
from mathmodel_ai.paper.validators import (
    CrossReferenceValidator,
    FactualIntegrityValidator,
    HallucinatedNumberDetector,
    SymbolNarrativeValidator,
)
from mathmodel_ai.schemas.paper import (
    CitationSupportDraft,
    CitationSupportStatus,
    Claim,
    ClaimEvidenceLink,
    ClaimEvidenceSupport,
    ClaimImportance,
    ClaimType,
    ComparisonClaimValue,
    ComparisonDirection,
    EvidenceRecord,
    EvidenceType,
    FigureType,
    NumericClaimValue,
    PaperBlock,
    PaperBlockType,
    ProblemRequirement,
    SubproblemCoverageRecord,
)
from tests.mathematical.helpers import lp_model
from tests.paper.helpers import (
    evidence_snapshot,
    numeric_claim,
    paper_ir,
    reference_record,
    result_evidence,
)


def _graph(claim: Claim, evidence: EvidenceRecord) -> ClaimEvidenceGraph:
    graph = ClaimEvidenceGraph([evidence])
    graph.add_claim(claim)
    graph.add_link(
        ClaimEvidenceLink(
            project_id=claim.project_id,
            claim_id=claim.claim_id,
            evidence_id=evidence.evidence_id,
            support=ClaimEvidenceSupport.DIRECT_SUPPORT,
            source_field=(
                claim.structured_value.source_field
                if isinstance(claim.structured_value, NumericClaimValue)
                else None
            ),
            rationale="adversarial direct binding",
        )
    )
    return graph


def _typed_evidence(
    base: EvidenceRecord, evidence_type: EvidenceType, payload: dict
) -> EvidenceRecord:
    data = base.model_dump(mode="json")
    data.update(
        {
            "evidence_id": str(uuid4()),
            "evidence_type": evidence_type.value,
            "source_type": evidence_type.value.casefold(),
            "source_id": f"{evidence_type.value.casefold()}-fixture",
            "structured_payload": payload,
        }
    )
    data["provenance"]["source_hash"] = sha256_json(payload)
    return EvidenceRecord.model_validate(data)


def _render_fixture(evidence: EvidenceRecord, claim: Claim):
    model = lp_model()
    equations = EquationRegistry.from_model(model)
    paper = paper_ir(evidence, claim)
    rendered = LaTeXRenderer().render(
        paper=paper,
        claims=[claim],
        equations=equations,
        figures=FigureRegistry([]),
        tables=TableRegistry([]),
        citations=CitationRegistry([]),
    )
    return paper, model, equations, rendered


def test_b_numeric_text_structured_rounding_units_and_comparison_attacks() -> None:
    evidence = result_evidence(objective=12.43782, unit="kg")
    rounded = numeric_claim(evidence, value=12.43782, text="The value is 12.44 kg.")
    linked = _graph(rounded, evidence).evidence_for(rounded.claim_id)
    assert NumericClaimValidator().validate(rounded, linked) == []
    assert PaperNumericFormattingPolicy().text_matches(
        rounded.model_copy(update={"text": "The value is 12.4 kg."})
    )
    assert not PaperNumericFormattingPolicy().text_matches(
        rounded.model_copy(update={"text": "The value is 13 kg."})
    )

    drifted_text = rounded.model_copy(update={"text": "The value is 31 kg."})
    assert "NUMERIC_CLAIM_MISMATCH" in {
        item.code for item in NumericClaimValidator().validate(drifted_text, linked)
    }

    tonne_evidence = result_evidence(objective=1000, unit="kg")
    one_tonne = numeric_claim(
        tonne_evidence, value=1, unit="t", text="The verified objective is 1 t."
    )
    assert (
        NumericClaimValidator().validate(
            one_tonne, _graph(one_tonne, tonne_evidence).evidence_for(one_tonne.claim_id)
        )
        == []
    )
    wrong_tonnes = numeric_claim(
        result_evidence(objective=30, unit="kg"),
        value=30,
        unit="t",
        text="The verified objective is 30 t.",
    )
    wrong_evidence = result_evidence(objective=30, unit="kg")
    wrong_tonnes = wrong_tonnes.model_copy(
        update={
            "project_id": wrong_evidence.project_id,
            "evidence_refs": [wrong_evidence.evidence_id],
        }
    )
    assert "UNIT_CLAIM_MISMATCH" in {
        item.code
        for item in NumericClaimValidator().validate(
            wrong_tonnes,
            _graph(wrong_tonnes, wrong_evidence).evidence_for(wrong_tonnes.claim_id),
        )
    }

    comparison_evidence = _typed_evidence(
        result_evidence(),
        EvidenceType.RESULT,
        {
            "baseline": {"value": 100, "unit": "kg"},
            "verified": {"value": 80, "unit": "kg"},
        },
    )
    comparison = numeric_claim(comparison_evidence).model_copy(
        update={
            "claim_id": "CLAIM-comparison",
            "claim_type": ClaimType.COMPARISON,
            "text": "The value increased by 20%.",
            "structured_value": ComparisonClaimValue(
                baseline_value=100,
                verified_value=80,
                percentage_change=-20,
                direction=ComparisonDirection.DECREASE,
                unit="kg",
                baseline_source_field="baseline",
                verified_source_field="verified",
            ),
        }
    )
    graph = ClaimEvidenceGraph([comparison_evidence])
    graph.add_claim(comparison)
    for field in ("baseline", "verified"):
        graph.add_link(
            ClaimEvidenceLink(
                project_id=comparison.project_id,
                claim_id=comparison.claim_id,
                evidence_id=comparison_evidence.evidence_id,
                support=ClaimEvidenceSupport.DIRECT_SUPPORT,
                source_field=field,
                rationale="comparison operand",
            )
        )
    assert "NUMERIC_CLAIM_MISMATCH" in {
        item.code
        for item in NumericClaimValidator().validate(
            comparison, graph.evidence_for(comparison.claim_id)
        )
    }


def test_c_rendered_numeric_drift_is_detected() -> None:
    evidence = result_evidence()
    claim = numeric_claim(evidence)
    paper, _, equations, rendered = _render_fixture(evidence, claim)
    tampered_tex = rendered.tex.replace("30 kg", "31 kg")
    tampered = RenderedPaper(
        tex=tampered_tex,
        bibliography=rendered.bibliography,
        tex_hash=sha256_text(tampered_tex),
        bib_hash=rendered.bib_hash,
    )
    issues = RenderedContentValidator().validate(
        paper,
        tampered,
        equations,
        FigureRegistry([]),
        TableRegistry([]),
        CitationRegistry([]),
    )
    assert "DOCUMENT_INTEGRITY_ERROR" in {item.code for item in issues}


def test_f_g_equation_semantic_and_model_version_drift_are_detected() -> None:
    evidence = result_evidence()
    claim = numeric_claim(evidence)
    model = lp_model()
    equations = EquationRegistry.from_model(model)
    equation = model.equations[0]
    equation_block = PaperBlock(
        block_id="BLOCK-equation-audit",
        block_type=PaperBlockType.EQUATION,
        equation_ref=equation.equation_id,
    )
    paper = paper_ir(
        evidence,
        claim,
        blocks=[
            PaperBlock(
                block_id="BLOCK-result",
                block_type=PaperBlockType.CLAIM,
                claim_ref=claim.claim_id,
            ),
            equation_block,
        ],
        equation_refs=[equation.equation_id],
    )
    rendered = LaTeXRenderer().render(
        paper=paper,
        claims=[claim],
        equations=equations,
        figures=FigureRegistry([]),
        tables=TableRegistry([]),
        citations=CitationRegistry([]),
    )
    replacement = equation.latex.replace(r"\ge", r"\le")
    if replacement == equation.latex:
        replacement = equation.latex + "+1"
    altered_tex = rendered.tex.replace(equation.latex, replacement)
    altered = rendered.__class__(
        tex=altered_tex,
        bibliography=rendered.bibliography,
        tex_hash=sha256_text(altered_tex),
        bib_hash=rendered.bib_hash,
    )
    assert "DOCUMENT_INTEGRITY_ERROR" in {
        item.code
        for item in RenderedContentValidator().validate(
            paper,
            altered,
            equations,
            FigureRegistry([]),
            TableRegistry([]),
            CitationRegistry([]),
        )
    }

    snapshot = evidence_snapshot(evidence).model_copy(
        update={
            "verified_model_id": model.model_id,
            "verified_model_version": model.version,
            "verified_model_digest": mathematical_model_digest(model),
        }
    )
    wrong_registry = EquationRegistry.from_model(
        model.model_copy(update={"version": model.version + 1})
    )
    assert "WRONG_EQUATION_VERSION" in {
        item.code
        for item in EvidenceSnapshotValidator().validate(
            snapshot, [evidence], model, wrong_registry
        )
    }


@pytest.mark.asyncio
async def test_h_i_j_citation_support_and_metadata_are_content_bound() -> None:
    evidence = result_evidence()
    claim = numeric_claim(evidence).model_copy(update={"citation_refs": ["REF-real"]})
    reference = reference_record(evidence.project_id)
    source = FixtureLiteratureSource((reference,))
    metadata = await CitationMetadataVerifier().verify(reference, source)
    verified_reference = CitationMetadataVerifier.verified_copy(reference, metadata)
    trusted = verified_reference.trusted_excerpt
    assert trusted is not None
    support = CitationSupportVerifier().verify(
        claim=claim,
        reference=verified_reference,
        draft=CitationSupportDraft(
            status=CitationSupportStatus.SUPPORTED,
            supporting_excerpt=trusted,
            rationale="exact trusted excerpt",
        ),
        reviewer_is_mock=False,
    )
    changed_claim = claim.model_copy(update={"text": "An unrelated changed claim."})
    assert "REFERENCE_NOT_SUPPORTED" in {
        item.code
        for item in CitationFreshnessValidator().validate(
            [changed_claim], [verified_reference], [metadata], [support]
        )
    }
    tampered_reference = verified_reference.model_copy(update={"year": 2023})
    assert reference_digest(tampered_reference) != metadata.reference_digest
    assert "REFERENCE_METADATA_CONFLICT" in {
        item.code
        for item in CitationFreshnessValidator().validate(
            [claim], [tampered_reference], [metadata], [support]
        )
    }


def test_k_l_figure_and_table_semantic_swaps_are_detected(tmp_path) -> None:
    base = result_evidence()
    experiment_id = uuid4()
    sensitivity = _typed_evidence(
        base,
        EvidenceType.SENSITIVITY,
        {
            "experiments": [
                {
                    "experiment_id": str(experiment_id),
                    "objective_value": 33.0,
                    "perturbations": [{"symbol": "p", "fraction": 0.1}],
                },
                {
                    "experiment_id": str(uuid4()),
                    "objective_value": 27.0,
                    "perturbations": [{"symbol": "p", "fraction": -0.1}],
                },
            ]
        },
    )
    robustness = _typed_evidence(base, EvidenceType.ROBUSTNESS, {"summary": {}})
    store = LocalFileStore(tmp_path / "store")
    figure, _ = FigureAgent(store).generate(
        project_id=base.project_id,
        problem_id=base.problem_id,
        paper_id=uuid4(),
        paper_version=1,
        figure_id="FIG-semantic",
        title="Sensitivity of objective to p",
        caption="Verified sensitivity experiments.",
        figure_type=FigureType.SENSITIVITY_CURVE,
        data_payload={
            "x": [0.1, -0.1],
            "y": [33.0, 27.0],
            "parameter": ["p"],
            "metric": "objective",
            "experiment_ids": [
                str(experiment_id),
                str(sensitivity.structured_payload["experiments"][1]["experiment_id"]),
            ],
        },
        source_evidence_refs=[sensitivity.evidence_id],
        evidence=[sensitivity],
    )
    swapped_figure = figure.model_copy(
        update={
            "source_binding": {
                **figure.source_binding,
                "parameter": ["q"],
            }
        }
    )
    table, _ = TableAgent(store).generate(
        project_id=base.project_id,
        problem_id=base.problem_id,
        paper_id=figure.paper_id,
        paper_version=1,
        table_id="TAB-semantic",
        title="Sensitivity Results",
        caption="Purported sensitivity table.",
        columns=["Metric", "Value"],
        rows=[["failure rate", 0.4]],
        source_evidence_refs=[robustness.evidence_id],
        evidence=[robustness],
    )
    issues = AssetSemanticIntegrityValidator().validate(
        [swapped_figure], [table], [sensitivity, robustness]
    )
    assert sum(item.code == "DOCUMENT_INTEGRITY_ERROR" for item in issues) >= 2


def test_m_unresolved_critical_red_team_finding_blocks() -> None:
    base = result_evidence()
    red_team = _typed_evidence(
        base,
        EvidenceType.RED_TEAM,
        {
            "findings": [
                {
                    "finding_id": "RTF-critical",
                    "severity": "CRITICAL",
                    "resolved": False,
                }
            ]
        },
    )
    claim = Claim(
        claim_id="CLAIM-limitations",
        project_id=base.project_id,
        paper_id=uuid4(),
        paper_version=1,
        claim_type=ClaimType.CONCLUSION,
        text="The model limitations were reviewed.",
        structured_value={"finding_ids": []},
        section_id="SEC-results",
        evidence_refs=[red_team.evidence_id],
        importance=ClaimImportance.MAJOR,
        generated_by="adversarial-test",
    )
    assert "UNRESOLVED_RED_TEAM_CRITICAL" in {
        item.code for item in FactualIntegrityValidator().validate(_graph(claim, red_team))
    }


def test_n_required_subproblem_coverage_is_deterministic() -> None:
    evidence = result_evidence()
    claim = numeric_claim(evidence)
    paper = paper_ir(evidence, claim)
    requirements = [
        ProblemRequirement(subproblem_id="Q1", required_outputs=["answer"]),
        ProblemRequirement(subproblem_id="Q2", required_outputs=["recommendation"]),
    ]
    partial = paper.model_copy(
        update={
            "sections": [paper.sections[0].model_copy(update={"subproblem_refs": ["Q1"]})],
            "subproblem_coverage": [
                SubproblemCoverageRecord(
                    subproblem_id="Q1",
                    required_outputs=["answer"],
                    section_ids=[paper.sections[0].section_id],
                    claim_refs=[claim.claim_id],
                )
            ],
        }
    )
    assert "MISSING_SUBPROBLEM" in {
        item.code for item in SubproblemCoverageValidator().validate(partial, requirements)
    }


def test_o_p_hallucinated_number_exemptions_and_wrong_data_source() -> None:
    evidence = result_evidence()
    claim = numeric_claim(evidence)
    prose = PaperBlock(
        block_id="BLOCK-hallucinated",
        block_type=PaperBlockType.PARAGRAPH,
        text="The model improves performance by 17.3%.",
    )
    paper = paper_ir(evidence, claim, blocks=[prose])
    assert "UNSUPPORTED_CLAIM" in {
        item.code for item in HallucinatedNumberDetector().validate(paper, [claim])
    }
    safe_prose = prose.model_copy(
        update={"text": "Equation (3), Table 2, the year 2026, and Q1 are labels."}
    )
    safe_paper = paper_ir(evidence, claim, blocks=[safe_prose])
    assert HallucinatedNumberDetector().validate(safe_paper, [claim]) == []

    source_claim = Claim(
        claim_id="CLAIM-who",
        project_id=evidence.project_id,
        paper_id=uuid4(),
        paper_version=1,
        claim_type=ClaimType.FACTUAL,
        text="The data were collected from WHO.",
        section_id="SEC-results",
        evidence_refs=[evidence.evidence_id],
        importance=ClaimImportance.CRITICAL,
        generated_by="adversarial-test",
    )
    assert "UNSUPPORTED_CLAIM" in {
        item.code for item in FactualIntegrityValidator().validate(_graph(source_claim, evidence))
    }


def test_q_mock_critical_citation_cannot_be_fresh() -> None:
    evidence = result_evidence()
    claim = numeric_claim(evidence).model_copy(update={"citation_refs": ["REF-real"]})
    reference = reference_record(evidence.project_id).model_copy(update={"is_mock": True})
    issues = CitationFreshnessValidator().validate([claim], [reference], [], [])
    assert "MOCK_EVIDENCE" in {item.code for item in issues}


def test_r_s_latex_injection_is_rejected_and_valid_special_text_is_escaped() -> None:
    safety = LaTeXSafetyValidator()
    for attack in (
        r"\input{/etc/passwd}",
        r"\include{secret}",
        r"\write18{touch pwned}",
        r"\openout1=x",
        r"\read1 to \x",
        r"\usepackage{evil}",
    ):
        with pytest.raises(LatexRenderError):
            safety.validate_equation(attack)
    valid = r"50% reduction A&B x_y $100 {test} #"
    escaped = escape_latex(valid)
    assert all(token in escaped for token in (r"50\%", r"A\&B", r"x\_y", r"\$100", r"\#"))
    assert r"\input" not in escape_latex(r"}\input{secret}{")

    warnings, fatal = PDFCompiler._classify_warnings(
        "This is XeTeX\nCitation `REF-a' undefined\nThis is XeTeX\nOverfull \\hbox warning",
        "",
    )
    assert warnings and fatal == []
    _, final_fatal = PDFCompiler._classify_warnings("This is XeTeX\nCitation `REF-a' undefined", "")
    assert final_fatal


def test_symbol_narrative_and_x_y_result_semantics_are_not_trusted() -> None:
    evidence = result_evidence(status="TIME_LIMIT", is_optimal=False)
    claim = numeric_claim(
        evidence,
        text="Gurobi 11 obtained the exact global optimum of 30 kg from a nonlinear model.",
    )
    codes = {item.code for item in FactualIntegrityValidator().validate(_graph(claim, evidence))}
    assert {
        "SOLVER_CLAIM_MISMATCH",
        "SOLVER_VERSION_CLAIM_MISMATCH",
        "OPTIMALITY_CLAIM_MISMATCH",
    } <= codes

    robust = _typed_evidence(
        evidence,
        EvidenceType.ROBUSTNESS,
        {"summary": {"feasibility_rate": 0.6, "failed_runs": 4}},
    )
    robust_claim = Claim(
        claim_id="CLAIM-robust",
        project_id=evidence.project_id,
        paper_id=uuid4(),
        paper_version=1,
        claim_type=ClaimType.ROBUSTNESS,
        text="The model demonstrates strong robustness.",
        section_id="SEC-results",
        evidence_refs=[robust.evidence_id],
        importance=ClaimImportance.CRITICAL,
        generated_by="adversarial-test",
    )
    assert "ROBUSTNESS_CLAIM_MISMATCH" in {
        item.code for item in FactualIntegrityValidator().validate(_graph(robust_claim, robust))
    }

    model = lp_model()
    symbol = model.decision_variables[0].symbol
    paper = paper_ir(
        evidence,
        numeric_claim(evidence),
        blocks=[
            PaperBlock(
                block_id="BLOCK-symbol-drift",
                block_type=PaperBlockType.PARAGRAPH,
                text=f"{symbol} denotes capacity.",
            )
        ],
    )
    assert "SYMBOL_CONFLICT" in {
        item.code for item in SymbolNarrativeValidator().validate(paper, model)
    }


def test_v_w_undefined_citation_and_missing_registry_assets_fail() -> None:
    evidence = result_evidence()
    claim = numeric_claim(evidence).model_copy(update={"citation_refs": ["REF-missing"]})
    paper = paper_ir(evidence, claim).model_copy(update={"bibliography": ["REF-missing"]})
    with pytest.raises(LatexRenderError, match="unknown citation"):
        LaTeXRenderer().render(
            paper=paper,
            claims=[claim],
            equations=EquationRegistry.from_model(lp_model()),
            figures=FigureRegistry([]),
            tables=TableRegistry([]),
            citations=CitationRegistry([]),
        )

    missing = paper_ir(
        evidence,
        numeric_claim(evidence),
        equation_refs=["EQ-missing"],
        figure_refs=["FIG-missing"],
        table_refs=["TAB-missing"],
    )
    codes = {
        item.code
        for item in CrossReferenceValidator().validate(
            missing,
            equations=EquationRegistry.from_model(lp_model()),
            figures=FigureRegistry([]),
            tables=TableRegistry([]),
            citations=CitationRegistry([]),
        )
    }
    assert {"MISSING_EQUATION", "MISSING_FIGURE", "MISSING_TABLE"} <= codes
