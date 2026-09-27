from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from mathmodel_ai.paper.claims import (
    ClaimEvidenceGraph,
    ClaimEvidenceValidator,
    NumberConsistencyValidator,
    NumericClaimValidator,
    PaperNumericFormattingPolicy,
    resolve_source_field,
)
from mathmodel_ai.paper.hashing import sha256_json
from mathmodel_ai.paper.literature import (
    CitationMetadataVerifier,
    CitationSupportVerifier,
    FixtureLiteratureSource,
    reference_digest,
)
from mathmodel_ai.paper.validators import CitationQualityValidator, FactualIntegrityValidator
from mathmodel_ai.schemas.paper import (
    CitationMetadataCheck,
    CitationSupportDraft,
    CitationSupportStatus,
    Claim,
    ClaimEvidenceLink,
    ClaimEvidenceSupport,
    ClaimType,
    ClaimVerificationStatus,
    ComparisonClaimValue,
    ComparisonDirection,
    EvidenceVerificationStatus,
    LiteratureNeedType,
    LiteratureSearchNeed,
    NumericClaimValue,
    ReferenceAccessStatus,
    ReferenceMetadataOrigin,
    ReferenceMetadataStatus,
    ReferenceRecord,
    ReferenceSource,
)
from tests.paper.helpers import direct_link, numeric_claim, result_evidence


def _graph(claim: Claim, evidence, *, link: bool = True) -> ClaimEvidenceGraph:
    graph = ClaimEvidenceGraph([evidence])
    graph.add_claim(claim)
    if link:
        graph.add_link(direct_link(claim, evidence))
    return graph


def _reference(project_id, *, reference_id: str = "REF-real") -> ReferenceRecord:
    metadata = {
        "title": "Linear Programming and Scientific Management",
        "authors": ["George Dantzig"],
        "year": 1963,
        "venue": "RAND",
        "doi": "10.0000/fixture",
    }
    return ReferenceRecord(
        reference_id=reference_id,
        project_id=project_id,
        title=metadata["title"],
        authors=metadata["authors"],
        year=metadata["year"],
        venue=metadata["venue"],
        doi=metadata["doi"],
        url="https://example.invalid/fixture",
        abstract="Linear programming optimizes a linear objective under linear constraints.",
        trusted_excerpt="Linear programming optimizes a linear objective under linear constraints.",
        source=ReferenceSource.FIXTURE,
        source_id="fixture:lp",
        metadata_origin=ReferenceMetadataOrigin.RETRIEVED,
        retrieved_at=datetime.now(UTC),
        metadata_status=ReferenceMetadataStatus.PENDING,
        access_status=ReferenceAccessStatus.AVAILABLE,
        raw_metadata_hash=sha256_json(metadata),
    )


def test_a_numeric_claim_exact_value_passes_and_tamper_fails() -> None:
    evidence = result_evidence()
    claim = numeric_claim(evidence)
    assert (
        NumericClaimValidator().validate(
            claim, _graph(claim, evidence).evidence_for(claim.claim_id)
        )
        == []
    )

    tampered = claim.model_copy(
        update={
            "structured_value": claim.structured_value.model_copy(update={"value": 31})  # type: ignore[union-attr]
        }
    )
    issues = NumericClaimValidator().validate(
        tampered, _graph(tampered, evidence).evidence_for(tampered.claim_id)
    )
    assert {item.code for item in issues} == {"NUMERIC_CLAIM_MISMATCH"}


def test_b_unsupported_critical_claim_fails() -> None:
    evidence = result_evidence()
    claim = numeric_claim(evidence).model_copy(update={"evidence_refs": []})
    graph = ClaimEvidenceGraph([evidence])
    graph.add_claim(claim)
    assert "UNSUPPORTED_CLAIM" in {item.code for item in ClaimEvidenceValidator().validate(graph)}


def test_c_cross_section_numeric_conflict_fails() -> None:
    evidence = result_evidence()
    first = numeric_claim(evidence, claim_id="CLAIM-abstract", value=12.4)
    conflicting = numeric_claim(
        evidence,
        claim_id="CLAIM-conclusion",
        section_id="SEC-conclusion",
        value=15,
    )
    assert "CROSS_SECTION_VALUE_CONFLICT" in {
        item.code for item in NumberConsistencyValidator().validate([first, conflicting])
    }


def test_d_e_solver_name_claim_detects_scipy_gurobi_and_optimality_mismatch() -> None:
    evidence = result_evidence(status="TIME_LIMIT", is_optimal=False)
    claim = numeric_claim(
        evidence,
        text="Gurobi obtained the globally optimal solution with objective 30 kg.",
    )
    codes = {item.code for item in FactualIntegrityValidator().validate(_graph(claim, evidence))}
    assert codes == {"SOLVER_CLAIM_MISMATCH", "OPTIMALITY_CLAIM_MISMATCH"}


@pytest.mark.asyncio
async def test_f_g_fake_and_real_citation_metadata() -> None:
    project_id = uuid4()
    reference = _reference(project_id)
    source = FixtureLiteratureSource((reference,))
    real = await CitationMetadataVerifier().verify(reference, source)
    assert real.status is ReferenceMetadataStatus.VERIFIED

    fake = reference.model_copy(
        update={"reference_id": "REF-fake", "doi": "10.9999/does-not-exist"}
    )
    missing = await CitationMetadataVerifier().verify(fake, source)
    assert missing.status is ReferenceMetadataStatus.NOT_FOUND


@pytest.mark.asyncio
async def test_fixture_literature_search_resolution_and_metadata_conflict() -> None:
    project_id = uuid4()
    reference = _reference(project_id)
    source = FixtureLiteratureSource((reference,))
    matches = await source.search(
        LiteratureSearchNeed(
            need_id="LITNEED-fixture",
            need_type=LiteratureNeedType.THEORY,
            query="linear programming",
            purpose="method provenance",
        ),
        project_id,
    )
    assert matches == [reference]
    assert await source.resolve(f"https://doi.org/{reference.doi}", project_id) == reference
    assert await source.resolve("10.9999/missing", project_id) is None

    conflicting = reference.model_copy(update={"title": "Conflicting title"})
    check = await CitationMetadataVerifier().verify(conflicting, source)
    assert check.status is ReferenceMetadataStatus.CONFLICT
    assert check.field_matches["title"] is False
    assert check.errors == ["metadata conflict"]
    assert (
        CitationMetadataVerifier.verified_copy(conflicting, check).metadata_status
        is ReferenceMetadataStatus.CONFLICT
    )
    assert len(CitationMetadataVerifier.canonical_metadata_hash(reference)) == 64


def test_h_existing_reference_that_does_not_support_claim_is_rejected() -> None:
    evidence = result_evidence()
    claim = numeric_claim(evidence).model_copy(update={"citation_refs": ["REF-real"]})
    reference = _reference(evidence.project_id).model_copy(
        update={"metadata_status": ReferenceMetadataStatus.VERIFIED}
    )
    check = CitationSupportVerifier().verify(
        claim=claim,
        reference=reference,
        draft=CitationSupportDraft(
            status=CitationSupportStatus.SUPPORTED,
            supporting_excerpt="This text is not in the trusted source.",
            rationale="fixture",
        ),
        reviewer_is_mock=False,
    )
    assert check.status is CitationSupportStatus.NOT_SUPPORTED

    metadata_check = CitationMetadataCheck(
        project_id=evidence.project_id,
        reference_id=reference.reference_id,
        reference_digest=reference_digest(reference),
        status=ReferenceMetadataStatus.VERIFIED,
        field_matches={"title": True},
    )
    issues = CitationQualityValidator().validate([claim], [metadata_check], [check])
    assert [item.code for item in issues] == ["REFERENCE_NOT_SUPPORTED"]


def test_citation_support_fails_closed_for_unverified_missing_text_and_mock_review() -> None:
    evidence = result_evidence()
    reference = _reference(evidence.project_id)
    verifier = CitationSupportVerifier()
    supported_draft = CitationSupportDraft(
        status=CitationSupportStatus.SUPPORTED,
        supporting_excerpt=reference.trusted_excerpt,
        rationale="fixture",
    )
    unverified = verifier.verify(
        claim=numeric_claim(evidence, claim_id="CLAIM-unverified"),
        reference=reference,
        draft=supported_draft,
        reviewer_is_mock=False,
    )
    assert unverified.status is CitationSupportStatus.NOT_SUPPORTED

    no_text = reference.model_copy(
        update={
            "metadata_status": ReferenceMetadataStatus.VERIFIED,
            "trusted_excerpt": None,
            "abstract": None,
        }
    )
    insufficient = verifier.verify(
        claim=numeric_claim(evidence, claim_id="CLAIM-no-text"),
        reference=no_text,
        draft=supported_draft,
        reviewer_is_mock=False,
    )
    assert insufficient.status is CitationSupportStatus.INSUFFICIENT_TEXT
    assert insufficient.trusted_text_hash is None

    verified = reference.model_copy(update={"metadata_status": ReferenceMetadataStatus.VERIFIED})
    mocked = verifier.verify(
        claim=numeric_claim(evidence, claim_id="CLAIM-mock"),
        reference=verified,
        draft=supported_draft,
        reviewer_is_mock=True,
    )
    assert mocked.status is CitationSupportStatus.INSUFFICIENT_TEXT
    assert mocked.reviewer_is_mock


def test_r_mock_evidence_can_never_support_formal_claim() -> None:
    evidence = result_evidence(verified=False, is_mock=True)
    claim = numeric_claim(evidence)
    codes = {item.code for item in ClaimEvidenceValidator().validate(_graph(claim, evidence))}
    assert {"MOCK_EVIDENCE", "UNSUPPORTED_CLAIM"} <= codes


def test_t_unit_conflict_fails() -> None:
    evidence = result_evidence(unit="kg")
    claim = numeric_claim(evidence, unit="t")
    codes = {
        item.code
        for item in NumericClaimValidator().validate(
            claim, _graph(claim, evidence).evidence_for(claim.claim_id)
        )
    }
    assert "UNIT_CLAIM_MISMATCH" in codes


def test_claim_graph_rejects_duplicate_and_broken_links() -> None:
    evidence = result_evidence()
    claim = numeric_claim(evidence)
    with pytest.raises(ValueError, match="duplicate evidence"):
        ClaimEvidenceGraph([evidence, evidence])
    graph = ClaimEvidenceGraph([evidence])
    graph.add_claim(claim)
    with pytest.raises(ValueError, match="duplicate claim"):
        graph.add_claim(claim)
    unknown_evidence_claim = claim.model_copy(
        update={"claim_id": "CLAIM-unknown-evidence", "evidence_refs": [uuid4()]}
    )
    with pytest.raises(ValueError, match="unknown evidence"):
        graph.add_claim(unknown_evidence_claim)
    unknown_claim_link = direct_link(claim, evidence).model_copy(
        update={"claim_id": "CLAIM-not-registered"}
    )
    with pytest.raises(ValueError, match="unknown claim"):
        graph.add_link(unknown_claim_link)
    extra_evidence = result_evidence(
        project_id=evidence.project_id,
        problem_id=evidence.problem_id,
    )
    graph_with_extra = ClaimEvidenceGraph([evidence, extra_evidence])
    graph_with_extra.add_claim(claim)
    with pytest.raises(ValueError, match="declared on its claim"):
        graph_with_extra.add_link(
            ClaimEvidenceLink(
                project_id=claim.project_id,
                claim_id=claim.claim_id,
                evidence_id=extra_evidence.evidence_id,
                support=ClaimEvidenceSupport.DIRECT_SUPPORT,
                rationale="invalid undeclared evidence",
            )
        )
    valid = direct_link(claim, evidence)
    graph.add_link(valid)
    with pytest.raises(ValueError, match="duplicate claim/evidence link"):
        graph.add_link(valid)


def test_source_field_and_numeric_payload_validation_are_strict() -> None:
    assert resolve_source_field({"nested": {"value": 3}}, "nested.value") == 3
    with pytest.raises(ValueError, match="invalid"):
        resolve_source_field({"value": 3}, "_private")
    with pytest.raises(KeyError):
        resolve_source_field({"value": 3}, "missing")

    evidence = result_evidence()
    scalar_evidence = evidence.model_copy(update={"structured_payload": {"objective": 30}})
    scalar_claim = numeric_claim(evidence, unit=None)
    scalar_link = direct_link(scalar_claim, scalar_evidence)
    assert NumericClaimValidator().validate(scalar_claim, [(scalar_link, scalar_evidence)]) == []
    boolean_evidence = evidence.model_copy(update={"structured_payload": {"objective": True}})
    issues = NumericClaimValidator().validate(scalar_claim, [(scalar_link, boolean_evidence)])
    assert [item.code for item in issues] == ["NUMERIC_CLAIM_MISMATCH"]
    wrong_field_link = scalar_link.model_copy(update={"source_field": "other"})
    issues = NumericClaimValidator().validate(scalar_claim, [(wrong_field_link, scalar_evidence)])
    assert [item.code for item in issues] == ["NUMERIC_CLAIM_MISMATCH"]


def test_comparison_claim_checks_both_deterministic_evidence_fields() -> None:
    evidence = result_evidence().model_copy(
        update={
            "structured_payload": {
                "baseline": {"value": 40, "unit": "kg"},
                "verified": {"value": 30, "unit": "kg"},
            }
        }
    )
    claim = numeric_claim(evidence).model_copy(
        update={
            "claim_id": "CLAIM-comparison",
            "claim_type": ClaimType.COMPARISON,
            "text": "The verified value decreased by 25% from the baseline.",
            "structured_value": ComparisonClaimValue(
                baseline_value=40,
                verified_value=30,
                percentage_change=-25,
                direction=ComparisonDirection.DECREASE,
                unit="kg",
                baseline_source_field="baseline",
                verified_source_field="verified",
            ),
        }
    )
    graph = ClaimEvidenceGraph([evidence])
    graph.add_claim(claim)
    for source_field in ("baseline", "verified"):
        graph.add_link(
            ClaimEvidenceLink(
                project_id=claim.project_id,
                claim_id=claim.claim_id,
                evidence_id=evidence.evidence_id,
                support=ClaimEvidenceSupport.DIRECT_SUPPORT,
                source_field=source_field,
                rationale="deterministic comparison operand",
            )
        )
    assert NumericClaimValidator().validate(claim, graph.evidence_for(claim.claim_id)) == []


def test_comparison_claim_parses_typed_value_from_provider_json() -> None:
    raw = numeric_claim(result_evidence()).model_dump(mode="json")
    raw["claim_type"] = "COMPARISON"
    raw["structured_value"] = {
        "baseline_value": 40,
        "verified_value": 30,
        "percentage_change": -25,
        "direction": "DECREASE",
        "baseline_source_field": "baseline",
        "verified_source_field": "verified",
    }
    parsed = Claim.model_validate(raw)
    assert isinstance(parsed.structured_value, ComparisonClaimValue)
    raw["structured_value"]["percentage_change"] = -20
    with pytest.raises(ValidationError, match="deterministically calculated"):
        Claim.model_validate(raw)


def test_numeric_text_accepts_proven_synonyms_and_scientific_notation() -> None:
    evidence = result_evidence()
    base = numeric_claim(evidence)
    comparison = ComparisonClaimValue(
        baseline_value=0.2897727272263358,
        verified_value=0.4673295454545455,
        percentage_change=61.27450982974101,
        direction=ComparisonDirection.INCREASE,
        baseline_source_field="reviewed_metric_values.fixed_high_resource_J_final",
        verified_source_field="reviewed_metric_values.adaptive_high_resource_J_final",
    )
    claim = base.model_copy(
        update={
            "claim_type": ClaimType.COMPARISON,
            "text": "Adaptive abundance exceeds the comparator by 61.3%.",
            "structured_value": comparison,
        }
    )
    policy = PaperNumericFormattingPolicy()
    assert policy.text_matches(claim)
    assert not policy.text_matches(
        claim.model_copy(update={"text": "Adaptive abundance falls below the comparator by 61.3%."})
    )

    numeric = base.model_copy(
        update={
            "text": "The residual is 6.757644837709665e-16.",
            "structured_value": NumericClaimValue(
                value=6.757644837709665e-16,
                metric_name="equilibrium residual squared",
                source_field="reviewed_metric_values.equilibrium_residual_sq",
            ),
        }
    )
    assert policy.text_matches(numeric)
    assert not policy.text_matches(
        numeric.model_copy(update={"text": "The residual is 9.757644837709665e-16."})
    )


def test_unverified_evidence_status_and_claim_status_are_deterministic() -> None:
    evidence = result_evidence().model_copy(
        update={
            "verified": False,
            "verification_status": EvidenceVerificationStatus.UNVERIFIED,
        }
    )
    claim = numeric_claim(evidence)
    graph = _graph(claim, evidence)
    issues = ClaimEvidenceValidator().validate(graph)
    assert {item.code for item in issues} >= {"UNSUPPORTED_CLAIM", "UNVERIFIED_EVIDENCE"}
    updated = ClaimEvidenceValidator().apply_statuses(graph)
    assert updated[0].verification_status is ClaimVerificationStatus.UNSUPPORTED


def test_number_consistency_detects_unit_only_conflict() -> None:
    evidence = result_evidence()
    kilograms = numeric_claim(evidence, claim_id="CLAIM-kg", unit="kg")
    tonnes = numeric_claim(evidence, claim_id="CLAIM-t", unit="t")
    issues = NumberConsistencyValidator().validate([kilograms, tonnes])
    assert "UNIT_CLAIM_MISMATCH" in {item.code for item in issues}
