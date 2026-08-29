from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from mathmodel_ai.paper.hashing import sha256_json
from mathmodel_ai.schemas.paper import (
    AbstractRole,
    Claim,
    ClaimEvidenceLink,
    ClaimEvidenceSupport,
    ClaimImportance,
    ClaimType,
    CompetitionProfile,
    EvidenceProvenance,
    EvidenceRecord,
    EvidenceSnapshot,
    EvidenceType,
    EvidenceVerificationStatus,
    NumericClaimValue,
    PaperBlock,
    PaperBlockType,
    PaperIR,
    PaperQualityStatus,
    PaperSection,
    PaperSectionType,
    ReferenceAccessStatus,
    ReferenceMetadataOrigin,
    ReferenceMetadataStatus,
    ReferenceRecord,
    ReferenceSource,
)

HASH = "a" * 64


def result_evidence(
    *,
    project_id: UUID | None = None,
    problem_id: UUID | None = None,
    objective: float = 30,
    unit: str | None = "kg",
    solver: str = "SCIPY",
    status: str = "OPTIMAL",
    is_optimal: bool = True,
    verified: bool = True,
    is_mock: bool = False,
) -> EvidenceRecord:
    project_id = project_id or uuid4()
    problem_id = problem_id or uuid4()
    payload = {
        "objective": {"value": objective, "unit": unit},
        "solver": solver,
        "status": status,
        "is_optimal": is_optimal,
        "is_feasible": True,
    }
    return EvidenceRecord(
        project_id=project_id,
        problem_id=problem_id,
        evidence_type=EvidenceType.RESULT,
        source_type="solver_result",
        source_id="result-fixture",
        source_version="1",
        content_summary="fixture result",
        structured_payload=payload,
        verified=verified,
        verification_status=(
            EvidenceVerificationStatus.VERIFIED
            if verified
            else EvidenceVerificationStatus.MOCK
            if is_mock
            else EvidenceVerificationStatus.UNVERIFIED
        ),
        provenance=EvidenceProvenance(
            source_refs=["result:fixture"],
            source_hash=sha256_json(payload),
            is_mock=is_mock,
        ),
    )


def numeric_claim(
    evidence: EvidenceRecord,
    *,
    paper_id: UUID | None = None,
    value: float = 30,
    unit: str | None = "kg",
    section_id: str = "SEC-results",
    claim_id: str = "CLAIM-objective",
    text: str | None = None,
) -> Claim:
    return Claim(
        claim_id=claim_id,
        project_id=evidence.project_id,
        paper_id=paper_id or uuid4(),
        paper_version=1,
        claim_type=ClaimType.NUMERIC,
        text=text or f"The verified objective is {value}{f' {unit}' if unit is not None else ''}.",
        structured_value=NumericClaimValue(
            value=value,
            unit=unit,
            metric_name="objective",
            source_field="objective",
        ),
        section_id=section_id,
        evidence_refs=[evidence.evidence_id],
        importance=ClaimImportance.CRITICAL,
        generated_by="fixture",
    )


def direct_link(claim: Claim, evidence: EvidenceRecord) -> ClaimEvidenceLink:
    return ClaimEvidenceLink(
        project_id=claim.project_id,
        claim_id=claim.claim_id,
        evidence_id=evidence.evidence_id,
        support=ClaimEvidenceSupport.DIRECT_SUPPORT,
        source_field="objective" if claim.claim_type is ClaimType.NUMERIC else None,
        rationale="fixture direct support",
    )


def evidence_snapshot(evidence: EvidenceRecord) -> EvidenceSnapshot:
    return EvidenceSnapshot(
        verified_model_id=uuid4(),
        verified_model_version=1,
        verified_model_digest=HASH,
        verified_result_id=uuid4(),
        validation_id=uuid4(),
        sensitivity_id=uuid4(),
        robustness_id=uuid4(),
        red_team_report_id=uuid4(),
        evidence_ids=[evidence.evidence_id],
        snapshot_hash=HASH,
    )


def paper_ir(
    evidence: EvidenceRecord,
    claim: Claim,
    *,
    blocks: list[PaperBlock] | None = None,
    equation_refs: list[str] | None = None,
    figure_refs: list[str] | None = None,
    table_refs: list[str] | None = None,
    citation_refs: list[str] | None = None,
    profile: CompetitionProfile | None = None,
) -> PaperIR:
    actual_blocks = blocks or [
        PaperBlock(
            block_id="BLOCK-result-claim",
            block_type=PaperBlockType.CLAIM,
            claim_ref=claim.claim_id,
        )
    ]
    section = PaperSection(
        section_id="SEC-results",
        title="Results",
        section_type=PaperSectionType.RESULTS,
        blocks=actual_blocks,
        claim_refs=[claim.claim_id],
        equation_refs=equation_refs or [],
        figure_refs=figure_refs or [],
        table_refs=table_refs or [],
        citation_refs=citation_refs or [],
        order=1,
    )
    return PaperIR(
        paper_id=claim.paper_id,
        version=claim.paper_version,
        title="Verified fixture paper",
        abstract=[
            PaperBlock(
                block_id="BLOCK-abstract-problem",
                block_type=PaperBlockType.PARAGRAPH,
                text="We solve the stated optimization problem.",
                abstract_role=AbstractRole.PROBLEM,
            ),
            PaperBlock(
                block_id="BLOCK-abstract-method",
                block_type=PaperBlockType.PARAGRAPH,
                text="We apply the verified mathematical model.",
                abstract_role=AbstractRole.METHOD,
            ),
            PaperBlock(
                block_id="BLOCK-abstract",
                block_type=PaperBlockType.CLAIM,
                claim_ref=claim.claim_id,
                abstract_role=AbstractRole.KEY_RESULT,
            ),
            PaperBlock(
                block_id="BLOCK-abstract-conclusion",
                block_type=PaperBlockType.PARAGRAPH,
                text="The verified solution answers the modeled requirements.",
                abstract_role=AbstractRole.CONCLUSION,
            ),
        ],
        keywords=["verification"],
        sections=[section],
        claims=[claim],
        competition_profile=profile or CompetitionProfile(),
        evidence_snapshot=evidence_snapshot(evidence),
        status=PaperQualityStatus.DRAFT,
    )


def reference_record(
    project_id: UUID, *, status: ReferenceMetadataStatus = ReferenceMetadataStatus.VERIFIED
) -> ReferenceRecord:
    metadata = {
        "title": "Linear Programming and Scientific Management",
        "authors": ["George Dantzig"],
        "year": 1963,
        "venue": "RAND",
        "doi": "10.0000/fixture",
    }
    return ReferenceRecord(
        reference_id="REF-real",
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
        metadata_status=status,
        access_status=ReferenceAccessStatus.AVAILABLE,
        raw_metadata_hash=sha256_json(metadata),
    )
