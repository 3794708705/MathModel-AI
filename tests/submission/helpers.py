from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

from pypdf import PdfWriter

from mathmodel_ai.files.storage import LocalFileStore
from mathmodel_ai.schemas.paper import (
    ClaimVerificationStatus,
    PaperQualityStatus,
    SubproblemCoverageRecord,
)
from mathmodel_ai.schemas.problem_analysis import ProblemTaskType, SubProblem
from mathmodel_ai.schemas.submission import (
    CompetitionProfile,
    FinalJuryDraft,
    JuryDimensionScore,
    SubmissionArtifact,
    SubmissionArtifactRole,
    SubmissionCandidate,
)
from mathmodel_ai.submission.integrity import validate_competition_profile
from mathmodel_ai.submission.profiles import generic_modeling_test_profile
from tests.paper.helpers import numeric_claim, paper_ir, result_evidence

HASH = "a" * 64


def pdf_bytes(pages: int = 1) -> bytes:
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=612, height=792)
    output = bytearray()
    from io import BytesIO

    stream = BytesIO()
    writer.write(stream)
    output.extend(stream.getvalue())
    return bytes(output)


def stored_artifact(
    store: LocalFileStore,
    *,
    project_id: UUID,
    role: SubmissionArtifactRole,
    relative_path: str,
    data: bytes,
    mime_type: str,
    paper_id: UUID | None = None,
    paper_version: int = 1,
) -> SubmissionArtifact:
    artifact_id = uuid4()
    stored = store.store_artifact(
        data,
        project_id=project_id,
        artifact_id=artifact_id,
        filename=Path(relative_path).name,
    )
    return SubmissionArtifact(
        artifact_id=artifact_id,
        project_id=project_id,
        role=role,
        relative_path=relative_path,
        mime_type=mime_type,
        size_bytes=stored.size_bytes,
        sha256=stored.sha256,
        storage_key=stored.storage_key,
        source_artifact_id=artifact_id,
        paper_id=paper_id,
        paper_version=paper_version if paper_id is not None else None,
    )


def candidate_fixture(
    tmp_path: Path,
    *,
    profile: CompetitionProfile | None = None,
    pages: int = 1,
    paper_text: str = "Anonymous verified results.",
    extra_files: list[tuple[SubmissionArtifactRole, str, bytes, str]] | None = None,
) -> tuple[LocalFileStore, CompetitionProfile, SubmissionCandidate, dict[UUID, bytes]]:
    store = LocalFileStore(tmp_path / "objects")
    actual_profile = profile or generic_modeling_test_profile()
    project_id = uuid4()
    paper_id = uuid4()
    artifacts = [
        stored_artifact(
            store,
            project_id=project_id,
            role=SubmissionArtifactRole.PAPER_PDF,
            relative_path="paper.pdf",
            data=pdf_bytes(pages),
            mime_type="application/pdf",
            paper_id=paper_id,
        )
    ]
    for role, path, data, mime_type in extra_files or []:
        artifacts.append(
            stored_artifact(
                store,
                project_id=project_id,
                role=role,
                relative_path=path,
                data=data,
                mime_type=mime_type,
                paper_id=paper_id,
            )
        )
    candidate = SubmissionCandidate(
        project_id=project_id,
        problem_id=uuid4(),
        competition_profile_id=actual_profile.profile_id,
        competition_profile_version=actual_profile.version,
        competition_profile_digest=validate_competition_profile(actual_profile),
        paper_id=paper_id,
        paper_version=1,
        paper_status=PaperQualityStatus.READY_FOR_FINAL_JURY,
        paper_manifest_hash=HASH,
        paper_ir_hash=HASH,
        verified_model_id=uuid4(),
        verified_model_version=1,
        verified_model_digest=HASH,
        verified_result_id=uuid4(),
        phase5_verified=True,
        page_count=pages,
        section_types=["RESULTS"],
        paper_text=paper_text,
        artifacts=artifacts,
    )
    payloads = {item.artifact_id: store.read_bytes(item.storage_key) for item in artifacts}
    return store, actual_profile, candidate, payloads


def jury_draft(*, innovation: float = 5, critical: bool = False) -> FinalJuryDraft:
    profile = generic_modeling_test_profile()
    dimensions = [
        JuryDimensionScore(
            dimension=name,
            score=(innovation if name == "innovation" else maximum),
            maximum=maximum,
            rationale="deterministic acceptance fixture",
        )
        for name, maximum in profile.jury_weights.items()
    ]
    findings = []
    if critical:
        from mathmodel_ai.schemas.submission import JuryFinding, JurySeverity

        findings.append(
            JuryFinding(
                finding_id="JURY-agent-critical",
                category="model",
                severity=JurySeverity.CRITICAL,
                title="Critical defect",
                description="A critical unresolved defect remains.",
                impact="The result cannot be trusted.",
                recommendation="Repair and rerun.",
                confidence=1,
            )
        )
    return FinalJuryDraft(
        dimensions=dimensions,
        findings=findings,
        summary="Structured final review fixture.",
    )


def subproblem(subproblem_id: str, output: str) -> SubProblem:
    return SubProblem(
        subproblem_id=subproblem_id,
        order=int(subproblem_id[1:]),
        original_text=f"Answer {subproblem_id}.",
        normalized_goal=f"Produce {output}.",
        output_required=[output],
        task_types=[ProblemTaskType.EVALUATION],
    )


def covered_paper(*pairs: tuple[str, str]):
    evidence = result_evidence()
    claim = numeric_claim(evidence).model_copy(
        update={"verification_status": ClaimVerificationStatus.SUPPORTED}
    )
    paper = paper_ir(evidence, claim)
    records = [
        SubproblemCoverageRecord(
            subproblem_id=subproblem_id,
            required_outputs=[output],
            section_ids=["SEC-results"],
            claim_refs=[claim.claim_id],
            output_claim_refs={output: [claim.claim_id]},
        )
        for subproblem_id, output in pairs
    ]
    section = paper.sections[0].model_copy(
        update={"subproblem_refs": [subproblem_id for subproblem_id, _ in pairs]}
    )
    return paper.model_copy(update={"sections": [section], "subproblem_coverage": records})
