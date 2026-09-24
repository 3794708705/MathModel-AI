from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4

import pytest

from mathmodel_ai.paper.integrity import ArtifactIntegrityValidator
from mathmodel_ai.paper.repository import PaperRepository
from mathmodel_ai.paper.workflow import PaperWorkflow, PaperWorkflowError
from mathmodel_ai.schemas.paper import (
    PaperArtifactKind,
    PaperQualityReport,
    PaperQualityStatus,
    PaperValidationIssue,
    PaperValidationSeverity,
    PaperVersion,
)
from tests.paper.helpers import numeric_claim, paper_ir, reference_record, result_evidence


def test_paper_retry_reuses_immutable_verified_reference(monkeypatch):
    original = reference_record(uuid4())
    fresh = original.model_copy(update={
        "retrieved_at": datetime.now(UTC) + timedelta(seconds=5)
    })
    repository = object.__new__(PaperRepository)
    monkeypatch.setattr(repository, "list_references", lambda _project_id: [original])
    assert repository.canonicalize_references(original.project_id, [fresh]) == [original]

    changed = fresh.model_copy(update={"title": "Conflicting source title"})
    with pytest.raises(ValueError, match="conflicts with immutable reference"):
        repository.canonicalize_references(original.project_id, [changed])


def test_paper_retry_reuses_immutable_evidence_with_original_timestamp(monkeypatch):
    original = result_evidence()
    fresh = original.model_copy(update={
        "created_at": original.created_at + timedelta(seconds=5)
    })
    row = SimpleNamespace(
        id=original.evidence_id, project_id=original.project_id,
        record_json=original.model_dump(mode="json"),
    )

    @contextmanager
    def fake_session_scope(_factory):
        yield SimpleNamespace(scalars=lambda _statement: [row])

    monkeypatch.setattr("mathmodel_ai.paper.repository.session_scope", fake_session_scope)
    repository = object.__new__(PaperRepository)
    repository._session_factory = object()
    assert repository.canonicalize_evidence(original.project_id, (fresh,)) == (original,)

    changed = fresh.model_copy(update={"content_summary": "conflicting scientific content"})
    with pytest.raises(ValueError, match="conflicting content"):
        repository.canonicalize_evidence(original.project_id, (changed,))


def test_independent_factual_findings_remain_visible_in_quality_feedback() -> None:
    run = SimpleNamespace(
        output=SimpleNamespace(findings=["A comparison attributes an effect to the wrong model."])
    )
    issues = PaperWorkflow._audit_findings(run)
    assert len(issues) == 1
    assert issues[0].code == "INDEPENDENT_FACTUAL_FINDING"
    assert "wrong model" in issues[0].message


def test_deterministic_rerender_reuses_only_a_proven_pdf_extraction_false_positive(
    monkeypatch,
) -> None:
    evidence = result_evidence()
    claim = numeric_claim(evidence)
    prior_ir = paper_ir(evidence, claim).model_copy(
        update={"status": PaperQualityStatus.FAILED}
    )
    previous = PaperVersion(
        paper_id=prior_ir.paper_id,
        project_id=evidence.project_id,
        problem_id=evidence.problem_id,
        version=1,
        revision_reason="original author run",
        evidence_snapshot=prior_ir.evidence_snapshot,
        paper_ir=prior_ir,
        status=PaperQualityStatus.FAILED,
        paper_agent_run_id=uuid4(),
    )
    issue = PaperValidationIssue(
        code="DOCUMENT_INTEGRITY_ERROR",
        message="material claim is absent from final PDF",
        severity=PaperValidationSeverity.ERROR,
        object_ref=claim.claim_id,
    )
    quality = PaperQualityReport(
        paper_id=prior_ir.paper_id,
        paper_version=1,
        status=PaperQualityStatus.FAILED,
        checks={},
        issues=[issue],
        claim_count=1,
        supported_claim_count=1,
        unsupported_claim_count=0,
        reference_count=0,
        verified_reference_count=0,
        figure_count=0,
        table_count=0,
    )
    repository = SimpleNamespace(
        get_version=lambda *_args: previous,
        get_quality=lambda *_args: quality,
        list_version_artifacts=lambda *_args: [
            SimpleNamespace(kind=PaperArtifactKind.PDF, storage_key="old.pdf")
        ],
        get_compile=lambda *_args: SimpleNamespace(page_count=1),
    )
    workflow = object.__new__(PaperWorkflow)
    workflow._repository = repository
    workflow._store = SimpleNamespace(
        read_bytes=lambda _key: b"original PDF bytes",
        store_artifact=lambda *_args, **_kwargs: SimpleNamespace(storage_key="new.json"),
    )
    bundle = SimpleNamespace(
        state=SimpleNamespace(project_id=evidence.project_id),
        snapshot=prior_ir.evidence_snapshot.model_copy(update={"snapshot_id": uuid4()}),
    )
    monkeypatch.setattr(ArtifactIntegrityValidator, "_pdf_content", lambda *_args: [])
    rebased, authorship, key = workflow._reuse_previous_author(
        bundle, prior_ir.paper_id, 2, prior_ir.competition_profile
    )
    assert rebased.version == 2
    assert rebased.claims[0].paper_version == 2
    assert rebased.evidence_snapshot == bundle.snapshot
    assert previous.paper_ir.version == 1
    assert authorship.run_id == previous.paper_agent_run_id
    assert key == "new.json"

    monkeypatch.setattr(ArtifactIntegrityValidator, "_pdf_content", lambda *_args: [issue])
    with pytest.raises(PaperWorkflowError, match="still reproduces"):
        workflow._reuse_previous_author(bundle, prior_ir.paper_id, 2, prior_ir.competition_profile)

    coverage_issue = issue.model_copy(update={
        "code": "FINAL_REQUIREMENT_COVERAGE", "message": "old classifier rejected model claim"
    })
    quality.issues = [coverage_issue]
    monkeypatch.setattr(workflow, "_subproblems", lambda _state: [])
    monkeypatch.setattr(workflow, "_submission_coverage_issues", lambda *_args: [])
    rebased, _, _ = workflow._reuse_previous_author(
        bundle, prior_ir.paper_id, 2, prior_ir.competition_profile
    )
    assert rebased.version == 2
    monkeypatch.setattr(
        workflow, "_submission_coverage_issues", lambda *_args: [coverage_issue]
    )
    with pytest.raises(PaperWorkflowError, match="coverage still fails"):
        workflow._reuse_previous_author(bundle, prior_ir.paper_id, 2, prior_ir.competition_profile)
