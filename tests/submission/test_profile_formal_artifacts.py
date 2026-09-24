from mathmodel_ai.benchmark.profiles import comap_mcm_2024_profile
from mathmodel_ai.schemas.submission import SubmissionArtifactRole
from mathmodel_ai.submission.workflow import FinalSubmissionWorkflow
from tests.submission.helpers import candidate_fixture


def test_one_pdf_contest_excludes_internal_paper_sources(tmp_path) -> None:
    _store, _profile, candidate, _payloads = candidate_fixture(
        tmp_path,
        extra_files=[
            (SubmissionArtifactRole.PAPER_TEX, "paper.tex", b"draft", "text/plain"),
            (SubmissionArtifactRole.TABLE, "tables/result.json", b"{}", "application/json"),
        ],
    )
    formal = FinalSubmissionWorkflow._formal_artifacts(
        candidate.artifacts, comap_mcm_2024_profile()
    )
    assert [item.relative_path for item in formal] == ["paper.pdf"]
    assert len(FinalSubmissionWorkflow._formal_artifacts(candidate.artifacts, _profile)) == 3
