from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from mathmodel_ai.agents import AgentRunResult, AgentRunStatus
from mathmodel_ai.core.config import Settings
from mathmodel_ai.db.base import Base
from mathmodel_ai.db.models import (
    ClaimEvidenceLinkRecord,
    ClaimRecord,
    DocumentRegistryRecord,
    EvidenceRecordModel,
    FigureRecordModel,
    FinalJuryReportRecord,
    LiteratureSearchRecord,
    PaperArtifactRecord,
    PaperSectionRecord,
    PaperVersionRecord,
    ReferenceRecordModel,
    RequirementCoverageRecordModel,
    SubmissionArtifactRecordModel,
    SubmissionCheckRecord,
    SubmissionManifestRecord,
    SubmissionSnapshotRecord,
    TableRecordModel,
)
from mathmodel_ai.main import create_app
from mathmodel_ai.mathematical.algorithms import AlgorithmSelector
from mathmodel_ai.paper.assets import FigureAgent, TableAgent
from mathmodel_ai.paper.bundle import PaperBundleBuilder
from mathmodel_ai.paper.compiler import PDFCompiler
from mathmodel_ai.paper.evidence import VerifiedEvidenceBuilder
from mathmodel_ai.paper.literature import FixtureLiteratureSource
from mathmodel_ai.paper.rendering import LaTeXRenderer
from mathmodel_ai.paper.workflow import PaperWorkflow
from mathmodel_ai.providers.factory import ProviderRegistry
from mathmodel_ai.providers.mock import MockProvider
from mathmodel_ai.schemas.mathematical import MathematicalModelDraft
from mathmodel_ai.schemas.paper import (
    AbstractRole,
    CitationAgentInput,
    CitationSupportDraft,
    CitationSupportStatus,
    Claim,
    ClaimImportance,
    ClaimType,
    EvidenceType,
    LiteratureAgentInput,
    LiteratureNeedType,
    LiteraturePlan,
    LiteratureSearchNeed,
    NumericClaimValue,
    PaperAgentInput,
    PaperArtifactKind,
    PaperBlock,
    PaperBlockType,
    PaperCompileStatus,
    PaperFactualAuditDraft,
    PaperFactualAuditInput,
    PaperIR,
    PaperQualityStatus,
    PaperSection,
    PaperSectionType,
    SubproblemCoverageRecord,
)
from mathmodel_ai.schemas.submission import (
    FinalJuryDraft,
    FinalJuryInput,
    JuryDimensionScore,
    SubmissionArtifactRole,
)
from mathmodel_ai.schemas.verification import RedTeamDraft, RedTeamInput
from mathmodel_ai.submission.package import SubmissionPackageBuilder
from mathmodel_ai.submission.rules import RuleEngine
from mathmodel_ai.submission.workflow import FinalSubmissionWorkflow
from mathmodel_ai.verification.experiment_integrity import ExperimentIntegrityVerifier
from mathmodel_ai.verification.red_team import RedTeamAnalyzer
from mathmodel_ai.verification.robustness import RobustnessAnalyzer
from mathmodel_ai.verification.sensitivity import SensitivityAnalyzer
from mathmodel_ai.verification.validation import IndependentValidator
from mathmodel_ai.verification.workflow import VerificationWorkflow
from tests.paper.helpers import reference_record
from tests.reasoning.helpers import analysis_fixture, exploration_fixture, jury_fixture
from tests.verification.helpers import phase5_model

SOLVER_IMAGE = "mathmodel-ai-solver:phase4"
PAPER_IMAGE = "mathmodel-ai-paper:phase6"


def _images_available() -> bool:
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", SOLVER_IMAGE, PAPER_IMAGE],
            capture_output=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


pytestmark = [
    pytest.mark.integration,
    pytest.mark.solver,
    pytest.mark.skipif(
        not _images_available(),
        reason=f"Docker images {SOLVER_IMAGE} and {PAPER_IMAGE} are required",
    ),
]


def _agent_result(name: str, output, state, *, is_mock: bool) -> AgentRunResult:
    now = datetime.now(UTC)
    return AgentRunResult(
        agent_name=name,
        status=AgentRunStatus.SUCCEEDED,
        output=output,
        attempts=1,
        is_mock=is_mock,
        input_state_version=state.version,
        provider="fixture",
        model="fixture-reviewer",
        reasoning="high",
        prompt_version="test-6.0.0",
        latency_ms=1,
        started_at=now,
        ended_at=now,
    )


class _PassingRedTeamAgent:
    async def run(self, input_data: RedTeamInput, state, profile):
        del input_data, profile
        return _agent_result(
            "red_team_agent",
            RedTeamDraft(summary="Independent fixture review found no critical issue."),
            state,
            is_mock=False,
        )


class _UnusedRepairAgent:
    async def run(self, input_data, state, profile):
        del input_data, state, profile
        raise AssertionError("repair is not expected for a passing fixture review")


class _LiteratureAgent:
    async def run(self, input_data: LiteratureAgentInput, state, profile):
        del input_data, profile
        return _agent_result(
            "literature_agent",
            LiteraturePlan(
                needs=[
                    LiteratureSearchNeed(
                        need_id="LITNEED-lp",
                        need_type=LiteratureNeedType.THEORY,
                        query="linear programming",
                        purpose="support the linear programming method",
                        target_claim_types=[ClaimType.LITERATURE],
                    )
                ],
                selection_strategy="Use independently resolvable method metadata.",
            ),
            state,
            is_mock=True,
        )


class _CitationAgent:
    async def run(self, input_data: CitationAgentInput, state, profile):
        del profile
        trusted = input_data.reference.trusted_excerpt
        assert trusted is not None
        return _agent_result(
            "citation_agent",
            CitationSupportDraft(
                status=CitationSupportStatus.SUPPORTED,
                supporting_excerpt=trusted,
                rationale="The trusted source text directly states the method property.",
            ),
            state,
            is_mock=False,
        )


class _PaperAgent:
    async def run(self, input_data: PaperAgentInput, state, profile):
        del profile
        result_evidence = next(
            item for item in input_data.evidence if item.evidence_type is EvidenceType.RESULT
        )
        literature_evidence = next(
            item for item in input_data.evidence if item.evidence_type is EvidenceType.LITERATURE
        )
        objective = result_evidence.structured_payload["objective"]
        result_claim = Claim(
            claim_id="CLAIM-result",
            project_id=input_data.project_id,
            paper_id=input_data.assigned_paper_id,
            paper_version=input_data.assigned_version,
            claim_type=ClaimType.NUMERIC,
            text=(
                f"The verified objective value is {objective['value']}"
                + (f" {objective['unit']}" if objective["unit"] else "")
                + "."
            ),
            structured_value=NumericClaimValue(
                value=objective["value"],
                unit=objective["unit"],
                metric_name="objective",
                source_field="objective",
            ),
            section_id="SEC-results",
            evidence_refs=[result_evidence.evidence_id],
            importance=ClaimImportance.CRITICAL,
            generated_by="fixture-paper-agent",
        )
        reference_id = input_data.reference_ids[0]
        literature_claim = Claim(
            claim_id="CLAIM-literature",
            project_id=input_data.project_id,
            paper_id=input_data.assigned_paper_id,
            paper_version=input_data.assigned_version,
            claim_type=ClaimType.LITERATURE,
            text="Linear programming optimizes a linear objective under linear constraints.",
            structured_value={"method": "linear programming"},
            section_id="SEC-results",
            evidence_refs=[literature_evidence.evidence_id],
            citation_refs=[reference_id],
            importance=ClaimImportance.MAJOR,
            generated_by="fixture-paper-agent",
        )
        blocks = [
            PaperBlock(
                block_id="BLOCK-result",
                block_type=PaperBlockType.CLAIM,
                claim_ref=result_claim.claim_id,
            ),
            PaperBlock(
                block_id="BLOCK-literature",
                block_type=PaperBlockType.CLAIM,
                claim_ref=literature_claim.claim_id,
            ),
            PaperBlock(
                block_id="BLOCK-equation",
                block_type=PaperBlockType.EQUATION,
                equation_ref=input_data.equation_ids[0],
            ),
            PaperBlock(
                block_id="BLOCK-figure",
                block_type=PaperBlockType.FIGURE,
                figure_ref=input_data.figure_ids[0],
            ),
            *[
                PaperBlock(
                    block_id=f"BLOCK-table-{index}",
                    block_type=PaperBlockType.TABLE,
                    table_ref=table_id,
                )
                for index, table_id in enumerate(input_data.table_ids, start=1)
            ],
        ]
        paper = PaperIR(
            paper_id=input_data.assigned_paper_id,
            version=input_data.assigned_version,
            title=input_data.title,
            abstract=[
                PaperBlock(
                    block_id="BLOCK-abstract-problem",
                    block_type=PaperBlockType.PARAGRAPH,
                    text="We solve the stated resource-allocation problem.",
                    abstract_role=AbstractRole.PROBLEM,
                ),
                PaperBlock(
                    block_id="BLOCK-abstract-method",
                    block_type=PaperBlockType.CLAIM,
                    claim_ref=literature_claim.claim_id,
                    abstract_role=AbstractRole.METHOD,
                ),
                PaperBlock(
                    block_id="BLOCK-abstract-result",
                    block_type=PaperBlockType.CLAIM,
                    claim_ref=result_claim.claim_id,
                    abstract_role=AbstractRole.KEY_RESULT,
                ),
                PaperBlock(
                    block_id="BLOCK-abstract-conclusion",
                    block_type=PaperBlockType.PARAGRAPH,
                    text="The verified solution satisfies the modeled requirements.",
                    abstract_role=AbstractRole.CONCLUSION,
                ),
            ],
            keywords=["linear programming", "verification"],
            sections=[
                PaperSection(
                    section_id="SEC-results",
                    title="Verified model and results",
                    section_type=PaperSectionType.RESULTS,
                    blocks=blocks,
                    claim_refs=[result_claim.claim_id, literature_claim.claim_id],
                    equation_refs=[input_data.equation_ids[0]],
                    figure_refs=input_data.figure_ids,
                    table_refs=input_data.table_ids,
                    citation_refs=[reference_id],
                    subproblem_refs=[
                        item.subproblem_id for item in input_data.required_subproblems
                    ],
                    order=1,
                )
            ],
            bibliography=[reference_id],
            claims=[result_claim, literature_claim],
            competition_profile=input_data.competition_profile,
            subproblem_coverage=[
                SubproblemCoverageRecord(
                    subproblem_id=item.subproblem_id,
                    required_outputs=item.required_outputs,
                    section_ids=["SEC-results"],
                    claim_refs=[result_claim.claim_id],
                    output_claim_refs={
                        output: [result_claim.claim_id] for output in item.required_outputs
                    },
                )
                for item in input_data.required_subproblems
            ],
            evidence_snapshot=input_data.evidence_snapshot,
            status=PaperQualityStatus.DRAFT,
        )
        return _agent_result("paper_agent", paper, state, is_mock=True)


class _AuditAgent:
    async def run(self, input_data: PaperFactualAuditInput, state, profile):
        del profile
        return _agent_result(
            "paper_factual_audit_agent",
            PaperFactualAuditDraft(
                passed=True,
                reviewed_claim_ids=[item.claim_id for item in input_data.claims],
            ),
            state,
            is_mock=False,
        )


class _FinalJuryAgent:
    async def run(self, input_data: FinalJuryInput, state, profile):
        del profile
        return _agent_result(
            "final_jury_agent",
            FinalJuryDraft(
                dimensions=[
                    JuryDimensionScore(
                        dimension=name,
                        score=maximum,
                        maximum=maximum,
                        rationale="All deterministic Phase 5-7 fixture gates passed.",
                    )
                    for name, maximum in input_data.profile.jury_weights.items()
                ],
                summary="Mock structured jury fixture; deterministic gates remain real.",
            ),
            state,
            is_mock=True,
        )


def _paper_workflow(app, project_id: UUID) -> PaperWorkflow:
    reference = reference_record(project_id)
    store = app.state.file_store
    return PaperWorkflow(
        repository=app.state.paper_repository,
        evidence_builder=VerifiedEvidenceBuilder(
            mathematical_repository=app.state.mathematical_repository,
            verification_repository=app.state.verification_repository,
        ),
        literature_agent=_LiteratureAgent(),  # type: ignore[arg-type]
        literature_source=FixtureLiteratureSource((reference,)),
        citation_agent=_CitationAgent(),  # type: ignore[arg-type]
        paper_agent=_PaperAgent(),  # type: ignore[arg-type]
        audit_agent=_AuditAgent(),  # type: ignore[arg-type]
        figure_agent=FigureAgent(store),
        table_agent=TableAgent(store),
        renderer=LaTeXRenderer(),
        compiler=PDFCompiler(store=store, image=PAPER_IMAGE),
        bundle_builder=PaperBundleBuilder(store),
        store=store,
    )


def test_verified_phase5_to_real_pdf_and_frozen_submission_e2e(tmp_path: Path) -> None:
    template = phase5_model()
    draft = MathematicalModelDraft.model_validate(
        template.model_dump(
            exclude={
                "model_id",
                "project_id",
                "problem_id",
                "version",
                "source_selected_model_id",
                "status",
            }
        )
    )
    app = create_app(
        Settings(
            environment="test",
            database_url="sqlite+pysqlite:///:memory:",
            default_provider="mock",
            reasoning_max_retries=0,
            storage_root=tmp_path / "storage",
            sandbox_root=tmp_path / "sandbox",
            solver_sandbox_root=tmp_path / "solver-sandbox",
            solver_sandbox_image=SOLVER_IMAGE,
            paper_compiler_image=PAPER_IMAGE,
            sandbox_memory_mb=768,
            sandbox_timeout_seconds=20,
            sandbox_max_artifact_bytes=1_048_576,
        ),
        providers=ProviderRegistry(
            [
                MockProvider(
                    [
                        analysis_fixture().model_dump_json(),
                        exploration_fixture().model_dump_json(),
                        jury_fixture().model_dump_json(),
                        draft.model_dump_json(),
                    ]
                )
            ]
        ),
    )
    Base.metadata.create_all(app.state.engine)
    validator = IndependentValidator()
    integrity = ExperimentIntegrityVerifier(validator)
    app.state.verification_workflow = VerificationWorkflow(
        reasoning_repository=app.state.reasoning_repository,
        mathematical_repository=app.state.mathematical_repository,
        mathematical_workflow=app.state.mathematical_workflow,
        repository=app.state.verification_repository,
        validator=validator,
        sensitivity_analyzer=SensitivityAnalyzer(app.state.experiment_engine),
        robustness_analyzer=RobustnessAnalyzer(app.state.experiment_engine),
        red_team_agent=_PassingRedTeamAgent(),  # type: ignore[arg-type]
        red_team_analyzer=RedTeamAnalyzer(),
        model_repair_agent=_UnusedRepairAgent(),  # type: ignore[arg-type]
        algorithm_selector=AlgorithmSelector(),
        experiment_integrity_verifier=integrity,
    )

    with TestClient(app) as client:
        project_id = client.post(
            "/api/v1/projects",
            json={
                "name": "Phase 6 paper fixture",
                "title": "Evidence-grounded allocation",
                "raw_problem": "Minimize cost while satisfying the verified demand parameter.",
            },
        ).json()["project_id"]
        project_uuid = UUID(project_id)
        assert (
            client.post(f"/api/v1/projects/{project_id}/reasoning/run", json={}).status_code == 200
        )
        assert (
            client.post(f"/api/v1/projects/{project_id}/mathematical/run", json={}).status_code
            == 200
        )
        verification = client.post(
            f"/api/v1/projects/{project_id}/verification/run",
            json={
                "sensitivity": {"perturbation_fractions": [0.05]},
                "robustness": {"scenario_fractions": [-0.05, 0.05]},
            },
        )
        assert verification.status_code == 200, verification.text
        verified_result_id = verification.json()["red_team"]["verified_result_id"]
        assert verified_result_id is not None

        app.state.paper_workflow = _paper_workflow(app, project_uuid)
        paper_response = client.post(f"/api/v1/projects/{project_id}/paper/run", json={})
        assert paper_response.status_code == 200, paper_response.text
        payload = paper_response.json()
        if payload["compile_record"]["status"] != "SUCCEEDED":
            raise AssertionError(
                payload["compile_record"]["stdout"]
                + "\nSTDERR:\n"
                + payload["compile_record"]["stderr"]
            )
        assert payload["paper"]["status"] == "READY_FOR_FINAL_JURY", {
            "issues": payload["quality"]["issues"],
            "compile_status": payload["compile_record"]["status"],
            "compile_error": payload["compile_record"]["error"],
            "compile_stderr": payload["compile_record"]["stderr"],
            "compile_stdout": payload["compile_record"]["stdout"],
        }
        assert payload["paper"]["evidence_snapshot"]["verified_result_id"] == verified_result_id
        assert payload["paper"]["paper_agent_is_mock"] is True
        assert payload["quality"]["unsupported_claim_count"] == 0
        assert payload["quality"]["verified_reference_count"] == 1
        assert payload["compile_record"]["status"] == "SUCCEEDED"
        assert payload["compile_record"]["exit_code"] == 0
        assert {item["kind"] for item in payload["artifacts"]} >= {
            "TEX",
            "BIB",
            "PDF",
            "MANIFEST",
            "FIGURE_DATA",
            "FIGURE_CODE",
            "FIGURE_IMAGE",
            "TABLE_DATA",
        }
        assert client.get(f"/api/v1/projects/{project_id}/paper").status_code == 200
        assert client.post(f"/api/v1/projects/{project_id}/paper/validate").status_code == 200
        assert client.post(f"/api/v1/projects/{project_id}/paper/render").status_code == 200
        artifacts = client.get(f"/api/v1/projects/{project_id}/paper/artifacts")
        assert artifacts.status_code == 200
        assert any(item["kind"] == PaperArtifactKind.PDF.value for item in artifacts.json())

        with Session(app.state.engine) as session:
            assert session.scalar(select(func.count()).select_from(PaperVersionRecord)) == 1
            assert session.scalar(select(func.count()).select_from(EvidenceRecordModel)) >= 8
            assert session.scalar(select(func.count()).select_from(ClaimRecord)) == 2
            assert session.scalar(select(func.count()).select_from(ClaimEvidenceLinkRecord)) == 2
            assert session.scalar(select(func.count()).select_from(ReferenceRecordModel)) == 1
            assert session.scalar(select(func.count()).select_from(LiteratureSearchRecord)) == 1
            assert session.scalar(select(func.count()).select_from(PaperSectionRecord)) == 1
            assert session.scalar(select(func.count()).select_from(FigureRecordModel)) == 1
            assert session.scalar(select(func.count()).select_from(TableRecordModel)) == 2
            assert session.scalar(select(func.count()).select_from(DocumentRegistryRecord)) >= 8
            assert session.scalar(select(func.count()).select_from(PaperArtifactRecord)) >= 8
            persisted = session.scalar(select(PaperVersionRecord))
            assert persisted is not None
            assert persisted.verified_result_id == UUID(verified_result_id)
            assert persisted.status == PaperQualityStatus.READY_FOR_FINAL_JURY.value
            assert persisted.compile_json["status"] == PaperCompileStatus.SUCCEEDED.value

        app.state.final_submission_workflow = FinalSubmissionWorkflow(
            reasoning_repository=app.state.reasoning_repository,
            paper_repository=app.state.paper_repository,
            repository=app.state.submission_repository,
            profiles=app.state.competition_profile_registry,
            jury_agent=_FinalJuryAgent(),  # type: ignore[arg-type]
            rule_engine=RuleEngine(app.state.file_store),
            package_builder=SubmissionPackageBuilder(app.state.file_store),
            store=app.state.file_store,
        )
        profile = app.state.competition_profile_registry.list()[0]
        final_response = client.post(
            f"/api/v1/projects/{project_id}/final/run",
            json={
                "profile_id": str(profile.profile_id),
                "profile_version": profile.version,
                "paper_id": payload["paper"]["paper_id"],
                "paper_version": payload["paper"]["version"],
                "freeze_on_pass": True,
            },
        )
        assert final_response.status_code == 200, final_response.text
        final = final_response.json()
        assert final["jury"]["decision"] == "PASS"
        assert final["jury"]["reviewer_is_mock"] is True
        assert final["submission_check"]["status"] == "PASS"
        assert final["snapshot"]["status"] == "FROZEN"
        assert final["state"]["current_stage"] == "FINAL"
        assert final["state"]["submission_state"]["status"] == "FROZEN"
        assert final["manifest"]["paper_version"] == payload["paper"]["version"]
        assert final["manifest"]["verified_result_id"] == verified_result_id

        package_record = next(
            item
            for item in final["artifacts"]
            if item["role"] == SubmissionArtifactRole.PACKAGE.value
        )
        package_bytes = app.state.file_store.read_bytes(package_record["storage_key"])
        from io import BytesIO
        from zipfile import ZipFile

        with ZipFile(BytesIO(package_bytes)) as archive:
            assert "paper.pdf" in archive.namelist()
            assert "submission_manifest.json" in archive.namelist()
            assert archive.read("paper.pdf").startswith(b"%PDF-")
        frozen_response = client.get(f"/api/v1/projects/{project_id}/submission")
        assert frozen_response.json()["status"] == "FROZEN"
        assert client.get(f"/api/v1/projects/{project_id}/submission/artifacts").status_code == 200

        with Session(app.state.engine) as session:
            jury_row = session.scalar(select(FinalJuryReportRecord))
            assert jury_row is not None
            original_report = dict(jury_row.report_json)
            jury_row.report_json = {**original_report, "claimed_score": 0}
            session.commit()
        jury_dirty = client.get(f"/api/v1/projects/{project_id}/submission")
        assert jury_dirty.json()["status"] == "DIRTY"
        with Session(app.state.engine) as session:
            jury_row = session.scalar(select(FinalJuryReportRecord))
            assert jury_row is not None
            jury_row.report_json = original_report
            session.commit()
        restored = client.get(f"/api/v1/projects/{project_id}/submission")
        assert restored.json()["status"] == "FROZEN"

        with Session(app.state.engine) as session:
            assert session.scalar(select(func.count()).select_from(FinalJuryReportRecord)) == 1
            assert (
                session.scalar(select(func.count()).select_from(RequirementCoverageRecordModel))
                == 3
            )
            assert session.scalar(select(func.count()).select_from(SubmissionCheckRecord)) == 1
            assert session.scalar(select(func.count()).select_from(SubmissionSnapshotRecord)) == 1
            assert session.scalar(select(func.count()).select_from(SubmissionManifestRecord)) == 1
            assert (
                session.scalar(select(func.count()).select_from(SubmissionArtifactRecordModel)) >= 5
            )

        app.state.file_store.resolve(package_record["storage_key"]).write_bytes(
            package_bytes + b"tampered"
        )
        dirty_response = client.get(f"/api/v1/projects/{project_id}/submission")
        assert dirty_response.status_code == 200
        assert dirty_response.json()["status"] == "DIRTY"
