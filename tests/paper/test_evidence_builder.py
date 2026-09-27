from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID, uuid4

import pytest

from mathmodel_ai.mathematical.digests import mathematical_model_digest
from mathmodel_ai.mathematical.repository import ResultContext
from mathmodel_ai.paper.evidence import EvidenceBuildError, VerifiedEvidenceBuilder
from mathmodel_ai.schemas.mathematical import ModelAssumption
from mathmodel_ai.schemas.paper import EvidenceType, PaperQualityStatus, PaperVersionRef
from mathmodel_ai.schemas.problem_analysis import EvidenceType as StateEvidenceType
from mathmodel_ai.schemas.problem_state import (
    ProblemState,
    TraceableItem,
    VerificationStatus,
    WorkflowStage,
    WorkflowStatus,
)
from mathmodel_ai.schemas.quality import QualityGateResult, QualityGateStatus
from mathmodel_ai.schemas.results import ResultRecordRef
from mathmodel_ai.schemas.verification import (
    RedTeamDraft,
    RedTeamReport,
    RedTeamReportRef,
    RobustnessConfig,
    RobustnessReport,
    SensitivityConfig,
    SensitivityReport,
    ValidationReport,
)
from mathmodel_ai.verification.red_team import RedTeamAnalyzer
from mathmodel_ai.verification.robustness import RobustnessAnalyzer
from mathmodel_ai.verification.sensitivity import SensitivityAnalyzer
from mathmodel_ai.verification.validation import IndependentValidator
from tests.verification.helpers import experiment_engine, result_bundle


@dataclass
class _Chain:
    context: ResultContext
    validation: ValidationReport
    sensitivity: SensitivityReport
    robustness: RobustnessReport
    red_team: RedTeamReport
    state: ProblemState


class _MathematicalRepository:
    def __init__(self, contexts: dict[UUID, ResultContext]) -> None:
        self.contexts = contexts
        self.requested_ids: list[UUID] = []

    def get_result_context(self, project_id: UUID, result_id: UUID) -> ResultContext:
        context = self.contexts[result_id]
        assert context.result.project_id == project_id
        self.requested_ids.append(result_id)
        return context


class _VerificationRepository:
    def __init__(self, chain: _Chain) -> None:
        self.chain = chain
        self.accepted_science = chain.state

    def load_current(self, project_id: UUID) -> ProblemState:
        assert self.chain.state.project_id == project_id
        return self.chain.state

    def load_accepted_science(self, project_id: UUID) -> ProblemState:
        assert self.accepted_science.project_id == project_id
        return self.accepted_science

    def get_validation(self, project_id: UUID, report_id: UUID) -> ValidationReport:
        assert (project_id, report_id) == (
            self.chain.state.project_id,
            self.chain.validation.validation_id,
        )
        return self.chain.validation

    def get_sensitivity(self, project_id: UUID, report_id: UUID) -> SensitivityReport:
        assert (project_id, report_id) == (
            self.chain.state.project_id,
            self.chain.sensitivity.sensitivity_id,
        )
        return self.chain.sensitivity

    def get_robustness(self, project_id: UUID, report_id: UUID) -> RobustnessReport:
        assert (project_id, report_id) == (
            self.chain.state.project_id,
            self.chain.robustness.robustness_id,
        )
        return self.chain.robustness

    def get_red_team(self, project_id: UUID, report_id: UUID) -> RedTeamReport:
        assert (project_id, report_id) == (
            self.chain.state.project_id,
            self.chain.red_team.report_id,
        )
        return self.chain.red_team


def _verified_chain(*, with_assumptions: bool = False) -> _Chain:
    model, result, solver_run, execution, program, evidence = result_bundle()
    validation = IndependentValidator().validate(
        model=model,
        result=result,
        solver_run=solver_run,
        evidence=evidence,
    )
    sensitivity, _ = SensitivityAnalyzer(experiment_engine()).analyze(
        model=model,
        result=result,
        validation=validation,
        config=SensitivityConfig(perturbation_fractions=[0.1], max_runs=2),
    )
    robustness, _ = RobustnessAnalyzer(experiment_engine()).analyze(
        model=model,
        result=result,
        validation=validation,
        sensitivity=sensitivity,
        config=RobustnessConfig(scenario_fractions=[-0.1, 0.1], max_runs=2),
    )
    red_team = RedTeamAnalyzer().compile(
        model=model,
        validation=validation,
        sensitivity=sensitivity,
        robustness=robustness,
        draft=RedTeamDraft(summary="deterministic fixture passed"),
        reviewer_agent_run_id=uuid4(),
        review_is_mock=False,
    )
    if with_assumptions:
        model = model.model_copy(
            update={
                "assumptions": [
                    ModelAssumption(
                        assumption_id="ASSUMP-supported",
                        statement="Demand is fixed during each scenario.",
                        source_refs=["EVID-fact-1"],
                        supported=True,
                        support_reason="The verified scenario contract fixes demand per run.",
                    ),
                    ModelAssumption(
                        assumption_id="ASSUMP-proposed",
                        statement="All costs remain constant forever.",
                        source_refs=["agent:proposal"],
                        supported=False,
                    ),
                ]
            }
        )
        digest = mathematical_model_digest(model)
        result = result.model_copy(update={"model_digest": digest})
        solver_run = solver_run.model_copy(update={"model_digest": digest})
        execution = execution.model_copy(update={"model_digest": digest})
        validation = validation.model_copy(update={"model_digest": digest})
        sensitivity = sensitivity.model_copy(update={"model_digest": digest})
        robustness = robustness.model_copy(update={"model_digest": digest})
        red_team = red_team.model_copy(update={"model_digest": digest})
    context = ResultContext(
        mathematical_model_record_id=uuid4(),
        model=model,
        result=result,
        solver_run=solver_run,
        execution=execution,
        program=program,
        evidence=evidence,
    )
    state = ProblemState(
        project_id=model.project_id,
        problem_id=model.problem_id,
        title="Verified fixture",
        raw_problem="Minimize a verified linear objective.",
        version=10,
        result_records=[
            ResultRecordRef(
                result_id=result.result_id,
                model_id=result.model_id,
                model_version=result.model_version,
                model_digest=result.model_digest,
                solver_run_id=result.solver_run_id,
                status=result.status,
            )
        ],
        results=[
            TraceableItem(
                item_id=f"RESULT-{result.result_id}",
                kind=StateEvidenceType.RESULT,
                statement="verified fixture result",
                verification_status=VerificationStatus.VERIFIED,
            )
        ],
        red_team_reports=[
            RedTeamReportRef(
                report_id=red_team.report_id,
                model_id=red_team.model_id,
                model_version=red_team.model_version,
                model_digest=red_team.model_digest,
                robustness_id=red_team.robustness_id,
                critical_count=red_team.critical_count,
                status=red_team.status,
                review_is_mock=red_team.review_is_mock,
            )
        ],
        quality_gates=[
            QualityGateResult(
                gate="VERIFIED",
                subject_ref=f"result:{result.result_id}",
                status=QualityGateStatus.PASS,
                checks={
                    "validation": True,
                    "sensitivity": True,
                    "robustness": True,
                    "red_team": True,
                    "execution": True,
                },
            )
        ],
        verified_result_id=result.result_id,
        current_stage=WorkflowStage.RED_TEAM,
        status=WorkflowStatus.SUCCEEDED,
    )
    return _Chain(
        context=context,
        validation=validation,
        sensitivity=sensitivity,
        robustness=robustness,
        red_team=red_team,
        state=state,
    )


def _builder(
    chain: _Chain, contexts: dict[UUID, ResultContext] | None = None
) -> tuple[VerifiedEvidenceBuilder, _MathematicalRepository]:
    mathematical = _MathematicalRepository(
        contexts or {chain.context.result.result_id: chain.context}
    )
    return (
        VerifiedEvidenceBuilder(  # type: ignore[arg-type]
            mathematical_repository=mathematical,
            verification_repository=_VerificationRepository(chain),  # type: ignore[arg-type]
        ),
        mathematical,
    )


def test_failed_paper_retry_reuses_accepted_science_revision_and_hash() -> None:
    chain = _verified_chain()
    builder, _ = _builder(chain)
    original = builder.build(chain.state.project_id)
    current = chain.state.model_copy(
        update={
            "schema_version": 6,
            "version": chain.state.version + 1,
            "current_stage": WorkflowStage.PAPER,
            "status": WorkflowStatus.FAILED,
            "paper_versions": [
                PaperVersionRef(
                    paper_id=uuid4(),
                    version=1,
                    verified_result_id=chain.state.verified_result_id,
                    evidence_snapshot_hash=original.snapshot.snapshot_hash,
                    status=PaperQualityStatus.FAILED,
                )
            ],
        }
    )
    chain.state = current
    retried = builder.build(current.project_id)
    assert retried.state is current
    assert retried.snapshot.snapshot_hash == original.snapshot.snapshot_hash
    assert retried.snapshot.evidence_ids == original.snapshot.evidence_ids
    chain.state = current.model_copy(update={"title": "changed scientific input"})
    with pytest.raises(EvidenceBuildError, match="changed scientific state: title"):
        builder.build(current.project_id)


def test_q_evidence_builder_uses_only_explicit_verified_result_id() -> None:
    chain = _verified_chain()
    latest_id = uuid4()
    latest = chain.context.result.model_copy(
        update={
            "result_id": latest_id,
            "objective": 999,
            "evidence_refs": [
                f"model:{chain.context.model.model_id}:v{chain.context.model.version}",
                f"solver_run:{chain.context.result.solver_run_id}",
                f"execution:{chain.context.result.execution_record_id}",
            ],
        }
    )
    latest_context = ResultContext(
        mathematical_model_record_id=uuid4(),
        model=chain.context.model,
        result=latest,
        solver_run=chain.context.solver_run,
        execution=chain.context.execution,
        program=chain.context.program,
        evidence=chain.context.evidence,
    )
    builder, mathematical = _builder(
        chain,
        {
            chain.context.result.result_id: chain.context,
            latest_id: latest_context,
        },
    )

    bundle = builder.build(chain.state.project_id)

    assert mathematical.requested_ids == [chain.state.verified_result_id]
    assert bundle.snapshot.verified_result_id == chain.state.verified_result_id
    result_record = next(
        item for item in bundle.records if item.evidence_type is EvidenceType.RESULT
    )
    assert result_record.structured_payload["objective"]["value"] == 30


def test_q_repository_cannot_substitute_latest_result_for_verified_result() -> None:
    chain = _verified_chain()
    substituted = chain.context.result.model_copy(update={"result_id": uuid4(), "objective": 999})
    context = ResultContext(
        mathematical_model_record_id=chain.context.mathematical_model_record_id,
        model=chain.context.model,
        result=substituted,
        solver_run=chain.context.solver_run,
        execution=chain.context.execution,
        program=chain.context.program,
        evidence=chain.context.evidence,
    )
    builder, _ = _builder(chain, {chain.context.result.result_id: context})

    with pytest.raises(EvidenceBuildError, match="other than verified_result_id"):
        builder.build(chain.state.project_id)


def test_s_only_supported_model_assumptions_enter_verified_evidence() -> None:
    chain = _verified_chain(with_assumptions=True)
    builder, _ = _builder(chain)

    bundle = builder.build(chain.state.project_id)

    assumptions = [item for item in bundle.records if item.evidence_type is EvidenceType.ASSUMPTION]
    assert [item.source_id for item in assumptions] == ["ASSUMP-supported"]
    assert all(item.verified for item in assumptions)
