from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID, uuid4

from pydantic import BaseModel

from mathmodel_ai.agents import (
    AgentRunResult,
    AgentRunStatus,
    ModelRepairAgent,
    RedTeamAgent,
)
from mathmodel_ai.core.errors import AgentRunError, QualityGateError
from mathmodel_ai.mathematical.algorithms import AlgorithmSelector
from mathmodel_ai.mathematical.digests import mathematical_model_digest
from mathmodel_ai.mathematical.quality_gates import model_quality_gate
from mathmodel_ai.mathematical.repository import MathematicalRepository, ResultContext
from mathmodel_ai.mathematical.workflow import MathematicalWorkflow, SolveStageOutcome
from mathmodel_ai.reasoning.repository import ReasoningRepository
from mathmodel_ai.reasoning.state_machine import ensure_transition
from mathmodel_ai.routing.schemas import EscalationLevel, TaskProfile, TaskType
from mathmodel_ai.schemas.benchmark import CausalScienceCheck
from mathmodel_ai.schemas.independent_verification import ReviewedValidationEvidence
from mathmodel_ai.schemas.mathematical import MathematicalModelRef
from mathmodel_ai.schemas.problem_state import (
    EquationRef,
    ProblemState,
    SymbolRef,
    VerificationStatus,
    WorkflowStage,
    WorkflowStatus,
)
from mathmodel_ai.schemas.quality import QualityGateResult, QualityGateStatus, StageHistoryEntry
from mathmodel_ai.schemas.verification import (
    ModelRepairInput,
    ModelRepairOutput,
    RedTeamDraft,
    RedTeamInput,
    RedTeamReport,
    RedTeamReportRef,
    RepairCycleRecord,
    RepairCycleRef,
    RepairCycleStatus,
    RobustnessConfig,
    RobustnessReport,
    RobustnessReportRef,
    SensitivityConfig,
    SensitivityReport,
    SensitivityReportRef,
    ValidationCheckStatus,
    ValidationReport,
    ValidationReportRef,
    ValidationStatus,
)
from mathmodel_ai.solvers.base import SolverExecution
from mathmodel_ai.verification.causal_holdout import AuditedCausalEvidence
from mathmodel_ai.verification.experiment_integrity import ExperimentIntegrityVerifier
from mathmodel_ai.verification.experiments import ExperimentOutcome
from mathmodel_ai.verification.quality_gates import (
    red_team_quality_gate,
    repair_quality_gate,
    robustness_quality_gate,
    sensitivity_quality_gate,
    validation_quality_gate,
    verified_result_quality_gate,
)
from mathmodel_ai.verification.red_team import RedTeamAnalyzer
from mathmodel_ai.verification.repository import VerificationRepository
from mathmodel_ai.verification.robustness import RobustnessAnalyzer
from mathmodel_ai.verification.sensitivity import SensitivityAnalyzer
from mathmodel_ai.verification.validation import IndependentValidator


@dataclass(frozen=True)
class ValidationStageOutcome:
    state: ProblemState
    report: ValidationReport
    gate: QualityGateResult


@dataclass(frozen=True)
class SensitivityStageOutcome:
    state: ProblemState
    report: SensitivityReport
    gate: QualityGateResult


@dataclass(frozen=True)
class RobustnessStageOutcome:
    state: ProblemState
    report: RobustnessReport
    gate: QualityGateResult


@dataclass(frozen=True)
class RedTeamStageOutcome:
    state: ProblemState
    report: RedTeamReport
    run: AgentRunResult[RedTeamDraft]
    gate: QualityGateResult
    verified_gate: QualityGateResult


@dataclass(frozen=True)
class RepairStageOutcome:
    state: ProblemState
    output: ModelRepairOutput
    cycle: RepairCycleRecord
    run: AgentRunResult[ModelRepairOutput]
    model_gate: QualityGateResult
    repair_gate: QualityGateResult


@dataclass(frozen=True)
class VerificationRunOutcome:
    validation: ValidationStageOutcome
    sensitivity: SensitivityStageOutcome
    robustness: RobustnessStageOutcome
    red_team: RedTeamStageOutcome


@dataclass(frozen=True)
class RepairLoopIteration:
    repair: RepairStageOutcome
    solve: SolveStageOutcome | None
    verification: VerificationRunOutcome | None


@dataclass(frozen=True)
class RepairLoopOutcome:
    state: ProblemState
    iterations: list[RepairLoopIteration]
    final_red_team: RedTeamReport
    resolved: bool
    exhausted: bool


class VerificationWorkflow:
    def __init__(
        self,
        *,
        reasoning_repository: ReasoningRepository,
        mathematical_repository: MathematicalRepository,
        mathematical_workflow: MathematicalWorkflow,
        repository: VerificationRepository,
        validator: IndependentValidator,
        sensitivity_analyzer: SensitivityAnalyzer,
        robustness_analyzer: RobustnessAnalyzer,
        red_team_agent: RedTeamAgent,
        red_team_analyzer: RedTeamAnalyzer,
        model_repair_agent: ModelRepairAgent,
        algorithm_selector: AlgorithmSelector,
        experiment_integrity_verifier: ExperimentIntegrityVerifier,
        max_repair_cycles: int = 3,
    ) -> None:
        if max_repair_cycles < 1 or max_repair_cycles > 3:
            raise ValueError("automatic repair cycles must be in [1, 3]")
        self._reasoning_repository = reasoning_repository
        self._mathematical_repository = mathematical_repository
        self._mathematical_workflow = mathematical_workflow
        self._repository = repository
        self._validator = validator
        self._sensitivity_analyzer = sensitivity_analyzer
        self._robustness_analyzer = robustness_analyzer
        self._red_team_agent = red_team_agent
        self._red_team_analyzer = red_team_analyzer
        self._model_repair_agent = model_repair_agent
        self._algorithm_selector = algorithm_selector
        self._experiment_integrity_verifier = experiment_integrity_verifier
        self._max_repair_cycles = max_repair_cycles

    def validate(
        self,
        project_id: UUID,
        *,
        result_id: UUID | None = None,
        reviewed_evidence: ReviewedValidationEvidence | None = None,
        causal_auditor: Callable[[], AuditedCausalEvidence] | None = None,
    ) -> ValidationStageOutcome:
        state = self._reasoning_repository.load_current(project_id)
        ensure_transition(state.current_stage, WorkflowStage.VALIDATE)
        selected_result_id = result_id or (
            state.result_records[-1].result_id if state.result_records else None
        )
        if selected_result_id is None:
            raise QualityGateError("Phase 5 requires an explicit formal result reference")
        if not any(item.result_id == selected_result_id for item in state.result_records):
            raise QualityGateError("Phase 5 result is not registered in current ProblemState")
        context = self._context(project_id, state, selected_result_id)
        report = self._validator.validate(
            model=context.model,
            result=context.result,
            solver_run=context.solver_run,
            evidence=context.evidence,
            reviewed_evidence=reviewed_evidence,
            causal_evidence=causal_auditor() if causal_auditor is not None else None,
        )
        gate = validation_quality_gate(report)
        next_state = self._validation_state(state, report, gate)
        self._repository.persist_validation(report, next_state)
        return ValidationStageOutcome(state=next_state, report=report, gate=gate)

    def sensitivity(
        self,
        project_id: UUID,
        *,
        config: SensitivityConfig | None = None,
        reviewed_evidence: ReviewedValidationEvidence | None = None,
    ) -> SensitivityStageOutcome:
        state = self._reasoning_repository.load_current(project_id)
        ensure_transition(state.current_stage, WorkflowStage.SENSITIVITY)
        validation = self._repository.get_validation(project_id)
        context = self._context(project_id, state, validation.result_id)
        report, outcomes = self._sensitivity_analyzer.analyze(
            model=context.model,
            result=context.result,
            validation=validation,
            config=config or SensitivityConfig(),
            reviewed_evidence=reviewed_evidence,
        )
        gate = sensitivity_quality_gate(report, reviewed_evidence)
        next_state = self._sensitivity_state(
            state,
            report,
            gate,
            outcomes,
            fallback_execution_id=context.execution.run_id,
        )
        self._repository.persist_sensitivity(report, outcomes, next_state)
        return SensitivityStageOutcome(state=next_state, report=report, gate=gate)

    def robustness(
        self,
        project_id: UUID,
        *,
        config: RobustnessConfig | None = None,
        reviewed_evidence: ReviewedValidationEvidence | None = None,
    ) -> RobustnessStageOutcome:
        state = self._reasoning_repository.load_current(project_id)
        ensure_transition(state.current_stage, WorkflowStage.ROBUSTNESS)
        validation = self._repository.get_validation(project_id)
        sensitivity = self._repository.get_sensitivity(project_id)
        context = self._context(project_id, state, validation.result_id)
        report, outcomes = self._robustness_analyzer.analyze(
            model=context.model,
            result=context.result,
            validation=validation,
            sensitivity=sensitivity,
            config=config or RobustnessConfig(),
            reviewed_evidence=reviewed_evidence,
        )
        gate = robustness_quality_gate(report, reviewed_evidence)
        next_state = self._robustness_state(
            state,
            report,
            gate,
            outcomes,
            fallback_execution_id=context.execution.run_id,
        )
        self._repository.persist_robustness(report, outcomes, next_state)
        return RobustnessStageOutcome(state=next_state, report=report, gate=gate)

    async def red_team(
        self,
        project_id: UUID,
        *,
        user_guidance: list[str] | None = None,
        reviewed_evidence: ReviewedValidationEvidence | None = None,
        causal_auditor: Callable[[], AuditedCausalEvidence] | None = None,
    ) -> RedTeamStageOutcome:
        state = self._reasoning_repository.load_current(project_id)
        ensure_transition(state.current_stage, WorkflowStage.RED_TEAM)
        validation = self._repository.get_validation(project_id)
        sensitivity = self._repository.get_sensitivity(project_id)
        robustness = self._repository.get_robustness(project_id)
        context = self._context(project_id, state, validation.result_id)
        run = await self._red_team_agent.run(
            RedTeamInput(
                mathematical_model=context.model,
                validation=validation,
                sensitivity=sensitivity,
                robustness=robustness,
                user_guidance=user_guidance or [],
            ),
            state,
            TaskProfile(
                task_type=TaskType.RED_TEAM,
                complexity=4,
                reasoning_requirement=4,
                math_requirement=4,
                review_requirement=5,
                blast_radius=4,
                minimum_level=EscalationLevel.FLAGSHIP_XHIGH,
            ),
        )
        draft = self._require_agent_output(state, run, "Red Team")
        report = self._red_team_analyzer.compile(
            model=context.model,
            validation=validation,
            sensitivity=sensitivity,
            robustness=robustness,
            draft=draft,
            reviewer_agent_run_id=run.run_id,
            review_is_mock=run.is_mock,
        )
        gate = red_team_quality_gate(report)
        validation_errors = self._validator.audit_report(
            report=validation,
            model=context.model,
            result=context.result,
            solver_run=context.solver_run,
            evidence=context.evidence,
            reviewed_evidence=reviewed_evidence,
            causal_evidence=causal_auditor() if causal_auditor is not None else None,
        )
        sensitivity_errors = self._repository.audit_experiment_report(
            project_id=project_id,
            model=context.model,
            experiments=sensitivity.experiments,
            verifier=self._experiment_integrity_verifier,
            sensitivity_run_id=sensitivity.sensitivity_id,
        )
        robustness_errors = self._repository.audit_experiment_report(
            project_id=project_id,
            model=context.model,
            experiments=robustness.experiments,
            verifier=self._experiment_integrity_verifier,
            robustness_run_id=robustness.robustness_id,
        )
        verified_gate = verified_result_quality_gate(
            validation,
            sensitivity,
            robustness,
            report,
            validation_integrity_errors=validation_errors,
            sensitivity_integrity_errors=sensitivity_errors,
            robustness_integrity_errors=robustness_errors,
            reviewed_evidence=reviewed_evidence,
        )
        next_state = self._red_team_state(
            state,
            report,
            gate,
            verified_gate,
            run.run_id,
        )
        stored_run = self._repository.persist_red_team(report, next_state, run)
        return RedTeamStageOutcome(
            state=next_state,
            report=report,
            run=stored_run,
            gate=gate,
            verified_gate=verified_gate,
        )

    async def repair(
        self,
        project_id: UUID,
        *,
        user_guidance: list[str] | None = None,
    ) -> RepairStageOutcome:
        state = self._reasoning_repository.load_current(project_id)
        ensure_transition(state.current_stage, WorkflowStage.MODEL_REPAIR)
        report = self._repository.get_red_team(project_id)
        if report.critical_count == 0:
            raise QualityGateError("MODEL_REPAIR requires an unresolved critical finding")
        context = self._context(project_id, state, report.result_id)
        previous_cycles = self._repository.repair_cycle_count(context.model.model_id)
        if previous_cycles >= self._max_repair_cycles:
            self._mark_repair_limit(state, report)
            raise QualityGateError(
                "MODEL_REPAIR exhausted the configured automatic cycle limit "
                f"({self._max_repair_cycles}); human review required"
            )
        cycle_number = previous_cycles + 1
        validation = self._repository.get_validation(project_id, report.validation_id)
        sensitivity = self._repository.get_sensitivity(project_id, report.sensitivity_id)
        robustness = self._repository.get_robustness(project_id, report.robustness_id)
        run = await self._model_repair_agent.run(
            ModelRepairInput(
                assigned_version=context.model.version + 1,
                repair_cycle=cycle_number,
                current_model=context.model,
                red_team_report=report,
                validation=validation,
                sensitivity=sensitivity,
                robustness=robustness,
                user_guidance=user_guidance or [],
            ),
            state,
            TaskProfile(
                task_type=TaskType.MODEL_REPAIR,
                complexity=5,
                reasoning_requirement=5,
                math_requirement=5,
                review_requirement=5,
                blast_radius=5,
                minimum_level=EscalationLevel.FLAGSHIP_XHIGH,
            ),
        )
        output = self._require_agent_output(state, run, "Model Repair")
        repair_gate = repair_quality_gate(output, report, is_mock=run.is_mock)
        model_gate = model_quality_gate(output.revised_model, state)
        target_digest = mathematical_model_digest(output.revised_model)
        accepted = (
            repair_gate.status is QualityGateStatus.PASS
            and model_gate.status is QualityGateStatus.PASS
        )
        cycle = RepairCycleRecord(
            project_id=state.project_id,
            problem_id=state.problem_id,
            stable_model_id=context.model.model_id,
            source_model_version=context.model.version,
            source_model_digest=mathematical_model_digest(context.model),
            target_model_version=output.revised_model.version if accepted else None,
            target_model_digest=target_digest if accepted else None,
            red_team_report_id=report.report_id,
            repair_cycle=cycle_number,
            actions=output.actions,
            addressed_finding_ids=output.addressed_finding_ids,
            remaining_risks=output.remaining_risks,
            agent_run_id=run.run_id,
            is_mock=run.is_mock,
            status=(
                RepairCycleStatus.ACCEPTED
                if accepted
                else RepairCycleStatus.HUMAN_REVIEW
                if run.is_mock or repair_gate.status is QualityGateStatus.HUMAN_REVIEW
                else RepairCycleStatus.REJECTED
            ),
        )
        target_record_id = uuid4()
        next_state = self._repair_state(
            state,
            output=output,
            cycle=cycle,
            model_gate=model_gate,
            repair_gate=repair_gate,
            agent_run_id=run.run_id,
            target_record_id=target_record_id if accepted else None,
        )
        if accepted:
            stored = self._repository.persist_repair(
                source_model_record_id=context.mathematical_model_record_id,
                target_model_record_id=target_record_id,
                revised_model=output.revised_model,
                repair=cycle,
                state=next_state,
                run=run,
            )
        else:
            stored = self._repository.persist_repair_attempt(
                source_model_record_id=context.mathematical_model_record_id,
                repair=cycle,
                state=next_state,
                run=run,
            )
        return RepairStageOutcome(
            state=next_state,
            output=output,
            cycle=cycle,
            run=stored,
            model_gate=model_gate,
            repair_gate=repair_gate,
        )

    async def run(
        self,
        project_id: UUID,
        *,
        sensitivity_config: SensitivityConfig | None = None,
        robustness_config: RobustnessConfig | None = None,
        user_guidance: list[str] | None = None,
        reviewed_evidence: ReviewedValidationEvidence | None = None,
        causal_auditor: Callable[[], AuditedCausalEvidence] | None = None,
    ) -> VerificationRunOutcome:
        validation = self.validate(
            project_id, reviewed_evidence=reviewed_evidence, causal_auditor=causal_auditor
        )
        self._require_pass(validation.gate)
        if causal_auditor is not None and reviewed_evidence is None:
            self._require_complete_causal_policy(validation.report, causal_auditor())
        sensitivity = self.sensitivity(
            project_id, config=sensitivity_config, reviewed_evidence=reviewed_evidence
        )
        self._require_pass(sensitivity.gate)
        robustness = self.robustness(
            project_id, config=robustness_config, reviewed_evidence=reviewed_evidence
        )
        self._require_pass(robustness.gate)
        red_team = await self.red_team(
            project_id,
            user_guidance=user_guidance,
            reviewed_evidence=reviewed_evidence,
            causal_auditor=causal_auditor,
        )
        return VerificationRunOutcome(
            validation=validation,
            sensitivity=sensitivity,
            robustness=robustness,
            red_team=red_team,
        )

    @staticmethod
    def _require_complete_causal_policy(
        report: ValidationReport, evidence: AuditedCausalEvidence
    ) -> None:
        """Permit a model-independent host policy only with every science check."""
        required = {f"causal_science:{check.value}" for check in CausalScienceCheck}
        if report.status is not ValidationStatus.PASS:
            raise QualityGateError("CAUSAL_HOLDOUT_REVIEWED_REQUIREMENT_POLICY_MISSING")
        checks = {item.requirement: item for item in report.requirement_checks}
        if (
            evidence.formal_result_id != report.result_id
            or evidence.science_policy_sha256 is None
            or len(evidence.source_sha256) != 64
            or {f"causal_science:{check.value}" for check in evidence.required_scientific_checks}
            != required
            or not required <= checks.keys()
            or any(checks[name].status is not ValidationCheckStatus.PASS for name in required)
            or not any(
                ref == f"causal_science_policy_sha256:{evidence.science_policy_sha256}"
                for ref in report.evidence_refs
            )
        ):
            raise QualityGateError("CAUSAL_HOLDOUT_REVIEWED_REQUIREMENT_POLICY_MISSING")

    async def repair_until_clear(
        self,
        project_id: UUID,
        *,
        sensitivity_config: SensitivityConfig | None = None,
        robustness_config: RobustnessConfig | None = None,
        user_guidance: list[str] | None = None,
        repair_solver: Callable[
            [RepairStageOutcome],
            Awaitable[tuple[SolveStageOutcome, Callable[[], AuditedCausalEvidence] | None]],
        ]
        | None = None,
    ) -> RepairLoopOutcome:
        initial = self._repository.get_red_team(project_id)
        iterations: list[RepairLoopIteration] = []
        final_report = initial
        if initial.critical_count == 0:
            current_state = self._reasoning_repository.load_current(project_id)
            return RepairLoopOutcome(
                state=current_state,
                iterations=iterations,
                final_red_team=initial,
                resolved=current_state.verified_result_id == initial.result_id,
                exhausted=False,
            )
        completed_cycles = self._repository.repair_cycle_count(initial.model_id)
        remaining_cycles = max(self._max_repair_cycles - completed_cycles, 0)
        for _ in range(remaining_cycles):
            repair = await self.repair(project_id, user_guidance=user_guidance)
            if (
                repair.model_gate.status is not QualityGateStatus.PASS
                or repair.repair_gate.status is not QualityGateStatus.PASS
            ):
                iterations.append(RepairLoopIteration(repair=repair, solve=None, verification=None))
                return RepairLoopOutcome(
                    state=repair.state,
                    iterations=iterations,
                    final_red_team=final_report,
                    resolved=False,
                    exhausted=False,
                )
            causal_auditor: Callable[[], AuditedCausalEvidence] | None = None
            if repair_solver is None:
                solve = await self._mathematical_workflow.solve(project_id)
            else:
                solve, causal_auditor = await repair_solver(repair)
            if solve.gate.status is not QualityGateStatus.PASS:
                iterations.append(
                    RepairLoopIteration(repair=repair, solve=solve, verification=None)
                )
                return RepairLoopOutcome(
                    state=solve.state,
                    iterations=iterations,
                    final_red_team=final_report,
                    resolved=False,
                    exhausted=False,
                )
            if repair_solver is not None and causal_auditor is None:
                raise QualityGateError("causal repair solve needs a fresh holdout auditor")
            verification = await self.run(
                project_id,
                sensitivity_config=sensitivity_config,
                robustness_config=robustness_config,
                user_guidance=user_guidance,
                causal_auditor=causal_auditor,
            )
            final_report = verification.red_team.report
            iterations.append(
                RepairLoopIteration(
                    repair=repair,
                    solve=solve,
                    verification=verification,
                )
            )
            if verification.red_team.verified_gate.status is QualityGateStatus.PASS:
                return RepairLoopOutcome(
                    state=verification.red_team.state,
                    iterations=iterations,
                    final_red_team=final_report,
                    resolved=True,
                    exhausted=False,
                )
            if (
                verification.red_team.verified_gate.status is QualityGateStatus.HUMAN_REVIEW
                or final_report.critical_count == 0
            ):
                return RepairLoopOutcome(
                    state=verification.red_team.state,
                    iterations=iterations,
                    final_red_team=final_report,
                    resolved=False,
                    exhausted=False,
                )
        state = self._reasoning_repository.load_current(project_id)
        self._mark_repair_limit(state, final_report)
        return RepairLoopOutcome(
            state=self._reasoning_repository.load_current(project_id),
            iterations=iterations,
            final_red_team=final_report,
            resolved=False,
            exhausted=True,
        )

    def _context(
        self,
        project_id: UUID,
        state: ProblemState,
        result_id: UUID | None,
    ) -> ResultContext:
        if state.mathematical_model is None:
            raise QualityGateError("Phase 5 requires an accepted MathematicalModel")
        context = self._mathematical_repository.get_result_context(project_id, result_id)
        current = state.mathematical_model
        if (
            context.model.model_id,
            context.model.version,
            mathematical_model_digest(context.model),
        ) != (current.model_id, current.version, current.model_digest):
            raise QualityGateError("Phase 5 result does not belong to the current model revision")
        return context

    def _mark_repair_limit(self, state: ProblemState, report: RedTeamReport) -> None:
        next_version = state.version + 1
        gate = QualityGateResult(
            gate="MODEL_REPAIR",
            status=QualityGateStatus.HUMAN_REVIEW,
            checks={"automatic_cycle_limit_available": False},
            errors=["MODEL_REPAIR_GATE_FAIL:AUTOMATIC_CYCLE_LIMIT_EXHAUSTED"],
        )
        history = StageHistoryEntry(
            from_stage=state.current_stage.value,
            to_stage=state.current_stage.value,
            status=WorkflowStatus.HUMAN_REVIEW.value,
            input_version=state.version,
            output_version=next_version,
            updated_by="model_repair_agent",
            reason=(f"{self._max_repair_cycles} configured automatic repair cycles exhausted"),
            agent_run_id=report.reviewer_agent_run_id,
        )
        payload = state.model_dump()
        payload.update(
            {
                "schema_version": 5,
                "version": next_version,
                "status": WorkflowStatus.HUMAN_REVIEW,
                "quality_gates": [*state.quality_gates, gate],
                "stage_history": [*state.stage_history, history],
                "updated_by": "model_repair_agent",
                "update_reason": history.reason,
                "updated_at": datetime.now(UTC),
                "verified_result_id": None,
            }
        )
        self._repository.persist_state_only(ProblemState.model_validate(payload))

    def _require_agent_output[OutputT: BaseModel](
        self,
        state: ProblemState,
        run: AgentRunResult[OutputT],
        label: str,
    ) -> OutputT:
        if run.status is not AgentRunStatus.SUCCEEDED or run.output is None:
            self._reasoning_repository.record_run(
                state.project_id,
                state.problem_id,
                run,
            )
            raise AgentRunError(
                f"{label} agent did not produce accepted structured output: "
                f"{'; '.join(run.errors) or run.status.value}"
            )
        return run.output

    @staticmethod
    def _require_pass(gate: QualityGateResult) -> None:
        if gate.status is not QualityGateStatus.PASS:
            raise QualityGateError(
                f"{gate.gate} quality gate did not pass: " + ", ".join(gate.errors)
            )

    @staticmethod
    def _validation_state(
        state: ProblemState,
        report: ValidationReport,
        gate: QualityGateResult,
    ) -> ProblemState:
        passed = gate.status is QualityGateStatus.PASS
        target = WorkflowStage.VALIDATE if passed else WorkflowStage.SOLVE
        return VerificationWorkflow._report_state(
            state,
            target=target,
            gate=gate,
            updated_by="independent_validator",
            reason=(
                "independent validation passed"
                if passed
                else "independent validation requires a new solve or model review"
            ),
            execution_run_id=report.execution_record_id,
            field_name="validation_results",
            ref=ValidationReportRef(
                validation_id=report.validation_id,
                model_id=report.model_id,
                model_version=report.model_version,
                model_digest=report.model_digest,
                result_id=report.result_id,
                status=report.status,
            ),
        )

    @staticmethod
    def _sensitivity_state(
        state: ProblemState,
        report: SensitivityReport,
        gate: QualityGateResult,
        outcomes: list[ExperimentOutcome],
        *,
        fallback_execution_id: UUID,
    ) -> ProblemState:
        passed = gate.status is QualityGateStatus.PASS
        target = WorkflowStage.SENSITIVITY if passed else WorkflowStage.VALIDATE
        executions = [item.execution for item in outcomes if item.execution is not None]
        return VerificationWorkflow._report_state(
            state,
            target=target,
            gate=gate,
            updated_by="sensitivity_analyzer",
            reason=(
                "all configured sensitivity experiments passed"
                if passed
                else "sensitivity evidence requires retry or human review"
            ),
            execution_run_id=(
                executions[0].execution.record.run_id if executions else fallback_execution_id
            ),
            field_name="sensitivity_results",
            ref=SensitivityReportRef(
                sensitivity_id=report.sensitivity_id,
                model_id=report.model_id,
                model_version=report.model_version,
                model_digest=report.model_digest,
                validation_id=report.validation_id,
                status=report.status,
            ),
            executions=executions,
        )

    @staticmethod
    def _robustness_state(
        state: ProblemState,
        report: RobustnessReport,
        gate: QualityGateResult,
        outcomes: list[ExperimentOutcome],
        *,
        fallback_execution_id: UUID,
    ) -> ProblemState:
        passed = gate.status is QualityGateStatus.PASS
        target = WorkflowStage.ROBUSTNESS if passed else WorkflowStage.SENSITIVITY
        executions = [item.execution for item in outcomes if item.execution is not None]
        return VerificationWorkflow._report_state(
            state,
            target=target,
            gate=gate,
            updated_by="robustness_analyzer",
            reason=(
                "all configured robustness experiments passed"
                if passed
                else "robustness evidence requires retry or model repair"
            ),
            execution_run_id=(
                executions[0].execution.record.run_id if executions else fallback_execution_id
            ),
            field_name="robustness_results",
            ref=RobustnessReportRef(
                robustness_id=report.robustness_id,
                model_id=report.model_id,
                model_version=report.model_version,
                model_digest=report.model_digest,
                sensitivity_id=report.sensitivity_id,
                method=report.method,
                status=report.status,
            ),
            executions=executions,
        )

    @staticmethod
    def _red_team_state(
        state: ProblemState,
        report: RedTeamReport,
        gate: QualityGateResult,
        verified_gate: QualityGateResult,
        agent_run_id: UUID,
    ) -> ProblemState:
        verified = (
            gate.status is QualityGateStatus.PASS and verified_gate.status is QualityGateStatus.PASS
        )
        reason = (
            "complete Phase 5 evidence accepted the formal result"
            if verified
            else "Red Team requires model repair or human review"
        )
        next_state = VerificationWorkflow._report_state(
            state,
            target=WorkflowStage.RED_TEAM,
            gate=verified_gate,
            additional_gates=[gate],
            updated_by="red_team_agent",
            reason=reason,
            agent_run_id=agent_run_id,
            field_name="red_team_reports",
            ref=RedTeamReportRef(
                report_id=report.report_id,
                model_id=report.model_id,
                model_version=report.model_version,
                model_digest=report.model_digest,
                robustness_id=report.robustness_id,
                critical_count=report.critical_count,
                status=report.status,
                review_is_mock=report.review_is_mock,
            ),
        )
        payload = next_state.model_dump()
        payload["verified_result_id"] = report.result_id if verified else None
        payload["results"] = [
            item.model_copy(
                update={
                    "verification_status": (
                        VerificationStatus.VERIFIED
                        if verified and item.item_id == f"RESULT-{report.result_id}"
                        else VerificationStatus.UNVERIFIED
                    )
                }
            )
            for item in next_state.results
        ]
        return ProblemState.model_validate(payload)

    @staticmethod
    def _report_state(
        state: ProblemState,
        *,
        target: WorkflowStage,
        gate: QualityGateResult,
        updated_by: str,
        reason: str,
        field_name: str,
        ref: BaseModel,
        execution_run_id: UUID | None = None,
        agent_run_id: UUID | None = None,
        executions: list[SolverExecution] | None = None,
        additional_gates: list[QualityGateResult] | None = None,
    ) -> ProblemState:
        next_version = state.version + 1
        status = {
            QualityGateStatus.PASS: WorkflowStatus.SUCCEEDED,
            QualityGateStatus.HUMAN_REVIEW: WorkflowStatus.HUMAN_REVIEW,
            QualityGateStatus.ESCALATE: WorkflowStatus.ESCALATED,
            QualityGateStatus.RETRY: WorkflowStatus.RETRY,
        }[gate.status]
        history = StageHistoryEntry(
            from_stage=state.current_stage.value,
            to_stage=target.value,
            status=status.value,
            input_version=state.version,
            output_version=next_version,
            updated_by=updated_by,
            reason=reason,
            agent_run_id=agent_run_id,
            execution_run_id=execution_run_id,
        )
        payload = state.model_dump()
        payload[field_name] = [*payload[field_name], ref.model_dump()]
        payload.update(
            {
                "schema_version": 5,
                "version": next_version,
                "current_stage": target,
                "status": status,
                "quality_gates": [*state.quality_gates, *(additional_gates or []), gate],
                "stage_history": [*state.stage_history, history],
                "updated_by": updated_by,
                "update_reason": reason,
                "updated_at": datetime.now(UTC),
            }
        )
        if executions:
            payload["execution_records"] = [
                *state.execution_records,
                *(item.execution.record for item in executions),
            ]
            payload["tracked_artifacts"] = [
                *state.tracked_artifacts,
                *(artifact for item in executions for artifact in item.execution.artifact_records),
            ]
        return ProblemState.model_validate(payload)

    def _repair_state(
        self,
        state: ProblemState,
        *,
        output: ModelRepairOutput,
        cycle: RepairCycleRecord,
        model_gate: QualityGateResult,
        repair_gate: QualityGateResult,
        agent_run_id: UUID,
        target_record_id: UUID | None,
    ) -> ProblemState:
        accepted = target_record_id is not None
        next_version = state.version + 1
        model = output.revised_model
        target = WorkflowStage.MODEL_REPAIR if accepted else WorkflowStage.RED_TEAM
        status = (
            WorkflowStatus.SUCCEEDED
            if accepted
            else WorkflowStatus.HUMAN_REVIEW
            if repair_gate.status is QualityGateStatus.HUMAN_REVIEW
            else WorkflowStatus.RETRY
        )
        history = StageHistoryEntry(
            from_stage=state.current_stage.value,
            to_stage=target.value,
            status=status.value,
            input_version=state.version,
            output_version=next_version,
            updated_by="model_repair_agent",
            reason=(
                "repaired model revision passed MODEL and repair gates"
                if accepted
                else "repair proposal requires retry or human review"
            ),
            agent_run_id=agent_run_id,
        )
        payload = state.model_dump()
        payload.update(
            {
                "schema_version": 5,
                "version": next_version,
                "current_stage": target,
                "status": status,
                "quality_gates": [*state.quality_gates, model_gate, repair_gate],
                "stage_history": [*state.stage_history, history],
                "revisions": [
                    *state.revisions,
                    RepairCycleRef(
                        repair_id=cycle.repair_id,
                        stable_model_id=cycle.stable_model_id,
                        source_model_version=cycle.source_model_version,
                        target_model_version=cycle.target_model_version,
                        red_team_report_id=cycle.red_team_report_id,
                        repair_cycle=cycle.repair_cycle,
                        status=cycle.status,
                    ),
                ],
                "updated_by": "model_repair_agent",
                "update_reason": history.reason,
                "updated_at": datetime.now(UTC),
                "verified_result_id": None,
            }
        )
        if accepted:
            assert target_record_id is not None
            plan = self._algorithm_selector.select(model)
            payload.update(
                {
                    "mathematical_model": MathematicalModelRef(
                        record_id=target_record_id,
                        model_id=model.model_id,
                        version=model.version,
                        model_digest=mathematical_model_digest(model),
                        source_selected_model_id=model.source_selected_model_id,
                        status=model.status,
                    ),
                    "algorithm_plan": plan,
                    "variables": [
                        SymbolRef(
                            symbol_id=item.variable_id,
                            symbol=item.symbol,
                            meaning=item.description,
                            unit=item.unit.display if item.unit is not None else None,
                        )
                        for item in [
                            *model.decision_variables,
                            *model.state_variables,
                            *model.derived_variables,
                        ]
                    ],
                    "parameters": [
                        SymbolRef(
                            symbol_id=item.parameter_id,
                            symbol=item.symbol,
                            meaning=item.description,
                            unit=item.unit.display if item.unit is not None else None,
                        )
                        for item in [*model.parameters, *model.constants]
                    ],
                    "units": {symbol: unit.display for symbol, unit in model.units.items()},
                    "equations": [
                        EquationRef(
                            equation_id=item.equation_id,
                            latex=item.latex,
                            meaning=item.meaning,
                            source_refs=item.source_refs,
                            variables=item.symbol_refs,
                            parameters=item.parameter_refs,
                            units=[
                                value.display
                                for value in (item.unit_lhs, item.unit_rhs)
                                if value is not None
                            ],
                        )
                        for item in model.equations
                    ],
                }
            )
        return ProblemState.model_validate(payload)
