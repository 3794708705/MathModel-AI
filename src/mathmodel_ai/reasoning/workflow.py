from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID, uuid4

from pydantic import BaseModel

from mathmodel_ai.agents import (
    AgentRunResult,
    AgentRunStatus,
    ModelExplorer,
    ModelJury,
    ProblemAgent,
)
from mathmodel_ai.core.errors import AgentRunError, QualityGateError
from mathmodel_ai.reasoning.quality_gates import (
    explore_quality_gate,
    select_quality_gate,
    understand_quality_gate,
)
from mathmodel_ai.reasoning.repository import ReasoningRepository
from mathmodel_ai.reasoning.state_machine import ensure_transition
from mathmodel_ai.routing.schemas import TaskProfile, TaskType
from mathmodel_ai.schemas.model_selection import (
    ModelDecisionEvidence,
    ModelExploration,
    ModelExplorerInput,
    ModelJuryInput,
    ModelJuryWeights,
    ModelSelection,
)
from mathmodel_ai.schemas.problem_analysis import ProblemAgentInput, ProblemAnalysis
from mathmodel_ai.schemas.problem_state import ProblemState, WorkflowStage, WorkflowStatus
from mathmodel_ai.schemas.quality import QualityGateResult, StageHistoryEntry

ReasoningAgentRun = (
    AgentRunResult[ProblemAnalysis]
    | AgentRunResult[ModelExploration]
    | AgentRunResult[ModelSelection]
)


@dataclass(frozen=True)
class StageOutcome[OutputT: BaseModel]:
    state: ProblemState
    output: OutputT
    run: AgentRunResult[OutputT]
    gate: QualityGateResult


@dataclass(frozen=True)
class ReasoningWorkflowOutcome:
    state: ProblemState
    runs: tuple[ReasoningAgentRun, ...]

    @property
    def is_mock(self) -> bool:
        return bool(self.runs) and all(run.is_mock for run in self.runs)


class ReasoningWorkflow:
    def __init__(
        self,
        *,
        repository: ReasoningRepository,
        problem_agent: ProblemAgent,
        model_explorer: ModelExplorer,
        model_jury: ModelJury,
        weights: ModelJuryWeights,
    ) -> None:
        self._repository = repository
        self._problem_agent = problem_agent
        self._model_explorer = model_explorer
        self._model_jury = model_jury
        self._weights = weights

    async def analyze(
        self, project_id: UUID, *, user_notes: list[str] | None = None
    ) -> StageOutcome[ProblemAnalysis]:
        state = self._repository.load_current(project_id)
        ensure_transition(state.current_stage, WorkflowStage.UNDERSTAND)
        result = await self._problem_agent.run(
            ProblemAgentInput(
                title=state.title,
                raw_problem=state.raw_problem,
                competition_context=state.competition,
                optional_user_notes=user_notes or [],
            ),
            state,
            TaskProfile(
                task_type=TaskType.PROBLEM_UNDERSTANDING,
                complexity=3,
                reasoning_requirement=4,
                math_requirement=3,
                blast_radius=3,
            ),
        )
        output = self._require_output(state, result)
        gate = understand_quality_gate(output)
        self._require_gate(state, result, gate)
        next_state = self._advance_state(
            state,
            stage=WorkflowStage.UNDERSTAND,
            run=result,
            gate=gate,
            updated_by=self._problem_agent.name,
            reason="structured problem analysis accepted",
            changes={
                "problem_analysis": output,
                "evidence_items": output.all_evidence(),
                "proposed_assumptions": output.assumptions_required,
                "subproblems": output.subproblems,
                "ambiguities": output.ambiguities,
            },
        )
        stored_run = self._repository.save_revision(
            project_id=project_id, state=next_state, run=result
        )
        return StageOutcome(state=next_state, output=output, run=stored_run, gate=gate)

    async def explore(
        self, project_id: UUID, *, user_guidance: list[str] | None = None
    ) -> StageOutcome[ModelExploration]:
        state = self._repository.load_current(project_id)
        ensure_transition(state.current_stage, WorkflowStage.EXPLORE)
        if state.problem_analysis is None:
            raise InvalidStateError("problem analysis is absent at EXPLORE")
        result = await self._model_explorer.run(
            ModelExplorerInput(
                analysis=state.problem_analysis,
                user_guidance=user_guidance or [],
            ),
            state,
            TaskProfile(
                task_type=TaskType.MODEL_EXPLORATION,
                complexity=4,
                reasoning_requirement=4,
                math_requirement=4,
                review_requirement=3,
                blast_radius=4,
            ),
        )
        output = self._require_output(state, result)
        required = {item.subproblem_id for item in state.problem_analysis.subproblems}
        gate = explore_quality_gate(output, required)
        self._require_gate(state, result, gate)
        next_state = self._advance_state(
            state,
            stage=WorkflowStage.EXPLORE,
            run=result,
            gate=gate,
            updated_by=self._model_explorer.name,
            reason="distinct model candidate set accepted",
            changes={"candidate_models": output.candidates},
        )
        stored_run = self._repository.save_revision(
            project_id=project_id, state=next_state, run=result
        )
        return StageOutcome(state=next_state, output=output, run=stored_run, gate=gate)

    async def select(
        self, project_id: UUID, *, jury_notes: list[str] | None = None
    ) -> StageOutcome[ModelSelection]:
        state = self._repository.load_current(project_id)
        ensure_transition(state.current_stage, WorkflowStage.SELECT)
        if state.problem_analysis is None:
            raise InvalidStateError("problem analysis is absent at SELECT")
        result = await self._model_jury.run(
            ModelJuryInput(
                analysis=state.problem_analysis,
                candidates=state.candidate_models,
                jury_notes=jury_notes or [],
            ),
            state,
            TaskProfile(
                task_type=TaskType.MODEL_JURY,
                complexity=4,
                reasoning_requirement=4,
                math_requirement=4,
                review_requirement=4,
                blast_radius=4,
            ),
        )
        output = self._require_output(state, result)
        candidate_ids = {candidate.candidate_id for candidate in state.candidate_models}
        gate = select_quality_gate(output, candidate_ids)
        self._require_gate(state, result, gate)
        by_id = {candidate.candidate_id: candidate for candidate in state.candidate_models}
        decision = ModelDecisionEvidence(
            decision_id=f"DECISION-{uuid4().hex}",
            candidate_scores=output.score_matrix,
            weights=self._weights,
            jury_rationale=output.decision_reason,
            selected_model_id=output.selected_model_id,
            backup_model_id=output.backup_model_id,
            agent_run_id=result.run_id,
        )
        next_state = self._advance_state(
            state,
            stage=WorkflowStage.SELECT,
            run=result,
            gate=gate,
            updated_by=self._model_jury.name,
            reason="deterministic weighted model selection accepted",
            changes={
                "model_scores": output.score_matrix,
                "selected_model": by_id[output.selected_model_id],
                "backup_model": by_id[output.backup_model_id],
                "model_selection": output,
                "model_selection_version": self._weights.version,
                "decision_evidence": [*state.decision_evidence, decision],
            },
        )
        stored_run = self._repository.save_revision(
            project_id=project_id,
            state=next_state,
            run=result,
            decision=decision,
        )
        return StageOutcome(state=next_state, output=output, run=stored_run, gate=gate)

    async def run(
        self,
        project_id: UUID,
        *,
        user_notes: list[str] | None = None,
        user_guidance: list[str] | None = None,
        jury_notes: list[str] | None = None,
    ) -> ReasoningWorkflowOutcome:
        understand = await self.analyze(project_id, user_notes=user_notes)
        explore = await self.explore(project_id, user_guidance=user_guidance)
        select = await self.select(project_id, jury_notes=jury_notes)
        runs: tuple[ReasoningAgentRun, ...] = (
            understand.run,
            explore.run,
            select.run,
        )
        return ReasoningWorkflowOutcome(state=select.state, runs=runs)

    def _require_output[OutputT: BaseModel](
        self, state: ProblemState, result: AgentRunResult[OutputT]
    ) -> OutputT:
        if result.status is not AgentRunStatus.SUCCEEDED or result.output is None:
            self._repository.record_run(state.project_id, state.problem_id, result)
            raise AgentRunError(
                f"{result.agent_name} did not produce accepted structured output: "
                f"{'; '.join(result.errors) or result.status.value}"
            )
        return result.output

    def _require_gate[OutputT: BaseModel](
        self,
        state: ProblemState,
        result: AgentRunResult[OutputT],
        gate: QualityGateResult,
    ) -> None:
        if gate.status.value != "PASS":
            rejected = result.model_copy(
                update={
                    "status": AgentRunStatus.RETRY,
                    "errors": [
                        *result.errors,
                        f"{gate.gate} gate: {', '.join(gate.errors)}",
                    ],
                }
            )
            self._repository.record_run(state.project_id, state.problem_id, rejected)
            raise QualityGateError(
                f"{gate.gate} quality gate rejected output: {', '.join(gate.errors)}"
            )

    @staticmethod
    def _advance_state[OutputT: BaseModel](
        state: ProblemState,
        *,
        stage: WorkflowStage,
        run: AgentRunResult[OutputT],
        gate: QualityGateResult,
        updated_by: str,
        reason: str,
        changes: dict[str, object],
    ) -> ProblemState:
        next_version = state.version + 1
        history = StageHistoryEntry(
            from_stage=state.current_stage.value,
            to_stage=stage.value,
            status=WorkflowStatus.SUCCEEDED.value,
            input_version=state.version,
            output_version=next_version,
            updated_by=updated_by,
            reason=reason,
            agent_run_id=run.run_id,
        )
        payload = state.model_dump()
        payload.update(
            {
                **changes,
                "version": next_version,
                "current_stage": stage,
                "status": WorkflowStatus.SUCCEEDED,
                "quality_gates": [*state.quality_gates, gate],
                "stage_history": [*state.stage_history, history],
                "updated_by": updated_by,
                "update_reason": reason,
                "updated_at": datetime.now(UTC),
            }
        )
        return ProblemState.model_validate(payload)


class InvalidStateError(QualityGateError):
    """A persisted state is missing data required by its declared stage."""
