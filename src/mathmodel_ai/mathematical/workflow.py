from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID, uuid4

from mathmodel_ai.agents import AgentRunResult, AgentRunStatus, CodeAgent, MathModeler
from mathmodel_ai.core.errors import AgentRunError, QualityGateError
from mathmodel_ai.mathematical.algorithms import AlgorithmSelector
from mathmodel_ai.mathematical.digests import mathematical_model_digest
from mathmodel_ai.mathematical.evidence import EvidenceIntegrityVerifier
from mathmodel_ai.mathematical.quality_gates import model_quality_gate, solve_quality_gate
from mathmodel_ai.mathematical.repository import MathematicalRepository
from mathmodel_ai.mathematical.strategy import ExecutionStrategySelector
from mathmodel_ai.reasoning.repository import ReasoningRepository
from mathmodel_ai.reasoning.state_machine import ensure_transition
from mathmodel_ai.routing.schemas import EscalationLevel, TaskProfile, TaskType
from mathmodel_ai.schemas.mathematical import (
    MathematicalModel,
    MathematicalModelRef,
    MathModelerInput,
)
from mathmodel_ai.schemas.problem_analysis import EvidenceType
from mathmodel_ai.schemas.problem_state import (
    ArtifactRef,
    EquationRef,
    ProblemState,
    SymbolRef,
    TraceableItem,
    VerificationStatus,
    WorkflowStage,
    WorkflowStatus,
)
from mathmodel_ai.schemas.program import (
    CodeAgentInput,
    ExecutionStrategy,
    ExecutionStrategyDecision,
    GeneratedProgram,
    GeneratedProgramRef,
)
from mathmodel_ai.schemas.quality import QualityGateResult, QualityGateStatus, StageHistoryEntry
from mathmodel_ai.schemas.results import ResultRecord, ResultRecordRef
from mathmodel_ai.schemas.solver import (
    AlgorithmPlan,
    SolverOptions,
    SolverRouteDecision,
    SolverRun,
    SolverRunRef,
)
from mathmodel_ai.solvers.base import SolverExecution
from mathmodel_ai.solvers.generated import GeneratedProgramExecutor
from mathmodel_ai.solvers.router import SolverRouter


@dataclass(frozen=True)
class ModelStageOutcome:
    state: ProblemState
    model: MathematicalModel
    plan: AlgorithmPlan
    run: AgentRunResult[MathematicalModel]
    gate: QualityGateResult


@dataclass(frozen=True)
class SolveStageOutcome:
    state: ProblemState
    execution: SolverExecution
    solver_run: SolverRun
    result: ResultRecord
    route: SolverRouteDecision
    strategy: ExecutionStrategyDecision
    code_agent_run: AgentRunResult[GeneratedProgram] | None
    gate: QualityGateResult


@dataclass(frozen=True)
class MathematicalRunOutcome:
    model_stage: ModelStageOutcome
    solve_stage: SolveStageOutcome


class MathematicalWorkflow:
    def __init__(
        self,
        *,
        reasoning_repository: ReasoningRepository,
        repository: MathematicalRepository,
        math_modeler: MathModeler,
        algorithm_selector: AlgorithmSelector,
        solver_router: SolverRouter,
        code_agent: CodeAgent,
        strategy_selector: ExecutionStrategySelector,
        generated_executor: GeneratedProgramExecutor,
        evidence_verifier: EvidenceIntegrityVerifier,
    ) -> None:
        self._reasoning_repository = reasoning_repository
        self._repository = repository
        self._math_modeler = math_modeler
        self._algorithm_selector = algorithm_selector
        self._solver_router = solver_router
        self._code_agent = code_agent
        self._strategy_selector = strategy_selector
        self._generated_executor = generated_executor
        self._evidence_verifier = evidence_verifier

    async def build_model(
        self,
        project_id: UUID,
        *,
        user_guidance: list[str] | None = None,
    ) -> ModelStageOutcome:
        state = self._reasoning_repository.load_current(project_id)
        ensure_transition(state.current_stage, WorkflowStage.MODEL)
        if state.selected_model is None or state.problem_analysis is None:
            raise QualityGateError("MODEL requires accepted problem analysis and model selection")
        model_id, model_version = self._repository.next_model_identity(project_id)
        result = await self._math_modeler.run(
            MathModelerInput(
                assigned_model_id=model_id,
                assigned_version=model_version,
                selected_model=state.selected_model,
                problem_analysis=state.problem_analysis,
                data_understanding=state.data_understanding,
                data_profiles=state.data_profiles,
                user_guidance=user_guidance or [],
            ),
            state,
            TaskProfile(
                task_type=TaskType.MATHEMATICAL_MODELING,
                complexity=4,
                reasoning_requirement=4,
                math_requirement=4,
                review_requirement=4,
                blast_radius=4,
                minimum_level=EscalationLevel.FLAGSHIP_XHIGH,
            ),
        )
        model = self._require_output(state, result)
        gate = model_quality_gate(model, state)
        if gate.status is not QualityGateStatus.PASS:
            rejected = result.model_copy(
                update={
                    "status": AgentRunStatus.RETRY,
                    "errors": [*result.errors, *gate.errors],
                }
            )
            self._reasoning_repository.record_run(project_id, state.problem_id, rejected)
            raise QualityGateError(f"MODEL quality gate rejected output: {', '.join(gate.errors)}")
        plan = self._algorithm_selector.select(model)
        record_id = uuid4()
        next_state = self._model_state(
            state,
            model=model,
            model_record_id=record_id,
            plan=plan,
            run=result,
            gate=gate,
        )
        stored_run = self._repository.persist_model_with_state(
            record_id=record_id,
            model=model,
            state=next_state,
            run=result,
        )
        return ModelStageOutcome(
            state=next_state,
            model=model,
            plan=plan,
            run=stored_run,
            gate=gate,
        )

    async def solve(
        self,
        project_id: UUID,
        *,
        options: SolverOptions | None = None,
        execution_strategy: ExecutionStrategy = ExecutionStrategy.AUTO,
        user_guidance: list[str] | None = None,
    ) -> SolveStageOutcome:
        state = self._reasoning_repository.load_current(project_id)
        ensure_transition(state.current_stage, WorkflowStage.SOLVE)
        if state.mathematical_model is None or state.algorithm_plan is None:
            raise QualityGateError("SOLVE requires a persisted MathematicalModel and AlgorithmPlan")
        model = self._repository.get_model(project_id, state.mathematical_model.record_id)
        selected = self._strategy_selector.select(
            requested=execution_strategy,
            model=model,
            plan=state.algorithm_plan,
            options=options or SolverOptions(),
            solver_router=self._solver_router,
        )
        code_agent_run: AgentRunResult[GeneratedProgram] | None = None
        if selected.decision.selected is ExecutionStrategy.DETERMINISTIC:
            if selected.routed_solver is None:
                raise QualityGateError("deterministic strategy did not resolve a solver")
            execution = selected.routed_solver.solver.solve(model, selected.options)
            route = selected.routed_solver.decision
        else:
            code_agent_run = await self._code_agent.run(
                CodeAgentInput(
                    mathematical_model=model,
                    algorithm_plan=state.algorithm_plan,
                    user_guidance=user_guidance or [],
                ),
                state,
                TaskProfile(
                    task_type=TaskType.CODE_GENERATION,
                    complexity=4,
                    reasoning_requirement=3,
                    math_requirement=4,
                    coding_requirement=4,
                    blast_radius=4,
                    minimum_level=EscalationLevel.FLAGSHIP_HIGH,
                ),
            )
            program = self._require_code_output(state, code_agent_run)
            program = program.model_copy(update={"generator_agent_run_id": code_agent_run.run_id})
            try:
                execution = self._generated_executor.execute(program, model, selected.options)
            except Exception:
                self._reasoning_repository.record_run(
                    state.project_id,
                    state.problem_id,
                    code_agent_run,
                )
                raise
            route = self._solver_router.generated_decision(
                model,
                selected.options,
                solver_name=execution.result.solver_name,
                target=program.solver_target,
            )
        result_id = uuid4()
        solver_run_id = uuid4()
        model_digest = mathematical_model_digest(model)
        result = ResultRecord(
            result_id=result_id,
            project_id=state.project_id,
            problem_id=state.problem_id,
            model_id=model.model_id,
            model_version=model.version,
            model_digest=model_digest,
            solver_run_id=solver_run_id,
            execution_record_id=execution.execution.record.run_id,
            solver=execution.result.solver_name,
            objective=execution.result.objective_value,
            key_outputs=dict(sorted(execution.result.variable_values.items())[:100]),
            status=execution.result.status,
            evidence_refs=[
                f"model:{model.model_id}:v{model.version}",
                f"solver_run:{solver_run_id}",
                f"execution:{execution.execution.record.run_id}",
            ],
            artifact_refs=[item.artifact_id for item in execution.execution.record.artifacts],
        )
        solver_run = SolverRun(
            solver_run_id=solver_run_id,
            project_id=state.project_id,
            problem_id=state.problem_id,
            model_id=model.model_id,
            model_version=model.version,
            model_digest=model_digest,
            generated_program_id=execution.program.program_id,
            execution_origin=execution.program.execution_origin,
            routing_decision_id=route.routing_decision_id,
            routing_decision=route,
            solver=execution.result.solver_name,
            solver_version=execution.result.solver_version,
            options=selected.options,
            start_time=execution.execution.record.start_time,
            end_time=execution.execution.record.end_time,
            status=execution.result.status,
            objective=execution.result.objective_value,
            runtime_seconds=execution.result.runtime_seconds,
            result_ref=result_id,
            execution_ref=execution.execution.record.run_id,
            error=(
                execution.execution.record.error
                if execution.execution.record.error is not None
                else None
            ),
            result=execution.result,
        )
        gate = solve_quality_gate(
            model,
            result,
            solver_run,
            execution.execution.record,
            execution.program,
            self._evidence_verifier,
        )
        next_state = self._solve_state(
            state,
            execution=execution,
            solver_run=solver_run,
            result=result,
            gate=gate,
        )
        self._repository.persist_solve_with_state(
            model_record_id=state.mathematical_model.record_id,
            model=model,
            execution=execution.execution.record,
            artifacts=execution.execution.artifact_records,
            program=execution.program,
            solver_run=solver_run,
            result=result,
            state=next_state,
            code_agent_run=code_agent_run,
        )
        return SolveStageOutcome(
            state=next_state,
            execution=execution,
            solver_run=solver_run,
            result=result,
            route=route,
            strategy=selected.decision,
            code_agent_run=code_agent_run,
            gate=gate,
        )

    async def run(
        self,
        project_id: UUID,
        *,
        user_guidance: list[str] | None = None,
        options: SolverOptions | None = None,
        execution_strategy: ExecutionStrategy = ExecutionStrategy.AUTO,
    ) -> MathematicalRunOutcome:
        model_stage = await self.build_model(project_id, user_guidance=user_guidance)
        solve_stage = await self.solve(
            project_id,
            options=options,
            execution_strategy=execution_strategy,
            user_guidance=user_guidance,
        )
        return MathematicalRunOutcome(model_stage=model_stage, solve_stage=solve_stage)

    def _require_output(
        self,
        state: ProblemState,
        result: AgentRunResult[MathematicalModel],
    ) -> MathematicalModel:
        if result.status is not AgentRunStatus.SUCCEEDED or result.output is None:
            self._reasoning_repository.record_run(
                state.project_id,
                state.problem_id,
                result,
            )
            raise AgentRunError(
                "math_modeler did not produce accepted structured output: "
                f"{'; '.join(result.errors) or result.status.value}"
            )
        return result.output

    def _require_code_output(
        self,
        state: ProblemState,
        result: AgentRunResult[GeneratedProgram],
    ) -> GeneratedProgram:
        if result.status is not AgentRunStatus.SUCCEEDED or result.output is None:
            self._reasoning_repository.record_run(state.project_id, state.problem_id, result)
            raise AgentRunError(
                "CodeAgent did not produce an executable GeneratedProgram: "
                f"{'; '.join(result.errors) or result.status.value}"
            )
        return result.output

    @staticmethod
    def _model_state(
        state: ProblemState,
        *,
        model: MathematicalModel,
        model_record_id: UUID,
        plan: AlgorithmPlan,
        run: AgentRunResult[MathematicalModel],
        gate: QualityGateResult,
    ) -> ProblemState:
        next_version = state.version + 1
        model_ref = MathematicalModelRef(
            record_id=model_record_id,
            model_id=model.model_id,
            version=model.version,
            model_digest=mathematical_model_digest(model),
            source_selected_model_id=model.source_selected_model_id,
            status=model.status,
        )
        variables = [
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
        ]
        parameters = [
            SymbolRef(
                symbol_id=item.parameter_id,
                symbol=item.symbol,
                meaning=item.description,
                unit=item.unit.display if item.unit is not None else None,
            )
            for item in [*model.parameters, *model.constants]
        ]
        equations = [
            EquationRef(
                equation_id=item.equation_id,
                latex=item.latex,
                meaning=item.meaning,
                source_refs=item.source_refs,
                variables=item.symbol_refs,
                parameters=item.parameter_refs,
                units=[
                    value.display for value in (item.unit_lhs, item.unit_rhs) if value is not None
                ],
            )
            for item in model.equations
        ]
        objective = (
            TraceableItem(
                item_id=model.objective.objective_id,
                kind=EvidenceType.DERIVATION,
                statement=model.objective.description,
                source_refs=model.objective.source_refs,
            )
            if model.objective is not None
            else None
        )
        model_constraints = [
            TraceableItem(
                item_id=item.constraint_id,
                kind=EvidenceType.DERIVATION,
                statement=item.description,
                source_refs=item.source_refs,
            )
            for item in model.constraints
        ]
        algorithm = TraceableItem(
            item_id=f"ALGORITHM-{model.model_id}-v{model.version}",
            kind=EvidenceType.DERIVATION,
            statement=plan.reason,
            source_refs=[f"model:{model.model_id}:v{model.version}"],
        )
        history = StageHistoryEntry(
            from_stage=state.current_stage.value,
            to_stage=WorkflowStage.MODEL.value,
            status=WorkflowStatus.SUCCEEDED.value,
            input_version=state.version,
            output_version=next_version,
            updated_by="math_modeler",
            reason="structured mathematical model passed MODEL quality gate",
            agent_run_id=run.run_id,
        )
        payload = state.model_dump()
        payload.update(
            {
                "schema_version": 5,
                "version": next_version,
                "current_stage": WorkflowStage.MODEL,
                "status": WorkflowStatus.SUCCEEDED,
                "mathematical_model": model_ref,
                "algorithm_plan": plan,
                "variables": variables,
                "parameters": parameters,
                "units": {symbol: unit.display for symbol, unit in model.units.items()},
                "equations": equations,
                "objective": objective,
                "model_constraints": model_constraints,
                "algorithm": algorithm,
                "quality_gates": [*state.quality_gates, gate],
                "stage_history": [*state.stage_history, history],
                "updated_by": "math_modeler",
                "update_reason": "structured mathematical model accepted",
                "updated_at": datetime.now(UTC),
            }
        )
        return ProblemState.model_validate(payload)

    @staticmethod
    def _solve_state(
        state: ProblemState,
        *,
        execution: SolverExecution,
        solver_run: SolverRun,
        result: ResultRecord,
        gate: QualityGateResult,
    ) -> ProblemState:
        next_version = state.version + 1
        passed = gate.status is QualityGateStatus.PASS
        target_stage = WorkflowStage.SOLVE if passed else state.current_stage
        code_artifact = next(
            item
            for item in execution.execution.artifact_records
            if item.artifact_id == execution.execution.record.code_artifact_id
        )
        history = StageHistoryEntry(
            from_stage=state.current_stage.value,
            to_stage=target_stage.value,
            status=(WorkflowStatus.SUCCEEDED.value if passed else WorkflowStatus.RETRY.value),
            input_version=state.version,
            output_version=next_version,
            updated_by="solver_router",
            reason=(
                "real solver result passed SOLVE quality gate"
                if passed
                else "solver attempt persisted but SOLVE quality gate requested retry"
            ),
            execution_run_id=execution.execution.record.run_id,
        )
        result_trace = TraceableItem(
            item_id=f"RESULT-{result.result_id}",
            kind=EvidenceType.RESULT,
            statement=f"Persisted canonical solver result {result.result_id}",
            source_refs=result.evidence_refs,
            verification_status=VerificationStatus.UNVERIFIED,
        )
        payload = state.model_dump()
        payload.update(
            {
                "version": next_version,
                "current_stage": target_stage,
                "status": WorkflowStatus.SUCCEEDED if passed else WorkflowStatus.RETRY,
                "verified_result_id": None,
                "code_files": [
                    *state.code_files,
                    ArtifactRef(
                        artifact_id=str(code_artifact.artifact_id),
                        kind=code_artifact.kind.value,
                        path=code_artifact.storage_key,
                        content_hash=code_artifact.sha256,
                        metadata={
                            "generated_program_id": str(execution.program.program_id),
                            "execution_run_id": str(execution.execution.record.run_id),
                        },
                    ),
                ],
                "execution_records": [
                    *state.execution_records,
                    execution.execution.record,
                ],
                "tracked_artifacts": [
                    *state.tracked_artifacts,
                    *execution.execution.artifact_records,
                ],
                "generated_programs": [
                    *state.generated_programs,
                    GeneratedProgramRef(
                        program_id=execution.program.program_id,
                        model_id=execution.program.model_id,
                        model_version=execution.program.model_version,
                        model_digest=execution.program.model_digest,
                        solver_target=execution.program.solver_target,
                        code_hash=execution.program.code_hash,
                        execution_origin=execution.program.execution_origin,
                        status=execution.program.status,
                    ),
                ],
                "solver_runs": [
                    *state.solver_runs,
                    SolverRunRef(
                        solver_run_id=solver_run.solver_run_id,
                        model_id=solver_run.model_id,
                        model_version=solver_run.model_version,
                        model_digest=solver_run.model_digest,
                        solver=solver_run.solver,
                        status=solver_run.status,
                        execution_ref=solver_run.execution_ref,
                        result_ref=solver_run.result_ref,
                    ),
                ],
                "result_records": [
                    *state.result_records,
                    ResultRecordRef(
                        result_id=result.result_id,
                        model_id=result.model_id,
                        model_version=result.model_version,
                        model_digest=result.model_digest,
                        solver_run_id=result.solver_run_id,
                        status=result.status,
                    ),
                ],
                "results": [*state.results, result_trace],
                "quality_gates": [*state.quality_gates, gate],
                "stage_history": [*state.stage_history, history],
                "updated_by": "solver_router",
                "update_reason": history.reason,
                "updated_at": datetime.now(UTC),
            }
        )
        return ProblemState.model_validate(payload)
