from __future__ import annotations

from uuid import UUID, uuid4

from pydantic import BaseModel, ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from mathmodel_ai.agents.base import AgentRunResult
from mathmodel_ai.core.errors import ResourceNotFoundError
from mathmodel_ai.db.models import (
    AgentRunRecord,
    ArtifactRecordModel,
    ExecutionRecordModel,
    GeneratedProgramRecord,
    MathematicalModelRecord,
    Problem,
    ProblemStateRecord,
    ResultRecordModel,
    SolverRunRecord,
)
from mathmodel_ai.db.session import session_scope
from mathmodel_ai.mathematical.digests import mathematical_model_digest
from mathmodel_ai.mathematical.evidence import EvidenceIntegrityVerifier
from mathmodel_ai.schemas.execution import ExecutionRecord
from mathmodel_ai.schemas.files import ArtifactRecord
from mathmodel_ai.schemas.mathematical import MathematicalModel
from mathmodel_ai.schemas.problem_state import ProblemState
from mathmodel_ai.schemas.program import GeneratedProgram
from mathmodel_ai.schemas.results import (
    EvidenceChainReport,
    EvidenceErrorCode,
    EvidenceIssue,
    ResultRecord,
)
from mathmodel_ai.schemas.solver import SolverRun


class MathematicalRepository:
    """Atomic Phase 4 persistence with exact model-version evidence links."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        evidence_verifier: EvidenceIntegrityVerifier | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._evidence_verifier = evidence_verifier or EvidenceIntegrityVerifier()

    def next_model_identity(self, project_id: UUID) -> tuple[UUID, int]:
        with session_scope(self._session_factory) as session:
            self._ensure_project(session, project_id)
            latest = session.scalar(
                select(MathematicalModelRecord)
                .where(MathematicalModelRecord.project_id == project_id)
                .order_by(MathematicalModelRecord.version.desc())
                .limit(1)
            )
            if latest is None:
                return uuid4(), 1
            return latest.model_id, latest.version + 1

    def persist_model_with_state[OutputT: BaseModel](
        self,
        *,
        record_id: UUID,
        model: MathematicalModel,
        state: ProblemState,
        run: AgentRunResult[OutputT],
    ) -> AgentRunResult[OutputT]:
        with session_scope(self._session_factory) as session:
            current = self._lock_current(session, state.project_id)
            self._validate_next_revision(current, state)
            session.add(
                MathematicalModelRecord(
                    id=record_id,
                    project_id=model.project_id,
                    problem_id=model.problem_id,
                    model_id=model.model_id,
                    version=model.version,
                    model_digest=mathematical_model_digest(model),
                    state_version=state.version,
                    selected_model_id=model.source_selected_model_id,
                    status=model.status.value,
                    model_json=model.model_dump(mode="json"),
                )
            )
            stored_run = run.model_copy(update={"output_state_version": state.version})
            session.add(self._agent_model(state.project_id, state.problem_id, stored_run))
            self._replace_state(session, current, state)
        return stored_run

    def persist_solve_with_state(
        self,
        *,
        model_record_id: UUID,
        model: MathematicalModel,
        execution: ExecutionRecord,
        artifacts: list[ArtifactRecord],
        program: GeneratedProgram,
        solver_run: SolverRun,
        result: ResultRecord,
        state: ProblemState,
        code_agent_run: AgentRunResult[GeneratedProgram] | None = None,
    ) -> None:
        with session_scope(self._session_factory) as session:
            current = self._lock_current(session, state.project_id)
            self._validate_next_revision(current, state)
            model_row = session.get(MathematicalModelRecord, model_record_id)
            if model_row is None:
                raise ResourceNotFoundError(f"mathematical model {model_record_id} was not found")
            if (model_row.model_id, model_row.version) != (model.model_id, model.version):
                raise ValueError("solve references a different mathematical model version")
            if code_agent_run is not None:
                stored_code_run = code_agent_run.model_copy(
                    update={"output_state_version": state.version}
                )
                session.add(self._agent_model(state.project_id, state.problem_id, stored_code_run))
            session.add(self._execution_model(execution))
            session.flush()
            session.add_all(self._artifact_model(item) for item in artifacts)
            session.add(
                GeneratedProgramRecord(
                    id=program.program_id,
                    project_id=program.project_id,
                    problem_id=program.problem_id,
                    mathematical_model_record_id=model_record_id,
                    model_id=program.model_id,
                    model_version=program.model_version,
                    model_digest=program.model_digest,
                    solver_target=program.solver_target,
                    code_hash=program.code_hash,
                    execution_origin=program.execution_origin.value,
                    generator_agent_run_id=program.generator_agent_run_id,
                    status=program.status.value,
                    is_mock=program.is_mock,
                    program_json=program.model_dump(mode="json"),
                    created_at=program.created_at,
                )
            )
            session.flush()
            session.add(
                SolverRunRecord(
                    id=solver_run.solver_run_id,
                    project_id=solver_run.project_id,
                    problem_id=solver_run.problem_id,
                    mathematical_model_record_id=model_record_id,
                    model_id=solver_run.model_id,
                    model_version=solver_run.model_version,
                    model_digest=solver_run.model_digest,
                    generated_program_id=solver_run.generated_program_id,
                    execution_record_id=solver_run.execution_ref,
                    execution_origin=solver_run.execution_origin.value,
                    routing_decision_id=solver_run.routing_decision_id,
                    routing_json=solver_run.routing_decision.model_dump(mode="json"),
                    solver=solver_run.solver.value,
                    solver_version=solver_run.solver_version,
                    status=solver_run.status.value,
                    objective=solver_run.objective,
                    runtime_seconds=solver_run.runtime_seconds,
                    options_json=solver_run.options.model_dump(mode="json"),
                    result_json=solver_run.model_dump(mode="json"),
                    error=solver_run.error,
                    start_time=solver_run.start_time,
                    end_time=solver_run.end_time,
                )
            )
            session.flush()
            session.add(
                ResultRecordModel(
                    id=result.result_id,
                    project_id=result.project_id,
                    problem_id=result.problem_id,
                    mathematical_model_record_id=model_record_id,
                    model_id=result.model_id,
                    model_version=result.model_version,
                    model_digest=result.model_digest,
                    solver_run_id=result.solver_run_id,
                    execution_record_id=result.execution_record_id,
                    status=result.status.value,
                    objective=result.objective,
                    record_json=result.model_dump(mode="json"),
                    created_at=result.created_at,
                )
            )
            self._replace_state(session, current, state)

    def get_model(self, project_id: UUID, record_id: UUID | None = None) -> MathematicalModel:
        with session_scope(self._session_factory) as session:
            statement = select(MathematicalModelRecord).where(
                MathematicalModelRecord.project_id == project_id
            )
            if record_id is not None:
                statement = statement.where(MathematicalModelRecord.id == record_id)
            else:
                statement = statement.order_by(MathematicalModelRecord.version.desc()).limit(1)
            row = session.scalar(statement)
            if row is None:
                raise ResourceNotFoundError("mathematical model was not found")
            return MathematicalModel.model_validate(row.model_json)

    def list_solver_runs(self, project_id: UUID) -> list[SolverRun]:
        with session_scope(self._session_factory) as session:
            self._ensure_project(session, project_id)
            rows = list(
                session.scalars(
                    select(SolverRunRecord)
                    .where(SolverRunRecord.project_id == project_id)
                    .order_by(SolverRunRecord.start_time)
                )
            )
            return [self._solver_run_schema(row) for row in rows]

    def list_results(self, project_id: UUID) -> list[ResultRecord]:
        with session_scope(self._session_factory) as session:
            self._ensure_project(session, project_id)
            rows = list(
                session.scalars(
                    select(ResultRecordModel)
                    .where(ResultRecordModel.project_id == project_id)
                    .order_by(ResultRecordModel.created_at)
                )
            )
            return [ResultRecord.model_validate(row.record_json) for row in rows]

    def list_programs(self, project_id: UUID) -> list[GeneratedProgram]:
        with session_scope(self._session_factory) as session:
            self._ensure_project(session, project_id)
            rows = list(
                session.scalars(
                    select(GeneratedProgramRecord)
                    .where(GeneratedProgramRecord.project_id == project_id)
                    .order_by(GeneratedProgramRecord.created_at)
                )
            )
            return [GeneratedProgram.model_validate(row.program_json) for row in rows]

    def verify_result_evidence(self, project_id: UUID, result_id: UUID) -> EvidenceChainReport:
        with session_scope(self._session_factory) as session:
            row = session.scalar(
                select(ResultRecordModel).where(
                    ResultRecordModel.project_id == project_id,
                    ResultRecordModel.id == result_id,
                )
            )
            if row is None:
                raise ResourceNotFoundError(f"result {result_id} was not found")
            solver_row = session.get(SolverRunRecord, row.solver_run_id)
            model_row = session.get(MathematicalModelRecord, row.mathematical_model_record_id)
            execution_row = session.get(ExecutionRecordModel, row.execution_record_id)
            program_row = (
                session.get(GeneratedProgramRecord, solver_row.generated_program_id)
                if solver_row is not None and solver_row.generated_program_id is not None
                else None
            )
            issues: list[EvidenceIssue] = []
            result = self._parse_result(row, issues)
            solver = self._parse_solver_run(solver_row, issues)
            model = self._parse_model(model_row, issues)
            execution = self._parse_execution(execution_row, issues)
            program = self._parse_program(program_row, issues)
            if result is None:
                return EvidenceChainReport(
                    result_id=result_id,
                    valid=False,
                    model_found=model is not None,
                    solver_run_found=solver is not None,
                    execution_found=execution is not None,
                    generated_program_found=(
                        program is not None if program_row is not None else None
                    ),
                    exact_model_version=False,
                    model_digest_match=False,
                    code_hash_match=False if program_row is not None else None,
                    errors=issues,
                )
            self._check_persisted_columns(
                row=row,
                result=result,
                solver_row=solver_row,
                solver=solver,
                model_row=model_row,
                model=model,
                execution_row=execution_row,
                execution=execution,
                program_row=program_row,
                program=program,
                issues=issues,
            )
            return self._evidence_verifier.verify(
                result=result,
                solver_run=solver,
                execution=execution,
                model=model,
                program=program,
                additional_issues=issues,
            )

    @staticmethod
    def _solver_run_schema(row: SolverRunRecord) -> SolverRun:
        return SolverRun.model_validate(row.result_json)

    @staticmethod
    def _parse_result(
        row: ResultRecordModel,
        issues: list[EvidenceIssue],
    ) -> ResultRecord | None:
        try:
            return ResultRecord.model_validate(row.record_json)
        except ValidationError:
            issues.append(
                EvidenceIssue(
                    code=EvidenceErrorCode.BROKEN_RESULT_CHAIN,
                    message="persisted ResultRecord payload is invalid",
                )
            )
            return None

    @staticmethod
    def _parse_solver_run(
        row: SolverRunRecord | None,
        issues: list[EvidenceIssue],
    ) -> SolverRun | None:
        if row is None:
            return None
        try:
            return SolverRun.model_validate(row.result_json)
        except ValidationError:
            issues.append(
                EvidenceIssue(
                    code=EvidenceErrorCode.BROKEN_RESULT_CHAIN,
                    message="persisted SolverRun payload is invalid",
                )
            )
            return None

    @staticmethod
    def _parse_model(
        row: MathematicalModelRecord | None,
        issues: list[EvidenceIssue],
    ) -> MathematicalModel | None:
        if row is None:
            return None
        try:
            return MathematicalModel.model_validate(row.model_json)
        except ValidationError:
            issues.append(
                EvidenceIssue(
                    code=EvidenceErrorCode.BROKEN_RESULT_CHAIN,
                    message="persisted MathematicalModel payload is invalid",
                )
            )
            return None

    @staticmethod
    def _parse_execution(
        row: ExecutionRecordModel | None,
        issues: list[EvidenceIssue],
    ) -> ExecutionRecord | None:
        if row is None:
            return None
        try:
            return ExecutionRecord.model_validate(row.record_json)
        except ValidationError:
            issues.append(
                EvidenceIssue(
                    code=(
                        EvidenceErrorCode.MOCK_EXECUTION
                        if row.is_mock or row.record_json.get("is_mock") is True
                        else EvidenceErrorCode.EXECUTION_FAILED
                    ),
                    message="persisted ExecutionRecord payload is invalid",
                )
            )
            return None

    @staticmethod
    def _parse_program(
        row: GeneratedProgramRecord | None,
        issues: list[EvidenceIssue],
    ) -> GeneratedProgram | None:
        if row is None:
            return None
        try:
            return GeneratedProgram.model_validate(row.program_json)
        except ValidationError:
            issues.append(
                EvidenceIssue(
                    code=EvidenceErrorCode.CODE_HASH_MISMATCH,
                    message="persisted GeneratedProgram payload or aggregate hash is invalid",
                )
            )
            return None

    @staticmethod
    def _check_persisted_columns(
        *,
        row: ResultRecordModel,
        result: ResultRecord,
        solver_row: SolverRunRecord | None,
        solver: SolverRun | None,
        model_row: MathematicalModelRecord | None,
        model: MathematicalModel | None,
        execution_row: ExecutionRecordModel | None,
        execution: ExecutionRecord | None,
        program_row: GeneratedProgramRecord | None,
        program: GeneratedProgram | None,
        issues: list[EvidenceIssue],
    ) -> None:
        if row.objective != result.objective:
            issues.append(
                EvidenceIssue(
                    code=EvidenceErrorCode.OBJECTIVE_MISMATCH,
                    message="result column and payload objective differ",
                )
            )
        if row.status != result.status.value:
            issues.append(
                EvidenceIssue(
                    code=EvidenceErrorCode.STATUS_MISMATCH,
                    message="result column and payload status differ",
                )
            )
        if row.model_digest != result.model_digest:
            issues.append(
                EvidenceIssue(
                    code=EvidenceErrorCode.MODEL_DIGEST_MISMATCH,
                    message="result column and payload digest differ",
                )
            )
        if (row.model_id, row.model_version) != (result.model_id, result.model_version):
            issues.append(
                EvidenceIssue(
                    code=EvidenceErrorCode.MODEL_VERSION_MISMATCH,
                    message="result columns and payload identify different model revisions",
                )
            )
        if solver_row is not None and solver is not None:
            if solver_row.id != solver.solver_run_id or solver_row.id != result.solver_run_id:
                issues.append(
                    EvidenceIssue(
                        code=EvidenceErrorCode.RESULT_REF_MISMATCH,
                        message="solver row identity differs from result chain",
                    )
                )
            if solver_row.objective != solver.objective:
                issues.append(
                    EvidenceIssue(
                        code=EvidenceErrorCode.OBJECTIVE_MISMATCH,
                        message="solver column and payload objective differ",
                    )
                )
            if solver_row.status != solver.status.value:
                issues.append(
                    EvidenceIssue(
                        code=EvidenceErrorCode.STATUS_MISMATCH,
                        message="solver column and payload status differ",
                    )
                )
            if (solver_row.model_id, solver_row.model_version) != (
                solver.model_id,
                solver.model_version,
            ):
                issues.append(
                    EvidenceIssue(
                        code=EvidenceErrorCode.MODEL_VERSION_MISMATCH,
                        message="solver columns and payload identify different model revisions",
                    )
                )
            if solver_row.model_digest != solver.model_digest:
                issues.append(
                    EvidenceIssue(
                        code=EvidenceErrorCode.MODEL_DIGEST_MISMATCH,
                        message="solver column and payload digest differ",
                    )
                )
        if model_row is not None and model is not None:
            if (model_row.model_id, model_row.version) != (model.model_id, model.version):
                issues.append(
                    EvidenceIssue(
                        code=EvidenceErrorCode.MODEL_VERSION_MISMATCH,
                        message="model columns and payload identify different revisions",
                    )
                )
            if model_row.model_digest != mathematical_model_digest(model):
                issues.append(
                    EvidenceIssue(
                        code=EvidenceErrorCode.MODEL_DIGEST_MISMATCH,
                        message="model column differs from canonical digest",
                    )
                )
        if execution_row is not None:
            if execution_row.is_mock:
                issues.append(
                    EvidenceIssue(
                        code=EvidenceErrorCode.MOCK_EXECUTION,
                        message="execution column marks the run as Mock",
                    )
                )
            if execution_row.status != "SUCCEEDED" or execution_row.exit_code != 0:
                issues.append(
                    EvidenceIssue(
                        code=EvidenceErrorCode.EXECUTION_FAILED,
                        message="execution columns do not prove success",
                    )
                )
            if execution is not None and execution_row.code_hash != execution.code_hash:
                issues.append(
                    EvidenceIssue(
                        code=EvidenceErrorCode.CODE_HASH_MISMATCH,
                        message="execution column and payload code hash differ",
                    )
                )
            if execution is not None and (
                execution_row.execution_origin != execution.execution_origin.value
                or execution_row.model_digest != execution.model_digest
                or execution_row.generated_program_id != execution.generated_program_id
            ):
                issues.append(
                    EvidenceIssue(
                        code=EvidenceErrorCode.BROKEN_RESULT_CHAIN,
                        message="execution columns and payload provenance differ",
                    )
                )
            if (
                execution is not None
                and execution_row.executed_bundle_hash != execution.executed_bundle_hash
            ):
                issues.append(
                    EvidenceIssue(
                        code=EvidenceErrorCode.CODE_HASH_MISMATCH,
                        message="execution column and payload bundle hash differ",
                    )
                )
        if program_row is not None and program is not None:
            if program_row.id != program.program_id or (
                program_row.model_id,
                program_row.model_version,
            ) != (program.model_id, program.model_version):
                issues.append(
                    EvidenceIssue(
                        code=EvidenceErrorCode.MODEL_VERSION_MISMATCH,
                        message="program columns and payload identify different model revisions",
                    )
                )
            if program_row.code_hash != program.code_hash:
                issues.append(
                    EvidenceIssue(
                        code=EvidenceErrorCode.CODE_HASH_MISMATCH,
                        message="program column and payload aggregate hash differ",
                    )
                )
            if program_row.model_digest != program.model_digest:
                issues.append(
                    EvidenceIssue(
                        code=EvidenceErrorCode.MODEL_DIGEST_MISMATCH,
                        message="program column and payload digest differ",
                    )
                )

    @staticmethod
    def _ensure_project(session: Session, project_id: UUID) -> None:
        row = session.scalar(select(Problem.id).where(Problem.project_id == project_id))
        if row is None:
            raise ResourceNotFoundError(f"project {project_id} was not found")

    @staticmethod
    def _lock_current(session: Session, project_id: UUID) -> ProblemStateRecord:
        row = session.scalar(
            select(ProblemStateRecord)
            .join(Problem, Problem.id == ProblemStateRecord.problem_id)
            .where(Problem.project_id == project_id, ProblemStateRecord.is_current.is_(True))
            .with_for_update()
        )
        if row is None:
            raise ResourceNotFoundError(f"project {project_id} was not found")
        return row

    @staticmethod
    def _validate_next_revision(current: ProblemStateRecord, state: ProblemState) -> None:
        if current.revision + 1 != state.version:
            raise ValueError(
                f"stale state update: current={current.revision}, requested={state.version}"
            )

    @staticmethod
    def _replace_state(session: Session, current: ProblemStateRecord, state: ProblemState) -> None:
        current.is_current = False
        session.flush()
        session.add(
            ProblemStateRecord(
                problem_id=state.problem_id,
                revision=state.version,
                schema_version=state.schema_version,
                current_stage=state.current_stage.value,
                status=state.status.value,
                is_current=True,
                state_json=state.model_dump(mode="json"),
                updated_by=state.updated_by,
                update_reason=state.update_reason,
            )
        )

    @staticmethod
    def _agent_model[OutputT: BaseModel](
        project_id: UUID,
        problem_id: UUID,
        run: AgentRunResult[OutputT],
    ) -> AgentRunRecord:
        return AgentRunRecord(
            id=run.run_id,
            project_id=project_id,
            problem_id=problem_id,
            agent_name=run.agent_name,
            input_state_version=run.input_state_version,
            output_state_version=run.output_state_version,
            provider=run.provider,
            model=run.model,
            reasoning_level=run.reasoning,
            prompt_version=run.prompt_version,
            started_at=run.started_at,
            ended_at=run.ended_at,
            latency_ms=run.latency_ms,
            token_usage=run.token_usage.model_dump(mode="json"),
            status=run.status.value,
            retry_count=max(run.attempts - 1, 0),
            error="\n".join(run.errors) or None,
            is_mock=run.is_mock,
            route_json=[route.model_dump(mode="json") for route in run.routes],
        )

    @staticmethod
    def _artifact_model(item: ArtifactRecord) -> ArtifactRecordModel:
        return ArtifactRecordModel(
            id=item.artifact_id,
            project_id=item.project_id,
            problem_id=item.problem_id,
            source_file_id=item.source_file_id,
            execution_run_id=item.execution_run_id,
            kind=item.kind.value,
            name=item.name,
            mime_type=item.mime_type,
            size_bytes=item.size_bytes,
            sha256=item.sha256,
            storage_key=item.storage_key,
            artifact_metadata=item.metadata,
            created_at=item.created_at,
        )

    @staticmethod
    def _execution_model(item: ExecutionRecord) -> ExecutionRecordModel:
        return ExecutionRecordModel(
            id=item.run_id,
            project_id=item.project_id,
            problem_id=item.problem_id,
            code_hash=item.code_hash,
            executed_bundle_hash=item.executed_bundle_hash,
            execution_origin=item.execution_origin.value,
            model_digest=item.model_digest,
            generated_program_id=item.generated_program_id,
            image=item.image,
            image_id=item.image_id,
            start_time=item.start_time,
            end_time=item.end_time,
            runtime_seconds=item.runtime_seconds,
            status=item.status.value,
            exit_code=item.exit_code,
            is_mock=item.is_mock,
            record_json=item.model_dump(mode="json"),
        )
