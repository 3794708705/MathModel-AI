from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, ValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from mathmodel_ai.agents.base import AgentRunResult
from mathmodel_ai.core.errors import ResourceNotFoundError
from mathmodel_ai.db.models import (
    AgentRunRecord,
    ArtifactRecordModel,
    ExecutionRecordModel,
    MathematicalModelRecord,
    Problem,
    ProblemStateRecord,
    RedTeamReportRecord,
    RepairCycleRecordModel,
    RobustnessRunRecord,
    SensitivityRunRecord,
    ValidationRunRecord,
    VerificationExperimentRecord,
)
from mathmodel_ai.db.session import session_scope
from mathmodel_ai.mathematical.digests import mathematical_model_digest
from mathmodel_ai.sandbox.executor import SandboxExecution
from mathmodel_ai.schemas.execution import ExecutionRecord
from mathmodel_ai.schemas.files import ArtifactRecord
from mathmodel_ai.schemas.mathematical import MathematicalModel
from mathmodel_ai.schemas.problem_state import ProblemState
from mathmodel_ai.schemas.program import GeneratedProgram
from mathmodel_ai.schemas.solver import SolverResult
from mathmodel_ai.schemas.verification import (
    ExperimentRun,
    RedTeamDraft,
    RedTeamReport,
    RepairCycleRecord,
    RobustnessReport,
    SensitivityReport,
    ValidationReport,
)
from mathmodel_ai.solvers.base import SolverExecution
from mathmodel_ai.verification.experiment_integrity import ExperimentIntegrityVerifier
from mathmodel_ai.verification.experiments import ExperimentOutcome


class VerificationRepository:
    """Atomic Phase 5 records and immutable state revisions."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def persist_validation(self, report: ValidationReport, state: ProblemState) -> None:
        with session_scope(self._session_factory) as session:
            current = self._lock_current(session, state.project_id)
            self._validate_next_revision(current, state)
            model_record_id = self._model_record_id(
                session,
                state.project_id,
                report.model_id,
                report.model_version,
            )
            session.add(
                ValidationRunRecord(
                    id=report.validation_id,
                    project_id=report.project_id,
                    problem_id=report.problem_id,
                    mathematical_model_record_id=model_record_id,
                    model_id=report.model_id,
                    model_version=report.model_version,
                    model_digest=report.model_digest,
                    result_id=report.result_id,
                    solver_run_id=report.solver_run_id,
                    execution_record_id=report.execution_record_id,
                    status=report.status.value,
                    report_json=report.model_dump(mode="json"),
                    created_at=report.created_at,
                )
            )
            self._replace_state(session, current, state)

    def persist_sensitivity(
        self,
        report: SensitivityReport,
        outcomes: list[ExperimentOutcome],
        state: ProblemState,
    ) -> None:
        with session_scope(self._session_factory) as session:
            current = self._lock_current(session, state.project_id)
            self._validate_next_revision(current, state)
            model_record_id = self._model_record_id(
                session,
                state.project_id,
                report.model_id,
                report.model_version,
            )
            session.add(
                SensitivityRunRecord(
                    id=report.sensitivity_id,
                    project_id=report.project_id,
                    problem_id=report.problem_id,
                    mathematical_model_record_id=model_record_id,
                    model_id=report.model_id,
                    model_version=report.model_version,
                    model_digest=report.model_digest,
                    result_id=report.result_id,
                    validation_run_id=report.validation_id,
                    status=report.status.value,
                    config_json=report.config.model_dump(mode="json"),
                    report_json=report.model_dump(mode="json"),
                    created_at=report.created_at,
                )
            )
            session.flush()
            self._persist_experiments(
                session,
                model_record_id=model_record_id,
                sensitivity_run_id=report.sensitivity_id,
                robustness_run_id=None,
                outcomes=outcomes,
            )
            self._replace_state(session, current, state)

    def persist_robustness(
        self,
        report: RobustnessReport,
        outcomes: list[ExperimentOutcome],
        state: ProblemState,
    ) -> None:
        with session_scope(self._session_factory) as session:
            current = self._lock_current(session, state.project_id)
            self._validate_next_revision(current, state)
            model_record_id = self._model_record_id(
                session,
                state.project_id,
                report.model_id,
                report.model_version,
            )
            session.add(
                RobustnessRunRecord(
                    id=report.robustness_id,
                    project_id=report.project_id,
                    problem_id=report.problem_id,
                    mathematical_model_record_id=model_record_id,
                    model_id=report.model_id,
                    model_version=report.model_version,
                    model_digest=report.model_digest,
                    result_id=report.result_id,
                    validation_run_id=report.validation_id,
                    sensitivity_run_id=report.sensitivity_id,
                    method=report.method.value,
                    status=report.status.value,
                    config_json=report.config.model_dump(mode="json"),
                    report_json=report.model_dump(mode="json"),
                    created_at=report.created_at,
                )
            )
            session.flush()
            self._persist_experiments(
                session,
                model_record_id=model_record_id,
                sensitivity_run_id=None,
                robustness_run_id=report.robustness_id,
                outcomes=outcomes,
            )
            self._replace_state(session, current, state)

    def persist_red_team(
        self,
        report: RedTeamReport,
        state: ProblemState,
        run: AgentRunResult[RedTeamDraft],
    ) -> AgentRunResult[RedTeamDraft]:
        with session_scope(self._session_factory) as session:
            current = self._lock_current(session, state.project_id)
            self._validate_next_revision(current, state)
            model_record_id = self._model_record_id(
                session,
                state.project_id,
                report.model_id,
                report.model_version,
            )
            stored_run = run.model_copy(update={"output_state_version": state.version})
            session.add(self._agent_model(state.project_id, state.problem_id, stored_run))
            session.flush()
            session.add(
                RedTeamReportRecord(
                    id=report.report_id,
                    project_id=report.project_id,
                    problem_id=report.problem_id,
                    mathematical_model_record_id=model_record_id,
                    model_id=report.model_id,
                    model_version=report.model_version,
                    model_digest=report.model_digest,
                    result_id=report.result_id,
                    validation_run_id=report.validation_id,
                    sensitivity_run_id=report.sensitivity_id,
                    robustness_run_id=report.robustness_id,
                    reviewer_agent_run_id=report.reviewer_agent_run_id,
                    status=report.status.value,
                    critical_count=report.critical_count,
                    major_count=report.major_count,
                    minor_count=report.minor_count,
                    review_is_mock=report.review_is_mock,
                    report_json=report.model_dump(mode="json"),
                    created_at=report.created_at,
                )
            )
            self._replace_state(session, current, state)
        return stored_run

    def persist_repair[OutputT: BaseModel](
        self,
        *,
        source_model_record_id: UUID,
        target_model_record_id: UUID,
        revised_model: MathematicalModel,
        repair: RepairCycleRecord,
        state: ProblemState,
        run: AgentRunResult[OutputT],
    ) -> AgentRunResult[OutputT]:
        with session_scope(self._session_factory) as session:
            current = self._lock_current(session, state.project_id)
            self._validate_next_revision(current, state)
            source = session.get(MathematicalModelRecord, source_model_record_id)
            if source is None:
                raise ResourceNotFoundError("source mathematical model was not found")
            if (source.model_id, source.version) != (
                revised_model.model_id,
                revised_model.version - 1,
            ):
                raise ValueError("repair source does not precede the revised model version")
            stored_run = run.model_copy(update={"output_state_version": state.version})
            session.add(self._agent_model(state.project_id, state.problem_id, stored_run))
            session.flush()
            session.add(
                MathematicalModelRecord(
                    id=target_model_record_id,
                    project_id=revised_model.project_id,
                    problem_id=revised_model.problem_id,
                    model_id=revised_model.model_id,
                    version=revised_model.version,
                    model_digest=mathematical_model_digest(revised_model),
                    state_version=state.version,
                    selected_model_id=revised_model.source_selected_model_id,
                    status=revised_model.status.value,
                    model_json=revised_model.model_dump(mode="json"),
                )
            )
            session.flush()
            session.add(
                RepairCycleRecordModel(
                    id=repair.repair_id,
                    project_id=repair.project_id,
                    problem_id=repair.problem_id,
                    stable_model_id=repair.stable_model_id,
                    source_model_record_id=source_model_record_id,
                    target_model_record_id=target_model_record_id,
                    red_team_report_id=repair.red_team_report_id,
                    agent_run_id=repair.agent_run_id,
                    repair_cycle=repair.repair_cycle,
                    status=repair.status.value,
                    is_mock=repair.is_mock,
                    record_json=repair.model_dump(mode="json"),
                    created_at=repair.created_at,
                )
            )
            self._replace_state(session, current, state)
        return stored_run

    def persist_repair_attempt[OutputT: BaseModel](
        self,
        *,
        source_model_record_id: UUID,
        repair: RepairCycleRecord,
        state: ProblemState,
        run: AgentRunResult[OutputT],
    ) -> AgentRunResult[OutputT]:
        with session_scope(self._session_factory) as session:
            current = self._lock_current(session, state.project_id)
            self._validate_next_revision(current, state)
            stored_run = run.model_copy(update={"output_state_version": state.version})
            session.add(self._agent_model(state.project_id, state.problem_id, stored_run))
            session.flush()
            session.add(
                RepairCycleRecordModel(
                    id=repair.repair_id,
                    project_id=repair.project_id,
                    problem_id=repair.problem_id,
                    stable_model_id=repair.stable_model_id,
                    source_model_record_id=source_model_record_id,
                    target_model_record_id=None,
                    red_team_report_id=repair.red_team_report_id,
                    agent_run_id=repair.agent_run_id,
                    repair_cycle=repair.repair_cycle,
                    status=repair.status.value,
                    is_mock=repair.is_mock,
                    record_json=repair.model_dump(mode="json"),
                    created_at=repair.created_at,
                )
            )
            self._replace_state(session, current, state)
        return stored_run

    def persist_state_only(self, state: ProblemState) -> None:
        with session_scope(self._session_factory) as session:
            current = self._lock_current(session, state.project_id)
            self._validate_next_revision(current, state)
            self._replace_state(session, current, state)

    def get_validation(self, project_id: UUID, report_id: UUID | None = None) -> ValidationReport:
        return self._get_report(
            ValidationRunRecord,
            ValidationReport,
            project_id,
            report_id,
        )

    def get_sensitivity(
        self,
        project_id: UUID,
        report_id: UUID | None = None,
    ) -> SensitivityReport:
        return self._get_report(
            SensitivityRunRecord,
            SensitivityReport,
            project_id,
            report_id,
        )

    def get_robustness(
        self,
        project_id: UUID,
        report_id: UUID | None = None,
    ) -> RobustnessReport:
        return self._get_report(
            RobustnessRunRecord,
            RobustnessReport,
            project_id,
            report_id,
        )

    def get_red_team(self, project_id: UUID, report_id: UUID | None = None) -> RedTeamReport:
        return self._get_report(
            RedTeamReportRecord,
            RedTeamReport,
            project_id,
            report_id,
        )

    def repair_cycle_count(self, model_id: UUID) -> int:
        with session_scope(self._session_factory) as session:
            return int(
                session.scalar(
                    select(func.count())
                    .select_from(RepairCycleRecordModel)
                    .where(RepairCycleRecordModel.stable_model_id == model_id)
                )
                or 0
            )

    def list_repairs(self, project_id: UUID) -> list[RepairCycleRecord]:
        with session_scope(self._session_factory) as session:
            rows = list(
                session.scalars(
                    select(RepairCycleRecordModel)
                    .where(RepairCycleRecordModel.project_id == project_id)
                    .order_by(RepairCycleRecordModel.repair_cycle)
                )
            )
            return [RepairCycleRecord.model_validate(row.record_json) for row in rows]

    def audit_experiment_report(
        self,
        *,
        project_id: UUID,
        model: MathematicalModel,
        experiments: list[ExperimentRun],
        verifier: ExperimentIntegrityVerifier,
        sensitivity_run_id: UUID | None = None,
        robustness_run_id: UUID | None = None,
    ) -> list[str]:
        if (sensitivity_run_id is None) == (robustness_run_id is None):
            raise ValueError("exactly one experiment report owner must be supplied")
        errors: list[str] = []
        expected = {item.experiment_id: item for item in experiments}
        with session_scope(self._session_factory) as session:
            statement = select(VerificationExperimentRecord).where(
                VerificationExperimentRecord.project_id == project_id
            )
            if sensitivity_run_id is not None:
                statement = statement.where(
                    VerificationExperimentRecord.sensitivity_run_id == sensitivity_run_id
                )
            else:
                statement = statement.where(
                    VerificationExperimentRecord.robustness_run_id == robustness_run_id
                )
            rows = list(session.scalars(statement))
            row_ids = {row.id for row in rows}
            if row_ids != set(expected):
                errors.append("PERSISTED_EXPERIMENT_SET_MISMATCH")
            for row in rows:
                report_record = expected.get(row.id)
                if report_record is None:
                    errors.append(f"UNEXPECTED_PERSISTED_EXPERIMENT:{row.id}")
                    continue
                try:
                    persisted_record = ExperimentRun.model_validate(row.record_json)
                except ValidationError:
                    errors.append(f"INVALID_PERSISTED_EXPERIMENT:{row.id}")
                    continue
                if persisted_record != report_record:
                    errors.append(f"EXPERIMENT_REPORT_ROW_MISMATCH:{row.id}")
                if (
                    row.base_model_digest != report_record.base_model_digest
                    or row.scenario_model_digest != report_record.scenario_model_digest
                    or row.experiment_type != report_record.experiment_type
                    or row.status != report_record.status.value
                    or row.execution_record_id != report_record.execution_record_id
                    or row.solver
                    != (report_record.solver.value if report_record.solver is not None else None)
                    or row.solver_status
                    != (
                        report_record.solver_status.value
                        if report_record.solver_status is not None
                        else None
                    )
                    or row.objective != report_record.objective_value
                    or row.perturbation_json
                    != [item.model_dump(mode="json") for item in report_record.perturbations]
                ):
                    errors.append(f"EXPERIMENT_COLUMN_MISMATCH:{row.id}")
                if row.execution_record_id is None:
                    errors.append(f"EXPERIMENT_EXECUTION_MISSING:{row.id}")
                    continue
                execution_row = session.get(ExecutionRecordModel, row.execution_record_id)
                if (
                    execution_row is None
                    or row.program_json is None
                    or row.solver_result_json is None
                ):
                    errors.append(f"EXPERIMENT_EVIDENCE_MISSING:{row.id}")
                    continue
                try:
                    execution_record = ExecutionRecord.model_validate(execution_row.record_json)
                    program = GeneratedProgram.model_validate(row.program_json)
                    solver_result = SolverResult.model_validate(row.solver_result_json)
                except ValidationError:
                    errors.append(f"EXPERIMENT_EVIDENCE_INVALID:{row.id}")
                    continue
                if (
                    execution_row.project_id != execution_record.project_id
                    or execution_row.problem_id != execution_record.problem_id
                    or execution_row.code_hash != execution_record.code_hash
                    or execution_row.executed_bundle_hash != execution_record.executed_bundle_hash
                    or execution_row.execution_origin != execution_record.execution_origin.value
                    or execution_row.model_digest != execution_record.model_digest
                    or execution_row.generated_program_id != execution_record.generated_program_id
                    or execution_row.status != execution_record.status.value
                    or execution_row.exit_code != execution_record.exit_code
                    or execution_row.is_mock != execution_record.is_mock
                ):
                    errors.append(f"EXPERIMENT_EXECUTION_COLUMN_MISMATCH:{row.id}")
                model_row = session.get(
                    MathematicalModelRecord,
                    row.mathematical_model_record_id,
                )
                if (
                    model_row is None
                    or model_row.project_id != model.project_id
                    or model_row.problem_id != model.problem_id
                    or model_row.model_id != model.model_id
                    or model_row.version != model.version
                    or model_row.model_digest != mathematical_model_digest(model)
                ):
                    errors.append(f"EXPERIMENT_MODEL_LINK_MISMATCH:{row.id}")
                audit = verifier.audit(
                    base_model=model,
                    record=report_record,
                    execution=SolverExecution(
                        result=solver_result,
                        execution=SandboxExecution(
                            record=execution_record,
                            artifact_records=[],
                        ),
                        program=program,
                    ),
                )
                errors.extend(f"{row.id}:{item}" for item in audit.errors)
        return list(dict.fromkeys(errors))

    def _persist_experiments(
        self,
        session: Session,
        *,
        model_record_id: UUID,
        sensitivity_run_id: UUID | None,
        robustness_run_id: UUID | None,
        outcomes: list[ExperimentOutcome],
    ) -> None:
        for outcome in outcomes:
            execution = outcome.execution
            if execution is not None:
                session.add(self._execution_model(execution.execution.record))
                session.flush()
                session.add_all(
                    self._artifact_model(item) for item in execution.execution.artifact_records
                )
            record = outcome.record
            session.add(
                VerificationExperimentRecord(
                    id=record.experiment_id,
                    project_id=(
                        execution.execution.record.project_id
                        if execution is not None
                        else self._project_id_for_model(session, model_record_id)
                    ),
                    problem_id=(
                        execution.execution.record.problem_id
                        if execution is not None
                        else self._problem_id_for_model(session, model_record_id)
                    ),
                    mathematical_model_record_id=model_record_id,
                    sensitivity_run_id=sensitivity_run_id,
                    robustness_run_id=robustness_run_id,
                    execution_record_id=record.execution_record_id,
                    experiment_type=record.experiment_type,
                    base_model_digest=record.base_model_digest,
                    scenario_model_digest=record.scenario_model_digest,
                    solver=record.solver.value if record.solver is not None else None,
                    solver_status=(
                        record.solver_status.value if record.solver_status is not None else None
                    ),
                    status=record.status.value,
                    objective=record.objective_value,
                    perturbation_json=[
                        item.model_dump(mode="json") for item in record.perturbations
                    ],
                    program_json=(
                        execution.program.model_dump(mode="json") if execution is not None else None
                    ),
                    solver_result_json=(
                        execution.result.model_dump(mode="json") if execution is not None else None
                    ),
                    record_json=record.model_dump(mode="json"),
                )
            )

    def _get_report[SchemaT: BaseModel](
        self,
        record_type: Any,
        schema_type: type[SchemaT],
        project_id: UUID,
        report_id: UUID | None,
    ) -> SchemaT:
        id_column = record_type.id
        project_column = record_type.project_id
        created_column = record_type.created_at
        with session_scope(self._session_factory) as session:
            statement = select(record_type).where(project_column == project_id)
            if report_id is not None:
                statement = statement.where(id_column == report_id)
            else:
                statement = statement.order_by(created_column.desc()).limit(1)
            row = session.scalar(statement)
            if row is None:
                raise ResourceNotFoundError(f"{schema_type.__name__} was not found")
            return schema_type.model_validate(row.report_json)

    @staticmethod
    def _model_record_id(
        session: Session,
        project_id: UUID,
        model_id: UUID,
        version: int,
    ) -> UUID:
        record_id = session.scalar(
            select(MathematicalModelRecord.id).where(
                MathematicalModelRecord.project_id == project_id,
                MathematicalModelRecord.model_id == model_id,
                MathematicalModelRecord.version == version,
            )
        )
        if record_id is None:
            raise ResourceNotFoundError("mathematical model revision was not found")
        return record_id

    @staticmethod
    def _project_id_for_model(session: Session, record_id: UUID) -> UUID:
        value = session.scalar(
            select(MathematicalModelRecord.project_id).where(
                MathematicalModelRecord.id == record_id
            )
        )
        if value is None:
            raise ResourceNotFoundError("mathematical model revision was not found")
        return value

    @staticmethod
    def _problem_id_for_model(session: Session, record_id: UUID) -> UUID:
        value = session.scalar(
            select(MathematicalModelRecord.problem_id).where(
                MathematicalModelRecord.id == record_id
            )
        )
        if value is None:
            raise ResourceNotFoundError("mathematical model revision was not found")
        return value

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
