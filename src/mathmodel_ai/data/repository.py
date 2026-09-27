from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from mathmodel_ai.core.errors import ResourceNotFoundError
from mathmodel_ai.db.models import (
    ArtifactRecordModel,
    DataProfileRecordModel,
    DatasetRecordModel,
    ExecutionRecordModel,
    FileRecordModel,
    Problem,
    ProblemStateRecord,
    Project,
)
from mathmodel_ai.db.session import session_scope
from mathmodel_ai.files.pipeline import ProcessedFile
from mathmodel_ai.schemas.data import DataProfile, DatasetRecord
from mathmodel_ai.schemas.execution import ExecutionRecord
from mathmodel_ai.schemas.files import ArtifactRecord, RegisteredFile
from mathmodel_ai.schemas.problem_state import ProblemState


class DataRepository:
    """Atomic metadata/state persistence; file bytes stay in the file store."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def persist_processed_with_state(self, processed: ProcessedFile, state: ProblemState) -> None:
        with session_scope(self._session_factory) as session:
            current = self._lock_current(session, state.project_id)
            self._validate_next_revision(current, state)
            file = processed.parsed_file.file
            session.add(self._file_model(file))
            session.flush()
            session.add_all(self._artifact_model(item) for item in processed.artifacts)
            session.add_all(self._dataset_model(item) for item in processed.datasets)
            session.flush()
            if processed.profile_bundle is not None:
                session.add_all(
                    self._profile_model(state.project_id, state.problem_id, item)
                    for item in processed.profile_bundle.profiles
                )
            self._replace_state(session, current, state)

    def persist_execution_with_state(
        self,
        record: ExecutionRecord,
        artifacts: list[ArtifactRecord],
        state: ProblemState,
    ) -> None:
        with session_scope(self._session_factory) as session:
            current = self._lock_current(session, state.project_id)
            self._validate_next_revision(current, state)
            session.add(self._execution_model(record))
            session.flush()
            session.add_all(self._artifact_model(item) for item in artifacts)
            self._replace_state(session, current, state)

    def persist_auxiliary_execution(
        self, record: ExecutionRecord, artifacts: list[ArtifactRecord]
    ) -> None:
        """Persist a verification-side run without inventing a state transition."""
        artifact_by_id = {item.artifact_id: item for item in artifacts}
        if (
            len(artifact_by_id) != len(artifacts)
            or record.code_artifact_id not in artifact_by_id
            or artifact_by_id[record.code_artifact_id].sha256 != record.code_hash
            or any(
                item.project_id != record.project_id
                or item.problem_id != record.problem_id
                or item.execution_run_id != record.run_id
                for item in artifacts
            )
            or any(
                (persisted := artifact_by_id.get(item.artifact_id)) is None
                or persisted.sha256 != item.sha256
                or persisted.storage_key != item.storage_key
                for item in record.artifacts
            )
        ):
            raise ValueError("auxiliary execution artifacts do not match the execution record")
        with session_scope(self._session_factory) as session:
            self._ensure_project(session, record.project_id)
            session.add(self._execution_model(record))
            session.flush()
            session.add_all(self._artifact_model(item) for item in artifacts)

    def list_files(self, project_id: UUID) -> list[RegisteredFile]:
        with session_scope(self._session_factory) as session:
            self._ensure_project(session, project_id)
            rows = list(
                session.scalars(
                    select(FileRecordModel)
                    .where(FileRecordModel.project_id == project_id)
                    .order_by(FileRecordModel.created_at)
                )
            )
            return [self._file_schema(row) for row in rows]

    def get_file(self, project_id: UUID, file_id: UUID) -> RegisteredFile:
        with session_scope(self._session_factory) as session:
            row = session.scalar(
                select(FileRecordModel).where(
                    FileRecordModel.project_id == project_id,
                    FileRecordModel.id == file_id,
                )
            )
            if row is None:
                raise ResourceNotFoundError(f"file {file_id} was not found")
            return self._file_schema(row)

    def list_datasets(self, project_id: UUID) -> list[DatasetRecord]:
        with session_scope(self._session_factory) as session:
            self._ensure_project(session, project_id)
            rows = list(
                session.scalars(
                    select(DatasetRecordModel)
                    .where(DatasetRecordModel.project_id == project_id)
                    .order_by(DatasetRecordModel.created_at)
                )
            )
            return [self._dataset_schema(row) for row in rows]

    def list_profiles(self, project_id: UUID) -> list[DataProfile]:
        with session_scope(self._session_factory) as session:
            self._ensure_project(session, project_id)
            rows = list(
                session.scalars(
                    select(DataProfileRecordModel)
                    .where(DataProfileRecordModel.project_id == project_id)
                    .order_by(DataProfileRecordModel.generated_at)
                )
            )
            return [DataProfile.model_validate(row.profile_json) for row in rows]

    def list_artifacts(self, project_id: UUID) -> list[ArtifactRecord]:
        with session_scope(self._session_factory) as session:
            self._ensure_project(session, project_id)
            rows = list(
                session.scalars(
                    select(ArtifactRecordModel)
                    .where(ArtifactRecordModel.project_id == project_id)
                    .order_by(ArtifactRecordModel.created_at)
                )
            )
            return [self._artifact_schema(row) for row in rows]

    def list_executions(self, project_id: UUID) -> list[ExecutionRecord]:
        with session_scope(self._session_factory) as session:
            self._ensure_project(session, project_id)
            rows = list(
                session.scalars(
                    select(ExecutionRecordModel)
                    .where(ExecutionRecordModel.project_id == project_id)
                    .order_by(ExecutionRecordModel.start_time)
                )
            )
            return [ExecutionRecord.model_validate(row.record_json) for row in rows]

    @staticmethod
    def _ensure_project(session: Session, project_id: UUID) -> None:
        if session.scalar(select(Project.id).where(Project.id == project_id)) is None:
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
    def _file_model(item: RegisteredFile) -> FileRecordModel:
        return FileRecordModel(
            id=item.file_id,
            project_id=item.project_id,
            problem_id=item.problem_id,
            original_name=item.original_name,
            safe_name=item.safe_name,
            extension=item.extension,
            kind=item.kind.value,
            declared_mime_type=item.declared_mime_type,
            detected_mime_type=item.detected_mime_type,
            size_bytes=item.size_bytes,
            sha256=item.sha256,
            storage_key=item.storage_key,
            status=item.status.value,
            validation_warnings=item.validation_warnings,
            parser_name=item.parser_name,
            parser_version=item.parser_version,
            error=item.error,
            created_at=item.created_at,
        )

    @staticmethod
    def _file_schema(row: FileRecordModel) -> RegisteredFile:
        return RegisteredFile(
            file_id=row.id,
            project_id=row.project_id,
            problem_id=row.problem_id,
            original_name=row.original_name,
            safe_name=row.safe_name,
            extension=row.extension,
            kind=row.kind,
            declared_mime_type=row.declared_mime_type,
            detected_mime_type=row.detected_mime_type,
            size_bytes=row.size_bytes,
            sha256=row.sha256,
            storage_key=row.storage_key,
            status=row.status,
            validation_warnings=row.validation_warnings,
            parser_name=row.parser_name,
            parser_version=row.parser_version,
            error=row.error,
            created_at=row.created_at,
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
    def _artifact_schema(row: ArtifactRecordModel) -> ArtifactRecord:
        return ArtifactRecord(
            artifact_id=row.id,
            project_id=row.project_id,
            problem_id=row.problem_id,
            source_file_id=row.source_file_id,
            execution_run_id=row.execution_run_id,
            kind=row.kind,
            name=row.name,
            mime_type=row.mime_type,
            size_bytes=row.size_bytes,
            sha256=row.sha256,
            storage_key=row.storage_key,
            metadata=row.artifact_metadata,
            created_at=row.created_at,
        )

    @staticmethod
    def _dataset_model(item: DatasetRecord) -> DatasetRecordModel:
        return DatasetRecordModel(
            id=item.dataset_id,
            project_id=item.project_id,
            problem_id=item.problem_id,
            source_file_id=item.source_file_id,
            name=item.name,
            sheet_name=item.sheet_name,
            layer=item.layer.value,
            row_count=item.row_count,
            column_count=item.column_count,
            columns_json=item.columns,
            created_at=item.created_at,
        )

    @staticmethod
    def _dataset_schema(row: DatasetRecordModel) -> DatasetRecord:
        return DatasetRecord(
            dataset_id=row.id,
            project_id=row.project_id,
            problem_id=row.problem_id,
            source_file_id=row.source_file_id,
            name=row.name,
            sheet_name=row.sheet_name,
            layer=row.layer,
            row_count=row.row_count,
            column_count=row.column_count,
            columns=row.columns_json,
            created_at=row.created_at,
        )

    @staticmethod
    def _profile_model(
        project_id: UUID, problem_id: UUID, item: DataProfile
    ) -> DataProfileRecordModel:
        return DataProfileRecordModel(
            id=item.profile_id,
            project_id=project_id,
            problem_id=problem_id,
            dataset_id=item.dataset_id,
            source_file_id=item.source_file_id,
            profile_json=item.model_dump(mode="json"),
            generated_at=item.generated_at,
        )

    @staticmethod
    def _execution_model(item: ExecutionRecord) -> ExecutionRecordModel:
        return ExecutionRecordModel(
            id=item.run_id,
            project_id=item.project_id,
            problem_id=item.problem_id,
            code_hash=item.code_hash,
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
