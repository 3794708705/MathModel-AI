from dataclasses import dataclass
from datetime import UTC, datetime
from typing import BinaryIO
from uuid import UUID

from mathmodel_ai.agents import AgentRunResult, AgentRunStatus, DataAgent
from mathmodel_ai.core.errors import AgentRunError, QualityGateError
from mathmodel_ai.data.quality_gates import (
    data_quality_gate,
    execution_quality_gate,
    files_quality_gate,
)
from mathmodel_ai.data.repository import DataRepository
from mathmodel_ai.files.pipeline import FilePipeline, ProcessedFile
from mathmodel_ai.reasoning.repository import ReasoningRepository
from mathmodel_ai.routing.schemas import TaskProfile, TaskType
from mathmodel_ai.sandbox.executor import SandboxExecution, SandboxExecutor
from mathmodel_ai.schemas.data import DataAgentInput, DataUnderstanding
from mathmodel_ai.schemas.files import ArtifactKind
from mathmodel_ai.schemas.problem_state import ArtifactRef, ProblemState
from mathmodel_ai.schemas.quality import (
    DataStageHistoryEntry,
    DataWorkflowStage,
    QualityGateResult,
    QualityGateStatus,
)


@dataclass(frozen=True)
class FileStageOutcome:
    state: ProblemState
    processed: ProcessedFile
    gate: QualityGateResult


@dataclass(frozen=True)
class DataStageOutcome:
    state: ProblemState
    output: DataUnderstanding
    run: AgentRunResult[DataUnderstanding]
    gate: QualityGateResult


@dataclass(frozen=True)
class ExecutionStageOutcome:
    state: ProblemState
    execution: SandboxExecution
    gate: QualityGateResult


class DataExecutionWorkflow:
    """Traceable FILES -> DATA -> EXECUTION sub-workflow for Phase 3."""

    def __init__(
        self,
        *,
        reasoning_repository: ReasoningRepository,
        data_repository: DataRepository,
        file_pipeline: FilePipeline,
        data_agent: DataAgent,
        sandbox: SandboxExecutor,
    ) -> None:
        self._reasoning_repository = reasoning_repository
        self._data_repository = data_repository
        self._file_pipeline = file_pipeline
        self._data_agent = data_agent
        self._sandbox = sandbox

    def ingest_file(
        self,
        project_id: UUID,
        stream: BinaryIO,
        *,
        original_name: str,
        declared_mime_type: str | None,
    ) -> FileStageOutcome:
        state = self._reasoning_repository.load_current(project_id)
        processed = self._file_pipeline.ingest(
            stream,
            project_id=project_id,
            problem_id=state.problem_id,
            original_name=original_name,
            declared_mime_type=declared_mime_type,
        )
        gate = files_quality_gate(processed, self._file_pipeline.store)
        file = processed.parsed_file.file
        profiles = processed.profile_bundle.profiles if processed.profile_bundle is not None else []
        relationships = (
            processed.profile_bundle.cross_dataset_relationships
            if processed.profile_bundle is not None
            else []
        )
        target_stage = (
            DataWorkflowStage.FILES if gate.status is QualityGateStatus.PASS else state.data_stage
        )
        original_artifact = next(
            item for item in processed.artifacts if item.kind is ArtifactKind.ORIGINAL_FILE
        )
        next_state = self._next_state(
            state,
            target_stage=target_stage,
            gate=gate,
            updated_by="file_pipeline",
            reason=(
                f"file {file.safe_name} passed deterministic ingestion"
                if gate.status is QualityGateStatus.PASS
                else f"file {file.safe_name} failed FILES quality gate"
            ),
            source_id=file.file_id,
            changes={
                "files": [
                    *state.files,
                    ArtifactRef(
                        artifact_id=str(original_artifact.artifact_id),
                        kind=original_artifact.kind.value,
                        path=original_artifact.storage_key,
                        content_hash=original_artifact.sha256,
                        is_original=True,
                        metadata={"file_id": str(file.file_id)},
                    ),
                ],
                "registered_files": [*state.registered_files, file],
                "tracked_artifacts": [*state.tracked_artifacts, *processed.artifacts],
                "datasets": [*state.datasets, *processed.datasets],
                "data_profiles": [*state.data_profiles, *profiles],
                "cross_dataset_relationships": [
                    *state.cross_dataset_relationships,
                    *relationships,
                ],
                # Any accepted new attachment invalidates prior semantic interpretation.
                "data_understanding": (
                    None if gate.status is QualityGateStatus.PASS else state.data_understanding
                ),
            },
        )
        self._data_repository.persist_processed_with_state(processed, next_state)
        return FileStageOutcome(state=next_state, processed=processed, gate=gate)

    async def analyze_data(
        self,
        project_id: UUID,
        *,
        media_file_ids: list[UUID] | None = None,
        user_guidance: list[str] | None = None,
    ) -> DataStageOutcome:
        state = self._reasoning_repository.load_current(project_id)
        if state.data_stage is not DataWorkflowStage.FILES:
            raise QualityGateError("DATA requires the FILES stage to pass first")
        media_assets = [
            self._file_pipeline.multimodal_asset(
                self._data_repository.get_file(project_id, file_id)
            )
            for file_id in media_file_ids or []
        ]
        if not state.data_profiles and not media_assets:
            raise QualityGateError(
                "DATA requires at least one deterministic profile or media attachment"
            )
        profile_size = sum(len(profile.model_dump_json()) for profile in state.data_profiles)
        result = await self._data_agent.run(
            DataAgentInput(
                raw_problem=state.raw_problem,
                problem_analysis=state.problem_analysis,
                profiles=state.data_profiles,
                cross_dataset_relationships=state.cross_dataset_relationships,
                media_assets=media_assets,
                user_guidance=user_guidance or [],
            ),
            state,
            TaskProfile(
                task_type=TaskType.DATA_UNDERSTANDING,
                complexity=3,
                reasoning_requirement=3,
                multimodal_requirement=4 if media_assets else 0,
                long_context_requirement=4 if profile_size >= 100_000 else 2,
                review_requirement=3,
                blast_radius=3,
            ),
        )
        output = self._require_output(state, result)
        gate = data_quality_gate(
            state.data_profiles,
            output,
            media_present=bool(media_assets),
        )
        if gate.status is not QualityGateStatus.PASS:
            rejected = result.model_copy(
                update={
                    "status": AgentRunStatus.RETRY,
                    "errors": [*result.errors, f"DATA gate: {', '.join(gate.errors)}"],
                }
            )
            self._reasoning_repository.record_run(project_id, state.problem_id, rejected)
            raise QualityGateError(f"DATA quality gate rejected output: {', '.join(gate.errors)}")

        next_state = self._next_state(
            state,
            target_stage=DataWorkflowStage.DATA,
            gate=gate,
            updated_by=self._data_agent.name,
            reason="structured data understanding passed deterministic reference checks",
            source_id=result.run_id,
            changes={"data_understanding": output},
        )
        stored_run = self._reasoning_repository.save_revision(
            project_id=project_id,
            state=next_state,
            run=result,
        )
        return DataStageOutcome(
            state=next_state,
            output=output,
            run=stored_run,
            gate=gate,
        )

    def execute_code(
        self,
        project_id: UUID,
        code: str,
        *,
        input_file_ids: list[UUID] | None = None,
    ) -> ExecutionStageOutcome:
        state = self._reasoning_repository.load_current(project_id)
        if state.data_stage is not DataWorkflowStage.DATA:
            raise QualityGateError("EXECUTION requires the DATA stage to pass first")
        input_files = [
            self._data_repository.get_file(project_id, file_id) for file_id in input_file_ids or []
        ]
        execution = self._sandbox.execute(
            code,
            project_id=project_id,
            problem_id=state.problem_id,
            input_files=input_files,
        )
        gate = execution_quality_gate(
            execution.record,
            self._file_pipeline.store,
            execution.artifact_records,
        )
        target_stage = (
            DataWorkflowStage.EXECUTION
            if gate.status is QualityGateStatus.PASS
            else DataWorkflowStage.DATA
        )
        code_artifact = next(
            item
            for item in execution.artifact_records
            if item.artifact_id == execution.record.code_artifact_id
        )
        next_state = self._next_state(
            state,
            target_stage=target_stage,
            gate=gate,
            updated_by="sandbox_executor",
            reason=(
                "isolated Python execution passed the EXECUTION gate"
                if gate.status is QualityGateStatus.PASS
                else "isolated Python execution was recorded but did not pass its gate"
            ),
            source_id=execution.record.run_id,
            changes={
                "code_files": [
                    *state.code_files,
                    ArtifactRef(
                        artifact_id=str(code_artifact.artifact_id),
                        kind=code_artifact.kind.value,
                        path=code_artifact.storage_key,
                        content_hash=code_artifact.sha256,
                        metadata={"execution_run_id": str(execution.record.run_id)},
                    ),
                ],
                "execution_records": [*state.execution_records, execution.record],
                "tracked_artifacts": [
                    *state.tracked_artifacts,
                    *execution.artifact_records,
                ],
            },
        )
        self._data_repository.persist_execution_with_state(
            execution.record,
            execution.artifact_records,
            next_state,
        )
        return ExecutionStageOutcome(state=next_state, execution=execution, gate=gate)

    def _require_output(
        self,
        state: ProblemState,
        result: AgentRunResult[DataUnderstanding],
    ) -> DataUnderstanding:
        if result.status is not AgentRunStatus.SUCCEEDED or result.output is None:
            self._reasoning_repository.record_run(
                state.project_id,
                state.problem_id,
                result,
            )
            raise AgentRunError(
                "data_agent did not produce accepted structured output: "
                f"{'; '.join(result.errors) or result.status.value}"
            )
        return result.output

    @staticmethod
    def _next_state(
        state: ProblemState,
        *,
        target_stage: DataWorkflowStage,
        gate: QualityGateResult,
        updated_by: str,
        reason: str,
        source_id: UUID,
        changes: dict[str, object],
    ) -> ProblemState:
        next_version = state.version + 1
        history = DataStageHistoryEntry(
            from_stage=state.data_stage,
            to_stage=target_stage,
            status=gate.status,
            input_version=state.version,
            output_version=next_version,
            updated_by=updated_by,
            reason=reason,
            source_id=source_id,
        )
        payload = state.model_dump()
        payload.update(
            {
                **changes,
                "version": next_version,
                "data_stage": target_stage,
                "data_stage_history": [*state.data_stage_history, history],
                "quality_gates": [*state.quality_gates, gate],
                "updated_by": updated_by,
                "update_reason": reason,
                "updated_at": datetime.now(UTC),
            }
        )
        return ProblemState.model_validate(payload)
