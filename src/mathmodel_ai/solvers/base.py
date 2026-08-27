from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from mathmodel_ai.files.storage import FileStore
from mathmodel_ai.sandbox.executor import SandboxExecution, SandboxExecutor, SecretMount
from mathmodel_ai.schemas.execution import ExecutionStatus
from mathmodel_ai.schemas.mathematical import MathematicalModel
from mathmodel_ai.schemas.program import (
    GeneratedProgram,
    GeneratedProgramStatus,
    GeneratedSourceFile,
)
from mathmodel_ai.schemas.solver import (
    SolverCapability,
    SolverFamily,
    SolverHealth,
    SolverName,
    SolverOptions,
    SolverResult,
    SolverStatus,
    SupportAssessment,
)


class RawSolverOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    solver_version: str | None = None
    native_status: int
    native_status_name: str | None = None
    success: bool
    message: str = ""
    objective: float | None = Field(default=None, allow_inf_nan=False)
    variables: dict[str, float] = Field(default_factory=dict)
    iterations: int | None = Field(default=None, ge=0)
    nodes: int | None = Field(default=None, ge=0)
    mip_gap: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    runtime_seconds: float = Field(ge=0)


@dataclass(frozen=True)
class SolverExecution:
    result: SolverResult
    execution: SandboxExecution
    program: GeneratedProgram


class BaseSolver(ABC):
    name: SolverName
    families: frozenset[SolverFamily]
    capabilities: frozenset[SolverCapability]

    def __init__(self, *, sandbox: SandboxExecutor, store: FileStore) -> None:
        self._sandbox = sandbox
        self._store = store

    def supports(self, model: MathematicalModel) -> bool:
        return self.assess_support(model).supported

    @abstractmethod
    def assess_support(self, model: MathematicalModel) -> SupportAssessment:
        raise NotImplementedError

    @abstractmethod
    def solve(self, model: MathematicalModel, options: SolverOptions) -> SolverExecution:
        raise NotImplementedError

    @abstractmethod
    def health_check(self) -> SolverHealth:
        raise NotImplementedError

    def get_capabilities(self) -> frozenset[SolverCapability]:
        return self.capabilities

    def _run_program(
        self,
        program: GeneratedProgram,
        *,
        secret_mounts: tuple[SecretMount, ...] = (),
    ) -> tuple[SandboxExecution, RawSolverOutput | None, GeneratedProgram]:
        entrypoint = next(item for item in program.files if item.path == program.entrypoint)
        execution = self._sandbox.execute(
            entrypoint.content,
            project_id=program.project_id,
            problem_id=program.problem_id,
            secret_mounts=secret_mounts,
            execution_origin=program.execution_origin,
            model_digest=program.model_digest,
            generated_program_id=program.program_id,
            entrypoint=program.entrypoint,
            source_files={
                item.path: item.content for item in program.files if item.path != program.entrypoint
            },
        )
        code_artifact_id = execution.record.code_artifact_id
        code_artifact = next(
            item for item in execution.artifact_records if item.artifact_id == code_artifact_id
        )
        files = [
            GeneratedSourceFile(
                path=item.path,
                content=item.content,
                sha256=item.sha256,
                artifact_id=(
                    code_artifact.artifact_id if item.path == program.entrypoint else None
                ),
            )
            for item in program.files
        ]
        status = (
            GeneratedProgramStatus.EXECUTED
            if execution.record.status is ExecutionStatus.SUCCEEDED
            else GeneratedProgramStatus.FAILED
        )
        finalized = program.model_copy(update={"files": files, "status": status})
        if execution.record.status is not ExecutionStatus.SUCCEEDED:
            return execution, None, finalized
        result_artifact = next(
            (item for item in execution.artifact_records if item.name == "result.json"),
            None,
        )
        if result_artifact is None:
            return execution, None, finalized
        try:
            payload: Any = json.loads(
                self._store.read_bytes(result_artifact.storage_key).decode("utf-8")
            )
            raw = RawSolverOutput.model_validate(payload)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return execution, None, finalized
        return execution, raw, finalized

    @staticmethod
    def dependency_version(raw: RawSolverOutput | None) -> str | None:
        return raw.solver_version if raw is not None else None

    def execution_failure_result(self, execution: SandboxExecution) -> SolverResult:
        status = {
            ExecutionStatus.TIMEOUT: SolverStatus.TIME_LIMIT,
            ExecutionStatus.UNAVAILABLE: SolverStatus.SOLVER_UNAVAILABLE,
        }.get(execution.record.status, SolverStatus.EXECUTION_ERROR)
        return SolverResult(
            solver_name=self.name,
            status=status,
            runtime_seconds=execution.record.runtime_seconds,
            message=execution.record.error or "solver execution produced no valid result artifact",
            raw_status=execution.record.status.value,
            execution_record_id=execution.record.run_id,
        )

    @staticmethod
    def license_mount(path: Path | None) -> tuple[SecretMount, ...]:
        if path is None:
            return ()
        return (
            SecretMount(
                source=path,
                target_name="gurobi.lic",
                environment_name="GRB_LICENSE_FILE",
            ),
        )
