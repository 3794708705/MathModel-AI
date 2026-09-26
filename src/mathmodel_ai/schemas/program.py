from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import PurePosixPath
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mathmodel_ai.schemas.execution import ExecutionOrigin
from mathmodel_ai.schemas.mathematical import MathematicalModel
from mathmodel_ai.schemas.solver import AlgorithmPlan


class ProgramLanguage(StrEnum):
    PYTHON = "PYTHON"


class GeneratedProgramStatus(StrEnum):
    DRAFT = "DRAFT"
    READY = "READY"
    CODE_GENERATION_BLOCKED = "CODE_GENERATION_BLOCKED"
    EXECUTED = "EXECUTED"
    FAILED = "FAILED"


class ExecutionStrategy(StrEnum):
    AUTO = "AUTO"
    DETERMINISTIC = "DETERMINISTIC"
    GENERATED = "GENERATED"


class GeneratedSourceFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1, max_length=255)
    content: str = Field(min_length=1, max_length=2 * 1024 * 1024)
    sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    artifact_id: UUID | None = None

    @model_validator(mode="after")
    def validate_relative_path_and_hash(self) -> GeneratedSourceFile:
        parsed = PurePosixPath(self.path.replace("\\", "/"))
        if parsed.is_absolute() or ".." in parsed.parts or not parsed.parts:
            raise ValueError("generated source path must be traversal-safe and relative")
        digest = hashlib.sha256(self.content.encode("utf-8")).hexdigest()
        if self.sha256 is not None and self.sha256 != digest:
            raise ValueError("generated source sha256 does not match content")
        return self

    def with_digest(self) -> GeneratedSourceFile:
        return self.model_copy(
            update={"sha256": hashlib.sha256(self.content.encode("utf-8")).hexdigest()}
        )


class GeneratedProgramDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    language: ProgramLanguage = ProgramLanguage.PYTHON
    entrypoint: str = Field(min_length=1, max_length=255)
    files: list[GeneratedSourceFile] = Field(min_length=1, max_length=20)
    dependencies: list[str] = Field(default_factory=list, max_length=100)
    solver_target: str = Field(min_length=1, max_length=64)
    explanation: str = Field(min_length=1)

    @model_validator(mode="after")
    def entrypoint_must_exist(self) -> GeneratedProgramDraft:
        paths = {item.path.replace("\\", "/") for item in self.files}
        if self.entrypoint.replace("\\", "/") not in paths:
            raise ValueError("generated program entrypoint must reference a source file")
        if len(paths) != len(self.files):
            raise ValueError("generated program file paths must be unique")
        return self


class GeneratedProgram(GeneratedProgramDraft):
    program_id: UUID = Field(default_factory=uuid4)
    project_id: UUID
    problem_id: UUID
    model_id: UUID
    model_version: int = Field(ge=1)
    model_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    code_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    generated_by: str = Field(min_length=1)
    generator_agent_run_id: UUID | None = None
    prompt_version: str = Field(min_length=1)
    execution_origin: ExecutionOrigin
    status: GeneratedProgramStatus
    is_mock: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def code_hash_matches_files(self) -> GeneratedProgram:
        if self.code_hash != generated_program_hash(self.files):
            raise ValueError("generated program code_hash does not match source files")
        return self


class GeneratedProgramRef(BaseModel):
    program_id: UUID
    model_id: UUID
    model_version: int = Field(ge=1)
    model_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    solver_target: str = Field(min_length=1, max_length=64)
    code_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    execution_origin: ExecutionOrigin
    status: GeneratedProgramStatus


class ExecutionStrategyDecision(BaseModel):
    requested: ExecutionStrategy
    selected: ExecutionStrategy
    reason: str = Field(min_length=1)


class CodeAgentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mathematical_model: MathematicalModel
    algorithm_plan: AlgorithmPlan
    input_manifest: list[dict[str, str]] = Field(default_factory=list)
    user_guidance: list[str] = Field(default_factory=list)


def generated_program_hash(files: list[GeneratedSourceFile]) -> str:
    digest = hashlib.sha256()
    for item in sorted(files, key=lambda source: source.path):
        digest.update(item.path.replace("\\", "/").encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.content.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()
