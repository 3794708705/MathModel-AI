from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ExecutionStatus(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    UNAVAILABLE = "UNAVAILABLE"
    REJECTED = "REJECTED"


class SandboxLimits(BaseModel):
    cpu_cores: float = Field(gt=0, le=64)
    memory_mb: int = Field(ge=64, le=262_144)
    timeout_seconds: float = Field(gt=0, le=3600)
    pids_limit: int = Field(ge=16, le=4096)
    max_output_bytes: int = Field(ge=1024)
    max_artifacts: int = Field(ge=0, le=1000)
    max_artifact_bytes: int = Field(ge=1024)


class ExecutionArtifact(BaseModel):
    artifact_id: UUID
    name: str = Field(min_length=1)
    mime_type: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    storage_key: str = Field(min_length=1)


class ExecutionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: UUID = Field(default_factory=uuid4)
    project_id: UUID
    problem_id: UUID
    code_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    code_artifact_id: UUID
    image: str = Field(min_length=1)
    image_id: str | None = None
    environment: dict[str, str] = Field(default_factory=dict)
    start_time: datetime = Field(default_factory=lambda: datetime.now(UTC))
    end_time: datetime
    runtime_seconds: float = Field(ge=0)
    status: ExecutionStatus
    stdout: str = ""
    stderr: str = ""
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    exit_code: int | None = None
    limits: SandboxLimits
    metrics: dict[str, int | float | str | bool | None] = Field(default_factory=dict)
    artifacts: list[ExecutionArtifact] = Field(default_factory=list)
    network_disabled: bool = True
    non_root: bool = True
    read_only_root: bool = True
    error: str | None = None
    is_mock: bool = False

    @model_validator(mode="after")
    def successful_run_requires_zero_exit(self) -> "ExecutionRecord":
        if self.status is ExecutionStatus.SUCCEEDED and self.exit_code != 0:
            raise ValueError("successful execution requires exit_code=0")
        if self.is_mock and self.status is ExecutionStatus.SUCCEEDED:
            raise ValueError("mock execution cannot claim success")
        return self
