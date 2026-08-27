import io
import subprocess
import tarfile
from pathlib import Path
from uuid import uuid4

import pytest

from mathmodel_ai.core.errors import SandboxError
from mathmodel_ai.data.quality_gates import execution_quality_gate
from mathmodel_ai.files.storage import LocalFileStore
from mathmodel_ai.sandbox.executor import SandboxExecutor, SecretMount
from mathmodel_ai.schemas.execution import ExecutionStatus, SandboxLimits
from mathmodel_ai.schemas.quality import QualityGateStatus


def limits(*, timeout: float = 2) -> SandboxLimits:
    return SandboxLimits(
        cpu_cores=0.5,
        memory_mb=128,
        timeout_seconds=timeout,
        pids_limit=32,
        max_output_bytes=1024,
        max_artifacts=5,
        max_artifact_bytes=4096,
    )


class SuccessfulRunner:
    def __init__(self) -> None:
        self.run_command: list[str] = []

    def __call__(self, command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        if command[1:3] == ["image", "inspect"]:
            return subprocess.CompletedProcess(command, 0, stdout=b"sha256:image-id\n", stderr=b"")
        if command[1] == "run":
            self.run_command = command
            return subprocess.CompletedProcess(command, 0, stdout=b"container-id\n", stderr=b"")
        if command[1] == "logs":
            return subprocess.CompletedProcess(command, 0, stdout=b"computed\n", stderr=b"")
        if command[1] == "exec" and command[3] == "python":
            stream = io.BytesIO()
            with tarfile.open(fileobj=stream, mode="w") as archive:
                for name, data in {
                    ".mathmodel-exit": b"0",
                    "result.json": b'{"verified":true}',
                }.items():
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    archive.addfile(info, io.BytesIO(data))
            return subprocess.CompletedProcess(command, 0, stdout=stream.getvalue(), stderr=b"")
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")


def test_executor_records_success_and_all_required_docker_boundaries(tmp_path: Path) -> None:
    store = LocalFileStore(tmp_path / "store")
    runner = SuccessfulRunner()
    executor = SandboxExecutor(
        store=store,
        root=tmp_path / "runs",
        image="sandbox:test",
        limits=limits(),
        runner=runner,
    )

    outcome = executor.execute(
        "print('computed')",
        project_id=uuid4(),
        problem_id=uuid4(),
    )

    record = outcome.record
    assert record.status is ExecutionStatus.SUCCEEDED
    assert record.image_id == "sha256:image-id"
    assert record.stdout == "computed\n"
    assert record.exit_code == 0
    assert len(record.artifacts) == 1
    assert {"--network", "none", "--read-only", "--cap-drop", "--user"} <= set(runner.run_command)
    assert "65532:65532" in runner.run_command
    assert "--cpus" in runner.run_command
    assert "--memory" in runner.run_command
    assert "--pids-limit" in runner.run_command
    assert "--rm" not in runner.run_command
    assert any(item.startswith("/output:") for item in runner.run_command)
    assert not any(",target=/output" in item for item in runner.run_command)
    assert (
        execution_quality_gate(record, store, outcome.artifact_records).status
        is QualityGateStatus.PASS
    )
    assert not (tmp_path / "runs" / record.run_id.hex).exists()


class TimeoutRunner:
    def __call__(self, command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        if command[1:3] == ["image", "inspect"]:
            return subprocess.CompletedProcess(command, 0, stdout=b"sha256:image-id", stderr=b"")
        if command[1:3] == ["rm", "--force"]:
            return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")
        raise subprocess.TimeoutExpired(
            command, timeout=float(kwargs["timeout"]), output=b"partial"
        )


def test_executor_records_timeout_without_claiming_success(tmp_path: Path) -> None:
    executor = SandboxExecutor(
        store=LocalFileStore(tmp_path / "store"),
        root=tmp_path / "runs",
        image="sandbox:test",
        limits=limits(timeout=0.1),
        runner=TimeoutRunner(),
    )

    outcome = executor.execute("while True: pass", project_id=uuid4(), problem_id=uuid4())

    assert outcome.record.status is ExecutionStatus.TIMEOUT
    assert outcome.record.exit_code == 124
    assert outcome.record.error is not None
    assert "exceeded" in outcome.record.error


def test_executor_records_unavailable_image(tmp_path: Path) -> None:
    def unavailable(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(command, 1, stdout=b"", stderr=b"not found")

    executor = SandboxExecutor(
        store=LocalFileStore(tmp_path / "store"),
        root=tmp_path / "runs",
        image="missing:test",
        limits=limits(),
        runner=unavailable,
    )

    outcome = executor.execute("print(1)", project_id=uuid4(), problem_id=uuid4())

    assert outcome.record.status is ExecutionStatus.UNAVAILABLE
    assert outcome.record.image_id is None
    assert outcome.record.is_mock is False


class UnsafeArchiveRunner(SuccessfulRunner):
    def __call__(self, command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        if command[1] == "exec" and command[3] == "python":
            stream = io.BytesIO()
            with tarfile.open(fileobj=stream, mode="w") as archive:
                data = b"escape"
                info = tarfile.TarInfo("../outside.txt")
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
            return subprocess.CompletedProcess(command, 0, stdout=stream.getvalue(), stderr=b"")
        return super().__call__(command, **kwargs)


def test_executor_rejects_unsafe_output_archive_path(tmp_path: Path) -> None:
    executor = SandboxExecutor(
        store=LocalFileStore(tmp_path / "store"),
        root=tmp_path / "runs",
        image="sandbox:test",
        limits=limits(),
        runner=UnsafeArchiveRunner(),
    )

    outcome = executor.execute("print(1)", project_id=uuid4(), problem_id=uuid4())

    assert outcome.record.status is ExecutionStatus.REJECTED
    assert outcome.record.error == "sandbox output archive contains an unsafe path"
    assert not (tmp_path / "outside.txt").exists()


def test_secret_mount_rejects_missing_source_and_unsafe_names(tmp_path: Path) -> None:
    with pytest.raises(SandboxError, match="existing regular file"):
        SecretMount(
            source=tmp_path / "missing.lic",
            target_name="gurobi.lic",
            environment_name="GRB_LICENSE_FILE",
        )

    secret = tmp_path / "runtime.lic"
    secret.write_text("fixture-only", encoding="utf-8")
    with pytest.raises(SandboxError, match="target name"):
        SecretMount(
            source=secret,
            target_name="../gurobi.lic",
            environment_name="GRB_LICENSE_FILE",
        )
    with pytest.raises(SandboxError, match="environment name"):
        SecretMount(
            source=secret,
            target_name="gurobi.lic",
            environment_name="lowercase",
        )
