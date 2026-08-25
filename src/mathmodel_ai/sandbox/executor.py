import hashlib
import logging
import mimetypes
import os
import shutil
import stat
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from uuid import UUID, uuid4

from mathmodel_ai.core.errors import SandboxError
from mathmodel_ai.files.storage import FileStore
from mathmodel_ai.schemas.execution import (
    ExecutionArtifact,
    ExecutionRecord,
    ExecutionStatus,
    SandboxLimits,
)
from mathmodel_ai.schemas.files import ArtifactKind, ArtifactRecord, RegisteredFile

CommandRunner = Callable[..., subprocess.CompletedProcess[bytes]]


@dataclass(frozen=True)
class SandboxExecution:
    record: ExecutionRecord
    artifact_records: list[ArtifactRecord]


class SandboxExecutor:
    """Executes Python in a resource-bounded, networkless Docker container."""

    def __init__(
        self,
        *,
        store: FileStore,
        root: Path,
        image: str,
        limits: SandboxLimits,
        runner: CommandRunner = subprocess.run,
        docker_binary: str = "docker",
        max_code_bytes: int = 2 * 1024 * 1024,
    ) -> None:
        self._root = root.resolve(strict=False)
        self._store = store
        self._image = image
        self._limits = limits
        self._runner = runner
        self._docker = docker_binary
        self._max_code_bytes = max_code_bytes
        self._logger = logging.getLogger("mathmodel_ai.sandbox.executor")

    def execute(
        self,
        code: str,
        *,
        project_id: UUID,
        problem_id: UUID,
        input_files: Sequence[RegisteredFile] = (),
    ) -> SandboxExecution:
        encoded = code.encode("utf-8")
        if not encoded:
            raise SandboxError("sandbox code cannot be empty")
        if len(encoded) > self._max_code_bytes:
            raise SandboxError("sandbox code exceeds configured size limit")

        run_id = uuid4()
        container_name = f"mathmodel-ai-{run_id.hex}"
        run_root = self._root / run_id.hex
        workspace = run_root / "workspace"
        output = run_root / "output"
        code_hash = hashlib.sha256(encoded).hexdigest()
        try:
            workspace.mkdir(parents=True)
            output.mkdir()
            code_path = workspace / "main.py"
            code_path.write_bytes(encoded)
            input_directory = workspace / "inputs"
            input_directory.mkdir()
            for index, item in enumerate(input_files, start=1):
                data = self._store.read_bytes(item.storage_key)
                input_name = f"{index:04d}-{item.file_id.hex}{item.extension}"
                (input_directory / input_name).write_bytes(data)
            self._restrict_workspace(workspace, output)
            code_record = self._store_code_artifact(
                encoded,
                project_id=project_id,
                problem_id=problem_id,
                run_id=run_id,
            )
        except Exception:
            self._cleanup(run_root)
            raise
        started = datetime.now(UTC)
        clock_started = monotonic()
        image_id = self._image_id()
        if image_id is None:
            ended = datetime.now(UTC)
            record = ExecutionRecord(
                run_id=run_id,
                project_id=project_id,
                problem_id=problem_id,
                code_hash=code_hash,
                code_artifact_id=code_record.artifact_id,
                image=self._image,
                image_id=None,
                environment={"executor": "docker"},
                start_time=started,
                end_time=ended,
                runtime_seconds=max(0.0, monotonic() - clock_started),
                status=ExecutionStatus.UNAVAILABLE,
                exit_code=None,
                limits=self._limits,
                error="configured sandbox image is unavailable",
            )
            self._cleanup(run_root)
            self._logger.warning(
                "sandbox image unavailable",
                extra={"run_id": str(run_id), "project_id": str(project_id)},
            )
            return SandboxExecution(record=record, artifact_records=[code_record])

        command = self._command(
            container_name=container_name,
            workspace=workspace,
            image_reference=image_id,
        )
        timed_out = False
        result: subprocess.CompletedProcess[bytes] | None = None
        runner_error: str | None = None
        try:
            result = self._runner(
                command,
                capture_output=True,
                check=False,
                timeout=self._limits.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            result = subprocess.CompletedProcess(
                command,
                returncode=124,
                stdout=exc.stdout or b"",
                stderr=exc.stderr or b"",
            )
        except (OSError, subprocess.SubprocessError) as exc:
            runner_error = f"docker execution failed: {type(exc).__name__}"

        copy_error: str | None = None
        if not timed_out and runner_error is None:
            copy_error = self._copy_outputs(container_name, output)
        self._force_remove(container_name)

        ended = datetime.now(UTC)
        stdout, stdout_truncated = self._bounded_output(
            result.stdout if result is not None else b""
        )
        stderr, stderr_truncated = self._bounded_output(
            result.stderr if result is not None else b""
        )
        status = ExecutionStatus.FAILED
        exit_code = result.returncode if result is not None else None
        error = runner_error
        if timed_out:
            status = ExecutionStatus.TIMEOUT
            error = f"execution exceeded {self._limits.timeout_seconds} seconds"
        elif result is not None and result.returncode == 0:
            status = ExecutionStatus.SUCCEEDED
        if copy_error is not None:
            status = ExecutionStatus.REJECTED
            error = copy_error

        try:
            artifact_records, execution_artifacts, artifact_error = self._collect_artifacts(
                output,
                project_id=project_id,
                problem_id=problem_id,
                run_id=run_id,
            )
        except Exception:
            self._cleanup(run_root)
            raise
        if artifact_error is not None:
            status = ExecutionStatus.REJECTED
            error = artifact_error

        record = ExecutionRecord(
            run_id=run_id,
            project_id=project_id,
            problem_id=problem_id,
            code_hash=code_hash,
            code_artifact_id=code_record.artifact_id,
            image=self._image,
            image_id=image_id,
            environment={
                "executor": "docker",
                "python_mode": "isolated",
                "input_file_count": str(len(input_files)),
            },
            start_time=started,
            end_time=ended,
            runtime_seconds=max(0.0, monotonic() - clock_started),
            status=status,
            stdout=stdout,
            stderr=stderr,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
            exit_code=exit_code,
            limits=self._limits,
            metrics={
                "stdout_bytes": len(result.stdout) if result is not None else 0,
                "stderr_bytes": len(result.stderr) if result is not None else 0,
                "artifact_count": len(execution_artifacts),
            },
            artifacts=execution_artifacts,
            error=error,
        )
        self._cleanup(run_root)
        self._logger.info(
            "sandbox execution completed",
            extra={
                "run_id": str(run_id),
                "project_id": str(project_id),
                "status": record.status.value,
                "runtime_seconds": record.runtime_seconds,
                "artifact_count": len(record.artifacts),
            },
        )
        return SandboxExecution(
            record=record,
            artifact_records=[code_record, *artifact_records],
        )

    def _command(
        self,
        *,
        container_name: str,
        workspace: Path,
        image_reference: str,
    ) -> list[str]:
        memory = f"{self._limits.memory_mb}m"
        return [
            self._docker,
            "run",
            "--name",
            container_name,
            "--network",
            "none",
            "--cpus",
            str(self._limits.cpu_cores),
            "--memory",
            memory,
            "--memory-swap",
            memory,
            "--pids-limit",
            str(self._limits.pids_limit),
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--user",
            "65532:65532",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,nodev,size=64m",
            "--tmpfs",
            (
                "/output:rw,noexec,nosuid,nodev,mode=1777,uid=65532,gid=65532,"
                f"size={self._limits.max_artifact_bytes}"
            ),
            "--mount",
            f"type=bind,source={workspace},target=/workspace,readonly",
            "--workdir",
            "/workspace",
            "--env",
            "HOME=/tmp",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            image_reference,
            "python",
            "-I",
            "-B",
            "/workspace/main.py",
        ]

    def _image_id(self) -> str | None:
        try:
            result = self._runner(
                [self._docker, "image", "inspect", self._image, "--format", "{{.Id}}"],
                capture_output=True,
                check=False,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if result.returncode != 0:
            return None
        value = result.stdout.decode("utf-8", errors="replace").strip()
        return value or None

    def _copy_outputs(self, container_name: str, output: Path) -> str | None:
        try:
            result = self._runner(
                [self._docker, "cp", f"{container_name}:/output/.", str(output)],
                capture_output=True,
                check=False,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return f"sandbox output collection failed: {type(exc).__name__}"
        if result.returncode != 0:
            return "sandbox output collection failed"
        return None

    def _force_remove(self, container_name: str) -> None:
        try:
            self._runner(
                [self._docker, "rm", "--force", container_name],
                capture_output=True,
                check=False,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return

    def _store_code_artifact(
        self, data: bytes, *, project_id: UUID, problem_id: UUID, run_id: UUID
    ) -> ArtifactRecord:
        artifact_id = uuid4()
        stored = self._store.store_artifact(
            data,
            project_id=project_id,
            artifact_id=artifact_id,
            filename="main.py",
        )
        return ArtifactRecord(
            artifact_id=artifact_id,
            project_id=project_id,
            problem_id=problem_id,
            execution_run_id=run_id,
            kind=ArtifactKind.GENERATED_CODE,
            name="main.py",
            mime_type="text/x-python",
            size_bytes=stored.size_bytes,
            sha256=stored.sha256,
            storage_key=stored.storage_key,
            metadata={"code_hash": stored.sha256},
        )

    def _collect_artifacts(
        self,
        output: Path,
        *,
        project_id: UUID,
        problem_id: UUID,
        run_id: UUID,
    ) -> tuple[list[ArtifactRecord], list[ExecutionArtifact], str | None]:
        paths = sorted(path for path in output.rglob("*") if path.is_file())
        if len(paths) > self._limits.max_artifacts:
            return [], [], "sandbox produced too many artifacts"
        total_size = sum(path.stat().st_size for path in paths)
        if total_size > self._limits.max_artifact_bytes:
            return [], [], "sandbox artifacts exceed aggregate size limit"
        for path in paths:
            if path.is_symlink() or not path.resolve().is_relative_to(output.resolve()):
                return [], [], "sandbox artifact attempted to escape output directory"
            if path.stat().st_size > self._limits.max_artifact_bytes:
                return [], [], f"sandbox artifact {path.name!r} exceeds size limit"
        records: list[ArtifactRecord] = []
        execution_artifacts: list[ExecutionArtifact] = []
        for path in paths:
            data = path.read_bytes()
            artifact_id = uuid4()
            relative = path.relative_to(output).as_posix()
            safe_name = relative.replace("/", "__")[:255]
            stored = self._store.store_artifact(
                data,
                project_id=project_id,
                artifact_id=artifact_id,
                filename=safe_name,
            )
            mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            record = ArtifactRecord(
                artifact_id=artifact_id,
                project_id=project_id,
                problem_id=problem_id,
                execution_run_id=run_id,
                kind=ArtifactKind.SANDBOX_OUTPUT,
                name=safe_name,
                mime_type=mime_type,
                size_bytes=stored.size_bytes,
                sha256=stored.sha256,
                storage_key=stored.storage_key,
                metadata={"relative_output_path": relative},
            )
            records.append(record)
            execution_artifacts.append(
                ExecutionArtifact(
                    artifact_id=artifact_id,
                    name=safe_name,
                    mime_type=mime_type,
                    size_bytes=stored.size_bytes,
                    sha256=stored.sha256,
                    storage_key=stored.storage_key,
                )
            )
        return records, execution_artifacts, None

    def _bounded_output(self, data: bytes) -> tuple[str, bool]:
        truncated = len(data) > self._limits.max_output_bytes
        bounded = data[: self._limits.max_output_bytes]
        return bounded.decode("utf-8", errors="replace"), truncated

    @staticmethod
    def _restrict_workspace(workspace: Path, output: Path) -> None:
        for path in workspace.rglob("*"):
            os.chmod(path, 0o555 if path.is_dir() else 0o444)
        os.chmod(workspace, 0o555)
        os.chmod(output, 0o777)

    def _cleanup(self, run_root: Path) -> None:
        resolved = run_root.resolve(strict=False)
        if resolved.parent != self._root or len(resolved.name) != 32:
            raise SandboxError("refusing to clean unexpected sandbox path")
        if not resolved.exists():
            return

        def restore_write_and_retry(
            function: Callable[[str], object], path: str, _exc: object
        ) -> None:
            os.chmod(path, stat.S_IWRITE)
            function(path)

        try:
            shutil.rmtree(resolved, onexc=restore_write_and_retry)
        except OSError as exc:
            raise SandboxError("failed to clean isolated execution workspace") from exc
