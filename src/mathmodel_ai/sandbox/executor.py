import hashlib
import io
import logging
import mimetypes
import os
import re
import shutil
import stat
import subprocess
import tarfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from time import monotonic
from uuid import UUID, uuid4

from mathmodel_ai.core.errors import SandboxError
from mathmodel_ai.files.storage import FileStore
from mathmodel_ai.schemas.execution import (
    ExecutionArtifact,
    ExecutionOrigin,
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


@dataclass(frozen=True)
class SecretMount:
    """Runtime-only read-only secret file; contents and host path are never audited."""

    source: Path
    target_name: str
    environment_name: str

    def __post_init__(self) -> None:
        resolved = self.source.resolve(strict=False)
        if not resolved.is_file():
            raise SandboxError("sandbox secret source must be an existing regular file")
        if re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", self.target_name) is None:
            raise SandboxError("sandbox secret target name is invalid")
        if re.fullmatch(r"[A-Z][A-Z0-9_]{0,99}", self.environment_name) is None:
            raise SandboxError("sandbox secret environment name is invalid")


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
        secret_mounts: Sequence[SecretMount] = (),
        execution_origin: ExecutionOrigin = ExecutionOrigin.USER_CODE,
        model_digest: str | None = None,
        generated_program_id: UUID | None = None,
        entrypoint: str = "main.py",
        source_files: Mapping[str, str] | None = None,
    ) -> SandboxExecution:
        normalized_entrypoint = self._safe_source_path(entrypoint)
        normalized_sources = {
            self._safe_source_path(path): content for path, content in (source_files or {}).items()
        }
        if normalized_entrypoint in normalized_sources:
            raise SandboxError("supporting source files cannot replace the entrypoint")
        encoded = code.encode("utf-8")
        if not encoded:
            raise SandboxError("sandbox code cannot be empty")
        if len(encoded) > self._max_code_bytes:
            raise SandboxError("sandbox code exceeds configured size limit")
        if (
            sum(len(item.encode("utf-8")) for item in normalized_sources.values()) + len(encoded)
            > self._max_code_bytes
        ):
            raise SandboxError("sandbox source bundle exceeds configured size limit")

        run_id = uuid4()
        container_name = f"mathmodel-ai-{run_id.hex}"
        run_root = self._root / run_id.hex
        workspace = run_root / "workspace"
        output = run_root / "output"
        code_hash = hashlib.sha256(encoded).hexdigest()
        bundle_hash = self._source_bundle_hash({normalized_entrypoint: code, **normalized_sources})
        try:
            workspace.mkdir(parents=True)
            output.mkdir()
            code_path = workspace / Path(*PurePosixPath(normalized_entrypoint).parts)
            code_path.parent.mkdir(parents=True, exist_ok=True)
            code_path.write_bytes(encoded)
            for relative, content in normalized_sources.items():
                source_path = workspace / Path(*PurePosixPath(relative).parts)
                source_path.parent.mkdir(parents=True, exist_ok=True)
                source_path.write_text(content, encoding="utf-8")
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
                filename=PurePosixPath(normalized_entrypoint).name,
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
                executed_bundle_hash=bundle_hash,
                code_artifact_id=code_record.artifact_id,
                execution_origin=execution_origin,
                model_digest=model_digest,
                generated_program_id=generated_program_id,
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
            secret_mounts=secret_mounts,
            entrypoint=normalized_entrypoint,
        )
        timed_out = False
        result: subprocess.CompletedProcess[bytes] | None = None
        runner_error: str | None = None
        container_started = False
        completion_observed = False
        try:
            start_result = self._runner(
                command,
                capture_output=True,
                check=False,
                timeout=min(self._limits.timeout_seconds, 15),
            )
            result = start_result
            if start_result.returncode == 0:
                container_started = True
                wait_result = self._runner(
                    self._completion_command(container_name),
                    capture_output=True,
                    check=False,
                    timeout=self._limits.timeout_seconds,
                )
                completion_observed = wait_result.returncode == 0
                if not completion_observed:
                    runner_error = "sandbox container stopped before reporting completion"
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
        if container_started:
            logs = self._container_logs(container_name)
            if logs is not None:
                result = subprocess.CompletedProcess(
                    command,
                    returncode=result.returncode if result is not None else 1,
                    stdout=logs.stdout,
                    stderr=logs.stderr,
                )
            elif runner_error is None:
                runner_error = "sandbox container logs could not be collected"
        if completion_observed and not timed_out and runner_error is None:
            copy_error = self._copy_outputs(container_name, output)
            if copy_error is None:
                marker_error, process_exit_code = self._consume_completion_marker(output)
                if marker_error is not None:
                    copy_error = marker_error
                elif result is not None:
                    result = subprocess.CompletedProcess(
                        command,
                        returncode=process_exit_code,
                        stdout=result.stdout,
                        stderr=result.stderr,
                    )
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
        elif result is not None and error is None:
            error = f"sandbox process exited with code {result.returncode}"
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
            executed_bundle_hash=bundle_hash,
            code_artifact_id=code_record.artifact_id,
            execution_origin=execution_origin,
            model_digest=model_digest,
            generated_program_id=generated_program_id,
            image=self._image,
            image_id=image_id,
            environment={
                "executor": "docker",
                "python_mode": "isolated",
                "input_file_count": str(len(input_files)),
                "runtime_secret_count": str(len(secret_mounts)),
                "source_file_count": str(len(normalized_sources) + 1),
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
        secret_mounts: Sequence[SecretMount] = (),
        entrypoint: str = "main.py",
    ) -> list[str]:
        memory = f"{self._limits.memory_mb}m"
        command = [
            self._docker,
            "run",
            "--name",
            container_name,
            "--detach",
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
        ]
        for secret in secret_mounts:
            target = f"/run/secrets/{secret.target_name}"
            command.extend(
                [
                    "--mount",
                    f"type=bind,source={secret.source.resolve()},target={target},readonly",
                    "--env",
                    f"{secret.environment_name}={target}",
                ]
            )
        command.extend(
            [
                image_reference,
                "/bin/sh",
                "-c",
                (
                    f"python -I -B /workspace/{entrypoint}; "
                    "status=$?; printf '%s' \"$status\" > /output/.mathmodel-exit; "
                    "while :; do sleep 3600; done"
                ),
            ]
        )
        return command

    def _completion_command(self, container_name: str) -> list[str]:
        return [
            self._docker,
            "exec",
            container_name,
            "/bin/sh",
            "-c",
            "until [ -f /output/.mathmodel-exit ]; do sleep 0.05; done",
        ]

    def _container_logs(self, container_name: str) -> subprocess.CompletedProcess[bytes] | None:
        try:
            result = self._runner(
                [self._docker, "logs", container_name],
                capture_output=True,
                check=False,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result if result.returncode == 0 else None

    @staticmethod
    def _consume_completion_marker(output: Path) -> tuple[str | None, int]:
        marker = output / ".mathmodel-exit"
        try:
            raw_exit_code = marker.read_text(encoding="ascii")
            exit_code = int(raw_exit_code)
        except (OSError, UnicodeDecodeError, ValueError):
            return "sandbox completion marker is missing or invalid", 1
        finally:
            marker.unlink(missing_ok=True)
        if not 0 <= exit_code <= 255:
            return "sandbox completion marker contains an invalid exit code", 1
        return None, exit_code

    def image_id(self) -> str | None:
        return self._image_id()

    def probe_python_module(self, module: str) -> bool:
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]{0,199}", module) is None:
            raise SandboxError("invalid Python module probe")
        image_id = self._image_id()
        if image_id is None:
            return False
        command = [
            self._docker,
            "run",
            "--rm",
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--user",
            "65532:65532",
            "--pids-limit",
            str(self._limits.pids_limit),
            "--memory",
            f"{self._limits.memory_mb}m",
            image_id,
            "python",
            "-I",
            "-c",
            (
                "import importlib.util,sys;"
                f"sys.exit(0 if importlib.util.find_spec({module!r}) else 1)"
            ),
        ]
        try:
            result = self._runner(
                command,
                capture_output=True,
                check=False,
                timeout=min(self._limits.timeout_seconds, 15),
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return result.returncode == 0

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
                [
                    self._docker,
                    "exec",
                    container_name,
                    "python",
                    "-I",
                    "-c",
                    (
                        "import sys,tarfile;"
                        "archive=tarfile.open(fileobj=sys.stdout.buffer,mode='w|');"
                        "archive.add('/output',arcname='.');archive.close()"
                    ),
                ],
                capture_output=True,
                check=False,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return f"sandbox output collection failed: {type(exc).__name__}"
        if result.returncode != 0:
            return "sandbox output collection failed"
        archive_limit = self._limits.max_artifact_bytes + max(
            20 * 1024,
            self._limits.max_artifacts * 2048,
        )
        if len(result.stdout) > archive_limit:
            return "sandbox output archive exceeds configured size limit"
        try:
            with tarfile.open(fileobj=io.BytesIO(result.stdout), mode="r:*") as archive:
                seen: set[Path] = set()
                for member in archive.getmembers():
                    if "\\" in member.name or ":" in member.name or "\x00" in member.name:
                        return "sandbox output archive contains an unsafe path"
                    parsed = PurePosixPath(member.name)
                    components = [item for item in parsed.parts if item != "."]
                    if not components:
                        continue
                    if parsed.is_absolute() or ".." in components:
                        return "sandbox output archive contains an unsafe path"
                    relative = Path(*components)
                    destination = output / relative
                    if member.isdir():
                        destination.mkdir(parents=True, exist_ok=True)
                        continue
                    if not member.isfile():
                        return "sandbox output archive contains a non-regular file"
                    if relative in seen:
                        return "sandbox output archive contains a duplicate path"
                    seen.add(relative)
                    source = archive.extractfile(member)
                    if source is None:
                        return "sandbox output archive contains an unreadable file"
                    data = source.read(self._limits.max_artifact_bytes + 1)
                    if len(data) > self._limits.max_artifact_bytes:
                        return f"sandbox artifact {relative.name!r} exceeds size limit"
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(data)
        except (tarfile.TarError, OSError):
            return "sandbox output archive is invalid"
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
        self,
        data: bytes,
        *,
        project_id: UUID,
        problem_id: UUID,
        run_id: UUID,
        filename: str = "main.py",
    ) -> ArtifactRecord:
        artifact_id = uuid4()
        stored = self._store.store_artifact(
            data,
            project_id=project_id,
            artifact_id=artifact_id,
            filename=filename,
        )
        return ArtifactRecord(
            artifact_id=artifact_id,
            project_id=project_id,
            problem_id=problem_id,
            execution_run_id=run_id,
            kind=ArtifactKind.GENERATED_CODE,
            name=filename,
            mime_type="text/x-python",
            size_bytes=stored.size_bytes,
            sha256=stored.sha256,
            storage_key=stored.storage_key,
            metadata={"code_hash": stored.sha256},
        )

    @staticmethod
    def _source_bundle_hash(files: Mapping[str, str]) -> str:
        digest = hashlib.sha256()
        for path, content in sorted(files.items()):
            digest.update(path.replace("\\", "/").encode("utf-8"))
            digest.update(b"\0")
            digest.update(content.encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()

    @staticmethod
    def _safe_source_path(path: str) -> str:
        normalized = path.replace("\\", "/")
        parsed = PurePosixPath(normalized)
        if parsed.is_absolute() or not parsed.parts or ".." in parsed.parts:
            raise SandboxError("sandbox source path must be traversal-safe and relative")
        if any(re.fullmatch(r"[A-Za-z0-9_.-]+", part) is None for part in parsed.parts):
            raise SandboxError("sandbox source path contains unsupported characters")
        return parsed.as_posix()

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
            if not os.path.islink(path):
                os.chmod(path, os.stat(path).st_mode | stat.S_IWRITE)
            function(path)

        try:
            # POSIX unlink needs write/search permission on the parent directory;
            # clearing a file's Windows read-only flag alone cannot restore it.
            # Restore only this validated run tree, without following symlinks.
            for directory, _subdirectories, _files in os.walk(resolved, followlinks=False):
                os.chmod(directory, os.stat(directory).st_mode | stat.S_IRWXU)
            shutil.rmtree(resolved, onexc=restore_write_and_retry)
        except OSError as exc:
            raise SandboxError("failed to clean isolated execution workspace") from exc
