from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from time import monotonic
from uuid import UUID, uuid4

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from mathmodel_ai.files.storage import FileStore
from mathmodel_ai.paper.rendering import LaTeXSafetyValidator, RenderedPaper
from mathmodel_ai.schemas.paper import (
    PaperArtifact,
    PaperArtifactKind,
    PaperCompileRecord,
    PaperCompileStatus,
)

CommandRunner = Callable[..., subprocess.CompletedProcess[bytes]]
_IMAGE_NAME = re.compile(r"^FIG-[A-Za-z0-9_-]+\.png$")
_FATAL_WARNING = re.compile(
    r"undefined (?:citation|reference)|citation [`'].+[`'] undefined|"
    r"file [`'].+[`'] not found|missing character",
    re.IGNORECASE,
)
_WARNING = re.compile(r"warning|undefined|overfull|underfull", re.IGNORECASE)


@dataclass(frozen=True)
class PaperCompilation:
    record: PaperCompileRecord
    artifacts: tuple[PaperArtifact, ...]


class PDFCompiler:
    """Compile system-rendered TeX in a non-root, networkless Docker container."""

    def __init__(
        self,
        *,
        store: FileStore,
        image: str = "mathmodel-ai-paper:phase6",
        timeout_seconds: float = 60.0,
        cpu_cores: float = 1.0,
        memory_mb: int = 768,
        pids_limit: int = 128,
        docker_binary: str = "docker",
        runner: CommandRunner = subprocess.run,
    ) -> None:
        if timeout_seconds <= 0 or cpu_cores <= 0 or memory_mb < 128 or pids_limit < 16:
            raise ValueError("invalid PDF compiler limits")
        self._store = store
        self._image = image
        self._timeout = timeout_seconds
        self._cpu = cpu_cores
        self._memory = memory_mb
        self._pids = pids_limit
        self._docker = docker_binary
        self._runner = runner
        self._logger = logging.getLogger("mathmodel_ai.paper.compiler")

    def compile(
        self,
        rendered: RenderedPaper,
        *,
        project_id: UUID,
        problem_id: UUID,
        paper_id: UUID,
        paper_version: int,
        figure_images: Mapping[str, bytes] | None = None,
    ) -> PaperCompilation:
        LaTeXSafetyValidator().validate_rendered(rendered.tex)
        compile_id = uuid4()
        container_name = f"mathmodel-paper-{compile_id.hex}"
        started = datetime.now(UTC)
        clock = monotonic()
        artifacts = [
            self._store_artifact(
                rendered.tex.encode("utf-8"),
                project_id,
                problem_id,
                paper_id,
                paper_version,
                PaperArtifactKind.TEX,
                "paper.tex",
                "application/x-tex",
            ),
            self._store_artifact(
                rendered.bibliography.encode("utf-8"),
                project_id,
                problem_id,
                paper_id,
                paper_version,
                PaperArtifactKind.BIB,
                "references.bib",
                "application/x-bibtex",
            ),
        ]
        image_id = self._image_id()
        if image_id is None:
            return PaperCompilation(
                record=self._record(
                    compile_id,
                    paper_id,
                    paper_version,
                    rendered,
                    started,
                    clock,
                    PaperCompileStatus.UNAVAILABLE,
                    None,
                    "",
                    "",
                    None,
                    None,
                    "configured paper compiler image is unavailable",
                ),
                artifacts=tuple(artifacts),
            )
        run_root = Path(tempfile.mkdtemp(prefix="mathmodel-paper-"))
        workspace = run_root / "workspace"
        output = run_root / "output"
        try:
            workspace.mkdir()
            output.mkdir()
            figures = workspace / "figures"
            figures.mkdir()
            (workspace / "paper.tex").write_text(rendered.tex, encoding="utf-8")
            (workspace / "references.bib").write_text(rendered.bibliography, encoding="utf-8")
            for name, data in (figure_images or {}).items():
                if _IMAGE_NAME.fullmatch(name) is None:
                    raise ValueError("unsafe figure filename")
                (figures / name).write_bytes(data)
            self._restrict(workspace, output)
            try:
                result = self._runner(
                    self._command(container_name, workspace, output, image_id),
                    capture_output=True,
                    check=False,
                    timeout=self._timeout,
                )
                status = (
                    PaperCompileStatus.SUCCEEDED
                    if result.returncode == 0
                    else PaperCompileStatus.FAILED
                )
                exit_code = result.returncode
                stdout = self._bounded(result.stdout)
                stderr = self._bounded(result.stderr)
                error = None if status is PaperCompileStatus.SUCCEEDED else "xelatex failed"
            except subprocess.TimeoutExpired as exc:
                self._force_remove(container_name)
                status = PaperCompileStatus.TIMEOUT
                exit_code = 124
                stdout = self._bounded(exc.stdout or b"")
                stderr = self._bounded(exc.stderr or b"")
                error = f"PDF compilation exceeded {self._timeout} seconds"
            except (OSError, subprocess.SubprocessError) as exc:
                status = PaperCompileStatus.FAILED
                exit_code = None
                stdout = ""
                stderr = ""
                error = f"Docker invocation failed: {type(exc).__name__}"
            finally:
                self._force_remove(container_name)
            warnings, fatal_warnings = self._classify_warnings(stdout, stderr)
            if status is PaperCompileStatus.SUCCEEDED and fatal_warnings:
                status = PaperCompileStatus.FAILED
                error = "compiler emitted a fatal document-integrity warning"
            pdf_path = output / "paper.pdf"
            pdf_artifact: PaperArtifact | None = None
            page_count: int | None = None
            if status is PaperCompileStatus.SUCCEEDED:
                if not pdf_path.is_file():
                    status = PaperCompileStatus.FAILED
                    error = "compiler reported success without paper.pdf"
                else:
                    pdf = pdf_path.read_bytes()
                    if not pdf.startswith(b"%PDF-") or len(pdf) < 100:
                        status = PaperCompileStatus.FAILED
                        error = "compiler output is not a valid non-empty PDF"
                    else:
                        try:
                            reader = PdfReader(BytesIO(pdf), strict=True)
                            page_count = len(reader.pages)
                        except (PdfReadError, ValueError, TypeError, KeyError) as exc:
                            status = PaperCompileStatus.FAILED
                            error = (
                                f"compiler output failed strict PDF parsing: {type(exc).__name__}"
                            )
                        if not page_count:
                            status = PaperCompileStatus.FAILED
                            error = "compiler output has no readable PDF pages"
                        if status is PaperCompileStatus.SUCCEEDED:
                            pdf_artifact = self._store_artifact(
                                pdf,
                                project_id,
                                problem_id,
                                paper_id,
                                paper_version,
                                PaperArtifactKind.PDF,
                                "paper.pdf",
                                "application/pdf",
                            )
                            artifacts.append(pdf_artifact)
            record = self._record(
                compile_id,
                paper_id,
                paper_version,
                rendered,
                started,
                clock,
                status,
                exit_code,
                stdout,
                stderr,
                image_id,
                pdf_artifact,
                error,
                warnings,
                fatal_warnings,
                page_count,
            )
            return PaperCompilation(record=record, artifacts=tuple(artifacts))
        finally:
            shutil.rmtree(run_root, ignore_errors=True)

    def _command(
        self, container_name: str, workspace: Path, output: Path, image_id: str
    ) -> list[str]:
        return [
            self._docker,
            "run",
            "--rm",
            "--name",
            container_name,
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--user",
            "65532:65532",
            "--cpus",
            str(self._cpu),
            "--memory",
            f"{self._memory}m",
            "--pids-limit",
            str(self._pids),
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=128m",
            "--mount",
            f"type=bind,source={workspace},target=/workspace,readonly",
            "--mount",
            f"type=bind,source={output},target=/output",
            image_id,
        ]

    def _image_id(self) -> str | None:
        try:
            result = self._runner(
                [self._docker, "image", "inspect", "--format", "{{.Id}}", self._image],
                capture_output=True,
                check=False,
                timeout=15,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if result.returncode != 0:
            return None
        value = result.stdout.decode("utf-8", errors="replace").strip()
        return value or None

    def _force_remove(self, container_name: str) -> None:
        try:
            self._runner(
                [self._docker, "rm", "--force", container_name],
                capture_output=True,
                check=False,
                timeout=15,
            )
        except (OSError, subprocess.SubprocessError):
            self._logger.warning("failed to remove PDF compiler container")

    @staticmethod
    def _restrict(workspace: Path, output: Path) -> None:
        if os.name != "nt":
            for directory in (workspace, workspace / "figures"):
                directory.chmod(0o555)
            for file in workspace.rglob("*"):
                if file.is_file():
                    file.chmod(0o444)
            output.chmod(0o777)

    @staticmethod
    def _bounded(value: bytes) -> str:
        return value[:1_000_000].decode("utf-8", errors="replace")

    @staticmethod
    def _classify_warnings(stdout: str, stderr: str) -> tuple[list[str], list[str]]:
        combined = f"{stdout}\n{stderr}"
        final_pass_start = combined.rfind("This is XeTeX")
        final_pass = combined[final_pass_start:] if final_pass_start >= 0 else combined
        lines = [line.strip() for line in final_pass.splitlines() if _WARNING.search(line)]
        warnings = list(dict.fromkeys(lines))
        return warnings, [line for line in warnings if _FATAL_WARNING.search(line)]

    def _store_artifact(
        self,
        data: bytes,
        project_id: UUID,
        problem_id: UUID,
        paper_id: UUID,
        paper_version: int,
        kind: PaperArtifactKind,
        name: str,
        mime_type: str,
    ) -> PaperArtifact:
        artifact_id = uuid4()
        stored = self._store.store_artifact(
            data,
            project_id=project_id,
            artifact_id=artifact_id,
            filename=name,
        )
        return PaperArtifact(
            artifact_id=artifact_id,
            project_id=project_id,
            problem_id=problem_id,
            paper_id=paper_id,
            paper_version=paper_version,
            kind=kind,
            name=name,
            mime_type=mime_type,
            size_bytes=stored.size_bytes,
            sha256=stored.sha256,
            storage_key=stored.storage_key,
        )

    def _record(
        self,
        compile_id: UUID,
        paper_id: UUID,
        paper_version: int,
        rendered: RenderedPaper,
        started: datetime,
        clock: float,
        status: PaperCompileStatus,
        exit_code: int | None,
        stdout: str,
        stderr: str,
        image_id: str | None,
        pdf_artifact: PaperArtifact | None,
        error: str | None = None,
        warnings: list[str] | None = None,
        fatal_warnings: list[str] | None = None,
        page_count: int | None = None,
    ) -> PaperCompileRecord:
        del started
        return PaperCompileRecord(
            compile_id=compile_id,
            paper_id=paper_id,
            paper_version=paper_version,
            compiler="xelatex+bibtex",
            container_image=self._image,
            container_image_id=image_id,
            status=status,
            exit_code=exit_code,
            runtime_seconds=max(0.0, monotonic() - clock),
            stdout=stdout,
            stderr=stderr,
            warnings=warnings or [],
            fatal_warnings=fatal_warnings or [],
            page_count=page_count,
            tex_hash=rendered.tex_hash,
            bib_hash=rendered.bib_hash,
            pdf_artifact_id=(pdf_artifact.artifact_id if pdf_artifact else None),
            error=error,
        )
