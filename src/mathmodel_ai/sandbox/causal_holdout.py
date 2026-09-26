"""Run a pointwise holdout through an isolated, networkless Docker predictor.

The trusted host owns the full CSV and sends only training labels and one
held-out pre-outcome feature per prediction. The generated program never has
the source CSV mounted or a batch of held-out features in its workspace.
"""

import hashlib
import json
import queue
import shutil
import subprocess
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from uuid import UUID, uuid4

from mathmodel_ai.files.storage import FileStore
from mathmodel_ai.schemas.execution import (
    ExecutionArtifact,
    ExecutionOrigin,
    ExecutionRecord,
    ExecutionStatus,
    SandboxLimits,
)
from mathmodel_ai.schemas.files import ArtifactKind, ArtifactRecord
from mathmodel_ai.verification.causal_binary import CausalBinarySpec, CausalFeature
from mathmodel_ai.verification.causal_holdout import (
    CausalHoldoutResult,
    causal_trace_payload,
    evaluate_causal_holdout,
    verify_causal_trace,
)


@dataclass(frozen=True)
class IsolatedCausalHoldout:
    result: CausalHoldoutResult | None
    execution: ExecutionRecord
    artifacts: tuple[ArtifactRecord, ...]


class _ProtocolError(Exception):
    pass


class _DockerPredictor:
    def __init__(
        self,
        process: subprocess.Popen[str],
        *,
        deadline: float,
    ) -> None:
        self._process = process
        self._deadline = deadline
        self._responses: queue.Queue[str | None] = queue.Queue(maxsize=1)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        assert self._process.stdout is not None
        while True:
            line = self._process.stdout.readline(8193)
            try:
                self._responses.put(line if line else None, timeout=1)
            except queue.Full:
                return
            if not line:
                return

    def _request(self, value: dict[str, object]) -> dict[str, object]:
        remaining = self._deadline - monotonic()
        if remaining <= 0:
            raise TimeoutError("isolated predictor exceeded its total time limit")
        assert self._process.stdin is not None
        try:
            self._process.stdin.write(json.dumps(value, separators=(",", ":")) + "\n")
            self._process.stdin.flush()
            line = self._responses.get(timeout=remaining)
        except (BrokenPipeError, OSError, queue.Empty) as exc:
            if isinstance(exc, queue.Empty):
                raise TimeoutError("isolated predictor response timed out") from exc
            raise _ProtocolError("isolated predictor pipe closed") from exc
        if line is None or len(line) > 8192 or not line.endswith("\n"):
            raise _ProtocolError("isolated predictor response missing or oversized")
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            raise _ProtocolError("isolated predictor response is not JSON") from exc
        if not isinstance(response, dict) or not all(isinstance(key, str) for key in response):
            raise _ProtocolError("isolated predictor response is not an object")
        if "error" in response:
            raise _ProtocolError("isolated predictor rejected a request")
        return response

    def fit(self, feature: CausalFeature, outcome: float) -> None:
        response = self._request({"command": "fit", "feature": asdict(feature), "outcome": outcome})
        if response != {"ok": True}:
            raise _ProtocolError("isolated predictor fit acknowledgement is invalid")

    def predict(self, feature: CausalFeature) -> float:
        response = self._request({"command": "predict", "feature": asdict(feature)})
        if set(response) != {"probability"}:
            raise _ProtocolError("isolated predictor prediction response is invalid")
        probability = response["probability"]
        if isinstance(probability, bool) or not isinstance(probability, (int, float)):
            raise _ProtocolError("isolated predictor probability is not numeric")
        return float(probability)

    def stop(self) -> None:
        if self._request({"command": "stop"}) != {"ok": True}:
            raise _ProtocolError("isolated predictor stop acknowledgement is invalid")


def _artifact(
    store: FileStore,
    data: bytes,
    *,
    project_id: UUID,
    problem_id: UUID,
    run_id: UUID,
    name: str,
    kind: ArtifactKind,
    mime_type: str,
) -> ArtifactRecord:
    artifact_id = uuid4()
    stored = store.store_artifact(
        data, project_id=project_id, artifact_id=artifact_id, filename=name
    )
    return ArtifactRecord(
        artifact_id=artifact_id,
        project_id=project_id,
        problem_id=problem_id,
        execution_run_id=run_id,
        kind=kind,
        name=name,
        mime_type=mime_type,
        size_bytes=stored.size_bytes,
        sha256=stored.sha256,
        storage_key=stored.storage_key,
    )


def run_isolated_causal_holdout(
    source: bytes,
    spec: CausalBinarySpec,
    predictor_code: str,
    *,
    fraction: float,
    salt: str,
    store: FileStore,
    root: Path,
    image: str,
    limits: SandboxLimits,
    project_id: UUID,
    problem_id: UUID,
    docker_binary: str = "docker",
    execution_origin: ExecutionOrigin = ExecutionOrigin.USER_CODE,
    model_digest: str | None = None,
    generated_program_id: UUID | None = None,
    formal_result_id: UUID | None = None,
    science_policy_sha256: str | None = None,
) -> IsolatedCausalHoldout:
    """Return a real execution record even when Docker or predictor code fails."""
    encoded = predictor_code.encode("utf-8")
    if science_policy_sha256 is not None and (
        len(science_policy_sha256) != 64
        or any(character not in "0123456789abcdef" for character in science_policy_sha256)
    ):
        raise ValueError("causal science policy digest is invalid")
    if not encoded or len(encoded) > 2 * 1024 * 1024:
        raise ValueError("isolated predictor code must contain at most 2 MiB")
    run_id = uuid4()
    code_artifact = _artifact(
        store,
        encoded,
        project_id=project_id,
        problem_id=problem_id,
        run_id=run_id,
        name="predictor.py",
        kind=ArtifactKind.GENERATED_CODE,
        mime_type="text/x-python",
    )
    driver = Path(__file__).with_name("causal_driver.py").read_bytes()
    driver_artifact = _artifact(
        store,
        driver,
        project_id=project_id,
        problem_id=problem_id,
        run_id=run_id,
        name="causal_driver.py",
        kind=ArtifactKind.GENERATED_CODE,
        mime_type="text/x-python",
    )
    bundle_hash = hashlib.sha256(
        hashlib.sha256(encoded).digest() + hashlib.sha256(driver).digest()
    ).hexdigest()
    started = datetime.now(UTC)
    clock_started = monotonic()
    deadline = clock_started + limits.timeout_seconds
    workdir = root.resolve(strict=False) / run_id.hex
    container_name = f"mathmodel-causal-{run_id.hex}"
    image_id: str | None = None
    result: CausalHoldoutResult | None = None
    trace_artifact: ArtifactRecord | None = None
    status = ExecutionStatus.FAILED
    exit_code: int | None = None
    error: str | None = None
    process: subprocess.Popen[str] | None = None
    try:
        inspected = subprocess.run(
            [docker_binary, "image", "inspect", "--format", "{{.Id}}", image],
            capture_output=True,
            check=False,
            timeout=min(10, limits.timeout_seconds),
        )
        if inspected.returncode != 0:
            status = ExecutionStatus.UNAVAILABLE
            raise _ProtocolError("isolated predictor image is unavailable")
        image_id = inspected.stdout.decode("ascii", errors="replace").strip()
        if not image_id.startswith("sha256:"):
            status = ExecutionStatus.UNAVAILABLE
            raise _ProtocolError("isolated predictor image digest is unavailable")
        workdir.mkdir(parents=True)
        (workdir / "predictor.py").write_bytes(encoded)
        (workdir / "driver.py").write_bytes(driver)
        memory = f"{limits.memory_mb}m"
        command = [
            docker_binary,
            "run",
            "--name",
            container_name,
            "--rm",
            "--interactive",
            "--network",
            "none",
            "--cpus",
            str(limits.cpu_cores),
            "--memory",
            memory,
            "--memory-swap",
            memory,
            "--pids-limit",
            str(limits.pids_limit),
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--user",
            "65532:65532",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,nodev,size=64m",
            "--mount",
            f"type=bind,source={workdir},target=/workspace,readonly",
            "--workdir",
            "/workspace",
            image_id,
            "python",
            "-I",
            "-B",
            "/workspace/driver.py",
        ]
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        predictor = _DockerPredictor(process, deadline=deadline)
        result = evaluate_causal_holdout(source, spec, predictor, fraction=fraction, salt=salt)
        predictor.stop()
        if process.stdin is not None:
            process.stdin.close()
        exit_code = process.wait(timeout=max(0.1, deadline - monotonic()))
        if exit_code != 0:
            raise _ProtocolError("isolated predictor exited unsuccessfully")
        trace = causal_trace_payload(spec, result, fraction=fraction, salt=salt)
        verify_causal_trace(source, trace)
        if len(trace) > limits.max_artifact_bytes:
            raise _ProtocolError("isolated prediction trace exceeds artifact limit")
        trace_artifact = _artifact(
            store,
            trace,
            project_id=project_id,
            problem_id=problem_id,
            run_id=run_id,
            name="causal-holdout.json",
            kind=ArtifactKind.VERIFICATION_TRACE,
            mime_type="application/json",
        )
        status = ExecutionStatus.SUCCEEDED
    except (OSError, subprocess.SubprocessError, _ProtocolError, ValueError) as exc:
        if isinstance(exc, (TimeoutError, subprocess.TimeoutExpired)):
            status = ExecutionStatus.TIMEOUT
        error = str(exc)[:500]
        result = None
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        if process is not None:
            try:
                subprocess.run(
                    [docker_binary, "rm", "--force", container_name],
                    capture_output=True,
                    check=False,
                    timeout=10,
                )
            except (OSError, subprocess.SubprocessError):
                pass
        shutil.rmtree(workdir, ignore_errors=True)
    ended = datetime.now(UTC)
    artifacts = (code_artifact, driver_artifact) + ((trace_artifact,) if trace_artifact else ())
    record = ExecutionRecord(
        run_id=run_id,
        project_id=project_id,
        problem_id=problem_id,
        code_hash=hashlib.sha256(encoded).hexdigest(),
        executed_bundle_hash=bundle_hash,
        code_artifact_id=code_artifact.artifact_id,
        execution_origin=execution_origin,
        model_digest=model_digest,
        generated_program_id=generated_program_id,
        image=image,
        image_id=image_id,
        environment={
            "executor": "docker-causal-stream",
            "source_sha256": spec.source_sha256,
            **({"formal_result_id": str(formal_result_id)} if formal_result_id else {}),
            **(
                {"science_policy_sha256": science_policy_sha256}
                if science_policy_sha256 is not None
                else {}
            ),
        },
        start_time=started,
        end_time=ended,
        runtime_seconds=max(0.0, monotonic() - clock_started),
        status=status,
        exit_code=exit_code if status is ExecutionStatus.SUCCEEDED else None,
        limits=limits,
        artifacts=[
            ExecutionArtifact(
                artifact_id=item.artifact_id,
                name=item.name,
                mime_type=item.mime_type,
                size_bytes=item.size_bytes,
                sha256=item.sha256,
                storage_key=item.storage_key,
            )
            for item in artifacts[2:]
        ],
        metrics={"heldout_predictions": len(result.predictions) if result else 0},
        error=error,
    )
    return IsolatedCausalHoldout(result=result, execution=record, artifacts=artifacts)


def verify_recorded_causal_holdout(
    source: bytes,
    run: IsolatedCausalHoldout,
    store: FileStore,
    *,
    expected_policy: tuple[CausalBinarySpec, float, str] | None = None,
) -> CausalHoldoutResult:
    """Check stored code, driver, execution identity and host-recorded trace."""
    record = run.execution
    if (
        record.status is not ExecutionStatus.SUCCEEDED
        or record.is_mock
        or not record.network_disabled
        or not record.non_root
        or not record.read_only_root
        or record.exit_code != 0
        or record.image_id is None
        or len(run.artifacts) != 3
        or len(record.artifacts) != 1
    ):
        raise ValueError("CAUSAL_EXECUTION_NOT_VERIFIABLE")
    code, driver, trace = run.artifacts
    if (
        code.kind is not ArtifactKind.GENERATED_CODE
        or driver.kind is not ArtifactKind.GENERATED_CODE
        or trace.kind is not ArtifactKind.VERIFICATION_TRACE
        or code.artifact_id != record.code_artifact_id
        or any(
            item.execution_run_id != record.run_id
            or item.project_id != record.project_id
            or item.problem_id != record.problem_id
            for item in run.artifacts
        )
        or record.artifacts[0].artifact_id != trace.artifact_id
        or record.artifacts[0].sha256 != trace.sha256
        or record.artifacts[0].storage_key != trace.storage_key
        or record.environment.get("source_sha256") != hashlib.sha256(source).hexdigest()
    ):
        raise ValueError("CAUSAL_EXECUTION_ARTIFACT_MISMATCH")
    for item in run.artifacts:
        data = store.read_bytes(item.storage_key)
        if len(data) != item.size_bytes or hashlib.sha256(data).hexdigest() != item.sha256:
            raise ValueError("CAUSAL_EXECUTION_ARTIFACT_HASH_MISMATCH")
    code_bytes = store.read_bytes(code.storage_key)
    driver_bytes = store.read_bytes(driver.storage_key)
    if driver_bytes != Path(__file__).with_name("causal_driver.py").read_bytes():
        raise ValueError("CAUSAL_EXECUTION_DRIVER_MISMATCH")
    if (
        hashlib.sha256(code_bytes).hexdigest() != record.code_hash
        or hashlib.sha256(
            hashlib.sha256(code_bytes).digest() + hashlib.sha256(driver_bytes).digest()
        ).hexdigest()
        != record.executed_bundle_hash
    ):
        raise ValueError("CAUSAL_EXECUTION_BUNDLE_MISMATCH")
    trace_bytes = store.read_bytes(trace.storage_key)
    result = verify_causal_trace(source, trace_bytes)
    if expected_policy is not None:
        spec, fraction, salt = expected_policy
        if causal_trace_payload(spec, result, fraction=fraction, salt=salt) != trace_bytes:
            raise ValueError("CAUSAL_EXECUTION_POLICY_MISMATCH")
    if run.result is not None and result != run.result:
        raise ValueError("CAUSAL_EXECUTION_RESULT_MISMATCH")
    return result
