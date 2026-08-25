import json
import subprocess
from pathlib import Path
from uuid import uuid4

import pytest

from mathmodel_ai.data.quality_gates import execution_quality_gate
from mathmodel_ai.files.storage import LocalFileStore
from mathmodel_ai.sandbox.executor import SandboxExecutor
from mathmodel_ai.schemas.execution import ExecutionStatus, SandboxLimits
from mathmodel_ai.schemas.files import FileKind, RegisteredFile
from mathmodel_ai.schemas.quality import QualityGateStatus

IMAGE = "mathmodel-ai-sandbox:phase3"


def _sandbox_image_available() -> bool:
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", IMAGE],
            capture_output=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


pytestmark = [
    pytest.mark.sandbox,
    pytest.mark.skipif(
        not _sandbox_image_available(),
        reason=f"Docker image {IMAGE} is not available",
    ),
]


def _limits(*, timeout: float = 5) -> SandboxLimits:
    return SandboxLimits(
        cpu_cores=0.5,
        memory_mb=128,
        timeout_seconds=timeout,
        pids_limit=32,
        max_output_bytes=8192,
        max_artifacts=5,
        max_artifact_bytes=8192,
    )


def test_real_container_is_non_root_networkless_and_read_only(tmp_path: Path) -> None:
    store = LocalFileStore(tmp_path / "store")
    project_id = uuid4()
    problem_id = uuid4()
    input_file_id = uuid4()
    stored_input = store.store_artifact(
        b"value\n42\n",
        project_id=project_id,
        artifact_id=uuid4(),
        filename="input.csv",
    )
    input_file = RegisteredFile(
        file_id=input_file_id,
        project_id=project_id,
        problem_id=problem_id,
        original_name="input.csv",
        safe_name="input.csv",
        extension=".csv",
        kind=FileKind.CSV,
        detected_mime_type="text/csv",
        size_bytes=stored_input.size_bytes,
        sha256=stored_input.sha256,
        storage_key=stored_input.storage_key,
    )
    executor = SandboxExecutor(
        store=store,
        root=tmp_path / "runs",
        image=IMAGE,
        limits=_limits(),
    )
    code = """
import json
import os
import socket
from pathlib import Path

checks = {"uid": os.geteuid()}
input_path = next(Path("/workspace/inputs").iterdir())
checks["input_readable"] = input_path.read_text() == "value\\n42\\n"
try:
    input_path.write_text("bad")
    checks["input_read_only"] = False
except OSError:
    checks["input_read_only"] = True
try:
    Path("/forbidden.txt").write_text("bad")
    checks["root_read_only"] = False
except OSError:
    checks["root_read_only"] = True
try:
    Path("/workspace/main.py").write_text("bad")
    checks["workspace_read_only"] = False
except OSError:
    checks["workspace_read_only"] = True
try:
    socket.create_connection(("1.1.1.1", 53), timeout=0.2)
    checks["network_disabled"] = False
except OSError:
    checks["network_disabled"] = True
Path("/output/isolation.json").write_text(json.dumps(checks))
print(json.dumps(checks))
"""

    outcome = executor.execute(
        code,
        project_id=project_id,
        problem_id=problem_id,
        input_files=[input_file],
    )

    assert outcome.record.status is ExecutionStatus.SUCCEEDED
    observed = json.loads(outcome.record.stdout)
    assert observed == {
        "uid": 65532,
        "input_readable": True,
        "input_read_only": True,
        "root_read_only": True,
        "workspace_read_only": True,
        "network_disabled": True,
    }
    assert (
        execution_quality_gate(outcome.record, store, outcome.artifact_records).status
        is QualityGateStatus.PASS
    )


def test_real_container_timeout_is_enforced(tmp_path: Path) -> None:
    executor = SandboxExecutor(
        store=LocalFileStore(tmp_path / "store"),
        root=tmp_path / "runs",
        image=IMAGE,
        limits=_limits(timeout=0.5),
    )

    outcome = executor.execute("while True: pass", project_id=uuid4(), problem_id=uuid4())

    assert outcome.record.status is ExecutionStatus.TIMEOUT
    assert outcome.record.exit_code == 124
