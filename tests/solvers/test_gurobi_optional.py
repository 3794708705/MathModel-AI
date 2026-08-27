import os
from pathlib import Path

import pytest

from mathmodel_ai.files.storage import LocalFileStore
from mathmodel_ai.sandbox.executor import SandboxExecutor
from mathmodel_ai.schemas.execution import SandboxLimits
from mathmodel_ai.schemas.solver import SolverOptions, SolverStatus
from mathmodel_ai.solvers.gurobi import GurobiSolver
from tests.mathematical.helpers import lp_model

GUROBI_IMAGE = os.getenv("MM_GUROBI_SANDBOX_IMAGE")
GUROBI_LICENSE = os.getenv("MM_GUROBI_LICENSE_FILE")

pytestmark = [pytest.mark.gurobi, pytest.mark.solver]


@pytest.mark.skipif(
    not GUROBI_IMAGE or not GUROBI_LICENSE,
    reason=(
        "optional Gurobi integration requires MM_GUROBI_SANDBOX_IMAGE and MM_GUROBI_LICENSE_FILE"
    ),
)
def test_optional_gurobi_runtime_solves_real_lp(tmp_path: Path) -> None:
    assert GUROBI_IMAGE is not None
    assert GUROBI_LICENSE is not None
    store = LocalFileStore(tmp_path / "store")
    solver = GurobiSolver(
        sandbox=SandboxExecutor(
            store=store,
            root=tmp_path / "runs",
            image=GUROBI_IMAGE,
            limits=SandboxLimits(
                cpu_cores=1,
                memory_mb=768,
                timeout_seconds=20,
                pids_limit=64,
                max_output_bytes=32_768,
                max_artifacts=10,
                max_artifact_bytes=1_048_576,
            ),
        ),
        store=store,
        license_file=Path(GUROBI_LICENSE),
    )
    health = solver.health_check()
    if not health.available:
        pytest.skip(health.reason)

    outcome = solver.solve(lp_model(), SolverOptions())

    assert outcome.result.status is SolverStatus.OPTIMAL
    assert outcome.result.objective_value == pytest.approx(30)
    assert outcome.result.variable_values == pytest.approx({"x": 10, "y": 0})
    assert outcome.execution.record.is_mock is False
