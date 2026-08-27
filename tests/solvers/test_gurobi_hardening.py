from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from mathmodel_ai.core.errors import SolverUnavailableError
from mathmodel_ai.files.storage import LocalFileStore
from mathmodel_ai.sandbox.executor import SandboxExecution, SandboxExecutor, SecretMount
from mathmodel_ai.schemas.execution import ExecutionRecord, ExecutionStatus, SandboxLimits
from mathmodel_ai.schemas.program import GeneratedProgram
from mathmodel_ai.schemas.solver import (
    SolverHealth,
    SolverName,
    SolverOptions,
    SolverStatus,
)
from mathmodel_ai.solvers.base import RawSolverOutput
from mathmodel_ai.solvers.gurobi import GurobiSolver, canonicalize_gurobi_status
from tests.mathematical.helpers import lp_model, nlp_model


def _limits() -> SandboxLimits:
    return SandboxLimits(
        cpu_cores=1,
        memory_mb=64,
        timeout_seconds=20,
        pids_limit=16,
        max_output_bytes=1024,
        max_artifacts=2,
        max_artifact_bytes=1024,
    )


class MissingImageRunner:
    def __call__(self, command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        del kwargs
        return subprocess.CompletedProcess(command, 1, stdout=b"", stderr=b"missing")


class StubGurobiSolver(GurobiSolver):
    def __init__(self, *, sandbox: SandboxExecutor, store: LocalFileStore) -> None:
        super().__init__(sandbox=sandbox, store=store)
        self.captured_program: GeneratedProgram | None = None
        self.raw_output: RawSolverOutput | None = RawSolverOutput(
            solver_version="fixture",
            native_status=9,
            native_status_name="TIME_LIMIT",
            success=False,
            message="time limit reached with incumbent",
            objective=30,
            variables={"x": 10, "y": 0},
            runtime_seconds=12,
        )
        self.execution_status = ExecutionStatus.SUCCEEDED

    def health_check(self) -> SolverHealth:
        return SolverHealth(
            solver_name=SolverName.GUROBI,
            available=True,
            version="fixture",
            reason="protocol stub available without a license-dependent solve",
        )

    def _run_program(
        self,
        program: GeneratedProgram,
        *,
        secret_mounts: tuple[SecretMount, ...] = (),
    ) -> tuple[SandboxExecution, RawSolverOutput | None, GeneratedProgram]:
        del secret_mounts
        self.captured_program = program
        entrypoint = next(item for item in program.files if item.path == program.entrypoint)
        assert entrypoint.sha256 is not None
        now = datetime.now(UTC)
        execution = ExecutionRecord(
            project_id=program.project_id,
            problem_id=program.problem_id,
            code_hash=entrypoint.sha256,
            executed_bundle_hash=program.code_hash,
            code_artifact_id=uuid4(),
            execution_origin=program.execution_origin,
            model_digest=program.model_digest,
            generated_program_id=program.program_id,
            image="gurobi:stub",
            image_id="sha256:stub",
            start_time=now,
            end_time=now,
            runtime_seconds=12,
            status=self.execution_status,
            exit_code=0 if self.execution_status is ExecutionStatus.SUCCEEDED else None,
            limits=_limits(),
            error=(
                None if self.execution_status is ExecutionStatus.SUCCEEDED else "fixture timeout"
            ),
        )
        return SandboxExecution(record=execution, artifact_records=[]), self.raw_output, program


@pytest.mark.parametrize(
    ("native", "canonical"),
    [
        (2, SolverStatus.OPTIMAL),
        (3, SolverStatus.INFEASIBLE),
        (4, SolverStatus.INFEASIBLE_OR_UNBOUNDED),
        (5, SolverStatus.UNBOUNDED),
        (7, SolverStatus.ITERATION_LIMIT),
        (9, SolverStatus.TIME_LIMIT),
        (12, SolverStatus.NUMERICAL_ERROR),
        (13, SolverStatus.FEASIBLE),
        (999, SolverStatus.UNKNOWN),
    ],
)
def test_gurobi_native_status_mapping_is_canonical(
    native: int,
    canonical: SolverStatus,
) -> None:
    assert canonicalize_gurobi_status(native) is canonical


def test_gurobi_translation_propagates_options_and_preserves_time_limit_incumbent(
    tmp_path: Path,
) -> None:
    store = LocalFileStore(tmp_path / "store")
    solver = StubGurobiSolver(
        sandbox=SandboxExecutor(
            store=store,
            root=tmp_path / "runs",
            image="gurobi:stub",
            limits=_limits(),
            runner=MissingImageRunner(),
        ),
        store=store,
    )

    outcome = solver.solve(
        lp_model(),
        SolverOptions(time_limit_seconds=12, iteration_limit=100, mip_gap=0.01),
    )

    assert solver.captured_program is not None
    source = solver.captured_program.files[0].content
    assert '"time_limit_seconds":12.0' in source
    assert '"iteration_limit":100' in source
    assert '"mip_gap":0.01' in source
    assert outcome.result.status is SolverStatus.TIME_LIMIT
    assert outcome.result.is_feasible is True
    assert outcome.result.is_optimal is False
    assert outcome.result.objective_value == pytest.approx(30)


def test_gurobi_execution_timeout_maps_to_time_limit_without_fake_optimality(
    tmp_path: Path,
) -> None:
    store = LocalFileStore(tmp_path / "store")
    solver = StubGurobiSolver(
        sandbox=SandboxExecutor(
            store=store,
            root=tmp_path / "runs",
            image="gurobi:stub",
            limits=_limits(),
            runner=MissingImageRunner(),
        ),
        store=store,
    )
    solver.raw_output = None
    solver.execution_status = ExecutionStatus.TIMEOUT

    outcome = solver.solve(lp_model(), SolverOptions())

    assert outcome.result.status is SolverStatus.TIME_LIMIT
    assert outcome.result.is_feasible is False
    assert outcome.result.is_optimal is False


def test_gurobi_rejects_unsupported_nonlinear_translation_and_missing_license(
    tmp_path: Path,
) -> None:
    store = LocalFileStore(tmp_path / "store")
    solver = GurobiSolver(
        sandbox=SandboxExecutor(
            store=store,
            root=tmp_path / "runs",
            image="missing:gurobi",
            limits=_limits(),
            runner=MissingImageRunner(),
        ),
        store=store,
        license_file=None,
    )

    assessment = solver.assess_support(nlp_model())
    assert assessment.supported is False
    assert any("supports LP and linear integer" in reason for reason in assessment.reasons)
    with pytest.raises(SolverUnavailableError, match="SOLVER_UNAVAILABLE"):
        solver.solve(lp_model(), SolverOptions())
