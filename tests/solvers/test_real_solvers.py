import subprocess
from pathlib import Path

import pytest

from mathmodel_ai.core.errors import SolverUnavailableError
from mathmodel_ai.files.storage import LocalFileStore
from mathmodel_ai.mathematical.algorithms import AlgorithmSelector
from mathmodel_ai.sandbox.executor import SandboxExecutor
from mathmodel_ai.schemas.execution import ExecutionStatus, SandboxLimits
from mathmodel_ai.schemas.mathematical import ExpressionKind, MathExpression
from mathmodel_ai.schemas.solver import SolverName, SolverOptions, SolverStatus
from mathmodel_ai.solvers.base import RawSolverOutput
from mathmodel_ai.solvers.gurobi import GurobiSolver
from mathmodel_ai.solvers.ortools import ORToolsSolver
from mathmodel_ai.solvers.router import SolverRouter
from mathmodel_ai.solvers.scipy import SciPySolver, canonicalize_scipy_status
from tests.mathematical.helpers import (
    infeasible_model,
    lp_model,
    milp_model,
    nlp_model,
    unbounded_model,
)

IMAGE = "mathmodel-ai-solver:phase4"


def _image_available() -> bool:
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


def _limits(*, timeout: float = 20) -> SandboxLimits:
    return SandboxLimits(
        cpu_cores=1,
        memory_mb=768,
        timeout_seconds=timeout,
        pids_limit=64,
        max_output_bytes=32_768,
        max_artifacts=10,
        max_artifact_bytes=1_048_576,
    )


def _solver(tmp_path: Path) -> SciPySolver:
    store = LocalFileStore(tmp_path / "store")
    return SciPySolver(
        sandbox=SandboxExecutor(
            store=store,
            root=tmp_path / "runs",
            image=IMAGE,
            limits=_limits(),
        ),
        store=store,
    )


pytestmark = [
    pytest.mark.solver,
    pytest.mark.skipif(not _image_available(), reason=f"Docker image {IMAGE} is unavailable"),
]


@pytest.mark.parametrize(
    ("factory", "status", "objective"),
    [
        (lp_model, SolverStatus.OPTIMAL, 30.0),
        (milp_model, SolverStatus.OPTIMAL, 12.0),
        (nlp_model, SolverStatus.OPTIMAL, 0.0),
        (infeasible_model, SolverStatus.INFEASIBLE, None),
        (unbounded_model, SolverStatus.UNBOUNDED, None),
    ],
)
def test_real_scipy_solver_statuses_and_values(
    tmp_path: Path,
    factory: object,
    status: SolverStatus,
    objective: float | None,
) -> None:
    model = factory()  # type: ignore[operator]
    outcome = _solver(tmp_path).solve(
        model,
        SolverOptions(initial_point={"x": 0, "y": 0}),
    )

    assert outcome.result.status is status
    assert outcome.execution.record.status is ExecutionStatus.SUCCEEDED
    assert outcome.execution.record.is_mock is False
    assert outcome.execution.record.image_id is not None
    assert outcome.program.is_mock is False
    assert outcome.program.files[0].artifact_id == outcome.execution.record.code_artifact_id
    if objective is not None:
        assert outcome.result.objective_value == pytest.approx(objective, abs=1e-6)
        assert outcome.result.is_feasible is True
        assert outcome.result.feasibility is not None
        assert outcome.result.feasibility.max_constraint_violation == pytest.approx(0, abs=1e-6)
    else:
        assert outcome.result.objective_value is None
        assert outcome.result.is_optimal is False


def test_real_lp_and_milp_decision_values_are_not_mocked_or_rounded_relaxation(
    tmp_path: Path,
) -> None:
    lp = _solver(tmp_path / "lp").solve(lp_model(), SolverOptions())
    milp = _solver(tmp_path / "milp").solve(milp_model(), SolverOptions())

    assert lp.result.variable_values == pytest.approx({"x": 10, "y": 0}, abs=1e-7)
    assert milp.result.variable_values == pytest.approx({"x": 0, "y": 2}, abs=1e-7)
    assert 4 * milp.result.variable_values["x"] + 3 * milp.result.variable_values["y"] <= 6


def test_real_ortools_cp_sat_solves_integral_model(tmp_path: Path) -> None:
    store = LocalFileStore(tmp_path / "store")
    solver = ORToolsSolver(
        sandbox=SandboxExecutor(
            store=store,
            root=tmp_path / "runs",
            image=IMAGE,
            limits=_limits(),
        ),
        store=store,
    )

    outcome = solver.solve(milp_model(), SolverOptions())

    assert outcome.result.solver_name is SolverName.ORTOOLS
    assert outcome.result.status is SolverStatus.OPTIMAL
    assert outcome.result.objective_value == pytest.approx(12)
    assert outcome.result.variable_values == pytest.approx({"x": 0, "y": 2})


def test_router_falls_back_from_unlicensed_gurobi_to_real_scipy(tmp_path: Path) -> None:
    store = LocalFileStore(tmp_path / "store")
    sandbox = SandboxExecutor(
        store=store,
        root=tmp_path / "runs",
        image=IMAGE,
        limits=_limits(),
    )
    router = SolverRouter(
        [
            GurobiSolver(sandbox=sandbox, store=store, license_file=None),
            SciPySolver(sandbox=sandbox, store=store),
            ORToolsSolver(sandbox=sandbox, store=store),
        ]
    )
    model = milp_model()

    routed = router.route(model, AlgorithmSelector().select(model), SolverOptions())

    assert routed.solver.name is SolverName.SCIPY
    assert routed.decision.fallback_used is True
    assert routed.decision.attempted_families[0].value == "GUROBI"


def test_cp_sat_rejects_nonintegral_coefficients_without_hidden_scaling(tmp_path: Path) -> None:
    store = LocalFileStore(tmp_path / "store")
    solver = ORToolsSolver(
        sandbox=SandboxExecutor(
            store=store,
            root=tmp_path / "runs",
            image=IMAGE,
            limits=_limits(),
        ),
        store=store,
    )
    model = milp_model()
    assert model.objective is not None
    nonintegral = model.model_copy(
        update={
            "objective": model.objective.model_copy(
                update={
                    "expression": MathExpression(
                        kind=ExpressionKind.MULTIPLY,
                        operands=[
                            MathExpression(kind=ExpressionKind.CONSTANT, value=0.5),
                            MathExpression(kind=ExpressionKind.SYMBOL, symbol="x"),
                        ],
                    )
                }
            )
        }
    )

    assessment = solver.assess_support(nonintegral)
    assert assessment.supported is False
    assert any("no explicit scaling plan" in reason for reason in assessment.reasons)


def test_router_reports_solver_unavailable_without_crashing(tmp_path: Path) -> None:
    store = LocalFileStore(tmp_path / "store")
    solver = SciPySolver(
        sandbox=SandboxExecutor(
            store=store,
            root=tmp_path / "runs",
            image="missing-phase4-solver:test",
            limits=_limits(),
        ),
        store=store,
    )
    model = lp_model()

    with pytest.raises(SolverUnavailableError, match="SOLVER_UNAVAILABLE"):
        SolverRouter([solver]).route(
            model,
            AlgorithmSelector().select(model),
            SolverOptions(),
        )


class TimeoutRunner:
    def __call__(self, command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        if command[1:3] == ["image", "inspect"]:
            return subprocess.CompletedProcess(command, 0, stdout=b"sha256:image", stderr=b"")
        if command[1:3] == ["rm", "--force"]:
            return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")
        raise subprocess.TimeoutExpired(command, timeout=float(kwargs["timeout"]))


def test_scipy_adapter_canonicalizes_sandbox_timeout(tmp_path: Path) -> None:
    store = LocalFileStore(tmp_path / "store")
    solver = SciPySolver(
        sandbox=SandboxExecutor(
            store=store,
            root=tmp_path / "runs",
            image="solver:test",
            limits=_limits(timeout=0.01),
            runner=TimeoutRunner(),
        ),
        store=store,
    )

    outcome = solver.solve(lp_model(), SolverOptions())

    assert outcome.execution.record.status is ExecutionStatus.TIMEOUT
    assert outcome.result.status is SolverStatus.TIME_LIMIT
    assert outcome.result.is_optimal is False


def test_scipy_nlp_statuses_are_not_misclassified_as_lp_unbounded() -> None:
    raw = RawSolverOutput(
        native_status=3,
        success=False,
        message="More than 3*n iterations in LSQ subproblem",
        runtime_seconds=0.01,
    )

    assert canonicalize_scipy_status(nlp_model(), raw) is SolverStatus.NUMERICAL_ERROR
    assert canonicalize_scipy_status(lp_model(), raw) is SolverStatus.UNBOUNDED
    assert (
        canonicalize_scipy_status(
            nlp_model(),
            raw.model_copy(update={"native_status": 9}),
        )
        is SolverStatus.ITERATION_LIMIT
    )
