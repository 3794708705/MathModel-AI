import math
import subprocess

import pytest

from mathmodel_ai.files.storage import LocalFileStore
from mathmodel_ai.mathematical.digests import mathematical_model_digest
from mathmodel_ai.sandbox.executor import SandboxExecutor
from mathmodel_ai.schemas.execution import SandboxLimits
from mathmodel_ai.schemas.independent_verification import (
    DynamicReplaySpec,
    IndependentStatus,
    MetricSpec,
    ScenarioSpec,
)
from mathmodel_ai.schemas.mathematical import VariableRole
from mathmodel_ai.solvers.router import SolverRouter
from mathmodel_ai.solvers.scipy import SciPySolver
from mathmodel_ai.verification.replay_integrity import audit_replay
from mathmodel_ai.verification.scenario_replay import ScenarioReplayer
from tests.mathematical.helpers import constant, multiply, symbol, variable
from tests.verification.helpers import phase5_model


def dynamic_model():
    model = phase5_model()
    equation = model.equations[0].model_copy(
        update={
            "equation_id": "EQ-decay",
            "lhs": symbol("dx"),
            "rhs": multiply(constant(-1), symbol("demand"), symbol("x")),
        }
    )
    return model.model_copy(
        update={
            "objective": None,
            "constraints": [],
            "decision_variables": [],
            "equations": [equation],
            "state_variables": [variable("x").model_copy(update={"role": VariableRole.STATE})],
        }
    )


def decay_spec(value=1.0):
    return ScenarioSpec(
        scenario_id="decay",
        version="1",
        parameter_values={"demand": value},
        dynamic=DynamicReplaySpec(
            equation_by_state={"x": "EQ-decay"}, initial_state={"x": 1.0}, stop=1.0
        ),
        metrics=[
            MetricSpec(
                metric_id="terminal",
                key="final_value",
                version="1",
                series_key="x",
                lower_threshold=0.0,
                upper_threshold=1.0,
                threshold_provenance="Exact synthetic decay domain.",
            )
        ],
    )


@pytest.fixture
def real_replayer(tmp_path):
    # Environment gate: never replace a failed run with a mock execution.
    inspected = subprocess.run(
        ["docker", "image", "inspect", "mathmodel-ai-solver:phase4"],
        capture_output=True,
        timeout=15,
    )
    if inspected.returncode:
        pytest.skip("real replay requires the existing solver Docker image")
    store = LocalFileStore(tmp_path / "store")
    sandbox = SandboxExecutor(
        store=store,
        root=tmp_path / "sandbox",
        image="mathmodel-ai-solver:phase4",
        limits=SandboxLimits(
            cpu_cores=1,
            memory_mb=512,
            timeout_seconds=30,
            pids_limit=64,
            max_output_bytes=1000000,
            max_artifacts=4,
            max_artifact_bytes=16 * 1024 * 1024,
        ),
    )
    replayer = ScenarioReplayer(
        store=store,
        root=tmp_path / "sandbox",
        image="mathmodel-ai-solver:phase4",
        solver_router=SolverRouter([SciPySolver(sandbox=sandbox, store=store)]),
    )
    return replayer, store


def test_input_contracts_and_seed():
    model, spec = dynamic_model(), decay_spec()
    noisy = spec.model_copy(update={"seed": 42, "noise_fraction": 0.1})
    assert ScenarioReplayer.parameters(model, noisy) == ScenarioReplayer.parameters(model, noisy)
    assert ScenarioReplayer.parameters(model, noisy) != ScenarioReplayer.parameters(model, spec)
    with pytest.raises(ValueError, match="unknown parameters"):
        ScenarioReplayer.parameters(
            model, spec.model_copy(update={"parameter_values": {"evil": 1}})
        )
    with pytest.raises(ValueError, match="state binding"):
        ScenarioReplayer.dynamic_code(
            model,
            spec.model_copy(
                update={
                    "dynamic": spec.dynamic.model_copy(
                        update={"equation_by_state": {"evil": "EQ-decay"}}
                    )
                }
            ),
            {},
        )


@pytest.mark.solver
def test_real_dynamic_replay_fresh_execution_and_tamper(real_replayer):
    replayer, store = real_replayer
    model, spec = dynamic_model(), decay_spec()
    original = mathematical_model_digest(model)
    first = replayer.execute(model, spec)
    assert first.status is IndependentStatus.PASS, first.execution.stderr
    assert first.metrics[0].verified == pytest.approx(math.exp(-1), rel=1e-6)
    assert audit_replay(model, spec, first, store) == []
    stress = decay_spec(2.0)
    second = replayer.execute(model, stress)
    assert second.status is IndependentStatus.PASS
    assert second.metrics[0].verified == pytest.approx(math.exp(-2), rel=1e-6)
    assert first.execution.run_id != second.execution.run_id
    assert first.execution.code_hash != second.execution.code_hash
    assert mathematical_model_digest(model) == original
    assert "REPLAY_INPUT_MISMATCH" in audit_replay(model, stress, first, store)
    assert "REPLAY_EXECUTED_CODE_MISMATCH" in audit_replay(model, stress, first, store)
    tampered = first.model_copy(deep=True)
    tampered.execution.network_disabled = False
    assert "REPLAY_NOT_REAL_ISOLATED_SUCCESS" in audit_replay(model, spec, tampered, store)
    tampered = first.model_copy(deep=True)
    tampered.execution.model_digest = "f" * 64
    assert "REPLAY_MODEL_BINDING_MISMATCH" in audit_replay(model, spec, tampered, store)
    tampered = first.model_copy(deep=True)
    tampered.execution.limits.timeout_seconds = 1
    assert "REPLAY_LIMIT_MISMATCH" in audit_replay(model, spec, tampered, store)
    tampered = first.model_copy(update={"output_artifact_id": None})
    assert "REPLAY_OUTPUT_MISSING" in audit_replay(model, spec, tampered, store)
    output = next(a for a in first.artifacts if a.artifact_id == first.output_artifact_id)
    store.resolve(output.storage_key).write_bytes(b'{"series":{"x":[999]}}')
    assert "REPLAY_ARTIFACT_TAMPER" in audit_replay(model, spec, first, store)


@pytest.mark.solver
def test_dynamic_replay_ignores_frozen_bounds_and_unused_diagnostics(real_replayer):
    replayer, store = real_replayer
    model, spec = dynamic_model(), decay_spec()
    parameter_boundary = model.equations[0].model_copy(
        update={
            "equation_id": "EQ-demand-lower-bound",
            "lhs": symbol("demand"),
            "rhs": constant(0),
            "normalized_expression": "demand >= 0",
        }
    )
    derivative_diagnostic = model.equations[0].model_copy(
        update={
            "equation_id": "EQ-unused-derivative-diagnostic",
            "lhs": symbol("unused_diagnostic"),
            "rhs": symbol("dx"),
            "normalized_expression": "unused_diagnostic = dx",
        }
    )
    bounded = model.model_copy(
        update={
            "equations": [
                *model.equations,
                parameter_boundary,
                derivative_diagnostic,
            ]
        }
    )

    replay = replayer.execute(bounded, spec)

    assert replay.status is IndependentStatus.PASS, replay.execution.stderr
    assert audit_replay(bounded, spec, replay, store) == []


@pytest.mark.solver
def test_real_optimizer_replay(real_replayer):
    replayer, store = real_replayer
    model = phase5_model()
    spec = ScenarioSpec(
        scenario_id="demand_stress",
        version="1",
        parameter_values={"demand": 11.0},
        metrics=[
            MetricSpec(metric_id="objective", key="objective", version="1"),
            MetricSpec(metric_id="feasible", key="feasible", version="1"),
        ],
    )
    replay = replayer.execute(model, spec)
    assert replay.status is IndependentStatus.PASS, replay.execution.stderr
    assert replay.metrics[0].verified == pytest.approx(33.0)
    assert audit_replay(model, spec, replay, store) == []
    no_program = replay.model_copy(update={"program": None})
    assert "REPLAY_PROGRAM_MISSING" in audit_replay(model, spec, no_program, store)


@pytest.mark.solver
def test_real_seeded_replays_reproduce_and_failed_execution_is_invalid(real_replayer):
    replayer, store = real_replayer
    model = dynamic_model()
    spec = decay_spec().model_copy(update={"seed": 42, "noise_fraction": 0.1})
    first = replayer.execute(model, spec)
    second = replayer.execute(model, spec)
    assert first.status is IndependentStatus.PASS and second.status is IndependentStatus.PASS
    assert first.execution.run_id != second.execution.run_id
    assert first.output_digest == second.output_digest
    assert first.input_parameters == second.input_parameters
    assert audit_replay(model, spec, second, store) == []
    # Division by zero in a derivative must produce a real failed execution, not invented output.
    from mathmodel_ai.schemas.mathematical import MathExpression

    bad_equation = model.equations[0].model_copy(
        update={"rhs": MathExpression(kind="DIVIDE", operands=[constant(1), constant(0)])}
    )
    broken = model.model_copy(update={"equations": [bad_equation]})
    failed = replayer.execute(broken, decay_spec())
    assert failed.execution.exit_code != 0
    assert failed.status is IndependentStatus.INVALID
    assert failed.output_artifact_id is None
    assert "REPLAY_NOT_REAL_ISOLATED_SUCCESS" in audit_replay(broken, decay_spec(), failed, store)
