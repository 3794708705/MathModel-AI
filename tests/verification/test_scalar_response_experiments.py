from __future__ import annotations

import subprocess
from pathlib import Path
from uuid import uuid4

import pytest

from mathmodel_ai.files.storage import LocalFileStore
from mathmodel_ai.mathematical.algorithms import AlgorithmSelector
from mathmodel_ai.mathematical.digests import mathematical_model_digest
from mathmodel_ai.sandbox.executor import SandboxExecutor
from mathmodel_ai.schemas.execution import SandboxLimits
from mathmodel_ai.schemas.mathematical import (
    EquationDefinition,
    ParameterDefinition,
    ParameterSourceType,
    VariableRole,
)
from mathmodel_ai.schemas.model_selection import ModelFamily
from mathmodel_ai.schemas.solver import SolverName
from mathmodel_ai.schemas.verification import (
    RedTeamReport,
    RobustnessConfig,
    SensitivityConfig,
    ValidationCheckCategory,
)
from mathmodel_ai.solvers.router import SolverRouter
from mathmodel_ai.solvers.scalar_response import ScalarResponseSolver
from mathmodel_ai.verification.experiment_integrity import ExperimentIntegrityVerifier
from mathmodel_ai.verification.experiments import ExperimentEngine
from mathmodel_ai.verification.quality_gates import (
    robustness_quality_gate,
    sensitivity_quality_gate,
    verified_result_quality_gate,
)
from mathmodel_ai.verification.robustness import RobustnessAnalyzer
from mathmodel_ai.verification.scalar_response import evaluate_scalar_responses
from mathmodel_ai.verification.sensitivity import SensitivityAnalyzer
from mathmodel_ai.verification.validation import IndependentValidator
from tests.mathematical.helpers import lp_model, multiply, symbol
from tests.verification.helpers import valid_report


def _model():
    model = lp_model()
    parameter = ParameterDefinition(
        parameter_id="PAR-rate",
        symbol="rate",
        description="sourced response rate",
        value=3.0,
        source_type=ParameterSourceType.PROBLEM_FACT,
        source_ref="EVID-fact-1",
        confidence=1,
    )
    derived = model.decision_variables[1].model_copy(update={"role": VariableRole.DERIVED})
    equation = EquationDefinition(
        equation_id="EQ-response",
        latex="y=rate x",
        normalized_expression="y = rate * x",
        lhs=symbol("y"),
        rhs=multiply(symbol("rate"), symbol("x")),
        meaning="scalar response",
        source_refs=["EVID-fact-1"],
        derivation="Direct fixed-point response",
    )
    return model.model_copy(
        update={
            "model_family": ModelFamily.DYNAMIC_SYSTEM,
            "objective": None,
            "decision_variables": [model.decision_variables[0]],
            "derived_variables": [derived],
            "parameters": [parameter],
            "constraints": [],
            "equations": [equation],
        }
    )


def test_independent_scalar_response_fails_closed_on_missing_definition() -> None:
    model = _model()
    assert evaluate_scalar_responses(model, {"x": 2.0}) == {"y": 6.0}
    with pytest.raises(ValueError, match="exactly one defining equation"):
        evaluate_scalar_responses(model.model_copy(update={"equations": []}), {"x": 2.0})
    with pytest.raises(ValueError, match="fixed point"):
        evaluate_scalar_responses(model, {"x": 2.0, "other": 0.0})


@pytest.mark.solver
def test_nonobjective_response_experiments_are_real_and_recomputed(tmp_path: Path) -> None:
    image = "mathmodel-ai-solver:phase4"
    if subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode:
        pytest.skip("versioned solver image is unavailable")
    store = LocalFileStore(tmp_path / "store")
    sandbox = SandboxExecutor(
        store=store,
        root=tmp_path / "runs",
        image=image,
        limits=SandboxLimits(
            cpu_cores=1,
            memory_mb=768,
            timeout_seconds=20,
            pids_limit=64,
            max_output_bytes=32_768,
            max_artifacts=10,
            max_artifact_bytes=1_048_576,
        ),
    )
    solver = ScalarResponseSolver(sandbox=sandbox, store=store)
    validator = IndependentValidator()
    engine = ExperimentEngine(
        algorithm_selector=AlgorithmSelector(),
        solver_router=SolverRouter([]),
        validator=validator,
        response_solver=solver,
    )
    model = _model()
    _, base_result, base_validation, _ = valid_report()
    result = base_result.model_copy(
        update={
            "model_id": model.model_id,
            "model_digest": mathematical_model_digest(model),
            "objective": None,
            "key_outputs": {"x": 2.0, "y": 6.0},
        }
    )
    sensitivity, outcomes = SensitivityAnalyzer(engine).analyze(
        model=model,
        result=result,
        validation=base_validation,
        config=SensitivityConfig(perturbation_fractions=[0.1]),
    )
    assert sensitivity.status.value == "PASS"
    assert sensitivity.baseline_objective is None
    assert sensitivity.response_ranges["y"] == pytest.approx((5.4, 6.6))
    assert all(item.record.solver is SolverName.SCALAR_RESPONSE for item in outcomes)
    assert sensitivity_quality_gate(sensitivity).status.value == "PASS"
    execution = outcomes[0].execution
    assert execution is not None
    forged = outcomes[0].record.model_copy(
        update={"fixed_decision_values": {"x": 0.5}}
    )
    audit = ExperimentIntegrityVerifier(validator).audit(
        base_model=model, record=forged, execution=execution
    )
    assert not audit.valid
    assert "RESPONSE_FIXED_POINT_MISMATCH" in audit.errors
    robustness, _ = RobustnessAnalyzer(engine).analyze(
        model=model,
        result=result,
        validation=base_validation,
        sensitivity=sensitivity,
        config=RobustnessConfig(scenario_fractions=[-0.1, 0.1]),
    )
    assert robustness.status.value == "PASS"
    assert robustness.response_ranges["y"] == pytest.approx((5.4, 6.6))
    assert robustness_quality_gate(robustness).status.value == "PASS"
    tampered = sensitivity.model_copy(update={"response_ranges": {"y": (0.0, 6.6)}})
    assert sensitivity_quality_gate(tampered).status.value != "PASS"
    validation = base_validation.model_copy(
        update={
            "model_id": model.model_id,
            "model_digest": result.model_digest,
            "result_id": result.result_id,
            "metric_recalculations": [
                item.model_copy(update={"category": ValidationCheckCategory.OUTPUT})
                for item in base_validation.metric_recalculations
            ],
        }
    )
    red_team = RedTeamReport(
        project_id=model.project_id,
        problem_id=model.problem_id,
        model_id=model.model_id,
        model_version=model.version,
        model_digest=result.model_digest,
        result_id=result.result_id,
        validation_id=validation.validation_id,
        sensitivity_id=sensitivity.sensitivity_id,
        robustness_id=robustness.robustness_id,
        critical_count=0,
        major_count=0,
        minor_count=0,
        summary="Independent test review.",
        reviewer_agent_run_id=uuid4(),
        status="PASS",
    )
    verified = verified_result_quality_gate(
        validation,
        sensitivity,
        robustness,
        red_team,
        validation_integrity_errors=[],
        sensitivity_integrity_errors=[],
        robustness_integrity_errors=[],
    )
    assert verified.status.value == "PASS", verified.errors
