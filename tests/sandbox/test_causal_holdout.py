import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from mathmodel_ai.benchmark.causal_inputs import split_causal_csv
from mathmodel_ai.benchmark.empirical_scenarios import (
    audit_isolated_empirical_scenario,
    empirical_scenario_coefficients,
)
from mathmodel_ai.files.storage import LocalFileStore
from mathmodel_ai.mathematical.digests import mathematical_model_digest
from mathmodel_ai.sandbox.causal_holdout import (
    run_isolated_causal_holdout,
    verify_recorded_causal_holdout,
)
from mathmodel_ai.sandbox.empirical_predictor import empirical_predictor_source
from mathmodel_ai.schemas.execution import ExecutionOrigin, ExecutionStatus, SandboxLimits
from mathmodel_ai.schemas.files import ArtifactKind
from mathmodel_ai.schemas.mathematical import (
    EmpiricalBinaryRiskDefinition,
    ExpressionKind,
    MathExpression,
    VariableDomain,
)
from mathmodel_ai.schemas.results import ResultRecord
from mathmodel_ai.schemas.solver import SolverName, SolverStatus
from mathmodel_ai.verification.causal_binary import CausalBinarySpec
from mathmodel_ai.verification.causal_holdout import assess_binary_calibration
from mathmodel_ai.verification.empirical_binary import (
    audit_empirical_binary_claim,
    evaluate_empirical_binary_risk,
)
from tests.mathematical.helpers import lp_model, variable

IMAGE = "mathmodel-ai-sandbox:phase3"


def _image_available() -> bool:
    try:
        return (
            subprocess.run(
                ["docker", "image", "inspect", IMAGE],
                capture_output=True,
                check=False,
                timeout=10,
            ).returncode
            == 0
        )
    except (OSError, subprocess.SubprocessError):
        return False


pytestmark = [pytest.mark.sandbox, pytest.mark.skipif(not _image_available(), reason="no image")]


def _source() -> bytes:
    return (
        b"match,server,winner,after_point\n"
        b"A,1,1,1\nA,2,2,2\n"
        b"B,1,2,1\nB,2,1,2\n"
        b"C,1,1,1\nC,2,1,2\n"
        b"D,1,2,1\nD,2,2,2\n"
    )


def _run(
    tmp_path: Path,
    code: str,
    *,
    execution_origin: ExecutionOrigin = ExecutionOrigin.USER_CODE,
    model_digest: str | None = None,
    formal_result_id: UUID | None = None,
    project_id: UUID | None = None,
    problem_id: UUID | None = None,
):
    source = _source()
    spec = CausalBinarySpec(
        source_sha256=hashlib.sha256(source).hexdigest(),
        group_column="match",
        condition_column="server",
        outcome_column="winner",
        positive_value="1",
        negative_value="2",
    )
    store = LocalFileStore(tmp_path / "store")
    execution = run_isolated_causal_holdout(
        source,
        spec,
        code,
        fraction=0.25,
        salt="test-v1",
        store=store,
        root=tmp_path / "runs",
        image=IMAGE,
        limits=SandboxLimits(
            cpu_cores=0.5,
            memory_mb=128,
            timeout_seconds=30,
            pids_limit=32,
            max_output_bytes=8192,
            max_artifacts=2,
            max_artifact_bytes=8192,
        ),
        project_id=project_id or uuid4(),
        problem_id=problem_id or uuid4(),
        execution_origin=execution_origin,
        model_digest=model_digest,
        formal_result_id=formal_result_id,
    )
    return execution, store


def test_real_causal_container_receives_only_training_labels_and_one_test_feature(
    tmp_path: Path,
) -> None:
    code = """
from pathlib import Path

class Predictor:
    def __init__(self):
        assert not (Path('/workspace') / 'inputs').exists()
        self.count = 0
        self.positive = 0

    def fit(self, feature, outcome):
        assert feature['group'] in {'A', 'B', 'C', 'D'}
        self.count += 1
        self.positive += outcome

    def predict(self, feature):
        assert feature['group'] in {'A', 'B', 'C', 'D'}
        return (self.positive + 1) / (self.count + 2)
"""
    execution, store = _run(tmp_path, code)
    assert execution.execution.status is ExecutionStatus.SUCCEEDED, execution.execution.error
    assert execution.execution.image_id is not None
    assert execution.execution.network_disabled
    assert execution.execution.read_only_root
    assert execution.result is not None
    assert len(execution.result.predictions) == 2
    assert len(execution.result.observations) == 2
    assert len(execution.artifacts) == 3
    assert execution.artifacts[2].kind is ArtifactKind.VERIFICATION_TRACE
    assert verify_recorded_causal_holdout(_source(), execution, store) == execution.result
    trace = json.loads(store.read_bytes(execution.artifacts[2].storage_key))
    assert trace["spec"]["source_sha256"] == execution.execution.environment["source_sha256"]
    assert trace["predictions"] == list(execution.result.predictions)
    altered = replace(
        execution,
        execution=execution.execution.model_copy(update={"code_hash": "0" * 64}),
    )
    with pytest.raises(ValueError, match="CAUSAL_EXECUTION_BUNDLE_MISMATCH"):
        verify_recorded_causal_holdout(_source(), altered, store)
    replacement = b"print('unreviewed driver')\n"
    replacement_id = uuid4()
    stored = store.store_artifact(
        replacement,
        project_id=execution.execution.project_id,
        artifact_id=replacement_id,
        filename="causal_driver.py",
    )
    driver = execution.artifacts[1].model_copy(
        update={
            "artifact_id": replacement_id,
            "size_bytes": stored.size_bytes,
            "sha256": stored.sha256,
            "storage_key": stored.storage_key,
        }
    )
    code_bytes = store.read_bytes(execution.artifacts[0].storage_key)
    tampered_bundle = hashlib.sha256(
        hashlib.sha256(code_bytes).digest() + hashlib.sha256(replacement).digest()
    ).hexdigest()
    tampered = replace(
        execution,
        execution=execution.execution.model_copy(update={"executed_bundle_hash": tampered_bundle}),
        artifacts=(execution.artifacts[0], driver, execution.artifacts[2]),
    )
    with pytest.raises(ValueError, match="CAUSAL_EXECUTION_DRIVER_MISMATCH"):
        verify_recorded_causal_holdout(_source(), tampered, store)


def test_invalid_predictor_is_recorded_as_failure(tmp_path: Path) -> None:
    code = """
class Predictor:
    def fit(self, feature, outcome):
        pass
    def predict(self, feature):
        return 2.0
"""
    execution, _ = _run(tmp_path, code)
    assert execution.result is None
    assert execution.execution.status is ExecutionStatus.FAILED
    assert execution.execution.error == "HELDOUT_PREDICTION_NOT_A_PROBABILITY"
    assert len(execution.artifacts) == 2


def test_empirical_fixed_coefficient_scenario_isolated_and_independently_replayed(
    tmp_path: Path,
) -> None:
    source = _source()
    official_spec = CausalBinarySpec(
        source_sha256=hashlib.sha256(source).hexdigest(),
        group_column="match",
        condition_column="server",
        outcome_column="winner",
        positive_value="1",
        negative_value="2",
    )
    split = split_causal_csv(source, official_spec, fraction=0.25, salt="test-v1")
    risk = EmpiricalBinaryRiskDefinition(
        training_dataset_id=uuid4(),
        training_sha256=split.training_sha256,
        group_column="match",
        condition_column="server",
        outcome_column="winner",
        positive_value="1",
        negative_value="2",
        history_window=official_spec.history_window,
        reference_condition="1",
        source_refs=["EVID-test"],
        logit=MathExpression(
            kind=ExpressionKind.ADD,
            operands=[
                MathExpression(
                    kind=ExpressionKind.MULTIPLY,
                    operands=[
                        MathExpression.symbol_ref("beta"),
                        MathExpression.symbol_ref("row_condition_delta"),
                    ],
                ),
                MathExpression.symbol_ref("intercept"),
            ],
        ),
    )
    model = lp_model().model_copy(
        update={
            "objective": None,
            "empirical_binary_risk": risk,
            "decision_variables": [
                variable("beta", domain=VariableDomain.CONTINUOUS, lower=None),
                variable("intercept", domain=VariableDomain.CONTINUOUS, lower=None),
            ],
        }
    )
    solver_run_id = uuid4()
    formal_execution_id = uuid4()
    formal_result = ResultRecord(
        project_id=model.project_id,
        problem_id=model.problem_id,
        model_id=model.model_id,
        model_version=model.version,
        model_digest=mathematical_model_digest(model),
        solver_run_id=solver_run_id,
        execution_record_id=formal_execution_id,
        solver=SolverName.SCIPY,
        objective=0.4,
        key_outputs={"beta": 1.0, "intercept": -0.5},
        status=SolverStatus.OPTIMAL,
        evidence_refs=[
            f"model:{model.model_id}:v{model.version}",
            f"solver_run:{solver_run_id}",
            f"execution:{formal_execution_id}",
        ],
    )
    fractions = {"beta": 1.0}
    coefficients = empirical_scenario_coefficients(model, formal_result, fractions)
    assert coefficients == {"beta": 2.0, "intercept": -0.5}
    code = empirical_predictor_source(risk, coefficients)
    execution, store = _run(
        tmp_path,
        code,
        execution_origin=ExecutionOrigin.DETERMINISTIC_SOLVER_ADAPTER,
        model_digest=mathematical_model_digest(model),
        formal_result_id=formal_result.result_id,
        project_id=model.project_id,
        problem_id=model.problem_id,
    )
    assert execution.execution.status is ExecutionStatus.SUCCEEDED, execution.execution.error
    assert not execution.execution.is_mock
    assert execution.execution.execution_origin is ExecutionOrigin.DETERMINISTIC_SOLVER_ADAPTER
    assert execution.execution.network_disabled
    assert execution.execution.read_only_root
    assert execution.result is not None
    assert verify_recorded_causal_holdout(source, execution, store) == execution.result
    training = evaluate_empirical_binary_risk(split.training_csv, risk, coefficients)
    audited = audit_empirical_binary_claim(
        risk=risk,
        training_csv=split.training_csv,
        official_csv=source,
        official_spec=official_spec,
        heldout_groups=split.heldout_groups,
        coefficients=coefficients,
        reported_training_log_loss=training.mean_log_loss,
        reported_heldout_predictions=execution.result.predictions,
    )
    assert audited == training
    assert execution.artifacts[0].sha256 == hashlib.sha256(code.encode()).hexdigest()
    scenario = audit_isolated_empirical_scenario(
        model=model,
        formal_result=formal_result,
        official_csv=source,
        official_spec=official_spec,
        training_csv=split.training_csv,
        fraction=0.25,
        salt="test-v1",
        fractions=fractions,
        run=execution,
        store=store,
    )
    assert scenario.training == training
    assert scenario.holdout == execution.result
    assert scenario.execution_id == execution.execution.run_id
    with pytest.raises(ValueError, match="EMPIRICAL_SCENARIO_FORMAL_COEFFICIENT_MISSING"):
        empirical_scenario_coefficients(
            model,
            formal_result.model_copy(update={"key_outputs": {"beta": 1.0}}),
            fractions,
        )
    with pytest.raises(ValueError, match="EMPIRICAL_SCENARIO_FORMAL_RESULT_NOT_BOUND"):
        empirical_scenario_coefficients(
            model,
            formal_result.model_copy(update={"model_digest": "0" * 64}),
            fractions,
        )
    with pytest.raises(ValueError, match="EMPIRICAL_SCENARIO_PERTURBATION_INVALID"):
        empirical_scenario_coefficients(model, formal_result, {"unknown": 0.1})
    with pytest.raises(ValueError, match="EMPIRICAL_SCENARIO_PERTURBATION_INEFFECTIVE"):
        empirical_scenario_coefficients(model, formal_result, {"beta": 0.0})
    with pytest.raises(ValueError, match="EMPIRICAL_SCENARIO_PERTURBATION_INEFFECTIVE"):
        empirical_scenario_coefficients(
            model,
            formal_result.model_copy(update={"key_outputs": {"beta": 0.0, "intercept": -0.5}}),
            {"beta": 0.1},
        )
    with pytest.raises(ValueError, match="EMPIRICAL_SCENARIO_FORMAL_RESULT_NOT_BOUND"):
        audit_isolated_empirical_scenario(
            model=model,
            formal_result=formal_result.model_copy(update={"result_id": uuid4()}),
            official_csv=source,
            official_spec=official_spec,
            training_csv=split.training_csv,
            fraction=0.25,
            salt="test-v1",
            fractions=fractions,
            run=execution,
            store=store,
        )
    with pytest.raises(ValueError, match="EMPIRICAL_SCENARIO_PREDICTOR_NOT_BOUND"):
        audit_isolated_empirical_scenario(
            model=model,
            formal_result=formal_result,
            official_csv=source,
            official_spec=official_spec,
            training_csv=split.training_csv,
            fraction=0.25,
            salt="test-v1",
            fractions={"beta": 0.5},
            run=execution,
            store=store,
        )
    with pytest.raises(ValueError, match="EMPIRICAL_SCENARIO_PREDICTOR_NOT_BOUND"):
        audit_isolated_empirical_scenario(
            model=model,
            formal_result=formal_result.model_copy(
                update={"key_outputs": {"beta": 0.75, "intercept": -0.5}}
            ),
            official_csv=source,
            official_spec=official_spec,
            training_csv=split.training_csv,
            fraction=0.25,
            salt="test-v1",
            fractions=fractions,
            run=execution,
            store=store,
        )


def test_empirical_predictor_rejects_unbound_or_nonfinite_coefficients() -> None:
    risk = EmpiricalBinaryRiskDefinition(
        training_dataset_id=uuid4(),
        training_sha256="a" * 64,
        group_column="match",
        condition_column="server",
        outcome_column="winner",
        positive_value="1",
        negative_value="2",
        history_window=8,
        reference_condition="1",
        source_refs=["EVID-test"],
        logit=MathExpression.symbol_ref("row_condition_delta"),
    )
    with pytest.raises(ValueError, match="EMPIRICAL_SCENARIO_COEFFICIENTS_INVALID"):
        empirical_predictor_source(risk, {"beta": float("nan")})
    with pytest.raises(ValueError, match="EMPIRICAL_SCENARIO_COEFFICIENTS_INVALID"):
        empirical_predictor_source(risk, {"row_condition_delta": 1.0})
    with pytest.raises(ValueError, match="EMPIRICAL_SCENARIO_COEFFICIENTS_INVALID"):
        empirical_predictor_source(
            risk.model_copy(update={"logit": MathExpression.symbol_ref("unknown")}),
            {},
        )


def test_official_c_csv_stream_isolates_entire_matches(tmp_path: Path) -> None:
    """Local diagnostic only; this reviewed baseline is not the auto-model benchmark."""
    source_path = (
        Path(__file__).resolve().parents[2]
        / "var/benchmarks/cache/BENCH-MCM2024-C/Wimbledon_featured_matches.csv"
    )
    if not source_path.is_file():
        pytest.skip("official runtime cache is unavailable")
    source = source_path.read_bytes()
    spec = CausalBinarySpec(
        source_sha256="b1788d0ea169b65629b0e9fb0f91d007507b306e404507bbf90bd5f700a3c229",
        group_column="match_id",
        condition_column="server",
        outcome_column="point_victor",
        positive_value="1",
        negative_value="2",
    )
    code = """
from pathlib import Path

class Predictor:
    def __init__(self):
        assert not (Path('/workspace') / 'inputs').exists()
    def fit(self, feature, outcome):
        pass
    def predict(self, feature):
        return (1 + feature['prior_condition_positive']) / (2 + feature['prior_condition_count'])
"""
    store = LocalFileStore(tmp_path / "store")
    run = run_isolated_causal_holdout(
        source,
        spec,
        code,
        fraction=0.2,
        salt="causal-protocol-v1",
        store=store,
        root=tmp_path / "runs",
        image=IMAGE,
        limits=SandboxLimits(
            cpu_cores=1,
            memory_mb=256,
            timeout_seconds=300,
            pids_limit=32,
            max_output_bytes=8192,
            max_artifacts=2,
            max_artifact_bytes=1024 * 1024,
        ),
        project_id=uuid4(),
        problem_id=uuid4(),
    )
    assert run.execution.status is ExecutionStatus.SUCCEEDED, run.execution.error
    assert run.result is not None
    assert len(run.result.heldout_groups) == 6
    assert len(run.result.training_groups) == 25
    assert len(run.result.predictions) > 100
    assert run.result.brier == pytest.approx(run.result.baseline_brier)
    calibration = assess_binary_calibration(run.result)
    assert calibration.count == len(run.result.predictions)
    assert sum(item.count for item in calibration.bins) == calibration.count
    assert 0 <= calibration.ece <= 1
    assert verify_recorded_causal_holdout(source, run, store) == run.result
