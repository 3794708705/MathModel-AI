import json
import math
import subprocess
from dataclasses import replace
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from mathmodel_ai.benchmark.causal_evaluation import CausalBenchmarkEvaluator
from mathmodel_ai.core.errors import QualityGateError
from mathmodel_ai.data.repository import DataRepository
from mathmodel_ai.db.base import Base
from mathmodel_ai.db.models import Problem, Project
from mathmodel_ai.files.storage import LocalFileStore
from mathmodel_ai.mathematical.digests import mathematical_model_digest
from mathmodel_ai.mathematical.repository import MathematicalRepository
from mathmodel_ai.paper.hashing import sha256_json
from mathmodel_ai.schemas.benchmark import CausalScienceCheck
from mathmodel_ai.schemas.data import DatasetRecord
from mathmodel_ai.schemas.execution import (
    ExecutionArtifact,
    ExecutionOrigin,
    ExecutionRecord,
    ExecutionStatus,
    SandboxLimits,
)
from mathmodel_ai.schemas.files import ArtifactKind, ArtifactRecord, FileKind, RegisteredFile
from mathmodel_ai.schemas.mathematical import (
    EmpiricalBinaryRiskDefinition,
    ExpressionKind,
    MathExpression,
)
from mathmodel_ai.schemas.problem_state import ProblemState
from mathmodel_ai.schemas.program import (
    GeneratedProgram,
    GeneratedProgramStatus,
    GeneratedSourceFile,
    generated_program_hash,
)
from mathmodel_ai.schemas.results import ResultRecord
from mathmodel_ai.schemas.solver import SolverName, SolverStatus
from mathmodel_ai.verification.causal_binary import CausalBinarySpec, iter_causal_binary_points
from mathmodel_ai.verification.causal_holdout import SWING_PROTOCOL, CausalHoldoutResult
from mathmodel_ai.verification.empirical_binary import (
    evaluate_empirical_binary_risk,
    predict_empirical_binary,
)
from tests.benchmark.test_causal_inputs import _fixture
from tests.mathematical.helpers import lp_model

IMAGE = "mathmodel-ai-solver:phase4"


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


def _setup(tmp_path: Path, *, include_predictor: bool):
    registry, cache = _fixture(tmp_path)
    bundle = registry.load_blind_solve_bundle("BENCH-CAUSAL-TEST", cache)
    assert bundle.causal_split is not None
    model = lp_model()
    files = [GeneratedSourceFile(path="solve.py", content="print('solve')").with_digest()]
    if include_predictor:
        files.append(
            GeneratedSourceFile(
                path="causal_predictor.py",
                content=(
                    "class Predictor:\n"
                    "    def __init__(self): self.n=0; self.s=0\n"
                    "    def fit(self, feature, outcome): self.n+=1; self.s+=outcome\n"
                    "    def predict(self, feature): return (self.s+1)/(self.n+2)\n"
                ),
            ).with_digest()
        )
    program = GeneratedProgram(
        project_id=model.project_id,
        problem_id=model.problem_id,
        model_id=model.model_id,
        model_version=model.version,
        model_digest=mathematical_model_digest(model),
        entrypoint="solve.py",
        files=files,
        solver_target="SCIPY",
        explanation="fixture",
        code_hash=generated_program_hash(files),
        generated_by="code_agent",
        prompt_version="fixture",
        execution_origin=ExecutionOrigin.GENERATED_PROGRAM,
        status=GeneratedProgramStatus.EXECUTED,
    )
    store = LocalFileStore(tmp_path / "store")
    file_id = uuid4()
    staged = store.stage_stream(BytesIO(bundle.causal_split.training_csv), max_bytes=1024 * 1024)
    stored_training = store.commit_upload(
        staged, project_id=model.project_id, file_id=file_id, extension=".csv"
    )
    registered = RegisteredFile(
        file_id=file_id,
        project_id=model.project_id,
        problem_id=model.problem_id,
        original_name="data.csv",
        safe_name="data.csv",
        extension=".csv",
        kind=FileKind.CSV,
        detected_mime_type="text/csv",
        size_bytes=len(bundle.causal_split.training_csv),
        sha256=bundle.causal_split.training_sha256,
        storage_key=stored_training.storage_key,
    )
    state = ProblemState(
        project_id=model.project_id,
        problem_id=model.problem_id,
        title="fixture",
        raw_problem="fixture",
        registered_files=[registered],
    )
    limits = SandboxLimits(
        cpu_cores=0.5,
        memory_mb=128,
        timeout_seconds=30,
        pids_limit=32,
        max_output_bytes=8192,
        max_artifacts=2,
        max_artifact_bytes=8192,
    )
    formal_execution = ExecutionRecord(
        project_id=model.project_id,
        problem_id=model.problem_id,
        code_hash=files[0].sha256,
        executed_bundle_hash=program.code_hash,
        code_artifact_id=uuid4(),
        execution_origin=ExecutionOrigin.GENERATED_PROGRAM,
        model_digest=program.model_digest,
        generated_program_id=program.program_id,
        image=IMAGE,
        end_time=datetime.now(UTC),
        runtime_seconds=0,
        status=ExecutionStatus.SUCCEEDED,
        exit_code=0,
        limits=limits,
    )
    solver_run_id, result_id = uuid4(), uuid4()
    result = SimpleNamespace(
        result_id=result_id,
        project_id=model.project_id,
        problem_id=model.problem_id,
        model_id=model.model_id,
        model_version=model.version,
        model_digest=program.model_digest,
        solver_run_id=solver_run_id,
        execution_record_id=formal_execution.run_id,
    )
    solver_run = SimpleNamespace(
        solver_run_id=solver_run_id,
        project_id=model.project_id,
        problem_id=model.problem_id,
        model_id=model.model_id,
        model_version=model.version,
        model_digest=program.model_digest,
        execution_origin=ExecutionOrigin.GENERATED_PROGRAM,
        generated_program_id=program.program_id,
        execution_ref=formal_execution.run_id,
        result_ref=result_id,
    )
    mathematical = SimpleNamespace(
        model_stage=SimpleNamespace(model=model),
        solve_stage=SimpleNamespace(
            execution=SimpleNamespace(
                program=program,
                execution=SimpleNamespace(record=formal_execution),
            ),
            solver_run=solver_run,
            result=result,
        ),
    )
    repository = Mock(spec=DataRepository)
    repository.list_files.return_value = [registered]
    mathematics_repository = Mock(spec=MathematicalRepository)
    mathematics_repository.get_result_context.return_value = SimpleNamespace(
        evidence=SimpleNamespace(valid=True),
        model=model,
        result=result,
        solver_run=solver_run,
        execution=formal_execution,
        program=program,
    )

    def capture_execution(record, artifacts):
        repository.list_executions.return_value = [formal_execution, record]
        repository.list_artifacts.return_value = artifacts

    repository.persist_auxiliary_execution.side_effect = capture_execution
    evaluator = CausalBenchmarkEvaluator(
        store=store,
        repository=repository,
        mathematics=mathematics_repository,
        root=tmp_path / "runs",
        image=IMAGE,
        limits=limits,
    )
    return evaluator, bundle, mathematical, state, repository


def test_generated_predictor_source_is_required_before_execution(tmp_path: Path) -> None:
    evaluator, bundle, mathematical, state, repository = _setup(tmp_path, include_predictor=False)
    with pytest.raises(QualityGateError, match="CAUSAL_HOLDOUT_PREDICTOR_SOURCE_MISSING"):
        evaluator.evaluate(bundle=bundle, mathematical=mathematical, state=state)  # type: ignore[arg-type]
    repository.persist_auxiliary_execution.assert_not_called()


def test_randomness_claim_must_come_from_exact_formal_result_artifact(tmp_path: Path) -> None:
    evaluator, _, _, state, repository = _setup(tmp_path, include_predictor=True)
    formal = evaluator._mathematics.get_result_context.return_value
    formal.solver_run.result = SimpleNamespace(
        solver_name=SolverName.SCIPY,
        status=SolverStatus.FEASIBLE,
        objective_value=None,
        variable_values={},
    )
    raw = json.dumps(
        {
            "solver_name": "SCIPY",
            "model_digest": formal.result.model_digest,
            "status": "FEASIBLE",
            "variable_values": {},
            "series": {"match_flow": [0.0, 0.5]},
            "metrics": {
                "conditional_randomness_statistic": 0.125,
                "conditional_randomness_p": 0.2,
                "conditional_randomness_replicates": 499,
                "swing_event_protocol": SWING_PROTOCOL,
            },
            "is_feasible": True,
        }
    ).encode()
    artifact_id = uuid4()
    stored = evaluator._store.store_artifact(
        raw, project_id=state.project_id, artifact_id=artifact_id, filename="result.json"
    )
    artifact = ArtifactRecord(
        artifact_id=artifact_id,
        project_id=state.project_id,
        problem_id=state.problem_id,
        execution_run_id=formal.execution.run_id,
        kind=ArtifactKind.SANDBOX_OUTPUT,
        name="result.json",
        mime_type="application/json",
        size_bytes=stored.size_bytes,
        sha256=stored.sha256,
        storage_key=stored.storage_key,
    )
    repository.list_artifacts.return_value = [artifact]
    formal.execution.artifacts.append(
        ExecutionArtifact(
            artifact_id=artifact.artifact_id,
            name=artifact.name,
            mime_type=artifact.mime_type,
            size_bytes=artifact.size_bytes,
            sha256=artifact.sha256,
            storage_key=artifact.storage_key,
        )
    )
    claim = evaluator._randomness_claim(state.project_id, formal.result.result_id)
    assert claim is not None
    assert claim.observed_statistic == 0.125
    assert claim.two_sided_p == 0.2
    assert claim.result_artifact_sha256 == stored.sha256
    flow_claim = evaluator._match_flow_claim(state.project_id, formal.result.result_id)
    assert flow_claim is not None
    assert flow_claim.values == (0.0, 0.5)
    assert flow_claim.result_artifact_sha256 == stored.sha256
    swing_claim = evaluator._swing_claim(state.project_id, formal.result.result_id)
    assert swing_claim is not None
    assert swing_claim.protocol == SWING_PROTOCOL
    assert swing_claim.result_artifact_sha256 == stored.sha256
    repository.list_artifacts.return_value = [artifact.model_copy(update={"sha256": "0" * 64})]
    with pytest.raises(QualityGateError, match="CAUSAL_SCIENCE_RESULT_ARTIFACT_MISMATCH"):
        evaluator._randomness_claim(state.project_id, formal.result.result_id)


def test_causal_evaluator_rejects_wrong_training_and_model_identity(tmp_path: Path) -> None:
    evaluator, bundle, mathematical, state, repository = _setup(tmp_path, include_predictor=True)
    state.registered_files[0].sha256 = "0" * 64
    with pytest.raises(QualityGateError, match="CAUSAL_HOLDOUT_TRAINING_INPUT_NOT_BOUND"):
        evaluator.evaluate(bundle=bundle, mathematical=mathematical, state=state)  # type: ignore[arg-type]
    state.registered_files[0].sha256 = bundle.causal_split.training_sha256  # type: ignore[union-attr]
    mathematical.solve_stage.execution.program.model_digest = "0" * 64
    with pytest.raises(QualityGateError, match="CAUSAL_HOLDOUT_GENERATED_PROGRAM_NOT_BOUND"):
        evaluator.evaluate(bundle=bundle, mathematical=mathematical, state=state)  # type: ignore[arg-type]
    mathematical.solve_stage.execution.program.model_digest = mathematical_model_digest(
        mathematical.model_stage.model
    )
    revised = mathematical.model_stage.model.model_copy(
        update={"version": mathematical.model_stage.model.version + 1}
    )
    with pytest.raises(QualityGateError, match="CAUSAL_HOLDOUT_FORMAL_RESULT_NOT_BOUND"):
        evaluator.evaluate_solve(
            bundle=bundle,
            model=revised,
            solve=mathematical.solve_stage,
            state=state,
        )  # type: ignore[arg-type]
    repository.persist_auxiliary_execution.assert_not_called()


def test_causal_evaluator_rejects_a_result_from_another_solver_run(tmp_path: Path) -> None:
    evaluator, bundle, mathematical, state, repository = _setup(tmp_path, include_predictor=True)
    mathematical.solve_stage.result.solver_run_id = uuid4()
    with pytest.raises(QualityGateError, match="CAUSAL_HOLDOUT_FORMAL_RESULT_NOT_BOUND"):
        evaluator.evaluate(bundle=bundle, mathematical=mathematical, state=state)  # type: ignore[arg-type]
    repository.persist_auxiliary_execution.assert_not_called()


def test_empirical_audit_binds_formal_coefficients_loss_and_sealed_forecasts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluator, bundle, mathematical, state, repository = _setup(tmp_path, include_predictor=True)
    assert bundle.causal_policy is not None and bundle.causal_split is not None
    policy, split = bundle.causal_policy, bundle.causal_split
    source = next(
        item.content for item in bundle.artifacts if item.resource.resource_id == policy.resource_id
    )
    risk = EmpiricalBinaryRiskDefinition(
        training_dataset_id=uuid4(),
        training_sha256=split.training_sha256,
        group_column=policy.group_column,
        condition_column=policy.condition_column,
        outcome_column=policy.outcome_column,
        positive_value=policy.positive_value,
        negative_value=policy.negative_value,
        history_window=policy.history_window,
        reference_condition="1",
        source_refs=["EVID-fact-1"],
        logit=MathExpression(
            kind=ExpressionKind.MULTIPLY,
            operands=[
                MathExpression.symbol_ref("x"),
                MathExpression.symbol_ref("row_condition_delta"),
            ],
        ),
    )
    model = mathematical.model_stage.model.model_copy(update={"empirical_binary_risk": risk})
    formal = evaluator._mathematics.get_result_context.return_value
    formal.model = model
    dataset = DatasetRecord(
        dataset_id=risk.training_dataset_id,
        project_id=state.project_id,
        problem_id=state.problem_id,
        source_file_id=state.registered_files[0].file_id,
        name="data.csv",
        row_count=split.training_rows,
        column_count=4,
        columns=["match", "server", "winner", "after"],
    )
    repository.list_datasets.return_value = [dataset]
    values = {"x": 1.0, "y": 0.0}
    training_loss = evaluate_empirical_binary_risk(split.training_csv, risk, values).mean_log_loss
    values["training_log_loss"] = training_loss
    formal.solver_run.result = SimpleNamespace(variable_values=values)
    formal.result.key_outputs = values.copy()
    payload = SimpleNamespace(
        metrics={"training_log_loss": training_loss}, variable_values=values.copy()
    )
    monkeypatch.setattr(evaluator, "_formal_payload", lambda *_: (payload, "a" * 64))
    official_spec = CausalBinarySpec(
        source_sha256=policy.source_sha256,
        group_column=policy.group_column,
        condition_column=policy.condition_column,
        outcome_column=policy.outcome_column,
        positive_value=policy.positive_value,
        negative_value=policy.negative_value,
        history_window=policy.history_window,
    )
    predictions = tuple(
        predict_empirical_binary(risk, feature, values)[0]
        for feature, _ in iter_causal_binary_points(source, official_spec)
        if feature.group in split.heldout_groups
    )
    result = CausalHoldoutResult(
        heldout_groups=split.heldout_groups,
        training_groups=split.training_groups,
        predictions=predictions,
        observations=tuple(0.0 for _ in predictions),
        baseline_predictions=tuple(0.5 for _ in predictions),
        brier=0.0,
        baseline_brier=0.0,
    )
    kwargs = dict(
        bundle=bundle,
        project_id=state.project_id,
        formal_result_id=formal.result.result_id,
        model=model,
        training_bytes=split.training_csv,
        source=source,
        official_spec=official_spec,
        result=result,
    )
    audited = evaluator._audit_empirical_risk(**kwargs)
    assert audited is not None and math.isclose(audited.mean_log_loss, training_loss)
    formal.result.key_outputs["training_log_loss"] += 0.1
    with pytest.raises(QualityGateError, match="TRAINING_LOSS_NOT_RESULT_BOUND"):
        evaluator._audit_empirical_risk(**kwargs)
    formal.result.key_outputs["training_log_loss"] = training_loss
    bad = replace(result, predictions=(predictions[0] + 0.1, *predictions[1:]))
    with pytest.raises(QualityGateError, match="EMPIRICAL_HELDOUT_PREDICTION_MISMATCH"):
        evaluator._audit_empirical_risk(**{**kwargs, "result": bad})


@pytest.mark.sandbox
@pytest.mark.skipif(not _image_available(), reason="Docker image unavailable")
def test_empirical_scenario_is_persisted_and_reaudited_from_database(tmp_path: Path) -> None:
    evaluator, bundle, mathematical, state, _ = _setup(tmp_path, include_predictor=True)
    assert bundle.causal_policy is not None and bundle.causal_split is not None
    policy, split = bundle.causal_policy, bundle.causal_split
    risk = EmpiricalBinaryRiskDefinition(
        training_dataset_id=uuid4(),
        training_sha256=split.training_sha256,
        group_column=policy.group_column,
        condition_column=policy.condition_column,
        outcome_column=policy.outcome_column,
        positive_value=policy.positive_value,
        negative_value=policy.negative_value,
        history_window=policy.history_window,
        reference_condition="1",
        source_refs=["EVID-fact-1"],
        logit=MathExpression(
            kind=ExpressionKind.MULTIPLY,
            operands=[
                MathExpression.symbol_ref("x"),
                MathExpression.symbol_ref("row_condition_delta"),
            ],
        ),
    )
    model = mathematical.model_stage.model.model_copy(
        update={"objective": None, "empirical_binary_risk": risk}
    )
    digest = mathematical_model_digest(model)
    program = mathematical.solve_stage.execution.program
    program.model_digest = digest
    formal_execution = mathematical.solve_stage.execution.execution.record
    formal_execution.model_digest = digest
    formal_solver_run = mathematical.solve_stage.solver_run
    formal_solver_run.model_digest = digest
    result = ResultRecord(
        result_id=formal_solver_run.result_ref,
        project_id=model.project_id,
        problem_id=model.problem_id,
        model_id=model.model_id,
        model_version=model.version,
        model_digest=digest,
        solver_run_id=formal_solver_run.solver_run_id,
        execution_record_id=formal_execution.run_id,
        solver=SolverName.SCIPY,
        status=SolverStatus.OPTIMAL,
        key_outputs={"x": 1.0, "y": 9.0},
        evidence_refs=[
            f"model:{model.model_id}:v{model.version}",
            f"solver_run:{formal_solver_run.solver_run_id}",
            f"execution:{formal_execution.run_id}",
        ],
    )
    formal = evaluator._mathematics.get_result_context.return_value
    formal.model = model
    formal.result = result
    formal.program = program
    formal.execution = formal_execution
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = sessionmaker(engine, expire_on_commit=False)
    dataset = DatasetRecord(
        dataset_id=risk.training_dataset_id,
        project_id=state.project_id,
        problem_id=state.problem_id,
        source_file_id=state.registered_files[0].file_id,
        name="data.csv",
        row_count=split.training_rows,
        column_count=4,
        columns=["match", "server", "winner", "after"],
    )
    with factory() as session:
        session.add(Project(id=state.project_id, name="empirical fixture"))
        session.add(
            Problem(
                id=state.problem_id,
                project_id=state.project_id,
                title="fixture",
                raw_problem="fixture",
            )
        )
        session.add(DataRepository._file_model(state.registered_files[0]))
        session.add(DataRepository._dataset_model(dataset))
        session.commit()
    repository = DataRepository(factory)
    code = ArtifactRecord(
        artifact_id=formal_execution.code_artifact_id,
        project_id=state.project_id,
        problem_id=state.problem_id,
        execution_run_id=formal_execution.run_id,
        kind=ArtifactKind.GENERATED_CODE,
        name="solve.py",
        mime_type="text/x-python",
        size_bytes=14,
        sha256=formal_execution.code_hash,
        storage_key="fixture/solve.py",
    )
    repository.persist_auxiliary_execution(formal_execution, [code])
    evaluator._repository = repository
    fractions = {"x": 0.1}
    audited = evaluator.evaluate_empirical_scenario(
        bundle=bundle,
        project_id=state.project_id,
        formal_result_id=result.result_id,
        fractions=fractions,
    )
    assert audited.formal_result_id == result.result_id
    assert audited.execution_id != formal_execution.run_id
    assert audited.training.row_count == split.training_rows
    assert len(audited.holdout.predictions) == split.heldout_rows
    assert len(repository.list_executions(state.project_id)) == 2
    assert (
        evaluator.audit_persisted_empirical_scenario(
            bundle=bundle,
            project_id=state.project_id,
            formal_result_id=result.result_id,
            fractions=fractions,
            execution_id=audited.execution_id,
        )
        == audited
    )
    with pytest.raises(QualityGateError, match="EMPIRICAL_SCENARIO_AUDIT_FAILED"):
        evaluator.audit_persisted_empirical_scenario(
            bundle=bundle,
            project_id=state.project_id,
            formal_result_id=result.result_id,
            fractions={"x": 0.2},
            execution_id=audited.execution_id,
        )
    with pytest.raises(QualityGateError, match="EMPIRICAL_SCENARIO_FORMAL_CONSTRAINT_FAILED"):
        evaluator.evaluate_empirical_scenario(
            bundle=bundle,
            project_id=state.project_id,
            formal_result_id=result.result_id,
            fractions={"x": -0.1},
        )
    assert len(repository.list_executions(state.project_id)) == 3
    formal_solver_run.result_ref = uuid4()
    with pytest.raises(QualityGateError, match="EMPIRICAL_SCENARIO_FORMAL_RESULT_NOT_BOUND"):
        evaluator.audit_persisted_empirical_scenario(
            bundle=bundle,
            project_id=state.project_id,
            formal_result_id=result.result_id,
            fractions=fractions,
            execution_id=audited.execution_id,
        )


@pytest.mark.sandbox
@pytest.mark.skipif(not _image_available(), reason="Docker image unavailable")
def test_generated_predictor_evaluates_in_isolation_and_persists_evidence(
    tmp_path: Path,
) -> None:
    evaluator, bundle, mathematical, state, repository = _setup(tmp_path, include_predictor=True)
    assert bundle.causal_policy is not None
    bundle = replace(
        bundle,
        causal_policy=bundle.causal_policy.model_copy(
            update={
                "problem_sha256": bundle.manifest.resources[0].sha256,
                "history_window": 1,
                "required_scientific_checks": [
                    CausalScienceCheck.HELDOUT_PREDICTION,
                    CausalScienceCheck.MATCH_FLOW,
                    CausalScienceCheck.RANDOMNESS_TEST,
                    CausalScienceCheck.SWING_PREDICTION,
                ],
            }
        ),
    )
    result = evaluator.evaluate_solve(
        bundle=bundle,
        model=mathematical.model_stage.model,
        solve=mathematical.solve_stage,
        state=state,
    )  # type: ignore[arg-type]
    assert result.execution.status is ExecutionStatus.SUCCEEDED
    assert result.execution.execution_origin is ExecutionOrigin.GENERATED_PROGRAM
    assert result.execution.environment["formal_result_id"] == str(
        mathematical.solve_stage.result.result_id
    )
    assert result.execution.environment["science_policy_sha256"] == sha256_json(
        bundle.causal_policy
    )
    assert (
        result.execution.generated_program_id
        == mathematical.solve_stage.execution.program.program_id
    )
    assert result.result is not None
    assert len(result.result.predictions) == bundle.causal_split.heldout_rows  # type: ignore[union-attr]
    repository.persist_auxiliary_execution.assert_called_once()
    replayed = evaluator.audit_persisted(
        bundle=bundle,
        project_id=state.project_id,
        formal_result_id=mathematical.solve_stage.result.result_id,
        holdout_execution_id=result.execution.run_id,
    )
    assert replayed == result.result
    validation_evidence = evaluator.audit_validation_evidence(
        bundle=bundle,
        project_id=state.project_id,
        formal_result_id=mathematical.solve_stage.result.result_id,
        holdout_execution_id=result.execution.run_id,
    )
    assert validation_evidence.result == replayed
    assert validation_evidence.calibration is not None
    assert validation_evidence.calibration.count == len(replayed.predictions)
    assert validation_evidence.randomness is not None
    assert validation_evidence.match_flow is not None
    assert validation_evidence.match_flow_claim is None
    assert len(validation_evidence.match_flow.values) == bundle.causal_split.training_rows
    assert validation_evidence.swing is not None
    assert validation_evidence.swing.eligible_points == 1
    assert validation_evidence.swing_claim is None
    assert validation_evidence.randomness.point_count == bundle.causal_split.training_rows
    assert validation_evidence.randomness.replicates == 499
    assert 0 < validation_evidence.randomness.two_sided_p <= 1
    assert validation_evidence.formal_result_id == mathematical.solve_stage.result.result_id
    assert validation_evidence.source_sha256 == bundle.causal_split.source_sha256  # type: ignore[union-attr]
    artifacts = repository.list_artifacts.return_value
    repository.list_artifacts.return_value = [
        item.model_copy(update={"sha256": "0" * 64}) if item.name == "causal-holdout.json" else item
        for item in artifacts
    ]
    with pytest.raises(ValueError, match="CAUSAL_EXECUTION_ARTIFACT_MISMATCH"):
        evaluator.audit_validation_evidence(
            bundle=bundle,
            project_id=state.project_id,
            formal_result_id=mathematical.solve_stage.result.result_id,
            holdout_execution_id=result.execution.run_id,
        )
    repository.list_artifacts.return_value = artifacts
    assert bundle.causal_policy is not None
    altered_policy = replace(
        bundle, causal_policy=bundle.causal_policy.model_copy(update={"salt": "changed-salt"})
    )
    with pytest.raises(QualityGateError, match="CAUSAL_HOLDOUT_SCIENCE_POLICY_MISMATCH"):
        evaluator.audit_persisted(
            bundle=altered_policy,
            project_id=state.project_id,
            formal_result_id=mathematical.solve_stage.result.result_id,
            holdout_execution_id=result.execution.run_id,
        )
    changed_obligations = replace(
        bundle,
        causal_policy=bundle.causal_policy.model_copy(
            update={"required_scientific_checks": ["heldout_prediction", "match_flow"]}
        ),
    )
    with pytest.raises(QualityGateError, match="CAUSAL_HOLDOUT_SCIENCE_POLICY_MISMATCH"):
        evaluator.audit_validation_evidence(
            bundle=changed_obligations,
            project_id=state.project_id,
            formal_result_id=mathematical.solve_stage.result.result_id,
            holdout_execution_id=result.execution.run_id,
        )
    LocalFileStore(tmp_path / "store").resolve(state.registered_files[0].storage_key).write_bytes(
        b"match,server,winner,after\n"
    )
    with pytest.raises(QualityGateError, match="CAUSAL_HOLDOUT_TRAINING_BYTES_MISMATCH"):
        evaluator.audit_persisted(
            bundle=bundle,
            project_id=state.project_id,
            formal_result_id=mathematical.solve_stage.result.result_id,
            holdout_execution_id=result.execution.run_id,
        )


@pytest.mark.sandbox
@pytest.mark.skipif(not _image_available(), reason="Docker image unavailable")
def test_causal_evaluator_rejects_missing_persisted_trace(tmp_path: Path) -> None:
    evaluator, bundle, mathematical, state, repository = _setup(tmp_path, include_predictor=True)
    repository.list_artifacts.return_value = []

    def omit_artifacts(record, artifacts):
        del artifacts
        repository.list_executions.return_value = [
            mathematical.solve_stage.execution.execution.record,
            record,
        ]

    repository.persist_auxiliary_execution.side_effect = omit_artifacts
    with pytest.raises(QualityGateError, match="CAUSAL_HOLDOUT_PERSISTED_EVIDENCE_MISMATCH"):
        evaluator.evaluate(bundle=bundle, mathematical=mathematical, state=state)  # type: ignore[arg-type]


@pytest.mark.sandbox
@pytest.mark.skipif(not _image_available(), reason="Docker image unavailable")
def test_generated_holdout_is_reaudited_from_real_database(tmp_path: Path) -> None:
    evaluator, bundle, mathematical, state, _ = _setup(tmp_path, include_predictor=True)
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = sessionmaker(engine, expire_on_commit=False)
    with factory() as session:
        session.add(Project(id=state.project_id, name="causal fixture"))
        session.add(
            Problem(
                id=state.problem_id,
                project_id=state.project_id,
                title="causal fixture",
                raw_problem="fixture",
            )
        )
        session.add(DataRepository._file_model(state.registered_files[0]))
        session.commit()
    repository = DataRepository(factory)
    formal = mathematical.solve_stage.execution.execution.record
    code = ArtifactRecord(
        artifact_id=formal.code_artifact_id,
        project_id=state.project_id,
        problem_id=state.problem_id,
        execution_run_id=formal.run_id,
        kind=ArtifactKind.GENERATED_CODE,
        name="solve.py",
        mime_type="text/x-python",
        size_bytes=14,
        sha256=formal.code_hash,
        storage_key="fixture/solve.py",
    )
    repository.persist_auxiliary_execution(formal, [code])
    evaluator._repository = repository
    run = evaluator.evaluate(bundle=bundle, mathematical=mathematical, state=state)  # type: ignore[arg-type]
    assert run.result is not None
    assert len(repository.list_executions(state.project_id)) == 2
    assert (
        evaluator.audit_persisted(
            bundle=bundle,
            project_id=state.project_id,
            formal_result_id=mathematical.solve_stage.result.result_id,
            holdout_execution_id=run.execution.run_id,
        )
        == run.result
    )
    formal_validation = evaluator.audit_validation_evidence(
        bundle=bundle,
        project_id=state.project_id,
        formal_result_id=mathematical.solve_stage.result.result_id,
        holdout_execution_id=run.execution.run_id,
    )
    assert formal_validation.result == run.result
    assert formal_validation.randomness is None
    assert formal_validation.holdout_execution_id == run.execution.run_id
