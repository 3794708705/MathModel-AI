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
from mathmodel_ai.schemas.execution import (
    ExecutionOrigin,
    ExecutionRecord,
    ExecutionStatus,
    SandboxLimits,
)
from mathmodel_ai.schemas.files import ArtifactKind, ArtifactRecord, FileKind, RegisteredFile
from mathmodel_ai.schemas.problem_state import ProblemState
from mathmodel_ai.schemas.program import (
    GeneratedProgram,
    GeneratedProgramStatus,
    GeneratedSourceFile,
    generated_program_hash,
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


def test_causal_evaluator_rejects_wrong_training_and_model_identity(tmp_path: Path) -> None:
    evaluator, bundle, mathematical, state, repository = _setup(tmp_path, include_predictor=True)
    state.registered_files[0].sha256 = "0" * 64
    with pytest.raises(QualityGateError, match="CAUSAL_HOLDOUT_TRAINING_INPUT_NOT_BOUND"):
        evaluator.evaluate(bundle=bundle, mathematical=mathematical, state=state)  # type: ignore[arg-type]
    state.registered_files[0].sha256 = bundle.causal_split.training_sha256  # type: ignore[union-attr]
    mathematical.solve_stage.execution.program.model_digest = "0" * 64
    with pytest.raises(QualityGateError, match="CAUSAL_HOLDOUT_GENERATED_PROGRAM_NOT_BOUND"):
        evaluator.evaluate(bundle=bundle, mathematical=mathematical, state=state)  # type: ignore[arg-type]
    repository.persist_auxiliary_execution.assert_not_called()


def test_causal_evaluator_rejects_a_result_from_another_solver_run(tmp_path: Path) -> None:
    evaluator, bundle, mathematical, state, repository = _setup(tmp_path, include_predictor=True)
    mathematical.solve_stage.result.solver_run_id = uuid4()
    with pytest.raises(QualityGateError, match="CAUSAL_HOLDOUT_FORMAL_RESULT_NOT_BOUND"):
        evaluator.evaluate(bundle=bundle, mathematical=mathematical, state=state)  # type: ignore[arg-type]
    repository.persist_auxiliary_execution.assert_not_called()


@pytest.mark.sandbox
@pytest.mark.skipif(not _image_available(), reason="Docker image unavailable")
def test_generated_predictor_evaluates_in_isolation_and_persists_evidence(
    tmp_path: Path,
) -> None:
    evaluator, bundle, mathematical, state, repository = _setup(tmp_path, include_predictor=True)
    result = evaluator.evaluate(bundle=bundle, mathematical=mathematical, state=state)  # type: ignore[arg-type]
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
    assert formal_validation.holdout_execution_id == run.execution.run_id
