import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mathmodel_ai.benchmark.causal_evaluation import CausalBenchmarkEvaluator
from mathmodel_ai.core.errors import QualityGateError
from mathmodel_ai.data.repository import DataRepository
from mathmodel_ai.files.storage import LocalFileStore
from mathmodel_ai.mathematical.digests import mathematical_model_digest
from mathmodel_ai.schemas.execution import ExecutionOrigin, ExecutionStatus, SandboxLimits
from mathmodel_ai.schemas.files import FileKind, RegisteredFile
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
    registered = RegisteredFile(
        project_id=model.project_id,
        problem_id=model.problem_id,
        original_name="data.csv",
        safe_name="data.csv",
        extension=".csv",
        kind=FileKind.CSV,
        detected_mime_type="text/csv",
        size_bytes=len(bundle.causal_split.training_csv),
        sha256=bundle.causal_split.training_sha256,
        storage_key="fixture/training.csv",
    )
    state = ProblemState(
        project_id=model.project_id,
        problem_id=model.problem_id,
        title="fixture",
        raw_problem="fixture",
        registered_files=[registered],
    )
    mathematical = SimpleNamespace(
        model_stage=SimpleNamespace(model=model),
        solve_stage=SimpleNamespace(execution=SimpleNamespace(program=program)),
    )
    repository = Mock(spec=DataRepository)
    evaluator = CausalBenchmarkEvaluator(
        store=LocalFileStore(tmp_path / "store"),
        repository=repository,
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


@pytest.mark.sandbox
@pytest.mark.skipif(not _image_available(), reason="Docker image unavailable")
def test_generated_predictor_evaluates_in_isolation_and_persists_evidence(
    tmp_path: Path,
) -> None:
    evaluator, bundle, mathematical, state, repository = _setup(tmp_path, include_predictor=True)
    result = evaluator.evaluate(bundle=bundle, mathematical=mathematical, state=state)  # type: ignore[arg-type]
    assert result.execution.status is ExecutionStatus.SUCCEEDED
    assert result.execution.execution_origin is ExecutionOrigin.GENERATED_PROGRAM
    assert (
        result.execution.generated_program_id
        == mathematical.solve_stage.execution.program.program_id
    )
    assert result.result is not None
    assert len(result.result.predictions) == bundle.causal_split.heldout_rows  # type: ignore[union-attr]
    repository.persist_auxiliary_execution.assert_called_once()
