"""Verification-side executions must persist without changing problem state."""

import hashlib
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from mathmodel_ai.data.repository import DataRepository
from mathmodel_ai.db.base import Base
from mathmodel_ai.reasoning.repository import ReasoningRepository
from mathmodel_ai.schemas.execution import ExecutionRecord, ExecutionStatus, SandboxLimits
from mathmodel_ai.schemas.files import ArtifactKind, ArtifactRecord


def test_auxiliary_execution_is_atomic_and_does_not_advance_state() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = sessionmaker(engine, expire_on_commit=False)
    reasoning = ReasoningRepository(factory)
    state = reasoning.create_project_problem(name="auxiliary", title="test", raw_problem="test")
    repository = DataRepository(factory)
    run_id = uuid4()
    artifact = ArtifactRecord(
        project_id=state.project_id,
        problem_id=state.problem_id,
        execution_run_id=run_id,
        kind=ArtifactKind.GENERATED_CODE,
        name="predictor.py",
        mime_type="text/x-python",
        size_bytes=4,
        sha256=hashlib.sha256(b"code").hexdigest(),
        storage_key="auxiliary/predictor.py",
    )
    record = ExecutionRecord(
        run_id=run_id,
        project_id=state.project_id,
        problem_id=state.problem_id,
        code_hash=artifact.sha256,
        code_artifact_id=artifact.artifact_id,
        image="test-image",
        end_time=datetime.now(UTC),
        runtime_seconds=0,
        status=ExecutionStatus.FAILED,
        limits=SandboxLimits(
            cpu_cores=0.5,
            memory_mb=64,
            timeout_seconds=5,
            pids_limit=16,
            max_output_bytes=1024,
            max_artifacts=1,
            max_artifact_bytes=1024,
        ),
    )
    with pytest.raises(ValueError, match="artifacts do not match"):
        repository.persist_auxiliary_execution(
            record, [artifact.model_copy(update={"execution_run_id": uuid4()})]
        )
    with pytest.raises(ValueError, match="artifacts do not match"):
        repository.persist_auxiliary_execution(
            record, [artifact.model_copy(update={"sha256": "0" * 64})]
        )
    assert repository.list_executions(state.project_id) == []
    assert repository.list_artifacts(state.project_id) == []

    repository.persist_auxiliary_execution(record, [artifact])
    assert repository.list_executions(state.project_id) == [record]
    persisted = repository.list_artifacts(state.project_id)
    assert len(persisted) == 1
    assert persisted[0].model_dump(exclude={"created_at"}) == artifact.model_dump(
        exclude={"created_at"}
    )
    assert reasoning.load_current(state.project_id).version == state.version
