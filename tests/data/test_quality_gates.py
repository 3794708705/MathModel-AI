from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest
from pydantic import ValidationError

from mathmodel_ai.agents.data import DataAgent
from mathmodel_ai.core.errors import ProviderResponseError
from mathmodel_ai.data.quality_gates import (
    data_column_reference_errors,
    data_quality_gate,
    execution_quality_gate,
)
from mathmodel_ai.files.storage import LocalFileStore
from mathmodel_ai.reasoning.prompts import PromptRegistry
from mathmodel_ai.schemas.data import (
    ColumnProfile,
    DataAgentInput,
    DataProfile,
    DataSemanticType,
    DatasetUnderstanding,
    DataUnderstanding,
)
from mathmodel_ai.schemas.execution import ExecutionRecord, ExecutionStatus, SandboxLimits
from mathmodel_ai.schemas.files import ArtifactKind, ArtifactRecord
from mathmodel_ai.schemas.quality import QualityGateStatus


def limits() -> SandboxLimits:
    return SandboxLimits(
        cpu_cores=1,
        memory_mb=128,
        timeout_seconds=5,
        pids_limit=32,
        max_output_bytes=4096,
        max_artifacts=5,
        max_artifact_bytes=4096,
    )


def test_data_gate_rejects_agent_column_or_dataset_invention() -> None:
    dataset_id = uuid4()
    profile = DataProfile(
        dataset_id=dataset_id,
        source_file_id=uuid4(),
        dataset_name="observations",
        row_count=2,
        column_count=1,
        columns=[
            ColumnProfile(
                name="observed",
                source_name="observed",
                physical_dtype="Int64",
                semantic_type=DataSemanticType.INTEGER,
                missing_count=0,
                missing_rate=0,
                unique_count=2,
                unique_rate=1,
            )
        ],
        duplicate_row_count=0,
        duplicate_row_rate=0,
        quality_score=100,
    )
    invented = DataUnderstanding(
        datasets=[
            DatasetUnderstanding(
                dataset_id=dataset_id,
                purpose="test",
                potential_targets=["invented"],
            )
        ],
        confidence=0.8,
    )

    gate = data_quality_gate([profile], invented)

    assert gate.status is QualityGateStatus.RETRY
    assert gate.checks["agent_references_known_columns"] is False
    assert data_column_reference_errors([profile], invented) == [
        f"dataset {dataset_id} references unknown columns: invented"
    ]


@pytest.mark.asyncio
async def test_data_agent_retries_with_exact_invalid_column_feedback() -> None:
    dataset_id = uuid4()
    profile = DataProfile(
        dataset_id=dataset_id,
        source_file_id=uuid4(),
        dataset_name="observations",
        row_count=1,
        column_count=1,
        columns=[
            ColumnProfile(
                name="observed",
                source_name="observed",
                physical_dtype="Int64",
                semantic_type=DataSemanticType.INTEGER,
                missing_count=0,
                missing_rate=0,
                unique_count=1,
                unique_rate=1,
            )
        ],
        duplicate_row_count=0,
        duplicate_row_rate=0,
        quality_score=100,
    )
    input_data = DataAgentInput(raw_problem="Analyze observations", profiles=[profile])
    invented = DataUnderstanding(
        datasets=[
            DatasetUnderstanding(
                dataset_id=dataset_id,
                purpose="test",
                potential_features=["imagined"],
            )
        ],
        confidence=0.5,
    )

    class Provider:
        async def structured_generate(self, _request, _schema):
            return SimpleNamespace(parsed=invented, response=SimpleNamespace())

    agent = DataAgent(router=None, providers=None, prompts=PromptRegistry())
    with pytest.raises(ProviderResponseError, match="unknown columns: imagined") as exc:
        await agent.execute(
            input_data,
            None,
            Provider(),
            SimpleNamespace(selected_model="fixture", selected_reasoning=None),
        )
    repaired = agent.prepare_attempt_input(input_data, None, (str(exc.value),))
    assert "unknown columns: imagined" in repaired.user_guidance[-1]
    assert input_data.user_guidance == []


def test_data_gate_allows_explicit_media_only_evidence_but_not_empty_input() -> None:
    understanding = DataUnderstanding(
        multimodal_observations=["Fixture observation only."],
        confidence=0.4,
    )
    empty = data_quality_gate([], understanding)
    media = data_quality_gate([], understanding, media_present=True)

    assert empty.status is QualityGateStatus.RETRY
    assert empty.checks["data_evidence_present"] is False
    assert media.status is QualityGateStatus.PASS


def test_execution_schema_and_gate_never_accept_mock_success(tmp_path) -> None:
    now = datetime.now(UTC)
    project_id = uuid4()
    problem_id = uuid4()
    artifact_id = uuid4()
    store = LocalFileStore(tmp_path / "store")
    code = b"print(1)\n"
    stored = store.store_artifact(
        code,
        project_id=project_id,
        artifact_id=artifact_id,
        filename="main.py",
    )
    code_artifact = ArtifactRecord(
        artifact_id=artifact_id,
        project_id=project_id,
        problem_id=problem_id,
        kind=ArtifactKind.GENERATED_CODE,
        name="main.py",
        mime_type="text/x-python",
        size_bytes=stored.size_bytes,
        sha256=stored.sha256,
        storage_key=stored.storage_key,
    )
    common = {
        "project_id": project_id,
        "problem_id": problem_id,
        "code_hash": stored.sha256,
        "code_artifact_id": artifact_id,
        "image": "sandbox:test",
        "start_time": now,
        "end_time": now,
        "runtime_seconds": 0,
        "status": ExecutionStatus.SUCCEEDED,
        "exit_code": 0,
        "limits": limits(),
    }
    try:
        ExecutionRecord(**common, is_mock=True)
    except ValidationError as exc:
        assert "mock execution cannot claim success" in str(exc)
    else:
        raise AssertionError("mock success should not validate")

    record = ExecutionRecord(**common, is_mock=False)
    gate = execution_quality_gate(record, store, [code_artifact])
    assert gate.status is QualityGateStatus.PASS
