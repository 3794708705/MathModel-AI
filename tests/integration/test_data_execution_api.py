from io import BytesIO
from pathlib import Path
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from mathmodel_ai.core.config import Settings
from mathmodel_ai.db.base import Base
from mathmodel_ai.main import create_app
from mathmodel_ai.providers.factory import ProviderRegistry
from mathmodel_ai.providers.mock import MockProvider
from mathmodel_ai.schemas.data import DatasetUnderstanding, DataUnderstanding
from mathmodel_ai.schemas.execution import ExecutionStatus
from mathmodel_ai.schemas.quality import DataWorkflowStage


@pytest.mark.integration
def test_files_data_execution_api_preserves_order_and_truth(tmp_path: Path) -> None:
    mock = MockProvider()
    app = create_app(
        Settings(
            environment="test",
            database_url="sqlite+pysqlite:///:memory:",
            default_provider="mock",
            reasoning_max_retries=0,
            storage_root=tmp_path / "storage",
            sandbox_root=tmp_path / "sandbox",
            sandbox_image="deliberately-missing-phase3-image:test",
        ),
        providers=ProviderRegistry([mock]),
    )
    Base.metadata.create_all(app.state.engine)

    with TestClient(app) as client:
        created = client.post(
            "/api/v1/projects",
            json={
                "name": "Phase 3 workflow",
                "title": "Analyze observations",
                "raw_problem": (
                    "Analyze the supplied observations, identify usable variables, "
                    "and prepare a reproducible computation."
                ),
            },
        )
        assert created.status_code == 201, created.text
        project_id = created.json()["project_id"]
        project_uuid = UUID(project_id)

        upload = client.post(
            f"/api/v1/projects/{project_id}/files",
            files={
                "upload": (
                    "observations.csv",
                    b"id,value,target\n1,2,4\n2,3,6\n3,,8\n",
                    "text/csv",
                )
            },
        )
        assert upload.status_code == 201, upload.text
        uploaded = upload.json()
        assert uploaded["gate"]["status"] == "PASS"
        assert uploaded["data_stage"] == "FILES"
        assert uploaded["parsed_file"]["file"]["status"] == "PARSED"
        dataset_id = UUID(uploaded["datasets"][0]["dataset_id"])

        mock.queue(
            DataUnderstanding(
                datasets=[
                    DatasetUnderstanding(
                        dataset_id=dataset_id,
                        purpose="Schema-level workflow fixture only",
                        potential_features=["value"],
                        potential_targets=["target"],
                        data_quality_risks=["value contains a missing observation"],
                    )
                ],
                problem_data_alignment=["Fixture output; no mathematical claim."],
                confidence=0.5,
            ).model_dump_json()
        )
        analyzed = client.post(
            f"/api/v1/projects/{project_id}/data/analyze",
            json={"media_file_ids": [], "user_guidance": []},
        )
        assert analyzed.status_code == 200, analyzed.text
        assert analyzed.json()["gate"]["status"] == "PASS"
        assert analyzed.json()["data_stage"] == "DATA"
        assert analyzed.json()["is_mock"] is True

        execution = client.post(
            f"/api/v1/projects/{project_id}/executions",
            json={"code": "print('never run without the image')", "input_file_ids": []},
        )
        assert execution.status_code == 200, execution.text
        assert execution.json()["execution"]["status"] == ExecutionStatus.UNAVAILABLE
        assert execution.json()["gate"]["status"] == "RETRY"
        assert execution.json()["data_stage"] == "DATA"

        assert len(client.get(f"/api/v1/projects/{project_id}/files").json()) == 1
        assert len(client.get(f"/api/v1/projects/{project_id}/datasets").json()) == 1
        assert len(client.get(f"/api/v1/projects/{project_id}/data-profiles").json()) == 1
        assert len(client.get(f"/api/v1/projects/{project_id}/artifacts").json()) >= 4
        executions = client.get(f"/api/v1/projects/{project_id}/executions").json()
        assert len(executions) == 1
        assert executions[0]["status"] == "UNAVAILABLE"

        restored = app.state.reasoning_repository.load_current(project_uuid)
        assert restored.data_stage is DataWorkflowStage.DATA
        assert restored.data_understanding is not None
        assert restored.execution_records[0].status is ExecutionStatus.UNAVAILABLE
        assert len(restored.files) == 1
        assert restored.files[0].is_original is True
        assert len(restored.code_files) == 1
        assert restored.code_files[0].content_hash == restored.execution_records[0].code_hash
        assert [entry.to_stage.value for entry in restored.data_stage_history] == [
            "FILES",
            "DATA",
            "DATA",
        ]


@pytest.mark.integration
def test_execution_endpoint_rejects_skipping_files_and_data(tmp_path: Path) -> None:
    app = create_app(
        Settings(
            environment="test",
            database_url="sqlite+pysqlite:///:memory:",
            storage_root=tmp_path / "storage",
            sandbox_root=tmp_path / "sandbox",
        )
    )
    Base.metadata.create_all(app.state.engine)
    with TestClient(app) as client:
        created = client.post(
            "/api/v1/projects",
            json={
                "name": "No skipping",
                "title": "Guard workflow order",
                "raw_problem": (
                    "Verify that execution cannot skip required data preparation stages."
                ),
            },
        )
        project_id = created.json()["project_id"]
        response = client.post(
            f"/api/v1/projects/{project_id}/executions",
            json={"code": "print(1)"},
        )
    assert response.status_code == 400
    assert "requires the DATA stage" in response.json()["detail"]


@pytest.mark.integration
def test_upload_transport_rejects_oversized_multipart_before_parsing(tmp_path: Path) -> None:
    app = create_app(
        Settings(
            environment="test",
            database_url="sqlite+pysqlite:///:memory:",
            storage_root=tmp_path / "storage",
            sandbox_root=tmp_path / "sandbox",
            max_upload_bytes=1024,
        )
    )
    Base.metadata.create_all(app.state.engine)
    with TestClient(app) as client:
        created = client.post(
            "/api/v1/projects",
            json={
                "name": "Upload limit",
                "title": "Guard attachment size",
                "raw_problem": "Reject oversized competition attachments before parsing them.",
            },
        )
        project_id = created.json()["project_id"]
        response = client.post(
            f"/api/v1/projects/{project_id}/files",
            files={"upload": ("large.csv", b"a" * 70_000, "text/csv")},
        )

    assert response.status_code == 413


@pytest.mark.integration
def test_media_only_attachment_can_reach_multimodal_data_agent(tmp_path: Path) -> None:
    mock = MockProvider()
    app = create_app(
        Settings(
            environment="test",
            database_url="sqlite+pysqlite:///:memory:",
            default_provider="mock",
            reasoning_max_retries=0,
            storage_root=tmp_path / "storage",
            sandbox_root=tmp_path / "sandbox",
        ),
        providers=ProviderRegistry([mock]),
    )
    Base.metadata.create_all(app.state.engine)
    image = BytesIO()
    Image.new("RGB", (8, 8), color="white").save(image, format="PNG")

    with TestClient(app) as client:
        created = client.post(
            "/api/v1/projects",
            json={
                "name": "Media only",
                "title": "Interpret diagram",
                "raw_problem": "Interpret the supplied diagram without inventing numeric data.",
            },
        )
        project_id = created.json()["project_id"]
        uploaded = client.post(
            f"/api/v1/projects/{project_id}/files",
            files={"upload": ("diagram.png", image.getvalue(), "image/png")},
        )
        assert uploaded.status_code == 201, uploaded.text
        assert uploaded.json()["datasets"] == []
        file_id = uploaded.json()["parsed_file"]["file"]["file_id"]
        mock.queue(
            DataUnderstanding(
                multimodal_observations=["Schema fixture: diagram attached."],
                confidence=0.4,
            ).model_dump_json()
        )

        analyzed = client.post(
            f"/api/v1/projects/{project_id}/data/analyze",
            json={"media_file_ids": [file_id]},
        )

    assert analyzed.status_code == 200, analyzed.text
    assert analyzed.json()["data_stage"] == "DATA"
    assert analyzed.json()["output"]["datasets"] == []
    assert analyzed.json()["is_mock"] is True
