from io import BytesIO
from pathlib import Path
from uuid import uuid4

import pymupdf  # type: ignore[import-untyped]
import pytest
from openpyxl import Workbook
from PIL import Image

from mathmodel_ai.core.errors import FileValidationError, StorageError
from mathmodel_ai.data.quality_gates import files_quality_gate
from mathmodel_ai.schemas.files import ArtifactKind, FileKind, FileStatus
from mathmodel_ai.schemas.quality import QualityGateStatus
from tests.data.helpers import pipeline_for, stream


def test_csv_ingestion_profiles_real_values_and_preserves_raw(
    tmp_path: Path,
) -> None:
    pipeline = pipeline_for(tmp_path / "store")
    raw = (
        b"id,value,target,region\n"
        b"1,1,10,north\n2,2,20,south\n3,,30,north\n4,3,40,south\n"
        b"5,4,50,north\n6,100,60,south\n5,4,50,north\n"
    )

    result = pipeline.ingest(
        stream(raw),
        project_id=uuid4(),
        problem_id=uuid4(),
        original_name="observations.csv",
        declared_mime_type="text/csv",
    )

    assert result.parsed_file.file.status is FileStatus.PARSED
    assert pipeline.store.read_bytes(result.parsed_file.file.storage_key) == raw
    assert len(result.datasets) == 1
    assert result.datasets[0].row_count == 7
    assert result.profile_bundle is not None
    profile = result.profile_bundle.profiles[0]
    by_name = {column.name: column for column in profile.columns}
    assert by_name["value"].missing_count == 1
    assert by_name["value"].numeric_statistics is not None
    assert by_name["value"].numeric_statistics.maximum == 100
    assert by_name["value"].outliers is not None
    assert by_name["value"].outliers.count == 1
    assert profile.duplicate_row_count == 1
    assert profile.deterministic is True
    assert {item.kind for item in result.artifacts} >= {
        ArtifactKind.ORIGINAL_FILE,
        ArtifactKind.DATASET_PREVIEW,
        ArtifactKind.DATA_PROFILE,
    }


def test_excel_parser_discovers_cross_sheet_key_overlap(tmp_path: Path) -> None:
    workbook = Workbook()
    customers = workbook.active
    customers.title = "customers"
    customers.append(["customer_id", "region"])
    customers.append([1, "north"])
    customers.append([2, "south"])
    orders = workbook.create_sheet("orders")
    orders.append(["customer_id", "amount"])
    orders.append([1, 10.0])
    orders.append([2, 20.0])
    buffer = BytesIO()
    workbook.save(buffer)
    pipeline = pipeline_for(tmp_path / "store")

    result = pipeline.ingest(
        stream(buffer.getvalue()),
        project_id=uuid4(),
        problem_id=uuid4(),
        original_name="relations.xlsx",
        declared_mime_type=("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
    )

    assert len(result.datasets) == 2
    assert result.profile_bundle is not None
    relationships = result.profile_bundle.cross_dataset_relationships
    assert len(relationships) == 1
    assert relationships[0].left_column == "customer_id"
    assert relationships[0].overlap_ratio == 1


def test_pdf_and_image_interfaces_emit_traceable_artifacts(
    tmp_path: Path,
) -> None:
    pipeline = pipeline_for(tmp_path / "store")
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "Mathematical modeling attachment")
    pdf_bytes = document.tobytes()
    document.close()

    pdf = pipeline.ingest(
        stream(pdf_bytes),
        project_id=uuid4(),
        problem_id=uuid4(),
        original_name="problem.pdf",
        declared_mime_type="application/pdf",
    )
    assert pdf.parsed_file.text_extraction is not None
    assert "Mathematical modeling" in pdf.parsed_file.text_extraction.text
    assert ArtifactKind.EXTRACTED_TEXT in {item.kind for item in pdf.artifacts}
    assert ArtifactKind.PDF_PAGE_IMAGE in {item.kind for item in pdf.artifacts}
    media = pipeline.multimodal_asset(pdf.parsed_file.file)
    assert media.mime_type == "application/pdf"
    assert media.byte_size == len(pdf_bytes)

    image_buffer = BytesIO()
    Image.new("RGB", (12, 8), color="white").save(image_buffer, format="PNG")
    image = pipeline.ingest(
        stream(image_buffer.getvalue()),
        project_id=pdf.parsed_file.file.project_id,
        problem_id=pdf.parsed_file.file.problem_id,
        original_name="diagram.png",
        declared_mime_type="image/png",
    )
    assert image.parsed_file.file.kind is FileKind.IMAGE
    assert image.parsed_file.image_metadata is not None
    assert image.parsed_file.image_metadata.width == 12


def test_text_attachment_is_normalized_to_utf8_artifact(tmp_path: Path) -> None:
    pipeline = pipeline_for(tmp_path / "store")
    result = pipeline.ingest(
        stream(b"competition notes\nsecond line"),
        project_id=uuid4(),
        problem_id=uuid4(),
        original_name="notes.txt",
        declared_mime_type="text/plain",
    )

    assert result.parsed_file.file.status is FileStatus.PARSED
    assert result.parsed_file.text_extraction is not None
    assert "competition notes" in result.parsed_file.text_extraction.text
    extracted = next(item for item in result.artifacts if item.kind is ArtifactKind.EXTRACTED_TEXT)
    assert (
        pipeline.store.read_bytes(extracted.storage_key)
        .decode("utf-8")
        .startswith("competition notes")
    )


def test_validation_rejects_path_traversal_and_mime_mismatch(
    tmp_path: Path,
) -> None:
    pipeline = pipeline_for(tmp_path / "store")
    with pytest.raises(FileValidationError, match="filename"):
        pipeline.ingest(
            stream(b"a,b\n1,2\n"),
            project_id=uuid4(),
            problem_id=uuid4(),
            original_name="../escape.csv",
            declared_mime_type="text/csv",
        )
    with pytest.raises(FileValidationError, match="declared MIME"):
        pipeline.ingest(
            stream(b"a,b\n1,2\n"),
            project_id=uuid4(),
            problem_id=uuid4(),
            original_name="table.csv",
            declared_mime_type="image/png",
        )


def test_storage_never_resolves_untrusted_keys(tmp_path: Path) -> None:
    pipeline = pipeline_for(tmp_path / "store")
    with pytest.raises(StorageError, match="invalid storage key"):
        pipeline.store.read_bytes("../secret")


def test_files_gate_detects_raw_byte_tampering(tmp_path: Path) -> None:
    pipeline = pipeline_for(tmp_path / "store")
    result = pipeline.ingest(
        stream(b"a,b\n1,2\n"),
        project_id=uuid4(),
        problem_id=uuid4(),
        original_name="table.csv",
        declared_mime_type="text/csv",
    )
    assert files_quality_gate(result, pipeline.store).status is QualityGateStatus.PASS

    pipeline.store.resolve(result.parsed_file.file.storage_key).write_bytes(b"tampered")
    gate = files_quality_gate(result, pipeline.store)

    assert gate.status is QualityGateStatus.RETRY
    assert gate.checks["raw_hash_matches"] is False


def test_image_pixel_limit_is_enforced_before_decode(tmp_path: Path) -> None:
    pipeline = pipeline_for(tmp_path / "store")
    image_buffer = BytesIO()
    Image.new("RGB", (1001, 1000), color="white").save(image_buffer, format="PNG")

    with pytest.raises(FileValidationError, match="pixel count"):
        pipeline.ingest(
            stream(image_buffer.getvalue()),
            project_id=uuid4(),
            problem_id=uuid4(),
            original_name="oversized.png",
            declared_mime_type="image/png",
        )
