import base64
import binascii
from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FileKind(StrEnum):
    CSV = "csv"
    EXCEL = "excel"
    PDF = "pdf"
    IMAGE = "image"
    TEXT = "text"


class FileStatus(StrEnum):
    VALIDATED = "VALIDATED"
    PARSED = "PARSED"
    FAILED = "FAILED"


class ArtifactKind(StrEnum):
    ORIGINAL_FILE = "original_file"
    EXTRACTED_TEXT = "extracted_text"
    DATASET_PREVIEW = "dataset_preview"
    DATA_PROFILE = "data_profile"
    PDF_PAGE_IMAGE = "pdf_page_image"
    SANDBOX_OUTPUT = "sandbox_output"
    GENERATED_CODE = "generated_code"
    VERIFICATION_TRACE = "verification_trace"


class RegisteredFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_id: UUID = Field(default_factory=uuid4)
    project_id: UUID
    problem_id: UUID
    original_name: str = Field(min_length=1, max_length=255)
    safe_name: str = Field(min_length=1, max_length=255)
    extension: str = Field(pattern=r"^\.[a-z0-9]+$")
    kind: FileKind
    declared_mime_type: str | None = None
    detected_mime_type: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    storage_key: str = Field(min_length=1)
    status: FileStatus = FileStatus.VALIDATED
    validation_warnings: list[str] = Field(default_factory=list)
    parser_name: str | None = None
    parser_version: str | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ArtifactRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_id: UUID = Field(default_factory=uuid4)
    project_id: UUID
    problem_id: UUID
    source_file_id: UUID | None = None
    execution_run_id: UUID | None = None
    kind: ArtifactKind
    name: str = Field(min_length=1, max_length=255)
    mime_type: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    storage_key: str = Field(min_length=1)
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class TextExtraction(BaseModel):
    text: str
    page_count: int | None = Field(default=None, ge=0)
    pages_with_text: int | None = Field(default=None, ge=0)
    warnings: list[str] = Field(default_factory=list)


class ImageMetadata(BaseModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    format: str = Field(min_length=1)
    mode: str = Field(min_length=1)
    frame_count: int = Field(default=1, ge=1)


class MultimodalAsset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_id: UUID
    mime_type: str = Field(pattern=r"^(image/(png|jpeg|webp)|application/pdf)$")
    data_base64: str = Field(min_length=1)
    byte_size: int = Field(gt=0)
    description: str = Field(min_length=1)

    @model_validator(mode="after")
    def inline_size_matches_base64_contract(self) -> "MultimodalAsset":
        try:
            decoded = base64.b64decode(self.data_base64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("media payload is not valid base64") from exc
        if len(decoded) != self.byte_size:
            raise ValueError("base64 media payload size does not match source bytes")
        return self


class ParsedFile(BaseModel):
    file: RegisteredFile
    dataset_ids: list[UUID] = Field(default_factory=list)
    artifact_ids: list[UUID] = Field(default_factory=list)
    text_extraction: TextExtraction | None = None
    image_metadata: ImageMetadata | None = None
    warnings: list[str] = Field(default_factory=list)
