import base64
import json
import logging
from dataclasses import dataclass
from typing import BinaryIO
from uuid import UUID, uuid4

from mathmodel_ai.core.errors import FileParseError, StorageError
from mathmodel_ai.data.profiler import DataProfiler
from mathmodel_ai.files.parsers import ParserRegistry
from mathmodel_ai.files.storage import FileStore
from mathmodel_ai.files.validation import FileValidator
from mathmodel_ai.schemas.data import DataProfileBundle, DatasetRecord
from mathmodel_ai.schemas.files import (
    ArtifactKind,
    ArtifactRecord,
    FileKind,
    FileStatus,
    MultimodalAsset,
    ParsedFile,
    RegisteredFile,
)


@dataclass(frozen=True)
class ProcessedFile:
    parsed_file: ParsedFile
    datasets: list[DatasetRecord]
    profile_bundle: DataProfileBundle | None
    artifacts: list[ArtifactRecord]


class FilePipeline:
    """Upload-to-profile pipeline; persistence is delegated to a repository."""

    def __init__(
        self,
        *,
        store: FileStore,
        validator: FileValidator,
        parsers: ParserRegistry,
        profiler: DataProfiler,
        max_upload_bytes: int,
        max_multimodal_inline_bytes: int,
    ) -> None:
        self._store = store
        self._validator = validator
        self._parsers = parsers
        self._profiler = profiler
        self._max_upload_bytes = max_upload_bytes
        self._max_multimodal_inline_bytes = max_multimodal_inline_bytes
        self._logger = logging.getLogger("mathmodel_ai.files.pipeline")

    @property
    def store(self) -> FileStore:
        return self._store

    def ingest(
        self,
        stream: BinaryIO,
        *,
        project_id: UUID,
        problem_id: UUID,
        original_name: str,
        declared_mime_type: str | None,
    ) -> ProcessedFile:
        file_id = uuid4()
        staged = self._store.stage_stream(stream, max_bytes=self._max_upload_bytes)
        try:
            validated = self._validator.validate(
                staged.path,
                original_name=original_name,
                declared_mime_type=declared_mime_type,
            )
            stored = self._store.commit_upload(
                staged,
                project_id=project_id,
                file_id=file_id,
                extension=validated.extension,
            )
        except Exception:
            self._store.discard_staged(staged)
            raise

        base_file = RegisteredFile(
            file_id=file_id,
            project_id=project_id,
            problem_id=problem_id,
            original_name=validated.original_name,
            safe_name=validated.safe_name,
            extension=validated.extension,
            kind=validated.kind,
            declared_mime_type=validated.declared_mime_type,
            detected_mime_type=validated.detected_mime_type,
            size_bytes=stored.size_bytes,
            sha256=stored.sha256,
            storage_key=stored.storage_key,
            validation_warnings=validated.warnings,
        )
        artifacts = [self._original_artifact(base_file)]
        try:
            parsed = self._parsers.parse(
                validated.kind,
                self._store.resolve(stored.storage_key),
                name=validated.safe_name,
            )
            datasets = [
                DatasetRecord(
                    project_id=project_id,
                    problem_id=problem_id,
                    source_file_id=file_id,
                    name=item.name,
                    sheet_name=item.sheet_name,
                    row_count=item.frame.height,
                    column_count=item.frame.width,
                    columns=item.frame.columns,
                )
                for item in parsed.datasets
            ]
            pairs = list(zip(datasets, parsed.datasets, strict=True))
            profile_bundle = self._profiler.profile(pairs) if pairs else None
            for generated in parsed.artifacts:
                artifacts.append(
                    self._store_generated_artifact(
                        generated.data,
                        project_id=project_id,
                        problem_id=problem_id,
                        source_file_id=file_id,
                        kind=generated.kind,
                        name=generated.name,
                        mime_type=generated.mime_type,
                        metadata=generated.metadata,
                    )
                )
            if profile_bundle is not None:
                for profile in profile_bundle.profiles:
                    artifacts.append(
                        self._store_generated_artifact(
                            profile.model_dump_json(indent=2).encode("utf-8"),
                            project_id=project_id,
                            problem_id=problem_id,
                            source_file_id=file_id,
                            kind=ArtifactKind.DATA_PROFILE,
                            name=f"profile-{profile.profile_id}.json",
                            mime_type="application/json",
                            metadata={"dataset_id": str(profile.dataset_id)},
                        )
                    )
            completed_file = base_file.model_copy(
                update={
                    "status": FileStatus.PARSED,
                    "parser_name": parsed.parser_name,
                    "parser_version": parsed.parser_version,
                }
            )
            parsed_file = ParsedFile(
                file=completed_file,
                dataset_ids=[item.dataset_id for item in datasets],
                artifact_ids=[item.artifact_id for item in artifacts],
                text_extraction=parsed.text_extraction,
                image_metadata=parsed.image_metadata,
                warnings=[*validated.warnings, *parsed.warnings],
            )
            self._logger.info(
                "file ingestion completed",
                extra={
                    "project_id": str(project_id),
                    "problem_id": str(problem_id),
                    "file_id": str(file_id),
                    "kind": validated.kind.value,
                    "status": FileStatus.PARSED.value,
                },
            )
            return ProcessedFile(
                parsed_file=parsed_file,
                datasets=datasets,
                profile_bundle=profile_bundle,
                artifacts=artifacts,
            )
        except (FileParseError, StorageError, ValueError) as exc:
            self._logger.warning(
                "file parsing failed",
                extra={
                    "project_id": str(project_id),
                    "problem_id": str(problem_id),
                    "file_id": str(file_id),
                    "kind": validated.kind.value,
                },
            )
            failed_file = base_file.model_copy(
                update={"status": FileStatus.FAILED, "error": str(exc)}
            )
            return ProcessedFile(
                parsed_file=ParsedFile(
                    file=failed_file,
                    artifact_ids=[item.artifact_id for item in artifacts],
                    warnings=validated.warnings,
                ),
                datasets=[],
                profile_bundle=None,
                artifacts=artifacts,
            )

    def multimodal_asset(self, file: RegisteredFile) -> MultimodalAsset:
        if file.kind not in {FileKind.IMAGE, FileKind.PDF}:
            raise FileParseError("only image and PDF files can become multimodal assets")
        data = self._store.read_bytes(file.storage_key, max_bytes=self._max_multimodal_inline_bytes)
        return MultimodalAsset(
            file_id=file.file_id,
            mime_type=file.detected_mime_type,
            data_base64=base64.b64encode(data).decode("ascii"),
            byte_size=len(data),
            description=f"Original competition attachment {file.safe_name}",
        )

    @staticmethod
    def _original_artifact(file: RegisteredFile) -> ArtifactRecord:
        return ArtifactRecord(
            project_id=file.project_id,
            problem_id=file.problem_id,
            source_file_id=file.file_id,
            kind=ArtifactKind.ORIGINAL_FILE,
            name=file.safe_name,
            mime_type=file.detected_mime_type,
            size_bytes=file.size_bytes,
            sha256=file.sha256,
            storage_key=file.storage_key,
            metadata={"immutable": True},
        )

    def _store_generated_artifact(
        self,
        data: bytes,
        *,
        project_id: UUID,
        problem_id: UUID,
        source_file_id: UUID,
        kind: ArtifactKind,
        name: str,
        mime_type: str,
        metadata: dict[str, str | int | float | bool | None],
    ) -> ArtifactRecord:
        artifact_id = uuid4()
        stored = self._store.store_artifact(
            data,
            project_id=project_id,
            artifact_id=artifact_id,
            filename=name,
        )
        return ArtifactRecord(
            artifact_id=artifact_id,
            project_id=project_id,
            problem_id=problem_id,
            source_file_id=source_file_id,
            kind=kind,
            name=name,
            mime_type=mime_type,
            size_bytes=stored.size_bytes,
            sha256=stored.sha256,
            storage_key=stored.storage_key,
            metadata=json.loads(json.dumps(metadata)),
        )
