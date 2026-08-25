import hashlib
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Protocol
from uuid import UUID, uuid4

from mathmodel_ai.core.errors import FileValidationError, StorageError


@dataclass(frozen=True)
class StagedObject:
    path: Path
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class StoredObject:
    storage_key: str
    size_bytes: int
    sha256: str


class FileStore(Protocol):
    """Storage boundary implemented locally in Phase 3 and replaceable by object storage."""

    @property
    def root(self) -> Path: ...

    def stage_stream(
        self,
        stream: BinaryIO,
        *,
        max_bytes: int,
        chunk_size: int = 1024 * 1024,
    ) -> StagedObject: ...

    def commit_upload(
        self,
        staged: StagedObject,
        *,
        project_id: UUID,
        file_id: UUID,
        extension: str,
    ) -> StoredObject: ...

    def store_artifact(
        self,
        data: bytes,
        *,
        project_id: UUID,
        artifact_id: UUID,
        filename: str,
    ) -> StoredObject: ...

    def read_bytes(self, storage_key: str, *, max_bytes: int | None = None) -> bytes: ...

    def resolve(self, storage_key: str) -> Path: ...

    def exists(self, storage_key: str) -> bool: ...

    def discard_staged(self, staged: StagedObject) -> None: ...


class LocalFileStore:
    """Local immutable store with generated keys and guarded path resolution."""

    def __init__(self, root: Path) -> None:
        self._root = root.resolve(strict=False)
        self._staging = self._root / ".staging"

    @property
    def root(self) -> Path:
        return self._root

    def stage_stream(
        self,
        stream: BinaryIO,
        *,
        max_bytes: int,
        chunk_size: int = 1024 * 1024,
    ) -> StagedObject:
        self._staging.mkdir(parents=True, exist_ok=True)
        staging_path = self._staging / f"{uuid4().hex}.upload"
        digest = hashlib.sha256()
        size = 0
        try:
            with staging_path.open("xb") as destination:
                while chunk := stream.read(chunk_size):
                    size += len(chunk)
                    if size > max_bytes:
                        raise FileValidationError(
                            f"upload exceeds configured limit of {max_bytes} bytes"
                        )
                    digest.update(chunk)
                    destination.write(chunk)
                destination.flush()
                os.fsync(destination.fileno())
        except Exception:
            staging_path.unlink(missing_ok=True)
            raise
        return StagedObject(path=staging_path, size_bytes=size, sha256=digest.hexdigest())

    def commit_upload(
        self,
        staged: StagedObject,
        *,
        project_id: UUID,
        file_id: UUID,
        extension: str,
    ) -> StoredObject:
        key = f"projects/{project_id}/files/{file_id}/raw{extension}"
        target = self._resolve_key(key)
        target.parent.mkdir(parents=True, exist_ok=False)
        if target.exists():
            raise StorageError("immutable upload target already exists")
        try:
            os.replace(staged.path, target)
        except OSError as exc:
            raise StorageError("failed to commit uploaded file") from exc
        return StoredObject(
            storage_key=key,
            size_bytes=staged.size_bytes,
            sha256=staged.sha256,
        )

    def store_artifact(
        self,
        data: bytes,
        *,
        project_id: UUID,
        artifact_id: UUID,
        filename: str,
    ) -> StoredObject:
        safe_filename = self._artifact_filename(filename)
        key = f"projects/{project_id}/artifacts/{artifact_id}/{safe_filename}"
        target = self._resolve_key(key)
        target.parent.mkdir(parents=True, exist_ok=False)
        try:
            with target.open("xb") as destination:
                destination.write(data)
                destination.flush()
                os.fsync(destination.fileno())
        except OSError as exc:
            raise StorageError("failed to store immutable artifact") from exc
        return StoredObject(
            storage_key=key,
            size_bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
        )

    def read_bytes(self, storage_key: str, *, max_bytes: int | None = None) -> bytes:
        path = self.resolve(storage_key)
        if max_bytes is not None and path.stat().st_size > max_bytes:
            raise StorageError("stored object exceeds requested read limit")
        return path.read_bytes()

    def resolve(self, storage_key: str) -> Path:
        path = self._resolve_key(storage_key)
        if not path.is_file() or path.is_symlink():
            raise StorageError("stored object does not exist or is not a regular file")
        return path

    def exists(self, storage_key: str) -> bool:
        try:
            self.resolve(storage_key)
        except StorageError:
            return False
        return True

    def discard_staged(self, staged: StagedObject) -> None:
        resolved = staged.path.resolve(strict=False)
        if resolved.parent != self._staging:
            raise StorageError("refusing to discard a path outside staging")
        resolved.unlink(missing_ok=True)

    def _resolve_key(self, storage_key: str) -> Path:
        key = PurePosixPath(storage_key)
        if key.is_absolute() or not key.parts or any(part in {"", ".", ".."} for part in key.parts):
            raise StorageError("invalid storage key")
        candidate = self._root.joinpath(*key.parts).resolve(strict=False)
        if not candidate.is_relative_to(self._root):
            raise StorageError("storage key escapes configured root")
        return candidate

    @staticmethod
    def _artifact_filename(filename: str) -> str:
        if (
            not filename
            or len(filename) > 255
            or "/" in filename
            or "\\" in filename
            or filename in {".", ".."}
            or any(ord(character) < 32 for character in filename)
        ):
            raise StorageError("invalid artifact filename")
        return filename
