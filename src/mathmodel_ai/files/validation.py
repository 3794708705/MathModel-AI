import csv
import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from charset_normalizer import from_bytes
from PIL import Image, UnidentifiedImageError

from mathmodel_ai.core.errors import FileValidationError
from mathmodel_ai.schemas.files import FileKind


@dataclass(frozen=True)
class ValidatedFile:
    original_name: str
    safe_name: str
    extension: str
    kind: FileKind
    declared_mime_type: str | None
    detected_mime_type: str
    warnings: list[str]
    text_encoding: str | None = None


@dataclass(frozen=True)
class FileTypeSpec:
    kind: FileKind
    detected_mime: str
    declared_mimes: frozenset[str]


_SPECS: dict[str, FileTypeSpec] = {
    ".csv": FileTypeSpec(
        FileKind.CSV,
        "text/csv",
        frozenset({"text/csv", "application/csv", "text/plain"}),
    ),
    ".xlsx": FileTypeSpec(
        FileKind.EXCEL,
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        frozenset({"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"}),
    ),
    ".pdf": FileTypeSpec(
        FileKind.PDF,
        "application/pdf",
        frozenset({"application/pdf"}),
    ),
    ".png": FileTypeSpec(FileKind.IMAGE, "image/png", frozenset({"image/png"})),
    ".jpg": FileTypeSpec(FileKind.IMAGE, "image/jpeg", frozenset({"image/jpeg"})),
    ".jpeg": FileTypeSpec(FileKind.IMAGE, "image/jpeg", frozenset({"image/jpeg"})),
    ".webp": FileTypeSpec(FileKind.IMAGE, "image/webp", frozenset({"image/webp"})),
    ".txt": FileTypeSpec(FileKind.TEXT, "text/plain", frozenset({"text/plain"})),
}


def csv_delimiter(text: str) -> str:
    """Select a delimiter only when strict CSV parsing gives a consistent table."""
    candidates: list[tuple[int, str]] = []
    for delimiter in (",", ";", "\t", "|"):
        try:
            rows = [
                row for row in csv.reader(
                    io.StringIO(text, newline=""), delimiter=delimiter, strict=True
                )
                if row
            ]
        except csv.Error:
            continue
        if not rows or len(rows[0]) < 2:
            continue
        width = len(rows[0])
        if all(len(row) == width for row in rows):
            candidates.append((width, delimiter))
    if not candidates:
        raise ValueError("CSV has no consistent multi-column delimiter")
    maximum = max(width for width, _ in candidates)
    winners = [delimiter for width, delimiter in candidates if width == maximum]
    if len(winners) != 1:
        raise ValueError("CSV delimiter is ambiguous")
    return winners[0]


class FileValidator:
    def __init__(
        self,
        *,
        max_archive_entries: int,
        max_archive_uncompressed_bytes: int,
        max_archive_ratio: float,
        max_image_pixels: int,
    ) -> None:
        self._max_archive_entries = max_archive_entries
        self._max_archive_uncompressed_bytes = max_archive_uncompressed_bytes
        self._max_archive_ratio = max_archive_ratio
        self._max_image_pixels = max_image_pixels

    def validate(
        self, path: Path, *, original_name: str, declared_mime_type: str | None
    ) -> ValidatedFile:
        safe_name, extension = self._validate_name(original_name)
        spec = _SPECS.get(extension)
        if spec is None:
            raise FileValidationError(f"unsupported file extension: {extension or '<none>'}")
        if path.stat().st_size == 0:
            raise FileValidationError("empty files are not accepted")

        warnings: list[str] = []
        normalized_declared = (
            declared_mime_type.split(";", maxsplit=1)[0].strip().lower()
            if declared_mime_type
            else None
        )
        if normalized_declared in {None, "", "application/octet-stream"}:
            warnings.append("client MIME type was absent or generic; byte validation was used")
        elif normalized_declared not in spec.declared_mimes:
            raise FileValidationError(
                f"declared MIME {normalized_declared!r} does not match {extension}"
            )

        encoding: str | None = None
        if spec.kind is FileKind.PDF:
            self._validate_pdf(path)
        elif spec.kind is FileKind.EXCEL:
            self._validate_xlsx(path)
        elif spec.kind is FileKind.IMAGE:
            self._validate_image(path, spec.detected_mime)
        elif spec.kind in {FileKind.CSV, FileKind.TEXT}:
            encoding = self._validate_text(path, require_csv=spec.kind is FileKind.CSV)
        return ValidatedFile(
            original_name=original_name,
            safe_name=safe_name,
            extension=extension,
            kind=spec.kind,
            declared_mime_type=normalized_declared,
            detected_mime_type=spec.detected_mime,
            warnings=warnings,
            text_encoding=encoding,
        )

    @staticmethod
    def _validate_name(original_name: str) -> tuple[str, str]:
        if (
            not original_name
            or len(original_name) > 255
            or "/" in original_name
            or "\\" in original_name
            or original_name in {".", ".."}
            or any(ord(character) < 32 for character in original_name)
        ):
            raise FileValidationError("unsafe upload filename")
        extension = Path(original_name).suffix.casefold()
        stem = Path(original_name).stem
        safe_stem = re.sub(r"[^\w.-]+", "_", stem, flags=re.UNICODE).strip("._")
        if not safe_stem:
            safe_stem = "upload"
        return f"{safe_stem}{extension}", extension

    @staticmethod
    def _validate_pdf(path: Path) -> None:
        with path.open("rb") as source:
            if source.read(5) != b"%PDF-":
                raise FileValidationError("PDF signature is missing")
            source.seek(-min(path.stat().st_size, 2048), io.SEEK_END)
            if b"%%EOF" not in source.read():
                raise FileValidationError("PDF end marker is missing")

    def _validate_xlsx(self, path: Path) -> None:
        try:
            with zipfile.ZipFile(path) as archive:
                infos = archive.infolist()
                if len(infos) > self._max_archive_entries:
                    raise FileValidationError("XLSX contains too many archive entries")
                names = {info.filename for info in infos}
                required = {"[Content_Types].xml", "xl/workbook.xml"}
                if not required <= names:
                    raise FileValidationError("file is not a valid XLSX package")
                total_uncompressed = 0
                total_compressed = 0
                for info in infos:
                    normalized = PurePosixPath(info.filename.replace("\\", "/"))
                    if normalized.is_absolute() or ".." in normalized.parts:
                        raise FileValidationError("XLSX archive contains an unsafe path")
                    total_uncompressed += info.file_size
                    total_compressed += max(info.compress_size, 1)
                if total_uncompressed > self._max_archive_uncompressed_bytes:
                    raise FileValidationError("XLSX expanded size exceeds configured limit")
                if total_uncompressed / max(total_compressed, 1) > self._max_archive_ratio:
                    raise FileValidationError("XLSX compression ratio exceeds configured limit")
                bad_member = archive.testzip()
                if bad_member is not None:
                    raise FileValidationError("XLSX archive checksum validation failed")
        except (zipfile.BadZipFile, OSError) as exc:
            raise FileValidationError("invalid XLSX archive") from exc

    def _validate_image(self, path: Path, expected_mime: str) -> None:
        previous_limit = Image.MAX_IMAGE_PIXELS
        # Enforce the configured hard limit explicitly before pixel decoding.
        Image.MAX_IMAGE_PIXELS = None
        try:
            with Image.open(path) as image:
                actual = Image.MIME.get(image.format or "")
                if actual != expected_mime:
                    raise FileValidationError(
                        f"image bytes are {actual or 'unknown'}, expected {expected_mime}"
                    )
                if image.width * image.height > self._max_image_pixels:
                    raise FileValidationError("image pixel count exceeds configured limit")
                image.verify()
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise FileValidationError("invalid or unsafe image") from exc
        finally:
            Image.MAX_IMAGE_PIXELS = previous_limit

    @staticmethod
    def _validate_text(path: Path, *, require_csv: bool) -> str:
        data = path.read_bytes()
        if b"\x00" in data:
            raise FileValidationError("text file contains NUL bytes")
        match = from_bytes(data).best()
        if match is None or match.encoding is None:
            raise FileValidationError("text encoding could not be determined")
        try:
            text = str(match)
        except UnicodeError as exc:
            raise FileValidationError("text decoding failed") from exc
        if require_csv:
            try:
                delimiter = csv_delimiter(text)
                rows = list(
                    csv.reader(
                        io.StringIO(text, newline=""), delimiter=delimiter, strict=True
                    )
                )
            except (csv.Error, ValueError) as exc:
                raise FileValidationError("CSV dialect could not be validated") from exc
            if not rows or max((len(row) for row in rows), default=0) < 1:
                raise FileValidationError("CSV contains no readable rows")
        return match.encoding
