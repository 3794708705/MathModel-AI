from __future__ import annotations

from collections import defaultdict

from mathmodel_ai.paper.hashing import canonical_json_bytes, sha256_bytes, sha256_json
from mathmodel_ai.schemas.mathematical import MathematicalModel
from mathmodel_ai.schemas.paper import (
    DocumentObjectType,
    DocumentRegistryEntry,
    FigureRecord,
    ReferenceRecord,
    RegistryGenerationStatus,
    TableRecord,
)

_PREFIXES = {
    DocumentObjectType.SECTION: "SEC-",
    DocumentObjectType.EQUATION: "EQ-",
    DocumentObjectType.FIGURE: "FIG-",
    DocumentObjectType.TABLE: "TAB-",
    DocumentObjectType.REFERENCE: "REF-",
    DocumentObjectType.CLAIM: "CLAIM-",
}


class DocumentRegistry:
    """Single deterministic namespace for all rendered document objects."""

    def __init__(self, entries: list[DocumentRegistryEntry] | None = None) -> None:
        self._entries: dict[str, DocumentRegistryEntry] = {}
        self._counts: dict[DocumentObjectType, int] = defaultdict(int)
        for entry in entries or []:
            self.register(entry)

    def register(self, entry: DocumentRegistryEntry) -> DocumentRegistryEntry:
        if not entry.object_id.startswith(_PREFIXES[entry.object_type]):
            raise ValueError("document object prefix does not match object type")
        if entry.object_id in self._entries:
            raise ValueError(f"duplicate document object ID: {entry.object_id}")
        if any(
            item.object_type is entry.object_type and item.source_id == entry.source_id
            for item in self._entries.values()
        ):
            raise ValueError("document source is already registered")
        self._entries[entry.object_id] = entry
        self._counts[entry.object_type] = max(self._counts[entry.object_type], entry.order)
        return entry

    def add(
        self,
        object_type: DocumentObjectType,
        *,
        source_id: str,
        content: object,
        object_id: str | None = None,
    ) -> DocumentRegistryEntry:
        order = self._counts[object_type] + 1
        identifier = object_id or f"{_PREFIXES[object_type]}{order:03d}"
        entry = DocumentRegistryEntry(
            object_id=identifier,
            object_type=object_type,
            source_id=source_id,
            order=order,
            content_hash=sha256_json({"content": content}),
        )
        return self.register(entry)

    def get(self, object_id: str) -> DocumentRegistryEntry | None:
        return self._entries.get(object_id)

    def require(self, object_id: str) -> DocumentRegistryEntry:
        entry = self.get(object_id)
        if entry is None:
            raise KeyError(object_id)
        return entry

    def entries(self) -> list[DocumentRegistryEntry]:
        return sorted(
            self._entries.values(),
            key=lambda item: (item.object_type.value, item.order, item.object_id),
        )

    @classmethod
    def from_model(cls, model: MathematicalModel) -> DocumentRegistry:
        registry = cls()
        for equation in model.equations:
            registry.add(
                DocumentObjectType.EQUATION,
                source_id=equation.equation_id,
                object_id=equation.equation_id,
                content=equation.model_dump(mode="json"),
            )
        return registry


class CitationRegistry:
    def __init__(self, references: list[ReferenceRecord]) -> None:
        identifiers: set[str] = set()
        dois: set[str] = set()
        sources: set[tuple[str, str]] = set()
        self._records: dict[str, ReferenceRecord] = {}
        for record in references:
            if record.reference_id in identifiers:
                raise ValueError("duplicate reference ID")
            if record.doi is not None and record.doi in dois:
                raise ValueError("duplicate DOI")
            source_key = (record.source.value, record.source_id)
            if source_key in sources:
                raise ValueError("duplicate literature source record")
            identifiers.add(record.reference_id)
            if record.doi is not None:
                dois.add(record.doi)
            sources.add(source_key)
            self._records[record.reference_id] = record

    def get(self, reference_id: str) -> ReferenceRecord | None:
        return self._records.get(reference_id)

    def records(self) -> list[ReferenceRecord]:
        return sorted(self._records.values(), key=lambda item: item.reference_id)

    def identifiers(self) -> set[str]:
        return set(self._records)


class FigureRegistry:
    def __init__(self, figures: list[FigureRecord]) -> None:
        self._figures = self._unique(figures)

    @staticmethod
    def _unique(figures: list[FigureRecord]) -> dict[str, FigureRecord]:
        result: dict[str, FigureRecord] = {}
        for figure in figures:
            if figure.figure_id in result:
                raise ValueError("duplicate figure ID")
            result[figure.figure_id] = figure
        return result

    def verify(
        self,
        figure_id: str,
        *,
        data_bytes: bytes,
        code_bytes: bytes,
        image_bytes: bytes,
    ) -> list[str]:
        figure = self._figures.get(figure_id)
        if figure is None:
            return ["MISSING_FIGURE"]
        errors: list[str] = []
        if sha256_bytes(data_bytes) != figure.data_hash:
            errors.append("DOCUMENT_INTEGRITY_ERROR:FIGURE_DATA")
        if sha256_bytes(code_bytes) != figure.code_hash:
            errors.append("DOCUMENT_INTEGRITY_ERROR:FIGURE_CODE")
        if sha256_bytes(image_bytes) != figure.image_hash:
            errors.append("DOCUMENT_INTEGRITY_ERROR:FIGURE_IMAGE")
        if canonical_json_bytes(figure.data_payload) != data_bytes:
            errors.append("DOCUMENT_INTEGRITY_ERROR:FIGURE_PAYLOAD")
        if figure.generation_status is not RegistryGenerationStatus.VERIFIED:
            errors.append("UNVERIFIED_EVIDENCE:FIGURE")
        return errors

    def records(self) -> list[FigureRecord]:
        return sorted(self._figures.values(), key=lambda item: item.figure_id)

    def identifiers(self) -> set[str]:
        return set(self._figures)


class TableRegistry:
    def __init__(self, tables: list[TableRecord]) -> None:
        self._tables: dict[str, TableRecord] = {}
        for table in tables:
            if table.table_id in self._tables:
                raise ValueError("duplicate table ID")
            self._tables[table.table_id] = table

    def verify(self, table_id: str, *, data_bytes: bytes) -> list[str]:
        table = self._tables.get(table_id)
        if table is None:
            return ["MISSING_TABLE"]
        expected = canonical_json_bytes({"columns": table.columns, "rows": table.rows})
        errors: list[str] = []
        if sha256_bytes(data_bytes) != table.data_hash or data_bytes != expected:
            errors.append("DOCUMENT_INTEGRITY_ERROR:TABLE_DATA")
        if table.generation_status is not RegistryGenerationStatus.VERIFIED:
            errors.append("UNVERIFIED_EVIDENCE:TABLE")
        return errors

    def records(self) -> list[TableRecord]:
        return sorted(self._tables.values(), key=lambda item: item.table_id)

    def identifiers(self) -> set[str]:
        return set(self._tables)
