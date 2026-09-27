from __future__ import annotations

import inspect
import io
import math
from typing import Any
from uuid import UUID, uuid4

from PIL import Image, ImageDraw

from mathmodel_ai.files.storage import FileStore
from mathmodel_ai.mathematical.registry import SymbolRegistry
from mathmodel_ai.paper.hashing import canonical_json_bytes, sha256_bytes
from mathmodel_ai.schemas.mathematical import MathematicalModel
from mathmodel_ai.schemas.paper import (
    EvidenceRecord,
    FigureRecord,
    FigureType,
    PaperArtifact,
    PaperArtifactKind,
    RegistryGenerationStatus,
    TableRecord,
)


class ArtifactGenerationError(ValueError):
    pass


class FigureAgent:
    """Deterministic system-code figure generator with three-part provenance."""

    def __init__(self, store: FileStore) -> None:
        self._file_store = store

    def generate(
        self,
        *,
        project_id: UUID,
        problem_id: UUID,
        paper_id: UUID,
        paper_version: int,
        figure_id: str,
        title: str,
        caption: str,
        figure_type: FigureType,
        data_payload: dict[str, Any],
        source_evidence_refs: list[UUID],
        evidence: list[EvidenceRecord],
    ) -> tuple[FigureRecord, list[PaperArtifact]]:
        self._require_verified_sources(source_evidence_refs, evidence)
        sources = self._source_binding(source_evidence_refs, evidence)
        data_bytes = canonical_json_bytes(data_payload)
        code_bytes = inspect.getsource(self._render_png).encode("utf-8")
        image_bytes = self._render_png(data_payload)
        data_artifact = self._persist(
            project_id,
            problem_id,
            paper_id,
            paper_version,
            PaperArtifactKind.FIGURE_DATA,
            f"{figure_id}.json",
            "application/json",
            data_bytes,
        )
        code_artifact = self._persist(
            project_id,
            problem_id,
            paper_id,
            paper_version,
            PaperArtifactKind.FIGURE_CODE,
            f"{figure_id}.py",
            "text/x-python",
            code_bytes,
        )
        image_artifact = self._persist(
            project_id,
            problem_id,
            paper_id,
            paper_version,
            PaperArtifactKind.FIGURE_IMAGE,
            f"{figure_id}.png",
            "image/png",
            image_bytes,
        )
        record = FigureRecord(
            figure_id=figure_id,
            project_id=project_id,
            paper_id=paper_id,
            paper_version=paper_version,
            title=title,
            caption=caption,
            figure_type=figure_type,
            source_evidence_refs=source_evidence_refs,
            source_evidence_types=[item.evidence_type for item in sources],
            source_binding={
                "evidence": [
                    {
                        "evidence_id": str(item.evidence_id),
                        "evidence_type": item.evidence_type.value,
                        "source_id": item.source_id,
                        "source_hash": item.provenance.source_hash,
                    }
                    for item in sources
                ],
                "metric": data_payload.get("metric"),
                "parameter": data_payload.get("parameter"),
                "experiment_ids": data_payload.get("experiment_ids", []),
            },
            data_payload=data_payload,
            data_hash=sha256_bytes(data_bytes),
            code_hash=sha256_bytes(code_bytes),
            image_hash=sha256_bytes(image_bytes),
            data_artifact_id=data_artifact.artifact_id,
            code_artifact_id=code_artifact.artifact_id,
            image_artifact_id=image_artifact.artifact_id,
            generation_status=RegistryGenerationStatus.VERIFIED,
        )
        return record, [data_artifact, code_artifact, image_artifact]

    @staticmethod
    def _render_png(data_payload: dict[str, Any]) -> bytes:
        raw_x = data_payload.get("x")
        raw_y = data_payload.get("y")
        if not isinstance(raw_x, list) or not isinstance(raw_y, list) or len(raw_x) != len(raw_y):
            raise ArtifactGenerationError("figure payload requires equal-length x and y arrays")
        if len(raw_x) < 2:
            raise ArtifactGenerationError("figure requires at least two observations")
        if any(
            isinstance(item, bool) or not isinstance(item, int | float) for item in raw_x + raw_y
        ):
            raise ArtifactGenerationError("figure values must be numeric")
        x = [float(item) for item in raw_x]
        y = [float(item) for item in raw_y]
        if any(not math.isfinite(item) for item in x + y):
            raise ArtifactGenerationError("figure values must be finite")
        width, height = 900, 540
        left, top, right, bottom = 110, 48, 865, 445
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((left, top, right, bottom), outline="black")
        min_x, max_x = min(x), max(x)
        min_y, max_y = min(y), max(y)
        x_span = max_x - min_x or 1.0
        y_span = max_y - min_y
        if y_span == 0:
            y_span = max(abs(min_y) * 0.1, 1e-9)
        y_min = min_y - 0.05 * y_span
        y_max = max_y + 0.05 * y_span
        points = [
            (
                left + (item_x - min_x) / x_span * (right - left),
                bottom - (item_y - y_min) / (y_max - y_min) * (bottom - top),
            )
            for item_x, item_y in zip(x, y, strict=True)
        ]
        for index in range(5):
            value = y_min + (y_max - y_min) * index / 4
            position = bottom - (bottom - top) * index / 4
            draw.line((left - 5, position, left, position), fill="black", width=2)
            draw.text((8, position - 6), f"{value:.5g}", fill="black")
        tick_step = max(1, math.ceil(len(x) / 15))
        for index, (item_x, _) in enumerate(points):
            if index % tick_step == 0 or index == len(x) - 1:
                draw.line((item_x, bottom, item_x, bottom + 5), fill="black", width=2)
                draw.text((item_x - 7, bottom + 10), f"{x[index]:g}", fill="black")
        x_label = str(data_payload.get("x_label") or "x")
        y_label = str(data_payload.get("y_label") or "value")
        draw.text((left, height - 47), x_label[:100], fill="black")
        draw.text((left, 15), y_label[:100], fill="black")
        draw.line(points, fill="#1f77b4", width=4)
        for point in points:
            draw.ellipse(
                (point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5),
                fill="#d62728",
            )
        output = io.BytesIO()
        image.save(output, format="PNG", optimize=False)
        return output.getvalue()

    @staticmethod
    def _require_verified_sources(source_refs: list[UUID], evidence: list[EvidenceRecord]) -> None:
        by_id = {item.evidence_id: item for item in evidence}
        if not source_refs or any(
            reference not in by_id or not by_id[reference].verified for reference in source_refs
        ):
            raise ArtifactGenerationError("asset source evidence must exist and be verified")

    @staticmethod
    def _source_binding(
        source_refs: list[UUID], evidence: list[EvidenceRecord]
    ) -> list[EvidenceRecord]:
        by_id = {item.evidence_id: item for item in evidence}
        return [by_id[reference] for reference in source_refs]

    def _persist(
        self,
        project_id: UUID,
        problem_id: UUID,
        paper_id: UUID,
        paper_version: int,
        kind: PaperArtifactKind,
        name: str,
        mime_type: str,
        data: bytes,
    ) -> PaperArtifact:
        artifact_id = uuid4()
        stored = self._file_store.store_artifact(
            data,
            project_id=project_id,
            artifact_id=artifact_id,
            filename=name,
        )
        return PaperArtifact(
            artifact_id=artifact_id,
            project_id=project_id,
            problem_id=problem_id,
            paper_id=paper_id,
            paper_version=paper_version,
            kind=kind,
            name=name,
            mime_type=mime_type,
            size_bytes=stored.size_bytes,
            sha256=stored.sha256,
            storage_key=stored.storage_key,
        )


class TableAgent:
    def __init__(self, store: FileStore) -> None:
        self._store = store

    def generate(
        self,
        *,
        project_id: UUID,
        problem_id: UUID,
        paper_id: UUID,
        paper_version: int,
        table_id: str,
        title: str,
        caption: str,
        columns: list[str],
        rows: list[list[str | int | float | bool | None]],
        source_evidence_refs: list[UUID],
        evidence: list[EvidenceRecord],
    ) -> tuple[TableRecord, PaperArtifact]:
        FigureAgent._require_verified_sources(source_evidence_refs, evidence)
        sources = FigureAgent._source_binding(source_evidence_refs, evidence)
        data_bytes = canonical_json_bytes({"columns": columns, "rows": rows})
        artifact_id = uuid4()
        stored = self._store.store_artifact(
            data_bytes,
            project_id=project_id,
            artifact_id=artifact_id,
            filename=f"{table_id}.json",
        )
        artifact = PaperArtifact(
            artifact_id=artifact_id,
            project_id=project_id,
            problem_id=problem_id,
            paper_id=paper_id,
            paper_version=paper_version,
            kind=PaperArtifactKind.TABLE_DATA,
            name=f"{table_id}.json",
            mime_type="application/json",
            size_bytes=stored.size_bytes,
            sha256=stored.sha256,
            storage_key=stored.storage_key,
        )
        table = TableRecord(
            table_id=table_id,
            project_id=project_id,
            paper_id=paper_id,
            paper_version=paper_version,
            title=title,
            caption=caption,
            columns=columns,
            rows=rows,
            source_evidence_refs=source_evidence_refs,
            source_evidence_types=[item.evidence_type for item in sources],
            source_binding={
                "evidence": [
                    {
                        "evidence_id": str(item.evidence_id),
                        "evidence_type": item.evidence_type.value,
                        "source_id": item.source_id,
                        "source_hash": item.provenance.source_hash,
                    }
                    for item in sources
                ]
            },
            data_hash=stored.sha256,
            data_artifact_id=artifact_id,
            generation_status=RegistryGenerationStatus.VERIFIED,
        )
        return table, artifact


class SymbolTableBuilder:
    @staticmethod
    def rows(
        model: MathematicalModel,
    ) -> tuple[list[str], list[list[str | int | float | bool | None]]]:
        registry = SymbolRegistry.from_model(model)
        if not registry.report.valid:
            raise ArtifactGenerationError("verified mathematical model has symbol conflicts")
        rows: list[list[str | int | float | bool | None]] = [
            [
                item.symbol,
                item.meaning,
                item.unit.display if item.unit else "1",
                item.kind.value,
                item.domain.value if item.domain else "-",
            ]
            for item in registry.definitions
        ]
        return (
            ["Symbol", "Meaning", "Unit", "Kind", "Domain"],
            rows,
        )
