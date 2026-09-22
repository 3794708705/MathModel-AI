from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import httpx
import polars as pl
from pypdf import PdfReader

from mathmodel_ai.benchmark.manifests import BenchmarkManifestRegistry, BlindSolveBundle
from mathmodel_ai.paper.hashing import sha256_bytes
from mathmodel_ai.schemas.benchmark import (
    BenchmarkCaseManifest,
    BenchmarkPhase,
    BenchmarkResource,
    ExpectedStructuralFact,
)


@dataclass(frozen=True)
class StructuralFactCheck:
    fact: ExpectedStructuralFact
    actual: str | int | float | bool
    passed: bool


class BenchmarkSourceError(ValueError):
    """An official source could not be fetched or verified exactly."""


class BenchmarkResourceCache:
    """Fetch hash-pinned solve inputs without credentials or redirects."""

    def __init__(
        self,
        root: Path,
        *,
        client: httpx.Client | None = None,
        allowed_hosts: frozenset[str] = frozenset({"www.contest.comap.com"}),
        max_resource_bytes: int = 25 * 1024 * 1024,
        timeout_seconds: float = 30,
    ) -> None:
        self._root = root
        self._allowed_hosts = allowed_hosts
        self._max_resource_bytes = max_resource_bytes
        self._client = client or httpx.Client(
            timeout=timeout_seconds,
            follow_redirects=False,
        )
        self._owns_client = client is None

    def materialize(
        self,
        manifest: BenchmarkCaseManifest,
        registry: BenchmarkManifestRegistry,
    ) -> BlindSolveBundle:
        case_root = self._case_root(manifest.benchmark_id)
        case_root.mkdir(parents=True, exist_ok=True)
        for resource in manifest.resources:
            if resource.phase is BenchmarkPhase.SOLVE:
                self._materialize_resource(resource, case_root)
        return registry.load_blind_solve_bundle(manifest.benchmark_id, case_root)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _case_root(self, benchmark_id: str) -> Path:
        self._root.mkdir(parents=True, exist_ok=True)
        root = self._root.resolve(strict=True)
        candidate = root / benchmark_id
        if candidate.parent != root:
            raise BenchmarkSourceError("benchmark cache path escaped its configured root")
        return candidate

    def _materialize_resource(self, resource: BenchmarkResource, case_root: Path) -> None:
        destination = case_root / resource.local_filename
        if destination.exists():
            info = os.lstat(destination)
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                raise BenchmarkSourceError("cached benchmark input is not a regular file")
            if hasattr(info, "st_file_attributes") and info.st_file_attributes & 0x400:
                raise BenchmarkSourceError("cached benchmark input is a reparse point")
            data = destination.read_bytes()
            if self._matches(resource, data):
                return
            raise BenchmarkSourceError(
                f"cached resource failed immutable identity check: {resource.resource_id}"
            )
        host = (urlparse(resource.source_url).hostname or "").casefold()
        if host not in self._allowed_hosts:
            raise BenchmarkSourceError(f"benchmark source host is not allowlisted: {host}")
        response = self._client.get(resource.source_url)
        if response.status_code != 200:
            raise BenchmarkSourceError(
                f"official source returned HTTP {response.status_code}: {resource.resource_id}"
            )
        response_host = (response.url.host or "").casefold()
        if response_host != host:
            raise BenchmarkSourceError("benchmark source changed host")
        declared_length = response.headers.get("content-length")
        if declared_length is not None:
            try:
                length = int(declared_length)
            except ValueError as exc:
                raise BenchmarkSourceError(
                    "official source returned invalid content-length"
                ) from exc
            if length > self._max_resource_bytes:
                raise BenchmarkSourceError("benchmark resource exceeds configured byte limit")
        content = bytearray()
        for chunk in response.iter_bytes():
            content.extend(chunk)
            if len(content) > self._max_resource_bytes:
                raise BenchmarkSourceError("benchmark resource exceeded configured byte limit")
        data = bytes(content)
        if not self._matches(resource, data):
            raise BenchmarkSourceError(
                f"downloaded resource failed immutable identity check: {resource.resource_id}"
            )
        temporary = destination.with_suffix(f"{destination.suffix}.download")
        temporary.write_bytes(data)
        temporary.replace(destination)

    @staticmethod
    def _matches(resource: BenchmarkResource, data: bytes) -> bool:
        return (
            resource.sha256 is not None
            and sha256_bytes(data) == resource.sha256
            and (resource.expected_size_bytes is None or len(data) == resource.expected_size_bytes)
        )


class BenchmarkStructureInspector:
    """Recalculate manifest structural facts from raw bytes, independently of agents."""

    def inspect(self, bundle: BlindSolveBundle) -> list[StructuralFactCheck]:
        by_id = {item.resource.resource_id: item for item in bundle.artifacts}
        checks: list[StructuralFactCheck] = []
        for fact in bundle.manifest.expected_structural_facts:
            artifact = by_id.get(fact.resource_id)
            if artifact is None:
                raise BenchmarkSourceError(
                    f"structural fact references unavailable solve input: {fact.resource_id}"
                )
            actual = self._value(artifact.resource, artifact.content, fact.field)
            checks.append(
                StructuralFactCheck(
                    fact=fact,
                    actual=actual,
                    passed=self._equal(actual, fact.expected),
                )
            )
        return checks

    @staticmethod
    def _value(
        resource: BenchmarkResource,
        content: bytes,
        field: str,
    ) -> str | int | float | bool:
        if field == "pdf_page_count" and resource.media_type == "application/pdf":
            return len(PdfReader(BytesIO(content)).pages)
        if resource.media_type == "text/csv":
            frame = pl.read_csv(BytesIO(content))
            if field == "row_count":
                return frame.height
            if field == "column_count":
                return frame.width
            if field == "null_cell_count":
                return sum(int(value) for value in frame.null_count().row(0))
        raise BenchmarkSourceError(
            f"unsupported structural fact {field!r} for {resource.media_type!r}"
        )

    @staticmethod
    def _equal(actual: str | int | float | bool, expected: str | int | float | bool) -> bool:
        if isinstance(actual, float) or isinstance(expected, float):
            return abs(float(actual) - float(expected)) <= 1e-9
        return actual == expected
