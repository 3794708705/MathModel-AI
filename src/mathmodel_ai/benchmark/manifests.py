from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path

from mathmodel_ai.paper.hashing import sha256_bytes, sha256_json
from mathmodel_ai.schemas.benchmark import (
    BenchmarkCaseManifest,
    BenchmarkPhase,
    BenchmarkResource,
    benchmark_manifest_digest,
)


@dataclass(frozen=True)
class BlindSolveArtifact:
    resource: BenchmarkResource
    content: bytes
    trust_classification: str = "UNTRUSTED_DATA"


@dataclass(frozen=True)
class BlindSolveBundle:
    manifest: BenchmarkCaseManifest
    artifacts: tuple[BlindSolveArtifact, ...]
    solve_input_digest: str


class BenchmarkManifestRegistry:
    """Load versioned case manifests while keeping evaluation material out of solve input."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._manifests: dict[str, BenchmarkCaseManifest] = {}
        if root.exists():
            for path in sorted(root.glob("case-*/manifest.json")):
                manifest = BenchmarkCaseManifest.model_validate_json(path.read_text("utf-8"))
                if manifest.benchmark_id in self._manifests:
                    raise ValueError(f"duplicate benchmark id {manifest.benchmark_id}")
                self._manifests[manifest.benchmark_id] = manifest

    def get(self, benchmark_id: str) -> BenchmarkCaseManifest:
        try:
            manifest = self._manifests[benchmark_id]
        except KeyError as exc:
            raise KeyError(f"benchmark case {benchmark_id!r} is not registered") from exc
        return manifest.model_copy(deep=True)

    def list(self) -> list[BenchmarkCaseManifest]:
        return [self._manifests[key].model_copy(deep=True) for key in sorted(self._manifests)]

    def digest(self, benchmark_id: str) -> str:
        return benchmark_manifest_digest(self.get(benchmark_id))

    def load_blind_solve_bundle(self, benchmark_id: str, cache_root: Path) -> BlindSolveBundle:
        manifest = self.get(benchmark_id)
        solve_resources = [
            item for item in manifest.resources if item.phase is BenchmarkPhase.SOLVE
        ]
        artifacts: list[BlindSolveArtifact] = []
        digest_payload: list[dict[str, str | int]] = []
        for resource in sorted(solve_resources, key=lambda item: item.resource_id):
            path = cache_root / resource.local_filename
            data = _verified_resource_bytes(path, cache_root, resource)
            artifacts.append(BlindSolveArtifact(resource=resource, content=data))
            digest_payload.append(
                {
                    "resource_id": resource.resource_id,
                    "sha256": sha256_bytes(data),
                    "size_bytes": len(data),
                }
            )
        return BlindSolveBundle(
            manifest=manifest,
            artifacts=tuple(artifacts),
            solve_input_digest=sha256_json(
                {
                    "manifest_digest": benchmark_manifest_digest(manifest),
                    "resources": digest_payload,
                }
            ),
        )


def _verified_resource_bytes(
    path: Path,
    cache_root: Path,
    resource: BenchmarkResource,
) -> bytes:
    resolved_root = cache_root.resolve(strict=True)
    resolved = path.resolve(strict=True)
    if resolved.parent != resolved_root:
        raise ValueError("benchmark resource escaped its case cache root")
    info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError("benchmark resources must be regular non-symlink files")
    file_attributes = int(getattr(info, "st_file_attributes", 0))
    reparse_point = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    if file_attributes & reparse_point:
        raise ValueError("Windows reparse-point benchmark resources are forbidden")
    data = path.read_bytes()
    if resource.expected_size_bytes is not None and len(data) != resource.expected_size_bytes:
        raise ValueError(f"resource size mismatch for {resource.resource_id}")
    actual_hash = sha256_bytes(data)
    if resource.sha256 is None or actual_hash != resource.sha256:
        raise ValueError(f"resource hash mismatch for {resource.resource_id}")
    return data


def manifest_json(manifest: BenchmarkCaseManifest) -> str:
    return json.dumps(
        manifest.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, indent=2
    )
