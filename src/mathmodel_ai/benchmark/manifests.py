from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path

from mathmodel_ai.benchmark.causal_inputs import CausalInputSplit, split_causal_csv
from mathmodel_ai.paper.hashing import sha256_bytes, sha256_json
from mathmodel_ai.schemas.benchmark import (
    BenchmarkCaseManifest,
    BenchmarkPhase,
    BenchmarkResource,
    BenchmarkResourceRole,
    CausalHoldoutPolicy,
    ModelingCategory,
    benchmark_manifest_digest,
)
from mathmodel_ai.verification.causal_binary import CausalBinarySpec


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
    visible_artifacts: tuple[BlindSolveArtifact, ...] | None = None
    causal_split: CausalInputSplit | None = None
    causal_policy: CausalHoldoutPolicy | None = None

    @property
    def solver_artifacts(self) -> tuple[BlindSolveArtifact, ...]:
        return self.visible_artifacts if self.visible_artifacts is not None else self.artifacts


class BenchmarkManifestRegistry:
    """Load versioned case manifests while keeping evaluation material out of solve input."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._manifests: dict[str, BenchmarkCaseManifest] = {}
        self._manifest_paths: dict[str, Path] = {}
        if root.exists():
            for path in sorted(root.glob("case-*/manifest.json")):
                manifest = BenchmarkCaseManifest.model_validate_json(path.read_text("utf-8"))
                if manifest.benchmark_id in self._manifests:
                    raise ValueError(f"duplicate benchmark id {manifest.benchmark_id}")
                self._manifests[manifest.benchmark_id] = manifest
                self._manifest_paths[manifest.benchmark_id] = path

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
        policy = self._causal_policy(benchmark_id)
        if (
            policy is not None
            and manifest.modeling_category is not ModelingCategory.DATA_PREDICTION
        ):
            raise ValueError("causal holdout requires a data-prediction benchmark")
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
        visible_artifacts: tuple[BlindSolveArtifact, ...] | None = None
        split: CausalInputSplit | None = None
        if policy is not None:
            if policy.problem_sha256 is not None:
                problems = [
                    item
                    for item in artifacts
                    if item.resource.role is BenchmarkResourceRole.PROBLEM
                ]
                if len(problems) != 1 or sha256_bytes(problems[0].content) != policy.problem_sha256:
                    raise ValueError("CAUSAL_SCIENCE_PROBLEM_SOURCE_MISMATCH")
            target = next(
                (item for item in artifacts if item.resource.resource_id == policy.resource_id),
                None,
            )
            if (
                target is None
                or target.resource.media_type != "text/csv"
                or target.resource.sha256 != policy.source_sha256
            ):
                raise ValueError("CAUSAL_HOLDOUT_RESOURCE_IDENTITY_MISMATCH")
            split = split_causal_csv(
                target.content,
                CausalBinarySpec(
                    source_sha256=policy.source_sha256,
                    group_column=policy.group_column,
                    condition_column=policy.condition_column,
                    outcome_column=policy.outcome_column,
                    positive_value=policy.positive_value,
                    negative_value=policy.negative_value,
                    history_window=policy.history_window,
                ),
                fraction=policy.fraction,
                salt=policy.salt,
            )
            visible_artifacts = tuple(
                BlindSolveArtifact(
                    resource=item.resource,
                    content=split.training_csv if item is target else item.content,
                    trust_classification=item.trust_classification,
                )
                for item in artifacts
            )
            digest_payload = [
                {
                    "resource_id": item.resource.resource_id,
                    "sha256": sha256_bytes(item.content),
                    "size_bytes": len(item.content),
                }
                for item in visible_artifacts
            ]
        digest_input: dict[str, object] = {
            "manifest_digest": benchmark_manifest_digest(manifest),
            "resources": digest_payload,
        }
        if split is not None:
            assert policy is not None
            digest_input["causal_split"] = {
                "source_sha256": split.source_sha256,
                "training_sha256": split.training_sha256,
                "policy_sha256": split.policy_sha256,
                "heldout_groups": split.heldout_groups,
                "training_groups": split.training_groups,
            }
            digest_input["causal_science_policy_sha256"] = sha256_json(policy)
        return BlindSolveBundle(
            manifest=manifest,
            artifacts=tuple(artifacts),
            solve_input_digest=sha256_json(digest_input),
            visible_artifacts=visible_artifacts,
            causal_split=split,
            causal_policy=policy,
        )

    def _causal_policy(self, benchmark_id: str) -> CausalHoldoutPolicy | None:
        path = self._manifest_paths[benchmark_id].with_name("causal-holdout-v1.json")
        if not os.path.lexists(path):
            return None
        info = os.lstat(path)
        reparse_point = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
        if (
            not stat.S_ISREG(info.st_mode)
            or stat.S_ISLNK(info.st_mode)
            or int(getattr(info, "st_file_attributes", 0)) & reparse_point
            or info.st_size > 10 * 1024
        ):
            raise ValueError("causal holdout policy is not a bounded regular file")
        return CausalHoldoutPolicy.model_validate_json(path.read_bytes())


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
