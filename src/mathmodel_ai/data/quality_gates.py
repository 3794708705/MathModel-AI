import hashlib

from mathmodel_ai.core.errors import StorageError
from mathmodel_ai.files.pipeline import ProcessedFile
from mathmodel_ai.files.storage import FileStore
from mathmodel_ai.schemas.data import DataProfile, DataUnderstanding
from mathmodel_ai.schemas.execution import ExecutionRecord, ExecutionStatus
from mathmodel_ai.schemas.files import ArtifactKind, ArtifactRecord, FileStatus
from mathmodel_ai.schemas.quality import QualityGateResult, QualityGateStatus


def files_quality_gate(processed: ProcessedFile, store: FileStore) -> QualityGateResult:
    file = processed.parsed_file.file
    checks = {
        "file_parsed": file.status is FileStatus.PARSED,
        "raw_exists": store.exists(file.storage_key),
        "raw_hash_matches": _hash_matches(store, file.storage_key, file.sha256),
        "artifacts_exist": all(store.exists(item.storage_key) for item in processed.artifacts),
        "artifact_hashes_match": all(
            _hash_matches(store, item.storage_key, item.sha256) for item in processed.artifacts
        ),
    }
    errors = [name for name, passed in checks.items() if not passed]
    if file.error:
        errors.append(file.error)
    return QualityGateResult(
        gate="FILES",
        status=QualityGateStatus.PASS if not errors else QualityGateStatus.RETRY,
        checks=checks,
        errors=errors,
        warnings=processed.parsed_file.warnings,
    )


def data_quality_gate(
    profiles: list[DataProfile],
    understanding: DataUnderstanding | None = None,
    *,
    media_present: bool = False,
) -> QualityGateResult:
    profile_ids = {profile.dataset_id for profile in profiles}
    understanding_ids = (
        {item.dataset_id for item in understanding.datasets} if understanding is not None else set()
    )
    references_valid = understanding is None or not data_column_reference_errors(
        profiles, understanding
    )
    checks = {
        "data_evidence_present": bool(profiles) or media_present,
        "profiles_deterministic": all(profile.deterministic for profile in profiles),
        "schemas_present": all(
            profile.column_count == len(profile.columns) for profile in profiles
        ),
        "agent_covers_profiles": understanding is None or understanding_ids == profile_ids,
        "agent_references_known_columns": references_valid,
    }
    errors = [name for name, passed in checks.items() if not passed]
    warnings = [
        issue.message
        for profile in profiles
        for issue in profile.quality_issues
        if issue.severity.value in {"WARNING", "ERROR"}
    ]
    return QualityGateResult(
        gate="DATA",
        status=QualityGateStatus.PASS if not errors else QualityGateStatus.RETRY,
        checks=checks,
        errors=errors,
        warnings=warnings,
    )


def data_column_reference_errors(
    profiles: list[DataProfile],
    understanding: DataUnderstanding,
) -> list[str]:
    """Describe invented column bindings without relaxing the DATA gate."""
    known = {
        profile.dataset_id: {column.name for column in profile.columns} for profile in profiles
    }
    errors: list[str] = []
    for dataset in understanding.datasets:
        columns = known.get(dataset.dataset_id, set())
        referenced = {
            *dataset.potential_features,
            *dataset.potential_targets,
            *(item.column for item in dataset.key_columns),
        }
        unknown = sorted(referenced - columns)
        if unknown:
            errors.append(
                f"dataset {dataset.dataset_id} references unknown columns: " + ", ".join(unknown)
            )
    return errors


def execution_quality_gate(
    record: ExecutionRecord,
    store: FileStore,
    tracked_artifacts: list[ArtifactRecord],
) -> QualityGateResult:
    by_id = {item.artifact_id: item for item in tracked_artifacts}
    code = by_id.get(record.code_artifact_id)
    checks = {
        "execution_succeeded": record.status is ExecutionStatus.SUCCEEDED,
        "zero_exit_code": record.exit_code == 0,
        "network_disabled": record.network_disabled,
        "non_root": record.non_root,
        "read_only_root": record.read_only_root,
        "not_mock": not record.is_mock,
        "code_artifact_tracked": (
            code is not None
            and code.kind is ArtifactKind.GENERATED_CODE
            and code.sha256 == record.code_hash
            and _hash_matches(store, code.storage_key, code.sha256)
        ),
        "artifacts_exist": all(store.exists(item.storage_key) for item in record.artifacts),
        "artifact_hashes_match": all(
            _hash_matches(store, item.storage_key, item.sha256) for item in record.artifacts
        ),
    }
    errors = [name for name, passed in checks.items() if not passed]
    if record.error:
        errors.append(record.error)
    return QualityGateResult(
        gate="EXECUTION",
        status=QualityGateStatus.PASS if not errors else QualityGateStatus.RETRY,
        checks=checks,
        errors=errors,
        warnings=[],
    )


def _hash_matches(store: FileStore, storage_key: str, expected: str) -> bool:
    try:
        path = store.resolve(storage_key)
        digest = hashlib.sha256()
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest() == expected
    except (OSError, StorageError):
        return False
