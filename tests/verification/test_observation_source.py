import hashlib
from uuid import uuid4

import pytest
from pydantic import ValidationError

from mathmodel_ai.files.storage import LocalFileStore
from mathmodel_ai.schemas.files import FileKind, RegisteredFile
from mathmodel_ai.schemas.independent_verification import CsvObservationSpec, ObservationData
from mathmodel_ai.verification.observation_source import (
    derive_csv_observations,
    verify_csv_observations,
    verify_registered_csv_observations,
)


def _observation(source: bytes) -> ObservationData:
    return ObservationData(
        observations=[1.0, 0.0, 1.0],
        source_csv_sha256=hashlib.sha256(source).hexdigest(),
        source_column="point_victor",
        positive_value="1",
        negative_value="2",
    )


def test_binary_observations_are_recomputed_from_exact_csv_bytes_and_order() -> None:
    source = b"point_victor,server\n1,2\n2,1\n1,1\n"
    observation = _observation(source)

    assert verify_csv_observations(observation, source) == [1.0, 0.0, 1.0]
    spec = CsvObservationSpec(
        source_csv_sha256=observation.source_csv_sha256,
        source_column="point_victor",
        positive_value="1",
        negative_value="2",
    )
    assert derive_csv_observations(spec, source) == [1.0, 0.0, 1.0]
    with pytest.raises(ValueError, match="OBSERVATION_CSV_HASH_MISMATCH"):
        verify_csv_observations(observation, source.replace(b"2,1", b"1,1"))
    with pytest.raises(ValueError, match="OBSERVATION_VALUES_DO_NOT_MATCH_CSV"):
        verify_csv_observations(
            observation.model_copy(update={"observations": [1.0, 1.0, 0.0]}), source
        )


def test_binary_observation_source_rejects_unknown_codes_and_duplicate_columns() -> None:
    unknown = b"point_victor,server\n1,2\n3,1\n1,1\n"
    with pytest.raises(ValueError, match="OBSERVATION_CSV_LABEL_INVALID"):
        verify_csv_observations(_observation(unknown), unknown)
    duplicate = b"point_victor,point_victor\n1,1\n2,2\n1,1\n"
    with pytest.raises(ValueError, match="OBSERVATION_CSV_COLUMN_MISSING_OR_DUPLICATE"):
        verify_csv_observations(_observation(duplicate), duplicate)
    empty = b"point_victor,server\n"
    with pytest.raises(ValueError, match="OBSERVATION_CSV_NO_ROWS"):
        derive_csv_observations(
            CsvObservationSpec(
                source_csv_sha256=hashlib.sha256(empty).hexdigest(),
                source_column="point_victor",
                positive_value="1",
                negative_value="2",
            ),
            empty,
        )


def test_binary_observation_provenance_is_complete_and_legacy_json_stays_valid() -> None:
    with pytest.raises(ValidationError, match="provenance must be complete"):
        ObservationData(observations=[1.0], source_column="point_victor")
    with pytest.raises(ValidationError, match="labels must differ"):
        ObservationData(
            observations=[1.0],
            source_csv_sha256="a" * 64,
            source_column="point_victor",
            positive_value="1",
            negative_value="1",
        )
    plain = ObservationData.model_validate_json('{"observations":[1.0,0.0]}')
    assert plain.observations == [1.0, 0.0]
    with pytest.raises(ValueError, match="OBSERVATION_CSV_PROVENANCE_MISSING"):
        verify_csv_observations(plain, b"point_victor\n1\n2\n")


def test_registered_observation_source_requires_unique_size_bound_file(tmp_path) -> None:
    source = b"point_victor,server\n1,2\n2,1\n1,1\n"
    project_id = uuid4()
    store = LocalFileStore(tmp_path)
    stored = store.store_artifact(
        source, project_id=project_id, artifact_id=uuid4(), filename="source.csv"
    )
    record = RegisteredFile(
        project_id=project_id,
        problem_id=uuid4(),
        original_name="source.csv",
        safe_name="source.csv",
        extension=".csv",
        kind=FileKind.CSV,
        detected_mime_type="text/csv",
        size_bytes=stored.size_bytes,
        sha256=stored.sha256,
        storage_key=stored.storage_key,
    )
    observation = _observation(source)
    assert verify_registered_csv_observations(observation, [record], store) == [1.0, 0.0, 1.0]
    assert verify_registered_csv_observations(
        CsvObservationSpec(
            source_csv_sha256=record.sha256,
            source_column="point_victor",
            positive_value="1",
            negative_value="2",
        ),
        [record],
        store,
    ) == [1.0, 0.0, 1.0]
    with pytest.raises(ValueError, match="OBSERVATION_CSV_SOURCE_NOT_UNIQUE"):
        verify_registered_csv_observations(observation, [], store)
    with pytest.raises(ValueError, match="OBSERVATION_CSV_SOURCE_NOT_UNIQUE"):
        verify_registered_csv_observations(
            observation, [record, record.model_copy(update={"file_id": uuid4()})], store
        )
    with pytest.raises(ValueError, match="OBSERVATION_CSV_SIZE_MISMATCH"):
        verify_registered_csv_observations(
            observation, [record.model_copy(update={"size_bytes": 1})], store
        )
