"""Independent, exact-order binary observations from a hash-bound CSV source."""

import csv
import hashlib
import io
from collections.abc import Sequence

from mathmodel_ai.files.storage import FileStore
from mathmodel_ai.schemas.files import RegisteredFile
from mathmodel_ai.schemas.independent_verification import CsvObservationSpec, ObservationData


def derive_csv_observations(spec: CsvObservationSpec, source: bytes) -> list[float]:
    """Parse an exact, bounded binary target without trusting a solver output."""
    if hashlib.sha256(source).hexdigest() != spec.source_csv_sha256:
        raise ValueError("OBSERVATION_CSV_HASH_MISMATCH")
    try:
        reader = csv.DictReader(io.StringIO(source.decode("utf-8-sig"), newline=""), strict=True)
        names = reader.fieldnames
        if names is None or len(names) != len(set(names)) or spec.source_column not in names:
            raise ValueError("OBSERVATION_CSV_COLUMN_MISSING_OR_DUPLICATE")
        expected: list[float] = []
        for row in reader:
            if None in row or None in row.values():
                raise ValueError("OBSERVATION_CSV_ROW_MALFORMED")
            value = row[spec.source_column]
            if value not in {spec.positive_value, spec.negative_value}:
                raise ValueError("OBSERVATION_CSV_LABEL_INVALID")
            expected.append(float(value == spec.positive_value))
            if len(expected) > 100000:
                raise ValueError("OBSERVATION_CSV_TOO_MANY_ROWS")
    except (UnicodeError, csv.Error) as exc:
        raise ValueError("OBSERVATION_CSV_PARSE_FAILED") from exc
    if not expected:
        raise ValueError("OBSERVATION_CSV_NO_ROWS")
    return expected


def verify_csv_observations(observation: ObservationData, source: bytes) -> list[float]:
    if (
        observation.source_csv_sha256 is None
        or observation.source_column is None
        or observation.positive_value is None
        or observation.negative_value is None
    ):
        raise ValueError("OBSERVATION_CSV_PROVENANCE_MISSING")
    expected = derive_csv_observations(
        CsvObservationSpec(
            source_csv_sha256=observation.source_csv_sha256,
            source_column=observation.source_column,
            positive_value=observation.positive_value,
            negative_value=observation.negative_value,
        ),
        source,
    )
    if expected != observation.observations:
        raise ValueError("OBSERVATION_VALUES_DO_NOT_MATCH_CSV")
    return expected


def verify_registered_csv_observations(
    observation: ObservationData | CsvObservationSpec,
    files: Sequence[RegisteredFile],
    store: FileStore,
) -> list[float]:
    sources = [
        item
        for item in files
        if item.sha256 == observation.source_csv_sha256 and item.extension == ".csv"
    ]
    if len(sources) != 1:
        raise ValueError("OBSERVATION_CSV_SOURCE_NOT_UNIQUE")
    source = sources[0]
    source_bytes = store.read_bytes(source.storage_key, max_bytes=16 * 1024 * 1024)
    if len(source_bytes) != source.size_bytes:
        raise ValueError("OBSERVATION_CSV_SIZE_MISMATCH")
    return (
        verify_csv_observations(observation, source_bytes)
        if isinstance(observation, ObservationData)
        else derive_csv_observations(observation, source_bytes)
    )
