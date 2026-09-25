"""Immutable whole-group split of official CSV for a blind solve.

The official bytes stay with the trusted benchmark host. Agents and their
solver sandbox receive only the derived training CSV; labels from held-out
groups are reserved for the separate online predictor protocol.
"""

import csv
import hashlib
import io
import json
from dataclasses import asdict, dataclass

from mathmodel_ai.verification.causal_binary import CausalBinarySpec, iter_causal_binary_points
from mathmodel_ai.verification.causal_holdout import heldout_groups


@dataclass(frozen=True)
class CausalInputSplit:
    source_sha256: str
    training_sha256: str
    policy_sha256: str
    training_csv: bytes
    heldout_groups: tuple[str, ...]
    training_groups: tuple[str, ...]
    training_rows: int
    heldout_rows: int


def split_causal_csv(
    source: bytes,
    spec: CausalBinarySpec,
    *,
    fraction: float,
    salt: str,
) -> CausalInputSplit:
    """Validate the full source, then materialize only non-held-out rows."""
    group_counts: dict[str, int] = {}
    for feature, _ in iter_causal_binary_points(source, spec):
        group_counts[feature.group] = group_counts.get(feature.group, 0) + 1
    groups = set(group_counts)
    heldout = heldout_groups(groups, fraction=fraction, salt=salt)
    heldout_set = set(heldout)
    expected_training = sum(
        count for group, count in group_counts.items() if group not in heldout_set
    )
    expected_heldout = sum(count for group, count in group_counts.items() if group in heldout_set)
    reader = csv.DictReader(io.StringIO(source.decode("utf-8-sig"), newline=""), strict=True)
    if reader.fieldnames is None:
        raise ValueError("CAUSAL_CSV_COLUMNS_INVALID")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=reader.fieldnames, lineterminator="\n")
    writer.writeheader()
    training_rows = 0
    for row in reader:
        if row[spec.group_column] not in heldout_set:
            writer.writerow(row)
            training_rows += 1
    if training_rows != expected_training or expected_heldout <= 0:
        raise ValueError("CAUSAL_SPLIT_ROW_MISMATCH")
    training_csv = output.getvalue().encode("utf-8")
    policy_bytes = json.dumps(
        {
            "protocol": "whole-group-causal-v1",
            "spec": asdict(spec),
            "fraction": fraction,
            "salt": salt,
        },
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return CausalInputSplit(
        source_sha256=spec.source_sha256,
        training_sha256=hashlib.sha256(training_csv).hexdigest(),
        policy_sha256=hashlib.sha256(policy_bytes).hexdigest(),
        training_csv=training_csv,
        heldout_groups=heldout,
        training_groups=tuple(sorted(groups - heldout_set)),
        training_rows=training_rows,
        heldout_rows=expected_heldout,
    )
