"""Causal, pointwise binary features from an immutable grouped CSV.

The caller must disclose only the yielded feature to a predictor, then reveal
the associated outcome before requesting the next feature. Materializing all
features for a held-out group at once would disclose future outcomes indirectly.
"""

import csv
import hashlib
import io
from collections import defaultdict, deque
from collections.abc import Iterator
from dataclasses import dataclass


@dataclass(frozen=True)
class CausalBinarySpec:
    source_sha256: str
    group_column: str
    condition_column: str
    outcome_column: str
    positive_value: str
    negative_value: str
    history_window: int = 8

    def __post_init__(self) -> None:
        if len(self.source_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.source_sha256
        ):
            raise ValueError("causal source digest must be lowercase SHA-256")
        if (
            not all((self.group_column, self.condition_column, self.outcome_column))
            or len({self.group_column, self.condition_column, self.outcome_column}) != 3
        ):
            raise ValueError("causal group, condition, and outcome columns must be distinct")
        if self.positive_value == self.negative_value or not 1 <= self.history_window <= 1000:
            raise ValueError("causal outcome codes and history window are invalid")


@dataclass(frozen=True)
class CausalFeature:
    group: str
    condition: str
    point_index: int
    prior_group_count: int
    prior_group_positive: int
    prior_condition_count: int
    prior_condition_positive: int
    recent_count: int
    recent_positive: int


@dataclass
class _History:
    count: int
    positive: int
    by_condition: dict[str, tuple[int, int]]
    recent: deque[int]


def iter_causal_binary_points(
    source: bytes, spec: CausalBinarySpec
) -> Iterator[tuple[CausalFeature, float]]:
    """Yield pre-outcome features; update history only after each yield resumes."""
    if len(source) > 16 * 1024 * 1024 or hashlib.sha256(source).hexdigest() != spec.source_sha256:
        raise ValueError("CAUSAL_CSV_SIZE_OR_HASH_MISMATCH")
    try:
        reader = csv.DictReader(io.StringIO(source.decode("utf-8-sig"), newline=""), strict=True)
        names = reader.fieldnames
        if (
            names is None
            or len(names) != len(set(names))
            or not {
                spec.group_column,
                spec.condition_column,
                spec.outcome_column,
            }
            <= set(names)
        ):
            raise ValueError("CAUSAL_CSV_COLUMNS_INVALID")
        histories: dict[str, _History] = defaultdict(
            lambda: _History(0, 0, {}, deque(maxlen=spec.history_window))
        )
        total = 0
        for row in reader:
            if None in row or None in row.values():
                raise ValueError("CAUSAL_CSV_ROW_MALFORMED")
            group = row[spec.group_column]
            condition = row[spec.condition_column]
            label = row[spec.outcome_column]
            if (
                not group
                or not condition
                or label
                not in {
                    spec.positive_value,
                    spec.negative_value,
                }
            ):
                raise ValueError("CAUSAL_CSV_VALUE_INVALID")
            total += 1
            if total > 100000:
                raise ValueError("CAUSAL_CSV_TOO_MANY_ROWS")
            history = histories[group]
            condition_count, condition_positive = history.by_condition.get(condition, (0, 0))
            feature = CausalFeature(
                group=group,
                condition=condition,
                point_index=history.count,
                prior_group_count=history.count,
                prior_group_positive=history.positive,
                prior_condition_count=condition_count,
                prior_condition_positive=condition_positive,
                recent_count=len(history.recent),
                recent_positive=sum(history.recent),
            )
            outcome = int(label == spec.positive_value)
            yield feature, float(outcome)
            history.count += 1
            history.positive += outcome
            history.by_condition[condition] = (
                condition_count + 1,
                condition_positive + outcome,
            )
            history.recent.append(outcome)
        if total == 0:
            raise ValueError("CAUSAL_CSV_NO_ROWS")
    except (UnicodeError, csv.Error) as exc:
        raise ValueError("CAUSAL_CSV_PARSE_FAILED") from exc
