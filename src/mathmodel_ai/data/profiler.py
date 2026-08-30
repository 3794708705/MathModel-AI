import math
import re
import statistics
from collections import Counter
from datetime import date, datetime
from itertools import combinations
from typing import Any

import polars as pl

from mathmodel_ai.files.parsers import ParsedDataset
from mathmodel_ai.schemas.data import (
    ColumnProfile,
    CorrelationRecord,
    CrossDatasetRelationship,
    DataProfile,
    DataProfileBundle,
    DataQualitySeverity,
    DataSemanticType,
    DatasetRecord,
    FeatureCandidate,
    FeatureRole,
    NumericStatistics,
    OutlierSummary,
    QualityIssue,
    ValueCount,
)

_UNIT_PATTERN = re.compile(r"(?:\(([^()]+)\)|\[([^\[\]]+)\])\s*$")
_TARGET_NAMES = frozenset({"target", "label", "response", "outcome", "demand", "cost"})
_TIME_NAMES = frozenset({"time", "date", "datetime", "timestamp", "year", "month", "day"})
_SPATIAL_NAMES = frozenset({"lat", "latitude", "lon", "lng", "longitude", "x", "y"})


def _missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and not math.isfinite(value))


def _safe_sample(value: Any) -> Any:
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


class DataProfiler:
    def profile(
        self,
        datasets: list[tuple[DatasetRecord, ParsedDataset]],
    ) -> DataProfileBundle:
        profiles = [self._profile_dataset(record, parsed.frame) for record, parsed in datasets]
        relationships = self._relationships(datasets)
        return DataProfileBundle(
            profiles=profiles,
            cross_dataset_relationships=relationships,
        )

    def _profile_dataset(self, record: DatasetRecord, frame: pl.DataFrame) -> DataProfile:
        columns = [
            self._profile_column(
                name,
                frame[name].to_list(),
                frame.height,
                physical_dtype=str(frame[name].dtype),
            )
            for name in frame.columns
        ]
        duplicate_count = self._duplicate_count(frame)
        duplicate_rate = duplicate_count / frame.height if frame.height else 0.0
        issues: list[QualityIssue] = []
        deductions = 0.0
        if frame.height == 0:
            issues.append(
                QualityIssue(
                    issue_id="empty-dataset",
                    severity=DataQualitySeverity.ERROR,
                    category="empty",
                    message="Dataset contains headers but no data rows.",
                )
            )
            deductions += 50
        if duplicate_count:
            issues.append(
                QualityIssue(
                    issue_id="duplicate-rows",
                    severity=DataQualitySeverity.WARNING,
                    category="duplicates",
                    message=f"Dataset contains {duplicate_count} duplicate rows.",
                    evidence={"count": duplicate_count, "rate": duplicate_rate},
                )
            )
            deductions += min(20, duplicate_rate * 50)
        for column in columns:
            if column.missing_rate > 0:
                severity = (
                    DataQualitySeverity.ERROR
                    if column.missing_rate >= 0.5
                    else DataQualitySeverity.WARNING
                )
                issues.append(
                    QualityIssue(
                        issue_id=f"missing-{column.name}",
                        severity=severity,
                        category="missing_values",
                        message=(
                            f"Column {column.name!r} has {column.missing_count} missing values."
                        ),
                        column=column.name,
                        evidence={"rate": column.missing_rate},
                    )
                )
                deductions += min(10, column.missing_rate * 10)
            if frame.height > 1 and column.unique_count <= 1:
                issues.append(
                    QualityIssue(
                        issue_id=f"constant-{column.name}",
                        severity=DataQualitySeverity.WARNING,
                        category="constant_column",
                        message=f"Column {column.name!r} is constant.",
                        column=column.name,
                    )
                )
                deductions += 3
            if column.outliers is not None and column.outliers.rate > 0.05:
                issues.append(
                    QualityIssue(
                        issue_id=f"outliers-{column.name}",
                        severity=DataQualitySeverity.INFO,
                        category="outliers",
                        message=(
                            f"Column {column.name!r} has {column.outliers.count} IQR outliers."
                        ),
                        column=column.name,
                        evidence={"rate": column.outliers.rate},
                    )
                )
                deductions += min(5, column.outliers.rate * 10)
        return DataProfile(
            dataset_id=record.dataset_id,
            source_file_id=record.source_file_id,
            dataset_name=record.name,
            sheet_name=record.sheet_name,
            row_count=frame.height,
            column_count=frame.width,
            columns=columns,
            duplicate_row_count=duplicate_count,
            duplicate_row_rate=duplicate_rate,
            correlations=self._correlations(frame, columns),
            feature_candidates=[
                candidate for column in columns for candidate in self._feature_candidates(column)
            ],
            quality_score=max(0.0, round(100 - deductions, 2)),
            quality_issues=issues,
        )

    def _profile_column(
        self,
        name: str,
        values: list[Any],
        row_count: int,
        *,
        physical_dtype: str,
    ) -> ColumnProfile:
        missing_count = sum(_missing(value) for value in values)
        present = [value for value in values if not _missing(value)]
        unique_strings = {str(value) for value in present}
        semantic = self._semantic_type(name, present, len(unique_strings), row_count)
        numeric = [
            float(value)
            for value in present
            if isinstance(value, int | float) and not isinstance(value, bool)
        ]
        numeric_statistics = self._numeric_statistics(numeric)
        outliers = self._outliers(numeric)
        counts = Counter(str(value) for value in present)
        return ColumnProfile(
            name=name,
            source_name=name,
            physical_dtype=physical_dtype,
            semantic_type=semantic,
            inferred_unit=self._unit(name),
            missing_count=missing_count,
            missing_rate=missing_count / row_count if row_count else 0.0,
            unique_count=len(unique_strings),
            unique_rate=len(unique_strings) / len(present) if present else 0.0,
            sample_values=[_safe_sample(value) for value in present[:10]],
            numeric_statistics=numeric_statistics,
            top_values=[
                ValueCount(value=value, count=count) for value, count in counts.most_common(10)
            ],
            outliers=outliers,
        )

    @staticmethod
    def _semantic_type(
        name: str, values: list[Any], unique_count: int, row_count: int
    ) -> DataSemanticType:
        if not values:
            return DataSemanticType.UNKNOWN
        lowered = name.casefold()
        if all(isinstance(value, bool) for value in values):
            return DataSemanticType.BOOLEAN
        if all(isinstance(value, datetime | date) for value in values):
            return DataSemanticType.DATETIME
        if all(isinstance(value, int) and not isinstance(value, bool) for value in values):
            if (lowered == "id" or lowered.endswith("_id")) and unique_count == len(values):
                return DataSemanticType.IDENTIFIER
            return DataSemanticType.INTEGER
        if all(isinstance(value, int | float) and not isinstance(value, bool) for value in values):
            return DataSemanticType.CONTINUOUS
        if (
            unique_count == len(values)
            and row_count >= 2
            and (lowered == "id" or lowered.endswith("_id") or lowered.endswith("code"))
        ):
            return DataSemanticType.IDENTIFIER
        if unique_count <= max(20, int(math.sqrt(max(row_count, 1)))):
            return DataSemanticType.CATEGORICAL
        return DataSemanticType.TEXT

    @staticmethod
    def _numeric_statistics(values: list[float]) -> NumericStatistics | None:
        if not values:
            return None
        ordered = sorted(values)
        q1, q3 = DataProfiler._quartiles(ordered)
        return NumericStatistics(
            count=len(values),
            minimum=min(values),
            maximum=max(values),
            mean=statistics.fmean(values),
            median=statistics.median(values),
            standard_deviation=statistics.stdev(values) if len(values) > 1 else 0.0,
            q1=q1,
            q3=q3,
        )

    @staticmethod
    def _outliers(values: list[float]) -> OutlierSummary | None:
        if len(values) < 4:
            return None
        q1, q3 = DataProfiler._quartiles(sorted(values))
        spread = q3 - q1
        lower = q1 - 1.5 * spread
        upper = q3 + 1.5 * spread
        count = sum(value < lower or value > upper for value in values)
        return OutlierSummary(
            lower_bound=lower,
            upper_bound=upper,
            count=count,
            rate=count / len(values),
        )

    @staticmethod
    def _quartiles(ordered: list[float]) -> tuple[float, float]:
        if len(ordered) == 1:
            return ordered[0], ordered[0]
        cuts = statistics.quantiles(ordered, n=4, method="inclusive")
        return cuts[0], cuts[2]

    @staticmethod
    def _duplicate_count(frame: pl.DataFrame) -> int:
        rows = [tuple(str(value) for value in row) for row in frame.iter_rows()]
        return len(rows) - len(set(rows))

    @staticmethod
    def _correlations(frame: pl.DataFrame, columns: list[ColumnProfile]) -> list[CorrelationRecord]:
        numeric_names = [
            column.name
            for column in columns
            if column.semantic_type in {DataSemanticType.INTEGER, DataSemanticType.CONTINUOUS}
        ][:30]
        records: list[CorrelationRecord] = []
        for left, right in combinations(numeric_names, 2):
            pairs = [
                (float(a), float(b))
                for a, b in zip(frame[left].to_list(), frame[right].to_list(), strict=True)
                if not _missing(a) and not _missing(b)
            ]
            if len(pairs) < 2:
                continue
            left_values = [pair[0] for pair in pairs]
            right_values = [pair[1] for pair in pairs]
            if len(set(left_values)) < 2 or len(set(right_values)) < 2:
                continue
            correlation = statistics.correlation(left_values, right_values)
            records.append(
                CorrelationRecord(
                    left_column=left,
                    right_column=right,
                    pearson=max(-1.0, min(1.0, correlation)),
                    pair_count=len(pairs),
                )
            )
        return records

    @staticmethod
    def _unit(name: str) -> str | None:
        match = _UNIT_PATTERN.search(name)
        if match:
            return match.group(1) or match.group(2)
        suffix = name.casefold().rsplit("_", maxsplit=1)[-1]
        return suffix if suffix in {"kg", "g", "km", "m", "s", "h", "usd", "rmb", "pct"} else None

    @staticmethod
    def _feature_candidates(column: ColumnProfile) -> list[FeatureCandidate]:
        name = column.name.casefold()
        if column.semantic_type is DataSemanticType.IDENTIFIER:
            return [
                FeatureCandidate(
                    column=column.name,
                    role=FeatureRole.IDENTIFIER,
                    confidence=0.9,
                    reason="Column values are unique and the name indicates an identifier.",
                )
            ]
        if column.semantic_type is DataSemanticType.DATETIME or name in _TIME_NAMES:
            return [
                FeatureCandidate(
                    column=column.name,
                    role=FeatureRole.TIME,
                    confidence=0.85,
                    reason="Physical type or column name indicates time ordering.",
                )
            ]
        if name in _SPATIAL_NAMES:
            return [
                FeatureCandidate(
                    column=column.name,
                    role=FeatureRole.SPATIAL,
                    confidence=0.7,
                    reason="Column name is a conventional spatial coordinate.",
                )
            ]
        if name in _TARGET_NAMES:
            return [
                FeatureCandidate(
                    column=column.name,
                    role=FeatureRole.TARGET,
                    confidence=0.55,
                    reason=(
                        "Name suggests a possible target; problem interpretation must confirm it."
                    ),
                )
            ]
        return [
            FeatureCandidate(
                column=column.name,
                role=FeatureRole.FEATURE,
                confidence=0.5,
                reason="Non-identifier column is a provisional feature candidate.",
            )
        ]

    @staticmethod
    def _relationships(
        datasets: list[tuple[DatasetRecord, ParsedDataset]],
    ) -> list[CrossDatasetRelationship]:
        relationships: list[CrossDatasetRelationship] = []
        for (left_record, left), (right_record, right) in combinations(datasets, 2):
            common = set(left.frame.columns) & set(right.frame.columns)
            for column in sorted(common):
                left_values = {
                    str(value) for value in left.frame[column].to_list() if not _missing(value)
                }
                right_values = {
                    str(value) for value in right.frame[column].to_list() if not _missing(value)
                }
                if not left_values or not right_values:
                    continue
                overlap = left_values & right_values
                ratio = len(overlap) / min(len(left_values), len(right_values))
                if ratio < 0.5:
                    continue
                relationships.append(
                    CrossDatasetRelationship(
                        relationship_id=(
                            f"REL-{left_record.dataset_id.hex[:8]}-"
                            f"{right_record.dataset_id.hex[:8]}-{column}"
                        ),
                        left_dataset_id=left_record.dataset_id,
                        left_column=column,
                        right_dataset_id=right_record.dataset_id,
                        right_column=column,
                        relationship_type="shared_key_candidate",
                        overlap_count=len(overlap),
                        overlap_ratio=ratio,
                        confidence=min(0.95, 0.5 + ratio / 2),
                        evidence=(
                            f"Shared column {column!r} has {len(overlap)} overlapping values."
                        ),
                    )
                )
        return relationships
