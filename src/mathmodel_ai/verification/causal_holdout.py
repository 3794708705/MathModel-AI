"""Group-separated, online binary prediction protocol.

This pure harness defines the scientific ordering. A production predictor must
be an isolated sandbox proxy: Python callbacks in this process are test doubles,
not a security boundary against an untrusted model reading the original CSV.
"""

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Protocol, cast
from uuid import UUID

from mathmodel_ai.schemas.benchmark import CausalScienceCheck
from mathmodel_ai.verification.causal_binary import (
    CausalBinarySpec,
    CausalFeature,
    iter_causal_binary_points,
)


class OnlineBinaryPredictor(Protocol):
    def fit(self, feature: CausalFeature, outcome: float) -> None: ...

    def predict(self, feature: CausalFeature) -> float: ...


@dataclass(frozen=True)
class CausalHoldoutResult:
    heldout_groups: tuple[str, ...]
    training_groups: tuple[str, ...]
    predictions: tuple[float, ...]
    observations: tuple[float, ...]
    baseline_predictions: tuple[float, ...]
    brier: float
    baseline_brier: float


@dataclass(frozen=True)
class CausalCalibrationBin:
    index: int
    count: int
    mean_prediction: float | None
    observed_rate: float | None


@dataclass(frozen=True)
class CausalCalibrationAssessment:
    count: int
    ece: float
    bins: tuple[CausalCalibrationBin, ...]


def assess_binary_calibration(
    result: CausalHoldoutResult, *, bin_count: int = 10
) -> CausalCalibrationAssessment:
    """Fixed-width reliability bins and descriptive ECE; no quality threshold."""
    if (
        not 2 <= bin_count <= 100
        or not result.predictions
        or len(result.predictions) != len(result.observations)
    ):
        raise ValueError("CAUSAL_CALIBRATION_INPUT_INVALID")
    counts = [0] * bin_count
    predicted = [0.0] * bin_count
    observed = [0.0] * bin_count
    for probability, outcome in zip(result.predictions, result.observations, strict=True):
        if not math.isfinite(probability) or not 0 <= probability <= 1 or outcome not in (0.0, 1.0):
            raise ValueError("CAUSAL_CALIBRATION_INPUT_INVALID")
        index = min(int(probability * bin_count), bin_count - 1)
        counts[index] += 1
        predicted[index] += probability
        observed[index] += outcome
    total = len(result.predictions)
    bins = tuple(
        CausalCalibrationBin(
            index=index,
            count=counts[index],
            mean_prediction=predicted[index] / counts[index] if counts[index] else None,
            observed_rate=observed[index] / counts[index] if counts[index] else None,
        )
        for index in range(bin_count)
    )
    return CausalCalibrationAssessment(
        count=total,
        ece=math.fsum(abs(p - y) for p, y in zip(predicted, observed, strict=True)) / total,
        bins=bins,
    )


@dataclass(frozen=True)
class AuditedCausalEvidence:
    """A formal-result-bound holdout replay, constructed by the trusted evaluator."""

    formal_result_id: UUID
    holdout_execution_id: UUID
    source_sha256: str
    trace_sha256: str
    result: CausalHoldoutResult
    science_policy_sha256: str | None = None
    required_scientific_checks: tuple[CausalScienceCheck, ...] = ()
    calibration: CausalCalibrationAssessment | None = None


def causal_trace_payload(
    spec: CausalBinarySpec,
    result: CausalHoldoutResult,
    *,
    fraction: float,
    salt: str,
) -> bytes:
    """Canonical host-recorded transcript, not a model-authored metrics file."""
    return json.dumps(
        {
            "protocol": "causal-binary-holdout-v1",
            "spec": asdict(spec),
            "fraction": fraction,
            "salt": salt,
            "heldout_groups": result.heldout_groups,
            "training_groups": result.training_groups,
            "predictions": result.predictions,
            "observations": result.observations,
            "baseline_predictions": result.baseline_predictions,
            "brier": result.brier,
            "baseline_brier": result.baseline_brier,
        },
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


class _RecordedPredictor:
    def __init__(self, predictions: list[float]) -> None:
        self._values = iter(predictions)

    def fit(self, feature: CausalFeature, outcome: float) -> None:
        del feature, outcome

    def predict(self, feature: CausalFeature) -> float:
        del feature
        return next(self._values)


def verify_causal_trace(source: bytes, payload: bytes) -> CausalHoldoutResult:
    """Independently rederive groups, labels, baseline and loss from the CSV."""
    if len(payload) > 4 * 1024 * 1024:
        raise ValueError("CAUSAL_TRACE_TOO_LARGE")
    try:
        trace = json.loads(payload)
        if not isinstance(trace, dict) or trace.get("protocol") != "causal-binary-holdout-v1":
            raise ValueError("CAUSAL_TRACE_PROTOCOL_INVALID")
        spec = CausalBinarySpec(**trace["spec"])
        fraction = trace["fraction"]
        salt = trace["salt"]
        predictions = trace["predictions"]
        if (
            isinstance(fraction, bool)
            or not isinstance(fraction, (int, float))
            or not isinstance(salt, str)
            or not isinstance(predictions, list)
            or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                for value in predictions
            )
        ):
            raise ValueError("CAUSAL_TRACE_FIELDS_INVALID")
        recorded = _RecordedPredictor(cast("list[float]", predictions))
        result = evaluate_causal_holdout(
            source, spec, recorded, fraction=float(fraction), salt=salt
        )
        if causal_trace_payload(spec, result, fraction=float(fraction), salt=salt) != payload:
            raise ValueError("CAUSAL_TRACE_RECOMPUTATION_MISMATCH")
        return result
    except (KeyError, TypeError, StopIteration, json.JSONDecodeError) as exc:
        raise ValueError("CAUSAL_TRACE_FIELDS_INVALID") from exc


def _group_hash(group: str, salt: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{salt}:{group}".encode()).digest(), "big")


def heldout_groups(groups: set[str], *, fraction: float, salt: str) -> tuple[str, ...]:
    """Choose groups by identity alone, never by their labels or model scores."""
    if len(groups) < 2 or not math.isfinite(fraction) or not 0 < fraction < 1 or not salt:
        raise ValueError("GROUP_HOLDOUT_DESIGN_INVALID")
    ranked = sorted(groups, key=lambda group: (_group_hash(group, salt), group))
    count = max(1, min(len(ranked) - 1, round(len(ranked) * fraction)))
    return tuple(sorted(ranked[:count]))


def evaluate_causal_holdout(
    source: bytes,
    spec: CausalBinarySpec,
    predictor: OnlineBinaryPredictor,
    *,
    fraction: float,
    salt: str,
) -> CausalHoldoutResult:
    """Fit on other groups, then issue one held-out query before its next query."""
    groups = {feature.group for feature, _ in iter_causal_binary_points(source, spec)}
    heldout = heldout_groups(groups, fraction=fraction, salt=salt)
    heldout_set = set(heldout)
    for feature, outcome in iter_causal_binary_points(source, spec):
        if feature.group not in heldout_set:
            predictor.fit(feature, outcome)
    predictions: list[float] = []
    observations: list[float] = []
    baseline: list[float] = []
    for feature, outcome in iter_causal_binary_points(source, spec):
        if feature.group not in heldout_set:
            continue
        value = predictor.predict(feature)
        if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("HELDOUT_PREDICTION_NOT_A_PROBABILITY")
        predictions.append(value)
        observations.append(outcome)
        baseline.append(
            (1 + feature.prior_condition_positive) / (2 + feature.prior_condition_count)
        )
    if not predictions:
        raise ValueError("HELDOUT_PREDICTIONS_MISSING")
    brier = math.fsum(
        (prediction - outcome) ** 2
        for prediction, outcome in zip(predictions, observations, strict=True)
    ) / len(predictions)
    baseline_brier = math.fsum(
        (prediction - outcome) ** 2
        for prediction, outcome in zip(baseline, observations, strict=True)
    ) / len(predictions)
    return CausalHoldoutResult(
        heldout_groups=heldout,
        training_groups=tuple(sorted(groups - heldout_set)),
        predictions=tuple(predictions),
        observations=tuple(observations),
        baseline_predictions=tuple(baseline),
        brier=brier,
        baseline_brier=baseline_brier,
    )
