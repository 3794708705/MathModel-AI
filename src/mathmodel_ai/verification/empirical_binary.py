"""Trusted row-wise empirical risk for grouped binary prediction models.

The expression describes a pre-outcome logit. Its row symbols are populated
only by the trusted causal CSV iterator, never by generated solver output.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

from mathmodel_ai.mathematical.expressions import (
    ExpressionError,
    evaluate_expression,
    referenced_symbols,
)
from mathmodel_ai.schemas.mathematical import (
    EMPIRICAL_ROW_SYMBOLS,
    EmpiricalBinaryRiskDefinition,
)
from mathmodel_ai.verification.causal_binary import (
    CausalBinarySpec,
    CausalFeature,
    iter_causal_binary_points,
)

EmpiricalBinaryRisk = EmpiricalBinaryRiskDefinition


def empirical_source_spec(risk: EmpiricalBinaryRiskDefinition) -> CausalBinarySpec:
    return CausalBinarySpec(
        source_sha256=risk.training_sha256,
        group_column=risk.group_column,
        condition_column=risk.condition_column,
        outcome_column=risk.outcome_column,
        positive_value=risk.positive_value,
        negative_value=risk.negative_value,
        history_window=risk.history_window,
    )


@dataclass(frozen=True)
class EmpiricalRiskResult:
    source_sha256: str
    row_count: int
    mean_log_loss: float
    mean_brier: float


def row_feature_values(feature: CausalFeature, *, reference_condition: str) -> dict[str, float]:
    """Compute bounded pre-outcome features with no access to the current label."""
    condition_rate = (feature.prior_condition_positive + 1) / (feature.prior_condition_count + 2)
    group_rate = (feature.prior_group_positive + 1) / (feature.prior_group_count + 2)
    recent_rate = (feature.recent_positive + 1) / (feature.recent_count + 2)
    return {
        "row_intercept": 1.0,
        "row_condition_reference": float(feature.condition == reference_condition),
        "row_condition_rate": condition_rate,
        "row_group_rate": group_rate,
        "row_recent_rate": recent_rate,
        "row_condition_delta": condition_rate - group_rate,
        "row_recent_delta": recent_rate - condition_rate,
    }


def predict_empirical_binary(
    risk: EmpiricalBinaryRiskDefinition,
    feature: CausalFeature,
    coefficients: Mapping[str, float],
) -> tuple[float, float]:
    """Return a stable sigmoid and its logit from an independently evaluated AST."""
    symbols = referenced_symbols(risk.logit)
    unknown = symbols - EMPIRICAL_ROW_SYMBOLS - coefficients.keys()
    if (
        unknown
        or EMPIRICAL_ROW_SYMBOLS & coefficients.keys()
        or any(
            isinstance(value, bool) or not math.isfinite(value) for value in coefficients.values()
        )
    ):
        raise ValueError("EMPIRICAL_COEFFICIENTS_INVALID")
    values = {
        **coefficients,
        **row_feature_values(feature, reference_condition=risk.reference_condition),
    }
    try:
        logit = evaluate_expression(risk.logit, values)
    except (ExpressionError, OverflowError, ValueError) as exc:
        raise ValueError("EMPIRICAL_LOGIT_NOT_EVALUABLE") from exc
    if not math.isfinite(logit):
        raise ValueError("EMPIRICAL_LOGIT_NOT_FINITE")
    if logit >= 0:
        probability = 1 / (1 + math.exp(-logit))
    else:
        exponential = math.exp(logit)
        probability = exponential / (1 + exponential)
    return probability, logit


def evaluate_empirical_binary_risk(
    source: bytes,
    risk: EmpiricalBinaryRiskDefinition,
    coefficients: Mapping[str, float],
) -> EmpiricalRiskResult:
    """Recompute every training-row loss from the exact hash-bound CSV bytes."""
    losses: list[float] = []
    briers: list[float] = []
    for feature, outcome in iter_causal_binary_points(source, empirical_source_spec(risk)):
        probability, logit = predict_empirical_binary(risk, feature, coefficients)
        losses.append(max(logit, 0.0) + math.log1p(math.exp(-abs(logit))) - outcome * logit)
        briers.append((probability - outcome) ** 2)
    if not losses:
        raise ValueError("EMPIRICAL_OBJECTIVE_NO_TRAINING_ROWS")
    return EmpiricalRiskResult(
        source_sha256=risk.training_sha256,
        row_count=len(losses),
        mean_log_loss=math.fsum(losses) / len(losses),
        mean_brier=math.fsum(briers) / len(briers),
    )


def audit_empirical_binary_claim(
    *,
    risk: EmpiricalBinaryRiskDefinition,
    training_csv: bytes,
    official_csv: bytes,
    official_spec: CausalBinarySpec,
    heldout_groups: Collection[str],
    coefficients: Mapping[str, float],
    reported_training_log_loss: float | None,
    reported_heldout_predictions: Sequence[float],
    tolerance: float = 1e-8,
) -> EmpiricalRiskResult:
    """Bind one persisted coefficient vector to training loss and sealed forecasts."""
    training_spec = empirical_source_spec(risk)
    if (
        not 0 < tolerance <= 1e-6
        or not heldout_groups
        or not set(heldout_groups)
        or any(
            getattr(training_spec, name) != getattr(official_spec, name)
            for name in (
                "group_column",
                "condition_column",
                "outcome_column",
                "positive_value",
                "negative_value",
                "history_window",
            )
        )
    ):
        raise ValueError("EMPIRICAL_POLICY_MISMATCH")
    heldout = set(heldout_groups)
    for feature, _ in iter_causal_binary_points(training_csv, training_spec):
        if feature.group in heldout:
            raise ValueError("EMPIRICAL_TRAINING_CONTAINS_HELDOUT_GROUP")
    training = evaluate_empirical_binary_risk(training_csv, risk, coefficients)
    if (
        reported_training_log_loss is None
        or not math.isfinite(reported_training_log_loss)
        or not math.isclose(
            reported_training_log_loss,
            training.mean_log_loss,
            rel_tol=tolerance,
            abs_tol=tolerance,
        )
    ):
        raise ValueError("EMPIRICAL_TRAINING_LOSS_MISMATCH")
    index = 0
    for feature, _ in iter_causal_binary_points(official_csv, official_spec):
        if feature.group not in heldout:
            continue
        if index >= len(reported_heldout_predictions):
            raise ValueError("EMPIRICAL_HELDOUT_TRACE_LENGTH_MISMATCH")
        expected, _ = predict_empirical_binary(risk, feature, coefficients)
        reported = reported_heldout_predictions[index]
        if not math.isfinite(reported) or not math.isclose(
            reported, expected, rel_tol=tolerance, abs_tol=tolerance
        ):
            raise ValueError(f"EMPIRICAL_HELDOUT_PREDICTION_MISMATCH:{index}")
        index += 1
    if index != len(reported_heldout_predictions):
        raise ValueError("EMPIRICAL_HELDOUT_TRACE_LENGTH_MISMATCH")
    return training
