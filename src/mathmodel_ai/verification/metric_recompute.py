from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from mathmodel_ai.mathematical.expressions import referenced_symbols
from mathmodel_ai.paper.hashing import sha256_bytes
from mathmodel_ai.schemas.independent_verification import (
    IndependentStatus,
    MetricKey,
    MetricSpec,
    RawMetricOutput,
    VerifiedMetric,
)
from mathmodel_ai.schemas.mathematical import MathematicalModel
from mathmodel_ai.schemas.verification import ValidationCheckStatus
from mathmodel_ai.verification.evaluator import IndependentExpressionEvaluator
from mathmodel_ai.verification.validation import IndependentValidator


def canonical_bytes(value: BaseModel | dict[str, Any] | list[Any]) -> bytes:
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def content_digest(value: BaseModel | dict[str, Any] | list[Any]) -> str:
    return sha256_bytes(canonical_bytes(value))


def calculator_digest() -> str:
    # Bind actual implementation bytes, including the independent evaluator.
    return sha256_bytes(
        b"".join(
            Path(__file__).with_name(name).read_bytes()
            for name in (
                "metric_recompute.py",
                "evaluator.py",
                "validation.py",
                "observation_source.py",
            )
        )
    )


def numeric_environment(model: MathematicalModel, raw: RawMetricOutput) -> dict[str, float]:
    parameters: dict[str, float] = {}
    for item in [*model.parameters, *model.constants]:
        if isinstance(item.value, bool) or not isinstance(item.value, int | float):
            raise ValueError("metric needs finite scalar model parameters")
        if not math.isfinite(item.value):
            raise ValueError("non-finite model parameter")
        parameters[item.symbol] = float(item.value)
    if set(parameters) & set(raw.variables):
        raise ValueError("output variables cannot override model parameters")
    return {**parameters, **raw.variables}


def constraint_violation(model: MathematicalModel, raw: RawMetricOutput) -> float:
    validator = IndependentValidator()
    values = numeric_environment(model, raw)
    variable_checks = [
        validator.check_variable(v, raw.variables)
        for v in [*model.decision_variables, *model.state_variables]
    ]
    constraints = [
        validator.check_constraint(c, values)
        for c in [*model.constraints, *model.initial_conditions, *model.boundary_conditions]
    ]
    if any(v.value is None for v in variable_checks) or any(
        c.status is ValidationCheckStatus.UNCHECKED for c in constraints
    ):
        raise ValueError("all model variables and hard constraints must be evaluable")
    return max(
        [
            0.0,
            *(
                max(v.bound_violation or 0.0, v.integrality_violation or 0.0)
                for v in variable_checks
            ),
            *(c.violation for c in constraints if c.violation is not None),
        ]
    )


def algebraic_value(model: MathematicalModel, raw: RawMetricOutput, symbol: str) -> float:
    """Resolve one declared derived scalar from the formal AST, never reported output."""

    derived = {item.symbol for item in model.derived_variables}
    if symbol not in derived:
        raise ValueError("algebraic scalar must name a declared derived variable")
    base_symbols = {item.symbol for item in [*model.decision_variables, *model.state_variables]}
    values = numeric_environment(
        model,
        raw.model_copy(
            update={"variables": {k: v for k, v in raw.variables.items() if k in base_symbols}}
        ),
    )
    assignments: dict[str, Any] = {}
    for equation in model.equations:
        if equation.lhs.kind.value != "SYMBOL" or equation.lhs.symbol not in derived:
            continue
        target = equation.lhs.symbol
        if target in assignments:
            raise ValueError("algebraic derived variable has multiple defining equations")
        assignments[target] = equation.rhs
    evaluator = IndependentExpressionEvaluator()
    remaining = dict(assignments)
    while remaining:
        progress = False
        for target, expression in list(remaining.items()):
            if not referenced_symbols(expression) <= set(values):
                continue
            value = evaluator.evaluate(expression, values)
            if not math.isfinite(value):
                raise ValueError("algebraic scalar calculation is non-finite")
            values[target] = float(value)
            del remaining[target]
            progress = True
        if symbol in values:
            return values[symbol]
        if not progress:
            break
    raise ValueError("algebraic scalar dependencies are unknown, cyclic, or incomplete")


def calculate(
    spec: MetricSpec,
    raw: RawMetricOutput,
    *,
    observations: list[float] | None = None,
    model: MathematicalModel | None = None,
) -> float:
    """Closed registry: no executable strings, dynamic imports, or reported inputs."""
    key = spec.key
    if key in {
        MetricKey.MAE,
        MetricKey.RMSE,
        MetricKey.R2,
        MetricKey.MAX_ERROR,
        MetricKey.BRIER,
        MetricKey.LOG_LOSS,
    }:
        if not observations or len(observations) != len(raw.predictions):
            raise ValueError(
                "independent observations and predictions must have equal nonzero size"
            )
        if not all(math.isfinite(v) and not isinstance(v, bool) for v in observations):
            raise ValueError("observations must be finite numbers")
        errors = [a - b for a, b in zip(observations, raw.predictions, strict=True)]
        if key in {MetricKey.BRIER, MetricKey.LOG_LOSS} and any(
            outcome not in (0.0, 1.0) or prediction < 0 or prediction > 1
            for outcome, prediction in zip(observations, raw.predictions, strict=True)
        ):
            raise ValueError("binary scoring requires 0/1 outcomes and probabilities in [0,1]")
        if key is MetricKey.MAE:
            value = math.fsum(abs(e) for e in errors) / len(errors)
        elif key is MetricKey.MAX_ERROR:
            value = max(abs(e) for e in errors)
        elif key is MetricKey.RMSE:
            value = math.hypot(*errors) / math.sqrt(len(errors))
        elif key is MetricKey.BRIER:
            value = math.fsum(e * e for e in errors) / len(errors)
        elif key is MetricKey.LOG_LOSS:
            if any(
                (outcome == 1.0 and prediction == 0.0) or (outcome == 0.0 and prediction == 1.0)
                for outcome, prediction in zip(observations, raw.predictions, strict=True)
            ):
                raise ValueError("log loss is infinite for a confidently wrong prediction")
            value = math.fsum(
                -math.log(prediction) if outcome == 1.0 else -math.log1p(-prediction)
                for outcome, prediction in zip(observations, raw.predictions, strict=True)
            ) / len(observations)
        else:
            if len(observations) < 2:
                raise ValueError("R2 requires at least two observations")
            mean = math.fsum(observations) / len(observations)
            denominator = math.fsum((v - mean) ** 2 for v in observations)
            if denominator == 0:
                raise ValueError(
                    "R2 is undefined for a constant target; no force-finite substitution"
                )
            value = 1.0 - math.fsum(e**2 for e in errors) / denominator
    elif key in {MetricKey.MEAN, MetricKey.MINIMUM, MetricKey.MAXIMUM, MetricKey.FINAL_VALUE}:
        series = raw.series.get(spec.series_key or "", [])
        if not series or len(series) > 100000:
            raise ValueError("metric requires a bounded nonempty raw series")
        if key is MetricKey.MEAN:
            value = math.fsum(series) / len(series)
        elif key is MetricKey.MINIMUM:
            value = min(series)
        elif key is MetricKey.MAXIMUM:
            value = max(series)
        else:
            value = series[-1]
    elif key is MetricKey.ALGEBRAIC_SCALAR:
        if model is None or spec.value_symbol is None:
            raise ValueError("algebraic scalar requires an exact model and value symbol")
        value = algebraic_value(model, raw, spec.value_symbol)
    else:
        if model is None:
            raise ValueError("optimization metric requires an exact mathematical model")
        if key in {MetricKey.CONSTRAINT_MAX_VIOLATION, MetricKey.FEASIBLE}:
            violation = constraint_violation(model, raw)
            value = (
                float(violation <= spec.absolute_tolerance)
                if key is MetricKey.FEASIBLE
                else violation
            )
        else:
            if model.objective is None:
                raise ValueError("model has no scalar objective")
            objective = IndependentExpressionEvaluator().evaluate(
                model.objective.expression, numeric_environment(model, raw)
            )
            if key is MetricKey.OBJECTIVE:
                value = objective
            elif key is MetricKey.MIP_GAP:
                if raw.best_bound is None or objective == 0:
                    raise ValueError(
                        "MIP gap requires a solver bound and nonzero incumbent objective"
                    )
                value = abs(objective - raw.best_bound) / abs(objective)
            else:
                raise ValueError("unregistered metric calculator")
    if not math.isfinite(value):
        raise ValueError("metric calculation is non-finite")
    return float(value)


def recompute(
    spec: MetricSpec,
    raw: RawMetricOutput,
    *,
    source_digest: str,
    observations: list[float] | None = None,
    model: MathematicalModel | None = None,
) -> VerifiedMetric:
    reported = raw.reported.get(spec.reported_key or spec.metric_id)
    identity = dict(
        metric_id=spec.metric_id,
        key=spec.key,
        version=spec.version,
        reported=reported,
        source_digest=source_digest,
        calculator_digest=calculator_digest(),
    )
    try:
        value = calculate(spec, raw, observations=observations, model=model)
        delta = reported - value if reported is not None else None
        if delta is not None and not math.isfinite(delta):
            raise ValueError("reported-versus-verified delta is non-finite")
        mismatch = delta is not None and (
            delta != 0
            if spec.exact or spec.key is MetricKey.FEASIBLE
            else abs(delta) > spec.absolute_tolerance + spec.relative_tolerance * abs(value)
        )
        errors = []
        if mismatch:
            errors.append("reported metric differs from independent value")
        if spec.reported_key is not None and reported is None:
            errors.append("required reported comparison is missing")
        if spec.lower_threshold is not None and value < spec.lower_threshold:
            errors.append("verified metric below lower threshold")
        if spec.upper_threshold is not None and value > spec.upper_threshold:
            errors.append("verified metric above upper threshold")
        if spec.key is MetricKey.FEASIBLE and value != 1:
            errors.append("independent hard-constraint feasibility failed")
        if spec.key is MetricKey.CONSTRAINT_MAX_VIOLATION and value > spec.absolute_tolerance:
            errors.append("independent hard-constraint tolerance exceeded")
        return VerifiedMetric(
            **identity,
            verified=value,
            delta=delta,
            status=IndependentStatus.FAIL if errors else IndependentStatus.PASS,
            error="; ".join(errors) or None,
        )
    except (ValueError, OverflowError, ArithmeticError) as exc:
        return VerifiedMetric(**identity, status=IndependentStatus.INVALID, error=str(exc))
