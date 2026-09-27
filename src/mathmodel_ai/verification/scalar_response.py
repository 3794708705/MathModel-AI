from __future__ import annotations

import math

from mathmodel_ai.mathematical.expressions import (
    ExpressionError,
    evaluate_expression,
    scalar_parameter_values,
)
from mathmodel_ai.schemas.mathematical import ExpressionKind, MathematicalModel, MathExpression
from mathmodel_ai.schemas.results import ResultRecord


def evaluate_scalar_responses(
    model: MathematicalModel, fixed_values: dict[str, float]
) -> dict[str, float]:
    """Independently recompute complete, uniquely defined scalar responses."""
    if model.objective is not None or model.state_variables:
        raise ValueError("scalar response requires an objective-free model without states")
    if not model.derived_variables or any(
        item.index_sets for item in [*model.decision_variables, *model.derived_variables]
    ):
        raise ValueError("scalar response requires non-indexed derived outputs")
    decisions = {item.symbol for item in model.decision_variables}
    if set(fixed_values) != decisions or any(
        not math.isfinite(value) for value in fixed_values.values()
    ):
        raise ValueError("fixed point must contain exactly the finite decision values")
    derived = {item.symbol for item in model.derived_variables}
    definitions: dict[str, list[MathExpression]] = {symbol: [] for symbol in derived}
    for equation in model.equations:
        if equation.lhs.kind is ExpressionKind.SYMBOL and equation.lhs.symbol in derived:
            assert equation.lhs.symbol is not None
            definitions[equation.lhs.symbol].append(equation.rhs)
    if any(len(items) != 1 for items in definitions.values()):
        raise ValueError("each scalar response needs exactly one defining equation")
    values = {
        **scalar_parameter_values([*model.parameters, *model.constants]),
        **fixed_values,
    }
    pending = {key: items[0] for key, items in definitions.items()}
    for _ in range(len(pending)):
        progressed = False
        for symbol, expression in list(pending.items()):
            try:
                value = evaluate_expression(expression, values)
            except ExpressionError as exc:
                if "no numeric value for symbol" in str(exc):
                    continue
                raise ValueError(f"response {symbol}: {exc}") from exc
            if not math.isfinite(value):
                raise ValueError(f"response {symbol} is not finite")
            values[symbol] = value
            del pending[symbol]
            progressed = True
        if not progressed:
            break
    if pending:
        raise ValueError("unresolved or cyclic scalar responses: " + ", ".join(sorted(pending)))
    return {symbol: values[symbol] for symbol in sorted(derived)}


def formal_baseline_responses(
    model: MathematicalModel, result: ResultRecord
) -> tuple[dict[str, float], dict[str, float]]:
    fixed = {
        item.symbol: result.key_outputs[item.symbol]
        for item in model.decision_variables
        if item.symbol in result.key_outputs
    }
    responses = evaluate_scalar_responses(model, fixed)
    for symbol, value in responses.items():
        recorded = result.key_outputs.get(symbol)
        if recorded is not None and not math.isclose(recorded, value, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"formal result response {symbol} disagrees with model equation")
    return fixed, responses


def response_ranges(
    responses: dict[str, float], outputs: list[dict[str, float]]
) -> dict[str, tuple[float, float]]:
    if not outputs or not responses:
        return {}
    return {
        symbol: (min(item[symbol] for item in outputs), max(item[symbol] for item in outputs))
        for symbol in responses
    }
