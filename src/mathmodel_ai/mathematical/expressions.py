from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from functools import reduce
from operator import mul

from mathmodel_ai.schemas.mathematical import (
    ExpressionKind,
    MathExpression,
    ParameterDefinition,
)


class ExpressionError(ValueError):
    """A typed expression cannot be evaluated by the requested deterministic path."""


def referenced_symbols(expression: MathExpression) -> set[str]:
    if expression.kind is ExpressionKind.SYMBOL:
        assert expression.symbol is not None
        return {expression.symbol}
    return {symbol for operand in expression.operands for symbol in referenced_symbols(operand)}


def evaluate_expression(expression: MathExpression, values: Mapping[str, float]) -> float:
    kind = expression.kind
    if kind is ExpressionKind.CONSTANT:
        assert expression.value is not None
        return expression.value
    if kind is ExpressionKind.SYMBOL:
        assert expression.symbol is not None
        try:
            return float(values[expression.symbol])
        except KeyError as exc:
            raise ExpressionError(f"no numeric value for symbol {expression.symbol!r}") from exc
    operands = [evaluate_expression(item, values) for item in expression.operands]
    if kind is ExpressionKind.ADD:
        return sum(operands)
    if kind is ExpressionKind.SUBTRACT:
        return operands[0] - operands[1]
    if kind is ExpressionKind.MULTIPLY:
        return reduce(mul, operands, 1.0)
    if kind is ExpressionKind.DIVIDE:
        if operands[1] == 0:
            raise ExpressionError("division by zero")
        return operands[0] / operands[1]
    if kind is ExpressionKind.POWER:
        try:
            result = operands[0] ** operands[1]
        except (OverflowError, ValueError) as exc:
            raise ExpressionError("invalid power expression") from exc
        if isinstance(result, complex) or not math.isfinite(float(result)):
            raise ExpressionError("power expression is not a finite real value")
        return float(result)
    if kind is ExpressionKind.NEGATE:
        return -operands[0]
    raise ExpressionError(f"unsupported expression kind {kind.value}")


@dataclass(frozen=True)
class LinearForm:
    coefficients: dict[str, float]
    constant: float = 0.0

    def plus(self, other: LinearForm) -> LinearForm:
        coefficients = dict(self.coefficients)
        for symbol, value in other.coefficients.items():
            coefficients[symbol] = coefficients.get(symbol, 0.0) + value
        return LinearForm(
            coefficients={key: value for key, value in coefficients.items() if value != 0},
            constant=self.constant + other.constant,
        )

    def scaled(self, factor: float) -> LinearForm:
        return LinearForm(
            coefficients={key: value * factor for key, value in self.coefficients.items()},
            constant=self.constant * factor,
        )


def linearize_expression(
    expression: MathExpression,
    *,
    variable_symbols: set[str],
    parameter_values: Mapping[str, float],
) -> LinearForm:
    kind = expression.kind
    if kind is ExpressionKind.CONSTANT:
        assert expression.value is not None
        return LinearForm({}, expression.value)
    if kind is ExpressionKind.SYMBOL:
        assert expression.symbol is not None
        if expression.symbol in variable_symbols:
            return LinearForm({expression.symbol: 1.0})
        try:
            return LinearForm({}, float(parameter_values[expression.symbol]))
        except KeyError as exc:
            raise ExpressionError(
                f"symbol {expression.symbol!r} is neither a scalar parameter nor variable"
            ) from exc

    operands = [
        linearize_expression(
            item,
            variable_symbols=variable_symbols,
            parameter_values=parameter_values,
        )
        for item in expression.operands
    ]
    if kind is ExpressionKind.ADD:
        return reduce(lambda left, right: left.plus(right), operands, LinearForm({}))
    if kind is ExpressionKind.SUBTRACT:
        return operands[0].plus(operands[1].scaled(-1))
    if kind is ExpressionKind.NEGATE:
        return operands[0].scaled(-1)
    if kind is ExpressionKind.MULTIPLY:
        result = LinearForm({}, 1.0)
        for operand in operands:
            if result.coefficients and operand.coefficients:
                raise ExpressionError("product of symbolic terms is nonlinear")
            if operand.coefficients:
                result = operand.scaled(result.constant)
            else:
                result = result.scaled(operand.constant)
        return result
    if kind is ExpressionKind.DIVIDE:
        denominator = operands[1]
        if denominator.coefficients:
            raise ExpressionError("division by a symbolic term is nonlinear")
        if denominator.constant == 0:
            raise ExpressionError("division by zero")
        return operands[0].scaled(1 / denominator.constant)
    if kind is ExpressionKind.POWER:
        exponent = operands[1]
        if exponent.coefficients:
            raise ExpressionError("symbolic exponent is nonlinear")
        if exponent.constant == 0:
            return LinearForm({}, 1.0)
        if exponent.constant == 1:
            return operands[0]
        raise ExpressionError("power greater than one is nonlinear")
    raise ExpressionError(f"unsupported expression kind {kind.value}")


def scalar_parameter_values(
    parameters: list[ParameterDefinition],
) -> dict[str, float]:
    values: dict[str, float] = {}
    for parameter in parameters:
        value = parameter.value
        if isinstance(value, bool):
            values[parameter.symbol] = float(value)
        elif isinstance(value, int | float):
            values[parameter.symbol] = float(value)
    return values


def expression_text(expression: MathExpression) -> str:
    if expression.kind is ExpressionKind.CONSTANT:
        assert expression.value is not None
        return str(expression.value)
    if expression.kind is ExpressionKind.SYMBOL:
        assert expression.symbol is not None
        return expression.symbol
    operands = [expression_text(item) for item in expression.operands]
    symbols = {
        ExpressionKind.ADD: " + ",
        ExpressionKind.SUBTRACT: " - ",
        ExpressionKind.MULTIPLY: " * ",
        ExpressionKind.DIVIDE: " / ",
        ExpressionKind.POWER: " ^ ",
    }
    if expression.kind is ExpressionKind.NEGATE:
        return f"-({operands[0]})"
    return f"({symbols[expression.kind].join(operands)})"
