from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from mathmodel_ai.mathematical.expressions import (
    ExpressionError,
    evaluate_expression,
    expression_text,
    linearize_expression,
    referenced_symbols,
    scalar_parameter_values,
)
from mathmodel_ai.schemas.mathematical import (
    ExpressionKind,
    MathExpression,
    ParameterDefinition,
    ParameterSourceType,
)
from tests.mathematical.helpers import add, constant, multiply, power, symbol


def _parameter(symbol_name: str, value: object) -> ParameterDefinition:
    return ParameterDefinition(
        parameter_id=f"PAR-{symbol_name}",
        symbol=symbol_name,
        description=f"fixture parameter {symbol_name}",
        value=value,
        source_type=ParameterSourceType.DATA,
        source_ref="EVID-fixture",
        confidence=1,
    )


def test_linear_expression_supports_parameters_negative_and_scientific_coefficients() -> None:
    expression = add(
        multiply(constant(-2.5), symbol("x")),
        multiply(constant(1e-6), symbol("y")),
        symbol("demand"),
    )

    form = linearize_expression(
        expression,
        variable_symbols={"x", "y"},
        parameter_values={"demand": 10},
    )

    assert form.coefficients == {"x": -2.5, "y": 1e-6}
    assert form.constant == 10
    assert evaluate_expression(expression, {"x": 2, "y": 1_000_000, "demand": 10}) == 6
    assert referenced_symbols(expression) == {"x", "y", "demand"}


def test_constant_negate_and_division_have_explicit_linear_semantics() -> None:
    negated = MathExpression(kind=ExpressionKind.NEGATE, operands=[symbol("x")])
    divided = MathExpression(
        kind=ExpressionKind.DIVIDE,
        operands=[add(symbol("x"), constant(4)), constant(2)],
    )

    assert (
        linearize_expression(constant(7), variable_symbols=set(), parameter_values={}).constant == 7
    )
    assert linearize_expression(
        negated, variable_symbols={"x"}, parameter_values={}
    ).coefficients == {"x": -1}
    divided_form = linearize_expression(divided, variable_symbols={"x"}, parameter_values={})
    assert divided_form.coefficients == pytest.approx({"x": 0.5})
    assert divided_form.constant == pytest.approx(2)
    assert evaluate_expression(divided, {"x": 6}) == 5
    assert expression_text(negated) == "-(x)"


def test_indexed_parameter_is_preserved_but_not_falsely_treated_as_scalar() -> None:
    indexed = _parameter("demand", {"north": 8, "south": 12})
    values = scalar_parameter_values([indexed])
    expression = symbol("demand[north]")

    assert values == {}
    with pytest.raises(ExpressionError, match="neither a scalar parameter nor variable"):
        linearize_expression(
            expression,
            variable_symbols=set(),
            parameter_values=values,
        )


@pytest.mark.parametrize(
    "expression",
    [
        multiply(symbol("x"), symbol("y")),
        power(symbol("x"), 2),
        MathExpression(
            kind=ExpressionKind.DIVIDE,
            operands=[symbol("x"), symbol("y")],
        ),
    ],
)
def test_nonlinear_operations_are_rejected_by_linear_translation(
    expression: MathExpression,
) -> None:
    with pytest.raises(ExpressionError, match="nonlinear"):
        linearize_expression(
            expression,
            variable_symbols={"x", "y"},
            parameter_values={},
        )


def test_power_and_division_runtime_errors_are_truthful() -> None:
    division = MathExpression(
        kind=ExpressionKind.DIVIDE,
        operands=[constant(1), constant(0)],
    )
    invalid_power = power(constant(-1), 0.5)

    with pytest.raises(ExpressionError, match="division by zero"):
        evaluate_expression(division, {})
    with pytest.raises(ExpressionError, match="finite real"):
        evaluate_expression(invalid_power, {})


def test_expression_schema_rejects_invalid_shape_operator_syntax_nan_and_infinity() -> None:
    with pytest.raises(ValidationError):
        MathExpression.model_validate({"kind": "MODULO", "operands": []})
    with pytest.raises(ValidationError, match="ADD requires at least two operands"):
        MathExpression(kind=ExpressionKind.ADD, operands=[constant(1)])
    with pytest.raises(ValidationError):
        MathExpression(kind=ExpressionKind.SYMBOL, symbol="x + y")
    for value in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValidationError):
            MathExpression(kind=ExpressionKind.CONSTANT, value=value)


def test_undefined_symbol_and_symbolic_zero_division_fail_explicitly() -> None:
    with pytest.raises(ExpressionError, match="no numeric value"):
        evaluate_expression(symbol("missing"), {})
    denominator = MathExpression(
        kind=ExpressionKind.DIVIDE,
        operands=[symbol("x"), constant(0)],
    )
    with pytest.raises(ExpressionError, match="division by zero"):
        linearize_expression(
            denominator,
            variable_symbols={"x"},
            parameter_values={},
        )
