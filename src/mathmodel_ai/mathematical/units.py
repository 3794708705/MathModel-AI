from __future__ import annotations

import re
from dataclasses import dataclass

from pydantic import BaseModel, Field

from mathmodel_ai.mathematical.registry import SymbolRegistry
from mathmodel_ai.schemas.mathematical import (
    BaseDimension,
    ExpressionKind,
    MathematicalModel,
    MathExpression,
    UnitCheckStatus,
    UnitExpression,
)

_UNIT_TOKEN = re.compile(r"^(?P<name>[^\^]+?)(?:\^(?P<power>-?\d+))?$")
_KNOWN_UNITS: dict[str, tuple[BaseDimension | None, float]] = {
    "1": (None, 1.0),
    "dimensionless": (None, 1.0),
    "kg": (BaseDimension.MASS, 1.0),
    "g": (BaseDimension.MASS, 0.001),
    "m": (BaseDimension.LENGTH, 1.0),
    "km": (BaseDimension.LENGTH, 1000.0),
    "s": (BaseDimension.TIME, 1.0),
    "sec": (BaseDimension.TIME, 1.0),
    "second": (BaseDimension.TIME, 1.0),
    "seconds": (BaseDimension.TIME, 1.0),
    "min": (BaseDimension.TIME, 60.0),
    "minute": (BaseDimension.TIME, 60.0),
    "hour": (BaseDimension.TIME, 3600.0),
    "hours": (BaseDimension.TIME, 3600.0),
    "h": (BaseDimension.TIME, 3600.0),
    "currency": (BaseDimension.CURRENCY, 1.0),
    "yuan": (BaseDimension.CURRENCY, 1.0),
    "元": (BaseDimension.CURRENCY, 1.0),
    "usd": (BaseDimension.CURRENCY, 1.0),
    "item": (BaseDimension.COUNT, 1.0),
    "items": (BaseDimension.COUNT, 1.0),
    "piece": (BaseDimension.COUNT, 1.0),
    "件": (BaseDimension.COUNT, 1.0),
    "j": (BaseDimension.ENERGY, 1.0),
    "joule": (BaseDimension.ENERGY, 1.0),
    "w": (BaseDimension.POWER, 1.0),
    "watt": (BaseDimension.POWER, 1.0),
}


def parse_unit(text: str) -> UnitExpression:
    normalized = text.strip()
    if not normalized:
        return UnitExpression(display=text or "1", unknown_units=["<empty>"])
    dimensions: dict[BaseDimension, int] = {}
    scale = 1.0
    unknown: list[str] = []
    sign = 1
    tokens = re.split(r"([*/])", normalized.replace(" ", ""))
    for token in tokens:
        if not token:
            continue
        if token == "*":
            sign = 1
            continue
        if token == "/":
            sign = -1
            continue
        match = _UNIT_TOKEN.fullmatch(token)
        if match is None:
            unknown.append(token)
            continue
        name = match.group("name").casefold()
        exponent = int(match.group("power") or "1") * sign
        known = _KNOWN_UNITS.get(name)
        if known is None:
            if token not in unknown:
                unknown.append(token)
            continue
        dimension, factor = known
        if dimension is not None:
            dimensions[dimension] = dimensions.get(dimension, 0) + exponent
        scale *= factor**exponent
    return UnitExpression(
        dimensions={key: value for key, value in dimensions.items() if value},
        scale=scale,
        display=normalized,
        unknown_units=unknown,
    )


def multiply_units(left: UnitExpression, right: UnitExpression) -> UnitExpression:
    dimensions = dict(left.dimensions)
    for dimension, exponent in right.dimensions.items():
        dimensions[dimension] = dimensions.get(dimension, 0) + exponent
    return UnitExpression(
        dimensions={key: value for key, value in dimensions.items() if value},
        scale=left.scale * right.scale,
        display=f"({left.display})*({right.display})",
        unknown_units=list(dict.fromkeys([*left.unknown_units, *right.unknown_units])),
    )


def divide_units(left: UnitExpression, right: UnitExpression) -> UnitExpression:
    dimensions = dict(left.dimensions)
    for dimension, exponent in right.dimensions.items():
        dimensions[dimension] = dimensions.get(dimension, 0) - exponent
    return UnitExpression(
        dimensions={key: value for key, value in dimensions.items() if value},
        scale=left.scale / right.scale,
        display=f"({left.display})/({right.display})",
        unknown_units=list(dict.fromkeys([*left.unknown_units, *right.unknown_units])),
    )


def power_unit(unit: UnitExpression, exponent: int) -> UnitExpression:
    return UnitExpression(
        dimensions={key: value * exponent for key, value in unit.dimensions.items() if value},
        scale=unit.scale**exponent,
        display=f"({unit.display})^{exponent}",
        unknown_units=list(unit.unknown_units),
    )


def same_dimension(left: UnitExpression, right: UnitExpression) -> bool:
    return left.dimensions == right.dimensions


class UnitIssue(BaseModel):
    reference: str
    status: UnitCheckStatus
    message: str


class UnitCheckReport(BaseModel):
    status: UnitCheckStatus
    issues: list[UnitIssue] = Field(default_factory=list)
    objective_unit: UnitExpression | None = None


@dataclass(frozen=True)
class _ExpressionUnit:
    unit: UnitExpression
    status: UnitCheckStatus
    issues: tuple[UnitIssue, ...] = ()


class UnitChecker:
    def __init__(self, symbols: SymbolRegistry) -> None:
        self._symbols = symbols

    def check_model(self, model: MathematicalModel) -> UnitCheckReport:
        issues: list[UnitIssue] = []
        objective_unit: UnitExpression | None = None
        if model.objective is not None:
            checked = self._expression_unit(
                model.objective.expression, model.objective.objective_id
            )
            objective_unit = checked.unit
            issues.extend(checked.issues)
            if model.objective.unit is not None:
                issues.extend(
                    self._compare_units(
                        checked.unit,
                        model.objective.unit,
                        model.objective.objective_id,
                        "objective declared unit",
                    )
                )
        for item in [
            *model.constraints,
            *model.initial_conditions,
            *model.boundary_conditions,
        ]:
            left = self._expression_unit(item.expression, item.constraint_id)
            right = self._expression_unit(item.rhs, item.constraint_id)
            issues.extend([*left.issues, *right.issues])
            issues.extend(
                self._compare_units(
                    left.unit,
                    right.unit,
                    item.constraint_id,
                    "constraint sides",
                )
            )
        for equation in model.equations:
            left = self._expression_unit(equation.lhs, equation.equation_id)
            right = self._expression_unit(equation.rhs, equation.equation_id)
            issues.extend([*left.issues, *right.issues])
            issues.extend(
                self._compare_units(
                    left.unit,
                    right.unit,
                    equation.equation_id,
                    "equation sides",
                )
            )
        status = UnitCheckStatus.PASS
        if any(item.status is UnitCheckStatus.FAIL for item in issues):
            status = UnitCheckStatus.FAIL
        elif any(item.status is UnitCheckStatus.UNKNOWN for item in issues):
            status = UnitCheckStatus.UNKNOWN
        return UnitCheckReport(status=status, issues=issues, objective_unit=objective_unit)

    def _expression_unit(self, expression: MathExpression, reference: str) -> _ExpressionUnit:
        if expression.kind is ExpressionKind.CONSTANT:
            return _ExpressionUnit(UnitExpression(), UnitCheckStatus.PASS)
        if expression.kind is ExpressionKind.SYMBOL:
            assert expression.symbol is not None
            definition = self._symbols.lookup(expression.symbol)
            if definition is None or definition.unit is None:
                unit = UnitExpression(display="unknown", unknown_units=[expression.symbol])
                return _ExpressionUnit(
                    unit,
                    UnitCheckStatus.UNKNOWN,
                    (
                        UnitIssue(
                            reference=reference,
                            status=UnitCheckStatus.UNKNOWN,
                            message=(
                                f"UNIT_CHECK_REQUIRED: unit for {expression.symbol!r} is unknown"
                            ),
                        ),
                    ),
                )
            if not definition.unit.is_known:
                return _ExpressionUnit(
                    definition.unit,
                    UnitCheckStatus.UNKNOWN,
                    (
                        UnitIssue(
                            reference=reference,
                            status=UnitCheckStatus.UNKNOWN,
                            message=f"UNIT_CHECK_REQUIRED: {definition.unit.display!r} is unknown",
                        ),
                    ),
                )
            return _ExpressionUnit(definition.unit, UnitCheckStatus.PASS)

        operands = [self._expression_unit(item, reference) for item in expression.operands]
        issues = tuple(issue for operand in operands for issue in operand.issues)
        if expression.kind in {ExpressionKind.ADD, ExpressionKind.SUBTRACT}:
            unit = operands[0].unit
            added_issues = list(issues)
            for operand in operands[1:]:
                added_issues.extend(
                    self._compare_units(unit, operand.unit, reference, "addition/subtraction")
                )
            status = self._status_from_issues(added_issues)
            return _ExpressionUnit(unit, status, tuple(added_issues))
        if expression.kind is ExpressionKind.MULTIPLY:
            unit = UnitExpression()
            for operand in operands:
                unit = multiply_units(unit, operand.unit)
            return _ExpressionUnit(unit, self._status_from_issues(list(issues)), issues)
        if expression.kind is ExpressionKind.DIVIDE:
            unit = divide_units(operands[0].unit, operands[1].unit)
            return _ExpressionUnit(unit, self._status_from_issues(list(issues)), issues)
        if expression.kind is ExpressionKind.NEGATE:
            return operands[0]
        if expression.kind is ExpressionKind.POWER:
            exponent_expression = expression.operands[1]
            if (
                operands[0].unit.is_known
                and not operands[0].unit.dimensions
                and operands[1].unit.is_known
                and not operands[1].unit.dimensions
            ):
                return _ExpressionUnit(
                    UnitExpression(), self._status_from_issues(list(issues)), issues
                )
            if (
                exponent_expression.kind is not ExpressionKind.CONSTANT
                or exponent_expression.value is None
                or not exponent_expression.value.is_integer()
            ):
                unknown = UnitIssue(
                    reference=reference,
                    status=UnitCheckStatus.UNKNOWN,
                    message="UNIT_CHECK_REQUIRED: unit exponent is not a constant integer",
                )
                return _ExpressionUnit(
                    operands[0].unit, UnitCheckStatus.UNKNOWN, (*issues, unknown)
                )
            unit = power_unit(operands[0].unit, int(exponent_expression.value))
            return _ExpressionUnit(unit, self._status_from_issues(list(issues)), issues)
        unknown = UnitIssue(
            reference=reference,
            status=UnitCheckStatus.UNKNOWN,
            message="UNIT_CHECK_REQUIRED: unsupported expression",
        )
        return _ExpressionUnit(UnitExpression(), UnitCheckStatus.UNKNOWN, (*issues, unknown))

    @staticmethod
    def _compare_units(
        left: UnitExpression,
        right: UnitExpression,
        reference: str,
        context: str,
    ) -> list[UnitIssue]:
        if not left.is_known or not right.is_known:
            return [
                UnitIssue(
                    reference=reference,
                    status=UnitCheckStatus.UNKNOWN,
                    message=f"UNIT_CHECK_REQUIRED: cannot compare {context}",
                )
            ]
        if not same_dimension(left, right):
            return [
                UnitIssue(
                    reference=reference,
                    status=UnitCheckStatus.FAIL,
                    message=(
                        f"dimension mismatch for {context}: {left.dimensions} != {right.dimensions}"
                    ),
                )
            ]
        return []

    @staticmethod
    def _status_from_issues(issues: list[UnitIssue]) -> UnitCheckStatus:
        if any(item.status is UnitCheckStatus.FAIL for item in issues):
            return UnitCheckStatus.FAIL
        if any(item.status is UnitCheckStatus.UNKNOWN for item in issues):
            return UnitCheckStatus.UNKNOWN
        return UnitCheckStatus.PASS
