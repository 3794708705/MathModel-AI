from __future__ import annotations

import math

from mathmodel_ai.mathematical.expressions import (
    ExpressionError,
    evaluate_expression,
    referenced_symbols,
    scalar_parameter_values,
)
from mathmodel_ai.schemas.mathematical import (
    ConstraintRelation,
    ExpressionKind,
    MathematicalModel,
    VariableDomain,
)
from mathmodel_ai.schemas.solver import FeasibilityReport


def check_feasibility(
    model: MathematicalModel,
    variable_values: dict[str, float],
    *,
    tolerance: float,
) -> FeasibilityReport:
    if not variable_values:
        return FeasibilityReport(
            checked=False,
            tolerance=tolerance,
            message="no candidate variable values were available for feasibility checking",
        )
    derived = {item.symbol: item for item in model.derived_variables if not item.index_sets}
    values = {
        **scalar_parameter_values([*model.parameters, *model.constants]),
        **{symbol: value for symbol, value in variable_values.items() if symbol not in derived},
    }
    bound_violations: list[str] = []
    max_violation = 0.0
    definitions = {
        symbol: [
            equation.rhs
            for equation in model.equations
            if equation.lhs.kind is ExpressionKind.SYMBOL and equation.lhs.symbol == symbol
        ]
        for symbol in derived
    }
    pending = {
        symbol: expressions[0]
        for symbol, expressions in definitions.items()
        if len(expressions) == 1
    }
    for _ in range(len(pending)):
        progressed = False
        for symbol, expression in list(pending.items()):
            if not referenced_symbols(expression) <= set(values):
                continue
            try:
                resolved = evaluate_expression(expression, values)
            except ExpressionError:
                continue
            if not math.isfinite(resolved):
                continue
            reported = variable_values.get(symbol)
            if reported is not None and not math.isclose(
                reported, resolved, rel_tol=tolerance, abs_tol=tolerance
            ):
                bound_violations.append(f"{derived[symbol].variable_id}:derived_mismatch")
                max_violation = max(max_violation, abs(reported - resolved))
            values[symbol] = resolved
            del pending[symbol]
            progressed = True
        if not progressed:
            break
    for variable in model.decision_variables:
        if variable.symbol not in variable_values:
            bound_violations.append(f"{variable.variable_id}:missing_value")
            continue
        value = variable_values[variable.symbol]
        lower = variable.lower_bound
        upper = variable.upper_bound
        if variable.domain in {
            VariableDomain.NONNEGATIVE_CONTINUOUS,
            VariableDomain.NONNEGATIVE_INTEGER,
            VariableDomain.BINARY,
        }:
            lower = max(lower or 0.0, 0.0)
        if variable.domain is VariableDomain.BINARY:
            upper = min(upper if upper is not None else 1.0, 1.0)
        if lower is not None and value < lower - tolerance:
            violation = lower - value
            max_violation = max(max_violation, violation)
            bound_violations.append(variable.variable_id)
        if upper is not None and value > upper + tolerance:
            violation = value - upper
            max_violation = max(max_violation, violation)
            bound_violations.append(variable.variable_id)
        if variable.domain in {
            VariableDomain.INTEGER,
            VariableDomain.NONNEGATIVE_INTEGER,
            VariableDomain.BINARY,
        }:
            violation = abs(value - round(value))
            if violation > tolerance:
                max_violation = max(max_violation, violation)
                bound_violations.append(f"{variable.variable_id}:integrality")

    violated_constraints: list[str] = []
    for constraint in [
        *model.constraints,
        *model.initial_conditions,
        *model.boundary_conditions,
    ]:
        try:
            left = evaluate_expression(constraint.expression, values)
            right = evaluate_expression(constraint.rhs, values)
        except ExpressionError:
            violated_constraints.append(f"{constraint.constraint_id}:unchecked")
            continue
        if constraint.relation is ConstraintRelation.LE:
            violation = max(left - right, 0.0)
        elif constraint.relation is ConstraintRelation.GE:
            violation = max(right - left, 0.0)
        else:
            violation = abs(left - right)
        max_violation = max(max_violation, violation)
        if violation > tolerance:
            violated_constraints.append(constraint.constraint_id)
    return FeasibilityReport(
        checked=True,
        max_constraint_violation=max_violation,
        violated_constraints=violated_constraints,
        bound_violations=list(dict.fromkeys(bound_violations)),
        tolerance=tolerance,
        message=(
            "implemented bounds and constraints satisfy tolerance"
            if not violated_constraints and not bound_violations
            else "one or more implemented bounds or constraints violate tolerance"
        ),
    )
