from __future__ import annotations

import math
from collections.abc import Mapping

from mathmodel_ai.schemas.mathematical import ExpressionKind, MathExpression


class IndependentEvaluationError(ValueError):
    """The independent validator cannot produce a finite real result."""


class IndependentExpressionEvaluator:
    """Bounded evaluator intentionally independent from the Phase 4 solver path."""

    def evaluate(self, expression: MathExpression, values: Mapping[str, float]) -> float:
        result = self._walk(expression, values)
        if not math.isfinite(result):
            raise IndependentEvaluationError("expression produced a non-finite value")
        return result

    def _walk(self, expression: MathExpression, values: Mapping[str, float]) -> float:
        if expression.kind is ExpressionKind.CONSTANT:
            if expression.value is None:
                raise IndependentEvaluationError("constant expression has no value")
            return float(expression.value)
        if expression.kind is ExpressionKind.SYMBOL:
            if expression.symbol is None or expression.symbol not in values:
                raise IndependentEvaluationError(
                    f"independent evaluator has no value for {expression.symbol!r}"
                )
            return float(values[expression.symbol])

        operands = [self._walk(item, values) for item in expression.operands]
        try:
            if expression.kind is ExpressionKind.ADD:
                result = math.fsum(operands)
            elif expression.kind is ExpressionKind.SUBTRACT:
                result = operands[0] - operands[1]
            elif expression.kind is ExpressionKind.MULTIPLY:
                result = math.prod(operands)
            elif expression.kind is ExpressionKind.DIVIDE:
                if operands[1] == 0:
                    raise IndependentEvaluationError("independent division by zero")
                result = operands[0] / operands[1]
            elif expression.kind is ExpressionKind.POWER:
                powered = operands[0] ** operands[1]
                if isinstance(powered, complex):
                    raise IndependentEvaluationError("independent power is complex")
                result = float(powered)
            elif expression.kind is ExpressionKind.NEGATE:
                result = -operands[0]
            else:
                raise IndependentEvaluationError(
                    f"unsupported independent expression kind {expression.kind.value}"
                )
        except IndependentEvaluationError:
            raise
        except (OverflowError, ValueError) as exc:
            raise IndependentEvaluationError("independent expression evaluation failed") from exc
        if not math.isfinite(result):
            raise IndependentEvaluationError("independent expression is not finite")
        return result
