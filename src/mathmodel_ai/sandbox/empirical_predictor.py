"""Deterministic, fixed-coefficient predictor for isolated empirical scenarios.

This source is executed by the existing causal holdout container. It is not a
fitter and never uses the outcomes delivered to ``fit``. The trusted host must
independently audit every forecast against the formal empirical risk contract.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping

from mathmodel_ai.mathematical.expressions import referenced_symbols
from mathmodel_ai.schemas.mathematical import (
    EMPIRICAL_ROW_SYMBOLS,
    EmpiricalBinaryRiskDefinition,
)

_PREDICTOR_BODY = """
def _evaluate(expression, values):
    kind = expression["kind"]
    if kind == "CONSTANT":
        return float(expression["value"])
    if kind == "SYMBOL":
        return float(values[expression["symbol"]])
    operands = [_evaluate(item, values) for item in expression["operands"]]
    if kind == "ADD":
        return sum(operands)
    if kind == "SUBTRACT":
        return operands[0] - operands[1]
    if kind == "MULTIPLY":
        product = 1.0
        for operand in operands:
            product *= operand
        return product
    if kind == "DIVIDE":
        return operands[0] / operands[1]
    if kind == "POWER":
        return operands[0] ** operands[1]
    if kind == "NEGATE":
        return -operands[0]
    raise ValueError("unsupported empirical expression")


class Predictor:
    def fit(self, feature, outcome):
        # Coefficients are fixed before the sealed holdout starts. Deliberately
        # ignore the label; there is no post-hoc refit or access to future labels.
        del feature, outcome

    def predict(self, feature):
        condition_rate = (feature["prior_condition_positive"] + 1) / (
            feature["prior_condition_count"] + 2
        )
        group_rate = (feature["prior_group_positive"] + 1) / (
            feature["prior_group_count"] + 2
        )
        recent_rate = (feature["recent_positive"] + 1) / (feature["recent_count"] + 2)
        values = {
            **PAYLOAD["coefficients"],
            "row_intercept": 1.0,
            "row_condition_reference": float(
                feature["condition"] == PAYLOAD["reference_condition"]
            ),
            "row_condition_rate": condition_rate,
            "row_group_rate": group_rate,
            "row_recent_rate": recent_rate,
            "row_condition_delta": condition_rate - group_rate,
            "row_recent_delta": recent_rate - condition_rate,
        }
        logit = _evaluate(PAYLOAD["logit"], values)
        if not isinstance(logit, (int, float)) or not math.isfinite(logit):
            raise ValueError("empirical scenario logit is not finite")
        if logit >= 0:
            return 1 / (1 + math.exp(-logit))
        exponential = math.exp(logit)
        return exponential / (1 + exponential)
"""


def empirical_predictor_source(
    risk: EmpiricalBinaryRiskDefinition,
    coefficients: Mapping[str, float],
) -> str:
    """Freeze an audited scenario vector into networkless predictor source."""
    symbols = referenced_symbols(risk.logit)
    if (
        symbols - EMPIRICAL_ROW_SYMBOLS != coefficients.keys()
        or EMPIRICAL_ROW_SYMBOLS & coefficients.keys()
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            for value in coefficients.values()
        )
    ):
        raise ValueError("EMPIRICAL_SCENARIO_COEFFICIENTS_INVALID")
    payload = json.dumps(
        {
            "coefficients": dict(sorted(coefficients.items())),
            "reference_condition": risk.reference_condition,
            "logit": risk.logit.model_dump(mode="json"),
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return (
        "import json\nimport math\nPAYLOAD = json.loads(" + repr(payload) + ")\n" + _PREDICTOR_BODY
    )
