"""Trusted sandbox source template, executed only by SandboxExecutor."""

DYNAMIC_REPLAY_CODE = r"""

import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp


def evaluate(expression, values):
    kind = expression["kind"]
    if kind == "CONSTANT":
        return expression["value"]
    if kind == "SYMBOL":
        return values[expression["symbol"]]
    args = [evaluate(e, values) for e in expression["operands"]]
    if kind == "ADD":
        return sum(args)
    if kind == "SUBTRACT":
        return args[0] - args[1]
    if kind == "MULTIPLY":
        return math.prod(args)
    if kind == "DIVIDE":
        return args[0] / args[1]
    if kind == "POWER":
        return args[0] ** args[1]
    if kind == "NEGATE":
        return -args[0]
    raise ValueError("unsupported AST operation")


def run(payload):
    dynamic = payload["scenario"]["dynamic"]
    model = payload["model"]
    states = sorted(dynamic["initial_state"])
    equations = {e["equation_id"]: e for e in model["equations"]}
    derivative_ids = set(dynamic["equation_by_state"].values())
    frozen_symbols = (set(payload["parameters"])
                      | set(payload["scenario"]["decision_values"])
                      | set(states))
    algebraic = [e for e in model["equations"] if e["equation_id"] not in derivative_ids
                 and e["lhs"]["kind"] == "SYMBOL"
                 and e["lhs"]["symbol"] not in frozen_symbols]

    def environment(y):
        values = {**payload["parameters"], **payload["scenario"]["decision_values"],
                  **dict(zip(states, y, strict=True))}
        remaining = list(algebraic)
        while remaining:
            progress = False
            for equation in remaining[:]:
                try:
                    value = evaluate(equation["rhs"], values)
                except KeyError:
                    continue
                symbol = equation["lhs"]["symbol"]
                if symbol in values:
                    raise ValueError("algebraic assignment overrides a frozen input")
                values[symbol] = value
                remaining.remove(equation)
                progress = True
            if not progress:
                # Formal models may also declare diagnostics that depend on
                # derivative outputs or operations recomputed by the independent
                # metric evaluator. They are not prerequisites for integration.
                # A missing algebraic value required by a derivative still fails
                # closed when that derivative is evaluated below.
                break
        return values

    def rhs(_time, y):
        values = environment(y)
        return [evaluate(equations[dynamic["equation_by_state"][s]]["rhs"], values)
                for s in states]

    times = np.linspace(dynamic["start"], dynamic["stop"], dynamic["samples"])
    solution = solve_ivp(rhs, (dynamic["start"], dynamic["stop"]),
        [dynamic["initial_state"][s] for s in states], t_eval=times,
        method="RK45", rtol=dynamic["rtol"], atol=dynamic["atol"])
    if not solution.success or solution.y.shape[1] != len(times):
        raise ValueError("dynamic integration did not finish all required samples")
    final = environment(solution.y[:, -1])
    variables = {key: float(value) for key, value in final.items()
                 if key not in payload["parameters"]}
    output = {"variables": variables, "series": {
        s: [float(x) for x in solution.y[i]] for i, s in enumerate(states)}}
    Path("/output/metrics.json").write_text(json.dumps(output, allow_nan=False), encoding="utf-8")
"""
