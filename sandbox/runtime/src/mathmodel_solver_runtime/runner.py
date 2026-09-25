"""Trusted deterministic runtime installed in the versioned Phase 4 solver image."""

from __future__ import annotations

import math
from time import monotonic
from typing import Any


def _parameters(model: dict[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for parameter in [*model.get("parameters", []), *model.get("constants", [])]:
        value = parameter.get("value")
        if isinstance(value, (int, float, bool)):
            values[str(parameter["symbol"])] = float(value)
    return values


def _evaluate(expression: dict[str, Any], values: dict[str, float]) -> float:
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
        result = 1.0
        for operand in operands:
            result *= operand
        return result
    if kind == "DIVIDE":
        return operands[0] / operands[1]
    if kind == "POWER":
        return operands[0] ** operands[1]
    if kind == "NEGATE":
        return -operands[0]
    raise ValueError(f"unsupported expression kind {kind}")


def _resolved_scalar_values(model: dict[str, Any], values: dict[str, float]) -> dict[str, float]:
    """Evaluate uniquely defined derived scalars from the current candidate point."""
    resolved = dict(values)
    derived = {str(item["symbol"]) for item in model.get("derived_variables", [])}
    definitions: dict[str, list[dict[str, Any]]] = {}
    for equation in model.get("equations", []):
        lhs = equation["lhs"]
        if lhs["kind"] == "SYMBOL" and lhs["symbol"] in derived:
            definitions.setdefault(str(lhs["symbol"]), []).append(equation["rhs"])
    pending = {
        symbol: expressions[0]
        for symbol, expressions in definitions.items()
        if len(expressions) == 1 and symbol not in resolved
    }
    for _ in range(len(pending)):
        progress = False
        for symbol, expression in list(pending.items()):
            try:
                value = _evaluate(expression, resolved)
            except KeyError:
                continue
            if isinstance(value, complex) or not math.isfinite(value):
                raise ValueError(f"derived scalar {symbol} is not finite real")
            resolved[symbol] = value
            del pending[symbol]
            progress = True
        if not progress:
            break
    return resolved


def _solve_scalar_response(payload: dict[str, Any]) -> dict[str, Any]:
    """Evaluate scalar model responses at a fixed, explicitly supplied decision point."""
    started = monotonic()
    model = payload["model"]
    if model.get("objective") is not None or model.get("state_variables"):
        raise ValueError("scalar response requires an objective-free model without states")
    decisions = model.get("decision_variables", [])
    derived = model.get("derived_variables", [])
    if not derived or any(item.get("index_sets") for item in [*decisions, *derived]):
        raise ValueError("scalar response requires non-indexed derived outputs")
    fixed = payload["options"].get("initial_point", {})
    symbols = {str(item["symbol"]) for item in decisions}
    if set(fixed) != symbols:
        raise ValueError("scalar response fixed point must cover exactly the decision symbols")
    values = _resolved_scalar_values(
        model, {**_parameters(model), **{key: float(value) for key, value in fixed.items()}}
    )
    outputs = {str(item["symbol"]): values[str(item["symbol"])] for item in derived}
    if any(not math.isfinite(value) for value in outputs.values()):
        raise ValueError("scalar response must be finite")
    return {
        "solver_version": "scalar-response-v1",
        "native_status": 0,
        "success": True,
        "message": "fixed-point scalar response evaluated",
        "objective": None,
        "variables": {**fixed, **outputs},
        "runtime_seconds": monotonic() - started,
    }


def _linearize(
    expression: dict[str, Any],
    variable_symbols: set[str],
    parameter_values: dict[str, float],
) -> tuple[dict[str, float], float]:
    kind = expression["kind"]
    if kind == "CONSTANT":
        return {}, float(expression["value"])
    if kind == "SYMBOL":
        symbol = str(expression["symbol"])
        if symbol in variable_symbols:
            return {symbol: 1.0}, 0.0
        return {}, parameter_values[symbol]
    operands = [
        _linearize(item, variable_symbols, parameter_values) for item in expression["operands"]
    ]

    def add(
        left: tuple[dict[str, float], float],
        right: tuple[dict[str, float], float],
    ) -> tuple[dict[str, float], float]:
        coefficients = dict(left[0])
        for symbol, value in right[0].items():
            coefficients[symbol] = coefficients.get(symbol, 0.0) + value
        return coefficients, left[1] + right[1]

    def scale(
        form: tuple[dict[str, float], float], factor: float
    ) -> tuple[dict[str, float], float]:
        return {symbol: value * factor for symbol, value in form[0].items()}, form[1] * factor

    if kind == "ADD":
        result: tuple[dict[str, float], float] = ({}, 0.0)
        for operand in operands:
            result = add(result, operand)
        return result
    if kind == "SUBTRACT":
        return add(operands[0], scale(operands[1], -1.0))
    if kind == "NEGATE":
        return scale(operands[0], -1.0)
    if kind == "MULTIPLY":
        result = ({}, 1.0)
        for operand in operands:
            if result[0] and operand[0]:
                raise ValueError("nonlinear product in linear solver")
            result = scale(operand, result[1]) if operand[0] else scale(result, operand[1])
        return result
    if kind == "DIVIDE":
        if operands[1][0] or operands[1][1] == 0:
            raise ValueError("nonlinear or zero denominator in linear solver")
        return scale(operands[0], 1.0 / operands[1][1])
    if kind == "POWER" and not operands[1][0]:
        if operands[1][1] == 0:
            return {}, 1.0
        if operands[1][1] == 1:
            return operands[0]
    raise ValueError("nonlinear expression in linear solver")


def _bounds(variable: dict[str, Any]) -> tuple[float | None, float | None]:
    lower = variable.get("lower_bound")
    upper = variable.get("upper_bound")
    domain = variable["domain"]
    if domain in {"NONNEGATIVE_CONTINUOUS", "NONNEGATIVE_INTEGER"}:
        lower = max(float(lower or 0), 0.0)
    if domain == "BINARY":
        lower = max(float(lower or 0), 0.0)
        upper = min(float(upper if upper is not None else 1), 1.0)
    return (
        float(lower) if lower is not None else None,
        float(upper) if upper is not None else None,
    )


def _linear_problem(model: dict[str, Any]) -> dict[str, Any]:
    variables = model["decision_variables"]
    symbols = [str(item["symbol"]) for item in variables]
    symbol_set = set(symbols)
    parameters = _parameters(model)
    objective_coefficients, objective_constant = _linearize(
        model["objective"]["expression"], symbol_set, parameters
    )
    rows: list[list[float]] = []
    lower: list[float] = []
    upper: list[float] = []
    for constraint in model.get("constraints", []):
        left = _linearize(constraint["expression"], symbol_set, parameters)
        right = _linearize(constraint["rhs"], symbol_set, parameters)
        coefficients = {
            symbol: left[0].get(symbol, 0.0) - right[0].get(symbol, 0.0) for symbol in symbols
        }
        constant = left[1] - right[1]
        rows.append([coefficients[symbol] for symbol in symbols])
        relation = constraint["relation"]
        if relation == "LE":
            lower.append(float("-inf"))
            upper.append(-constant)
        elif relation == "GE":
            lower.append(-constant)
            upper.append(float("inf"))
        else:
            lower.append(-constant)
            upper.append(-constant)
    return {
        "variables": variables,
        "symbols": symbols,
        "parameters": parameters,
        "objective": [objective_coefficients.get(symbol, 0.0) for symbol in symbols],
        "objective_constant": objective_constant,
        "rows": rows,
        "constraint_lower": lower,
        "constraint_upper": upper,
    }


def _solve_scipy(payload: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import scipy
    from scipy.optimize import Bounds, LinearConstraint, linprog, milp, minimize

    model = payload["model"]
    options = payload["options"]
    family = model["model_family"]
    started = monotonic()
    if family in {
        "linear_programming",
        "mixed_integer_linear_programming",
        "integer_programming",
    }:
        problem = _linear_problem(model)
        c_original = np.asarray(problem["objective"], dtype=float)
        maximize = model["objective"]["sense"] == "MAXIMIZE"
        c = -c_original if maximize else c_original
        variable_bounds = [_bounds(item) for item in problem["variables"]]
        solver_options: dict[str, Any] = {}
        if options.get("time_limit_seconds") is not None:
            solver_options["time_limit"] = float(options["time_limit_seconds"])
        if family == "linear_programming":
            if options.get("iteration_limit") is not None:
                solver_options["maxiter"] = int(options["iteration_limit"])
            a_ub: list[list[float]] = []
            b_ub: list[float] = []
            a_eq: list[list[float]] = []
            b_eq: list[float] = []
            for row, lower, upper in zip(
                problem["rows"],
                problem["constraint_lower"],
                problem["constraint_upper"],
                strict=True,
            ):
                if lower == upper:
                    a_eq.append(row)
                    b_eq.append(upper)
                else:
                    if upper != float("inf"):
                        a_ub.append(row)
                        b_ub.append(upper)
                    if lower != float("-inf"):
                        a_ub.append([-value for value in row])
                        b_ub.append(-lower)
            result = linprog(
                c,
                A_ub=np.asarray(a_ub) if a_ub else None,
                b_ub=np.asarray(b_ub) if b_ub else None,
                A_eq=np.asarray(a_eq) if a_eq else None,
                b_eq=np.asarray(b_eq) if b_eq else None,
                bounds=variable_bounds,
                method="highs",
                options=solver_options,
            )
        else:
            if options.get("iteration_limit") is not None:
                solver_options["node_limit"] = int(options["iteration_limit"])
            if options.get("mip_gap") is not None:
                solver_options["mip_rel_gap"] = float(options["mip_gap"])
            bounds = Bounds(
                [lower if lower is not None else -np.inf for lower, _ in variable_bounds],
                [upper if upper is not None else np.inf for _, upper in variable_bounds],
            )
            integrality = np.asarray(
                [
                    0 if item["domain"] in {"CONTINUOUS", "NONNEGATIVE_CONTINUOUS"} else 1
                    for item in problem["variables"]
                ]
            )
            constraints = None
            if problem["rows"]:
                constraints = LinearConstraint(
                    np.asarray(problem["rows"], dtype=float),
                    np.asarray(problem["constraint_lower"], dtype=float),
                    np.asarray(problem["constraint_upper"], dtype=float),
                )
            result = milp(
                c,
                integrality=integrality,
                bounds=bounds,
                constraints=constraints,
                options=solver_options,
            )
        values = (
            {
                symbol: float(value)
                for symbol, value in zip(problem["symbols"], result.x, strict=True)
            }
            if result.x is not None
            else {}
        )
        objective = None
        if result.x is not None:
            objective = float(np.dot(c_original, result.x) + problem["objective_constant"])
        return {
            "solver_version": scipy.__version__,
            "native_status": int(result.status),
            "success": bool(result.success),
            "message": str(result.message),
            "objective": objective,
            "variables": values,
            "iterations": int(result.nit) if getattr(result, "nit", None) is not None else None,
            "nodes": (
                int(result.mip_node_count)
                if getattr(result, "mip_node_count", None) is not None
                else None
            ),
            "mip_gap": (
                float(result.mip_gap)
                if getattr(result, "mip_gap", None) is not None and np.isfinite(result.mip_gap)
                else None
            ),
            "runtime_seconds": monotonic() - started,
        }

    variables = model["decision_variables"]
    symbols = [str(item["symbol"]) for item in variables]
    parameters = _parameters(model)
    maximize = model["objective"]["sense"] == "MAXIMIZE"

    def values(vector: Any) -> dict[str, float]:
        return _resolved_scalar_values(
            model,
            {
                **parameters,
                **{symbol: float(value) for symbol, value in zip(symbols, vector, strict=True)},
            },
        )

    def objective(vector: Any) -> float:
        observed = _evaluate(model["objective"]["expression"], values(vector))
        return -observed if maximize else observed

    nonlinear_constraints: list[dict[str, Any]] = []
    for constraint in model.get("constraints", []):
        relation = constraint["relation"]

        def difference(vector: Any, item: dict[str, Any] = constraint) -> float:
            observed = values(vector)
            return _evaluate(item["expression"], observed) - _evaluate(item["rhs"], observed)

        if relation == "LE":
            nonlinear_constraints.append(
                {"type": "ineq", "fun": lambda vector, fn=difference: -fn(vector)}
            )
        elif relation == "GE":
            nonlinear_constraints.append({"type": "ineq", "fun": difference})
        else:
            nonlinear_constraints.append({"type": "eq", "fun": difference})
    initial = []
    supplied = options.get("initial_point", {})
    variable_bounds = []
    for variable in variables:
        lower, upper = _bounds(variable)
        variable_bounds.append((lower, upper))
        if variable["symbol"] in supplied:
            initial.append(float(supplied[variable["symbol"]]))
        elif lower is not None and upper is not None:
            initial.append((lower + upper) / 2)
        elif lower is not None:
            initial.append(lower + 1)
        elif upper is not None:
            initial.append(upper - 1)
        else:
            initial.append(0.0)
    minimize_options: dict[str, Any] = {"ftol": float(options["feasibility_tolerance"])}
    if options.get("iteration_limit") is not None:
        minimize_options["maxiter"] = int(options["iteration_limit"])
    result = minimize(
        objective,
        np.asarray(initial, dtype=float),
        method="SLSQP",
        bounds=variable_bounds,
        constraints=nonlinear_constraints,
        options=minimize_options,
    )
    observed = values(result.x) if result.x is not None else {}
    result_symbols = set(symbols) | {
        str(item["symbol"]) for item in model.get("derived_variables", [])
    }
    observed_values = {
        symbol: value for symbol, value in observed.items() if symbol in result_symbols
    }
    observed_objective = (
        _evaluate(model["objective"]["expression"], observed) if result.x is not None else None
    )
    return {
        "solver_version": scipy.__version__,
        "native_status": int(result.status),
        "success": bool(result.success),
        "message": str(result.message),
        "objective": observed_objective,
        "variables": observed_values,
        "iterations": int(result.nit) if getattr(result, "nit", None) is not None else None,
        "nodes": None,
        "mip_gap": None,
        "runtime_seconds": monotonic() - started,
    }


def _solve_ortools(payload: dict[str, Any]) -> dict[str, Any]:
    import ortools
    from ortools.sat.python import cp_model

    model_data = payload["model"]
    options = payload["options"]
    linear = _linear_problem(model_data)
    started = monotonic()
    model = cp_model.CpModel()
    variables: dict[str, Any] = {}
    for item in linear["variables"]:
        lower, upper = _bounds(item)
        if lower is None or upper is None:
            raise ValueError("CP-SAT requires explicit finite variable bounds")
        if item["domain"] == "BINARY":
            variables[item["symbol"]] = model.new_bool_var(item["symbol"])
        else:
            variables[item["symbol"]] = model.new_int_var(int(lower), int(upper), item["symbol"])
    for row, lower, upper in zip(
        linear["rows"],
        linear["constraint_lower"],
        linear["constraint_upper"],
        strict=True,
    ):
        expression = sum(
            int(coefficient) * variables[symbol]
            for symbol, coefficient in zip(linear["symbols"], row, strict=True)
        )
        if lower == upper:
            model.add(expression == int(lower))
        else:
            if lower != float("-inf"):
                model.add(expression >= int(lower))
            if upper != float("inf"):
                model.add(expression <= int(upper))
    objective = sum(
        int(coefficient) * variables[symbol]
        for symbol, coefficient in zip(linear["symbols"], linear["objective"], strict=True)
    )
    if model_data["objective"]["sense"] == "MAXIMIZE":
        model.maximize(objective)
    else:
        model.minimize(objective)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    if options.get("time_limit_seconds") is not None:
        solver.parameters.max_time_in_seconds = float(options["time_limit_seconds"])
    if options.get("mip_gap") is not None:
        solver.parameters.relative_gap_limit = float(options["mip_gap"])
    status = solver.solve(model)
    feasible = status in {cp_model.OPTIMAL, cp_model.FEASIBLE}
    values = (
        {symbol: float(solver.value(variable)) for symbol, variable in variables.items()}
        if feasible
        else {}
    )
    objective_value = (
        float(solver.objective_value + linear["objective_constant"]) if feasible else None
    )
    return {
        "solver_version": ortools.__version__,
        "native_status": int(status),
        "native_status_name": solver.status_name(status),
        "success": feasible,
        "message": solver.response_stats(),
        "objective": objective_value,
        "variables": values,
        "iterations": None,
        "nodes": int(solver.num_branches),
        "mip_gap": None,
        "runtime_seconds": monotonic() - started,
    }


def _solve_gurobi(payload: dict[str, Any]) -> dict[str, Any]:
    import gurobipy as gp
    from gurobipy import GRB

    model_data = payload["model"]
    options = payload["options"]
    linear = _linear_problem(model_data)
    started = monotonic()
    model = gp.Model("mathmodel-ai")
    model.Params.OutputFlag = 0
    if options.get("time_limit_seconds") is not None:
        model.Params.TimeLimit = float(options["time_limit_seconds"])
    if options.get("mip_gap") is not None:
        model.Params.MIPGap = float(options["mip_gap"])
    variables: dict[str, Any] = {}
    domain_types = {
        "CONTINUOUS": GRB.CONTINUOUS,
        "NONNEGATIVE_CONTINUOUS": GRB.CONTINUOUS,
        "INTEGER": GRB.INTEGER,
        "NONNEGATIVE_INTEGER": GRB.INTEGER,
        "BINARY": GRB.BINARY,
    }
    for item in linear["variables"]:
        lower, upper = _bounds(item)
        variables[item["symbol"]] = model.addVar(
            lb=lower if lower is not None else -GRB.INFINITY,
            ub=upper if upper is not None else GRB.INFINITY,
            vtype=domain_types[item["domain"]],
            name=item["symbol"],
        )
    for row, lower, upper in zip(
        linear["rows"],
        linear["constraint_lower"],
        linear["constraint_upper"],
        strict=True,
    ):
        expression = gp.quicksum(
            coefficient * variables[symbol]
            for symbol, coefficient in zip(linear["symbols"], row, strict=True)
        )
        if lower == upper:
            model.addConstr(expression == lower)
        else:
            if lower != float("-inf"):
                model.addConstr(expression >= lower)
            if upper != float("inf"):
                model.addConstr(expression <= upper)
    objective = gp.quicksum(
        coefficient * variables[symbol]
        for symbol, coefficient in zip(linear["symbols"], linear["objective"], strict=True)
    )
    sense = GRB.MAXIMIZE if model_data["objective"]["sense"] == "MAXIMIZE" else GRB.MINIMIZE
    model.setObjective(objective, sense)
    model.optimize()
    feasible = model.SolCount > 0
    values = (
        {symbol: float(variable.X) for symbol, variable in variables.items()} if feasible else {}
    )
    return {
        "solver_version": gp.gurobi.version().__str__(),
        "native_status": int(model.Status),
        "native_status_name": str(model.Status),
        "success": model.Status == GRB.OPTIMAL,
        "message": f"Gurobi status {model.Status}",
        "objective": float(model.ObjVal + linear["objective_constant"]) if feasible else None,
        "variables": values,
        "iterations": int(model.IterCount),
        "nodes": int(model.NodeCount),
        "mip_gap": float(model.MIPGap) if feasible and model.IsMIP else None,
        "runtime_seconds": monotonic() - started,
    }


def solve(payload: dict[str, Any]) -> dict[str, Any]:
    backend = payload["backend"]
    if backend == "scalar_response":
        return _solve_scalar_response(payload)
    if backend == "scipy":
        return _solve_scipy(payload)
    if backend == "ortools":
        return _solve_ortools(payload)
    if backend == "gurobi":
        return _solve_gurobi(payload)
    raise ValueError(f"unknown solver backend {backend!r}")
