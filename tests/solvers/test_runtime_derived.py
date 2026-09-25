import runpy
from pathlib import Path


def test_runtime_resolves_only_unique_acyclic_derived_scalars() -> None:
    runtime = runpy.run_path(
        str(
            Path(__file__).resolve().parents[2]
            / "sandbox/runtime/src/mathmodel_solver_runtime/runner.py"
        )
    )
    resolve = runtime["_resolved_scalar_values"]
    model = {
        "derived_variables": [{"symbol": "flow"}, {"symbol": "hazard"}],
        "equations": [
            {
                "lhs": {"kind": "SYMBOL", "symbol": "flow"},
                "rhs": {
                    "kind": "ADD",
                    "operands": [
                        {"kind": "SYMBOL", "symbol": "x"},
                        {"kind": "CONSTANT", "value": 1.0},
                    ],
                },
            },
            {
                "lhs": {"kind": "SYMBOL", "symbol": "hazard"},
                "rhs": {
                    "kind": "MULTIPLY",
                    "operands": [
                        {"kind": "SYMBOL", "symbol": "flow"},
                        {"kind": "CONSTANT", "value": 0.5},
                    ],
                },
            },
        ],
    }

    assert resolve(model, {"x": 2.0}) == {"x": 2.0, "flow": 3.0, "hazard": 1.5}
    assert resolve(model, {"x": 4.0}) == {"x": 4.0, "flow": 5.0, "hazard": 2.5}
    ambiguous = {**model, "equations": [*model["equations"], model["equations"][0]]}
    assert resolve(ambiguous, {"x": 2.0}) == {"x": 2.0}
