from __future__ import annotations

from mathmodel_ai.schemas.mathematical import MathematicalModel, VariableDomain
from mathmodel_ai.schemas.model_selection import ModelFamily

_EXECUTABLE_OPTIMIZATION_FAMILIES = frozenset(
    {
        ModelFamily.LINEAR_PROGRAMMING,
        ModelFamily.MIXED_INTEGER_LINEAR_PROGRAMMING,
        ModelFamily.INTEGER_PROGRAMMING,
        ModelFamily.NONLINEAR_PROGRAMMING,
    }
)
_CONTINUOUS_DOMAINS = frozenset({VariableDomain.CONTINUOUS, VariableDomain.NONNEGATIVE_CONTINUOUS})


def canonicalize_scalar_optimization_family(model: MathematicalModel) -> MathematicalModel:
    """Use a conservative executable family for a scalar continuous objective.

    This only repairs the family label. It never changes equations, parameters,
    decision domains, the objective, or unsupported indexed/state models.
    """
    if (
        model.objective is None
        or not model.decision_variables
        or model.state_variables
        or model.model_family in _EXECUTABLE_OPTIMIZATION_FAMILIES
        or any(
            item.index_sets or item.domain not in _CONTINUOUS_DOMAINS
            for item in model.decision_variables
        )
        or any(item.index_sets for item in model.derived_variables)
    ):
        return model
    note = (
        f"Executable family normalized from {model.model_family.value} to "
        "nonlinear_programming for a scalar continuous objective; local "
        "optimization does not prove global optimality."
    )
    return model.model_copy(
        update={
            "model_family": ModelFamily.NONLINEAR_PROGRAMMING,
            "limitations": [*model.limitations, note],
        }
    )
