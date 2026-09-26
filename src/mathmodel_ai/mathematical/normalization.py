from __future__ import annotations

import math

from mathmodel_ai.schemas.mathematical import (
    DataBinding,
    MathematicalModel,
    ParameterSourceType,
    VariableDomain,
)
from mathmodel_ai.schemas.model_selection import ModelFamily
from mathmodel_ai.schemas.problem_state import ProblemState

_EXECUTABLE_OPTIMIZATION_FAMILIES = frozenset(
    {
        ModelFamily.LINEAR_PROGRAMMING,
        ModelFamily.MIXED_INTEGER_LINEAR_PROGRAMMING,
        ModelFamily.INTEGER_PROGRAMMING,
        ModelFamily.NONLINEAR_PROGRAMMING,
    }
)
_CONTINUOUS_DOMAINS = frozenset({VariableDomain.CONTINUOUS, VariableDomain.NONNEGATIVE_CONTINUOUS})


def deterministic_bound_scalar(binding: DataBinding, state: ProblemState) -> float | None:
    """Return a scalar only when its registered deterministic profile proves it."""
    datasets = [item for item in state.datasets if item.dataset_id == binding.dataset_id]
    profiles = [item for item in state.data_profiles if item.dataset_id == binding.dataset_id]
    if len(datasets) != 1 or len(profiles) != 1:
        return None
    dataset, profile = datasets[0], profiles[0]
    if (
        not profile.deterministic
        or profile.source_file_id != dataset.source_file_id
        or profile.row_count != dataset.row_count
        or binding.selector is not None
    ):
        return None
    columns = [item for item in profile.columns if item.name == binding.column]
    if len(columns) != 1:
        return None
    column = columns[0]
    value: float | None = None
    if binding.transform == "mean" and column.numeric_statistics is not None:
        value = column.numeric_statistics.mean
    elif binding.transform == "count_nonmissing":
        value = float(profile.row_count - column.missing_count)
    elif binding.transform and binding.transform.startswith("rate_eq:"):
        target = binding.transform.removeprefix("rate_eq:")
        matches = [item.count for item in column.top_values if item.value == target]
        denominator = profile.row_count - column.missing_count
        if len(matches) == 1 and denominator:
            value = matches[0] / denominator
    return float(value) if value is not None and math.isfinite(value) else None


def materialize_data_bound_scalars(
    model: MathematicalModel, state: ProblemState
) -> MathematicalModel:
    """Fill omitted DATA scalars only from exact registered profile statistics."""
    replacements: list[str] = []
    parameters = []
    for item in model.parameters:
        binding = item.data_binding
        expected = (
            deterministic_bound_scalar(binding, state)
            if binding is not None and item.source_type is ParameterSourceType.DATA
            else None
        )
        if expected is not None and item.value is None:
            item = item.model_copy(update={"value": expected})
            replacements.append(item.symbol)
        parameters.append(item)
    constants = []
    for item in model.constants:
        binding = item.data_binding
        expected = (
            deterministic_bound_scalar(binding, state)
            if binding is not None and item.source_type is ParameterSourceType.DATA
            else None
        )
        if expected is not None and item.value is None:
            item = item.model_copy(update={"value": expected})
            replacements.append(item.symbol)
        constants.append(item)
    if not replacements:
        return model
    return model.model_copy(
        update={
            "parameters": parameters,
            "constants": constants,
            "limitations": [
                *model.limitations,
                "DATA scalar values materialized from registered deterministic profiles: "
                + ", ".join(sorted(replacements)),
            ],
        }
    )


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
