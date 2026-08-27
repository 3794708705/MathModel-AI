from __future__ import annotations

from collections.abc import Mapping, Sequence

from mathmodel_ai.schemas.mathematical import MathematicalModel, VariableDomain
from mathmodel_ai.schemas.model_selection import ModelFamily
from mathmodel_ai.schemas.solver import AlgorithmFamily, AlgorithmPlan, SolverFamily

_DEFAULT_PREFERENCES: dict[ModelFamily, tuple[SolverFamily, ...]] = {
    ModelFamily.LINEAR_PROGRAMMING: (
        SolverFamily.SCIPY_HIGHS,
        SolverFamily.GUROBI,
    ),
    ModelFamily.MIXED_INTEGER_LINEAR_PROGRAMMING: (
        SolverFamily.GUROBI,
        SolverFamily.SCIPY_MILP,
        SolverFamily.ORTOOLS_CP_SAT,
    ),
    ModelFamily.INTEGER_PROGRAMMING: (
        SolverFamily.ORTOOLS_CP_SAT,
        SolverFamily.GUROBI,
        SolverFamily.SCIPY_MILP,
    ),
    ModelFamily.NONLINEAR_PROGRAMMING: (SolverFamily.SCIPY_MINIMIZE,),
}


class AlgorithmSelector:
    """Deterministic selection from declared mathematical structure, never keywords."""

    def __init__(
        self,
        preferences: Mapping[ModelFamily, Sequence[SolverFamily]] | None = None,
    ) -> None:
        merged = dict(_DEFAULT_PREFERENCES)
        if preferences is not None:
            for family, solvers in preferences.items():
                if not solvers:
                    raise ValueError(f"solver preference for {family.value} cannot be empty")
                merged[family] = tuple(solvers)
        self._preferences = merged

    def select(self, model: MathematicalModel) -> AlgorithmPlan:
        family = model.model_family
        domains = {item.domain for item in model.decision_variables}
        requirements = [
            f"{len(model.decision_variables)} decision variables",
            f"{len(model.constraints)} primary constraints",
        ]
        risks: list[str] = []
        if any(
            item.lower_bound is None or item.upper_bound is None
            for item in model.decision_variables
        ):
            risks.append("one or more variables have an open bound")
        if (
            model.algorithm_requirements.convexity.value == "UNKNOWN"
            and family is ModelFamily.NONLINEAR_PROGRAMMING
        ):
            risks.append("nonlinear convexity is unknown; a local solution may not be global")

        if family is ModelFamily.LINEAR_PROGRAMMING:
            algorithm = AlgorithmFamily.LINEAR_OPTIMIZATION
            reason = (
                "linear objective and constraints with continuous domains use a reliable LP path"
            )
        elif family is ModelFamily.MIXED_INTEGER_LINEAR_PROGRAMMING:
            algorithm = AlgorithmFamily.MIXED_INTEGER_OPTIMIZATION
            reason = "mixed discrete and continuous domains require a MILP-capable solver"
        elif family is ModelFamily.INTEGER_PROGRAMMING:
            algorithm = AlgorithmFamily.INTEGER_CONSTRAINT_PROGRAMMING
            reason = "all decision domains are discrete and admit integer/CP solving"
        elif family is ModelFamily.NONLINEAR_PROGRAMMING:
            algorithm = AlgorithmFamily.NONLINEAR_LOCAL_OPTIMIZATION
            reason = (
                "the typed nonlinear expression tree requires continuous nonlinear optimization"
            )
        elif family is ModelFamily.LEAST_SQUARES:
            algorithm = AlgorithmFamily.LEAST_SQUARES
            reason = "the declared family is a least-squares estimation problem"
        elif family in {ModelFamily.GRAPH, ModelFamily.NETWORK_FLOW}:
            algorithm = AlgorithmFamily.GRAPH_ALGORITHM
            reason = "the declared model is graph structured"
        elif family in {ModelFamily.SIMULATION, ModelFamily.DISCRETE_EVENT}:
            algorithm = AlgorithmFamily.SIMULATION
            reason = "the declared model requires simulation rather than deterministic optimization"
        elif family in {ModelFamily.STATISTICAL, ModelFamily.REGRESSION}:
            algorithm = AlgorithmFamily.STATISTICAL_ESTIMATION
            reason = (
                "the declared model is statistical rather than a supported optimization adapter"
            )
        else:
            algorithm = AlgorithmFamily.UNSUPPORTED
            reason = "Phase 4 has no deterministic adapter for the declared model family"

        declared_preference: tuple[SolverFamily, ...] = tuple(
            SolverFamily(item) for item in model.solver_requirements.preferred_solver_families
        )
        preference = tuple(
            dict.fromkeys([*declared_preference, *self._preferences.get(family, ())])
        )
        optimization_families = {
            ModelFamily.LINEAR_PROGRAMMING,
            ModelFamily.MIXED_INTEGER_LINEAR_PROGRAMMING,
            ModelFamily.INTEGER_PROGRAMMING,
            ModelFamily.NONLINEAR_PROGRAMMING,
            ModelFamily.MULTI_OBJECTIVE,
        }
        if family in optimization_families and model.objective is None:
            algorithm = AlgorithmFamily.UNSUPPORTED
            preference = ()
            reason = "an optimization model without an objective cannot be selected for solving"
        if family is ModelFamily.LINEAR_PROGRAMMING and any(
            domain not in {VariableDomain.CONTINUOUS, VariableDomain.NONNEGATIVE_CONTINUOUS}
            for domain in domains
        ):
            algorithm = AlgorithmFamily.UNSUPPORTED
            preference = ()
            reason = "an LP declaration cannot contain integer or binary variable domains"
        if family is ModelFamily.NONLINEAR_PROGRAMMING and any(
            domain
            in {
                VariableDomain.INTEGER,
                VariableDomain.NONNEGATIVE_INTEGER,
                VariableDomain.BINARY,
            }
            for domain in domains
        ):
            algorithm = AlgorithmFamily.UNSUPPORTED
            preference = ()
            reason = "Phase 4 has no deterministic mixed-integer nonlinear adapter"
        recommended = preference[0] if preference else None
        return AlgorithmPlan(
            algorithm_family=algorithm,
            recommended_solver_family=recommended,
            reason=reason,
            requirements=requirements,
            alternatives=list(preference[1:]),
            complexity_notes=[
                f"problem size={len(model.decision_variables)} variables/"
                f"{len(model.constraints)} constraints"
            ],
            numerical_risks=risks,
        )
