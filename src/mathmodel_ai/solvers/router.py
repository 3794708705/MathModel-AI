from __future__ import annotations

from dataclasses import dataclass

from mathmodel_ai.core.errors import SolverUnavailableError
from mathmodel_ai.mathematical.expressions import referenced_symbols
from mathmodel_ai.schemas.mathematical import MathematicalModel, VariableDomain
from mathmodel_ai.schemas.model_selection import ModelFamily
from mathmodel_ai.schemas.solver import (
    AlgorithmPlan,
    ProblemSize,
    ProblemSizeClass,
    ProblemSizeThresholds,
    SolverCapability,
    SolverFamily,
    SolverHardRejection,
    SolverName,
    SolverOptions,
    SolverRouteDecision,
    SolverScore,
)
from mathmodel_ai.solvers.base import BaseSolver


@dataclass(frozen=True)
class RoutedSolver:
    solver: BaseSolver
    decision: SolverRouteDecision
    options: SolverOptions


_SIZE_ADJUSTMENT: dict[SolverName, dict[ProblemSizeClass, float]] = {
    SolverName.SCIPY: {
        ProblemSizeClass.TINY: 8,
        ProblemSizeClass.SMALL: 4,
        ProblemSizeClass.MEDIUM: 0,
        ProblemSizeClass.LARGE: -10,
    },
    SolverName.ORTOOLS: {
        ProblemSizeClass.TINY: 6,
        ProblemSizeClass.SMALL: 4,
        ProblemSizeClass.MEDIUM: 1,
        ProblemSizeClass.LARGE: -5,
    },
    SolverName.GUROBI: {
        ProblemSizeClass.TINY: 0,
        ProblemSizeClass.SMALL: 2,
        ProblemSizeClass.MEDIUM: 8,
        ProblemSizeClass.LARGE: 15,
    },
}


class SolverRouter:
    def __init__(
        self,
        solvers: list[BaseSolver],
        *,
        size_thresholds: ProblemSizeThresholds | None = None,
        deadline_runtime_caps: dict[int, float] | None = None,
    ) -> None:
        self._by_family: dict[SolverFamily, BaseSolver] = {}
        for solver in solvers:
            for family in solver.families:
                if family in self._by_family:
                    raise ValueError(f"duplicate solver adapter for {family.value}")
                self._by_family[family] = solver
        self._thresholds = size_thresholds or ProblemSizeThresholds()
        self._deadline_runtime_caps = deadline_runtime_caps or {3: 300, 4: 120, 5: 60}

    def route(
        self,
        model: MathematicalModel,
        plan: AlgorithmPlan,
        options: SolverOptions,
    ) -> RoutedSolver:
        preference = self._preference(model, plan)
        problem_size = self.problem_size(model)
        resolved_options = self.resolve_options(model, options)
        runtime_budget = resolved_options.time_limit_seconds
        required = self._required_capabilities(model)
        scores: list[SolverScore] = []
        rejected: list[SolverHardRejection] = []
        deadline_weight = 1.0 + max(0, options.deadline_pressure - 2)

        for index, family in enumerate(preference):
            solver = self._by_family.get(family)
            if solver is None:
                rejected.append(
                    SolverHardRejection(family=family, reason="adapter is not registered")
                )
                continue
            if (
                not model.solver_requirements.allow_commercial_solver
                and solver.name is SolverName.GUROBI
            ):
                rejected.append(
                    SolverHardRejection(
                        family=family,
                        reason="model requirements disallow commercial solvers",
                    )
                )
                continue
            missing = required - solver.get_capabilities()
            if missing:
                rejected.append(
                    SolverHardRejection(
                        family=family,
                        reason="missing required capabilities: "
                        + ", ".join(sorted(item.value for item in missing)),
                    )
                )
                continue
            support = solver.assess_support(model)
            if not support.supported:
                rejected.append(
                    SolverHardRejection(family=family, reason="; ".join(support.reasons))
                )
                continue
            health = solver.health_check()
            if not health.available:
                rejected.append(SolverHardRejection(family=family, reason=health.reason))
                continue

            configured = (len(preference) - index) * 10 * deadline_weight
            size_adjustment = _SIZE_ADJUSTMENT[solver.name][problem_size.classification]
            declared_adjustment = self._declared_preference_adjustment(model, family)
            total = configured + size_adjustment + declared_adjustment
            scores.append(
                SolverScore(
                    family=family,
                    solver=solver.name,
                    score=total,
                    reasons=[
                        f"configured preference score={configured:g}",
                        f"configured {problem_size.classification.value} "
                        f"suitability={size_adjustment:g}",
                        f"model-declared preference={declared_adjustment:g}",
                        f"deadline pressure={options.deadline_pressure}",
                    ],
                )
            )

        if not scores:
            details = " | ".join(f"{item.family.value}: {item.reason}" for item in rejected)
            raise SolverUnavailableError(
                "SOLVER_UNAVAILABLE: " + (details or "no solver preference exists")
            )
        ranked = sorted(
            scores,
            key=lambda item: (-item.score, preference.index(item.family), item.family.value),
        )
        selected_score = ranked[0]
        solver = self._by_family[selected_score.family]
        decision = SolverRouteDecision(
            selected_solver=solver.name,
            selected_family=selected_score.family,
            attempted_families=preference,
            alternatives=[item.family for item in ranked[1:]],
            scores=ranked,
            hard_rejections=rejected,
            problem_size=problem_size,
            runtime_budget_seconds=runtime_budget,
            fallback_used=selected_score.family != preference[0],
            reason=(
                f"selected {selected_score.family.value} after capability/availability hard "
                f"filters and policy scoring; size={problem_size.classification.value}, "
                f"deadline_pressure={options.deadline_pressure}, runtime_budget={runtime_budget}"
            ),
        )
        return RoutedSolver(solver=solver, decision=decision, options=resolved_options)

    def resolve_options(
        self,
        model: MathematicalModel,
        options: SolverOptions,
    ) -> SolverOptions:
        return options.model_copy(
            update={"time_limit_seconds": self._runtime_budget(model, options)}
        )

    def generated_decision(
        self,
        model: MathematicalModel,
        options: SolverOptions,
        *,
        solver_name: SolverName,
        target: str,
    ) -> SolverRouteDecision:
        try:
            family = SolverFamily(target)
        except ValueError:
            family = {
                SolverName.GUROBI: SolverFamily.GUROBI,
                SolverName.ORTOOLS: SolverFamily.ORTOOLS_CP_SAT,
                SolverName.SCIPY: {
                    "linear_programming": SolverFamily.SCIPY_HIGHS,
                    "mixed_integer_linear_programming": SolverFamily.SCIPY_MILP,
                    "integer_programming": SolverFamily.SCIPY_MILP,
                    "nonlinear_programming": SolverFamily.SCIPY_MINIMIZE,
                }.get(model.model_family.value, SolverFamily.SCIPY_MINIMIZE),
            }[solver_name]
        resolved = self.resolve_options(model, options)
        return SolverRouteDecision(
            selected_solver=solver_name,
            selected_family=family,
            attempted_families=[family],
            scores=[
                SolverScore(
                    family=family,
                    solver=solver_name,
                    score=0,
                    reasons=["CodeAgent generated-program execution was explicitly selected"],
                )
            ],
            problem_size=self.problem_size(model),
            runtime_budget_seconds=resolved.time_limit_seconds,
            fallback_used=False,
            reason="generated program declared and executed this solver target",
        )

    @staticmethod
    def _preference(model: MathematicalModel, plan: AlgorithmPlan) -> list[SolverFamily]:
        declared: list[SolverFamily] = []
        for raw in model.solver_requirements.preferred_solver_families:
            try:
                declared.append(SolverFamily(raw))
            except ValueError as exc:
                raise SolverUnavailableError(
                    f"SOLVER_UNAVAILABLE: unknown preferred solver family {raw!r}"
                ) from exc
        planned = [
            *(
                [plan.recommended_solver_family]
                if plan.recommended_solver_family is not None
                else []
            ),
            *plan.alternatives,
        ]
        return list(dict.fromkeys([*declared, *planned]))

    @staticmethod
    def _declared_preference_adjustment(
        model: MathematicalModel,
        family: SolverFamily,
    ) -> float:
        declared = model.solver_requirements.preferred_solver_families
        if family.value not in declared:
            return 0
        return float((len(declared) - declared.index(family.value)) * 20)

    @staticmethod
    def _required_capabilities(model: MathematicalModel) -> frozenset[SolverCapability]:
        required: set[SolverCapability] = set()
        requirements = model.solver_requirements
        family_capability = {
            ModelFamily.LINEAR_PROGRAMMING: SolverCapability.LP,
            ModelFamily.MIXED_INTEGER_LINEAR_PROGRAMMING: SolverCapability.MILP,
            ModelFamily.INTEGER_PROGRAMMING: SolverCapability.INTEGER,
            ModelFamily.NONLINEAR_PROGRAMMING: SolverCapability.NLP,
            ModelFamily.MULTI_OBJECTIVE: SolverCapability.MULTIOBJECTIVE,
        }.get(model.model_family)
        if family_capability is not None:
            required.add(family_capability)
        for raw in requirements.required_capabilities:
            try:
                required.add(SolverCapability(raw))
            except ValueError as exc:
                raise SolverUnavailableError(
                    f"SOLVER_UNAVAILABLE: unknown required capability {raw!r}"
                ) from exc
        if requirements.requires_continuous:
            required.add(SolverCapability.CONTINUOUS)
        if requirements.requires_integer:
            required.add(SolverCapability.INTEGER)
        if requirements.requires_binary:
            required.add(SolverCapability.BINARY)
        if requirements.requires_nonlinear:
            required.add(SolverCapability.NLP)
        if requirements.requires_multiobjective:
            required.add(SolverCapability.MULTIOBJECTIVE)
        return frozenset(required)

    def _runtime_budget(
        self,
        model: MathematicalModel,
        options: SolverOptions,
    ) -> float | None:
        candidates = [
            item
            for item in (
                options.time_limit_seconds,
                model.solver_requirements.maximum_runtime_seconds,
            )
            if item is not None
        ]
        candidates.extend(
            cap
            for pressure, cap in self._deadline_runtime_caps.items()
            if options.deadline_pressure >= pressure
        )
        return min(candidates) if candidates else None

    def problem_size(self, model: MathematicalModel) -> ProblemSize:
        variables = model.decision_variables
        variable_symbols = {item.symbol for item in variables}
        expressions = []
        if model.objective is not None:
            expressions.append(model.objective.expression)
        for constraint in model.constraints:
            expressions.extend([constraint.expression, constraint.rhs])
        nonzeros = sum(
            len(referenced_symbols(expression) & variable_symbols) for expression in expressions
        )
        metrics = (len(variables), len(model.constraints), nonzeros)
        if self._fits(metrics, "tiny"):
            classification = ProblemSizeClass.TINY
        elif self._fits(metrics, "small"):
            classification = ProblemSizeClass.SMALL
        elif self._fits(metrics, "medium"):
            classification = ProblemSizeClass.MEDIUM
        else:
            classification = ProblemSizeClass.LARGE
        integer_domains = {
            VariableDomain.INTEGER,
            VariableDomain.NONNEGATIVE_INTEGER,
            VariableDomain.BINARY,
        }
        return ProblemSize(
            classification=classification,
            number_of_variables=len(variables),
            number_of_integer_variables=sum(item.domain in integer_domains for item in variables),
            number_of_binary_variables=sum(
                item.domain is VariableDomain.BINARY for item in variables
            ),
            number_of_constraints=len(model.constraints),
            number_of_nonzero_coefficients=nonzeros,
        )

    def _fits(self, metrics: tuple[int, int, int], size: str) -> bool:
        names = ("variables", "constraints", "nonzeros")
        return all(
            value <= getattr(self._thresholds, f"{size}_{name}")
            for value, name in zip(metrics, names, strict=True)
        )
