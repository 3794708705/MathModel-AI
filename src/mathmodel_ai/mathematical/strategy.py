from __future__ import annotations

from dataclasses import dataclass

from mathmodel_ai.core.errors import SolverUnavailableError
from mathmodel_ai.schemas.mathematical import MathematicalModel
from mathmodel_ai.schemas.program import (
    ExecutionStrategy,
    ExecutionStrategyDecision,
)
from mathmodel_ai.schemas.solver import AlgorithmPlan, SolverOptions
from mathmodel_ai.solvers.router import RoutedSolver, SolverRouter


@dataclass(frozen=True)
class SelectedExecutionStrategy:
    decision: ExecutionStrategyDecision
    routed_solver: RoutedSolver | None
    options: SolverOptions


class ExecutionStrategySelector:
    """Select deterministic execution when it can faithfully consume the model."""

    def __init__(self, solver_router: SolverRouter) -> None:
        self._solver_router = solver_router

    def select(
        self,
        *,
        requested: ExecutionStrategy,
        model: MathematicalModel,
        plan: AlgorithmPlan,
        options: SolverOptions,
        solver_router: SolverRouter | None = None,
    ) -> SelectedExecutionStrategy:
        router = solver_router or self._solver_router
        if requested is ExecutionStrategy.GENERATED:
            return SelectedExecutionStrategy(
                decision=ExecutionStrategyDecision(
                    requested=requested,
                    selected=ExecutionStrategy.GENERATED,
                    reason="caller explicitly requested the generated-program strategy",
                ),
                routed_solver=None,
                options=router.resolve_options(model, options),
            )
        try:
            routed = router.route(model, plan, options)
        except SolverUnavailableError:
            if requested is ExecutionStrategy.DETERMINISTIC:
                raise
            return SelectedExecutionStrategy(
                decision=ExecutionStrategyDecision(
                    requested=requested,
                    selected=ExecutionStrategy.GENERATED,
                    reason="no available deterministic adapter can fully consume the model",
                ),
                routed_solver=None,
                options=router.resolve_options(model, options),
            )
        return SelectedExecutionStrategy(
            decision=ExecutionStrategyDecision(
                requested=requested,
                selected=ExecutionStrategy.DETERMINISTIC,
                reason="a compatible deterministic adapter is available and preferred",
            ),
            routed_solver=routed,
            options=routed.options,
        )
