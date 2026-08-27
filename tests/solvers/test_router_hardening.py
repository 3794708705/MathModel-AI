from __future__ import annotations

from mathmodel_ai.core.errors import SolverUnavailableError
from mathmodel_ai.mathematical.algorithms import AlgorithmSelector
from mathmodel_ai.schemas.mathematical import MathematicalModel
from mathmodel_ai.schemas.solver import (
    ProblemSizeClass,
    ProblemSizeThresholds,
    SolverCapability,
    SolverFamily,
    SolverHealth,
    SolverName,
    SolverOptions,
    SupportAssessment,
)
from mathmodel_ai.solvers.base import BaseSolver, SolverExecution
from mathmodel_ai.solvers.router import SolverRouter
from tests.mathematical.helpers import lp_model, milp_model


class PolicySolver(BaseSolver):
    def __init__(
        self,
        *,
        name: SolverName,
        families: frozenset[SolverFamily],
        capabilities: frozenset[SolverCapability],
        available: bool = True,
        supported: bool = True,
    ) -> None:
        self.name = name
        self.families = families
        self.capabilities = capabilities
        self._available = available
        self._supported = supported

    def assess_support(self, model: MathematicalModel) -> SupportAssessment:
        del model
        return SupportAssessment(
            supported=self._supported,
            reasons=[] if self._supported else ["fixture translation is unsupported"],
        )

    def health_check(self) -> SolverHealth:
        return SolverHealth(
            solver_name=self.name,
            available=self._available,
            version="fixture-1" if self._available else None,
            reason="fixture available" if self._available else "fixture license unavailable",
        )

    def solve(self, model: MathematicalModel, options: SolverOptions) -> SolverExecution:
        del model, options
        raise AssertionError("routing policy tests must not execute a solver")


def _scipy(*, available: bool = True) -> PolicySolver:
    return PolicySolver(
        name=SolverName.SCIPY,
        families=frozenset(
            {
                SolverFamily.SCIPY_HIGHS,
                SolverFamily.SCIPY_MILP,
                SolverFamily.SCIPY_MINIMIZE,
            }
        ),
        capabilities=frozenset(
            {
                SolverCapability.LP,
                SolverCapability.MILP,
                SolverCapability.INTEGER,
                SolverCapability.BINARY,
                SolverCapability.CONTINUOUS,
                SolverCapability.NLP,
            }
        ),
        available=available,
    )


def _gurobi(*, available: bool = True) -> PolicySolver:
    return PolicySolver(
        name=SolverName.GUROBI,
        families=frozenset({SolverFamily.GUROBI}),
        capabilities=frozenset(
            {
                SolverCapability.LP,
                SolverCapability.MILP,
                SolverCapability.INTEGER,
                SolverCapability.BINARY,
                SolverCapability.CONTINUOUS,
            }
        ),
        available=available,
    )


def _ortools(*, available: bool = True) -> PolicySolver:
    return PolicySolver(
        name=SolverName.ORTOOLS,
        families=frozenset({SolverFamily.ORTOOLS_CP_SAT}),
        capabilities=frozenset(
            {
                SolverCapability.MILP,
                SolverCapability.INTEGER,
                SolverCapability.BINARY,
            }
        ),
        available=available,
    )


def test_problem_size_changes_policy_selection() -> None:
    model = lp_model()
    plan = AlgorithmSelector().select(model)
    tiny_route = SolverRouter([_scipy(), _gurobi()]).route(
        model,
        plan,
        SolverOptions(),
    )
    large_thresholds = ProblemSizeThresholds(
        tiny_variables=1,
        tiny_constraints=1,
        tiny_nonzeros=1,
        small_variables=2,
        small_constraints=2,
        small_nonzeros=2,
        medium_variables=3,
        medium_constraints=3,
        medium_nonzeros=3,
    )
    large_route = SolverRouter([_scipy(), _gurobi()], size_thresholds=large_thresholds).route(
        model, plan, SolverOptions()
    )

    assert tiny_route.decision.problem_size.classification is ProblemSizeClass.TINY
    assert tiny_route.decision.selected_solver is SolverName.SCIPY
    assert large_route.decision.problem_size.classification is ProblemSizeClass.LARGE
    assert large_route.decision.selected_solver is SolverName.GUROBI
    assert large_route.decision.scores[0].score > large_route.decision.scores[1].score


def test_deadline_and_model_runtime_caps_reach_resolved_solver_options() -> None:
    model = lp_model()
    requirements = model.solver_requirements.model_copy(update={"maximum_runtime_seconds": 180})
    model = model.model_copy(update={"solver_requirements": requirements})
    router = SolverRouter([_scipy(), _gurobi()], deadline_runtime_caps={3: 90, 4: 45, 5: 15})

    route = router.route(
        model,
        AlgorithmSelector().select(model),
        SolverOptions(time_limit_seconds=240, deadline_pressure=4),
    )

    assert route.options.time_limit_seconds == 45
    assert route.decision.runtime_budget_seconds == 45
    assert any("deadline pressure=4" in reason for reason in route.decision.scores[0].reasons)


def test_deadline_pressure_can_shift_selection_toward_configured_primary() -> None:
    model = lp_model()
    plan = AlgorithmSelector().select(model)
    thresholds = ProblemSizeThresholds(
        tiny_variables=1,
        tiny_constraints=1,
        tiny_nonzeros=1,
        small_variables=2,
        small_constraints=2,
        small_nonzeros=2,
        medium_variables=3,
        medium_constraints=3,
        medium_nonzeros=3,
    )
    router = SolverRouter([_scipy(), _gurobi()], size_thresholds=thresholds)

    normal = router.route(model, plan, SolverOptions(deadline_pressure=0))
    urgent = router.route(model, plan, SolverOptions(deadline_pressure=4))

    assert normal.decision.selected_solver is SolverName.GUROBI
    assert urgent.decision.selected_solver is SolverName.SCIPY
    assert urgent.options.time_limit_seconds == 120


def test_integer_route_records_unavailable_preference_and_falls_back() -> None:
    model = milp_model()
    route = SolverRouter([_gurobi(available=False), _scipy(), _ortools()]).route(
        model,
        AlgorithmSelector().select(model),
        SolverOptions(),
    )

    assert route.decision.selected_solver is SolverName.SCIPY
    assert route.decision.fallback_used is True
    assert route.decision.hard_rejections[0].family is SolverFamily.GUROBI
    assert "license unavailable" in route.decision.hard_rejections[0].reason


def test_user_preference_and_commercial_license_policy_are_hard_constraints() -> None:
    model = milp_model()
    requirements = model.solver_requirements.model_copy(
        update={
            "preferred_solver_families": ["GUROBI", "ORTOOLS_CP_SAT"],
            "allow_commercial_solver": False,
        }
    )
    model = model.model_copy(update={"solver_requirements": requirements})

    route = SolverRouter([_gurobi(), _ortools(), _scipy()]).route(
        model,
        AlgorithmSelector().select(model),
        SolverOptions(),
    )

    assert route.decision.selected_solver is SolverName.ORTOOLS
    assert any(
        rejection.family is SolverFamily.GUROBI and "disallow commercial" in rejection.reason
        for rejection in route.decision.hard_rejections
    )


def test_unsupported_required_capability_fails_with_auditable_rejections() -> None:
    model = lp_model()
    requirements = model.solver_requirements.model_copy(
        update={"required_capabilities": ["MULTIOBJECTIVE"]}
    )
    model = model.model_copy(update={"solver_requirements": requirements})

    try:
        SolverRouter([_scipy(), _gurobi()]).route(
            model,
            AlgorithmSelector().select(model),
            SolverOptions(),
        )
    except SolverUnavailableError as exc:
        message = str(exc)
    else:
        raise AssertionError("unsupported capability must not resolve a solver")

    assert "MULTIOBJECTIVE" in message
    assert "SCIPY_HIGHS" in message
    assert "GUROBI" in message
