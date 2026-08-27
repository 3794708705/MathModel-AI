from __future__ import annotations

from uuid import UUID, uuid4

from mathmodel_ai.mathematical.expressions import referenced_symbols
from mathmodel_ai.schemas.mathematical import (
    AlgorithmRequirements,
    ConstraintDefinition,
    ConstraintRelation,
    ConvexityStatus,
    EquationDefinition,
    ExpectedOutput,
    ExpressionKind,
    MathematicalModel,
    MathematicalModelStatus,
    MathExpression,
    ObjectiveDefinition,
    ObjectiveSense,
    SolverRequirements,
    UnitCheckStatus,
    UnitExpression,
    VariableDefinition,
    VariableDomain,
    VariableRole,
)
from mathmodel_ai.schemas.model_selection import ModelFamily
from mathmodel_ai.schemas.problem_state import ProblemState, WorkflowStage, WorkflowStatus
from tests.reasoning.helpers import analysis_fixture, candidate_fixture

EVIDENCE = "EVID-fact-1"


def constant(value: float) -> MathExpression:
    return MathExpression(kind=ExpressionKind.CONSTANT, value=value)


def symbol(name: str) -> MathExpression:
    return MathExpression(kind=ExpressionKind.SYMBOL, symbol=name)


def add(*operands: MathExpression) -> MathExpression:
    return MathExpression(kind=ExpressionKind.ADD, operands=list(operands))


def subtract(left: MathExpression, right: MathExpression) -> MathExpression:
    return MathExpression(kind=ExpressionKind.SUBTRACT, operands=[left, right])


def multiply(*operands: MathExpression) -> MathExpression:
    return MathExpression(kind=ExpressionKind.MULTIPLY, operands=list(operands))


def power(base: MathExpression, exponent: float) -> MathExpression:
    return MathExpression(kind=ExpressionKind.POWER, operands=[base, constant(exponent)])


def variable(
    name: str,
    *,
    domain: VariableDomain = VariableDomain.NONNEGATIVE_CONTINUOUS,
    lower: float | None = 0,
    upper: float | None = None,
    unit: UnitExpression | None = None,
) -> VariableDefinition:
    return VariableDefinition(
        variable_id=f"VAR-{name}",
        symbol=name,
        description=f"decision variable {name}",
        domain=domain,
        lower_bound=lower,
        upper_bound=upper,
        unit=unit or UnitExpression(),
        role=VariableRole.DECISION,
        source_refs=[EVIDENCE],
    )


def mathematical_model(
    *,
    family: ModelFamily,
    variables: list[VariableDefinition],
    objective_expression: MathExpression,
    constraints: list[tuple[MathExpression, ConstraintRelation, MathExpression]],
    sense: ObjectiveSense = ObjectiveSense.MINIMIZE,
    project_id: UUID | None = None,
    problem_id: UUID | None = None,
    model_id: UUID | None = None,
    source_selected_model_id: str = "CAND-lp",
    convexity: ConvexityStatus = ConvexityStatus.CONVEX,
) -> MathematicalModel:
    objective = ObjectiveDefinition(
        objective_id="OBJ-main",
        sense=sense,
        expression=objective_expression,
        description="deterministic fixture objective",
        unit=UnitExpression(),
        source_refs=[EVIDENCE],
        derivation="Fixture coefficients are stated mathematical inputs.",
        equation_ref="EQ-objective",
    )
    constraint_models = [
        ConstraintDefinition(
            constraint_id=f"CON-{index}",
            name=f"fixture constraint {index}",
            expression=left,
            relation=relation,
            rhs=right,
            normalized_expression=f"constraint-{index}",
            description=f"deterministic fixture constraint {index}",
            source_refs=[EVIDENCE],
            unit=UnitExpression(),
            equation_ref=f"EQ-constraint-{index}",
        )
        for index, (left, relation, right) in enumerate(constraints, start=1)
    ]
    equations = [
        EquationDefinition(
            equation_id="EQ-objective",
            latex="f(x)",
            normalized_expression="objective",
            lhs=objective_expression,
            rhs=objective_expression,
            meaning="objective definition",
            source_refs=[EVIDENCE],
            derivation="Directly from the objective definition.",
            symbol_refs=sorted(referenced_symbols(objective_expression)),
            dimension_status=UnitCheckStatus.PASS,
        ),
        *[
            EquationDefinition(
                equation_id=item.equation_ref,
                latex=item.normalized_expression,
                normalized_expression=item.normalized_expression,
                lhs=item.expression,
                rhs=item.rhs,
                meaning=item.description,
                source_refs=item.source_refs,
                derivation="Directly from the stated fixture constraint.",
                symbol_refs=sorted(
                    referenced_symbols(item.expression) | referenced_symbols(item.rhs)
                ),
                dimension_status=UnitCheckStatus.PASS,
            )
            for item in constraint_models
        ],
    ]
    domains = {item.domain for item in variables}
    return MathematicalModel(
        model_id=model_id or uuid4(),
        project_id=project_id or uuid4(),
        problem_id=problem_id or uuid4(),
        version=1,
        source_selected_model_id=source_selected_model_id,
        status=MathematicalModelStatus.READY,
        name=f"{family.value} fixture",
        description="A deterministic Phase 4 solver fixture.",
        target_subproblems=["Q2"],
        model_family=family,
        decision_variables=variables,
        objective=objective,
        constraints=constraint_models,
        equations=equations,
        algorithm_requirements=AlgorithmRequirements(
            exact_solution_required=family
            in {
                ModelFamily.LINEAR_PROGRAMMING,
                ModelFamily.MIXED_INTEGER_LINEAR_PROGRAMMING,
                ModelFamily.INTEGER_PROGRAMMING,
            },
            convexity=convexity,
        ),
        solver_requirements=SolverRequirements(
            requires_continuous=any(
                domain in {VariableDomain.CONTINUOUS, VariableDomain.NONNEGATIVE_CONTINUOUS}
                for domain in domains
            ),
            requires_integer=any(
                domain
                in {
                    VariableDomain.INTEGER,
                    VariableDomain.NONNEGATIVE_INTEGER,
                    VariableDomain.BINARY,
                }
                for domain in domains
            ),
            requires_binary=VariableDomain.BINARY in domains,
            requires_nonlinear=family is ModelFamily.NONLINEAR_PROGRAMMING,
        ),
        source_evidence=[EVIDENCE],
        expected_outputs=[
            ExpectedOutput(
                output_id="OUT-solution",
                name="solution",
                description="objective and decision variable values",
                unit=UnitExpression(),
                source_refs=[EVIDENCE],
            )
        ],
        validation_requirements=["recompute variable bounds and constraints"],
        limitations=["small deterministic fixture only"],
        confidence=1,
    )


def lp_model(**identities: object) -> MathematicalModel:
    return mathematical_model(
        family=ModelFamily.LINEAR_PROGRAMMING,
        variables=[variable("x"), variable("y")],
        objective_expression=add(
            multiply(constant(3), symbol("x")),
            multiply(constant(4), symbol("y")),
        ),
        constraints=[(add(symbol("x"), symbol("y")), ConstraintRelation.GE, constant(10))],
        **identities,
    )


def milp_model(**identities: object) -> MathematicalModel:
    return mathematical_model(
        family=ModelFamily.MIXED_INTEGER_LINEAR_PROGRAMMING,
        variables=[
            variable(
                "x",
                domain=VariableDomain.NONNEGATIVE_INTEGER,
                lower=0,
                upper=10,
            ),
            variable(
                "y",
                domain=VariableDomain.NONNEGATIVE_INTEGER,
                lower=0,
                upper=10,
            ),
        ],
        objective_expression=add(
            multiply(constant(10), symbol("x")),
            multiply(constant(6), symbol("y")),
        ),
        constraints=[
            (
                add(
                    multiply(constant(4), symbol("x")),
                    multiply(constant(3), symbol("y")),
                ),
                ConstraintRelation.LE,
                constant(6),
            )
        ],
        sense=ObjectiveSense.MAXIMIZE,
        **identities,
    )


def nlp_model(**identities: object) -> MathematicalModel:
    return mathematical_model(
        family=ModelFamily.NONLINEAR_PROGRAMMING,
        variables=[
            variable("x", domain=VariableDomain.CONTINUOUS, lower=-10, upper=10),
            variable("y", domain=VariableDomain.CONTINUOUS, lower=-10, upper=10),
        ],
        objective_expression=add(
            power(subtract(symbol("x"), constant(3)), 2),
            power(add(symbol("y"), constant(1)), 2),
        ),
        constraints=[(add(symbol("x"), symbol("y")), ConstraintRelation.GE, constant(0))],
        convexity=ConvexityStatus.CONVEX,
        **identities,
    )


def infeasible_model(**identities: object) -> MathematicalModel:
    return mathematical_model(
        family=ModelFamily.LINEAR_PROGRAMMING,
        variables=[variable("x")],
        objective_expression=symbol("x"),
        constraints=[
            (symbol("x"), ConstraintRelation.GE, constant(10)),
            (symbol("x"), ConstraintRelation.LE, constant(5)),
        ],
        **identities,
    )


def unbounded_model(**identities: object) -> MathematicalModel:
    return mathematical_model(
        family=ModelFamily.LINEAR_PROGRAMMING,
        variables=[variable("x")],
        objective_expression=symbol("x"),
        constraints=[(symbol("x"), ConstraintRelation.GE, constant(0))],
        sense=ObjectiveSense.MAXIMIZE,
        **identities,
    )


def selected_state(*, low_confidence_ambiguity: bool = False) -> ProblemState:
    analysis = analysis_fixture(low_confidence_ambiguity=low_confidence_ambiguity)
    candidate = candidate_fixture(
        "CAND-lp",
        name="Linear allocation",
        family=ModelFamily.LINEAR_PROGRAMMING,
        targets=["Q2"],
    )
    return ProblemState(
        project_id=uuid4(),
        title="Deterministic allocation fixture",
        raw_problem="Minimize allocation cost while satisfying the stated requirement.",
        current_stage=WorkflowStage.SELECT,
        status=WorkflowStatus.SUCCEEDED,
        problem_analysis=analysis,
        subproblems=analysis.subproblems,
        ambiguities=analysis.ambiguities,
        selected_model=candidate,
        candidate_models=[candidate],
    )
