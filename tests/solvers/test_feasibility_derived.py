from mathmodel_ai.schemas.mathematical import (
    ConstraintDefinition,
    ConstraintRelation,
    EquationDefinition,
    UnitCheckStatus,
    VariableRole,
)
from mathmodel_ai.solvers.feasibility import check_feasibility
from tests.mathematical.helpers import EVIDENCE, add, constant, lp_model, symbol, variable


def _model_with_derived_constraint():
    model = lp_model()
    derived = variable("z").model_copy(update={"role": VariableRole.DERIVED})
    equation = EquationDefinition(
        equation_id="EQ-derived",
        latex="z=x+1",
        normalized_expression="z = x + 1",
        lhs=symbol("z"),
        rhs=add(symbol("x"), constant(1)),
        meaning="A scalar derived from the candidate decision value.",
        source_refs=[EVIDENCE],
        derivation="Exact algebraic definition.",
        symbol_refs=["x", "z"],
        dimension_status=UnitCheckStatus.PASS,
    )
    constraint = ConstraintDefinition(
        constraint_id="CON-derived",
        name="derived scalar identity",
        expression=symbol("z"),
        relation=ConstraintRelation.EQ,
        rhs=add(symbol("x"), constant(1)),
        normalized_expression="z = x + 1",
        description="The definition remains a checked hard constraint.",
        source_refs=[EVIDENCE],
        equation_ref="EQ-derived",
    )
    return model.model_copy(
        update={
            "derived_variables": [derived],
            "equations": [*model.equations, equation],
            "constraints": [*model.constraints, constraint],
        }
    )


def test_unique_scalar_definition_is_checked_without_reported_derived_value() -> None:
    report = check_feasibility(_model_with_derived_constraint(), {"x": 2, "y": 8}, tolerance=1e-7)
    assert report.violated_constraints == []
    assert report.bound_violations == []


def test_reported_derived_value_cannot_override_formal_definition() -> None:
    report = check_feasibility(
        _model_with_derived_constraint(), {"x": 2, "y": 8, "z": 9}, tolerance=1e-7
    )
    assert report.bound_violations == ["VAR-z:derived_mismatch"]
    assert report.max_constraint_violation == 6


def test_ambiguous_scalar_definition_remains_unchecked() -> None:
    model = _model_with_derived_constraint()
    duplicate = model.equations[-1].model_copy(update={"equation_id": "EQ-duplicate"})
    ambiguous = model.model_copy(update={"equations": [*model.equations, duplicate]})
    report = check_feasibility(ambiguous, {"x": 2, "y": 8, "z": 3}, tolerance=1e-7)
    assert report.violated_constraints == ["CON-derived:unchecked"]
