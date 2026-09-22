from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from mathmodel_ai.mathematical.digests import mathematical_model_digest
from mathmodel_ai.mathematical.quality_gates import model_quality_gate, solve_quality_gate
from mathmodel_ai.mathematical.registry import (
    EquationRegistry,
    ParameterRegistry,
    RegistryIssueCode,
    SymbolRegistry,
)
from mathmodel_ai.mathematical.units import (
    UnitChecker,
    multiply_units,
    parse_unit,
    same_dimension,
)
from mathmodel_ai.schemas.execution import (
    ExecutionOrigin,
    ExecutionRecord,
    ExecutionStatus,
    SandboxLimits,
)
from mathmodel_ai.schemas.mathematical import (
    BaseDimension,
    ConstantDefinition,
    ExpressionKind,
    IndexDefinition,
    MathExpression,
    ParameterDefinition,
    ParameterSourceType,
    SetDefinition,
    UnitCheckStatus,
    UnitExpression,
)
from mathmodel_ai.schemas.quality import QualityGateStatus
from mathmodel_ai.schemas.results import ResultRecord
from mathmodel_ai.schemas.solver import (
    FeasibilityReport,
    ProblemSize,
    ProblemSizeClass,
    SolverFamily,
    SolverName,
    SolverOptions,
    SolverResult,
    SolverRouteDecision,
    SolverRun,
    SolverStatus,
)
from tests.mathematical.helpers import add, lp_model, selected_state, symbol


def test_mathematical_model_round_trips_with_typed_expression_tree() -> None:
    model = lp_model()
    restored = type(model).model_validate_json(model.model_dump_json())

    assert restored == model
    assert restored.objective is not None
    assert restored.objective.expression.operands[0].kind.value == "MULTIPLY"
    assert restored.version == 1


def test_constant_identity_and_equation_dependency_namespaces_remain_distinct() -> None:
    constant = ConstantDefinition(
        parameter_id="CONST-scale",
        symbol="scale",
        description="A dimensionless normalization constant",
        value=1,
        unit=UnitExpression(),
        source_type=ParameterSourceType.DERIVATION,
        source_ref="EVID-fact-1",
        confidence=1,
    )
    model = lp_model().model_copy(update={"constants": [constant]})
    assert SymbolRegistry.from_model(model).lookup("scale") is not None
    assert ParameterRegistry.from_model(model).lookup("scale") == constant
    assert EquationRegistry.from_model(model).report.valid

    wrong_fields = constant.model_dump()
    wrong_fields["constant_id"] = wrong_fields.pop("parameter_id")
    with pytest.raises(ValidationError) as error:
        ConstantDefinition.model_validate(wrong_fields)
    assert {item["type"] for item in error.value.errors()} == {"extra_forbidden", "missing"}

    bad_equation = model.equations[0].model_copy(
        update={"dependency_refs": [constant.parameter_id]}
    )
    invalid = model.model_copy(update={"equations": [bad_equation, *model.equations[1:]]})
    report = EquationRegistry.from_model(invalid).report
    assert not report.valid
    assert any(
        issue.code is RegistryIssueCode.UNDEFINED_EQUATION_DEPENDENCY
        and issue.reference == "CONST-scale"
        for issue in report.issues
    )
    valid_equation = bad_equation.model_copy(
        update={
            "parameter_refs": [constant.parameter_id],
            "dependency_refs": [model.equations[1].equation_id],
        }
    )
    repaired = model.model_copy(update={"equations": [valid_equation, *model.equations[1:]]})
    assert EquationRegistry.from_model(repaired).report.valid


def test_symbol_registry_detects_undefined_and_parameter_variable_conflict() -> None:
    model = lp_model()
    assert model.objective is not None
    undefined_objective = model.objective.model_copy(update={"expression": symbol("z")})
    undefined = model.model_copy(update={"objective": undefined_objective})
    undefined_report = SymbolRegistry.from_model(undefined).report
    assert any(
        issue.code is RegistryIssueCode.UNDEFINED_SYMBOL and issue.reference == "z"
        for issue in undefined_report.issues
    )

    conflicting = model.model_copy(
        update={
            "parameters": [
                ParameterDefinition(
                    parameter_id="PAR-x",
                    symbol="x",
                    description="conflicting demand parameter",
                    value=1,
                    unit=UnitExpression(),
                    source_type=ParameterSourceType.PROBLEM_FACT,
                    source_ref="EVID-fact-1",
                    confidence=1,
                )
            ]
        }
    )
    conflict_report = SymbolRegistry.from_model(conflicting).report
    assert any(
        issue.code is RegistryIssueCode.CONFLICTING_DEFINITION and issue.reference == "x"
        for issue in conflict_report.issues
    )


def test_symbol_registry_accepts_declared_indexed_symbol() -> None:
    model = lp_model()
    indexed_variable = model.decision_variables[0].model_copy(update={"index_sets": ["SET-I"]})
    assert model.objective is not None
    objective = model.objective.model_copy(update={"expression": symbol("x[i]")})
    indexed = model.model_copy(
        update={
            "sets": [
                SetDefinition(
                    set_id="SET-I",
                    symbol="I",
                    description="fixture index set",
                    values=[1, 2],
                )
            ],
            "indices": [
                IndexDefinition(
                    index_id="IDX-i",
                    symbol="i",
                    description="fixture index",
                    set_ref="SET-I",
                )
            ],
            "decision_variables": [indexed_variable, model.decision_variables[1]],
            "objective": objective,
        }
    )
    registry = SymbolRegistry.from_model(indexed)

    assert registry.lookup("x[i]") is not None
    assert not any(
        issue.code is RegistryIssueCode.UNDEFINED_SYMBOL and issue.reference == "x[i]"
        for issue in registry.report.issues
    )


def test_parameter_registry_rejects_duplicate_symbol() -> None:
    model = lp_model()
    first = ParameterDefinition(
        parameter_id="PAR-demand-a",
        symbol="demand",
        description="first demand definition",
        value=10,
        source_type=ParameterSourceType.PROBLEM_FACT,
        source_ref="EVID-fact-1",
        confidence=1,
    )
    second = first.model_copy(
        update={
            "parameter_id": "PAR-demand-b",
            "description": "conflicting duplicate demand definition",
        }
    )
    registry = ParameterRegistry(model.model_id, model.version)
    registry.register(first)
    registry.register(second)

    assert registry.lookup("demand") == first
    assert registry.report.issues[0].code is RegistryIssueCode.DUPLICATE_PARAMETER


def test_equation_registry_rejects_duplicate_in_same_model_version() -> None:
    model = lp_model()
    registry = EquationRegistry(model.model_id, model.version)
    equation = model.equations[0]
    registry.register(equation)
    registry.register(equation)

    assert registry.lookup(equation.equation_id) == equation
    assert registry.report.issues[0].code is RegistryIssueCode.DUPLICATE_EQUATION


def test_unit_parser_and_composition_are_dimensional_not_string_comparisons() -> None:
    mass_rate = parse_unit("kg/hour")
    assert mass_rate.dimensions == {
        BaseDimension.MASS: 1,
        BaseDimension.TIME: -1,
    }
    rate_times_count = parse_unit("kg/hour*item")
    assert rate_times_count.dimensions == {
        BaseDimension.MASS: 1,
        BaseDimension.TIME: -1,
        BaseDimension.COUNT: 1,
    }

    count = parse_unit("件")
    price = parse_unit("元/件")
    currency = parse_unit("元")
    assert same_dimension(multiply_units(count, price), currency)

    unknown = parse_unit("custom_flux_unit")
    assert not unknown.is_known
    assert unknown.unknown_units == ["custom_flux_unit"]


def test_unit_checker_fails_incompatible_addition_and_equation() -> None:
    model = lp_model()
    mass = parse_unit("kg")
    time = parse_unit("second")
    variables = [
        model.decision_variables[0].model_copy(update={"unit": mass}),
        model.decision_variables[1].model_copy(update={"unit": time}),
    ]
    assert model.objective is not None
    objective = model.objective.model_copy(
        update={"expression": add(symbol("x"), symbol("y")), "unit": mass}
    )
    equation = model.equations[0].model_copy(update={"lhs": symbol("x"), "rhs": symbol("y")})
    incompatible = model.model_copy(
        update={
            "decision_variables": variables,
            "objective": objective,
            "equations": [equation, *model.equations[1:]],
        }
    )

    report = UnitChecker(SymbolRegistry.from_model(incompatible)).check_model(incompatible)
    assert report.status is UnitCheckStatus.FAIL
    assert any("dimension mismatch" in issue.message for issue in report.issues)


def test_unit_checker_returns_unknown_for_unrecognized_custom_unit() -> None:
    model = lp_model()
    custom = parse_unit("custom_flux_unit")
    variables = [item.model_copy(update={"unit": custom}) for item in model.decision_variables]
    unknown_model = model.model_copy(update={"decision_variables": variables})

    report = UnitChecker(SymbolRegistry.from_model(unknown_model)).check_model(unknown_model)

    assert report.status is UnitCheckStatus.UNKNOWN
    assert any("UNIT_CHECK_REQUIRED" in issue.message for issue in report.issues)


def test_dimensionless_base_accepts_dimensionless_variable_exponent() -> None:
    model = lp_model()
    expression = MathExpression(
        kind=ExpressionKind.POWER,
        operands=[symbol("x"), symbol("y")],
    )
    assert model.objective is not None
    objective = model.objective.model_copy(update={"expression": expression})
    equation = model.equations[0].model_copy(update={"lhs": expression, "rhs": expression})
    powered = model.model_copy(
        update={"objective": objective, "equations": [equation, *model.equations[1:]]}
    )

    report = UnitChecker(SymbolRegistry.from_model(powered)).check_model(powered)

    assert report.status is UnitCheckStatus.PASS
    assert report.issues == []


def test_model_gate_accepts_valid_lp_and_rejects_required_failures() -> None:
    state = selected_state()
    model = lp_model(
        project_id=state.project_id,
        problem_id=state.problem_id,
        source_selected_model_id="CAND-lp",
    )
    assert model_quality_gate(model, state).status is QualityGateStatus.PASS

    missing_objective = model.model_copy(update={"objective": None})
    assert model_quality_gate(missing_objective, state).status is QualityGateStatus.RETRY

    assert model.objective is not None
    undefined = model.model_copy(
        update={"objective": model.objective.model_copy(update={"expression": symbol("z")})}
    )
    undefined_gate = model_quality_gate(undefined, state)
    assert undefined_gate.status is QualityGateStatus.RETRY
    assert any("UNDEFINED_SYMBOL:z" in error for error in undefined_gate.errors)

    conflict_model = model.model_copy(
        update={
            "decision_variables": [
                model.decision_variables[0].model_copy(update={"unit": parse_unit("kg")}),
                model.decision_variables[1].model_copy(update={"unit": parse_unit("hour")}),
            ],
            "objective": model.objective.model_copy(
                update={"expression": add(symbol("x"), symbol("y"))}
            ),
        }
    )
    assert model_quality_gate(conflict_model, state).status is QualityGateStatus.RETRY


def test_model_gate_blocks_unresolved_critical_ambiguity() -> None:
    state = selected_state(low_confidence_ambiguity=True)
    model = lp_model(
        project_id=state.project_id,
        problem_id=state.problem_id,
        source_selected_model_id="CAND-lp",
    ).model_copy(update={"target_subproblems": ["Q1"]})

    gate = model_quality_gate(model, state)
    assert gate.status is QualityGateStatus.RETRY
    assert any("UNRESOLVED_AMBIGUITY:AMB-demand" in error for error in gate.errors)


def test_result_record_rejects_broken_evidence_chain() -> None:
    result_id = uuid4()
    model_id = uuid4()
    with pytest.raises(ValidationError, match="model, solver run, and execution"):
        ResultRecord(
            result_id=result_id,
            project_id=uuid4(),
            problem_id=uuid4(),
            model_id=model_id,
            model_version=1,
            model_digest="0" * 64,
            solver_run_id=uuid4(),
            execution_record_id=uuid4(),
            solver=SolverName.SCIPY,
            status=SolverStatus.OPTIMAL,
            evidence_refs=["model:wrong:v1", "solver_run:wrong", "execution:wrong"],
        )


def test_solve_gate_rejects_constraint_violation_and_failed_execution() -> None:
    model = lp_model()
    execution_id = uuid4()
    solver_run_id = uuid4()
    result_id = uuid4()
    digest = mathematical_model_digest(model)
    feasibility = FeasibilityReport(
        checked=True,
        max_constraint_violation=1,
        violated_constraints=["CON-1"],
        tolerance=1e-7,
        message="fixture violation",
    )
    solver_result = SolverResult(
        solver_name=SolverName.SCIPY,
        status=SolverStatus.FEASIBLE,
        objective_value=27,
        variable_values={"x": 9, "y": 0},
        runtime_seconds=0.01,
        is_feasible=True,
        execution_record_id=execution_id,
        feasibility=feasibility,
    )
    result = ResultRecord(
        result_id=result_id,
        project_id=model.project_id,
        problem_id=model.problem_id,
        model_id=model.model_id,
        model_version=model.version,
        model_digest=digest,
        solver_run_id=solver_run_id,
        execution_record_id=execution_id,
        solver=SolverName.SCIPY,
        objective=27,
        key_outputs={"x": 9, "y": 0},
        status=SolverStatus.FEASIBLE,
        evidence_refs=[
            f"model:{model.model_id}:v{model.version}",
            f"solver_run:{solver_run_id}",
            f"execution:{execution_id}",
        ],
    )
    now = datetime.now(UTC)
    route = SolverRouteDecision(
        selected_solver=SolverName.SCIPY,
        selected_family=SolverFamily.SCIPY_HIGHS,
        attempted_families=[SolverFamily.SCIPY_HIGHS],
        problem_size=ProblemSize(
            classification=ProblemSizeClass.TINY,
            number_of_variables=2,
            number_of_integer_variables=0,
            number_of_binary_variables=0,
            number_of_constraints=1,
            number_of_nonzero_coefficients=2,
        ),
        fallback_used=False,
        reason="unit-test route",
    )
    solver_run = SolverRun(
        solver_run_id=solver_run_id,
        project_id=model.project_id,
        problem_id=model.problem_id,
        model_id=model.model_id,
        model_version=model.version,
        model_digest=digest,
        execution_origin=ExecutionOrigin.DETERMINISTIC_SOLVER_ADAPTER,
        routing_decision_id=route.routing_decision_id,
        routing_decision=route,
        solver=SolverName.SCIPY,
        options=SolverOptions(),
        start_time=now,
        end_time=now,
        status=SolverStatus.FEASIBLE,
        objective=27,
        runtime_seconds=0.01,
        result_ref=result_id,
        execution_ref=execution_id,
        result=solver_result,
    )
    execution = ExecutionRecord(
        run_id=execution_id,
        project_id=model.project_id,
        problem_id=model.problem_id,
        code_hash="0" * 64,
        code_artifact_id=uuid4(),
        execution_origin=ExecutionOrigin.DETERMINISTIC_SOLVER_ADAPTER,
        model_digest=digest,
        image="solver:test",
        end_time=now,
        runtime_seconds=0.01,
        status=ExecutionStatus.FAILED,
        exit_code=1,
        limits=SandboxLimits(
            cpu_cores=1,
            memory_mb=64,
            timeout_seconds=1,
            pids_limit=16,
            max_output_bytes=1024,
            max_artifacts=1,
            max_artifact_bytes=1024,
        ),
    )

    gate = solve_quality_gate(model, result, solver_run, execution)

    assert gate.status is QualityGateStatus.RETRY
    assert "SOLVE_GATE_FAIL:failure_not_wrapped_as_success" in gate.errors
    assert "SOLVE_GATE_FAIL:implemented_feasibility_passes" in gate.errors
