from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from mathmodel_ai.mathematical.digests import mathematical_model_digest
from mathmodel_ai.mathematical.normalization import materialize_data_bound_scalars
from mathmodel_ai.mathematical.quality_gates import (
    _self_referential_data_residuals,
    _state_relations_sufficient,
    model_quality_gate,
    solve_quality_gate,
)
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
from mathmodel_ai.schemas.data import (
    ColumnProfile,
    DataProfile,
    DataSemanticType,
    DatasetRecord,
    NumericStatistics,
    ValueCount,
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
    ConstraintDefinition,
    ConstraintRelation,
    DataBinding,
    EquationDefinition,
    ExpressionKind,
    IndexDefinition,
    MathExpression,
    ParameterDefinition,
    ParameterSourceType,
    SetDefinition,
    UnitCheckStatus,
    UnitExpression,
    VariableRole,
)
from mathmodel_ai.schemas.model_selection import ModelFamily
from mathmodel_ai.schemas.problem_analysis import (
    EvidenceItem,
    EvidenceSource,
    EvidenceStatus,
    EvidenceType,
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
from tests.mathematical.helpers import (
    add,
    constant,
    lp_model,
    multiply,
    power,
    selected_state,
    subtract,
    symbol,
    variable,
)


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


def test_model_gate_rejects_three_latent_states_with_only_one_sum_relation() -> None:
    state = selected_state()
    base = lp_model(
        project_id=state.project_id,
        problem_id=state.problem_id,
        source_selected_model_id="CAND-lp",
    )
    latent = [
        variable(name).model_copy(update={"role": VariableRole.STATE})
        for name in ("cold", "neutral", "hot")
    ]
    sum_equation = base.equations[0].model_copy(
        update={
            "equation_id": "EQ-state-sum",
            "lhs": add(*(symbol(item.symbol) for item in latent)),
            "rhs": constant(1),
        }
    )
    underdetermined = base.model_copy(
        update={
            "state_variables": latent,
            "equations": [*base.equations, sum_equation],
        }
    )
    gate = model_quality_gate(underdetermined, state)
    assert "MODEL_GATE_FAIL:state_relations_sufficient" in gate.errors

    transitions = [
        sum_equation.model_copy(
            update={
                "equation_id": f"EQ-state-{item.symbol}",
                "lhs": symbol(item.symbol),
            }
        )
        for item in latent
    ]
    complete = underdetermined.model_copy(
        update={
            "equations": [*base.equations, *transitions],
        }
    )
    assert _state_relations_sufficient(complete)


def test_model_gate_rejects_unmaterialized_data_bound_constraint_parameter() -> None:
    state = selected_state()
    base = lp_model(
        project_id=state.project_id,
        problem_id=state.problem_id,
        source_selected_model_id="CAND-lp",
    )
    baseline = ParameterDefinition(
        parameter_id="PAR-baseline",
        symbol="baseline",
        description="Row-level observation unavailable to the scalar verifier",
        data_binding=DataBinding(binding_id="BIND-baseline", dataset_id=uuid4(), column="observed"),
        source_type=ParameterSourceType.DATA,
        source_ref="EVID-fact-1",
        confidence=1,
    )
    constraint = base.constraints[0].model_copy(update={"rhs": symbol("baseline")})
    model = base.model_copy(update={"parameters": [baseline], "constraints": [constraint]})

    gate = model_quality_gate(model, state)

    assert "MODEL_GATE_FAIL:scalar_verifier_inputs_available" in gate.errors
    assert "MODEL_GATE_FAIL:UNAVAILABLE_SCALAR_INPUT:baseline" in gate.errors
    materialized = model.model_copy(
        update={"parameters": [baseline.model_copy(update={"value": 2.0})]}
    )
    assert (
        "MODEL_GATE_FAIL:scalar_verifier_inputs_available"
        not in model_quality_gate(materialized, state).errors
    )


def test_model_gate_rejects_data_literal_not_stated_in_cited_evidence() -> None:
    state = selected_state()
    state = state.model_copy(
        update={
            "evidence_items": [
                EvidenceItem(
                    evidence_id="EVID-data-file",
                    type=EvidenceType.DATA,
                    content="The supplied CSV contains match-level point records.",
                    source=EvidenceSource.PROBLEM_TEXT,
                    confidence=1,
                    status=EvidenceStatus.ACCEPTED,
                )
            ]
        }
    )
    base = lp_model(
        project_id=state.project_id,
        problem_id=state.problem_id,
        source_selected_model_id="CAND-lp",
    )
    parameter = ParameterDefinition(
        parameter_id="PAR-win-rate",
        symbol="win_rate",
        description="Claimed observed rate",
        value=0.5046,
        source_type=ParameterSourceType.DATA,
        source_ref="EVID-data-file",
        confidence=0.9,
    )
    model = base.model_copy(update={"parameters": [parameter]})

    gate = model_quality_gate(model, state)

    assert "MODEL_GATE_FAIL:data_literals_have_numeric_evidence" in gate.errors
    assert "MODEL_GATE_FAIL:UNSUPPORTED_DATA_LITERAL:win_rate" in gate.errors

    stated = state.evidence_items[0].model_copy(
        update={"content": "The observed win rate is 50.46% in the stated source."}
    )
    supported_state = state.model_copy(update={"evidence_items": [stated]})
    assert (
        "MODEL_GATE_FAIL:data_literals_have_numeric_evidence"
        in model_quality_gate(model, supported_state).errors
    )
    supported_state = supported_state.model_copy(
        update={"raw_problem": f"{state.raw_problem} The observed win rate is 50.46%."}
    )
    assert (
        "MODEL_GATE_FAIL:data_literals_have_numeric_evidence"
        not in model_quality_gate(model, supported_state).errors
    )


def test_model_gate_requires_declared_data_to_reach_core_equations() -> None:
    state = selected_state()
    dataset = DatasetRecord(
        project_id=state.project_id,
        problem_id=state.problem_id,
        source_file_id=uuid4(),
        name="observations.csv",
        row_count=5,
        column_count=1,
        columns=["rate"],
    )
    state = state.model_copy(update={"datasets": [dataset]})
    base = lp_model(
        project_id=state.project_id,
        problem_id=state.problem_id,
        source_selected_model_id="CAND-lp",
    )
    binding = DataBinding(binding_id="BIND-rate", dataset_id=dataset.dataset_id, column="rate")
    orphan = base.model_copy(update={"data_bindings": [binding]})
    assert "MODEL_GATE_FAIL:data_bindings_reach_core" in model_quality_gate(orphan, state).errors

    parameter = ParameterDefinition(
        parameter_id="PAR-rate",
        symbol="rate",
        description="Observed rate computed from the registered dataset",
        value=1.0,
        data_binding=binding,
        source_type=ParameterSourceType.DATA,
        source_ref="EVID-fact-1",
        confidence=1,
    )
    unused = orphan.model_copy(update={"parameters": [parameter]})
    assert "MODEL_GATE_FAIL:data_bindings_reach_core" in model_quality_gate(unused, state).errors

    assert base.objective is not None
    used = unused.model_copy(
        update={
            "objective": base.objective.model_copy(
                update={"expression": add(symbol("rate"), symbol("x"))}
            )
        }
    )
    assert model_quality_gate(used, state).checks["data_bindings_reach_core"]
    mismatched = used.model_copy(
        update={
            "parameters": [
                parameter.model_copy(
                    update={"data_binding": binding.model_copy(update={"column": "other"})}
                )
            ]
        }
    )
    assert (
        "MODEL_GATE_FAIL:data_bindings_reach_core" in model_quality_gate(mismatched, state).errors
    )


def test_model_gate_rejects_omitted_provided_dataset_for_target_subproblem() -> None:
    state = selected_state()
    base = lp_model(
        project_id=state.project_id,
        problem_id=state.problem_id,
        source_selected_model_id="CAND-lp",
    )
    model = base.model_copy(update={"target_subproblems": ["Q1"]})
    assert "MODEL_GATE_FAIL:required_data_is_bound" in model_quality_gate(model, state).errors
    dataset = DatasetRecord(
        project_id=state.project_id,
        problem_id=state.problem_id,
        source_file_id=uuid4(),
        name="demand.csv",
        row_count=5,
        column_count=1,
        columns=["demand"],
    )
    state = state.model_copy(update={"datasets": [dataset]})
    assert "MODEL_GATE_FAIL:required_data_is_bound" in model_quality_gate(model, state).errors
    assert model_quality_gate(base, state).checks["required_data_is_bound"]


def test_model_gate_recomputes_supported_bound_scalar_from_data_profile() -> None:
    state = selected_state()
    dataset = DatasetRecord(
        project_id=state.project_id,
        problem_id=state.problem_id,
        source_file_id=uuid4(),
        name="observations.csv",
        row_count=5,
        column_count=1,
        columns=["rate"],
    )
    profile = DataProfile(
        dataset_id=dataset.dataset_id,
        source_file_id=dataset.source_file_id,
        dataset_name=dataset.name,
        row_count=5,
        column_count=1,
        duplicate_row_count=0,
        duplicate_row_rate=0,
        columns=[
            ColumnProfile(
                name="rate",
                source_name="rate",
                physical_dtype="Float64",
                semantic_type=DataSemanticType.CONTINUOUS,
                missing_count=0,
                missing_rate=0,
                unique_count=2,
                unique_rate=0.4,
                numeric_statistics=NumericStatistics(count=5, mean=0.4),
            )
        ],
        quality_score=100,
    )
    state = state.model_copy(update={"datasets": [dataset], "data_profiles": [profile]})
    base = lp_model(
        project_id=state.project_id,
        problem_id=state.problem_id,
        source_selected_model_id="CAND-lp",
    )
    binding = DataBinding(
        binding_id="BIND-rate", dataset_id=dataset.dataset_id, column="rate", transform="mean"
    )
    parameter = ParameterDefinition(
        parameter_id="PAR-rate",
        symbol="rate",
        description="Mean of the registered rate column",
        value=0.4,
        data_binding=binding,
        source_type=ParameterSourceType.DATA,
        source_ref="EVID-fact-1",
        confidence=1,
    )
    assert base.objective is not None
    model = base.model_copy(
        update={
            "data_bindings": [binding],
            "parameters": [parameter],
            "objective": base.objective.model_copy(
                update={"expression": add(symbol("rate"), symbol("x"))}
            ),
        }
    )
    assert model_quality_gate(model, state).checks["bound_data_scalars_match_profile"]
    invented = model.model_copy(
        update={"parameters": [parameter.model_copy(update={"value": 0.5046})]}
    )
    assert (
        "MODEL_GATE_FAIL:UNVERIFIED_BOUND_SCALAR:rate" in model_quality_gate(invented, state).errors
    )
    assert materialize_data_bound_scalars(invented, state) == invented
    omitted = model.model_copy(
        update={"parameters": [parameter.model_copy(update={"value": None})]}
    )
    materialized = materialize_data_bound_scalars(omitted, state)
    assert materialized.parameters[0].value == 0.4
    assert omitted.parameters[0].value is None
    assert materialized.objective == omitted.objective
    assert materialized.data_bindings == omitted.data_bindings
    assert any("materialized" in item for item in materialized.limitations)
    assert model_quality_gate(materialized, state).checks["bound_data_scalars_match_profile"]
    assert materialize_data_bound_scalars(materialized, state) == materialized
    stale_state = state.model_copy(
        update={"data_profiles": [profile.model_copy(update={"source_file_id": uuid4()})]}
    )
    assert (
        "MODEL_GATE_FAIL:UNVERIFIED_BOUND_SCALAR:rate"
        in model_quality_gate(model, stale_state).errors
    )
    assert materialize_data_bound_scalars(omitted, stale_state) == omitted
    count_binding = binding.model_copy(update={"transform": "count_nonmissing"})
    count_model = model.model_copy(
        update={
            "data_bindings": [count_binding],
            "parameters": [
                parameter.model_copy(update={"value": 5.0, "data_binding": count_binding})
            ],
        }
    )
    assert model_quality_gate(count_model, state).checks["bound_data_scalars_match_profile"]
    rate_binding = binding.model_copy(update={"transform": "rate_eq:1"})
    rate_profile = profile.model_copy(
        update={
            "columns": [
                profile.columns[0].model_copy(
                    update={"top_values": [ValueCount(value="1", count=2)]}
                )
            ]
        }
    )
    rate_state = state.model_copy(update={"data_profiles": [rate_profile]})
    rate_model = model.model_copy(
        update={
            "data_bindings": [rate_binding],
            "parameters": [parameter.model_copy(update={"data_binding": rate_binding})],
        }
    )
    assert model_quality_gate(rate_model, rate_state).checks["bound_data_scalars_match_profile"]
    unsupported = model.model_copy(
        update={
            "data_bindings": [binding.model_copy(update={"transform": "custom"})],
            "parameters": [
                parameter.model_copy(
                    update={
                        "value": None,
                        "data_binding": binding.model_copy(update={"transform": "custom"}),
                    }
                )
            ],
        }
    )
    assert (
        "MODEL_GATE_FAIL:UNVERIFIED_BOUND_SCALAR:rate"
        in model_quality_gate(unsupported, state).errors
    )
    assert materialize_data_bound_scalars(unsupported, state) == unsupported
    assumed = omitted.model_copy(
        update={
            "parameters": [
                omitted.parameters[0].model_copy(
                    update={"source_type": ParameterSourceType.ASSUMPTION}
                )
            ]
        }
    )
    assert materialize_data_bound_scalars(assumed, state) == assumed


def test_model_gate_rejects_data_target_inside_its_own_squared_predictor() -> None:
    state = selected_state()
    base = lp_model(
        project_id=state.project_id,
        problem_id=state.problem_id,
        source_selected_model_id="CAND-lp",
    )
    assert base.objective is not None
    binding = DataBinding(
        binding_id="BIND-target", dataset_id=uuid4(), column="outcome", transform="mean"
    )
    target = ParameterDefinition(
        parameter_id="PAR-target",
        symbol="target",
        description="Observed target rate",
        value=0.5,
        data_binding=binding,
        source_type=ParameterSourceType.DATA,
        source_ref="EVID-fact-1",
        confidence=1,
    )
    leaked_loss = power(
        subtract(
            add(multiply(symbol("x"), symbol("target")), symbol("y")),
            symbol("target"),
        ),
        2,
    )
    model = base.model_copy(
        update={
            "parameters": [target],
            "objective": base.objective.model_copy(update={"expression": leaked_loss}),
        }
    )
    assert _self_referential_data_residuals(model) == {"target"}
    assert (
        "MODEL_GATE_FAIL:SELF_REFERENTIAL_DATA_TARGET:target"
        in model_quality_gate(model, state).errors
    )
    indirect = model.model_copy(
        update={
            "derived_variables": [
                variable("loss").model_copy(update={"role": VariableRole.DERIVED})
            ],
            "objective": base.objective.model_copy(update={"expression": symbol("loss")}),
            "equations": [
                base.equations[0].model_copy(update={"lhs": symbol("loss"), "rhs": leaked_loss})
            ],
        }
    )
    assert _self_referential_data_residuals(indirect) == {"target"}
    clean_loss = power(
        subtract(add(multiply(symbol("x"), symbol("y")), constant(0.1)), symbol("target")),
        2,
    )
    clean = model.model_copy(
        update={"objective": base.objective.model_copy(update={"expression": clean_loss})}
    )
    assert _self_referential_data_residuals(clean) == set()
    assert (
        _self_referential_data_residuals(
            model.model_copy(
                update={
                    "parameters": [
                        target.model_copy(update={"source_type": ParameterSourceType.ASSUMPTION})
                    ]
                }
            )
        )
        == set()
    )


def test_model_gate_rejects_objective_constant_through_derived_equation() -> None:
    state = selected_state()
    base = lp_model(
        project_id=state.project_id,
        problem_id=state.problem_id,
        source_selected_model_id="CAND-lp",
    )
    assert base.objective is not None
    score = variable("score").model_copy(update={"role": VariableRole.DERIVED})
    definition = base.equations[0].model_copy(
        update={
            "equation_id": "EQ-fixed-score",
            "lhs": symbol("score"),
            "rhs": constant(1),
        }
    )
    fixed_objective = base.objective.model_copy(
        update={"expression": symbol("score"), "equation_ref": "EQ-fixed-score"}
    )
    model = base.model_copy(
        update={
            "derived_variables": [score],
            "equations": [*base.equations, definition],
            "objective": fixed_objective,
        }
    )

    assert (
        "MODEL_GATE_FAIL:objective_depends_on_decision" in model_quality_gate(model, state).errors
    )
    assert (
        "MODEL_GATE_FAIL:objective_depends_on_decision"
        not in model_quality_gate(base, state).errors
    )
    coupled = model.model_copy(
        update={
            "equations": [
                *base.equations,
                definition.model_copy(update={"rhs": add(symbol("x"), constant(1))}),
            ]
        }
    )
    assert (
        "MODEL_GATE_FAIL:objective_depends_on_decision"
        not in model_quality_gate(coupled, state).errors
    )


def test_model_gate_rejects_fixed_infeasible_hard_constraint_before_solver() -> None:
    state = selected_state()
    base = lp_model(
        project_id=state.project_id,
        problem_id=state.problem_id,
        source_selected_model_id="CAND-lp",
    )
    flow = variable("flow").model_copy(update={"role": VariableRole.DERIVED})
    hazard = variable("hazard").model_copy(update={"role": VariableRole.DERIVED})
    flow_equation = EquationDefinition(
        equation_id="EQ-flow",
        latex="flow=0.2",
        normalized_expression="flow=0.2",
        lhs=symbol("flow"),
        rhs=constant(0.2),
        meaning="fixed scalar flow",
        source_refs=["EVID-fact-1"],
        derivation="fixture",
    )
    hazard_equation = flow_equation.model_copy(
        update={
            "equation_id": "EQ-hazard",
            "lhs": symbol("hazard"),
            "rhs": subtract(symbol("flow"), constant(1.5)),
        }
    )
    bound = ConstraintDefinition(
        constraint_id="CON-hazard-min",
        name="nonnegative hazard",
        expression=symbol("hazard"),
        relation=ConstraintRelation.GE,
        rhs=constant(0),
        normalized_expression="hazard>=0",
        description="hazard must be nonnegative",
        source_refs=["EVID-fact-1"],
        equation_ref="EQ-hazard",
    )
    model = base.model_copy(
        update={
            "derived_variables": [flow, hazard],
            "equations": [*base.equations, flow_equation, hazard_equation],
            "constraints": [*base.constraints, bound],
        }
    )

    gate = model_quality_gate(model, state)
    assert gate.status is QualityGateStatus.RETRY
    assert gate.checks["decision_independent_constraints_feasible"] is False
    assert (
        "MODEL_GATE_FAIL:DECISION_INDEPENDENT_CONSTRAINT_INFEASIBLE:CON-hazard-min" in gate.errors
    )
    feasible = model.model_copy(
        update={
            "equations": [
                *base.equations,
                flow_equation,
                hazard_equation.model_copy(update={"rhs": add(symbol("flow"), constant(0.5))}),
            ]
        }
    )
    assert model_quality_gate(feasible, state).checks["decision_independent_constraints_feasible"]
    decision_dependent = model.model_copy(
        update={
            "equations": [
                *base.equations,
                flow_equation.model_copy(update={"rhs": symbol("x")}),
                hazard_equation,
            ]
        }
    )
    assert model_quality_gate(decision_dependent, state).checks[
        "decision_independent_constraints_feasible"
    ]
    soft_bound = model.model_copy(
        update={"constraints": [*base.constraints, bound.model_copy(update={"is_hard": False})]}
    )
    assert model_quality_gate(soft_bound, state).checks["decision_independent_constraints_feasible"]
    assert model_quality_gate(base, state).status is QualityGateStatus.PASS


def test_model_gate_rejects_scalar_optimization_misclassified_as_time_series() -> None:
    state = selected_state()
    base = lp_model(
        project_id=state.project_id,
        problem_id=state.problem_id,
        source_selected_model_id="CAND-lp",
    )
    mislabeled = base.model_copy(update={"model_family": ModelFamily.TIME_SERIES})
    gate = model_quality_gate(mislabeled, state)
    assert gate.status is QualityGateStatus.RETRY
    assert "MODEL_GATE_FAIL:SCALAR_OPTIMIZATION_UNSUPPORTED_FAMILY:time_series" in gate.errors
    chained = base.model_copy(update={"model_family": ModelFamily.MODEL_CHAIN})
    chain_gate = model_quality_gate(chained, state)
    assert "MODEL_GATE_FAIL:SCALAR_OPTIMIZATION_UNSUPPORTED_FAMILY:model_chain" in chain_gate.errors
    assert model_quality_gate(base, state).status is QualityGateStatus.PASS


def test_model_gate_rejects_unmaterializable_indexed_initial_condition() -> None:
    state = selected_state()
    base = lp_model(
        project_id=state.project_id,
        problem_id=state.problem_id,
        source_selected_model_id="CAND-lp",
    )
    indexed = base.decision_variables[0].model_copy(update={"index_sets": ["SET-POINTS"]})
    initial = base.constraints[0].model_copy(
        update={
            "constraint_id": "CON-INIT-X",
            "name": "first indexed state value",
            "expression": symbol("x"),
            "rhs": constant(0),
            "relation": ConstraintRelation.EQ,
            "index_scope": ["SET-POINTS"],
        }
    )
    model = base.model_copy(
        update={
            "sets": [
                SetDefinition(
                    set_id="SET-POINTS",
                    symbol="T",
                    description="fixture time points",
                    values=[1, 2],
                )
            ],
            "decision_variables": [indexed, base.decision_variables[1]],
            "initial_conditions": [initial],
        }
    )
    gate = model_quality_gate(model, state)
    assert "MODEL_GATE_FAIL:INDEXED_HARD_CONSTRAINT_NOT_SCALAR:CON-INIT-X:x" in gate.errors
    assert model_quality_gate(base, state).status is QualityGateStatus.PASS


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
