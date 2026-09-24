from __future__ import annotations

import math
import re

from mathmodel_ai.mathematical.evidence import EvidenceIntegrityVerifier
from mathmodel_ai.mathematical.expressions import referenced_symbols, scalar_parameter_values
from mathmodel_ai.mathematical.registry import EquationRegistry, SymbolRegistry
from mathmodel_ai.mathematical.units import UnitChecker
from mathmodel_ai.schemas.execution import ExecutionRecord, ExecutionStatus
from mathmodel_ai.schemas.mathematical import (
    ConstraintRelation,
    ExpressionKind,
    InterpretationResolutionStatus,
    MathematicalModel,
    ParameterSourceType,
    UnitCheckStatus,
)
from mathmodel_ai.schemas.model_selection import ModelFamily
from mathmodel_ai.schemas.problem_analysis import EvidenceSource, EvidenceStatus
from mathmodel_ai.schemas.problem_state import ProblemState
from mathmodel_ai.schemas.program import GeneratedProgram
from mathmodel_ai.schemas.quality import QualityGateResult, QualityGateStatus
from mathmodel_ai.schemas.results import ResultRecord
from mathmodel_ai.schemas.solver import SolverCapability, SolverFamily, SolverRun, SolverStatus

_OPTIMIZATION_FAMILIES = {
    ModelFamily.LINEAR_PROGRAMMING,
    ModelFamily.MIXED_INTEGER_LINEAR_PROGRAMMING,
    ModelFamily.INTEGER_PROGRAMMING,
    ModelFamily.NONLINEAR_PROGRAMMING,
    ModelFamily.MULTI_OBJECTIVE,
    ModelFamily.NETWORK_FLOW,
}


def model_quality_gate(model: MathematicalModel, state: ProblemState) -> QualityGateResult:
    symbols = SymbolRegistry.from_model(model)
    equations = EquationRegistry.from_model(model)
    unit_report = UnitChecker(symbols).check_model(model)
    known_subproblems = {item.subproblem_id for item in state.subproblems}
    equation_ids = {item.equation_id for item in model.equations}
    required_equations = {
        *([model.objective.equation_ref] if model.objective is not None else []),
        *(item.equation_ref for item in model.constraints),
    }
    known_dataset_ids = {item.dataset_id for item in state.datasets}
    data_bindings_valid = all(
        binding.dataset_id in known_dataset_ids for binding in model.data_bindings
    )
    parameter_bindings_valid = all(
        parameter.data_binding is None or parameter.data_binding.dataset_id in known_dataset_ids
        for parameter in [*model.parameters, *model.constants]
    )

    resolutions = {item.ambiguity_id: item for item in model.interpretation_resolutions}
    targeted_ambiguities = {
        ambiguity_ref
        for subproblem in state.subproblems
        if subproblem.subproblem_id in model.target_subproblems
        for ambiguity_ref in subproblem.ambiguity_refs
    }
    ambiguity_by_id = {item.ambiguity_id: item for item in state.ambiguities}
    unresolved_critical: list[str] = []
    for ambiguity_id in targeted_ambiguities:
        ambiguity = ambiguity_by_id.get(ambiguity_id)
        resolution = resolutions.get(ambiguity_id)
        resolved_from_existing_evidence = bool(
            ambiguity is not None
            and ambiguity.preferred_interpretation_id is not None
            and ambiguity.confidence >= 0.7
        )
        explicitly_resolved = bool(
            resolution is not None
            and resolution.status
            in {
                InterpretationResolutionStatus.RESOLVED_FROM_EVIDENCE,
                InterpretationResolutionStatus.HUMAN_CONFIRMED,
            }
            and resolution.interpretation_id is not None
        )
        critical = resolution.critical if resolution is not None else True
        if critical and not (resolved_from_existing_evidence or explicitly_resolved):
            unresolved_critical.append(ambiguity_id)

    unsupported_assumptions = [
        item.assumption_id for item in model.assumptions if item.critical and not item.supported
    ]
    optimization = model.model_family in _OPTIMIZATION_FAMILIES
    solver_requirements_valid = all(
        item in {family.value for family in SolverFamily}
        for item in model.solver_requirements.preferred_solver_families
    ) and all(
        item in {capability.value for capability in SolverCapability}
        for item in model.solver_requirements.required_capabilities
    )
    unavailable_scalar_inputs = _unavailable_scalar_verifier_inputs(model)
    unsupported_data_literals = _unsupported_data_literals(model, state)
    objective_decision_coupling = _objective_depends_on_decision(model)
    checks = {
        "selected_model_exists": state.selected_model is not None,
        "selected_model_matches": (
            state.selected_model is not None
            and model.source_selected_model_id == state.selected_model.candidate_id
        ),
        "target_subproblems_known": set(model.target_subproblems) <= known_subproblems,
        "decision_variables_present": bool(model.decision_variables),
        "optimization_objective_present": not optimization or model.objective is not None,
        "objective_depends_on_decision": objective_decision_coupling,
        "symbol_registry_valid": symbols.report.valid,
        "parameter_sources_present": all(
            bool(item.source_ref) for item in [*model.parameters, *model.constants]
        ),
        "data_bindings_resolve": data_bindings_valid and parameter_bindings_valid,
        "equation_registry_valid": equations.report.valid,
        "state_relations_sufficient": _state_relations_sufficient(model),
        "scalar_verifier_inputs_available": not unavailable_scalar_inputs,
        "data_literals_have_numeric_evidence": not unsupported_data_literals,
        "required_equations_present": required_equations <= equation_ids,
        "no_explicit_unit_failure": unit_report.status is not UnitCheckStatus.FAIL,
        "critical_ambiguities_resolved": not unresolved_critical,
        "critical_assumptions_supported": not unsupported_assumptions,
        "solver_requirements_valid": solver_requirements_valid,
    }
    errors = [f"MODEL_GATE_FAIL:{name}" for name, passed in checks.items() if not passed]
    errors.extend(
        f"MODEL_GATE_FAIL:UNAVAILABLE_SCALAR_INPUT:{symbol}"
        for symbol in sorted(unavailable_scalar_inputs)
    )
    errors.extend(
        f"MODEL_GATE_FAIL:UNSUPPORTED_DATA_LITERAL:{symbol}"
        for symbol in sorted(unsupported_data_literals)
    )
    errors.extend(
        f"MODEL_GATE_FAIL:{item.code.value}:{item.reference}"
        for item in [*symbols.report.issues, *equations.report.issues]
        if item.critical
    )
    errors.extend(f"MODEL_GATE_FAIL:UNRESOLVED_AMBIGUITY:{item}" for item in unresolved_critical)
    errors.extend(
        f"MODEL_GATE_FAIL:UNSUPPORTED_CRITICAL_ASSUMPTION:{item}"
        for item in unsupported_assumptions
    )
    warnings = [
        f"{item.code.value}:{item.reference}" for item in symbols.report.issues if not item.critical
    ]
    if unit_report.status is UnitCheckStatus.UNKNOWN:
        warnings.append("UNIT_CHECK_REQUIRED")
    warnings.extend(
        item.message for item in unit_report.issues if item.status is not UnitCheckStatus.FAIL
    )
    return QualityGateResult(
        gate="MODEL",
        status=QualityGateStatus.PASS if not errors else QualityGateStatus.RETRY,
        checks=checks,
        errors=list(dict.fromkeys(errors)),
        warnings=list(dict.fromkeys(warnings)),
    )


def _state_relations_sufficient(model: MathematicalModel) -> bool:
    """Reject state symbols with fewer defining relations than unknown states."""
    states = {item.symbol for item in model.state_variables}
    if not states:
        return True
    objective_equation = model.objective.equation_ref if model.objective else None
    constraint_equations = {item.equation_ref for item in model.constraints}
    equality_equations = {
        item.equation_ref for item in model.constraints if item.relation is ConstraintRelation.EQ
    }
    relations = {
        equation.equation_id
        for equation in model.equations
        if equation.equation_id != objective_equation
        and (
            equation.equation_id not in constraint_equations
            or equation.equation_id in equality_equations
        )
        and referenced_symbols(equation.lhs) & states
    }
    relations.update(
        item.equation_ref
        for item in [*model.initial_conditions, *model.boundary_conditions]
        if item.relation is ConstraintRelation.EQ and referenced_symbols(item.expression) & states
    )
    return len(relations) >= len(states)


def _objective_depends_on_decision(model: MathematicalModel) -> bool:
    """Reject an objective proven constant with respect to every decision variable."""
    if model.objective is None:
        return True
    decisions = {item.symbol for item in model.decision_variables}
    parameters = {item.symbol for item in [*model.parameters, *model.constants]}
    definitions: dict[str, set[str]] = {}
    for equation in model.equations:
        if equation.lhs.kind is ExpressionKind.SYMBOL and equation.lhs.symbol is not None:
            definitions.setdefault(equation.lhs.symbol, set()).update(
                referenced_symbols(equation.rhs)
            )
    frontier = list(referenced_symbols(model.objective.expression))
    seen: set[str] = set()
    while frontier:
        symbol = frontier.pop()
        if symbol in decisions:
            return True
        if symbol in seen:
            continue
        seen.add(symbol)
        if symbol in definitions:
            frontier.extend(definitions[symbol] - seen)
        elif symbol not in parameters:
            # An implicit or externally defined symbol may depend on a decision;
            # this check rejects only objectives proven decision-independent.
            return True
    return False


def _unavailable_scalar_verifier_inputs(model: MathematicalModel) -> set[str]:
    """The scalar result contract cannot verify unmaterialized data-bound inputs."""
    available = {
        *(item.symbol for item in model.decision_variables),
        *(item.symbol for item in model.state_variables),
        *(item.symbol for item in model.derived_variables),
        *scalar_parameter_values([*model.parameters, *model.constants]),
    }
    expressions = [
        expression
        for constraint in [
            *model.constraints,
            *model.initial_conditions,
            *model.boundary_conditions,
        ]
        for expression in (constraint.expression, constraint.rhs)
    ]
    if model.objective is not None:
        expressions.append(model.objective.expression)
    referenced = set().union(*(referenced_symbols(expression) for expression in expressions))
    return referenced - available


_NUMERIC_LITERAL = re.compile(
    r"(?<![\w.])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?\s*%?(?!\w)"
)


def _unsupported_data_literals(model: MathematicalModel, state: ProblemState) -> set[str]:
    """Do not accept a model-generated DATA scalar with only a dataset-name citation."""
    evidence = {item.evidence_id: item for item in state.evidence_items}
    if state.problem_analysis is not None:
        evidence.update({item.evidence_id: item for item in state.problem_analysis.all_evidence()})
    unsupported: set[str] = set()
    for parameter in [*model.parameters, *model.constants]:
        if (
            parameter.source_type is not ParameterSourceType.DATA
            or parameter.data_binding is not None
            or isinstance(parameter.value, bool)
            or not isinstance(parameter.value, (int, float))
        ):
            continue
        cited = evidence.get(parameter.source_ref)
        if cited is None or cited.status is not EvidenceStatus.ACCEPTED:
            unsupported.add(parameter.symbol)
            continue
        if not _text_states_numeric_value(cited.content, float(parameter.value)):
            unsupported.add(parameter.symbol)
        elif (
            cited.source is EvidenceSource.PROBLEM_TEXT
            and not _text_states_numeric_value(state.raw_problem, float(parameter.value))
        ):
            unsupported.add(parameter.symbol)
    return unsupported


def _text_states_numeric_value(text: str, value: float) -> bool:
    for match in _NUMERIC_LITERAL.finditer(text):
        literal = match.group().strip()
        percent = literal.endswith("%")
        try:
            stated = float(literal.rstrip("%")) / (100 if percent else 1)
        except ValueError:
            continue
        if math.isclose(value, stated, rel_tol=1e-9, abs_tol=1e-12):
            return True
    return False


def solve_quality_gate(
    model: MathematicalModel,
    result: ResultRecord,
    solver_run: SolverRun,
    execution: ExecutionRecord,
    program: GeneratedProgram | None = None,
    evidence_verifier: EvidenceIntegrityVerifier | None = None,
) -> QualityGateResult:
    evidence = (evidence_verifier or EvidenceIntegrityVerifier()).verify(
        result=result,
        solver_run=solver_run,
        execution=execution,
        model=model,
        program=program,
    )
    terminal_truthful = solver_run.status in {
        SolverStatus.OPTIMAL,
        SolverStatus.FEASIBLE,
        SolverStatus.INFEASIBLE,
        SolverStatus.UNBOUNDED,
        SolverStatus.INFEASIBLE_OR_UNBOUNDED,
    }
    execution_completed = execution.status is ExecutionStatus.SUCCEEDED
    expected_variables = {item.symbol for item in model.decision_variables}
    observed_variables = set(solver_run.result.variable_values)
    feasible = solver_run.result.is_feasible
    feasibility = solver_run.result.feasibility
    no_constraint_violation = bool(
        not feasible
        or (
            feasibility is not None
            and feasibility.checked
            and not feasibility.violated_constraints
            and not feasibility.bound_violations
            and (
                feasibility.max_constraint_violation is None
                or feasibility.max_constraint_violation <= feasibility.tolerance
            )
        )
    )
    checks = {
        "canonical_solver_status": solver_run.status is solver_run.result.status,
        "execution_record_matches": solver_run.execution_ref == execution.run_id,
        "execution_is_non_mock": not execution.is_mock,
        "completed_execution_for_terminal_result": not terminal_truthful or execution_completed,
        "result_schema_links": (
            result.solver_run_id == solver_run.solver_run_id
            and result.execution_record_id == execution.run_id
            and result.model_id == model.model_id
            and result.model_version == model.version
        ),
        "key_variables_present_when_feasible": not feasible
        or expected_variables <= observed_variables,
        "objective_present_when_feasible": (
            not feasible or model.objective is None or solver_run.result.objective_value is not None
        ),
        "optimal_is_feasible": (
            solver_run.status is not SolverStatus.OPTIMAL
            or (solver_run.result.is_optimal and solver_run.result.is_feasible)
        ),
        "failure_not_wrapped_as_success": not (
            execution.status is not ExecutionStatus.SUCCEEDED
            and solver_run.status in {SolverStatus.OPTIMAL, SolverStatus.FEASIBLE}
        ),
        "implemented_feasibility_passes": no_constraint_violation,
        "evidence_integrity": evidence.valid,
    }
    errors = [f"SOLVE_GATE_FAIL:{name}" for name, passed in checks.items() if not passed]
    retry_statuses = {
        SolverStatus.TIME_LIMIT,
        SolverStatus.ITERATION_LIMIT,
        SolverStatus.NUMERICAL_ERROR,
        SolverStatus.MODEL_INVALID,
        SolverStatus.SOLVER_UNAVAILABLE,
        SolverStatus.EXECUTION_ERROR,
        SolverStatus.UNKNOWN,
    }
    if solver_run.status in retry_statuses:
        errors.append(f"SOLVE_GATE_FAIL:{solver_run.status.value}")
    errors.extend(f"SOLVE_GATE_FAIL:{item.code.value}" for item in evidence.errors)
    return QualityGateResult(
        gate="SOLVE",
        status=QualityGateStatus.PASS if not errors else QualityGateStatus.RETRY,
        checks=checks,
        errors=errors,
        warnings=list(solver_run.result.warnings),
    )
