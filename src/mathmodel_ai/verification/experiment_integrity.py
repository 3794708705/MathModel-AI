from __future__ import annotations

import ast
import json
import math
from dataclasses import dataclass

from mathmodel_ai.mathematical.digests import mathematical_model_digest
from mathmodel_ai.schemas.execution import ExecutionOrigin, ExecutionStatus
from mathmodel_ai.schemas.mathematical import MathematicalModel, ParameterDefinition
from mathmodel_ai.schemas.program import GeneratedProgramStatus, GeneratedSourceFile
from mathmodel_ai.schemas.solver import SolverName
from mathmodel_ai.schemas.verification import ExperimentRun, ParameterPerturbation
from mathmodel_ai.solvers.base import SolverExecution
from mathmodel_ai.verification.scalar_response import evaluate_scalar_responses
from mathmodel_ai.verification.validation import IndependentValidator


@dataclass(frozen=True)
class ExperimentIntegrityAudit:
    valid: bool
    errors: list[str]


class ExperimentIntegrityVerifier:
    """Binds perturbation metadata to the exact deterministic program execution."""

    def __init__(self, validator: IndependentValidator) -> None:
        self._validator = validator

    def audit(
        self,
        *,
        base_model: MathematicalModel,
        record: ExperimentRun,
        execution: SolverExecution,
    ) -> ExperimentIntegrityAudit:
        errors: list[str] = []
        scenario = self.reconstruct_scenario(base_model, record.perturbations, errors)
        base_digest = mathematical_model_digest(base_model)
        scenario_digest = mathematical_model_digest(scenario)
        program = execution.program
        execution_record = execution.execution.record
        solver_result = execution.result

        self._require_equal(
            errors, "BASE_MODEL_DIGEST_MISMATCH", record.base_model_digest, base_digest
        )
        self._require_equal(
            errors,
            "SCENARIO_MODEL_DIGEST_MISMATCH",
            record.scenario_model_digest,
            scenario_digest,
        )
        self._require_equal(
            errors, "PROGRAM_MODEL_ID_MISMATCH", program.model_id, scenario.model_id
        )
        self._require_equal(
            errors,
            "PROGRAM_MODEL_VERSION_MISMATCH",
            program.model_version,
            scenario.version,
        )
        self._require_equal(
            errors,
            "PROGRAM_MODEL_DIGEST_MISMATCH",
            program.model_digest,
            scenario_digest,
        )
        self._require_equal(
            errors,
            "EXECUTION_MODEL_DIGEST_MISMATCH",
            execution_record.model_digest,
            scenario_digest,
        )
        self._require_equal(
            errors,
            "EXECUTION_PROGRAM_ID_MISMATCH",
            execution_record.generated_program_id,
            program.program_id,
        )
        self._require_equal(
            errors,
            "EXECUTED_BUNDLE_HASH_MISMATCH",
            execution_record.executed_bundle_hash,
            program.code_hash,
        )
        entrypoint = next(
            (item for item in program.files if item.path == program.entrypoint),
            None,
        )
        if entrypoint is None or entrypoint.sha256 is None:
            errors.append("PROGRAM_ENTRYPOINT_HASH_MISSING")
        else:
            self._require_equal(
                errors,
                "EXECUTION_CODE_HASH_MISMATCH",
                execution_record.code_hash,
                entrypoint.sha256,
            )
        if program.execution_origin is not ExecutionOrigin.DETERMINISTIC_SOLVER_ADAPTER:
            errors.append("EXPERIMENT_PROGRAM_NOT_DETERMINISTIC")
        if execution_record.execution_origin is not ExecutionOrigin.DETERMINISTIC_SOLVER_ADAPTER:
            errors.append("EXPERIMENT_EXECUTION_NOT_DETERMINISTIC")
        if program.status is not GeneratedProgramStatus.EXECUTED:
            errors.append("EXPERIMENT_PROGRAM_NOT_EXECUTED")
        if program.is_mock:
            errors.append("EXPERIMENT_PROGRAM_IS_MOCK")
        if (
            execution_record.status is not ExecutionStatus.SUCCEEDED
            or execution_record.exit_code != 0
            or execution_record.is_mock
        ):
            errors.append("EXPERIMENT_EXECUTION_NOT_REAL_SUCCESS")
        self._require_equal(
            errors,
            "SOLVER_EXECUTION_REF_MISMATCH",
            solver_result.execution_record_id,
            execution_record.run_id,
        )
        self._require_equal(
            errors,
            "EXPERIMENT_EXECUTION_REF_MISMATCH",
            record.execution_record_id,
            execution_record.run_id,
        )
        self._require_equal(
            errors, "EXPERIMENT_SOLVER_MISMATCH", record.solver, solver_result.solver_name
        )
        self._require_equal(
            errors,
            "EXPERIMENT_SOLVER_STATUS_MISMATCH",
            record.solver_status,
            solver_result.status,
        )
        self._require_float_equal(
            errors,
            "EXPERIMENT_OBJECTIVE_MISMATCH",
            record.objective_value,
            solver_result.objective_value,
        )
        for key, value in record.key_outputs.items():
            self._require_float_equal(
                errors,
                f"EXPERIMENT_OUTPUT_MISMATCH:{key}",
                value,
                solver_result.variable_values.get(key),
            )

        feasible, max_violation, failed = self._validator.candidate_is_feasible(
            scenario,
            solver_result.variable_values,
        )
        expected_feasible = feasible and solver_result.is_feasible
        self._require_equal(
            errors,
            "EXPERIMENT_FEASIBILITY_MISMATCH",
            record.feasible,
            expected_feasible,
        )
        self._require_float_equal(
            errors,
            "EXPERIMENT_MAX_VIOLATION_MISMATCH",
            record.max_constraint_violation,
            max_violation,
        )
        if not expected_feasible:
            errors.append("EXPERIMENT_SCENARIO_INFEASIBLE:" + ",".join(failed))

        payload_model = self._payload_model(program.entrypoint, program.files, errors)
        if payload_model is not None:
            if payload_model.model_dump(mode="json") != scenario.model_dump(mode="json"):
                errors.append("EXECUTED_SCENARIO_PAYLOAD_MISMATCH")
            self._require_equal(
                errors,
                "EXECUTED_PAYLOAD_DIGEST_MISMATCH",
                mathematical_model_digest(payload_model),
                scenario_digest,
            )
        if record.solver is SolverName.SCALAR_RESPONSE:
            payload = self._payload_data(program.entrypoint, program.files, errors)
            if payload is None or payload.get("backend") != "scalar_response":
                errors.append("RESPONSE_BACKEND_MISMATCH")
            else:
                options = payload.get("options")
                fixed = options.get("initial_point") if isinstance(options, dict) else None
                if fixed != record.fixed_decision_values:
                    errors.append("RESPONSE_FIXED_POINT_MISMATCH")
            try:
                expected = evaluate_scalar_responses(scenario, record.fixed_decision_values)
            except ValueError as exc:
                errors.append(f"RESPONSE_NOT_EVALUABLE:{exc}")
            else:
                expected_values = {**record.fixed_decision_values, **expected}
                if set(solver_result.variable_values) != set(expected_values):
                    errors.append("RESPONSE_OUTPUT_SET_MISMATCH")
                for symbol, value in expected_values.items():
                    self._require_float_equal(
                        errors,
                        f"RESPONSE_OUTPUT_MISMATCH:{symbol}",
                        solver_result.variable_values.get(symbol),
                        value,
                    )
                if record.objective_value is not None or solver_result.objective_value is not None:
                    errors.append("RESPONSE_OBJECTIVE_FABRICATED")
        return ExperimentIntegrityAudit(valid=not errors, errors=list(dict.fromkeys(errors)))

    @staticmethod
    def reconstruct_scenario(
        base_model: MathematicalModel,
        perturbations: list[ParameterPerturbation],
        errors: list[str] | None = None,
    ) -> MathematicalModel:
        issues = errors if errors is not None else []
        parameters = {item.parameter_id: item for item in base_model.parameters}
        updates: dict[str, float] = {}
        for item in perturbations:
            parameter = parameters.get(item.parameter_id)
            if parameter is None:
                issues.append(f"UNKNOWN_PERTURBATION_PARAMETER:{item.parameter_id}")
                continue
            if item.parameter_id in updates:
                issues.append(f"DUPLICATE_PERTURBATION_PARAMETER:{item.parameter_id}")
                continue
            baseline = ExperimentIntegrityVerifier._scalar(parameter)
            if baseline is None:
                issues.append(f"NONSCALAR_PERTURBATION_PARAMETER:{item.parameter_id}")
                continue
            if item.symbol != parameter.symbol or item.source_ref != parameter.source_ref:
                issues.append(f"PERTURBATION_IDENTITY_MISMATCH:{item.parameter_id}")
            if not math.isclose(item.baseline_value, baseline, rel_tol=0, abs_tol=1e-12):
                issues.append(f"PERTURBATION_BASELINE_MISMATCH:{item.parameter_id}")
            expected = baseline * (1 + item.fraction)
            if not math.isclose(item.perturbed_value, expected, rel_tol=1e-12, abs_tol=1e-12):
                issues.append(f"PERTURBATION_VALUE_MISMATCH:{item.parameter_id}")
            updates[item.parameter_id] = item.perturbed_value
        revised = [
            item.model_copy(update={"value": updates[item.parameter_id]})
            if item.parameter_id in updates
            else item
            for item in base_model.parameters
        ]
        return base_model.model_copy(update={"parameters": revised})

    @staticmethod
    def _payload_model(
        entrypoint: str,
        files: list[GeneratedSourceFile],
        errors: list[str],
    ) -> MathematicalModel | None:
        payload = ExperimentIntegrityVerifier._payload_data(entrypoint, files, errors)
        if payload is None:
            return None
        try:
            return MathematicalModel.model_validate(payload["model"])
        except (KeyError, TypeError, ValueError):
            errors.append("DETERMINISTIC_PAYLOAD_INVALID")
            return None

    @staticmethod
    def _payload_data(
        entrypoint: str,
        files: list[GeneratedSourceFile],
        errors: list[str],
    ) -> dict[str, object] | None:
        source = next((item.content for item in files if item.path == entrypoint), None)
        if source is None:
            errors.append("EXECUTED_ENTRYPOINT_MISSING")
            return None
        try:
            tree = ast.parse(source)
        except SyntaxError:
            errors.append("EXECUTED_ENTRYPOINT_INVALID_PYTHON")
            return None
        payload_text: str | None = None
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if not any(
                isinstance(target, ast.Name) and target.id == "PAYLOAD" for target in node.targets
            ):
                continue
            value = node.value
            if (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Attribute)
                and isinstance(value.func.value, ast.Name)
                and value.func.value.id == "json"
                and value.func.attr == "loads"
                and len(value.args) == 1
                and isinstance(value.args[0], ast.Constant)
                and isinstance(value.args[0].value, str)
            ):
                payload_text = value.args[0].value
                break
        if payload_text is None:
            errors.append("DETERMINISTIC_PAYLOAD_NOT_FOUND")
            return None
        try:
            payload = json.loads(payload_text)
            if not isinstance(payload, dict):
                raise ValueError("payload must be an object")
            return payload
        except (TypeError, ValueError, json.JSONDecodeError):
            errors.append("DETERMINISTIC_PAYLOAD_INVALID")
            return None

    @staticmethod
    def _scalar(parameter: ParameterDefinition) -> float | None:
        value = parameter.value
        if isinstance(value, bool):
            return None
        if isinstance(value, int | float) and math.isfinite(float(value)):
            return float(value)
        return None

    @staticmethod
    def _require_equal(errors: list[str], code: str, actual: object, expected: object) -> None:
        if actual != expected:
            errors.append(code)

    @staticmethod
    def _require_float_equal(
        errors: list[str],
        code: str,
        actual: float | None,
        expected: float | None,
    ) -> None:
        if actual is None or expected is None:
            if actual != expected:
                errors.append(code)
            return
        if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9):
            errors.append(code)
