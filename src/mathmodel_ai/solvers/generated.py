from __future__ import annotations

import json
import math
import re
from typing import Any

from pydantic import ValidationError

from mathmodel_ai.core.errors import DependencyUnavailableError, SandboxError
from mathmodel_ai.files.storage import FileStore
from mathmodel_ai.mathematical.digests import mathematical_model_digest
from mathmodel_ai.mathematical.expressions import evaluate_expression, scalar_parameter_values
from mathmodel_ai.sandbox.executor import SandboxExecutor
from mathmodel_ai.schemas.execution import ExecutionStatus
from mathmodel_ai.schemas.mathematical import MathematicalModel
from mathmodel_ai.schemas.program import (
    GeneratedProgram,
    GeneratedProgramStatus,
    GeneratedSourceFile,
)
from mathmodel_ai.schemas.solver import (
    GeneratedResultPayload,
    SolverName,
    SolverOptions,
    SolverResult,
    SolverStatus,
)
from mathmodel_ai.solvers.base import SolverExecution
from mathmodel_ai.solvers.feasibility import check_feasibility

_APPROVED_DEPENDENCIES = {
    "numpy": "numpy",
    "scipy": "scipy",
    "pandas": "pandas",
    "polars": "polars",
    "sympy": "sympy",
    "networkx": "networkx",
    "statsmodels": "statsmodels",
    "scikit-learn": "sklearn",
    "ortools": "ortools",
}
_DYNAMIC_INSTALL = re.compile(
    r"(?:pip\s+install|python\s+-m\s+pip|subprocess\s*\.\s*(?:run|call|Popen).*pip)",
    re.IGNORECASE | re.DOTALL,
)


class GeneratedProgramExecutor:
    def __init__(self, *, sandbox: SandboxExecutor, store: FileStore) -> None:
        self._sandbox = sandbox
        self._store = store

    def execute(
        self,
        program: GeneratedProgram,
        model: MathematicalModel,
        options: SolverOptions,
    ) -> SolverExecution:
        self._validate(program, model)
        entrypoint = self._entrypoint(program)
        supporting = {
            item.path: item.content for item in program.files if item.path != program.entrypoint
        }
        execution = self._sandbox.execute(
            entrypoint.content,
            project_id=program.project_id,
            problem_id=program.problem_id,
            execution_origin=program.execution_origin,
            model_digest=program.model_digest,
            generated_program_id=program.program_id,
            entrypoint=program.entrypoint,
            source_files=supporting,
        )
        finalized_files = [
            item.model_copy(
                update={
                    "artifact_id": (
                        execution.record.code_artifact_id
                        if item.path == program.entrypoint
                        else item.artifact_id
                    )
                }
            )
            for item in program.files
        ]
        program_status = (
            GeneratedProgramStatus.EXECUTED
            if execution.record.status is ExecutionStatus.SUCCEEDED
            else GeneratedProgramStatus.FAILED
        )
        finalized = program.model_copy(update={"files": finalized_files, "status": program_status})
        payload, payload_error = self._result_payload(
            execution.record.status, execution.artifact_records
        )
        if payload is None:
            result_status = (
                SolverStatus.TIME_LIMIT
                if execution.record.status is ExecutionStatus.TIMEOUT
                else SolverStatus.EXECUTION_ERROR
            )
            return SolverExecution(
                result=SolverResult(
                    solver_name=self._declared_solver(program),
                    status=result_status,
                    runtime_seconds=execution.record.runtime_seconds,
                    message=execution.record.error
                    or payload_error
                    or "generated program produced no valid result.json",
                    raw_status=execution.record.status.value,
                    execution_record_id=execution.record.run_id,
                ),
                execution=execution,
                program=finalized,
            )
        if payload.model_digest != program.model_digest:
            raise SandboxError("generated result model_digest does not match its program")
        self._validate_solver_identity(program, payload.solver_name)
        feasibility = check_feasibility(
            model,
            payload.variable_values,
            tolerance=options.feasibility_tolerance,
        )
        feasible = bool(
            payload.is_feasible
            and feasibility.checked
            and not feasibility.violated_constraints
            and not feasibility.bound_violations
        )
        objective_matches = self._objective_matches(
            model,
            payload.variable_values,
            payload.objective,
            tolerance=options.feasibility_tolerance,
        )
        semantic_valid = bool(not payload.is_feasible or (feasible and objective_matches))
        result_status = payload.status if semantic_valid else SolverStatus.MODEL_INVALID
        result = SolverResult(
            solver_name=payload.solver_name,
            solver_version=payload.solver_version,
            status=result_status,
            objective_value=payload.objective,
            variable_values=payload.variable_values,
            runtime_seconds=execution.record.runtime_seconds,
            iterations=self._optional_int(payload.metrics.get("iterations")),
            nodes=self._optional_int(payload.metrics.get("nodes")),
            mip_gap=self._optional_float(payload.metrics.get("mip_gap")),
            message=payload.message,
            raw_status=payload.raw_status,
            is_optimal=payload.is_optimal and semantic_valid,
            is_feasible=feasible and semantic_valid,
            warnings=[
                *payload.warnings,
                *(
                    ["generated result failed recomputed feasibility or objective checks"]
                    if not semantic_valid
                    else []
                ),
            ],
            execution_record_id=execution.record.run_id,
            feasibility=feasibility,
        )
        return SolverExecution(result=result, execution=execution, program=finalized)

    def _validate(self, program: GeneratedProgram, model: MathematicalModel) -> None:
        if (program.model_id, program.model_version) != (model.model_id, model.version):
            raise SandboxError("generated program references a different mathematical model")
        if program.model_digest != mathematical_model_digest(model):
            raise SandboxError("generated program model_digest is not canonical for its model")
        unavailable = [item for item in program.dependencies if item not in _APPROVED_DEPENDENCIES]
        if unavailable:
            raise DependencyUnavailableError(
                "DEPENDENCY_UNAVAILABLE: " + ", ".join(sorted(unavailable))
            )
        for dependency in program.dependencies:
            module = _APPROVED_DEPENDENCIES[dependency]
            if not self._sandbox.probe_python_module(module):
                raise DependencyUnavailableError(f"DEPENDENCY_UNAVAILABLE: {dependency}")
        if any(_DYNAMIC_INSTALL.search(item.content) for item in program.files):
            raise DependencyUnavailableError(
                "DEPENDENCY_UNAVAILABLE: dynamic installation is forbidden"
            )

    @staticmethod
    def _validate_solver_identity(program: GeneratedProgram, solver: SolverName) -> None:
        target = program.solver_target.upper()
        expected = next((item for item in SolverName if target.startswith(item.value)), None)
        if expected is not None and expected is not solver:
            raise SandboxError("generated result solver does not match declared solver_target")

    @staticmethod
    def _objective_matches(
        model: MathematicalModel,
        variables: dict[str, float],
        objective: float | None,
        *,
        tolerance: float,
    ) -> bool:
        if model.objective is None:
            return objective is None
        if objective is None:
            return False
        values = {
            **scalar_parameter_values([*model.parameters, *model.constants]),
            **variables,
        }
        try:
            expected = evaluate_expression(model.objective.expression, values)
        except ValueError:
            return False
        return math.isclose(objective, expected, rel_tol=tolerance, abs_tol=tolerance)

    @staticmethod
    def _entrypoint(program: GeneratedProgram) -> GeneratedSourceFile:
        return next(item for item in program.files if item.path == program.entrypoint)

    def _result_payload(
        self,
        status: ExecutionStatus,
        artifacts: list[Any],
    ) -> tuple[GeneratedResultPayload | None, str | None]:
        if status is not ExecutionStatus.SUCCEEDED:
            return None, None
        artifact = next((item for item in artifacts if item.name == "result.json"), None)
        if artifact is None:
            return None, "generated program did not produce /output/result.json"
        try:
            data: Any = json.loads(self._store.read_bytes(artifact.storage_key).decode("utf-8"))
            return GeneratedResultPayload.model_validate(data), None
        except ValidationError as exc:
            details = "; ".join(
                f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
                for item in exc.errors(include_url=False, include_context=False)
            )
            return None, f"generated result.json failed schema validation: {details[:2000]}"
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None, "generated result.json is not valid UTF-8 JSON"
        except ValueError as exc:
            return None, f"generated result.json could not be read ({type(exc).__name__})"

    @staticmethod
    def _declared_solver(program: GeneratedProgram) -> SolverName:
        try:
            return SolverName(program.solver_target)
        except ValueError:
            return SolverName.SCIPY

    @staticmethod
    def _optional_int(value: object) -> int | None:
        return (
            value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None
        )

    @staticmethod
    def _optional_float(value: object) -> float | None:
        if isinstance(value, int | float) and not isinstance(value, bool) and value >= 0:
            return float(value)
        return None
