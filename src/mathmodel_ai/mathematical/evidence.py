from __future__ import annotations

import math
from collections.abc import Iterable

from mathmodel_ai.mathematical.digests import mathematical_model_digest
from mathmodel_ai.schemas.execution import ExecutionRecord, ExecutionStatus
from mathmodel_ai.schemas.mathematical import MathematicalModel
from mathmodel_ai.schemas.program import GeneratedProgram
from mathmodel_ai.schemas.results import (
    EvidenceChainReport,
    EvidenceErrorCode,
    EvidenceIssue,
    ResultRecord,
)
from mathmodel_ai.schemas.solver import SolverRun, SolverStatus


class EvidenceIntegrityVerifier:
    def __init__(self, *, absolute_tolerance: float = 1e-9, relative_tolerance: float = 1e-9):
        if absolute_tolerance < 0 or relative_tolerance < 0:
            raise ValueError("evidence tolerances must be non-negative")
        self._atol = absolute_tolerance
        self._rtol = relative_tolerance

    def verify(
        self,
        *,
        result: ResultRecord,
        solver_run: SolverRun | None,
        execution: ExecutionRecord | None,
        model: MathematicalModel | None,
        program: GeneratedProgram | None,
        additional_issues: Iterable[EvidenceIssue] = (),
    ) -> EvidenceChainReport:
        issues = list(additional_issues)
        if solver_run is None:
            issues.append(
                self._issue(EvidenceErrorCode.MISSING_SOLVER_RUN, "solver run is missing")
            )
        if execution is None:
            issues.append(self._issue(EvidenceErrorCode.MISSING_EXECUTION, "execution is missing"))
        if model is None:
            issues.append(
                self._issue(EvidenceErrorCode.MISSING_MODEL, "mathematical model is missing")
            )

        expected_digest = mathematical_model_digest(model) if model is not None else None
        exact_model_version = bool(
            model is not None
            and solver_run is not None
            and model.model_id == result.model_id == solver_run.model_id
            and model.version == result.model_version == solver_run.model_version
        )
        if not exact_model_version and model is not None and solver_run is not None:
            issues.append(
                self._issue(
                    EvidenceErrorCode.MODEL_VERSION_MISMATCH,
                    "result, solver run, and mathematical model identity/version differ",
                )
            )

        digest_values = [result.model_digest]
        if solver_run is not None:
            digest_values.append(solver_run.model_digest)
        if execution is not None and execution.model_digest is not None:
            digest_values.append(execution.model_digest)
        if program is not None:
            digest_values.append(program.model_digest)
        model_digest_match = bool(
            expected_digest is not None
            and digest_values
            and all(item == expected_digest for item in digest_values)
        )
        if expected_digest is not None and not model_digest_match:
            issues.append(
                self._issue(
                    EvidenceErrorCode.MODEL_DIGEST_MISMATCH,
                    "one or more records do not match the canonical mathematical model digest",
                )
            )

        if solver_run is not None:
            self._verify_result_and_run(result, solver_run, issues)
        if execution is not None:
            self._verify_execution(result, solver_run, execution, issues)

        program_required = solver_run is not None and solver_run.generated_program_id is not None
        code_hash_match: bool | None = None
        if program_required and program is None:
            issues.append(
                self._issue(
                    EvidenceErrorCode.MISSING_GENERATED_PROGRAM,
                    "solver run references a generated program that is missing",
                )
            )
        elif program is not None:
            code_hash_match = self._verify_program(
                result,
                solver_run,
                execution,
                program,
                issues,
            )

        unique = self._deduplicate(issues)
        codes = {item.code for item in unique}
        return EvidenceChainReport(
            result_id=result.result_id,
            valid=not unique,
            model_found=model is not None,
            solver_run_found=solver_run is not None,
            execution_found=execution is not None,
            generated_program_found=(program is not None if program_required else None),
            exact_model_version=(
                exact_model_version and EvidenceErrorCode.MODEL_VERSION_MISMATCH not in codes
            ),
            model_digest_match=(
                model_digest_match and EvidenceErrorCode.MODEL_DIGEST_MISMATCH not in codes
            ),
            code_hash_match=(
                code_hash_match if EvidenceErrorCode.CODE_HASH_MISMATCH not in codes else False
            ),
            errors=unique,
        )

    def _verify_result_and_run(
        self,
        result: ResultRecord,
        solver_run: SolverRun,
        issues: list[EvidenceIssue],
    ) -> None:
        if (
            result.solver_run_id != solver_run.solver_run_id
            or solver_run.result_ref != result.result_id
        ):
            issues.append(
                self._issue(
                    EvidenceErrorCode.RESULT_REF_MISMATCH,
                    "ResultRecord and SolverRun references are not reciprocal",
                )
            )
        statuses = {result.status, solver_run.status, solver_run.result.status}
        if len(statuses) != 1 or (
            result.status is SolverStatus.OPTIMAL
            and not (solver_run.result.is_optimal and solver_run.result.is_feasible)
        ):
            issues.append(
                self._issue(
                    EvidenceErrorCode.STATUS_MISMATCH,
                    "result status or canonical solver semantics differ",
                )
            )
        if (
            result.solver is not solver_run.solver
            or result.solver is not solver_run.result.solver_name
        ):
            issues.append(
                self._issue(
                    EvidenceErrorCode.SOLVER_IDENTITY_MISMATCH,
                    "ResultRecord, SolverRun, and SolverResult identify different solvers",
                )
            )
        if not self._optional_number_equal(
            result.objective, solver_run.objective
        ) or not self._optional_number_equal(result.objective, solver_run.result.objective_value):
            issues.append(
                self._issue(
                    EvidenceErrorCode.OBJECTIVE_MISMATCH,
                    "ResultRecord objective differs from persisted solver output",
                )
            )
        values = solver_run.result.variable_values
        if any(
            key not in values or not self._number_equal(value, values[key])
            for key, value in result.key_outputs.items()
        ):
            issues.append(
                self._issue(
                    EvidenceErrorCode.KEY_OUTPUT_MISMATCH,
                    "one or more ResultRecord key outputs differ from SolverResult variables",
                )
            )

    def _verify_execution(
        self,
        result: ResultRecord,
        solver_run: SolverRun | None,
        execution: ExecutionRecord,
        issues: list[EvidenceIssue],
    ) -> None:
        if result.execution_record_id != execution.run_id or (
            solver_run is not None and solver_run.execution_ref != execution.run_id
        ):
            issues.append(
                self._issue(
                    EvidenceErrorCode.BROKEN_RESULT_CHAIN,
                    "result or solver run references a different execution",
                )
            )
        if execution.is_mock:
            issues.append(
                self._issue(
                    EvidenceErrorCode.MOCK_EXECUTION, "real evidence cannot use Mock execution"
                )
            )
        if execution.status is not ExecutionStatus.SUCCEEDED or execution.exit_code != 0:
            issues.append(
                self._issue(
                    EvidenceErrorCode.EXECUTION_FAILED,
                    "execution did not complete successfully with exit code zero",
                )
            )
        if solver_run is not None and solver_run.execution_origin is not execution.execution_origin:
            issues.append(
                self._issue(
                    EvidenceErrorCode.BROKEN_RESULT_CHAIN,
                    "solver run and execution origins differ",
                )
            )

    def _verify_program(
        self,
        result: ResultRecord,
        solver_run: SolverRun | None,
        execution: ExecutionRecord | None,
        program: GeneratedProgram,
        issues: list[EvidenceIssue],
    ) -> bool:
        if solver_run is None or solver_run.generated_program_id != program.program_id:
            issues.append(
                self._issue(
                    EvidenceErrorCode.BROKEN_RESULT_CHAIN,
                    "solver run and generated program identifiers differ",
                )
            )
        if execution is not None and program.execution_origin is not execution.execution_origin:
            issues.append(
                self._issue(
                    EvidenceErrorCode.BROKEN_RESULT_CHAIN,
                    "generated program and execution origins differ",
                )
            )
        if program.model_id != result.model_id or program.model_version != result.model_version:
            issues.append(
                self._issue(
                    EvidenceErrorCode.MODEL_VERSION_MISMATCH,
                    "generated program references a different model revision",
                )
            )
        entrypoint = next(
            (
                item
                for item in program.files
                if item.path.replace("\\", "/") == program.entrypoint.replace("\\", "/")
            ),
            None,
        )
        matches = bool(
            execution is not None
            and entrypoint is not None
            and entrypoint.sha256 is not None
            and entrypoint.sha256 == execution.code_hash
            and program.code_hash == execution.executed_bundle_hash
            and execution.generated_program_id == program.program_id
        )
        if not matches:
            issues.append(
                self._issue(
                    EvidenceErrorCode.CODE_HASH_MISMATCH,
                    "generated source bundle or entrypoint hash does not match executed code",
                )
            )
        return matches

    def _number_equal(self, left: float, right: float) -> bool:
        return math.isclose(left, right, rel_tol=self._rtol, abs_tol=self._atol)

    def _optional_number_equal(self, left: float | None, right: float | None) -> bool:
        if left is None or right is None:
            return left is right
        return self._number_equal(left, right)

    @staticmethod
    def _issue(code: EvidenceErrorCode, message: str) -> EvidenceIssue:
        return EvidenceIssue(code=code, message=message)

    @staticmethod
    def _deduplicate(issues: list[EvidenceIssue]) -> list[EvidenceIssue]:
        unique: list[EvidenceIssue] = []
        seen: set[tuple[EvidenceErrorCode, str | None]] = set()
        for issue in issues:
            key = (issue.code, issue.reference)
            if key not in seen:
                seen.add(key)
                unique.append(issue)
        return unique
