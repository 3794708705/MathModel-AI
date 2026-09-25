"""Bind an automatically generated predictor to sealed benchmark evaluation."""

import hashlib
from pathlib import Path
from uuid import UUID

from mathmodel_ai.benchmark.manifests import BlindSolveBundle
from mathmodel_ai.core.errors import QualityGateError
from mathmodel_ai.data.repository import DataRepository
from mathmodel_ai.files.storage import FileStore
from mathmodel_ai.mathematical.digests import mathematical_model_digest
from mathmodel_ai.mathematical.repository import MathematicalRepository
from mathmodel_ai.mathematical.workflow import MathematicalRunOutcome
from mathmodel_ai.sandbox.causal_holdout import (
    IsolatedCausalHoldout,
    run_isolated_causal_holdout,
    verify_recorded_causal_holdout,
)
from mathmodel_ai.schemas.execution import ExecutionOrigin, ExecutionStatus, SandboxLimits
from mathmodel_ai.schemas.problem_state import ProblemState
from mathmodel_ai.schemas.program import GeneratedProgramStatus
from mathmodel_ai.verification.causal_binary import CausalBinarySpec
from mathmodel_ai.verification.causal_holdout import CausalHoldoutResult


class CausalBenchmarkEvaluator:
    """Run and persist a model-generated predictor, without trusting its metrics."""

    def __init__(
        self,
        *,
        store: FileStore,
        repository: DataRepository,
        mathematics: MathematicalRepository,
        root: Path,
        image: str,
        limits: SandboxLimits,
    ) -> None:
        self._store = store
        self._repository = repository
        self._mathematics = mathematics
        self._root = root
        self._image = image
        self._limits = limits

    def evaluate(
        self,
        *,
        bundle: BlindSolveBundle,
        mathematical: MathematicalRunOutcome,
        state: ProblemState,
    ) -> IsolatedCausalHoldout:
        policy = bundle.causal_policy
        split = bundle.causal_split
        if policy is None or split is None:
            raise QualityGateError("CAUSAL_HOLDOUT_POLICY_MISSING")
        source = next(
            (
                item.content
                for item in bundle.artifacts
                if item.resource.resource_id == policy.resource_id
            ),
            None,
        )
        if source is None or hashlib.sha256(source).hexdigest() != split.source_sha256:
            raise QualityGateError("CAUSAL_HOLDOUT_SOURCE_MISMATCH")
        resource = next(
            item.resource
            for item in bundle.artifacts
            if item.resource.resource_id == policy.resource_id
        )
        matched_inputs = [
            item
            for item in state.registered_files
            if item.original_name == resource.local_filename
            and item.sha256 == split.training_sha256
        ]
        if len(matched_inputs) != 1:
            raise QualityGateError("CAUSAL_HOLDOUT_TRAINING_INPUT_NOT_BOUND")
        solve = mathematical.solve_stage
        model = mathematical.model_stage.model
        program = solve.execution.program
        formal_execution = solve.execution.execution.record
        formal_result = solve.result
        solver_run = solve.solver_run
        model_digest = mathematical_model_digest(model)
        entrypoint = next((item for item in program.files if item.path == program.entrypoint), None)
        if (
            program.execution_origin is not ExecutionOrigin.GENERATED_PROGRAM
            or program.status is not GeneratedProgramStatus.EXECUTED
            or program.is_mock
            or program.model_digest != model_digest
        ):
            raise QualityGateError("CAUSAL_HOLDOUT_GENERATED_PROGRAM_NOT_BOUND")
        if (
            model.project_id != state.project_id
            or model.problem_id != state.problem_id
            or program.project_id != state.project_id
            or program.problem_id != state.problem_id
            or program.model_id != model.model_id
            or program.model_version != model.version
            or entrypoint is None
            or formal_execution.code_hash != entrypoint.sha256
            or formal_execution.project_id != state.project_id
            or formal_execution.problem_id != state.problem_id
            or formal_execution.execution_origin is not ExecutionOrigin.GENERATED_PROGRAM
            or formal_execution.status is not ExecutionStatus.SUCCEEDED
            or formal_execution.is_mock
            or formal_execution.model_digest != model_digest
            or formal_execution.generated_program_id != program.program_id
            or formal_execution.executed_bundle_hash != program.code_hash
            or formal_result.model_id != model.model_id
            or formal_result.model_version != model.version
            or formal_result.model_digest != model_digest
            or formal_result.project_id != state.project_id
            or formal_result.problem_id != state.problem_id
            or formal_result.solver_run_id != solver_run.solver_run_id
            or formal_result.execution_record_id != formal_execution.run_id
            or solver_run.project_id != state.project_id
            or solver_run.problem_id != state.problem_id
            or solver_run.model_id != model.model_id
            or solver_run.model_version != model.version
            or solver_run.model_digest != model_digest
            or solver_run.execution_origin is not ExecutionOrigin.GENERATED_PROGRAM
            or solver_run.generated_program_id != program.program_id
            or solver_run.execution_ref != formal_execution.run_id
            or solver_run.result_ref != formal_result.result_id
        ):
            raise QualityGateError("CAUSAL_HOLDOUT_FORMAL_RESULT_NOT_BOUND")
        persisted_formal = self._mathematics.get_result_context(
            state.project_id, formal_result.result_id
        )
        if (
            not persisted_formal.evidence.valid
            or persisted_formal.model != model
            or persisted_formal.result != formal_result
            or persisted_formal.solver_run != solver_run
            or persisted_formal.execution != formal_execution
            or persisted_formal.program != program
        ):
            raise QualityGateError("CAUSAL_HOLDOUT_PERSISTED_FORMAL_RESULT_MISMATCH")
        predictors = [item for item in program.files if item.path == "causal_predictor.py"]
        if len(predictors) != 1:
            raise QualityGateError("CAUSAL_HOLDOUT_PREDICTOR_SOURCE_MISSING")
        predictor = predictors[0]
        if hashlib.sha256(predictor.content.encode("utf-8")).hexdigest() != predictor.sha256:
            raise QualityGateError("CAUSAL_HOLDOUT_PREDICTOR_SOURCE_MISMATCH")
        spec = CausalBinarySpec(
            source_sha256=policy.source_sha256,
            group_column=policy.group_column,
            condition_column=policy.condition_column,
            outcome_column=policy.outcome_column,
            positive_value=policy.positive_value,
            negative_value=policy.negative_value,
            history_window=policy.history_window,
        )
        run = run_isolated_causal_holdout(
            source,
            spec,
            predictor.content,
            fraction=policy.fraction,
            salt=policy.salt,
            store=self._store,
            root=self._root,
            image=self._image,
            limits=self._limits,
            project_id=state.project_id,
            problem_id=state.problem_id,
            execution_origin=ExecutionOrigin.GENERATED_PROGRAM,
            model_digest=program.model_digest,
            generated_program_id=program.program_id,
            formal_result_id=formal_result.result_id,
        )
        self._repository.persist_auxiliary_execution(run.execution, list(run.artifacts))
        if run.execution.status is not ExecutionStatus.SUCCEEDED or run.result is None:
            raise QualityGateError(
                "CAUSAL_HOLDOUT_EXECUTION_FAILED: " + (run.execution.error or "unknown")[:300]
            )
        result = self.audit_persisted(
            bundle=bundle,
            project_id=state.project_id,
            formal_result_id=formal_result.result_id,
            holdout_execution_id=run.execution.run_id,
        )
        if result != run.result:
            raise QualityGateError("CAUSAL_HOLDOUT_TRACE_RECOMPUTATION_MISMATCH")
        return run

    def audit_persisted(
        self,
        *,
        bundle: BlindSolveBundle,
        project_id: UUID,
        formal_result_id: UUID,
        holdout_execution_id: UUID,
    ) -> CausalHoldoutResult:
        """Rebuild the holdout result from stored formal evidence and source bytes."""
        policy = bundle.causal_policy
        split = bundle.causal_split
        if policy is None or split is None:
            raise QualityGateError("CAUSAL_HOLDOUT_POLICY_MISSING")
        source = next(
            (
                item.content
                for item in bundle.artifacts
                if item.resource.resource_id == policy.resource_id
            ),
            None,
        )
        if source is None or hashlib.sha256(source).hexdigest() != split.source_sha256:
            raise QualityGateError("CAUSAL_HOLDOUT_SOURCE_MISMATCH")
        resource = next(
            item.resource
            for item in bundle.artifacts
            if item.resource.resource_id == policy.resource_id
        )
        registered = [
            item
            for item in self._repository.list_files(project_id)
            if item.original_name == resource.local_filename
            and item.sha256 == split.training_sha256
        ]
        if len(registered) != 1:
            raise QualityGateError("CAUSAL_HOLDOUT_TRAINING_INPUT_NOT_BOUND")
        training_bytes = self._store.read_bytes(
            registered[0].storage_key, max_bytes=16 * 1024 * 1024
        )
        if training_bytes != split.training_csv:
            raise QualityGateError("CAUSAL_HOLDOUT_TRAINING_BYTES_MISMATCH")
        formal = self._mathematics.get_result_context(project_id, formal_result_id)
        if (
            not formal.evidence.valid
            or formal.program is None
            or formal.program.is_mock
            or formal.program.status is not GeneratedProgramStatus.EXECUTED
            or formal.execution.is_mock
            or formal.execution.status is not ExecutionStatus.SUCCEEDED
            or formal.result.result_id != formal_result_id
            or formal.result.execution_record_id != formal.execution.run_id
            or formal.solver_run.result_ref != formal_result_id
            or formal.solver_run.execution_ref != formal.execution.run_id
            or formal.solver_run.generated_program_id != formal.program.program_id
        ):
            raise QualityGateError("CAUSAL_HOLDOUT_FORMAL_RESULT_NOT_BOUND")
        stored_executions = [
            item
            for item in self._repository.list_executions(project_id)
            if item.run_id in {formal.execution.run_id, holdout_execution_id}
        ]
        holdouts = [item for item in stored_executions if item.run_id == holdout_execution_id]
        stored_artifacts = {
            item.artifact_id: item for item in self._repository.list_artifacts(project_id)
        }
        if (
            len(stored_executions) != 2
            or formal.execution not in stored_executions
            or len(holdouts) != 1
        ):
            raise QualityGateError("CAUSAL_HOLDOUT_PERSISTED_EVIDENCE_MISMATCH")
        holdout = holdouts[0]
        if (
            holdout.execution_origin is not ExecutionOrigin.GENERATED_PROGRAM
            or holdout.model_digest != mathematical_model_digest(formal.model)
            or holdout.generated_program_id != formal.program.program_id
            or holdout.environment.get("formal_result_id") != str(formal_result_id)
            or holdout.environment.get("source_sha256") != split.source_sha256
        ):
            raise QualityGateError("CAUSAL_HOLDOUT_FORMAL_RESULT_NOT_BOUND")
        named = [
            item
            for item in stored_artifacts.values()
            if item.execution_run_id == holdout_execution_id
        ]
        by_name = {item.name: item for item in named}
        if len(named) != 3 or set(by_name) != {
            "predictor.py",
            "causal_driver.py",
            "causal-holdout.json",
        }:
            raise QualityGateError("CAUSAL_HOLDOUT_PERSISTED_EVIDENCE_MISMATCH")
        predictor = next(
            (item for item in formal.program.files if item.path == "causal_predictor.py"), None
        )
        if predictor is None or by_name["predictor.py"].sha256 != predictor.sha256:
            raise QualityGateError("CAUSAL_HOLDOUT_PREDICTOR_SOURCE_MISMATCH")
        run = IsolatedCausalHoldout(
            result=None,
            execution=holdout,
            artifacts=(
                by_name["predictor.py"],
                by_name["causal_driver.py"],
                by_name["causal-holdout.json"],
            ),
        )
        expected_spec = CausalBinarySpec(
            source_sha256=policy.source_sha256,
            group_column=policy.group_column,
            condition_column=policy.condition_column,
            outcome_column=policy.outcome_column,
            positive_value=policy.positive_value,
            negative_value=policy.negative_value,
            history_window=policy.history_window,
        )
        result = verify_recorded_causal_holdout(
            source,
            run,
            self._store,
            expected_policy=(expected_spec, policy.fraction, policy.salt),
        )
        if (
            result.heldout_groups != split.heldout_groups
            or result.training_groups != split.training_groups
            or len(result.predictions) != split.heldout_rows
        ):
            raise QualityGateError("CAUSAL_HOLDOUT_TRACE_SPLIT_MISMATCH")
        return result
