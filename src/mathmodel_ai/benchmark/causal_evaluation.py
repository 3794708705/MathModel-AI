"""Bind an automatically generated predictor to sealed benchmark evaluation."""

import hashlib
from pathlib import Path

from mathmodel_ai.benchmark.manifests import BlindSolveBundle
from mathmodel_ai.core.errors import QualityGateError
from mathmodel_ai.data.repository import DataRepository
from mathmodel_ai.files.storage import FileStore
from mathmodel_ai.mathematical.digests import mathematical_model_digest
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


class CausalBenchmarkEvaluator:
    """Run and persist a model-generated predictor, without trusting its metrics."""

    def __init__(
        self,
        *,
        store: FileStore,
        repository: DataRepository,
        root: Path,
        image: str,
        limits: SandboxLimits,
    ) -> None:
        self._store = store
        self._repository = repository
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
        program = mathematical.solve_stage.execution.program
        if (
            program.execution_origin is not ExecutionOrigin.GENERATED_PROGRAM
            or program.status is not GeneratedProgramStatus.EXECUTED
            or program.is_mock
            or program.model_digest != mathematical_model_digest(mathematical.model_stage.model)
        ):
            raise QualityGateError("CAUSAL_HOLDOUT_GENERATED_PROGRAM_NOT_BOUND")
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
        )
        self._repository.persist_auxiliary_execution(run.execution, list(run.artifacts))
        if run.execution.status is not ExecutionStatus.SUCCEEDED or run.result is None:
            raise QualityGateError(
                "CAUSAL_HOLDOUT_EXECUTION_FAILED: " + (run.execution.error or "unknown")[:300]
            )
        result = verify_recorded_causal_holdout(source, run, self._store)
        if (
            result.heldout_groups != split.heldout_groups
            or result.training_groups != split.training_groups
            or len(result.predictions) != split.heldout_rows
        ):
            raise QualityGateError("CAUSAL_HOLDOUT_TRACE_SPLIT_MISMATCH")
        return run
