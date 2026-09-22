"""Read-time reconstruction, not trust in persisted replay status or metric counts."""

import hashlib
import sys

from mathmodel_ai.files.storage import FileStore
from mathmodel_ai.mathematical.digests import mathematical_model_digest
from mathmodel_ai.schemas.execution import ExecutionOrigin, ExecutionStatus
from mathmodel_ai.schemas.independent_verification import ReplayRecord, ScenarioSpec
from mathmodel_ai.schemas.mathematical import MathematicalModel
from mathmodel_ai.schemas.program import GeneratedSourceFile, generated_program_hash
from mathmodel_ai.schemas.solver import SolverFamily, SolverStatus
from mathmodel_ai.solvers.programs import build_deterministic_program
from mathmodel_ai.verification.metric_recompute import content_digest
from mathmodel_ai.verification.scenario_replay import ScenarioReplayer


def audit_replay(
    model: MathematicalModel, spec: ScenarioSpec, record: ReplayRecord, store: FileStore
) -> list[str]:
    errors: list[str] = []
    values = ScenarioReplayer.parameters(model, spec)
    scenario = ScenarioReplayer.scenario_model(model, values)
    generator = f"python-random-MT19937-uniform:{sys.version_info.major}.{sys.version_info.minor}"
    expected_input = content_digest(
        {
            "model_digest": mathematical_model_digest(model),
            "scenario": spec.model_dump(mode="json"),
            "parameters": values,
            "generator": generator,
        }
    )
    if (
        record.scenario_id != spec.scenario_id
        or record.scenario_digest != content_digest(spec)
        or record.input_parameters != values
        or record.input_digest != expected_input
        or record.generator_version != generator
    ):
        errors.append("REPLAY_INPUT_MISMATCH")
    execution = record.execution
    if (
        execution.model_digest != mathematical_model_digest(scenario)
        or execution.project_id != model.project_id
        or execution.problem_id != model.problem_id
    ):
        errors.append("REPLAY_MODEL_BINDING_MISMATCH")
    if (
        execution.status is not ExecutionStatus.SUCCEEDED
        or execution.exit_code != 0
        or execution.is_mock
        or not execution.image_id
        or not all((execution.network_disabled, execution.non_root, execution.read_only_root))
        or execution.execution_origin is not ExecutionOrigin.DETERMINISTIC_SOLVER_ADAPTER
    ):
        errors.append("REPLAY_NOT_REAL_ISOLATED_SUCCESS")
    if spec.dynamic is not None:
        source = ScenarioReplayer.dynamic_code(model, spec, values)
        if execution.limits.timeout_seconds != spec.timeout_seconds:
            errors.append("REPLAY_LIMIT_MISMATCH")
    else:
        program, options = record.program, record.solver_options
        if program is None or options is None:
            return [*errors, "REPLAY_PROGRAM_MISSING"]
        backend = program.generated_by.removeprefix("deterministic:")
        if backend not in {"scipy", "ortools", "gurobi"}:
            return [*errors, "REPLAY_BACKEND_INVALID"]
        expected = build_deterministic_program(
            scenario,
            options,
            backend=backend,
            target=SolverFamily(program.solver_target),
            dependencies=program.dependencies,
        )
        source = expected.files[0].content
        if (
            program.is_mock
            or program.code_hash != expected.code_hash
            or program.model_digest != mathematical_model_digest(scenario)
            or execution.generated_program_id != program.program_id
            or record.solver_status not in {SolverStatus.OPTIMAL, SolverStatus.FEASIBLE}
            or options.time_limit_seconds is None
            or options.time_limit_seconds > spec.timeout_seconds
        ):
            errors.append("REPLAY_SOLVER_BINDING_INVALID")
    expected_hash = hashlib.sha256(source.encode()).hexdigest()
    expected_bundle = generated_program_hash([GeneratedSourceFile(path="main.py", content=source)])
    if execution.code_hash != expected_hash or execution.executed_bundle_hash != expected_bundle:
        errors.append("REPLAY_EXECUTED_CODE_MISMATCH")
    artifacts = {a.artifact_id: a for a in record.artifacts}
    if len(artifacts) != len(record.artifacts):
        errors.append("REPLAY_DUPLICATE_ARTIFACT")
    code = artifacts.get(execution.code_artifact_id)
    output = artifacts.get(record.output_artifact_id) if record.output_artifact_id else None
    if code is None or code.sha256 != expected_hash:
        errors.append("REPLAY_CODE_ARTIFACT_MISMATCH")
    if output is None or output.sha256 != record.output_digest:
        errors.append("REPLAY_OUTPUT_MISSING")
    if output and not any(
        a.artifact_id == output.artifact_id and a.sha256 == output.sha256
        for a in execution.artifacts
    ):
        errors.append("REPLAY_OUTPUT_EXECUTION_MISMATCH")
    for artifact in artifacts.values():
        if artifact.project_id != model.project_id or artifact.execution_run_id != execution.run_id:
            errors.append("REPLAY_ARTIFACT_BINDING_MISMATCH")
        data = store.read_bytes(artifact.storage_key, max_bytes=16 * 1024 * 1024)
        if len(data) != artifact.size_bytes or hashlib.sha256(data).hexdigest() != artifact.sha256:
            errors.append("REPLAY_ARTIFACT_TAMPER")
    return list(dict.fromkeys(errors))
