from __future__ import annotations

import hashlib
import math
import random
import sys
from pathlib import Path
from uuid import uuid4

from mathmodel_ai.files.storage import FileStore
from mathmodel_ai.mathematical.algorithms import AlgorithmSelector
from mathmodel_ai.mathematical.digests import mathematical_model_digest
from mathmodel_ai.sandbox.executor import SandboxExecutor
from mathmodel_ai.schemas.execution import ExecutionOrigin, ExecutionStatus, SandboxLimits
from mathmodel_ai.schemas.independent_verification import (
    IndependentStatus,
    RawMetricOutput,
    ReplayRecord,
    ScenarioSpec,
)
from mathmodel_ai.schemas.mathematical import MathematicalModel
from mathmodel_ai.schemas.solver import SolverOptions, SolverStatus
from mathmodel_ai.solvers.router import SolverRouter
from mathmodel_ai.verification.metric_recompute import canonical_bytes, content_digest, recompute
from mathmodel_ai.verification.replay_runtime import DYNAMIC_REPLAY_CODE


class ScenarioReplayer:
    def __init__(self, *, store: FileStore, root: Path, image: str, solver_router: SolverRouter):
        self._store = store
        self._root = root
        self._image = image
        self._solvers = solver_router

    @staticmethod
    def parameters(model: MathematicalModel, spec: ScenarioSpec) -> dict[str, float]:
        values: dict[str, float] = {}
        for p in [*model.parameters, *model.constants]:
            if isinstance(p.value, bool) or not isinstance(p.value, int | float):
                raise ValueError("scenario requires finite scalar parameters")
            if not math.isfinite(p.value):
                raise ValueError("scenario requires finite scalar parameters")
            values[p.symbol] = float(p.value)
        mutable = {p.symbol for p in model.parameters}
        if not set(spec.parameter_values) <= mutable:
            raise ValueError("scenario cannot change unknown parameters or model constants")
        values.update(spec.parameter_values)
        rng = random.Random(spec.seed)
        if spec.noise_fraction:
            for symbol in sorted(spec.parameter_values):
                values[symbol] *= 1 + rng.uniform(-spec.noise_fraction, spec.noise_fraction)
        if not set(spec.decision_values) <= {v.symbol for v in model.decision_variables}:
            raise ValueError("scenario decision binding is unknown")
        return values

    @staticmethod
    def scenario_model(model: MathematicalModel, values: dict[str, float]) -> MathematicalModel:
        return model.model_copy(
            update={
                "parameters": [
                    p.model_copy(update={"value": values[p.symbol]}) for p in model.parameters
                ]
            }
        )

    @staticmethod
    def dynamic_code(model: MathematicalModel, spec: ScenarioSpec, values: dict[str, float]) -> str:
        if spec.dynamic is None:
            raise ValueError("explicit dynamic binding is required")
        bindings = spec.dynamic.equation_by_state
        if not set(bindings) <= {v.symbol for v in model.state_variables}:
            raise ValueError("dynamic state binding is not in the mathematical model")
        if not set(bindings.values()) <= {e.equation_id for e in model.equations}:
            raise ValueError("dynamic derivative equation is not in the mathematical model")
        if len(set(bindings.values())) != len(bindings):
            raise ValueError("a derivative equation cannot ambiguously bind multiple states")
        payload = canonical_bytes(
            {
                "model": model.model_dump(mode="json"),
                "scenario": spec.model_dump(mode="json"),
                "parameters": values,
            }
        ).decode()
        return DYNAMIC_REPLAY_CODE + "\nrun(json.loads(" + repr(payload) + "))\n"

    def execute(
        self,
        model: MathematicalModel,
        spec: ScenarioSpec,
        *,
        observations: list[float] | None = None,
    ) -> ReplayRecord:
        values = self.parameters(model, spec)
        scenario = self.scenario_model(model, values)
        generator = (
            f"python-random-MT19937-uniform:{sys.version_info.major}.{sys.version_info.minor}"
        )
        input_digest = content_digest(
            {
                "model_digest": mathematical_model_digest(model),
                "scenario": spec.model_dump(mode="json"),
                "parameters": values,
                "generator": generator,
            }
        )
        program = None
        options = None
        solver_status = None
        if spec.dynamic is not None:
            code = self.dynamic_code(model, spec, values)
            sandbox = SandboxExecutor(
                store=self._store,
                root=self._root,
                image=self._image,
                limits=SandboxLimits(
                    cpu_cores=1,
                    memory_mb=512,
                    timeout_seconds=spec.timeout_seconds,
                    pids_limit=64,
                    max_output_bytes=1000000,
                    max_artifacts=4,
                    max_artifact_bytes=16 * 1024 * 1024,
                ),
            )
            execution = sandbox.execute(
                code,
                project_id=model.project_id,
                problem_id=model.problem_id,
                model_digest=mathematical_model_digest(scenario),
                execution_origin=ExecutionOrigin.DETERMINISTIC_SOLVER_ADAPTER,
            )
            output_name = "metrics.json"
        else:
            if spec.decision_values:
                raise ValueError(
                    "deterministic optimizer replay cannot override decision variables"
                )
            routed = self._solvers.route(
                scenario,
                AlgorithmSelector().select(scenario),
                SolverOptions(time_limit_seconds=spec.timeout_seconds),
            )
            solved = routed.solver.solve(scenario, routed.options)
            execution = solved.execution
            program = solved.program
            options = routed.options
            solver_status = solved.result.status
            output_name = "result.json"
        artifact = next((a for a in execution.artifact_records if a.name == output_name), None)
        metrics = []
        error = None
        status = IndependentStatus.INVALID
        if (
            execution.record.status is ExecutionStatus.SUCCEEDED
            and execution.record.exit_code == 0
            and not execution.record.is_mock
            and artifact is not None
            and (
                spec.dynamic is not None
                or solver_status in {SolverStatus.OPTIMAL, SolverStatus.FEASIBLE}
            )
        ):
            try:
                data = self._store.read_bytes(artifact.storage_key, max_bytes=16 * 1024 * 1024)
                if hashlib.sha256(data).hexdigest() != artifact.sha256:
                    raise ValueError("replay artifact digest mismatch")
                raw = parse_raw_output(data)
                metrics = [
                    recompute(
                        m,
                        raw,
                        source_digest=artifact.sha256,
                        model=scenario,
                        observations=observations,
                    )
                    for m in spec.metrics
                ]
                required = [
                    m
                    for m in metrics
                    if m.metric_id in {s.metric_id for s in spec.metrics if s.required}
                ]
                status = (
                    IndependentStatus.PASS
                    if all(m.status is IndependentStatus.PASS for m in required)
                    else IndependentStatus.FAIL
                )
            except ValueError as exc:
                error = str(exc)
        else:
            error = "replay execution or required output was unsuccessful"
        return ReplayRecord(
            replay_id=uuid4(),
            scenario_id=spec.scenario_id,
            scenario_digest=content_digest(spec),
            input_digest=input_digest,
            input_parameters=values,
            generator_version=generator,
            execution=execution.record,
            program=program,
            solver_options=options,
            solver_status=solver_status,
            artifacts=execution.artifact_records,
            output_artifact_id=artifact.artifact_id if artifact else None,
            output_digest=artifact.sha256 if artifact else None,
            status=status,
            metrics=metrics,
            error=error,
        )


def parse_raw_output(data: bytes) -> RawMetricOutput:
    import json

    from mathmodel_ai.schemas.solver import GeneratedResultPayload
    from mathmodel_ai.solvers.base import RawSolverOutput

    payload = json.loads(data)
    if not isinstance(payload, dict):
        raise ValueError("raw metric artifact must be a JSON object")
    if "native_status" in payload:
        native = RawSolverOutput.model_validate(payload)
        return RawMetricOutput(
            variables=native.variables,
            reported={"objective": native.objective} if native.objective is not None else {},
        )
    if "solver_name" in payload:
        generated = GeneratedResultPayload.model_validate(payload)
        reported = {
            key: float(value)
            for key, value in generated.metrics.items()
            if isinstance(value, int | float) and not isinstance(value, bool)
        }
        if generated.objective is not None:
            if "objective" in reported and reported["objective"] != generated.objective:
                raise ValueError("conflicting reported objectives")
            reported["objective"] = generated.objective
        return RawMetricOutput(
            variables=generated.variable_values,
            predictions=generated.predictions,
            series=generated.series,
            reported=reported,
        )
    return RawMetricOutput.model_validate(payload)
