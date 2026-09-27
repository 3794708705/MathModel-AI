"""Independent audit of an isolated fixed-coefficient empirical scenario.

The scenario predictor is a deterministic rendering of the formal logit, not
the original generated fitter. This is a fixed-coefficient perturbation, never
a claim that the coefficients were re-estimated on held-out observations.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from uuid import UUID

from mathmodel_ai.benchmark.causal_inputs import split_causal_csv
from mathmodel_ai.files.storage import FileStore
from mathmodel_ai.mathematical.digests import mathematical_model_digest
from mathmodel_ai.sandbox.causal_holdout import (
    IsolatedCausalHoldout,
    verify_recorded_causal_holdout,
)
from mathmodel_ai.sandbox.empirical_predictor import empirical_predictor_source
from mathmodel_ai.schemas.execution import ExecutionOrigin
from mathmodel_ai.schemas.mathematical import MathematicalModel
from mathmodel_ai.verification.causal_binary import CausalBinarySpec
from mathmodel_ai.verification.causal_holdout import CausalHoldoutResult
from mathmodel_ai.verification.empirical_binary import (
    EmpiricalRiskResult,
    audit_empirical_binary_claim,
    evaluate_empirical_binary_risk,
)


@dataclass(frozen=True)
class AuditedEmpiricalScenario:
    execution_id: UUID
    formal_result_id: UUID
    model_digest: str
    predictor_sha256: str
    training: EmpiricalRiskResult
    holdout: CausalHoldoutResult


def audit_isolated_empirical_scenario(
    *,
    model: MathematicalModel,
    formal_result_id: UUID,
    official_csv: bytes,
    official_spec: CausalBinarySpec,
    training_csv: bytes,
    fraction: float,
    salt: str,
    coefficients: dict[str, float],
    run: IsolatedCausalHoldout,
    store: FileStore,
) -> AuditedEmpiricalScenario:
    """Rebuild the exact split and every forecast from persisted execution bytes."""
    risk = model.empirical_binary_risk
    if risk is None:
        raise ValueError("EMPIRICAL_SCENARIO_MODEL_RISK_MISSING")
    split = split_causal_csv(official_csv, official_spec, fraction=fraction, salt=salt)
    if training_csv != split.training_csv or risk.training_sha256 != split.training_sha256:
        raise ValueError("EMPIRICAL_SCENARIO_TRAINING_SPLIT_MISMATCH")
    record = run.execution
    digest = mathematical_model_digest(model)
    if (
        record.execution_origin is not ExecutionOrigin.DETERMINISTIC_SOLVER_ADAPTER
        or record.project_id != model.project_id
        or record.problem_id != model.problem_id
        or record.model_digest != digest
        or record.generated_program_id is not None
        or record.environment.get("formal_result_id") != str(formal_result_id)
    ):
        raise ValueError("EMPIRICAL_SCENARIO_FORMAL_RESULT_NOT_BOUND")
    expected_code = empirical_predictor_source(risk, coefficients)
    predictor_sha256 = hashlib.sha256(expected_code.encode("utf-8")).hexdigest()
    if (
        record.code_hash != predictor_sha256
        or not run.artifacts
        or run.artifacts[0].sha256 != predictor_sha256
    ):
        raise ValueError("EMPIRICAL_SCENARIO_PREDICTOR_NOT_BOUND")
    holdout = verify_recorded_causal_holdout(
        official_csv,
        run,
        store,
        expected_policy=(official_spec, fraction, salt),
    )
    if holdout.heldout_groups != split.heldout_groups:
        raise ValueError("EMPIRICAL_SCENARIO_HELDOUT_SPLIT_MISMATCH")
    training = evaluate_empirical_binary_risk(training_csv, risk, coefficients)
    audited = audit_empirical_binary_claim(
        risk=risk,
        training_csv=training_csv,
        official_csv=official_csv,
        official_spec=official_spec,
        heldout_groups=holdout.heldout_groups,
        coefficients=coefficients,
        reported_training_log_loss=training.mean_log_loss,
        reported_heldout_predictions=holdout.predictions,
    )
    if audited != training:
        raise ValueError("EMPIRICAL_SCENARIO_TRAINING_RECOMPUTATION_MISMATCH")
    return AuditedEmpiricalScenario(
        execution_id=record.run_id,
        formal_result_id=formal_result_id,
        model_digest=digest,
        predictor_sha256=predictor_sha256,
        training=training,
        holdout=holdout,
    )
