import hashlib
import math
from uuid import uuid4

import pytest

from mathmodel_ai.schemas.mathematical import ExpressionKind, MathExpression
from mathmodel_ai.verification.causal_binary import CausalBinarySpec, iter_causal_binary_points
from mathmodel_ai.verification.empirical_binary import (
    EmpiricalBinaryRisk,
    audit_empirical_binary_claim,
    empirical_source_spec,
    evaluate_empirical_binary_risk,
    predict_empirical_binary,
    row_feature_values,
)


def _risk(source: bytes) -> EmpiricalBinaryRisk:
    spec = CausalBinarySpec(
        source_sha256=hashlib.sha256(source).hexdigest(),
        group_column="match",
        condition_column="server",
        outcome_column="winner",
        positive_value="1",
        negative_value="2",
        history_window=2,
    )
    return EmpiricalBinaryRisk(
        training_dataset_id=uuid4(),
        training_sha256=spec.source_sha256,
        group_column=spec.group_column,
        condition_column=spec.condition_column,
        outcome_column=spec.outcome_column,
        positive_value=spec.positive_value,
        negative_value=spec.negative_value,
        history_window=spec.history_window,
        reference_condition="1",
        source_refs=["EVID-fact-1"],
        logit=MathExpression(
            kind=ExpressionKind.MULTIPLY,
            operands=[
                MathExpression.symbol_ref("beta"),
                MathExpression.symbol_ref("row_condition_delta"),
            ],
        ),
    )


def test_empirical_risk_recomputes_every_row_and_changes_with_coefficients() -> None:
    source = b"match,server,winner\nA,1,1\nA,2,2\nA,1,1\nB,1,2\n"
    risk = _risk(source)
    zero = evaluate_empirical_binary_risk(source, risk, {"beta": 0.0})
    assert zero.row_count == 4
    assert zero.source_sha256 == hashlib.sha256(source).hexdigest()
    assert zero.mean_log_loss == pytest.approx(math.log(2))
    assert zero.mean_brier == pytest.approx(0.25)

    fitted = evaluate_empirical_binary_risk(source, risk, {"beta": 2.0})
    rows = list(iter_causal_binary_points(source, empirical_source_spec(risk)))
    expected = []
    for feature, outcome in rows:
        probability, _ = predict_empirical_binary(risk, feature, {"beta": 2.0})
        expected.append(-math.log(probability if outcome else 1 - probability))
    assert fitted.mean_log_loss == pytest.approx(sum(expected) / 4)
    assert fitted.mean_log_loss != pytest.approx(zero.mean_log_loss)


def test_empirical_features_are_pre_outcome_and_source_hash_is_enforced() -> None:
    source = b"match,server,winner\nA,1,1\nA,2,2\nA,1,1\n"
    risk = _risk(source)
    original = list(iter_causal_binary_points(source, empirical_source_spec(risk)))
    first = row_feature_values(original[0][0], reference_condition="1")
    assert first["row_condition_rate"] == first["row_group_rate"] == 0.5
    assert first["row_condition_reference"] == 1.0
    assert row_feature_values(original[1][0], reference_condition="1")[
        "row_group_rate"
    ] == pytest.approx(2 / 3)
    with pytest.raises(ValueError, match="CAUSAL_CSV_SIZE_OR_HASH_MISMATCH"):
        evaluate_empirical_binary_risk(source.replace(b"A,2,2", b"A,2,1"), risk, {"beta": 1.0})
    with pytest.raises(ValueError, match="EMPIRICAL_COEFFICIENTS_INVALID"):
        evaluate_empirical_binary_risk(source, risk, {"beta": float("nan")})
    with pytest.raises(ValueError, match="EMPIRICAL_COEFFICIENTS_INVALID"):
        evaluate_empirical_binary_risk(source, risk, {"beta": 1.0, "row_group_rate": 0.0})


def test_empirical_risk_rejects_constant_row_free_logit() -> None:
    source = b"match,server,winner\nA,1,1\n"
    with pytest.raises(ValueError, match="EMPIRICAL_OBJECTIVE_HAS_NO_ROW_FEATURE"):
        EmpiricalBinaryRisk(
            training_dataset_id=uuid4(),
            training_sha256=_risk(source).training_sha256,
            group_column="match",
            condition_column="server",
            outcome_column="winner",
            positive_value="1",
            negative_value="2",
            history_window=2,
            reference_condition="1",
            source_refs=["EVID-fact-1"],
            logit=MathExpression.symbol_ref("beta"),
        )


def test_empirical_risk_contract_round_trips_without_source_bytes() -> None:
    source = b"match,server,winner\nA,1,1\n"
    risk = _risk(source)
    restored = EmpiricalBinaryRisk.model_validate_json(risk.model_dump_json())
    assert restored == risk
    assert "match,server,winner" not in risk.model_dump_json()


def test_empirical_claim_binds_training_loss_and_heldout_predictions() -> None:
    training = b"match,server,winner\nA,1,1\nA,2,2\nA,1,1\n"
    official = training + b"B,1,2\nB,2,1\n"
    risk = _risk(training)
    official_spec = _risk(official)
    coefficients = {"beta": 1.0}
    reported_loss = evaluate_empirical_binary_risk(training, risk, coefficients).mean_log_loss
    predictions = [
        predict_empirical_binary(risk, feature, coefficients)[0]
        for feature, _ in iter_causal_binary_points(official, empirical_source_spec(official_spec))
        if feature.group == "B"
    ]
    assert len(predictions) == 2
    claim = dict(
        risk=risk,
        training_csv=training,
        official_csv=official,
        official_spec=empirical_source_spec(official_spec),
        heldout_groups={"B"},
        coefficients=coefficients,
        reported_training_log_loss=reported_loss,
        reported_heldout_predictions=predictions,
    )
    assert audit_empirical_binary_claim(**claim).row_count == 3
    with pytest.raises(ValueError, match="EMPIRICAL_TRAINING_LOSS_MISMATCH"):
        audit_empirical_binary_claim(**{**claim, "reported_training_log_loss": reported_loss + 0.1})
    with pytest.raises(ValueError, match="EMPIRICAL_HELDOUT_PREDICTION_MISMATCH"):
        audit_empirical_binary_claim(
            **{**claim, "reported_heldout_predictions": [predictions[0] + 0.1, predictions[1]]}
        )
    with pytest.raises(ValueError, match="EMPIRICAL_TRAINING_CONTAINS_HELDOUT_GROUP"):
        audit_empirical_binary_claim(
            **{
                **claim,
                "risk": risk.model_copy(
                    update={"training_sha256": _risk(official).training_sha256}
                ),
                "training_csv": official,
                "reported_training_log_loss": evaluate_empirical_binary_risk(
                    official,
                    risk.model_copy(update={"training_sha256": _risk(official).training_sha256}),
                    coefficients,
                ).mean_log_loss,
            }
        )
