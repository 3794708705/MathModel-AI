import hashlib
import json
from dataclasses import replace

import pytest

from mathmodel_ai.verification.causal_binary import CausalBinarySpec, CausalFeature
from mathmodel_ai.verification.causal_holdout import (
    CausalHoldoutResult,
    assess_binary_calibration,
    assess_conditional_randomness,
    assess_imminent_swing,
    assess_match_flow,
    causal_trace_payload,
    evaluate_causal_holdout,
    heldout_groups,
    verify_causal_trace,
)


class RecordingPredictor:
    def __init__(self) -> None:
        self.fit_groups: list[str] = []
        self.queries: list[CausalFeature] = []

    def fit(self, feature: CausalFeature, outcome: float) -> None:
        assert outcome in {0.0, 1.0}
        self.fit_groups.append(feature.group)

    def predict(self, feature: CausalFeature) -> float:
        self.queries.append(feature)
        return (1 + feature.prior_condition_positive) / (2 + feature.prior_condition_count)


def _fixture() -> tuple[bytes, CausalBinarySpec]:
    source = (
        b"match,server,winner,after_point\n"
        b"A,1,1,1\nA,1,2,2\nA,2,1,3\n"
        b"B,2,2,1\nB,2,1,2\nB,1,2,3\n"
        b"C,1,1,1\nC,2,1,2\nC,1,2,3\n"
        b"D,2,2,1\nD,1,2,2\nD,2,1,3\n"
    )
    spec = CausalBinarySpec(
        source_sha256=hashlib.sha256(source).hexdigest(),
        group_column="match",
        condition_column="server",
        outcome_column="winner",
        positive_value="1",
        negative_value="2",
        history_window=2,
    )
    return source, spec


def test_group_holdout_is_identity_only_and_sequential() -> None:
    source, spec = _fixture()
    predictor = RecordingPredictor()
    result = evaluate_causal_holdout(source, spec, predictor, fraction=0.25, salt="blind-v1")
    assert len(result.heldout_groups) == 1
    assert len(result.training_groups) == 3
    assert set(predictor.fit_groups) == set(result.training_groups)
    assert set(predictor.fit_groups).isdisjoint(result.heldout_groups)
    assert len(predictor.queries) == len(result.predictions) == len(result.observations) == 3
    assert all(query.group in result.heldout_groups for query in predictor.queries)
    assert [query.point_index for query in predictor.queries] == [0, 1, 2]
    assert result.predictions == result.baseline_predictions
    assert result.brier == pytest.approx(result.baseline_brier)
    assert heldout_groups({"A", "B", "C", "D"}, fraction=0.25, salt="blind-v1") == (
        result.heldout_groups
    )


def test_holdout_rejects_invalid_design_and_probabilities() -> None:
    source, spec = _fixture()
    with pytest.raises(ValueError, match="GROUP_HOLDOUT_DESIGN_INVALID"):
        heldout_groups({"A"}, fraction=0.5, salt="blind-v1")
    with pytest.raises(ValueError, match="GROUP_HOLDOUT_DESIGN_INVALID"):
        heldout_groups({"A", "B"}, fraction=1, salt="blind-v1")

    class InvalidPredictor(RecordingPredictor):
        def predict(self, feature: CausalFeature) -> float:
            del feature
            return float("nan")

    with pytest.raises(ValueError, match="HELDOUT_PREDICTION_NOT_A_PROBABILITY"):
        evaluate_causal_holdout(source, spec, InvalidPredictor(), fraction=0.25, salt="blind-v1")


def test_trace_recomputes_labels_split_baseline_and_loss() -> None:
    source, spec = _fixture()
    result = evaluate_causal_holdout(source, spec, RecordingPredictor(), fraction=0.25, salt="v1")
    payload = causal_trace_payload(spec, result, fraction=0.25, salt="v1")
    assert verify_causal_trace(source, payload) == result
    changed = json.loads(payload)
    changed["observations"][0] = 1 - changed["observations"][0]
    with pytest.raises(ValueError, match="CAUSAL_TRACE_RECOMPUTATION_MISMATCH"):
        verify_causal_trace(source, json.dumps(changed).encode())
    changed = json.loads(payload)
    changed["brier"] = 0
    with pytest.raises(ValueError, match="CAUSAL_TRACE_RECOMPUTATION_MISMATCH"):
        verify_causal_trace(source, json.dumps(changed).encode())
    altered_source = source.replace(b"A,1,1,1", b"A,1,2,1")
    with pytest.raises(ValueError, match="CAUSAL_CSV_SIZE_OR_HASH_MISMATCH"):
        verify_causal_trace(altered_source, payload)


def test_calibration_recomputes_fixed_probability_bins_without_skill_claim() -> None:
    result = CausalHoldoutResult(
        heldout_groups=("B",),
        training_groups=("A",),
        predictions=(0.0, 0.2, 0.8, 1.0),
        observations=(0.0, 1.0, 1.0, 0.0),
        baseline_predictions=(0.5,) * 4,
        brier=0.42,
        baseline_brier=0.25,
    )
    assessment = assess_binary_calibration(result, bin_count=5)
    assert assessment.count == 4
    assert assessment.ece == pytest.approx(0.4)
    assert [item.count for item in assessment.bins] == [1, 1, 0, 0, 2]
    assert assessment.bins[4].mean_prediction == pytest.approx(0.9)
    assert assessment.bins[4].observed_rate == pytest.approx(0.5)
    with pytest.raises(ValueError, match="CAUSAL_CALIBRATION_INPUT_INVALID"):
        assess_binary_calibration(replace(result, predictions=(float("nan"),)))


def test_conditional_randomness_is_source_bound_and_not_a_momentum_claim() -> None:
    source, spec = _fixture()
    first = assess_conditional_randomness(source, spec, replicates=99)
    assert first == assess_conditional_randomness(source, spec, replicates=99)
    assert first.point_count == 12
    assert first.transition_count == 8
    assert 0 < first.two_sided_p <= 1
    assert first.seed_sha256 == hashlib.sha256(b"conditional-randomness-v1:" + source).hexdigest()
    changed = source.replace(b"A,1,1,1", b"A,1,2,1")
    with pytest.raises(ValueError, match="CAUSAL_CSV_SIZE_OR_HASH_MISMATCH"):
        assess_conditional_randomness(changed, spec, replicates=99)
    with pytest.raises(ValueError, match="CAUSAL_RANDOMNESS_REPLICATES_INVALID"):
        assess_conditional_randomness(source, spec, replicates=1)


def test_conditional_randomness_preserves_server_strata() -> None:
    source = b"match,server,winner\nA,1,1\nA,2,2\nA,1,1\nA,2,2\n"
    spec = CausalBinarySpec(
        source_sha256=hashlib.sha256(source).hexdigest(),
        group_column="match",
        condition_column="server",
        outcome_column="winner",
        positive_value="1",
        negative_value="2",
    )
    result = assess_conditional_randomness(source, spec, replicates=99)
    assert result.observed_statistic == 0
    assert result.null_mean == 0
    assert result.two_sided_p == 1


def test_match_flow_is_pre_outcome_grouped_and_source_bound() -> None:
    source, spec = _fixture()
    flow = assess_match_flow(source, spec)
    assert len(flow.values) == 12
    assert flow.source_sha256 == spec.source_sha256
    assert flow.window == spec.history_window
    assert flow.values[0] == 0.0
    assert flow.values[1] == pytest.approx(0.5)
    assert flow.values[3] == 0.0  # A new match cannot inherit A's point history.
    altered = source.replace(b"A,1,1,1", b"A,1,2,1")
    with pytest.raises(ValueError, match="CAUSAL_CSV_SIZE_OR_HASH_MISMATCH"):
        assess_match_flow(altered, spec)


def test_imminent_swing_uses_only_pre_outcome_forecasts_and_replays_labels() -> None:
    source, spec = _fixture()
    result = evaluate_causal_holdout(source, spec, RecordingPredictor(), fraction=0.25, salt="v1")
    assessed = assess_imminent_swing(source, spec, result)
    assert assessed.eligible_points == len(result.predictions) - spec.history_window
    assert 0 <= assessed.observed_swings <= assessed.eligible_points
    assert 0 <= assessed.brier <= 1
    assert assessed.brier == pytest.approx(assessed.baseline_brier)
    with pytest.raises(ValueError, match="CAUSAL_SWING_TRACE_INVALID"):
        assess_imminent_swing(source, spec, replace(result, observations=(1.0,) * 3))
    with pytest.raises(ValueError, match="CAUSAL_SWING_TRACE_INVALID"):
        assess_imminent_swing(source, spec, replace(result, baseline_predictions=()))


def test_imminent_swing_forecast_is_a_counterfactual_before_current_label() -> None:
    source = b"match,server,winner\nA,1,1\nB,1,1\nB,1,1\nB,1,2\n"
    spec = CausalBinarySpec(
        source_sha256=hashlib.sha256(source).hexdigest(),
        group_column="match",
        condition_column="server",
        outcome_column="winner",
        positive_value="1",
        negative_value="2",
        history_window=2,
    )
    result = CausalHoldoutResult(
        heldout_groups=("B",),
        training_groups=("A",),
        predictions=(0.5, 0.5, 0.8),
        observations=(1.0, 1.0, 0.0),
        baseline_predictions=(0.5, 2 / 3, 0.75),
        brier=0.38,
        baseline_brier=0.0,
    )
    assessment = assess_imminent_swing(source, spec, result)
    assert assessment.eligible_points == 1
    assert assessment.observed_swings == 1
    assert assessment.brier == pytest.approx(0.64)
    assert assessment.baseline_brier == pytest.approx(0.5625)
