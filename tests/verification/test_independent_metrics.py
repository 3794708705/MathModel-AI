import math

import pytest
from pydantic import ValidationError

from mathmodel_ai.schemas.independent_verification import (
    IndependentStatus,
    MetricKey,
    MetricSpec,
    RawMetricOutput,
    ScenarioSpec,
)
from mathmodel_ai.verification.metric_recompute import calculate, canonical_bytes, recompute
from tests.mathematical.helpers import lp_model


def spec(key, **kwargs):
    return MetricSpec(metric_id=key, key=key, version="1", **kwargs)


@pytest.mark.parametrize(
    "key,expected", [("mae", 1), ("rmse", math.sqrt(2)), ("max_error", 2), ("r2", -1)]
)
def test_known_independent_prediction_vectors(key, expected):
    raw = RawMetricOutput(predictions=[1.0, 5.0])
    assert calculate(spec(key), raw, observations=[1.0, 3.0]) == pytest.approx(expected)


def test_binary_scores_recompute_from_observations_and_reject_false_certainty() -> None:
    raw = RawMetricOutput(predictions=[0.8, 0.3, 0.9], reported={"brier": 0.01})
    observations = [1.0, 0.0, 1.0]
    assert calculate(spec("brier"), raw, observations=observations) == pytest.approx(
        (0.2**2 + 0.3**2 + 0.1**2) / 3
    )
    assert calculate(spec("log_loss"), raw, observations=observations) == pytest.approx(
        (-math.log(0.8) - math.log(0.7) - math.log(0.9)) / 3
    )
    compared = recompute(spec("brier"), raw, observations=observations, source_digest="a" * 64)
    assert compared.status is IndependentStatus.FAIL
    assert compared.verified is not None and compared.verified > compared.reported
    invalid = recompute(
        spec("log_loss"),
        RawMetricOutput(predictions=[0.0]),
        observations=[1.0],
        source_digest="a" * 64,
    )
    assert invalid.status is IndependentStatus.INVALID
    with pytest.raises(ValueError, match="binary scoring requires"):
        calculate(spec("brier"), RawMetricOutput(predictions=[1.2]), observations=[1.0])


@pytest.mark.parametrize(
    "key,expected", [("mean", 1), ("minimum", -2), ("maximum", 4), ("final_value", 1)]
)
def test_series_metrics_do_not_need_an_objective(key, expected):
    raw = RawMetricOutput(series={"population": [-2.0, 4.0, 1.0]})
    assert calculate(spec(key, series_key="population"), raw) == expected


def test_fake_reported_metric_cannot_override_verified_value():
    result = recompute(
        spec(
            "rmse",
            upper_threshold=0.1,
            threshold_provenance="Synthetic fixture acceptance threshold.",
        ),
        RawMetricOutput(predictions=[0.5, 0.5], reported={"rmse": 0.01}),
        source_digest="a" * 64,
        observations=[0.0, 0.0],
    )
    assert result.verified == 0.5
    assert result.reported == 0.01
    assert result.delta == -0.49
    assert result.status is IndependentStatus.FAIL


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, True])
def test_raw_nonfinite_or_boolean_predictions_rejected(value):
    with pytest.raises(ValidationError):
        RawMetricOutput(predictions=[value])


@pytest.mark.parametrize(
    "observations,predictions", [([], []), ([1.0], []), ([1.0, 1.0], [1.0, 1.0]), ([1.0], [1.0])]
)
def test_undefined_r2_is_invalid_not_coerced_to_success(observations, predictions):
    output = recompute(
        spec("r2"),
        RawMetricOutput(predictions=predictions),
        observations=observations,
        source_digest="a" * 64,
    )
    assert output.status is IndependentStatus.INVALID
    assert output.verified is None


def test_optimization_uses_model_ast_and_rejects_fake_feasibility():
    model = lp_model()
    raw = RawMetricOutput(variables={"x": 2.0, "y": 0.0}, reported={"feasible": 1.0})
    assert calculate(spec("objective"), raw, model=model) == 6
    assert calculate(spec("constraint_max_violation"), raw, model=model) == 8
    result = recompute(spec("feasible"), raw, model=model, source_digest="b" * 64)
    assert result.verified == 0 and result.status is IndependentStatus.FAIL
    assert (
        calculate(spec("mip_gap"), raw.model_copy(update={"best_bound": 3.0}), model=model) == 0.5
    )


@pytest.mark.parametrize(
    "actual,exact,expected",
    [(1.0 + 1e-8, False, "PASS"), (1.0 + 1e-5, False, "FAIL"), (1.0 + 1e-8, True, "FAIL")],
)
def test_reported_comparison_tolerance_and_exact_mode(actual, exact, expected):
    output = recompute(
        spec("mean", series_key="v", exact=exact),
        RawMetricOutput(series={"v": [1.0]}, reported={"mean": actual}),
        source_digest="a" * 64,
    )
    assert output.status.value == expected


def test_canonicalization_and_specs_reject_unsafe_values():
    assert canonical_bytes({"b": 2, "a": 1}) == b'{"a":1,"b":2}'
    with pytest.raises(ValueError):
        canonical_bytes({"x": math.nan})
    for kwargs in (
        {"key": "__import__"},
        {"version": "2"},
        {"version": None},
        {"absolute_tolerance": -1},
        {"lower_threshold": 2, "upper_threshold": 1},
        {"calculator_python_code": "bad"},
    ):
        with pytest.raises(ValidationError):
            MetricSpec.model_validate({"metric_id": "m", "key": "mae", "version": "1", **kwargs})
    with pytest.raises(ValidationError):
        spec("mean")
    with pytest.raises(ValidationError):
        spec("algebraic_scalar")
    with pytest.raises(ValidationError):
        spec("rmse", value_symbol="derived")
    with pytest.raises(ValidationError):
        spec("rmse", upper_threshold=1.0)
    with pytest.raises(ValidationError):
        ScenarioSpec(scenario_id="noise", version="1", noise_fraction=0.1, metrics=[spec("mae")])


def test_missing_raw_values_and_parameters_never_become_valid_metrics():
    for key in (MetricKey.OBJECTIVE, MetricKey.FEASIBLE, MetricKey.MIP_GAP):
        out = recompute(spec(key), RawMetricOutput(), source_digest="a" * 64, model=lp_model())
        assert out.status is IndependentStatus.INVALID


def test_algebraic_scalar_recomputes_ast_and_ignores_reported_derived_value():
    from mathmodel_ai.schemas.mathematical import VariableRole
    from tests.mathematical.helpers import add, symbol, variable

    model = lp_model()
    derived = variable("derived").model_copy(update={"role": VariableRole.DERIVED})
    definition = model.equations[0].model_copy(
        update={
            "equation_id": "EQ-derived",
            "lhs": symbol("derived"),
            "rhs": add(symbol("x"), symbol("y")),
        }
    )
    model = model.model_copy(
        update={
            "derived_variables": [derived],
            "equations": [*model.equations, definition],
        }
    )
    metric = spec("algebraic_scalar", value_symbol="derived")
    raw = RawMetricOutput(variables={"x": 2.0, "y": 3.0, "derived": 999.0})
    assert calculate(metric, raw, model=model) == 5.0
    verified = recompute(metric, raw, model=model, source_digest="c" * 64)
    assert verified.status is IndependentStatus.PASS and verified.verified == 5.0
