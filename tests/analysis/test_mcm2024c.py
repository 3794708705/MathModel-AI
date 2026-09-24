from __future__ import annotations

import math
from pathlib import Path

import pytest

from analysis.mcm2024c.analyze import (
    adjusted_probability,
    analyze,
    fit_beta,
    grouped_cross_validation,
    load_matches,
    point_features,
    service_summary,
)


def test_point_features_use_only_past_points_and_condition_on_server() -> None:
    points = [(1, 1), (1, 0), (2, 1), (1, 1)]
    features = point_features(points, window=2)

    assert features[0] == (0.5, 0.0, 1)
    assert features[1][0] == pytest.approx(11 / 21)
    assert features[1][1] == pytest.approx(0.5)
    assert features[2][0] == 0.5
    assert features[2][1] == pytest.approx((0.5 - 11 / 21) / 2)
    changed_future = point_features([*points[:3], (1, 0)], window=2)
    assert changed_future[:3] == features[:3]


def test_offset_fit_is_zero_for_no_momentum_and_predicts_valid_probability() -> None:
    rows = [(0.5, 0.0, 1), (0.6, 0.0, 0), (0.4, 0.0, 1)]
    assert fit_beta(rows) == 0.0
    assert adjusted_probability(0.6, 0.0, 7.0) == pytest.approx(0.6)
    assert math.isfinite(adjusted_probability(0.6, 0.25, 7.0))


def test_official_schema_load_and_grouped_cv(tmp_path: Path) -> None:
    path = tmp_path / "Wimbledon_featured_matches.csv"
    rows = ["match_id,player1,player2,server,point_victor,p1_score,p2_score"]
    for match in range(5):
        for point in range(12):
            rows.append(
                f"m{match},P{match},Q{match},{1 + point % 2},{1 + (match + point) % 2},AD,40"
            )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    matches = load_matches(path)
    result = grouped_cross_validation(matches, window=4)

    assert len(matches) == 5
    assert service_summary(matches)["server_points"] == 60
    assert result["n_points"] == 60
    assert len(result["beta_by_fold"]) == 5
    assert 0 <= result["brier_baseline"] <= 1
    assert 0 <= result["brier_adjusted"] <= 1
    with pytest.raises(ValueError, match="digest does not match"):
        analyze(path)
    path.write_text(
        "match_id,player1,player2,server,point_victor\nm0,P,Q,3,1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid server"):
        load_matches(path)
