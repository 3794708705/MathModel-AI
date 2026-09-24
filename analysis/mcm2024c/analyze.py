"""Reproducible, non-benchmark analysis of COMAP 2024 MCM Problem C.

The input is the unmodified official Wimbledon_featured_matches.csv. Every
prediction uses only earlier points from the same match. This is an independent
case analysis, not a substitute for MathModel AI's failed verification gate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import deque
from pathlib import Path

PRIOR_WINS = 10.0
PRIOR_LOSSES = 10.0
MAIN_WINDOW = 8
NULL_REPLICATES = 500
NULL_SEED = 2024
FINAL_MATCH_ID = "2023-wimbledon-1701"
OFFICIAL_SHA256 = "b1788d0ea169b65629b0e9fb0f91d007507b306e404507bbf90bd5f700a3c229"


def load_matches(path: Path) -> dict[str, dict[str, object]]:
    required = {"match_id", "player1", "player2", "server", "point_victor"}
    matches: dict[str, dict[str, object]] = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise ValueError(f"official point data missing columns: {sorted(required)}")
        for row in reader:
            match_id = row["match_id"]
            if row["server"] not in {"1", "2"} or row["point_victor"] not in {"1", "2"}:
                raise ValueError(f"invalid server or winner code in {match_id}")
            match = matches.setdefault(
                match_id,
                {"player1": row["player1"], "player2": row["player2"], "points": []},
            )
            if (match["player1"], match["player2"]) != (row["player1"], row["player2"]):
                raise ValueError(f"player names changed inside {match_id}")
            points = match["points"]
            assert isinstance(points, list)
            points.append((int(row["server"]), int(row["point_victor"] == "1")))
    if not matches:
        raise ValueError("official point data is empty")
    return matches


def point_features(points: list[tuple[int, int]], window: int) -> list[tuple[float, float, int]]:
    """Online serve-conditioned baseline and past-only momentum signal."""
    wins = {1: 0, 2: 0}
    counts = {1: 0, 2: 0}
    recent: deque[float] = deque(maxlen=window)
    rows: list[tuple[float, float, int]] = []
    for server, outcome in points:
        p0 = (PRIOR_WINS + wins[server]) / (PRIOR_WINS + PRIOR_LOSSES + counts[server])
        momentum = sum(recent) / len(recent) if recent else 0.0
        rows.append((p0, momentum, outcome))
        recent.append(outcome - p0)
        counts[server] += 1
        wins[server] += outcome
    return rows


def logistic(value: float) -> float:
    if value >= 0:
        exponent = math.exp(-value)
        return 1 / (1 + exponent)
    exponent = math.exp(value)
    return exponent / (1 + exponent)


def adjusted_probability(p0: float, momentum: float, beta: float) -> float:
    return logistic(math.log(p0 / (1 - p0)) + beta * momentum)


def fit_beta(rows: list[tuple[float, float, int]]) -> float:
    """One-parameter logistic-offset MLE, with numerical ridge only."""
    beta = 0.0
    for _ in range(60):
        gradient = 0.0
        curvature = 1e-8
        for p0, momentum, outcome in rows:
            probability = adjusted_probability(p0, momentum, beta)
            gradient += momentum * (outcome - probability)
            curvature += momentum * momentum * probability * (1 - probability)
        step = max(-1.0, min(1.0, gradient / curvature))
        beta = max(-20.0, min(20.0, beta + step))
        if abs(step) < 1e-9:
            break
    return beta


def score(rows: list[tuple[float, float, int]], beta: float) -> dict[str, float]:
    brier_baseline = brier_adjusted = log_baseline = log_adjusted = 0.0
    for p0, momentum, outcome in rows:
        p1 = adjusted_probability(p0, momentum, beta)
        brier_baseline += (outcome - p0) ** 2
        brier_adjusted += (outcome - p1) ** 2
        log_baseline -= outcome * math.log(p0) + (1 - outcome) * math.log1p(-p0)
        log_adjusted -= outcome * math.log(p1) + (1 - outcome) * math.log1p(-p1)
    n = len(rows)
    return {
        "n": n,
        "brier_baseline": brier_baseline / n,
        "brier_adjusted": brier_adjusted / n,
        "logloss_baseline": log_baseline / n,
        "logloss_adjusted": log_adjusted / n,
    }


def grouped_cross_validation(
    matches: dict[str, dict[str, object]], window: int
) -> dict[str, object]:
    ids = sorted(matches)
    features: dict[str, list[tuple[float, float, int]]] = {}
    for match_id in ids:
        points = matches[match_id]["points"]
        assert isinstance(points, list)
        features[match_id] = point_features(points, window)
    all_test: list[tuple[float, float, int]] = []
    coefficients: list[float] = []
    predictions: list[float] = []
    fold_results: list[dict[str, float | int]] = []
    match_results: list[dict[str, float | int | str]] = []
    for fold in range(5):
        train = [
            row
            for index, match_id in enumerate(ids)
            if index % 5 != fold
            for row in features[match_id]
        ]
        test = [
            row
            for index, match_id in enumerate(ids)
            if index % 5 == fold
            for row in features[match_id]
        ]
        beta = fit_beta(train)
        coefficients.append(beta)
        all_test.extend(test)
        predictions.extend(adjusted_probability(p0, momentum, beta) for p0, momentum, _ in test)
        fold_score = score(test, beta)
        fold_results.append(
            {
                "fold": fold + 1,
                "n_matches": sum(index % 5 == fold for index in range(len(ids))),
                "beta": beta,
                **fold_score,
            }
        )
        for index, match_id in enumerate(ids):
            if index % 5 != fold:
                continue
            match_score = score(features[match_id], beta)
            match_results.append({"match_id": match_id, "fold": fold + 1, **match_score})
    brier_adjusted = sum(
        (y - p) ** 2 for (_, _, y), p in zip(all_test, predictions, strict=True)
    ) / len(all_test)
    log_adjusted = -sum(
        y * math.log(p) + (1 - y) * math.log1p(-p)
        for (_, _, y), p in zip(all_test, predictions, strict=True)
    ) / len(all_test)
    baseline = score(all_test, 0.0)
    ordered = sorted(zip(all_test, predictions, strict=True), key=lambda pair: pair[0][0])
    calibration: list[dict[str, float | int]] = []
    for decile in range(10):
        group = ordered[decile * len(ordered) // 10 : (decile + 1) * len(ordered) // 10]
        calibration.append(
            {
                "decile": decile + 1,
                "n": len(group),
                "baseline_mean": sum(row[0][0] for row in group) / len(group),
                "adjusted_mean": sum(row[1] for row in group) / len(group),
                "observed_rate": sum(row[0][2] for row in group) / len(group),
            }
        )
    return {
        "window": window,
        "folds": 5,
        "n_points": len(all_test),
        "beta_by_fold": coefficients,
        "brier_baseline": baseline["brier_baseline"],
        "brier_adjusted": brier_adjusted,
        "logloss_baseline": baseline["logloss_baseline"],
        "logloss_adjusted": log_adjusted,
        "fold_results": fold_results,
        "match_results": sorted(match_results, key=lambda item: str(item["match_id"])),
        "calibration": calibration,
    }


def persistence_statistic(matches: dict[str, dict[str, object]], window: int) -> float:
    total = 0.0
    for match in matches.values():
        points = match["points"]
        assert isinstance(points, list)
        total += sum(
            momentum * (outcome - p0) for p0, momentum, outcome in point_features(points, window)
        )
    return total


def null_test(matches: dict[str, dict[str, object]], window: int) -> dict[str, float | int]:
    """Sequential Bernoulli null preserving each match's observed server schedule."""
    observed = persistence_statistic(matches, window)
    rng = random.Random(NULL_SEED)
    simulated: list[float] = []
    for _ in range(NULL_REPLICATES):
        total = 0.0
        for match in matches.values():
            points = match["points"]
            assert isinstance(points, list)
            wins = {1: 0, 2: 0}
            counts = {1: 0, 2: 0}
            recent: deque[float] = deque(maxlen=window)
            for server, _ in points:
                p0 = (PRIOR_WINS + wins[server]) / (PRIOR_WINS + PRIOR_LOSSES + counts[server])
                momentum = sum(recent) / len(recent) if recent else 0.0
                outcome = int(rng.random() < p0)
                residual = outcome - p0
                total += momentum * residual
                recent.append(residual)
                counts[server] += 1
                wins[server] += outcome
        simulated.append(total)
    mean = sum(simulated) / len(simulated)
    simulated.sort()
    deviation = abs(observed - mean)
    extreme = sum(abs(item - mean) >= deviation for item in simulated)
    return {
        "window": window,
        "observed": observed,
        "null_mean": mean,
        "null_sd": math.sqrt(sum((item - mean) ** 2 for item in simulated) / len(simulated)),
        "replicates": NULL_REPLICATES,
        "two_sided_p": (extreme + 1) / (NULL_REPLICATES + 1),
        "seed": NULL_SEED,
        "null_q025": simulated[int(0.025 * (NULL_REPLICATES - 1))],
        "null_q975": simulated[int(0.975 * (NULL_REPLICATES - 1))],
        "extreme_count": extreme,
    }


def match_audit(matches: dict[str, dict[str, object]]) -> dict[str, object]:
    """Descriptive checks from the exact rows used by the analysis."""
    lengths: list[int] = []
    server_1_points = server_1_wins = server_2_points = server_2_wins = 0
    for match in matches.values():
        points = match["points"]
        assert isinstance(points, list)
        lengths.append(len(points))
        for server, outcome in points:
            if server == 1:
                server_1_points += 1
                server_1_wins += outcome
            else:
                server_2_points += 1
                server_2_wins += 1 - outcome
    lengths.sort()
    return {
        "min_points_per_match": lengths[0],
        "median_points_per_match": lengths[len(lengths) // 2],
        "max_points_per_match": lengths[-1],
        "server_1_points": server_1_points,
        "server_1_wins": server_1_wins,
        "server_2_points": server_2_points,
        "server_2_wins": server_2_wins,
    }


def final_match_summary(matches: dict[str, dict[str, object]]) -> dict[str, object]:
    match = matches[FINAL_MATCH_ID]
    points = match["points"]
    assert isinstance(points, list)
    features = point_features(points, MAIN_WINDOW)
    strengths = [item[1] for item in features]
    eligible = range(MAIN_WINDOW + 1, len(strengths))
    high = max(eligible, key=strengths.__getitem__)
    low = min(eligible, key=strengths.__getitem__)
    return {
        "match_id": FINAL_MATCH_ID,
        "player1": match["player1"],
        "player2": match["player2"],
        "n_points": len(points),
        "largest_prior_flow_for_player1": {
            "point_index_1based": high + 1,
            "value": strengths[high],
        },
        "largest_prior_flow_for_player2": {"point_index_1based": low + 1, "value": strengths[low]},
        "player1_point_wins": sum(outcome for _, outcome in points),
    }


def service_summary(matches: dict[str, dict[str, object]]) -> dict[str, float | int]:
    server_points = 0
    server_wins = 0
    for match in matches.values():
        points = match["points"]
        assert isinstance(points, list)
        for server, player1_won in points:
            server_points += 1
            server_wins += int((server == 1) == bool(player1_won))
    return {
        "server_points": server_points,
        "server_wins": server_wins,
        "server_win_rate": server_wins / server_points,
    }


def analyze(path: Path) -> dict[str, object]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != OFFICIAL_SHA256:
        raise ValueError("input CSV digest does not match the reviewed official dataset")
    matches = load_matches(path)
    result = {
        "source_file": "Wimbledon_featured_matches.csv",
        "source_sha256": digest,
        "n_matches": len(matches),
        "n_points": sum(len(match["points"]) for match in matches.values()),
        "definition": (
            "past-eight-point mean of (player1 win - sequential serve-conditioned baseline)"
        ),
        "prior": {"wins": PRIOR_WINS, "losses": PRIOR_LOSSES},
        "service": service_summary(matches),
        "match_audit": match_audit(matches),
        "cross_validation": [grouped_cross_validation(matches, window) for window in (4, 8, 12)],
        "null_test": null_test(matches, MAIN_WINDOW),
        "final_match": final_match_summary(matches),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("official_csv", type=Path)
    parser.add_argument("output_json", type=Path)
    args = parser.parse_args()
    result = analyze(args.official_csv)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
