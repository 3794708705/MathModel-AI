# ruff: noqa: E501
"""Independently review and, only on success, promote the Case A policy."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, text

from mathmodel_ai.mathematical.digests import mathematical_model_digest
from mathmodel_ai.mathematical.quality_gates import model_quality_gate
from mathmodel_ai.mathematical.registry import EquationRegistry, SymbolRegistry
from mathmodel_ai.mathematical.units import UnitChecker
from mathmodel_ai.schemas.independent_verification import (
    RawMetricOutput,
    VerificationRequirements,
)
from mathmodel_ai.schemas.mathematical import MathematicalModel, UnitCheckStatus
from mathmodel_ai.schemas.problem_state import ProblemState
from mathmodel_ai.schemas.quality import QualityGateStatus
from mathmodel_ai.verification.metric_recompute import algebraic_value, content_digest

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "benchmarks" / "case-003-mcm-2024-a"
OLD_DRAFT_DIGEST = "650560edde518156a05ed31b2b60357fd06826a60f60008b5c31ae174beb9384"
ASSUMPTION_PARAMETERS = {
    "sex_ratio_shape",
    "cap_min",
    "cap_slope",
    "birth_coef",
    "maturation",
    "mort_larva",
    "mort_adult",
    "fish_growth",
    "fish_capacity",
    "attack_rate",
    "parasite_growth",
    "parasite_capacity",
    "fish_benefit_par",
    "lamprey_cost_par",
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must be a JSON object")
    return value


def _semantic_digest(value: dict[str, Any]) -> str:
    return content_digest({key: item for key, item in value.items() if key != "content_digest"})


def _write(path: Path, value: dict[str, Any]) -> None:
    path.write_bytes(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    )


def _state(model: MathematicalModel) -> ProblemState:
    url = os.getenv(
        "MM_DATABASE_URL",
        "postgresql+psycopg://mathmodel:mathmodel@localhost:5432/mathmodel",
    )
    engine = create_engine(url)
    with engine.connect() as connection:
        state = connection.execute(
            text(
                "SELECT state FROM problem_states WHERE problem_id = :problem_id "
                "AND is_current = true"
            ),
            {"problem_id": str(model.problem_id)},
        ).scalar_one()
    return ProblemState.model_validate(state)


def _review_checks(
    contract: dict[str, Any], draft: dict[str, Any], model: MathematicalModel, state: ProblemState
) -> dict[str, bool]:
    symbols = SymbolRegistry.from_model(model)
    equations = EquationRegistry.from_model(model)
    units = UnitChecker(symbols).check_model(model)
    gate = model_quality_gate(model, state)
    reference = contract["state_contract"]["initial_state"]
    reference_raw = RawMetricOutput(variables={"E": 0.5, **reference})
    residual = algebraic_value(model, reference_raw, "equilibrium_residual_sq")
    low = algebraic_value(model, RawMetricOutput(variables={"E": 0.0}), "male_ratio")
    high = algebraic_value(model, RawMetricOutput(variables={"E": 1.0}), "male_ratio")
    fixed = model.model_copy(
        update={
            "parameters": [
                item.model_copy(update={"value": 0.0})
                if item.symbol == "adaptation_weight"
                else item
                for item in model.parameters
            ]
        }
    )
    fixed_ratio = algebraic_value(fixed, RawMetricOutput(variables={"E": 0.0}), "male_ratio")
    eigenvalues = [
        algebraic_value(model, reference_raw, symbol)
        for symbol in ("lambda_JA_plus", "lambda_JA_minus", "lambda_F", "lambda_P")
    ]
    dominant = algebraic_value(model, reference_raw, "dominant_real_part")
    policy = draft["candidate_policy"]
    sensitivity = [
        item for item in policy["scenarios"] if item["scenario_id"].startswith("sensitivity_")
    ]
    inventory = {item["symbol"] for item in contract["parameter_inventory"]}
    initial_symbols = {item.expression.symbol for item in model.initial_conditions}
    expected_initial = set(contract["state_contract"]["state_vector"])
    all_core = [
        *model.decision_variables,
        *model.state_variables,
        *model.derived_variables,
        *model.parameters,
        *model.constants,
    ]
    policy_metadata_complete = all(
        metric.get("quantity")
        and metric.get("calculation")
        and metric.get("inputs")
        and metric.get("unit")
        and metric.get("tolerance_provenance")
        and metric.get("model_binding") == contract["model_digest"]
        for metric in [
            *policy["metrics"],
            *(metric for scenario in policy["scenarios"] for metric in scenario["metrics"]),
        ]
    )
    scenario_metadata_complete = all(
        scenario.get("baseline")
        and scenario.get("perturbation")
        and scenario.get("reason")
        and scenario.get("comparison_quantity")
        and scenario.get("acceptance_criterion")
        and scenario.get("criterion_provenance")
        and set(scenario.get("input_changes", []))
        == set(scenario["parameter_values"]) | set(scenario["decision_values"])
        for scenario in policy["scenarios"]
    )
    historical = contract["source_bindings"]["historical_result_solver_execution_artifacts"]
    return {
        "contract_digest_valid": contract["content_digest"] == _semantic_digest(contract),
        "draft_digest_valid": draft["content_digest"] == _semantic_digest(draft),
        "model_version_is_new": model.version == 2,
        "model_digest_valid": contract["model_digest"] == mathematical_model_digest(model),
        "model_digest_changed": (
            contract["model_digest"] != contract["source_bindings"]["source_model_digest"]
        ),
        "model_quality_gate_pass": gate.status is QualityGateStatus.PASS,
        "symbol_registry_pass": symbols.report.valid and not symbols.report.issues,
        "equation_registry_pass": equations.report.valid and not equations.report.issues,
        "unit_audit_pass": units.status is UnitCheckStatus.PASS and not units.issues,
        "all_core_symbols_have_units": all(item.unit is not None for item in all_core),
        "initial_state_is_formal_and_ordered": (
            initial_symbols == expected_initial
            and list(reference) == contract["state_contract"]["state_vector"]
        ),
        "reference_state_is_equilibrium": abs(residual) <= 1e-20,
        "adaptive_anchor_identities_pass": abs(low - 0.78) <= 1e-12 and abs(high - 0.56) <= 1e-12,
        "fixed_ratio_identity_pass": abs(fixed_ratio - 0.67) <= 1e-12,
        "jacobian_spectrum_is_recomputed": (
            abs(dominant - max(eigenvalues)) <= 1e-12 and all(map(lambda x: x == x, eigenvalues))
        ),
        "all_fourteen_parameters_inventory_complete": inventory == ASSUMPTION_PARAMETERS,
        "all_fourteen_parameters_are_assumptions": all(
            item.source_type.value == "ASSUMPTION"
            for item in model.parameters
            if item.symbol in ASSUMPTION_PARAMETERS
        ),
        "local_sensitivity_coverage_complete": (
            len(sensitivity) == 14
            and {next(iter(item["parameter_values"])) for item in sensitivity}
            == ASSUMPTION_PARAMETERS
        ),
        "structural_comparator_pairs_complete": {
            "adaptive_low_resource",
            "fixed_low_resource",
            "adaptive_high_resource",
            "fixed_high_resource",
        }
        <= {item["scenario_id"] for item in policy["scenarios"]},
        "policy_metric_metadata_complete": bool(policy_metadata_complete),
        "policy_scenario_metadata_complete": bool(scenario_metadata_complete),
        "scientific_obligations_resolved": not contract["unresolved_obligations"]
        and not draft["unresolved_obligations"],
        "parasite_semantics_explicit": (
            contract["parasite_contract"]["P"]
            == "normalized abundance of a representative non-lamprey fish-parasite guild"
        ),
        "no_practical_extinction_threshold": (
            "No practical-extinction cutoff"
            in contract["resilience_persistence_contract"]["extinction"]
        ),
        "historical_execution_artifacts_audited": (
            len(historical) == 2
            and all(not item["is_mock"] for item in historical)
            and any(item["result_status"] == "FEASIBLE" for item in historical)
            and all(item["program_code_hash"] for item in historical)
            and all(item["execution_code_hash"] for item in historical)
        ),
        "no_retained_literature_ranges": (
            contract["source_bindings"]["retained_literature_reference_count"] == 0
        ),
    }


def _report(
    kind: str, contract_digest: str, checks: dict[str, bool], findings: list[dict[str, Any]]
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "report_version": "1",
        "review_kind": kind,
        "reviewer": f"independent-deterministic-{kind.lower()}-v1",
        "reviewer_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model_contract_digest": contract_digest,
        "checks": checks,
        "findings": findings,
        "critical_count": sum(item["severity"] == "CRITICAL" for item in findings),
        "major_count": sum(item["severity"] == "MAJOR" for item in findings),
        "minor_count": sum(item["severity"] == "MINOR" for item in findings),
        "status": "PASS"
        if all(checks.values()) and not any(item["severity"] == "CRITICAL" for item in findings)
        else "FAIL",
        "content_digest": None,
    }
    report["content_digest"] = _semantic_digest(report)
    return report


def main() -> None:
    contract = _load(CASE / "mathematical-model-v2.json")
    draft = _load(CASE / "independent-verification.draft.json")
    archived = _load(CASE / "independent-verification.v1-draft.json")
    if (
        archived.get("content_digest") != OLD_DRAFT_DIGEST
        or _semantic_digest(archived) != OLD_DRAFT_DIGEST
    ):
        raise ValueError("archived version-1 draft was not preserved byte-semantically")
    model = MathematicalModel.model_validate(contract["model"])
    state = _state(model)
    checks = _review_checks(contract, draft, model, state)
    red_team_findings = [
        {
            "severity": "MAJOR",
            "code": "PROVISIONAL_ECOLOGICAL_COEFFICIENTS",
            "finding": "All 14 ecological coefficients are modeling assumptions rather than retained empirical estimates.",
            "disposition": "ACCEPTED_LIMITATION_PAPER_VISIBLE_AND_FULLY_SENSITIVITY_COVERED",
        },
        {
            "severity": "MAJOR",
            "code": "LOCAL_STABILITY_ONLY",
            "finding": "The Jacobian contract establishes only local asymptotic behavior at a numerical equilibrium.",
            "disposition": "ACCEPTED_LIMITATION_GLOBAL_STABILITY_AND_RESILIENCE_CLAIMS_FORBIDDEN",
        },
        {
            "severity": "MAJOR",
            "code": "NORMALIZED_TIME_ONLY",
            "finding": "No physical T_ref exists, so calendar-time ecological claims are unsupported.",
            "disposition": "ACCEPTED_LIMITATION_PHYSICAL_TIME_CONVERSION_FORBIDDEN",
        },
        {
            "severity": "MINOR",
            "code": "INITIAL_STATE_DEPENDENCE",
            "finding": "Transient endpoint magnitudes depend on the common normalized reference equilibrium.",
            "disposition": "CONTROLLED_BY_IDENTICAL_PAIRED_INITIAL_STATE_AND_PAPER_DISCLOSURE",
        },
        {
            "severity": "MINOR",
            "code": "NO_PRACTICAL_EXTINCTION_THRESHOLD",
            "finding": "The contract cannot make practical-extinction claims without a sourced threshold.",
            "disposition": "BY_DESIGN_ONLY_MATHEMATICAL_PERSISTENCE_IS_AUTHORIZED",
        },
    ]
    red_team = _report("RED_TEAM", contract["content_digest"], checks, red_team_findings)
    jury_dimensions = {
        "scientific_coherence": 25
        if checks["scientific_obligations_resolved"] and checks["parasite_semantics_explicit"]
        else 0,
        "mathematical_well_posedness": 25
        if checks["model_quality_gate_pass"]
        and checks["unit_audit_pass"]
        and checks["reference_state_is_equilibrium"]
        else 0,
        "problem_relevance": 25
        if set(model.target_subproblems) == {"Q1", "Q2", "Q3", "Q4", "Q5"}
        and checks["structural_comparator_pairs_complete"]
        else 0,
        "verification_feasibility": 25
        if checks["policy_metric_metadata_complete"]
        and checks["policy_scenario_metadata_complete"]
        and checks["jacobian_spectrum_is_recomputed"]
        else 0,
    }
    jury_checks = {**checks, "minimum_score_reached": sum(jury_dimensions.values()) >= 85}
    jury = _report("MODEL_JURY", contract["content_digest"], jury_checks, [])
    jury["score_dimensions"] = jury_dimensions
    jury["score"] = sum(jury_dimensions.values())
    jury["decision"] = "PASS" if jury["status"] == "PASS" and jury["score"] >= 85 else "FAIL"
    jury["content_digest"] = _semantic_digest(jury)
    _write(CASE / "model-contract-red-team.json", red_team)
    _write(CASE / "model-contract-jury.json", jury)
    if red_team["status"] != "PASS" or red_team["critical_count"] != 0:
        raise ValueError("independent model-contract Red Team rejected promotion")
    if jury["decision"] != "PASS":
        raise ValueError("independent Model Jury rejected promotion")
    candidate = dict(draft["candidate_policy"])
    candidate.update(
        {
            "review_status": "REVIEWED",
            "production_eligible": True,
            "red_team_report_digest": red_team["content_digest"],
            "model_jury_report_digest": jury["content_digest"],
        }
    )
    policy = VerificationRequirements.model_validate(candidate)
    _write(CASE / "independent-verification.json", policy.model_dump(mode="json"))
    draft.update(
        {
            "review_status": "REVIEWED",
            "production_eligible": True,
            "candidate_policy": policy.model_dump(mode="json"),
            "promotion_blockers": [],
            "red_team_report_digest": red_team["content_digest"],
            "model_jury_report_digest": jury["content_digest"],
            "production_policy_digest": content_digest(policy),
        }
    )
    draft["content_digest"] = _semantic_digest(draft)
    _write(CASE / "independent-verification.draft.json", draft)
    summary: dict[str, Any] = {
        "review_version": "1",
        "model_contract_digest": contract["content_digest"],
        "model_digest": contract["model_digest"],
        "red_team_report_digest": red_team["content_digest"],
        "red_team_critical_count": red_team["critical_count"],
        "model_jury_report_digest": jury["content_digest"],
        "model_jury_score": jury["score"],
        "model_jury_decision": jury["decision"],
        "policy_digest": content_digest(policy),
        "unresolved_obligations": [],
        "fresh_case_a_run": False,
        "case_b_c_run": False,
        "content_digest": None,
    }
    summary["content_digest"] = _semantic_digest(summary)
    _write(CASE / "model-contract-review.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
