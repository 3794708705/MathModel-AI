import json
import shutil
from pathlib import Path

import pytest

from mathmodel_ai.mathematical.digests import mathematical_model_digest
from mathmodel_ai.mathematical.registry import EquationRegistry, SymbolRegistry
from mathmodel_ai.mathematical.units import UnitChecker
from mathmodel_ai.schemas.independent_verification import RawMetricOutput
from mathmodel_ai.schemas.mathematical import MathematicalModel, UnitCheckStatus
from mathmodel_ai.verification.metric_recompute import algebraic_value, content_digest
from mathmodel_ai.verification.requirements import VerificationRequirementRegistry

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / "benchmarks" / "case-003-mcm-2024-a"
OLD_DRAFT_DIGEST = "650560edde518156a05ed31b2b60357fd06826a60f60008b5c31ae174beb9384"
OLD_MODEL_DIGEST = "3334b664351b303aa89a0fd5e6efb2eec8862df18b579dab9f07268feb0c63cf"


def load(name: str):
    return json.loads((CASE / name).read_text("utf-8"))


def semantic_digest(value):
    return content_digest({key: item for key, item in value.items() if key != "content_digest"})


def test_case_a_v2_contract_is_new_complete_and_dimensionally_closed():
    contract = load("mathematical-model-v2.json")
    model = MathematicalModel.model_validate(contract["model"])

    assert contract["content_digest"] == semantic_digest(contract)
    assert model.version == 2
    assert contract["source_bindings"]["source_model_digest"] == OLD_MODEL_DIGEST
    assert contract["source_bindings"]["retained_literature_reference_count"] == 0
    historical = contract["source_bindings"]["historical_result_solver_execution_artifacts"]
    assert len(historical) == 2 and all(not item["is_mock"] for item in historical)
    assert contract["model_digest"] == mathematical_model_digest(model) != OLD_MODEL_DIGEST
    assert contract["unresolved_obligations"] == []
    assert [item.symbol for item in model.state_variables] == ["J", "A", "F", "P"]
    assert {item.expression.symbol for item in model.initial_conditions} == {"J", "A", "F", "P"}
    assert SymbolRegistry.from_model(model).report.issues == []
    assert EquationRegistry.from_model(model).report.issues == []
    unit_report = UnitChecker(SymbolRegistry.from_model(model)).check_model(model)
    assert unit_report.status is UnitCheckStatus.PASS and unit_report.issues == []


def test_case_a_formal_anchors_reference_equilibrium_and_spectrum_recompute():
    contract = load("mathematical-model-v2.json")
    model = MathematicalModel.model_validate(contract["model"])
    state = contract["state_contract"]["initial_state"]
    reference = RawMetricOutput(variables={"E": 0.5, **state})

    assert algebraic_value(
        model, RawMetricOutput(variables={"E": 0.0}), "male_ratio"
    ) == pytest.approx(0.78)
    assert algebraic_value(
        model, RawMetricOutput(variables={"E": 1.0}), "male_ratio"
    ) == pytest.approx(0.56)
    assert algebraic_value(model, reference, "equilibrium_residual_sq") == pytest.approx(
        0, abs=1e-20
    )
    eigenvalues = [
        algebraic_value(model, reference, symbol)
        for symbol in ("lambda_JA_plus", "lambda_JA_minus", "lambda_F", "lambda_P")
    ]
    assert algebraic_value(model, reference, "dominant_real_part") == pytest.approx(
        max(eigenvalues)
    )

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
    assert algebraic_value(
        fixed, RawMetricOutput(variables={"E": 0.0}), "male_ratio"
    ) == pytest.approx(0.67)


def test_case_a_policy_is_reviewed_bound_and_prior_draft_is_preserved():
    contract = load("mathematical-model-v2.json")
    draft = load("independent-verification.draft.json")
    archived = load("independent-verification.v1-draft.json")
    red_team = load("model-contract-red-team.json")
    jury = load("model-contract-jury.json")
    review = load("model-contract-review.json")
    policy = VerificationRequirementRegistry(ROOT / "benchmarks").get("BENCH-MCM2024-A")

    assert archived["content_digest"] == semantic_digest(archived) == OLD_DRAFT_DIGEST
    assert draft["content_digest"] == semantic_digest(draft)
    assert draft["promotion_blockers"] == []
    assert policy is not None
    assert policy.review_status == "REVIEWED" and policy.production_eligible is True
    assert policy.model_digest == contract["model_digest"]
    assert policy.model_contract_digest == contract["content_digest"]
    assert policy.unresolved_obligations == []
    assert len(policy.scenarios) == 19
    assert red_team["critical_count"] == 0 and red_team["status"] == "PASS"
    assert jury["decision"] == "PASS" and jury["score"] >= 85
    assert review["red_team_report_digest"] == policy.red_team_report_digest
    assert review["model_jury_report_digest"] == policy.model_jury_report_digest
    assert red_team["content_digest"] == semantic_digest(red_team)
    assert jury["content_digest"] == semantic_digest(jury)


def test_case_a_reviewed_model_registry_binds_all_review_evidence():
    registry = VerificationRequirementRegistry(ROOT / "benchmarks")
    reviewed = registry.reviewed_model("BENCH-MCM2024-A")

    assert reviewed is not None
    assert reviewed.model.version == 2
    assert reviewed.model_digest == mathematical_model_digest(reviewed.model)
    assert reviewed.model_digest == load("mathematical-model-v2.json")["model_digest"]
    assert reviewed.model_contract_digest == load("mathematical-model-v2.json")["content_digest"]
    assert reviewed.policy_digest == content_digest(reviewed.requirements)
    assert reviewed.subproblem_identity.canonical_ids == reviewed.model.target_subproblems
    assert reviewed.subproblem_identity.problem_sha256 == reviewed.requirements.problem_sha256
    assert reviewed.subproblem_identity_digest == load("subproblem-identity.json")["content_digest"]
    identity_review = load("subproblem-identity-review.json")
    assert identity_review["content_digest"] == semantic_digest(identity_review)
    assert identity_review["identity_contract_digest"] == reviewed.subproblem_identity_digest
    replay_review = load("verification-policy-replay-review.json")
    assert replay_review["content_digest"] == semantic_digest(replay_review)
    assert replay_review["policy_digest"] == reviewed.policy_digest
    assert replay_review["passed_metrics"] == replay_review["required_metrics"] == 1
    assert replay_review["passed_scenarios"] == replay_review["required_scenarios"] == 19
    assert replay_review["unique_execution_count"] == 19
    assert replay_review["critical_count"] == 0


def test_reviewed_model_registry_rejects_contract_tampering(tmp_path: Path):
    copied = tmp_path / "benchmarks" / CASE.name
    shutil.copytree(CASE, copied)
    path = copied / "mathematical-model-v2.json"
    contract = json.loads(path.read_text("utf-8"))
    contract["model"]["parameters"][0]["value"] = 99.0
    path.write_text(json.dumps(contract), encoding="utf-8")

    with pytest.raises(ValueError, match="REVIEWED_MODEL_CONTRACT_DIGEST_MISMATCH"):
        VerificationRequirementRegistry(tmp_path / "benchmarks").reviewed_model("BENCH-MCM2024-A")


def test_reviewed_model_registry_rejects_subproblem_identity_tampering(tmp_path: Path):
    copied = tmp_path / "benchmarks" / CASE.name
    shutil.copytree(CASE, copied)
    path = copied / "subproblem-identity.json"
    identity = json.loads(path.read_text("utf-8"))
    identity["bindings"][3]["canonical_subproblem_id"] = "Q-WRONG"
    path.write_text(json.dumps(identity), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=r"REVIEWED_SUBPROBLEM_IDENTITY_DIGEST_MISMATCH|unknown canonical subproblem",
    ):
        VerificationRequirementRegistry(tmp_path / "benchmarks").reviewed_model("BENCH-MCM2024-A")


def test_reviewed_model_registry_rejects_scenario_review_tampering(tmp_path: Path):
    copied = tmp_path / "benchmarks" / CASE.name
    shutil.copytree(CASE, copied)
    path = copied / "verification-policy-replay-review.json"
    review = json.loads(path.read_text("utf-8"))
    review["passed_scenarios"] = 0
    path.write_text(json.dumps(review), encoding="utf-8")

    with pytest.raises(ValueError, match="REVIEWED_VERIFICATION_REPLAY_REVIEW_DIGEST_MISMATCH"):
        VerificationRequirementRegistry(tmp_path / "benchmarks").reviewed_model("BENCH-MCM2024-A")


def test_case_a_sensitivity_and_comparator_scenarios_are_reproducible():
    policy = VerificationRequirementRegistry(ROOT / "benchmarks").get("BENCH-MCM2024-A")
    assert policy is not None
    sensitivity = [item for item in policy.scenarios if item.scenario_id.startswith("sensitivity_")]
    assert len(sensitivity) == 14
    assert len({next(iter(item.parameter_values)) for item in sensitivity}) == 14
    by_id = {item.scenario_id: item for item in policy.scenarios}
    structural = [
        by_id[scenario_id]
        for scenario_id in (
            "adaptive_low_resource",
            "fixed_low_resource",
            "adaptive_high_resource",
            "fixed_high_resource",
        )
    ]
    assert {item.dynamic.stop for item in structural if item.dynamic is not None} == {
        614.0226914650789
    }
    assert {item.dynamic.samples for item in structural if item.dynamic is not None} == {1601}
    for adaptive, fixed in (
        ("adaptive_low_resource", "fixed_low_resource"),
        ("adaptive_high_resource", "fixed_high_resource"),
    ):
        left, right = by_id[adaptive], by_id[fixed]
        assert left.dynamic == right.dynamic
        assert left.decision_values == right.decision_values
        assert left.parameter_values == {}
        assert right.parameter_values == {"adaptation_weight": 0.0}
