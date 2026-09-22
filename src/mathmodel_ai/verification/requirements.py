"""Reviewed evaluation-only sidecars; historical solve manifests remain byte-identical."""

import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mathmodel_ai.mathematical.digests import mathematical_model_digest
from mathmodel_ai.schemas.independent_verification import VerificationRequirements
from mathmodel_ai.schemas.mathematical import MathematicalModel
from mathmodel_ai.schemas.subproblem_identity import SubproblemIdentityContract
from mathmodel_ai.verification.metric_recompute import content_digest


@dataclass(frozen=True)
class ReviewedModelContract:
    benchmark_id: str
    model: MathematicalModel
    model_digest: str
    model_contract_digest: str
    policy_digest: str
    requirements: VerificationRequirements
    subproblem_identity: SubproblemIdentityContract
    subproblem_identity_digest: str


class VerificationRequirementRegistry:
    def __init__(self, root: Path):
        self._root = root

    def get(self, benchmark_id: str) -> VerificationRequirements | None:
        found = self._matching_policies(benchmark_id)
        return found[0][1] if found else None

    def reviewed_model(self, benchmark_id: str) -> ReviewedModelContract | None:
        found = self._matching_policies(benchmark_id)
        if not found:
            return None
        policy_path, policy = found[0]
        if policy.model_contract_digest is None:
            raise ValueError("REVIEWED_MODEL_CONTRACT_DIGEST_MISSING")
        model_path = policy_path.with_name(f"mathematical-model-v{policy.version}.json")
        contract = self._read_json(model_path)
        stated_contract_digest = contract.get("content_digest")
        actual_contract_digest = content_digest(
            {key: value for key, value in contract.items() if key != "content_digest"}
        )
        if (
            not isinstance(stated_contract_digest, str)
            or stated_contract_digest != actual_contract_digest
            or stated_contract_digest != policy.model_contract_digest
        ):
            raise ValueError("REVIEWED_MODEL_CONTRACT_DIGEST_MISMATCH")
        model = MathematicalModel.model_validate(contract.get("model"))
        digest = mathematical_model_digest(model)
        if contract.get("model_digest") != digest or policy.model_digest != digest:
            raise ValueError("REVIEWED_MODEL_DIGEST_MISMATCH")
        bindings = policy.validation_requirement_bindings
        if [item.requirement for item in bindings] != model.validation_requirements:
            raise ValueError("REVIEWED_VALIDATION_REQUIREMENT_COVERAGE_MISMATCH")
        known_model_refs = {
            *(item.assumption_id for item in model.assumptions),
            *(item.ambiguity_id for item in model.interpretation_resolutions),
            *(item.set_id for item in model.sets),
            *(item.index_id for item in model.indices),
            *(
                item.variable_id
                for item in [
                    *model.decision_variables,
                    *model.state_variables,
                    *model.derived_variables,
                ]
            ),
            *(item.parameter_id for item in [*model.parameters, *model.constants]),
            *(
                item.constraint_id
                for item in [
                    *model.constraints,
                    *model.initial_conditions,
                    *model.boundary_conditions,
                ]
            ),
            *(item.equation_id for item in model.equations),
            *(item.output_id for item in model.expected_outputs),
            *model.source_evidence,
        }
        if any(not set(item.model_evidence_refs) <= known_model_refs for item in bindings):
            raise ValueError("REVIEWED_VALIDATION_REQUIREMENT_MODEL_EVIDENCE_MISMATCH")
        identity_payload = self._read_json(policy_path.with_name("subproblem-identity.json"))
        identity = SubproblemIdentityContract.model_validate(identity_payload)
        identity_digest = content_digest(
            {key: value for key, value in identity_payload.items() if key != "content_digest"}
        )
        if identity.content_digest != identity_digest:
            raise ValueError("REVIEWED_SUBPROBLEM_IDENTITY_DIGEST_MISMATCH")
        if identity.benchmark_id != benchmark_id:
            raise ValueError("REVIEWED_SUBPROBLEM_BENCHMARK_MISMATCH")
        if identity.problem_sha256 != policy.problem_sha256:
            raise ValueError("REVIEWED_SUBPROBLEM_SOURCE_MISMATCH")
        if identity.canonical_ids != model.target_subproblems:
            raise ValueError("REVIEWED_SUBPROBLEM_MODEL_COVERAGE_MISMATCH")
        self._check_review_report(
            policy_path.with_name("model-contract-red-team.json"),
            expected_digest=policy.red_team_report_digest,
            required={"status": "PASS", "critical_count": 0},
        )
        self._check_review_report(
            policy_path.with_name("model-contract-jury.json"),
            expected_digest=policy.model_jury_report_digest,
            required={"status": "PASS", "decision": "PASS", "critical_count": 0},
        )
        identity_review = self._read_json(policy_path.with_name("subproblem-identity-review.json"))
        stated_review_digest = identity_review.get("content_digest")
        actual_review_digest = content_digest(
            {key: value for key, value in identity_review.items() if key != "content_digest"}
        )
        expected_review = {
            "status": "PASS",
            "critical_count": 0,
            "benchmark_id": benchmark_id,
            "identity_contract_digest": identity_digest,
            "model_digest": digest,
            "policy_digest": content_digest(policy),
        }
        if stated_review_digest != actual_review_digest:
            raise ValueError("REVIEWED_SUBPROBLEM_REVIEW_DIGEST_MISMATCH")
        if any(identity_review.get(key) != value for key, value in expected_review.items()):
            raise ValueError("REVIEWED_SUBPROBLEM_IDENTITY_REVIEW_DID_NOT_PASS")
        replay_review = self._read_json(
            policy_path.with_name("verification-policy-replay-review.json")
        )
        replay_review_digest = replay_review.get("content_digest")
        if replay_review_digest != content_digest(
            {key: value for key, value in replay_review.items() if key != "content_digest"}
        ):
            raise ValueError("REVIEWED_VERIFICATION_REPLAY_REVIEW_DIGEST_MISMATCH")
        required_metrics = sum(item.required for item in policy.metrics)
        required_scenarios = sum(item.required for item in policy.scenarios)
        expected_replay_review = {
            "benchmark_id": benchmark_id,
            "model_digest": digest,
            "model_contract_digest": stated_contract_digest,
            "policy_digest": content_digest(policy),
            "required_metrics": required_metrics,
            "passed_metrics": required_metrics,
            "required_scenarios": required_scenarios,
            "passed_scenarios": required_scenarios,
            "unique_execution_count": required_scenarios,
            "critical_count": 0,
            "status": "PASS",
        }
        if any(replay_review.get(key) != value for key, value in expected_replay_review.items()):
            raise ValueError("REVIEWED_VERIFICATION_REPLAY_REVIEW_DID_NOT_PASS")
        return ReviewedModelContract(
            benchmark_id=benchmark_id,
            model=model,
            model_digest=digest,
            model_contract_digest=stated_contract_digest,
            policy_digest=content_digest(policy),
            requirements=policy,
            subproblem_identity=identity,
            subproblem_identity_digest=identity_digest,
        )

    def _matching_policies(self, benchmark_id: str) -> list[tuple[Path, VerificationRequirements]]:
        found: list[tuple[Path, VerificationRequirements]] = []
        for path in sorted(self._root.glob("case-*/independent-verification.json")):
            policy = VerificationRequirements.model_validate(self._read_json(path))
            if policy.benchmark_id == benchmark_id:
                found.append((path, policy))
        if len(found) > 1:
            raise ValueError("duplicate independent verification policy")
        return found

    def _read_json(self, path: Path) -> dict[str, Any]:
        info = os.lstat(path)
        if (
            not stat.S_ISREG(info.st_mode)
            or stat.S_ISLNK(info.st_mode)
            or getattr(info, "st_file_attributes", 0) & 0x400
            or not path.resolve().is_relative_to(self._root.resolve())
            or info.st_size > 1024 * 1024
        ):
            raise ValueError("invalid independent verification policy file")
        value = json.loads(path.read_bytes())
        if not isinstance(value, dict):
            raise ValueError("reviewed verification sidecar must be a JSON object")
        return value

    def _check_review_report(
        self,
        path: Path,
        *,
        expected_digest: str | None,
        required: dict[str, str | int],
    ) -> None:
        if expected_digest is None:
            raise ValueError("REVIEWED_MODEL_REVIEW_DIGEST_MISSING")
        report = self._read_json(path)
        stated_digest = report.get("content_digest")
        actual_digest = content_digest(
            {key: value for key, value in report.items() if key != "content_digest"}
        )
        if stated_digest != actual_digest or stated_digest != expected_digest:
            raise ValueError("REVIEWED_MODEL_REVIEW_DIGEST_MISMATCH")
        if any(report.get(key) != value for key, value in required.items()):
            raise ValueError("REVIEWED_MODEL_REVIEW_DID_NOT_PASS")
