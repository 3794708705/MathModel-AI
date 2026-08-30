from __future__ import annotations

from collections.abc import Iterable
from typing import Any, cast

from mathmodel_ai.paper.hashing import sha256_json
from mathmodel_ai.schemas.submission import (
    CompetitionProfile,
    CompetitionRule,
    FinalJuryReport,
    ProfileVerificationStatus,
    RequirementCoverage,
    RuleResult,
    RuleSeverity,
    SubmissionArtifact,
    SubmissionCandidate,
    SubmissionCheckResult,
    SubmissionRequirement,
    SubmissionSnapshot,
)


class CompetitionProfileIntegrityError(ValueError):
    pass


def competition_profile_digest(profile: CompetitionProfile) -> str:
    return sha256_json(profile.model_dump(mode="json"))


def competition_rule_digest(rule: CompetitionRule) -> str:
    return sha256_json(rule.model_dump(mode="json"))


def validate_competition_profile(profile: CompetitionProfile) -> str:
    payload = profile.model_dump(mode="json")
    try:
        CompetitionProfile.model_validate(payload)
    except ValueError as exc:
        raise CompetitionProfileIntegrityError(
            "COMPETITION_PROFILE_INTEGRITY_ERROR: profile schema is invalid"
        ) from exc
    source_refs = set(profile.source_refs)
    if len(source_refs) != len(profile.source_refs):
        raise CompetitionProfileIntegrityError(
            "COMPETITION_PROFILE_INTEGRITY_ERROR: profile provenance is duplicated"
        )
    for rule in profile.rules:
        if not rule.source_ref or not rule.source_location:
            raise CompetitionProfileIntegrityError(
                "COMPETITION_PROFILE_INTEGRITY_ERROR: rule provenance is incomplete"
            )
        if rule.severity is RuleSeverity.BLOCKING and rule.source_ref not in source_refs:
            raise CompetitionProfileIntegrityError(
                "COMPETITION_PROFILE_INTEGRITY_ERROR: blocking rule source is not in "
                "profile provenance"
            )
    if profile.verification_status is ProfileVerificationStatus.VERIFIED:
        if any(_is_fixture_reference(item) for item in profile.source_refs):
            raise CompetitionProfileIntegrityError(
                "COMPETITION_PROFILE_INTEGRITY_ERROR: fixture provenance cannot be "
                "promoted to VERIFIED"
            )
        if any(
            rule.verification_status is not ProfileVerificationStatus.VERIFIED
            for rule in profile.rules
            if rule.severity is RuleSeverity.BLOCKING
        ):
            raise CompetitionProfileIntegrityError(
                "COMPETITION_PROFILE_INTEGRITY_ERROR: verified profile has unverified "
                "blocking rules"
            )
    if profile.special_rules.get("reproduction_required") is True and not any(
        rule.rule_type.value == "CUSTOM"
        and rule.severity is RuleSeverity.BLOCKING
        and rule.parameters.get("handler") == "REPRODUCTION"
        for rule in profile.rules
    ):
        raise CompetitionProfileIntegrityError(
            "COMPETITION_PROFILE_INTEGRITY_ERROR: required reproduction lacks a "
            "fail-closed blocking rule"
        )
    return competition_profile_digest(profile)


def requirement_digest(requirement: SubmissionRequirement) -> str:
    return sha256_json(requirement.model_dump(mode="json"))


def requirement_registry_digest(requirements: Iterable[SubmissionRequirement]) -> str:
    payload = [item.model_dump(mode="json") for item in requirements]
    payload.sort(key=lambda item: str(item["requirement_id"]))
    return sha256_json(payload)


def requirement_snapshot_digest(coverage: Iterable[RequirementCoverage]) -> str:
    payload = [item.model_dump(mode="json", exclude={"coverage_id"}) for item in coverage]
    payload.sort(key=lambda item: (str(item["requirement_id"]), str(item["subproblem_id"])))
    return sha256_json(payload)


def rule_result_snapshot_digest(results: Iterable[RuleResult]) -> str:
    payload = [
        item.model_dump(mode="json", exclude={"result_id", "evaluated_at"}) for item in results
    ]
    payload.sort(key=lambda item: str(item["rule_id"]))
    return sha256_json(payload)


def artifact_set_digest(artifacts: Iterable[SubmissionArtifact]) -> str:
    payload = [item.model_dump(mode="json", exclude={"storage_key"}) for item in artifacts]
    payload.sort(key=lambda item: str(item["relative_path"]))
    return sha256_json(payload)


def submission_candidate_digest(candidate: SubmissionCandidate) -> str:
    return sha256_json(candidate.model_dump(mode="json", exclude={"candidate_id"}))


def final_jury_report_digest(report: FinalJuryReport) -> str:
    return sha256_json(report.model_dump(mode="json", exclude={"report_digest"}))


def submission_check_digest(check: SubmissionCheckResult) -> str:
    return sha256_json(check.model_dump(mode="json", exclude={"check_digest"}))


def submission_snapshot_digest(snapshot: SubmissionSnapshot) -> str:
    return sha256_json(snapshot.model_dump(mode="json", exclude={"snapshot_digest"}))


def manifest_file_payload(item: Any) -> dict[str, Any]:
    return cast(dict[str, Any], item.model_dump(mode="json"))


def _is_fixture_reference(value: str) -> bool:
    lowered = value.casefold()
    return lowered.startswith(("fixture:", "mock:", "test:")) or any(
        marker in lowered for marker in ("/fixture", "test_fixture", "dummy", "placeholder")
    )
