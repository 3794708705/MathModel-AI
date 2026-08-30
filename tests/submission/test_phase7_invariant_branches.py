from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from mathmodel_ai.schemas.paper import PaperQualityStatus
from mathmodel_ai.schemas.submission import (
    CorrectionPlan,
    CorrectionScope,
    JuryDecision,
    RequirementCoverage,
    RequirementCoverageStatus,
    RuleResult,
    RuleResultStatus,
    RuleSeverity,
    RuleType,
    SubmissionCheckStatus,
)
from mathmodel_ai.submission.checks import SubmissionCheck
from mathmodel_ai.submission.corrections import CorrectionWorkflow, InvalidationEngine
from mathmodel_ai.submission.freeze import SubmissionFreeze
from mathmodel_ai.submission.jury import FinalJuryGate
from mathmodel_ai.submission.package import SubmissionPackageBuilder
from mathmodel_ai.submission.profiles import (
    CompetitionProfileRegistry,
    generic_modeling_test_profile,
)
from mathmodel_ai.submission.requirements import RequirementCoverageValidator, RequirementRegistry
from mathmodel_ai.submission.rules import RuleEngine
from tests.submission.helpers import (
    candidate_fixture,
    covered_paper,
    jury_draft,
    subproblem,
)


def _coverage(
    status: RequirementCoverageStatus = RequirementCoverageStatus.COVERED,
) -> list[RequirementCoverage]:
    return [
        RequirementCoverage(
            requirement_id="REQ-Q1-1",
            requirement_digest="a" * 64,
            subproblem_id="Q1",
            status=status,
            evidence_refs=[UUID(int=0x7101)] if status is RequirementCoverageStatus.COVERED else [],
            paper_refs=["SEC-results"] if status is RequirementCoverageStatus.COVERED else [],
            artifact_refs=[UUID(int=0x7102)] if status is RequirementCoverageStatus.COVERED else [],
        )
    ]


def _accepted_inputs(tmp_path):
    store, profile, candidate, payloads = candidate_fixture(tmp_path)
    paper = covered_paper(("Q1", "answer"))
    submission_requirements = RequirementRegistry.from_subproblems([subproblem("Q1", "answer")])
    coverage = RequirementCoverageValidator().validate(
        submission_requirements, paper, candidate.artifacts
    )
    rules = RuleEngine(store).evaluate(profile, candidate, artifact_bytes=payloads)
    jury = FinalJuryGate().build_report(
        draft=jury_draft(),
        profile=profile,
        candidate=candidate,
        requirements=coverage,
        rule_results=rules,
        reviewer_is_mock=True,
        agent_run_id=uuid4(),
    )
    check = SubmissionCheck().run(
        profile=profile,
        candidate=candidate,
        requirements=coverage,
        rules=rules,
        jury=jury,
        artifact_bytes=payloads,
    )
    assert jury.decision is JuryDecision.PASS
    assert check.status is SubmissionCheckStatus.PASS
    return (
        store,
        profile,
        candidate,
        payloads,
        paper,
        submission_requirements,
        coverage,
        rules,
        jury,
        check,
    )


def _plan(scope: CorrectionScope, **updates) -> CorrectionPlan:
    values = {
        "project_id": uuid4(),
        "finding_refs": ["JURY-1"],
        "scope": scope,
        "affected_components": ["paper"],
        "requires_model_change": False,
        "requires_result_change": False,
        "requires_paper_change": True,
        "required_revalidation": ["PAPER", "FINAL_JURY", "SUBMISSION"],
        "risk": RuleSeverity.MAJOR,
        "priority": 1,
    }
    values.update(updates)
    return CorrectionPlan(**values)


@pytest.mark.parametrize(
    ("plan", "message"),
    [
        (
            _plan(CorrectionScope.PAPER_ONLY, requires_model_change=True),
            "invalidate MODEL",
        ),
        (
            _plan(CorrectionScope.CITATION, requires_result_change=True),
            "invalidate SOLVE",
        ),
        (
            _plan(
                CorrectionScope.CODE,
                requires_paper_change=False,
                required_revalidation=["ROBUSTNESS"],
            ),
            "exceeds its deterministic scope",
        ),
    ],
)
def test_correction_scope_rejects_unauthorized_invalidation(plan, message) -> None:
    with pytest.raises(ValueError, match=message):
        InvalidationEngine().evaluate(plan)


def test_correction_scope_requires_paper_stage_and_limits_auto_fix(monkeypatch) -> None:
    malformed = _plan(
        CorrectionScope.PAPER_ONLY,
        required_revalidation=["FINAL_JURY"],
    )
    monkeypatch.setitem(
        InvalidationEngine._STAGES,
        CorrectionScope.PAPER_ONLY,
        ["FINAL_JURY", "SUBMISSION"],
    )
    with pytest.raises(ValueError, match="invalidate PAPER"):
        InvalidationEngine().evaluate(malformed)

    format_plan = _plan(CorrectionScope.FORMAT_ONLY, affected_components=["spacing"])
    assert CorrectionWorkflow().validate_auto_fix(format_plan)
    assert not CorrectionWorkflow().validate_auto_fix(
        format_plan.model_copy(update={"requires_result_change": True})
    )


def test_freeze_requires_passing_jury_and_submission_check(tmp_path) -> None:
    (
        store,
        profile,
        candidate,
        payloads,
        paper,
        submission_requirements,
        coverage,
        rules,
        jury,
        check,
    ) = _accepted_inputs(tmp_path)
    freeze = SubmissionFreeze(SubmissionPackageBuilder(store), RuleEngine(store))
    with pytest.raises(ValueError, match="Final Jury must pass"):
        freeze.freeze(
            profile=profile,
            candidate=candidate,
            paper=paper,
            submission_requirements=submission_requirements,
            requirements=coverage,
            rules=rules,
            jury=jury.model_copy(update={"decision": JuryDecision.FAIL}),
            check=check,
            artifact_bytes=payloads,
        )
    with pytest.raises(ValueError, match="SubmissionCheck must pass"):
        freeze.freeze(
            profile=profile,
            candidate=candidate,
            paper=paper,
            submission_requirements=submission_requirements,
            requirements=coverage,
            rules=rules,
            jury=jury,
            check=check.model_copy(update={"status": SubmissionCheckStatus.WARNING}),
            artifact_bytes=payloads,
        )


@pytest.mark.parametrize(
    "target",
    ["jury_project", "jury_paper", "jury_version", "jury_result", "check_project", "check_jury"],
)
def test_freeze_rejects_cross_candidate_identity_mix(tmp_path, target) -> None:
    (
        store,
        profile,
        candidate,
        payloads,
        paper,
        submission_requirements,
        coverage,
        rules,
        jury,
        check,
    ) = _accepted_inputs(tmp_path)
    if target == "jury_project":
        jury = jury.model_copy(update={"project_id": uuid4()})
    elif target == "jury_paper":
        jury = jury.model_copy(update={"paper_id": uuid4()})
    elif target == "jury_version":
        jury = jury.model_copy(update={"paper_version": 2})
    elif target == "jury_result":
        jury = jury.model_copy(update={"verified_result_id": uuid4()})
    elif target == "check_project":
        check = check.model_copy(update={"project_id": uuid4()})
    else:
        check = check.model_copy(update={"jury_report_id": uuid4()})
    with pytest.raises(ValueError, match="same approved candidate"):
        SubmissionFreeze(SubmissionPackageBuilder(store), RuleEngine(store)).freeze(
            profile=profile,
            candidate=candidate,
            paper=paper,
            submission_requirements=submission_requirements,
            requirements=coverage,
            rules=rules,
            jury=jury,
            check=check,
            artifact_bytes=payloads,
        )


def test_freeze_success_delegates_to_roundtrip_verified_builder(tmp_path) -> None:
    (
        store,
        profile,
        candidate,
        payloads,
        paper,
        submission_requirements,
        coverage,
        rules,
        jury,
        check,
    ) = _accepted_inputs(tmp_path)
    package = SubmissionFreeze(SubmissionPackageBuilder(store), RuleEngine(store)).freeze(
        profile=profile,
        candidate=candidate,
        paper=paper,
        submission_requirements=submission_requirements,
        requirements=coverage,
        rules=rules,
        jury=jury,
        check=check,
        artifact_bytes=payloads,
    )
    assert package.snapshot.submission_id == candidate.candidate_id


def test_profile_registry_enforces_exact_immutable_versions() -> None:
    profile = generic_modeling_test_profile()
    registry = CompetitionProfileRegistry([profile])
    registry.register(profile)
    assert registry.get(profile.profile_id, profile.version) == profile
    assert registry.list() == [profile]
    with pytest.raises(KeyError, match="was not found"):
        registry.get(profile.profile_id, profile.version + 1)
    with pytest.raises(ValueError, match="conflicting content"):
        registry.register(profile.model_copy(update={"name": "tampered"}))


def test_requirement_coverage_covers_not_applicable_and_missing_evidence(tmp_path) -> None:
    _, _, candidate, _ = candidate_fixture(tmp_path)
    requirement = RequirementRegistry.from_subproblems([subproblem("Q1", "answer")])[0]
    paper = covered_paper(("Q1", "answer"))

    not_applicable = requirement.model_copy(update={"required": False})
    result = RequirementCoverageValidator().validate([not_applicable], paper, candidate.artifacts)[
        0
    ]
    assert result.status is RequirementCoverageStatus.NOT_APPLICABLE

    no_pdf = RequirementCoverageValidator().validate([requirement], paper, [])[0]
    assert no_pdf.status is RequirementCoverageStatus.PARTIAL
    assert "formal paper artifact is missing" in no_pdf.reasons

    wrong_output = paper.subproblem_coverage[0].model_copy(
        update={"required_outputs": ["different"]}
    )
    result = RequirementCoverageValidator().validate(
        [requirement],
        paper.model_copy(update={"subproblem_coverage": [wrong_output]}),
        candidate.artifacts,
    )[0]
    assert result.status is RequirementCoverageStatus.PARTIAL
    assert "required output is absent" in result.reasons[0]


def test_jury_rejects_dimension_schema_score_and_candidate_tampering(tmp_path) -> None:
    store, profile, candidate, payloads = candidate_fixture(tmp_path)
    rules = RuleEngine(store).evaluate(profile, candidate, artifact_bytes=payloads)
    gate = FinalJuryGate()

    missing_dimension = jury_draft().model_copy(update={"dimensions": jury_draft().dimensions[:-1]})
    assert (
        gate.build_report(
            draft=missing_dimension,
            profile=profile,
            candidate=candidate,
            requirements=_coverage(),
            rule_results=rules,
            reviewer_is_mock=True,
            agent_run_id=None,
        ).decision
        is JuryDecision.FAIL
    )

    wrong_maximum = jury_draft()
    changed = wrong_maximum.dimensions[0].model_copy(
        update={"maximum": wrong_maximum.dimensions[0].maximum - 1}
    )
    wrong_maximum = wrong_maximum.model_copy(
        update={"dimensions": [changed, *wrong_maximum.dimensions[1:]]}
    )
    assert FinalJuryGate.recompute_score(wrong_maximum.dimensions, profile) == (0.0, False)

    report = gate.build_report(
        draft=jury_draft(),
        profile=profile,
        candidate=candidate,
        requirements=_coverage(),
        rule_results=rules,
        reviewer_is_mock=True,
        agent_run_id=None,
    )
    assert (
        gate.verify_report(
            report.model_copy(update={"paper_id": uuid4()}),
            profile=profile,
            candidate=candidate,
            requirements=_coverage(),
            rule_results=rules,
        )
        is JuryDecision.FAIL
    )


def test_jury_requires_independent_reviewer_and_minimum_score(tmp_path) -> None:
    store, profile, candidate, payloads = candidate_fixture(tmp_path)
    rules = RuleEngine(store).evaluate(profile, candidate, artifact_bytes=payloads)
    gate = FinalJuryGate()
    independent = profile.model_copy(update={"require_independent_reviewer": True})
    assert (
        gate.build_report(
            draft=jury_draft(),
            profile=independent,
            candidate=candidate,
            requirements=_coverage(),
            rule_results=rules,
            reviewer_is_mock=True,
            agent_run_id=None,
        ).decision
        is JuryDecision.HUMAN_REVIEW
    )
    high_threshold = profile.model_copy(update={"minimum_jury_score": 100})
    assert (
        gate.build_report(
            draft=jury_draft(innovation=0),
            profile=high_threshold,
            candidate=candidate,
            requirements=_coverage(),
            rule_results=rules,
            reviewer_is_mock=False,
            agent_run_id=None,
        ).decision
        is JuryDecision.FAIL
    )


def test_jury_deduplicates_deterministic_and_draft_findings(tmp_path) -> None:
    _store, profile, candidate, _payloads = candidate_fixture(tmp_path)
    candidate = candidate.model_copy(
        update={"phase5_verified": False, "paper_status": PaperQualityStatus.FAILED}
    )
    failed_rule = RuleResult(
        rule_id="RULE-block",
        rule_type=RuleType.CUSTOM,
        severity=RuleSeverity.BLOCKING,
        status=RuleResultStatus.FAIL,
        message="failed",
    )
    report = FinalJuryGate().build_report(
        draft=jury_draft(),
        profile=profile,
        candidate=candidate,
        requirements=_coverage(RequirementCoverageStatus.PARTIAL),
        rule_results=[failed_rule],
        reviewer_is_mock=True,
        agent_run_id=None,
    )
    assert report.decision is JuryDecision.FAIL
    assert {item.finding_id for item in report.findings} >= {
        "JURY-phase5",
        "JURY-paper",
        "JURY-REQ-Q1-1",
        "JURY-RULE-block",
    }


def test_jury_separately_rejects_paper_gate_and_blocking_rule(tmp_path) -> None:
    _store, profile, candidate, _payloads = candidate_fixture(tmp_path)
    paper_failed = candidate.model_copy(update={"paper_status": PaperQualityStatus.FAILED})
    assert (
        FinalJuryGate()
        .build_report(
            draft=jury_draft(),
            profile=profile,
            candidate=paper_failed,
            requirements=_coverage(),
            rule_results=[],
            reviewer_is_mock=True,
            agent_run_id=None,
        )
        .decision
        is JuryDecision.FAIL
    )
    failed_rule = RuleResult(
        rule_id="RULE-block",
        rule_type=RuleType.CUSTOM,
        severity=RuleSeverity.BLOCKING,
        status=RuleResultStatus.FAIL,
        message="failed",
    )
    assert (
        FinalJuryGate()
        .build_report(
            draft=jury_draft(),
            profile=profile,
            candidate=candidate,
            requirements=_coverage(),
            rule_results=[failed_rule],
            reviewer_is_mock=True,
            agent_run_id=None,
        )
        .decision
        is JuryDecision.FAIL
    )


def test_requirement_coverage_ignores_missing_and_unsupported_claim_refs(tmp_path) -> None:
    _, _, candidate, _ = candidate_fixture(tmp_path)
    requirement = RequirementRegistry.from_subproblems([subproblem("Q1", "answer")])
    paper = covered_paper(("Q1", "answer"))
    missing_ref = paper.subproblem_coverage[0].model_copy(
        update={"claim_refs": ["CLAIM-does-not-exist"]}
    )
    result = RequirementCoverageValidator().validate(
        requirement,
        paper.model_copy(update={"subproblem_coverage": [missing_ref]}),
        candidate.artifacts,
    )[0]
    assert result.status is RequirementCoverageStatus.PARTIAL

    unsupported = paper.claims[0].model_copy(update={"verification_status": "UNVERIFIED"})
    result = RequirementCoverageValidator().validate(
        requirement,
        paper.model_copy(update={"claims": [unsupported]}),
        candidate.artifacts,
    )[0]
    assert result.status is RequirementCoverageStatus.PARTIAL
