from types import SimpleNamespace
from uuid import uuid4

import pytest

from mathmodel_ai.agents.base import AgentRunStatus
from mathmodel_ai.agents.paper import (
    PaperAgent,
    _claim_value_errors,
    _normalize_comparison_arithmetic,
    _paper_draft_errors,
    _PaperIRPrevalidation,
)
from mathmodel_ai.core.errors import ProviderResponseError
from mathmodel_ai.paper.workflow import PaperWorkflow, PaperWorkflowError
from mathmodel_ai.reasoning.prompts import PromptRegistry
from mathmodel_ai.schemas.paper import (
    PaperAgentInput,
    PaperIR,
    PaperQualityStatus,
    PaperValidationSeverity,
    ProblemRequirement,
)
from mathmodel_ai.schemas.problem_state import ProblemState
from tests.paper.helpers import (
    evidence_snapshot,
    numeric_claim,
    paper_ir,
    reference_record,
    result_evidence,
)
from tests.submission.helpers import covered_paper, subproblem


class CapturingProvider:
    async def structured_generate(self, request, _schema):
        self.request = request
        return SimpleNamespace(parsed=SimpleNamespace(root={}), response=SimpleNamespace())


def test_paper_prompt_requires_nonempty_sections_and_verbatim_symbol_meanings():
    prompt = PromptRegistry().get("paper_agent").system
    assert "Every included formal section and appendix must contain" in prompt
    assert "registered symbol table" in prompt
    assert "Preserve the source's level of certainty" in prompt


@pytest.mark.asyncio
async def test_paper_agent_requests_room_for_full_structured_document():
    evidence = result_evidence()
    provider = CapturingProvider()
    agent = PaperAgent(router=None, providers=None, prompts=PromptRegistry())
    await agent.execute(
        PaperAgentInput(
            project_id=evidence.project_id,
            assigned_paper_id=uuid4(),
            assigned_version=1,
            title="Verified model",
            evidence=[evidence],
            evidence_snapshot=evidence_snapshot(evidence),
        ),
        ProblemState(
            project_id=evidence.project_id,
            problem_id=evidence.problem_id,
            title="Verified model",
            raw_problem="Model description",
        ),
        provider,
        SimpleNamespace(selected_model="test-model", selected_reasoning=None),
    )
    assert provider.request.max_output_tokens == 65_536
    assert "Keep the structured document concise" in provider.request.messages[0].content
    assert "baseline_source_field and verified_source_field" in provider.request.messages[0].content
    assert "not an evidence ID" in provider.request.messages[0].content
    assert (
        "limitations claim's structured_value.finding_ids" in provider.request.messages[0].content
    )
    assert "abstract KEY_RESULT must be a CLAIM block" in provider.request.messages[0].content
    author_schema = _PaperIRPrevalidation.model_json_schema()
    assert "evidence_snapshot" not in author_schema["properties"]
    assert "evidence_snapshot" not in author_schema["required"]
    assert "evidence_snapshot" in PaperIR.model_json_schema()["required"]


@pytest.mark.asyncio
async def test_paper_agent_attaches_only_canonical_system_snapshot():
    evidence = result_evidence()
    snapshot = evidence_snapshot(evidence)
    provider = CapturingProvider()
    agent = PaperAgent(router=None, providers=None, prompts=PromptRegistry())
    input_data = PaperAgentInput(
        project_id=evidence.project_id,
        assigned_paper_id=uuid4(),
        assigned_version=1,
        title="Verified model",
        evidence=[evidence],
        evidence_snapshot=snapshot,
    )
    state = ProblemState(
        project_id=evidence.project_id,
        problem_id=evidence.problem_id,
        title="Verified model",
        raw_problem="Model description",
    )
    result = await agent.execute(
        input_data,
        state,
        provider,
        SimpleNamespace(selected_model="test-model", selected_reasoning=None),
    )
    assert result.output["evidence_snapshot"] == snapshot.model_dump(mode="json")

    class ConflictingProvider(CapturingProvider):
        async def structured_generate(self, request, schema):
            result = await super().structured_generate(request, schema)
            result.parsed.root["evidence_snapshot"] = {"snapshot_hash": "wrong"}
            return result

    with pytest.raises(ProviderResponseError, match="conflicting evidence snapshot"):
        await agent.execute(
            input_data,
            state,
            ConflictingProvider(),
            SimpleNamespace(selected_model="test-model", selected_reasoning=None),
        )


def test_paper_agent_retry_receives_schema_feedback():
    evidence = result_evidence()
    agent = PaperAgent(router=None, providers=None, prompts=PromptRegistry())
    input_data = PaperAgentInput(
        project_id=evidence.project_id,
        assigned_paper_id=uuid4(),
        assigned_version=1,
        title="Verified model",
        evidence=[evidence],
        evidence_snapshot=evidence_snapshot(evidence),
        repair_feedback=["prior quality gate: missing limitations"],
    )
    state = ProblemState(
        project_id=evidence.project_id,
        problem_id=evidence.problem_id,
        title="Verified model",
        raw_problem="Model description",
    )
    repaired = agent.prepare_attempt_input(
        input_data, state, ("claims.9:comparison percentage invalid",)
    )
    assert repaired.repair_feedback == [
        "prior quality gate: missing limitations",
        "claims.9:comparison percentage invalid",
    ]
    assert input_data.repair_feedback == ["prior quality gate: missing limitations"]
    with pytest.raises(PaperWorkflowError, match=r"claims\.9"):
        PaperWorkflow._require_output(
            SimpleNamespace(
                status=AgentRunStatus.ESCALATED,
                output=None,
                errors=repaired.repair_feedback,
            ),
            "PaperAgent",
        )


def test_paper_agent_preflight_identifies_structural_draft_errors():
    evidence = result_evidence()
    claim = numeric_claim(evidence, text="P represents an unrelated population; objective 30.")
    paper = paper_ir(evidence, claim)
    paper.sections[0].figure_refs.append("FIG-001")
    paper.bibliography.append("REF-uncited")
    input_data = PaperAgentInput(
        project_id=evidence.project_id,
        assigned_paper_id=paper.paper_id,
        assigned_version=paper.version,
        title=paper.title,
        evidence=[evidence],
        evidence_snapshot=paper.evidence_snapshot,
        symbol_definitions={"P": "Dimensionless abundance of fish parasites."},
        figure_ids=["FIG-001"],
        required_subproblems=[
            ProblemRequirement(subproblem_id="Q1", required_outputs=["Population response"])
        ],
    )
    errors = _paper_draft_errors(paper, input_data)
    assert any("MISSING_SUBPROBLEM:Q1" in error for error in errors)
    assert any("SYMBOL_CONFLICT:P" in error for error in errors)
    assert any("FIGURE:FIG-001" in error for error in errors)
    assert any("BIBLIOGRAPHY:REF-uncited" in error for error in errors)


def test_paper_agent_preflight_rejects_unresolvable_claim_source_field():
    evidence = result_evidence()
    claim = numeric_claim(evidence)
    paper = paper_ir(evidence, claim)
    input_data = PaperAgentInput(
        project_id=evidence.project_id,
        assigned_paper_id=paper.paper_id,
        assigned_version=paper.version,
        title=paper.title,
        evidence=[evidence],
        evidence_snapshot=paper.evidence_snapshot,
    )
    assert not any(
        error.startswith("SOURCE_FIELD:") for error in _paper_draft_errors(paper, input_data)
    )

    invalid = paper.model_copy(deep=True)
    invalid.claims[0].structured_value.source_field = "reviewed_metric_values.missing"
    errors = _paper_draft_errors(invalid, input_data)
    assert errors[0] == (
        "SOURCE_FIELD:CLAIM-objective:reviewed_metric_values.missing "
        "does not resolve in linked evidence; exact field is absent from all supplied "
        "verified evidence; relink only if the evidence supports the same claim"
    )

    other = result_evidence(project_id=evidence.project_id, problem_id=evidence.problem_id)
    other.structured_payload["reviewed_metric_values"] = {"missing": 30.0}
    with_other = input_data.model_copy(update={"evidence": [evidence, other]})
    errors = _paper_draft_errors(invalid, with_other)
    assert f"exact field exists in other verified evidence {other.evidence_id}" in errors[0]


def test_comparison_arithmetic_recomputes_only_derived_fields():
    claim = {
        "claim_type": "COMPARISON",
        "text": "The verified value increased by 25%.",
        "structured_value": {
            "baseline_value": 4.0,
            "verified_value": 5.0,
            "percentage_change": 20.0,
            "direction": "DECREASE",
            "baseline_source_field": "a",
            "verified_source_field": "b",
        },
    }
    raw = {"claims": [claim]}
    _normalize_comparison_arithmetic(raw)
    assert claim["structured_value"]["baseline_value"] == 4.0
    assert claim["structured_value"]["verified_value"] == 5.0
    assert claim["structured_value"]["percentage_change"] == 25.0
    assert claim["structured_value"]["direction"] == "INCREASE"
    assert claim["text"] == "The verified value increased by 25%."

    incomplete = {"claim_type": "COMPARISON", "structured_value": {"baseline_value": 0.0}}
    _normalize_comparison_arithmetic({"claims": [incomplete]})
    assert incomplete["structured_value"] == {"baseline_value": 0.0}


def test_claim_diagnostics_expose_nested_value_errors_without_accepting_claim():
    raw = {
        "claims": [
            {
                "claim_type": "COMPARISON",
                "structured_value": {
                    "baseline_value": 4.0,
                    "verified_value": 5.0,
                    "percentage_change": 25.0,
                    "direction": "INCREASE",
                },
            }
        ]
    }
    details = _claim_value_errors(raw)
    assert "claims.0.structured_value.baseline_source_field:missing" in details
    assert "claims.0.structured_value.verified_source_field:missing" in details


def test_failed_paper_feedback_is_scoped_to_same_science_snapshot():
    evidence = result_evidence()
    snapshot = evidence_snapshot(evidence)
    repository = SimpleNamespace(
        get_version=lambda *args: SimpleNamespace(evidence_snapshot=snapshot),
        get_quality=lambda *args: SimpleNamespace(
            status=PaperQualityStatus.FAILED,
            issues=[
                SimpleNamespace(
                    code="MISSING_SUBPROBLEM",
                    object_ref="Q2",
                    message="coverage points to missing paper objects",
                    severity=PaperValidationSeverity.ERROR,
                )
            ],
        ),
    )
    workflow = object.__new__(PaperWorkflow)
    workflow._repository = repository
    bundle = SimpleNamespace(
        state=SimpleNamespace(project_id=evidence.project_id), snapshot=snapshot
    )
    feedback = workflow._previous_paper_feedback(bundle, uuid4(), 2)
    assert len(feedback) == 1
    assert "MISSING_SUBPROBLEM (Q2)" in feedback[0]
    repository.get_quality = lambda *args: SimpleNamespace(
        status=PaperQualityStatus.HUMAN_REVIEW,
        issues=[
            SimpleNamespace(
                code="INDEPENDENT_FACTUAL_FINDING",
                object_ref="CLAIM-006",
                message="The claim overstates a hypothesized mechanism.",
                severity=PaperValidationSeverity.ERROR,
            )
        ],
    )
    feedback = workflow._previous_paper_feedback(bundle, uuid4(), 2)
    assert "INDEPENDENT_FACTUAL_FINDING (CLAIM-006)" in feedback[0]
    assert "overstates a hypothesized mechanism" in feedback[0]
    repository.get_version = lambda *args: SimpleNamespace(
        evidence_snapshot=snapshot.model_copy(update={"verified_result_id": uuid4()})
    )
    assert workflow._previous_paper_feedback(bundle, uuid4(), 2) == []


def test_paper_retry_restores_reference_order_without_accepting_changed_identity():
    project_id = uuid4()
    first = reference_record(project_id)
    second = first.model_copy(update={"reference_id": "REF-second"})
    snapshot = evidence_snapshot(result_evidence()).model_copy(
        update={"citation_reference_ids": [second.reference_id, first.reference_id]}
    )
    assert PaperWorkflow._references_in_snapshot_order([first, second], snapshot) == [second, first]
    with pytest.raises(PaperWorkflowError, match="reference identities"):
        PaperWorkflow._references_in_snapshot_order([first], snapshot)
    with pytest.raises(PaperWorkflowError, match="reference identities"):
        PaperWorkflow._references_in_snapshot_order([first, first], snapshot)


def test_ready_paper_with_partial_final_coverage_receives_repair_feedback():
    output = "Evaluate final solution risk"
    paper = covered_paper(("Q1", output))
    issues = PaperWorkflow._submission_coverage_issues(paper, [subproblem("Q1", output)])
    assert len(issues) == 1
    assert issues[0].code == "FINAL_REQUIREMENT_COVERAGE"
    repository = SimpleNamespace(
        get_version=lambda *args: SimpleNamespace(
            evidence_snapshot=paper.evidence_snapshot, paper_ir=paper
        ),
        get_quality=lambda *args: SimpleNamespace(
            status=PaperQualityStatus.READY_FOR_FINAL_JURY, issues=[]
        ),
    )
    workflow = object.__new__(PaperWorkflow)
    workflow._repository = repository
    bundle = SimpleNamespace(
        state=SimpleNamespace(
            project_id=paper.claims[0].project_id,
            subproblems=[subproblem("Q1", output)],
            problem_analysis=None,
        ),
        snapshot=paper.evidence_snapshot,
    )
    feedback = workflow._previous_paper_feedback(bundle, paper.paper_id, 2)
    assert "FINAL_REQUIREMENT_COVERAGE (REQ-Q1-1)" in feedback[0]
