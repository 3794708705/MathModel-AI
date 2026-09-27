from mathmodel_ai.schemas.paper import ClaimType, PaperSectionType
from mathmodel_ai.schemas.problem_analysis import ProblemTaskType
from mathmodel_ai.submission.requirements import (
    RequirementCoverageValidator,
    RequirementRegistry,
    _allowed_claim_types,
)
from mathmodel_ai.submission.workflow import FinalSubmissionWorkflow
from tests.submission.helpers import covered_paper, subproblem


def test_dedicated_abstract_is_exposed_to_competition_rules() -> None:
    paper = covered_paper(("Q1", "answer"))
    assert paper.abstract
    assert PaperSectionType.ABSTRACT.value in FinalSubmissionWorkflow._section_types(paper)


def test_conceptual_outputs_accept_evidence_linked_model_claims() -> None:
    assert ClaimType.MODEL in _allowed_claim_types(
        "Model specification", [ProblemTaskType.EVALUATION]
    )
    assert ClaimType.MODEL in _allowed_claim_types(
        "Definition of the stability criterion", [ProblemTaskType.EVALUATION]
    )
    assert ClaimType.MODEL in _allowed_claim_types(
        "Analysis of ecological interactions", [ProblemTaskType.PREDICTION]
    )
    assert ClaimType.MODEL in _allowed_claim_types(
        "Explicit treatment of parasites", [ProblemTaskType.PREDICTION]
    )
    assert ClaimType.MODEL in _allowed_claim_types(
        "Identification of affected species and interactions", [ProblemTaskType.EVALUATION]
    )
    assert ClaimType.MODEL not in _allowed_claim_types(
        "Identification of optimal solution", [ProblemTaskType.OPTIMIZATION]
    )
    assert ClaimType.CONCLUSION in _allowed_claim_types(
        "Discussion of alternative interpretation", [ProblemTaskType.PREDICTION]
    )
    assert ClaimType.CONCLUSION in _allowed_claim_types(
        "Discussion of alternative interpretation", [ProblemTaskType.NETWORK]
    )
    assert ClaimType.NUMERIC in _allowed_claim_types(
        "Stability analysis of the ecosystem", [ProblemTaskType.EVALUATION]
    )
    assert ClaimType.MODEL not in _allowed_claim_types(
        "Stability analysis of the ecosystem", [ProblemTaskType.EVALUATION]
    )


def test_generic_risk_does_not_accept_model_or_unrelated_numeric_claim() -> None:
    allowed = _allowed_claim_types("Evaluate final solution risk", [ProblemTaskType.EVALUATION])
    assert ClaimType.MODEL not in allowed
    assert ClaimType.NUMERIC not in allowed


def test_paper_stage_can_reuse_final_coverage_without_faking_pdf_artifact() -> None:
    requirement = RequirementRegistry.from_subproblems([subproblem("Q1", "answer")])
    paper = covered_paper(("Q1", "answer"))
    coverage = RequirementCoverageValidator().validate(
        requirement, paper, [], require_paper_artifact=False
    )
    assert coverage[0].status.value == "COVERED"


def test_visible_claim_block_can_bind_output_without_redundant_section_index() -> None:
    requirement = RequirementRegistry.from_subproblems([subproblem("Q1", "answer")])
    paper = covered_paper(("Q1", "answer"))
    section = paper.sections[0].model_copy(update={"claim_refs": []})
    paper = paper.model_copy(update={"sections": [section]})
    covered = RequirementCoverageValidator().validate(
        requirement, paper, [], require_paper_artifact=False
    )
    assert covered[0].status.value == "COVERED"
    section = section.model_copy(update={"blocks": []})
    paper = paper.model_copy(update={"sections": [section]})
    missing = RequirementCoverageValidator().validate(
        requirement, paper, [], require_paper_artifact=False
    )
    assert missing[0].status.value == "PARTIAL"
