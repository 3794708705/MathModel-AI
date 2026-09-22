from __future__ import annotations

import pytest
from pydantic import ValidationError

from mathmodel_ai.reasoning.subproblem_identity import (
    identity_agent_note,
    normalize_problem_analysis,
)
from mathmodel_ai.schemas.problem_analysis import MissingInformation, RiskItem
from mathmodel_ai.schemas.subproblem_identity import (
    ReviewedAmbiguityBinding,
    SubproblemIdentityBinding,
    SubproblemIdentityContract,
    SubproblemResolutionMethod,
)
from tests.reasoning.helpers import analysis_fixture

ZERO = "0" * 64


def contract() -> SubproblemIdentityContract:
    return SubproblemIdentityContract(
        contract_version="1",
        benchmark_id="BENCH-TEST",
        problem_namespace="TEST-CASE",
        problem_sha256="1" * 64,
        bindings=[
            SubproblemIdentityBinding(
                canonical_subproblem_id=f"Q{index}",
                display_label=f"Official task {index}",
                source_binding=f"RESOURCE:page-1:task-{index}",
                internal_node_id=f"TEST-Q{index}",
                source_order=index,
                canonical_goal=f"Complete official task {index}.",
                reviewed_aliases=[f"Q-SEMANTIC-{index}"],
            )
            for index in range(1, 4)
        ],
        ambiguity_bindings=[
            ReviewedAmbiguityBinding(
                canonical_ambiguity_id="AMB-demand",
                display_label="Demand interpretation",
                source_binding="RESOURCE:page-1:demand",
                canonical_subproblem_ids=["Q1"],
                canonical_interpretation_id="forecast",
                resolution_reason="The planning period is future-facing.",
                confidence=0.9,
            )
        ],
        content_digest=ZERO,
    )


def aliased_analysis():
    analysis = analysis_fixture()
    mapping = {f"Q{index}": f"Q-SEMANTIC-{index}" for index in range(1, 4)}
    subproblems = [
        item.model_copy(
            update={
                "subproblem_id": mapping[item.subproblem_id],
                "input_dependencies": [mapping[value] for value in item.input_dependencies],
                "output_dependencies": [mapping[value] for value in item.output_dependencies],
            }
        )
        for item in analysis.subproblems
    ]
    dependencies = [
        item.model_copy(
            update={
                "upstream_id": mapping[item.upstream_id],
                "downstream_id": mapping[item.downstream_id],
            }
        )
        for item in analysis.dependencies_between_subproblems
    ]
    return analysis.model_copy(
        update={
            "subproblems": subproblems,
            "dependencies_between_subproblems": dependencies,
            "risks": [
                RiskItem(
                    risk_id="RISK-1",
                    description="A linked risk",
                    severity=2,
                    affected_subproblems=["Q-SEMANTIC-2"],
                )
            ],
            "missing_information": [
                MissingInformation(
                    item_id="MISS-1",
                    description="A linked missing input",
                    impact="May weaken task 3",
                    affected_subproblems=["Q-SEMANTIC-3"],
                )
            ],
        }
    )


def test_reviewed_aliases_rewrite_the_complete_subproblem_graph() -> None:
    normalized, resolution = normalize_problem_analysis(aliased_analysis(), contract())

    assert [item.subproblem_id for item in normalized.subproblems] == ["Q1", "Q2", "Q3"]
    assert normalized.subproblems[0].output_dependencies == ["Q2"]
    assert normalized.subproblems[1].input_dependencies == ["Q1"]
    assert normalized.dependencies_between_subproblems[1].downstream_id == "Q3"
    assert normalized.risks[0].affected_subproblems == ["Q2"]
    assert normalized.missing_information[0].affected_subproblems == ["Q3"]
    assert [item.internal_node_id for item in resolution.bindings] == [
        "TEST-Q1",
        "TEST-Q2",
        "TEST-Q3",
    ]
    assert {item.method for item in resolution.bindings} == {
        SubproblemResolutionMethod.REVIEWED_ALIAS
    }
    assert resolution.ambiguity_bindings[0].canonical_ambiguity_id == "AMB-demand"
    assert normalized.ambiguities[0].preferred_interpretation_id == "forecast"
    assert normalized.ambiguities[0].confidence == 0.9


def test_unreviewed_identity_and_wrong_source_order_fail_closed() -> None:
    analysis = aliased_analysis()
    unknown = analysis.model_copy(
        update={
            "subproblems": [
                analysis.subproblems[0].model_copy(update={"subproblem_id": "Q-UNKNOWN"}),
                *analysis.subproblems[1:],
            ]
        }
    )
    with pytest.raises(ValueError, match="UNREVIEWED_SUBPROBLEM_IDENTITY"):
        normalize_problem_analysis(unknown, contract())

    wrong_order = analysis.model_copy(
        update={
            "subproblems": [
                *analysis.subproblems[:2],
                analysis.subproblems[2].model_copy(update={"subproblem_id": "Q-SEMANTIC-2"}),
            ]
        }
    )
    with pytest.raises(ValueError, match="SUBPROBLEM_SOURCE_ORDER_MISMATCH"):
        normalize_problem_analysis(wrong_order, contract())


def test_contract_rejects_cross_task_alias_collision() -> None:
    value = contract().model_dump()
    value["bindings"][1]["reviewed_aliases"] = ["Q-SEMANTIC-1"]
    with pytest.raises(ValidationError, match="globally unique"):
        SubproblemIdentityContract.model_validate(value)


def test_agent_note_requires_exact_complete_canonical_identity() -> None:
    note = identity_agent_note(contract())

    assert "Do not merge, split, duplicate, or rename tasks" in note
    assert all(f"{index}. Q{index} |" in note for index in range(1, 4))
