from __future__ import annotations

from mathmodel_ai.schemas.problem_analysis import AmbiguityReviewStatus, ProblemAnalysis
from mathmodel_ai.schemas.subproblem_identity import (
    ResolvedAmbiguityIdentity,
    ResolvedSubproblemIdentity,
    SubproblemIdentityContract,
    SubproblemIdentityResolution,
    SubproblemResolutionMethod,
)


def identity_agent_note(contract: SubproblemIdentityContract) -> str:
    """Render reviewed task identities and explicitly sourced interpretation choices."""

    tasks = "\n".join(
        f"{item.source_order}. {item.canonical_subproblem_id} | "
        f"{item.display_label} | {item.canonical_goal} | source={item.source_binding}"
        for item in sorted(contract.bindings, key=lambda item: item.source_order)
    )
    ambiguities = "\n".join(
        f"- {item.canonical_ambiguity_id} | {item.display_label} | choose "
        f"{item.canonical_interpretation_id}: {item.resolution_reason} | "
        f"source={item.source_binding}"
        for item in contract.ambiguity_bindings
    )
    return (
        "REVIEWED_PROBLEM_BINDING_CONTRACT:\n"
        f"contract={contract.contract_version}; digest={contract.content_digest}; "
        f"namespace={contract.problem_namespace}.\n"
        "Return exactly one subproblem for every listed canonical id, in listed order. "
        "Use each canonical id verbatim. Do not merge, split, duplicate, or rename tasks.\n"
        f"{tasks}\n"
        "Use the following reviewed ambiguity ids and preferred interpretations verbatim; "
        "retain alternatives as recorded limitations.\n"
        f"{ambiguities}"
    )


def normalize_problem_analysis(
    analysis: ProblemAnalysis,
    contract: SubproblemIdentityContract,
) -> tuple[ProblemAnalysis, SubproblemIdentityResolution]:
    """Apply only exact reviewed identity mappings and rewrite every graph reference."""

    by_id = {item.canonical_subproblem_id: item for item in contract.bindings}
    aliases = {alias: item for item in contract.bindings for alias in item.reviewed_aliases}
    id_map: dict[str, str] = {}
    resolved: list[ResolvedSubproblemIdentity] = []
    for subproblem in analysis.subproblems:
        binding = by_id.get(subproblem.subproblem_id)
        method = SubproblemResolutionMethod.CANONICAL_ID
        if binding is None:
            binding = aliases.get(subproblem.subproblem_id)
            method = SubproblemResolutionMethod.REVIEWED_ALIAS
        if binding is None:
            raise ValueError(f"UNREVIEWED_SUBPROBLEM_IDENTITY:{subproblem.subproblem_id}")
        if subproblem.order != binding.source_order:
            raise ValueError(f"SUBPROBLEM_SOURCE_ORDER_MISMATCH:{subproblem.subproblem_id}")
        id_map[subproblem.subproblem_id] = binding.canonical_subproblem_id
        resolved.append(
            ResolvedSubproblemIdentity(
                source_subproblem_id=subproblem.subproblem_id,
                canonical_subproblem_id=binding.canonical_subproblem_id,
                internal_node_id=binding.internal_node_id,
                method=method,
            )
        )

    if len(id_map) != len(analysis.subproblems):
        raise ValueError("DUPLICATE_SOURCE_SUBPROBLEM_IDENTITY")
    if set(id_map.values()) != set(contract.canonical_ids):
        raise ValueError("CANONICAL_SUBPROBLEM_COVERAGE_MISMATCH")
    if len(id_map.values()) != len(set(id_map.values())):
        raise ValueError("CANONICAL_SUBPROBLEM_IDENTITY_COLLISION")

    ambiguity_by_id = {item.canonical_ambiguity_id: item for item in contract.ambiguity_bindings}
    ambiguity_aliases = {
        alias: item for item in contract.ambiguity_bindings for alias in item.reviewed_aliases
    }
    ambiguity_map: dict[str, str] = {}
    normalized_ambiguities = []
    resolved_ambiguities: list[ResolvedAmbiguityIdentity] = []
    for ambiguity in analysis.ambiguities:
        ambiguity_binding = ambiguity_by_id.get(ambiguity.ambiguity_id)
        method = SubproblemResolutionMethod.CANONICAL_ID
        if ambiguity_binding is None:
            ambiguity_binding = ambiguity_aliases.get(ambiguity.ambiguity_id)
            method = SubproblemResolutionMethod.REVIEWED_ALIAS
        if ambiguity_binding is None:
            normalized_ambiguities.append(ambiguity)
            continue
        interpretation_ids = {item.interpretation_id for item in ambiguity.interpretations}
        accepted = {
            ambiguity_binding.canonical_interpretation_id,
            *ambiguity_binding.reviewed_interpretation_aliases,
        }
        matches = interpretation_ids & accepted
        if len(matches) != 1:
            raise ValueError(f"REVIEWED_INTERPRETATION_BINDING_MISMATCH:{ambiguity.ambiguity_id}")
        source_interpretation_id = matches.pop()
        interpretations = [
            item.model_copy(
                update={"interpretation_id": ambiguity_binding.canonical_interpretation_id}
            )
            if item.interpretation_id == source_interpretation_id
            else item
            for item in ambiguity.interpretations
        ]
        ambiguity_map[ambiguity.ambiguity_id] = ambiguity_binding.canonical_ambiguity_id
        normalized_ambiguities.append(
            ambiguity.model_copy(
                update={
                    "ambiguity_id": ambiguity_binding.canonical_ambiguity_id,
                    "interpretations": interpretations,
                    "preferred_interpretation_id": (ambiguity_binding.canonical_interpretation_id),
                    "reason": ambiguity_binding.resolution_reason,
                    "confidence": ambiguity_binding.confidence,
                    "review_status": AmbiguityReviewStatus.RECORDED,
                }
            )
        )
        resolved_ambiguities.append(
            ResolvedAmbiguityIdentity(
                source_ambiguity_id=ambiguity.ambiguity_id,
                canonical_ambiguity_id=ambiguity_binding.canonical_ambiguity_id,
                source_interpretation_id=source_interpretation_id,
                canonical_interpretation_id=(ambiguity_binding.canonical_interpretation_id),
                method=method,
            )
        )

    def remap(value: str) -> str:
        try:
            return id_map[value]
        except KeyError as exc:
            raise ValueError(f"UNKNOWN_SUBPROBLEM_REFERENCE:{value}") from exc

    targeted_ambiguities = {value for item in analysis.subproblems for value in item.ambiguity_refs}
    if not targeted_ambiguities <= set(ambiguity_map):
        unresolved = sorted(targeted_ambiguities - set(ambiguity_map))
        raise ValueError(f"UNREVIEWED_TARGETED_AMBIGUITY:{','.join(unresolved)}")

    subproblems = [
        item.model_copy(
            update={
                "subproblem_id": remap(item.subproblem_id),
                "input_dependencies": [remap(value) for value in item.input_dependencies],
                "output_dependencies": [remap(value) for value in item.output_dependencies],
                "ambiguity_refs": [ambiguity_map[value] for value in item.ambiguity_refs],
            }
        )
        for item in analysis.subproblems
    ]
    dependencies = [
        item.model_copy(
            update={
                "upstream_id": remap(item.upstream_id),
                "downstream_id": remap(item.downstream_id),
            }
        )
        for item in analysis.dependencies_between_subproblems
    ]
    risks = [
        item.model_copy(
            update={"affected_subproblems": [remap(value) for value in item.affected_subproblems]}
        )
        for item in analysis.risks
    ]
    missing = [
        item.model_copy(
            update={"affected_subproblems": [remap(value) for value in item.affected_subproblems]}
        )
        for item in analysis.missing_information
    ]
    normalized = ProblemAnalysis.model_validate(
        analysis.model_dump()
        | {
            "subproblems": subproblems,
            "dependencies_between_subproblems": dependencies,
            "risks": risks,
            "missing_information": missing,
            "ambiguities": normalized_ambiguities,
        }
    )
    resolution = SubproblemIdentityResolution(
        contract_version=contract.contract_version,
        contract_digest=contract.content_digest,
        benchmark_id=contract.benchmark_id,
        problem_namespace=contract.problem_namespace,
        problem_sha256=contract.problem_sha256,
        bindings=sorted(
            resolved,
            key=lambda item: contract.canonical_ids.index(item.canonical_subproblem_id),
        ),
        ambiguity_bindings=sorted(
            resolved_ambiguities,
            key=lambda item: item.canonical_ambiguity_id,
        ),
    )
    return normalized, resolution
