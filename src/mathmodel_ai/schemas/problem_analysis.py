from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EvidenceType(StrEnum):
    FACT = "FACT"
    DATA = "DATA"
    ASSUMPTION = "ASSUMPTION"
    DERIVATION = "DERIVATION"
    RESULT = "RESULT"
    EXTERNAL_EVIDENCE = "EXTERNAL_EVIDENCE"


class EvidenceStatus(StrEnum):
    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class EvidenceSource(StrEnum):
    PROBLEM_TEXT = "problem_text"
    COMPETITION_CONTEXT = "competition_context"
    USER_NOTE = "user_note"
    AGENT_DERIVATION = "agent_derivation"


class EvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(pattern=r"^EVID-[A-Za-z0-9_-]+$")
    type: EvidenceType
    content: str = Field(min_length=1)
    source: EvidenceSource
    source_location: str | None = None
    confidence: float = Field(ge=0, le=1)
    status: EvidenceStatus
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def proposed_assumptions_are_not_accepted(self) -> "EvidenceItem":
        if self.type is EvidenceType.ASSUMPTION and self.status is not EvidenceStatus.PROPOSED:
            raise ValueError("ProblemAgent assumptions must remain proposed in Phase 2")
        if (
            self.type in {EvidenceType.FACT, EvidenceType.DATA}
            and self.status is EvidenceStatus.PROPOSED
        ):
            raise ValueError("facts and stated data cannot use proposed status")
        return self


class ProblemTaskType(StrEnum):
    PREDICTION = "prediction"
    OPTIMIZATION = "optimization"
    EVALUATION = "evaluation"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    TIME_SERIES = "time_series"
    NETWORK = "network"
    SIMULATION = "simulation"
    DECISION = "decision"
    STATISTICAL_ANALYSIS = "statistical_analysis"
    DYNAMIC_SYSTEM = "dynamic_system"
    MULTI_OBJECTIVE = "multi_objective"
    UNCERTAINTY = "uncertainty"


class StructuredText(BaseModel):
    item_id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    evidence_refs: list[str] = Field(default_factory=list)


class CoreProblem(BaseModel):
    statement: str = Field(min_length=1)
    success_criteria: list[str] = Field(min_length=1)
    evidence_refs: list[str] = Field(default_factory=list)


class ObjectiveItem(BaseModel):
    objective_id: str = Field(pattern=r"^OBJ-[A-Za-z0-9_-]+$")
    description: str = Field(min_length=1)
    priority: int = Field(default=1, ge=1)
    required_output: str = Field(min_length=1)
    evidence_refs: list[str] = Field(default_factory=list)


class ConditionItem(BaseModel):
    condition_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    evidence_refs: list[str] = Field(default_factory=list)
    is_explicit: bool


class DataAvailability(StrEnum):
    PROVIDED = "provided"
    DERIVABLE = "derivable"
    EXTERNAL_REQUIRED = "external_required"
    MISSING = "missing"
    UNKNOWN = "unknown"


class DataRequirement(BaseModel):
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    availability: DataAvailability
    evidence_refs: list[str] = Field(default_factory=list)


class SubProblem(BaseModel):
    subproblem_id: str = Field(pattern=r"^Q[A-Za-z0-9_-]+$")
    order: int = Field(ge=1)
    original_text: str = Field(min_length=1)
    normalized_goal: str = Field(min_length=1)
    output_required: list[str] = Field(min_length=1)
    task_types: list[ProblemTaskType] = Field(min_length=1)
    input_dependencies: list[str] = Field(default_factory=list)
    output_dependencies: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    data_requirements: list[DataRequirement] = Field(default_factory=list)
    ambiguity_refs: list[str] = Field(default_factory=list)


class SubProblemDependency(BaseModel):
    upstream_id: str = Field(min_length=1)
    downstream_id: str = Field(min_length=1)
    transferred_output: str = Field(min_length=1)

    @model_validator(mode="after")
    def reject_self_dependency(self) -> "SubProblemDependency":
        if self.upstream_id == self.downstream_id:
            raise ValueError("a subproblem cannot depend on itself")
        return self


class Interpretation(BaseModel):
    interpretation_id: str = Field(min_length=1)
    meaning: str = Field(min_length=1)
    support: str = Field(min_length=1)
    implications: list[str] = Field(default_factory=list)


class AmbiguityReviewStatus(StrEnum):
    RECORDED = "RECORDED"
    HUMAN_REVIEW_RECOMMENDED = "HUMAN_REVIEW_RECOMMENDED"


class Ambiguity(BaseModel):
    ambiguity_id: str = Field(pattern=r"^AMB-[A-Za-z0-9_-]+$")
    description: str = Field(min_length=1)
    interpretations: list[Interpretation] = Field(min_length=2)
    preferred_interpretation_id: str | None = None
    reason: str | None = None
    confidence: float = Field(ge=0, le=1)
    review_status: AmbiguityReviewStatus = AmbiguityReviewStatus.RECORDED

    @model_validator(mode="after")
    def preferred_interpretation_must_exist(self) -> "Ambiguity":
        ids = {item.interpretation_id for item in self.interpretations}
        if self.preferred_interpretation_id is not None:
            if self.preferred_interpretation_id not in ids:
                raise ValueError("preferred interpretation must reference an interpretation")
            if not self.reason:
                raise ValueError("a preferred interpretation requires a reason")
        return self


class EntityItem(BaseModel):
    entity_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    role: str = Field(min_length=1)
    attributes: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)


class ScopeItem(BaseModel):
    description: str = Field(min_length=1)
    start: str | None = None
    end: str | None = None
    granularity: str | None = None
    evidence_refs: list[str] = Field(default_factory=list)


class ResourceItem(BaseModel):
    resource_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    limitation: str | None = None
    evidence_refs: list[str] = Field(default_factory=list)


class RiskItem(BaseModel):
    risk_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    severity: int = Field(ge=1, le=5)
    affected_subproblems: list[str] = Field(default_factory=list)
    mitigation: str | None = None


class MissingInformation(BaseModel):
    item_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    impact: str = Field(min_length=1)
    affected_subproblems: list[str] = Field(default_factory=list)


class ProblemAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    background: list[StructuredText] = Field(default_factory=list)
    core_problem: CoreProblem
    objectives: list[ObjectiveItem] = Field(min_length=1)
    subproblems: list[SubProblem] = Field(min_length=1)

    facts: list[EvidenceItem] = Field(default_factory=list)
    data_items: list[EvidenceItem] = Field(default_factory=list)
    derivations: list[EvidenceItem] = Field(default_factory=list)
    explicit_constraints: list[ConditionItem] = Field(default_factory=list)
    implicit_conditions: list[ConditionItem] = Field(default_factory=list)

    ambiguities: list[Ambiguity] = Field(default_factory=list)
    assumptions_required: list[EvidenceItem] = Field(default_factory=list)

    entities: list[EntityItem] = Field(default_factory=list)
    time_scope: ScopeItem | None = None
    spatial_scope: ScopeItem | None = None
    resources: list[ResourceItem] = Field(default_factory=list)

    task_types: list[ProblemTaskType] = Field(min_length=1)
    dependencies_between_subproblems: list[SubProblemDependency] = Field(default_factory=list)
    risks: list[RiskItem] = Field(default_factory=list)
    missing_information: list[MissingInformation] = Field(default_factory=list)
    no_facts_reason: str | None = None
    confidence: float = Field(ge=0, le=1)
    human_review_recommended: bool = False

    @model_validator(mode="after")
    def validate_analysis_graph_and_evidence(self) -> "ProblemAnalysis":
        evidence = self.all_evidence()
        evidence_ids = [item.evidence_id for item in evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("evidence ids must be unique")
        if any(item.type is not EvidenceType.FACT for item in self.facts):
            raise ValueError("facts must contain FACT evidence")
        if any(item.type is not EvidenceType.DATA for item in self.data_items):
            raise ValueError("data_items must contain DATA evidence")
        if any(item.type is not EvidenceType.DERIVATION for item in self.derivations):
            raise ValueError("derivations must contain DERIVATION evidence")
        if any(item.type is not EvidenceType.ASSUMPTION for item in self.assumptions_required):
            raise ValueError("assumptions_required must contain ASSUMPTION evidence")

        subproblem_ids = {item.subproblem_id for item in self.subproblems}
        if len(subproblem_ids) != len(self.subproblems):
            raise ValueError("subproblem ids must be unique")
        ambiguity_ids = {item.ambiguity_id for item in self.ambiguities}
        graph: dict[str, set[str]] = {item_id: set() for item_id in subproblem_ids}
        indegree = dict.fromkeys(subproblem_ids, 0)
        for dependency in self.dependencies_between_subproblems:
            if dependency.upstream_id not in subproblem_ids:
                raise ValueError("dependency upstream must reference a subproblem")
            if dependency.downstream_id not in subproblem_ids:
                raise ValueError("dependency downstream must reference a subproblem")
            if dependency.downstream_id not in graph[dependency.upstream_id]:
                graph[dependency.upstream_id].add(dependency.downstream_id)
                indegree[dependency.downstream_id] += 1
        for subproblem in self.subproblems:
            linked = {*subproblem.input_dependencies, *subproblem.output_dependencies}
            if not linked <= subproblem_ids:
                raise ValueError("subproblem dependency lists contain an unknown id")
            if not set(subproblem.ambiguity_refs) <= ambiguity_ids:
                raise ValueError("subproblem ambiguity refs contain an unknown id")

        ready = [node for node, degree in indegree.items() if degree == 0]
        visited = 0
        while ready:
            node = ready.pop()
            visited += 1
            for downstream in graph[node]:
                indegree[downstream] -= 1
                if indegree[downstream] == 0:
                    ready.append(downstream)
        if visited != len(subproblem_ids):
            raise ValueError("subproblem dependency graph must be acyclic")

        evidence_ref_groups = [
            *(item.evidence_refs for item in self.background),
            self.core_problem.evidence_refs,
            *(item.evidence_refs for item in self.objectives),
            *(item.evidence_refs for item in self.explicit_constraints),
            *(item.evidence_refs for item in self.implicit_conditions),
            *(item.evidence_refs for item in self.entities),
            *(item.evidence_refs for item in self.resources),
            *(
                requirement.evidence_refs
                for subproblem in self.subproblems
                for requirement in subproblem.data_requirements
            ),
        ]
        if self.time_scope is not None:
            evidence_ref_groups.append(self.time_scope.evidence_refs)
        if self.spatial_scope is not None:
            evidence_ref_groups.append(self.spatial_scope.evidence_refs)
        referenced_evidence = {ref for group in evidence_ref_groups for ref in group}
        if not referenced_evidence <= set(evidence_ids):
            raise ValueError("analysis contains an unknown evidence reference")

        affected_refs = {ref for risk in self.risks for ref in risk.affected_subproblems} | {
            ref for missing in self.missing_information for ref in missing.affected_subproblems
        }
        if not affected_refs <= subproblem_ids:
            raise ValueError("risk or missing-information entry references an unknown subproblem")
        return self

    def all_evidence(self) -> list[EvidenceItem]:
        return [
            *self.facts,
            *self.data_items,
            *self.derivations,
            *self.assumptions_required,
        ]


class ProblemAgentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1)
    raw_problem: str = Field(min_length=20)
    competition_context: str | None = None
    optional_user_notes: list[str] = Field(default_factory=list)
