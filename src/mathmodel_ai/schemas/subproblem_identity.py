from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SubproblemResolutionMethod(StrEnum):
    CANONICAL_ID = "CANONICAL_ID"
    REVIEWED_ALIAS = "REVIEWED_ALIAS"


class SubproblemIdentityBinding(BaseModel):
    """Reviewed names for one official task; aliases are exact, never fuzzy."""

    model_config = ConfigDict(extra="forbid")

    canonical_subproblem_id: str = Field(pattern=r"^Q[A-Za-z0-9_-]+$")
    display_label: str = Field(min_length=1)
    source_binding: str = Field(min_length=1)
    internal_node_id: str = Field(min_length=1)
    source_order: int = Field(ge=1)
    canonical_goal: str = Field(min_length=1)
    reviewed_aliases: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def aliases_are_distinct(self) -> SubproblemIdentityBinding:
        if self.canonical_subproblem_id in self.reviewed_aliases:
            raise ValueError("canonical subproblem id cannot also be an alias")
        if len(self.reviewed_aliases) != len(set(self.reviewed_aliases)):
            raise ValueError("reviewed aliases must be unique")
        if any(not value.startswith("Q") for value in self.reviewed_aliases):
            raise ValueError("reviewed aliases must use the subproblem id namespace")
        return self


class ReviewedAmbiguityBinding(BaseModel):
    """Exact alias and reviewed interpretation for one known problem ambiguity."""

    model_config = ConfigDict(extra="forbid")

    canonical_ambiguity_id: str = Field(pattern=r"^AMB-[A-Za-z0-9_-]+$")
    display_label: str = Field(min_length=1)
    source_binding: str = Field(min_length=1)
    canonical_subproblem_ids: list[str] = Field(min_length=1)
    reviewed_aliases: list[str] = Field(default_factory=list)
    canonical_interpretation_id: str = Field(min_length=1)
    reviewed_interpretation_aliases: list[str] = Field(default_factory=list)
    resolution_reason: str = Field(min_length=1)
    confidence: float = Field(ge=0.7, le=1)

    @model_validator(mode="after")
    def reviewed_names_are_distinct(self) -> ReviewedAmbiguityBinding:
        if self.canonical_ambiguity_id in self.reviewed_aliases:
            raise ValueError("canonical ambiguity id cannot also be an alias")
        if len(self.reviewed_aliases) != len(set(self.reviewed_aliases)):
            raise ValueError("reviewed ambiguity aliases must be unique")
        if self.canonical_interpretation_id in self.reviewed_interpretation_aliases:
            raise ValueError("canonical interpretation id cannot also be an alias")
        if len(self.reviewed_interpretation_aliases) != len(
            set(self.reviewed_interpretation_aliases)
        ):
            raise ValueError("reviewed interpretation aliases must be unique")
        return self


class SubproblemIdentityContract(BaseModel):
    """Versioned official-task identity metadata bound to an immutable problem file."""

    model_config = ConfigDict(extra="forbid")

    contract_version: str = Field(min_length=1)
    benchmark_id: str = Field(min_length=1)
    problem_namespace: str = Field(min_length=1)
    problem_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bindings: list[SubproblemIdentityBinding] = Field(min_length=1)
    ambiguity_bindings: list[ReviewedAmbiguityBinding] = Field(default_factory=list)
    content_digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def identity_namespaces_are_unambiguous(self) -> SubproblemIdentityContract:
        canonical = [item.canonical_subproblem_id for item in self.bindings]
        orders = [item.source_order for item in self.bindings]
        nodes = [item.internal_node_id for item in self.bindings]
        aliases = [alias for item in self.bindings for alias in item.reviewed_aliases]
        if len(canonical) != len(set(canonical)):
            raise ValueError("canonical subproblem ids must be unique")
        if len(orders) != len(set(orders)):
            raise ValueError("source orders must be unique")
        if sorted(orders) != list(range(1, len(orders) + 1)):
            raise ValueError("source orders must be contiguous and one-based")
        if len(nodes) != len(set(nodes)):
            raise ValueError("internal graph node ids must be unique")
        if len(aliases) != len(set(aliases)) or set(aliases) & set(canonical):
            raise ValueError("reviewed aliases must be globally unique and non-canonical")
        known_subproblems = set(canonical)
        ambiguity_ids = [item.canonical_ambiguity_id for item in self.ambiguity_bindings]
        ambiguity_aliases = [
            alias for item in self.ambiguity_bindings for alias in item.reviewed_aliases
        ]
        if len(ambiguity_ids) != len(set(ambiguity_ids)):
            raise ValueError("canonical ambiguity ids must be unique")
        if len(ambiguity_aliases) != len(set(ambiguity_aliases)) or set(ambiguity_aliases) & set(
            ambiguity_ids
        ):
            raise ValueError("reviewed ambiguity aliases must be globally unique and non-canonical")
        if any(
            not set(item.canonical_subproblem_ids) <= known_subproblems
            for item in self.ambiguity_bindings
        ):
            raise ValueError("ambiguity binding references an unknown canonical subproblem")
        return self

    @property
    def canonical_ids(self) -> list[str]:
        return [
            item.canonical_subproblem_id
            for item in sorted(self.bindings, key=lambda item: item.source_order)
        ]


class ResolvedSubproblemIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_subproblem_id: str = Field(min_length=1)
    canonical_subproblem_id: str = Field(pattern=r"^Q[A-Za-z0-9_-]+$")
    internal_node_id: str = Field(min_length=1)
    method: SubproblemResolutionMethod


class ResolvedAmbiguityIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_ambiguity_id: str = Field(min_length=1)
    canonical_ambiguity_id: str = Field(pattern=r"^AMB-[A-Za-z0-9_-]+$")
    source_interpretation_id: str = Field(min_length=1)
    canonical_interpretation_id: str = Field(min_length=1)
    method: SubproblemResolutionMethod


class SubproblemIdentityResolution(BaseModel):
    """Persisted audit record for the exact identity rewrite applied to an analysis."""

    model_config = ConfigDict(extra="forbid")

    contract_version: str = Field(min_length=1)
    contract_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    benchmark_id: str = Field(min_length=1)
    problem_namespace: str = Field(min_length=1)
    problem_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bindings: list[ResolvedSubproblemIdentity] = Field(min_length=1)
    ambiguity_bindings: list[ResolvedAmbiguityIdentity] = Field(default_factory=list)

    @model_validator(mode="after")
    def resolved_identity_is_one_to_one(self) -> SubproblemIdentityResolution:
        sources = [item.source_subproblem_id for item in self.bindings]
        canonical = [item.canonical_subproblem_id for item in self.bindings]
        nodes = [item.internal_node_id for item in self.bindings]
        if len(sources) != len(set(sources)):
            raise ValueError("source subproblem identities must be unique")
        if len(canonical) != len(set(canonical)):
            raise ValueError("canonical subproblem identities must be unique")
        if len(nodes) != len(set(nodes)):
            raise ValueError("resolved internal graph nodes must be unique")
        ambiguity_ids = [item.canonical_ambiguity_id for item in self.ambiguity_bindings]
        if len(ambiguity_ids) != len(set(ambiguity_ids)):
            raise ValueError("resolved canonical ambiguity identities must be unique")
        return self
