import math
from typing import Any

from pydantic import RootModel, ValidationError

from mathmodel_ai.agents.base import AgentExecution, BaseAgent
from mathmodel_ai.core.errors import ProviderResponseError
from mathmodel_ai.paper.claims import resolve_source_field
from mathmodel_ai.providers.base import BaseModelProvider
from mathmodel_ai.providers.factory import ProviderRegistry
from mathmodel_ai.providers.schemas import GenerationRequest, ModelMessage
from mathmodel_ai.reasoning.prompts import PromptRegistry
from mathmodel_ai.routing.router import ModelRouter
from mathmodel_ai.routing.schemas import RouteDecision
from mathmodel_ai.schemas.paper import (
    ClaimType,
    ComparisonClaimValue,
    NumericClaimValue,
    PaperAgentInput,
    PaperBlockType,
    PaperFactualAuditDraft,
    PaperFactualAuditInput,
    PaperIR,
)
from mathmodel_ai.schemas.problem_state import ProblemState


class _PaperIRPrevalidation(RootModel[dict[str, Any]]):
    """Author schema excludes the system-owned immutable evidence snapshot."""

    @classmethod
    def model_json_schema(cls, *args: Any, **kwargs: Any) -> dict[str, Any]:
        schema = PaperIR.model_json_schema(*args, **kwargs)
        schema["properties"].pop("evidence_snapshot")
        schema["required"] = [field for field in schema["required"] if field != "evidence_snapshot"]
        return schema


class PaperAgent(BaseAgent[PaperAgentInput, PaperIR]):
    name = "paper_agent"
    role = "evidence-grounded PaperIR authoring"
    capabilities = frozenset({"structured_generation", "paper_ir", "claim_authoring"})
    input_schema = PaperAgentInput
    output_schema = PaperIR

    def validate_output(self, output: PaperIR | dict[str, Any]) -> PaperIR:
        try:
            return PaperIR.model_validate(output)
        except ValidationError as exc:
            errors = exc.errors(include_input=False, include_url=False)
            details = ", ".join(
                f"{'.'.join(map(str, error['loc'])) or '<root>'}:{error['type']}"
                for error in errors[:8]
            )
            claim_details = _claim_value_errors(output)
            if claim_details:
                details += "; claim values: " + ", ".join(claim_details[:8])
            raise ProviderResponseError(
                f"PaperIR failed strict schema validation ({exc.error_count()} errors: {details})"
            ) from exc

    def __init__(
        self,
        *,
        router: ModelRouter,
        providers: ProviderRegistry,
        prompts: PromptRegistry,
        max_retries: int = 2,
    ) -> None:
        super().__init__(router=router, providers=providers, max_retries=max_retries)
        self._prompts = prompts

    def prepare_attempt_input(
        self,
        input_data: PaperAgentInput,
        state: ProblemState,
        previous_errors: tuple[str, ...],
    ) -> PaperAgentInput:
        del state
        feedback = list(input_data.repair_feedback)
        if previous_errors:
            feedback = [*feedback[:2], previous_errors[-1]]
        return input_data.model_copy(update={"repair_feedback": feedback})

    async def execute(
        self,
        input_data: PaperAgentInput,
        state: ProblemState,
        provider: BaseModelProvider,
        route: RouteDecision,
    ) -> AgentExecution[PaperIR]:
        del state
        prompt = self._prompts.get("paper_agent")
        response = await provider.structured_generate(
            GenerationRequest(
                model=route.selected_model or "unselected",
                max_output_tokens=65_536,
                reasoning_effort=route.selected_reasoning,
                messages=[
                    ModelMessage(role="system", content=prompt.system),
                    ModelMessage(
                        role="user",
                        content=prompt.render_user(
                            paper_input_json=input_data.model_dump_json(indent=2)
                        ),
                    ),
                ],
                metadata={"agent": self.name, "prompt_version": prompt.version},
            ),
            _PaperIRPrevalidation,
        )
        raw = response.parsed.root
        canonical_snapshot = input_data.evidence_snapshot.model_dump(mode="json")
        if "evidence_snapshot" in raw and raw["evidence_snapshot"] != canonical_snapshot:
            raise ProviderResponseError("PaperAgent supplied a conflicting evidence snapshot")
        raw["evidence_snapshot"] = canonical_snapshot
        _normalize_comparison_arithmetic(raw)
        try:
            paper = PaperIR.model_validate(raw)
        except ValidationError:
            # BaseAgent reports schema failures with field-level feedback.
            paper = None
        if paper is not None:
            preflight_errors = _paper_draft_errors(paper, input_data)
            if preflight_errors:
                raise ProviderResponseError(
                    "PaperIR structural preflight failed: " + "; ".join(preflight_errors[:8])
                )
        return AgentExecution(
            output=raw,
            response=response.response,
            prompt_version=prompt.version,
        )


def _paper_draft_errors(paper: PaperIR, input_data: PaperAgentInput) -> list[str]:
    """Reject deterministic authoring defects before costly asset and PDF work."""
    from mathmodel_ai.paper.claims import PaperNumericFormattingPolicy
    from mathmodel_ai.paper.integrity import (
        DocumentCompletenessValidator,
        SubproblemCoverageValidator,
    )
    from mathmodel_ai.paper.validators import SymbolNarrativeValidator

    errors: list[str] = []
    evidence_by_id = {item.evidence_id: item for item in input_data.evidence}
    for claim in paper.claims:
        unknown = [str(ref) for ref in claim.evidence_refs if ref not in evidence_by_id]
        if unknown:
            errors.append(
                f"EVIDENCE_REF:{claim.claim_id}:unknown linked evidence {', '.join(unknown)}"
            )
            continue
        value = claim.structured_value
        if isinstance(value, NumericClaimValue):
            source_fields = [value.source_field]
        elif isinstance(value, ComparisonClaimValue):
            source_fields = [value.baseline_source_field, value.verified_source_field]
        else:
            continue
        linked = [evidence_by_id[ref] for ref in claim.evidence_refs]
        for field in source_fields:
            if not any(_source_field_resolves(item.structured_payload, field) for item in linked):
                elsewhere = [
                    str(item.evidence_id)
                    for item in input_data.evidence
                    if item.evidence_id not in claim.evidence_refs
                    and item.verified
                    and _source_field_resolves(item.structured_payload, field)
                ]
                location = (
                    f"; exact field exists in other verified evidence {', '.join(elsewhere)}"
                    if elsewhere
                    else "; exact field is absent from all supplied verified evidence"
                )
                errors.append(
                    f"SOURCE_FIELD:{claim.claim_id}:{field} does not resolve in linked evidence"
                    f"{location}; relink only if the evidence supports the same claim"
                )
    if (paper.paper_id, paper.version) != (
        input_data.assigned_paper_id,
        input_data.assigned_version,
    ):
        errors.append("use the assigned paper_id and version")
    for issue in SubproblemCoverageValidator().validate(paper, input_data.required_subproblems):
        errors.append(f"{issue.code}:{issue.object_ref}:{issue.message}")
    for issue in DocumentCompletenessValidator().validate(paper):
        errors.append(f"{issue.code}:{issue.object_ref}:{issue.message}")
    for issue in SymbolNarrativeValidator().validate_meanings(paper, input_data.symbol_definitions):
        errors.append(f"{issue.code}:{issue.object_ref}:use the exact supplied symbol meaning")
    blocks = [
        *paper.abstract,
        *(block for section in [*paper.sections, *paper.appendices] for block in section.blocks),
    ]
    for figure_id in input_data.figure_ids:
        if not any(
            block.block_type is PaperBlockType.FIGURE and block.figure_ref == figure_id
            for block in blocks
        ):
            errors.append(f"FIGURE:{figure_id}:include a visible FIGURE block")
    for table_id in input_data.table_ids:
        if not any(
            block.block_type is PaperBlockType.TABLE and block.table_ref == table_id
            for block in blocks
        ):
            errors.append(f"TABLE:{table_id}:include a visible TABLE block")
    cited = (
        {reference for claim in paper.claims for reference in claim.citation_refs}
        | {
            reference
            for section in [*paper.sections, *paper.appendices]
            for reference in section.citation_refs
        }
        | {reference for block in blocks for reference in block.citation_refs}
    )
    for reference in sorted(set(paper.bibliography) - cited):
        errors.append(f"BIBLIOGRAPHY:{reference}:remove uncited entry")
    for reference in sorted(cited - set(paper.bibliography)):
        errors.append(f"BIBLIOGRAPHY:{reference}:include cited entry")
    formatting = PaperNumericFormattingPolicy()
    for claim in paper.claims:
        if claim.claim_type in {
            ClaimType.NUMERIC,
            ClaimType.COMPARISON,
        } and not formatting.text_matches(claim):
            errors.append(
                f"NUMERIC_CLAIM_MISMATCH:{claim.claim_id}:correct printed value and direction"
            )
    return errors


def _source_field_resolves(payload: dict[str, Any], field: str) -> bool:
    try:
        resolve_source_field(payload, field)
    except (KeyError, ValueError):
        return False
    return True


def _normalize_comparison_arithmetic(raw: dict[str, Any]) -> None:
    """Recalculate only derived fields; never change source values or claim text."""
    claims = raw.get("claims")
    if not isinstance(claims, list):
        return
    for claim in claims:
        if not isinstance(claim, dict) or claim.get("claim_type") != "COMPARISON":
            continue
        value = claim.get("structured_value")
        if not isinstance(value, dict):
            continue
        baseline = value.get("baseline_value")
        verified = value.get("verified_value")
        if (
            isinstance(baseline, bool)
            or isinstance(verified, bool)
            or not isinstance(baseline, int | float)
            or not isinstance(verified, int | float)
            or not math.isfinite(baseline)
            or not math.isfinite(verified)
            or baseline == 0
        ):
            continue
        percentage = (verified - baseline) / abs(baseline) * 100
        value["percentage_change"] = percentage
        tolerance = value.get("tolerance", 1e-9)
        if not isinstance(tolerance, int | float) or isinstance(tolerance, bool):
            continue
        value["direction"] = (
            "INCREASE"
            if percentage > tolerance
            else "DECREASE"
            if percentage < -tolerance
            else "CHANGE"
        )


def _claim_value_errors(output: PaperIR | dict[str, Any]) -> list[str]:
    """Report nested claim-value failures hidden by the permissive evidence union."""
    if not isinstance(output, dict) or not isinstance(output.get("claims"), list):
        return []
    details: list[str] = []
    for index, claim in enumerate(output["claims"]):
        if not isinstance(claim, dict):
            continue
        claim_type = claim.get("claim_type")
        value_type: type[NumericClaimValue] | type[ComparisonClaimValue] | None
        if claim_type == ClaimType.NUMERIC.value:
            value_type = NumericClaimValue
        elif claim_type == ClaimType.COMPARISON.value:
            value_type = ComparisonClaimValue
        else:
            value_type = None
        if value_type is None:
            continue
        try:
            value_type.model_validate(claim.get("structured_value"))
        except ValidationError as exc:
            for error in exc.errors(include_input=False, include_url=False)[:2]:
                loc = ".".join(map(str, error["loc"])) or "<root>"
                details.append(f"claims.{index}.structured_value.{loc}:{error['type']}")
    return details


class PaperFactualAuditAgent(BaseAgent[PaperFactualAuditInput, PaperFactualAuditDraft]):
    name = "paper_factual_audit_agent"
    role = "independent final paper factual audit"
    capabilities = frozenset({"structured_review", "factual_audit"})
    input_schema = PaperFactualAuditInput
    output_schema = PaperFactualAuditDraft

    def __init__(
        self,
        *,
        router: ModelRouter,
        providers: ProviderRegistry,
        prompts: PromptRegistry,
        max_retries: int = 1,
    ) -> None:
        super().__init__(router=router, providers=providers, max_retries=max_retries)
        self._prompts = prompts

    async def execute(
        self,
        input_data: PaperFactualAuditInput,
        state: ProblemState,
        provider: BaseModelProvider,
        route: RouteDecision,
    ) -> AgentExecution[PaperFactualAuditDraft]:
        del state
        prompt = self._prompts.get("paper_factual_audit_agent")
        response = await provider.structured_generate(
            GenerationRequest(
                model=route.selected_model or "unselected",
                reasoning_effort=route.selected_reasoning,
                messages=[
                    ModelMessage(role="system", content=prompt.system),
                    ModelMessage(
                        role="user",
                        content=prompt.render_user(
                            audit_input_json=input_data.model_dump_json(indent=2)
                        ),
                    ),
                ],
                metadata={"agent": self.name, "prompt_version": prompt.version},
            ),
            PaperFactualAuditDraft,
        )
        return AgentExecution(
            output=response.parsed,
            response=response.response,
            prompt_version=prompt.version,
        )
