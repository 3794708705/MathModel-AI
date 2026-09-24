from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from mathmodel_ai.schemas.paper import (
    Claim,
    ClaimEvidenceLink,
    ClaimEvidenceSupport,
    ClaimImportance,
    ClaimType,
    ClaimVerificationStatus,
    ComparisonClaimValue,
    EvidenceRecord,
    NumericClaimValue,
    PaperValidationIssue,
    PaperValidationSeverity,
)

_TEXT_NUMBER = re.compile(
    r"(?<![A-Za-z0-9_])-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\s*"
    r"(%|kg|kilograms?|g|grams?|t|tonnes?|m|cm|km|s|min|h)?",
    re.IGNORECASE,
)
_UNIT_ALIASES = {
    "kilogram": "kg",
    "kilograms": "kg",
    "gram": "g",
    "grams": "g",
    "tonne": "t",
    "tonnes": "t",
}
_UNIT_SCALE: dict[str, tuple[str, float]] = {
    "g": ("mass", 0.001),
    "kg": ("mass", 1.0),
    "t": ("mass", 1000.0),
    "cm": ("length", 0.01),
    "m": ("length", 1.0),
    "km": ("length", 1000.0),
    "s": ("time", 1.0),
    "min": ("time", 60.0),
    "h": ("time", 3600.0),
    "%": ("percentage", 1.0),
}


def normalize_unit(unit: str | None) -> str | None:
    if unit is None:
        return None
    normalized = unit.strip().casefold()
    return _UNIT_ALIASES.get(normalized, normalized)


def convert_unit(value: float, source: str | None, target: str | None) -> float:
    source_unit = normalize_unit(source)
    target_unit = normalize_unit(target)
    if source_unit == target_unit:
        return value
    if source_unit is None or target_unit is None:
        raise ValueError("unit is missing")
    source_scale = _UNIT_SCALE.get(source_unit)
    target_scale = _UNIT_SCALE.get(target_unit)
    if source_scale is None or target_scale is None or source_scale[0] != target_scale[0]:
        raise ValueError("units are not safely convertible")
    return value * source_scale[1] / target_scale[1]


@dataclass(frozen=True)
class PaperNumericFormattingPolicy:
    """Separate publication rounding from solver feasibility tolerances."""

    max_decimal_places: int = 2

    def matches(
        self, displayed: float, expected: float, decimal_places: int,
        scientific_exponent: int | None = None,
    ) -> bool:
        if scientific_exponent is not None:
            precision = 0.5 * 10 ** (scientific_exponent - decimal_places)
            return math.isclose(displayed, expected, rel_tol=0.0, abs_tol=precision)
        if decimal_places > self.max_decimal_places:
            return math.isclose(displayed, expected, rel_tol=0.0, abs_tol=10**-decimal_places / 2)
        return math.isclose(
            displayed,
            round(expected, decimal_places),
            rel_tol=0.0,
            abs_tol=10 ** -(decimal_places + 4),
        )

    def text_matches(self, claim: Claim) -> bool:
        value = claim.structured_value
        tokens = []
        for match in _TEXT_NUMBER.finditer(claim.text):
            raw_number = match.group(0).strip()
            number_text = re.match(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", raw_number)
            if number_text is None:
                continue
            number = float(number_text.group(0))
            literal = number_text.group(0)
            mantissa, exponent_marker, exponent_text = literal.lower().partition("e")
            decimals = len(mantissa.partition(".")[2])
            exponent = int(exponent_text) if exponent_marker else None
            tokens.append((number, decimals, exponent, match.group(1)))
        if isinstance(value, NumericClaimValue):
            for number, decimals, exponent, unit in tokens:
                try:
                    expected = convert_unit(value.value, value.unit, unit or value.unit)
                except ValueError:
                    continue
                if self.matches(number, expected, decimals, exponent):
                    return True
            return False
        if isinstance(value, ComparisonClaimValue):
            percentage_matches = any(
                unit == "%"
                and self.matches(
                    abs(number), abs(value.percentage_change), decimals, exponent
                )
                for number, decimals, exponent, unit in tokens
            )
            text = claim.text.casefold()
            increasing = any(
                term in text for term in
                ("increase", "increased", "rose", "higher", "exceeds", "above", "greater than")
            )
            decreasing = any(
                term in text for term in
                ("decrease", "decreased", "reduced", "reduction", "lower", "below", "less than")
            )
            direction_matches = (
                (value.direction.value == "INCREASE" and increasing and not decreasing)
                or (value.direction.value == "DECREASE" and decreasing and not increasing)
                or (value.direction.value == "CHANGE" and not increasing and not decreasing)
            )
            return percentage_matches and direction_matches
        return True


def resolve_source_field(payload: dict[str, Any], source_field: str) -> Any:
    """Resolve a dot-separated field without evaluating user-controlled expressions."""

    current: Any = payload
    for part in source_field.split("."):
        if not part or part.startswith("_"):
            raise ValueError("invalid evidence source field")
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            raise KeyError(source_field)
    return current


class ClaimEvidenceGraph:
    """Deterministic, immutable-by-convention claim/evidence association graph."""

    def __init__(self, evidence: list[EvidenceRecord]) -> None:
        self._evidence = {item.evidence_id: item for item in evidence}
        if len(self._evidence) != len(evidence):
            raise ValueError("duplicate evidence identifiers")
        self._claims: dict[str, Claim] = {}
        self._links: dict[UUID, ClaimEvidenceLink] = {}

    @property
    def claims(self) -> list[Claim]:
        return list(self._claims.values())

    @property
    def links(self) -> list[ClaimEvidenceLink]:
        return list(self._links.values())

    def add_claim(self, claim: Claim) -> None:
        if claim.claim_id in self._claims:
            raise ValueError(f"duplicate claim identifier: {claim.claim_id}")
        unknown = set(claim.evidence_refs) - self._evidence.keys()
        if unknown:
            raise ValueError(f"claim references unknown evidence: {sorted(map(str, unknown))}")
        self._claims[claim.claim_id] = claim

    def add_link(self, link: ClaimEvidenceLink) -> None:
        if link.link_id in self._links:
            raise ValueError(f"duplicate claim/evidence link: {link.link_id}")
        claim = self._claims.get(link.claim_id)
        if claim is None:
            raise ValueError(f"unknown claim: {link.claim_id}")
        if link.evidence_id not in self._evidence:
            raise ValueError(f"unknown evidence: {link.evidence_id}")
        if link.evidence_id not in claim.evidence_refs:
            raise ValueError("link evidence must be declared on its claim")
        self._links[link.link_id] = link

    def evidence_for(self, claim_id: str) -> list[tuple[ClaimEvidenceLink, EvidenceRecord]]:
        return [
            (link, self._evidence[link.evidence_id])
            for link in self._links.values()
            if link.claim_id == claim_id
        ]


class NumericClaimValidator:
    def __init__(self, formatting: PaperNumericFormattingPolicy | None = None) -> None:
        self._formatting = formatting or PaperNumericFormattingPolicy()

    def validate(
        self,
        claim: Claim,
        linked_evidence: list[tuple[ClaimEvidenceLink, EvidenceRecord]],
    ) -> list[PaperValidationIssue]:
        value = claim.structured_value
        if not isinstance(value, NumericClaimValue | ComparisonClaimValue):
            return []
        issues: list[PaperValidationIssue] = []
        if not self._formatting.text_matches(claim):
            issues.append(
                self._issue(
                    "NUMERIC_CLAIM_MISMATCH",
                    claim,
                    "rendered claim text does not match its structured numeric value",
                )
            )
        fields = (
            [(value.source_field, value.value, value.unit)]
            if isinstance(value, NumericClaimValue)
            else [
                (value.baseline_source_field, value.baseline_value, value.unit),
                (value.verified_source_field, value.verified_value, value.unit),
            ]
        )
        for source_field, expected, unit in fields:
            matched = False
            for link, evidence in linked_evidence:
                if link.source_field != source_field:
                    continue
                matched = True
                try:
                    actual, actual_unit = self._numeric_value(
                        resolve_source_field(evidence.structured_payload, source_field)
                    )
                except (KeyError, TypeError, ValueError):
                    issues.append(
                        self._issue(
                            "NUMERIC_CLAIM_MISMATCH",
                            claim,
                            f"evidence field {source_field!r} is not a finite number",
                        )
                    )
                    continue
                try:
                    converted_actual = convert_unit(actual, actual_unit, unit)
                except ValueError:
                    issues.append(
                        self._issue(
                            "UNIT_CLAIM_MISMATCH",
                            claim,
                            f"claim unit {unit!r} is not safely convertible from {actual_unit!r}",
                        )
                    )
                    continue
                if not math.isclose(
                    converted_actual,
                    expected,
                    rel_tol=value.tolerance,
                    abs_tol=value.tolerance,
                ):
                    issues.append(
                        self._issue(
                            "NUMERIC_CLAIM_MISMATCH",
                            claim,
                            (
                                f"claim value {expected} differs from verified evidence "
                                f"{converted_actual}"
                            ),
                        )
                    )
                    if normalize_unit(unit) != normalize_unit(actual_unit):
                        issues.append(
                            self._issue(
                                "UNIT_CLAIM_MISMATCH",
                                claim,
                                "unit conversion changes the claimed magnitude",
                            )
                        )
            if not matched:
                issues.append(
                    self._issue(
                        "NUMERIC_CLAIM_MISMATCH",
                        claim,
                        f"no linked evidence maps source field {source_field!r}",
                    )
                )
        return issues

    @staticmethod
    def _numeric_value(raw: Any) -> tuple[float, str | None]:
        if isinstance(raw, bool):
            raise TypeError("boolean is not numeric evidence")
        if isinstance(raw, int | float):
            value = float(raw)
            unit = None
        elif isinstance(raw, dict):
            value = float(raw["value"])
            unit_value = raw.get("unit")
            unit = str(unit_value) if unit_value is not None else None
        else:
            raise TypeError("unsupported numeric evidence payload")
        if not math.isfinite(value):
            raise ValueError("non-finite evidence value")
        return value, unit

    @staticmethod
    def _issue(code: str, claim: Claim, message: str) -> PaperValidationIssue:
        return PaperValidationIssue(
            code=code,
            message=message,
            severity=PaperValidationSeverity.ERROR,
            object_ref=claim.claim_id,
        )


class ClaimEvidenceValidator:
    def __init__(self, numeric_validator: NumericClaimValidator | None = None) -> None:
        self._numeric = numeric_validator or NumericClaimValidator()

    def validate(self, graph: ClaimEvidenceGraph) -> list[PaperValidationIssue]:
        issues: list[PaperValidationIssue] = []
        for claim in graph.claims:
            linked = graph.evidence_for(claim.claim_id)
            direct = [
                evidence
                for link, evidence in linked
                if link.support is ClaimEvidenceSupport.DIRECT_SUPPORT
            ]
            verified_direct = [item for item in direct if item.verified]
            if claim.importance in {ClaimImportance.CRITICAL, ClaimImportance.MAJOR} and not (
                verified_direct
            ):
                issues.append(
                    PaperValidationIssue(
                        code="UNSUPPORTED_CLAIM",
                        message="critical/major claim lacks direct verified evidence",
                        severity=PaperValidationSeverity.ERROR,
                        object_ref=claim.claim_id,
                    )
                )
            if any(item.provenance.is_mock for _, item in linked):
                issues.append(
                    PaperValidationIssue(
                        code="MOCK_EVIDENCE",
                        message="Mock evidence cannot support a paper claim",
                        severity=PaperValidationSeverity.ERROR,
                        object_ref=claim.claim_id,
                    )
                )
            elif any(not item.verified for _, item in linked):
                issues.append(
                    PaperValidationIssue(
                        code="UNVERIFIED_EVIDENCE",
                        message="unverified evidence is linked to a paper claim",
                        severity=PaperValidationSeverity.ERROR,
                        object_ref=claim.claim_id,
                    )
                )
            expected_types = {
                ClaimType.ASSUMPTION: {"ASSUMPTION"},
                ClaimType.FACTUAL: {"PROBLEM_FACT", "DATA_FACT", "LITERATURE"},
                ClaimType.LITERATURE: {"LITERATURE"},
            }.get(claim.claim_type)
            if expected_types is not None and any(
                item.evidence_type.value not in expected_types for item in verified_direct
            ):
                issues.append(
                    PaperValidationIssue(
                        code="PROVENANCE_TYPE_MISMATCH",
                        message="claim type does not match its direct evidence provenance",
                        severity=PaperValidationSeverity.ERROR,
                        object_ref=claim.claim_id,
                    )
                )
            if claim.claim_type is ClaimType.ASSUMPTION and any(
                item.structured_payload.get("supported") is not True for item in verified_direct
            ):
                issues.append(
                    PaperValidationIssue(
                        code="UNSUPPORTED_ASSUMPTION",
                        message="only accepted/supported assumptions may enter the formal paper",
                        severity=PaperValidationSeverity.ERROR,
                        object_ref=claim.claim_id,
                    )
                )
            issues.extend(self._numeric.validate(claim, linked))
        return issues

    def apply_statuses(self, graph: ClaimEvidenceGraph) -> list[Claim]:
        by_claim: defaultdict[str, list[PaperValidationIssue]] = defaultdict(list)
        for issue in self.validate(graph):
            if issue.object_ref is not None:
                by_claim[issue.object_ref].append(issue)
        output: list[Claim] = []
        for claim in graph.claims:
            claim_issues = by_claim[claim.claim_id]
            status = (
                ClaimVerificationStatus.SUPPORTED
                if not claim_issues
                else ClaimVerificationStatus.UNSUPPORTED
            )
            output.append(claim.model_copy(update={"verification_status": status}))
        return output


class NumberConsistencyValidator:
    """Detect conflicting structured claims about the same evidence field."""

    def validate(self, claims: list[Claim]) -> list[PaperValidationIssue]:
        observed: dict[tuple[str, str, tuple[str, ...]], tuple[float, str | None, str]] = {}
        issues: list[PaperValidationIssue] = []
        for claim in claims:
            value = claim.structured_value
            if isinstance(value, NumericClaimValue):
                fields = [
                    (value.metric_name.casefold(), value.source_field, value.value, value.unit)
                ]
            elif isinstance(value, ComparisonClaimValue):
                fields = [
                    (
                        "comparison-baseline",
                        value.baseline_source_field,
                        value.baseline_value,
                        value.unit,
                    ),
                    (
                        "comparison-verified",
                        value.verified_source_field,
                        value.verified_value,
                        value.unit,
                    ),
                ]
            else:
                continue
            for metric_name, source_field, number, unit in fields:
                key = (metric_name, source_field, tuple(sorted(map(str, claim.evidence_refs))))
                previous = observed.get(key)
                if previous is None:
                    observed[key] = (number, unit, claim.claim_id)
                else:
                    try:
                        converted = convert_unit(number, unit, previous[1])
                    except ValueError:
                        converted = math.nan
                if previous is not None and not math.isfinite(converted):
                    issues.append(
                        PaperValidationIssue(
                            code="UNIT_CLAIM_MISMATCH",
                            message=f"claims use conflicting units for {source_field}",
                            severity=PaperValidationSeverity.ERROR,
                            object_ref=claim.claim_id,
                        )
                    )
                elif previous is not None and not math.isclose(
                    previous[0], converted, rel_tol=1e-9, abs_tol=1e-9
                ):
                    issues.append(
                        PaperValidationIssue(
                            code="CROSS_SECTION_VALUE_CONFLICT",
                            message=(
                                f"{claim.claim_id} conflicts with {previous[2]} for {source_field}"
                            ),
                            severity=PaperValidationSeverity.ERROR,
                            object_ref=claim.claim_id,
                        )
                    )
                    if normalize_unit(previous[1]) != normalize_unit(unit):
                        issues.append(
                            PaperValidationIssue(
                                code="UNIT_CLAIM_MISMATCH",
                                message=f"unit conversion changes the value for {source_field}",
                                severity=PaperValidationSeverity.ERROR,
                                object_ref=claim.claim_id,
                            )
                        )
        return issues
