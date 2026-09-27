from __future__ import annotations

import re
from collections import defaultdict
from typing import ClassVar

from mathmodel_ai.mathematical.registry import EquationRegistry, SymbolRegistry
from mathmodel_ai.paper.claims import (
    ClaimEvidenceGraph,
    ClaimEvidenceValidator,
    NumberConsistencyValidator,
)
from mathmodel_ai.paper.hashing import sha256_json
from mathmodel_ai.paper.integrity import CitationFreshnessValidator, DocumentCompletenessValidator
from mathmodel_ai.paper.literature import reference_digest
from mathmodel_ai.paper.registry import CitationRegistry, FigureRegistry, TableRegistry
from mathmodel_ai.schemas.mathematical import MathematicalModel
from mathmodel_ai.schemas.paper import (
    CitationMetadataCheck,
    CitationSupportCheck,
    CitationSupportStatus,
    Claim,
    DocumentObjectType,
    PaperBlock,
    PaperBlockType,
    PaperIR,
    PaperQualityReport,
    PaperQualityStatus,
    PaperValidationIssue,
    PaperValidationSeverity,
    ReferenceMetadataStatus,
)

_NUMBER = re.compile(r"(?<![A-Za-z0-9_])-?\d+(?:\.\d+)?%?")


def _issue(
    code: str,
    message: str,
    object_ref: str | None = None,
    severity: PaperValidationSeverity = PaperValidationSeverity.ERROR,
) -> PaperValidationIssue:
    return PaperValidationIssue(
        code=code,
        message=message,
        severity=severity,
        object_ref=object_ref,
    )


class CrossReferenceValidator:
    def validate(
        self,
        paper: PaperIR,
        *,
        equations: EquationRegistry,
        figures: FigureRegistry,
        tables: TableRegistry,
        citations: CitationRegistry,
    ) -> list[PaperValidationIssue]:
        issues: list[PaperValidationIssue] = []
        referenced_figures: set[str] = set()
        referenced_tables: set[str] = set()
        referenced_equations: set[str] = set()
        referenced_citations: set[str] = set()
        document_entries = {item.object_id: item for item in paper.document_registry}
        if len(document_entries) != len(paper.document_registry):
            issues.append(_issue("DOCUMENT_INTEGRITY_ERROR", "duplicate document registry ID"))
        sections = [*paper.sections, *paper.appendices]
        for section in sections:
            issues.extend(
                self._validate_reference_sets(
                    section.equation_refs,
                    section.figure_refs,
                    section.table_refs,
                    section.citation_refs,
                    section.section_id,
                    equations,
                    figures,
                    tables,
                    citations,
                )
            )
            referenced_equations.update(section.equation_refs)
            referenced_figures.update(section.figure_refs)
            referenced_tables.update(section.table_refs)
            referenced_citations.update(section.citation_refs)
            for block in section.blocks:
                issues.extend(self._validate_block(block, equations, figures, tables, citations))
                if block.equation_ref:
                    referenced_equations.add(block.equation_ref)
                if block.figure_ref:
                    referenced_figures.add(block.figure_ref)
                if block.table_ref:
                    referenced_tables.add(block.table_ref)
                referenced_citations.update(block.citation_refs)
        for block in paper.abstract:
            issues.extend(self._validate_block(block, equations, figures, tables, citations))
            if block.equation_ref:
                referenced_equations.add(block.equation_ref)
            if block.figure_ref:
                referenced_figures.add(block.figure_ref)
            if block.table_ref:
                referenced_tables.add(block.table_ref)
            referenced_citations.update(block.citation_refs)
        referenced_citations.update(
            reference for claim in paper.claims for reference in claim.citation_refs
        )
        bibliography = set(paper.bibliography)
        for reference in sorted(referenced_citations - bibliography):
            issues.append(
                _issue(
                    "MISSING_REFERENCE",
                    "cited reference is absent from PaperIR bibliography",
                    reference,
                )
            )
        for reference in sorted(bibliography - referenced_citations):
            issues.append(
                _issue(
                    "DOCUMENT_INTEGRITY_ERROR",
                    "bibliography entry is never cited",
                    reference,
                    PaperValidationSeverity.WARNING,
                )
            )
        for reference in sorted(bibliography - citations.identifiers()):
            issues.append(_issue("MISSING_REFERENCE", "unknown bibliography entry", reference))
        for entry in paper.document_registry:
            if entry.object_type is DocumentObjectType.SECTION and not any(
                item.section_id == entry.object_id for item in sections
            ):
                issues.append(
                    _issue(
                        "DOCUMENT_INTEGRITY_ERROR", "orphan section registry entry", entry.object_id
                    )
                )
        for section in sections:
            if section.section_id not in document_entries:
                issues.append(
                    _issue(
                        "DOCUMENT_INTEGRITY_ERROR",
                        "section is absent from document registry",
                        section.section_id,
                    )
                )
        expected_objects: dict[str, tuple[DocumentObjectType, object]] = {
            section.section_id: (DocumentObjectType.SECTION, section.model_dump(mode="json"))
            for section in sections
        }
        expected_objects.update(
            {
                claim.claim_id: (DocumentObjectType.CLAIM, claim.model_dump(mode="json"))
                for claim in paper.claims
            }
        )
        expected_objects.update(
            {
                item.figure_id: (DocumentObjectType.FIGURE, item.model_dump(mode="json"))
                for item in figures.records()
            }
        )
        expected_objects.update(
            {
                item.table_id: (DocumentObjectType.TABLE, item.model_dump(mode="json"))
                for item in tables.records()
            }
        )
        expected_objects.update(
            {
                item.reference_id: (DocumentObjectType.REFERENCE, item.model_dump(mode="json"))
                for item in citations.records()
            }
        )
        for equation_id in referenced_equations:
            equation = equations.lookup(equation_id)
            if equation is not None:
                expected_objects[equation_id] = (
                    DocumentObjectType.EQUATION,
                    equation.model_dump(mode="json"),
                )
        for object_id, (object_type, content) in expected_objects.items():
            resolved_entry = document_entries.get(object_id)
            if resolved_entry is None:
                issues.append(
                    _issue(
                        "DOCUMENT_INTEGRITY_ERROR",
                        f"{object_type.value.casefold()} is absent from document registry",
                        object_id,
                    )
                )
                continue
            expected_hash = sha256_json({"content": content})
            if (
                resolved_entry.object_type is not object_type
                or resolved_entry.content_hash != expected_hash
            ):
                issues.append(
                    _issue(
                        "DOCUMENT_INTEGRITY_ERROR",
                        "document registry type or content hash does not match source object",
                        object_id,
                    )
                )
        for object_id in document_entries.keys() - expected_objects.keys():
            entry = document_entries[object_id]
            if entry.object_type is not DocumentObjectType.EQUATION:
                issues.append(
                    _issue(
                        "DOCUMENT_INTEGRITY_ERROR",
                        "orphan document registry entry",
                        object_id,
                    )
                )
        unreferenced_figures = figures.identifiers() - referenced_figures
        unreferenced_tables = tables.identifiers() - referenced_tables
        orphan_severity = (
            PaperValidationSeverity.ERROR
            if paper.competition_profile.fail_on_unreferenced_assets
            else PaperValidationSeverity.WARNING
        )
        for item in sorted(unreferenced_figures):
            issues.append(
                _issue(
                    "UNREFERENCED_FIGURE",
                    "registered figure is not referenced",
                    item,
                    orphan_severity,
                )
            )
        for item in sorted(unreferenced_tables):
            issues.append(
                _issue(
                    "UNREFERENCED_TABLE",
                    "registered table is not referenced",
                    item,
                    orphan_severity,
                )
            )
        return issues

    def _validate_block(
        self,
        block: PaperBlock,
        equations: EquationRegistry,
        figures: FigureRegistry,
        tables: TableRegistry,
        citations: CitationRegistry,
    ) -> list[PaperValidationIssue]:
        return self._validate_reference_sets(
            [block.equation_ref] if block.equation_ref else [],
            [block.figure_ref] if block.figure_ref else [],
            [block.table_ref] if block.table_ref else [],
            block.citation_refs,
            block.block_id,
            equations,
            figures,
            tables,
            citations,
        )

    @staticmethod
    def _validate_reference_sets(
        equation_refs: list[str],
        figure_refs: list[str],
        table_refs: list[str],
        citation_refs: list[str],
        owner: str,
        equations: EquationRegistry,
        figures: FigureRegistry,
        tables: TableRegistry,
        citations: CitationRegistry,
    ) -> list[PaperValidationIssue]:
        issues: list[PaperValidationIssue] = []
        for ref in equation_refs:
            if equations.lookup(ref) is None:
                issues.append(_issue("MISSING_EQUATION", f"unknown equation {ref}", owner))
        for ref in figure_refs:
            if ref not in figures.identifiers():
                issues.append(_issue("MISSING_FIGURE", f"unknown figure {ref}", owner))
        for ref in table_refs:
            if ref not in tables.identifiers():
                issues.append(_issue("MISSING_TABLE", f"unknown table {ref}", owner))
        for ref in citation_refs:
            if ref not in citations.identifiers():
                issues.append(_issue("MISSING_REFERENCE", f"unknown reference {ref}", owner))
        return issues


class SymbolConsistencyValidator:
    def validate(
        self,
        model: MathematicalModel,
        declared_symbols: dict[str, tuple[str, str | None]] | None = None,
    ) -> list[PaperValidationIssue]:
        registry = SymbolRegistry.from_model(model)
        issues = [
            _issue("SYMBOL_CONFLICT", item.message, item.reference)
            for item in registry.report.issues
            if item.critical
        ]
        expected = {
            item.symbol: (item.meaning, item.unit.display if item.unit else None)
            for item in registry.definitions
        }
        for symbol, definition in (declared_symbols or {}).items():
            if symbol not in expected or expected[symbol] != definition:
                issues.append(
                    _issue(
                        "SYMBOL_CONFLICT",
                        f"paper symbol definition for {symbol!r} differs from the verified model",
                        symbol,
                    )
                )
        return issues


class SymbolNarrativeValidator:
    _DEFINITION = re.compile(
        r"\b([A-Za-z][A-Za-z0-9_]*)\s+(?:denotes|represents|means)\s+([^.;]+)",
        re.IGNORECASE,
    )

    def validate(self, paper: PaperIR, model: MathematicalModel) -> list[PaperValidationIssue]:
        expected = {
            item.symbol: item.meaning for item in SymbolRegistry.from_model(model).definitions
        }
        return self.validate_meanings(paper, expected)

    def validate_meanings(
        self, paper: PaperIR, definitions: dict[str, str]
    ) -> list[PaperValidationIssue]:
        expected = {
            symbol: " ".join(meaning.casefold().split()) for symbol, meaning in definitions.items()
        }
        issues: list[PaperValidationIssue] = []
        texts = [
            block.text
            for block in [
                *paper.abstract,
                *(
                    block
                    for section in [*paper.sections, *paper.appendices]
                    for block in section.blocks
                ),
            ]
            if block.text
        ]
        texts.extend(claim.text for claim in paper.claims)
        for text in texts:
            for match in self._DEFINITION.finditer(text):
                symbol = match.group(1)
                meaning = " ".join(match.group(2).casefold().split())
                canonical = expected.get(symbol)
                if canonical is not None and canonical not in meaning and meaning not in canonical:
                    issues.append(
                        _issue(
                            "SYMBOL_CONFLICT",
                            f"paper narrative redefines {symbol!r} inconsistently",
                            symbol,
                        )
                    )
        return issues


class HallucinatedNumberDetector:
    """Flag prose numbers that are not represented by a structured Claim block."""

    def validate(self, paper: PaperIR, claims: list[Claim]) -> list[PaperValidationIssue]:
        del claims
        issues: list[PaperValidationIssue] = []
        blocks = [*paper.abstract]
        blocks.extend(
            block for section in [*paper.sections, *paper.appendices] for block in section.blocks
        )
        for block in blocks:
            if block.block_type is PaperBlockType.CLAIM or block.text is None:
                continue
            for match in _NUMBER.finditer(block.text):
                token = match.group(0)
                context = block.text[max(0, match.start() - 24) : match.end() + 2]
                raw_number = token.removesuffix("%")
                if (
                    (raw_number.isdigit() and 1900 <= int(raw_number) <= 2100)
                    or re.search(
                        r"(?:equation|eq\.?|table|figure|fig\.?)\s*\(?\s*"
                        + re.escape(raw_number)
                        + r"\s*\)?",
                        context,
                        re.IGNORECASE,
                    )
                    or re.search(
                        r"(?:Q|SEC-|EQ-|FIG-|TAB-|REF-|CLAIM-)" + re.escape(raw_number), context
                    )
                ):
                    continue
                issues.append(
                    _issue(
                        "UNSUPPORTED_CLAIM",
                        f"prose number {token!r} has no structured claim/evidence binding",
                        block.block_id,
                    )
                )
        return issues


class UnsupportedFactualLanguageValidator:
    _unsupported = re.compile(
        r"\b(?:best|superior|significantly better|significantly improved|highly robust|"
        r"excellent performance)\b|\bdata (?:were|was|are|is) (?:collected|obtained) from\b",
        re.IGNORECASE,
    )

    def validate(self, paper: PaperIR) -> list[PaperValidationIssue]:
        issues: list[PaperValidationIssue] = []
        for block in [
            *paper.abstract,
            *(
                block
                for section in [*paper.sections, *paper.appendices]
                for block in section.blocks
            ),
        ]:
            if block.block_type is PaperBlockType.CLAIM or not block.text:
                continue
            if self._unsupported.search(block.text):
                issues.append(
                    _issue(
                        "UNSUPPORTED_CLAIM",
                        "material factual/comparative language lacks a ClaimRecord binding",
                        block.block_id,
                    )
                )
        return issues


class FactualIntegrityValidator:
    _solver_terms: ClassVar[dict[str, str]] = {
        "gurobi": "GUROBI",
        "scipy": "SCIPY",
        "or-tools": "ORTOOLS",
        "ortools": "ORTOOLS",
    }
    _optimal_terms = ("optimal solution", "globally optimal", "获得最优解", "全局最优")
    _exact_terms = ("global optimum", "exact optimum")
    _model_terms: ClassVar[dict[str, str]] = {
        "mixed integer": "MILP",
        "milp": "MILP",
        "linear programming": "LP",
        "linear program": "LP",
        "nonlinear programming": "NLP",
        "nonlinear program": "NLP",
    }

    def validate(self, graph: ClaimEvidenceGraph) -> list[PaperValidationIssue]:
        issues: list[PaperValidationIssue] = []
        for claim in graph.claims:
            linked = graph.evidence_for(claim.claim_id)
            result_payloads = [
                evidence.structured_payload
                for _, evidence in linked
                if evidence.evidence_type.value == "RESULT"
            ]
            if not result_payloads:
                continue
            payload = result_payloads[0]
            text = claim.text.casefold()
            for term, solver in self._solver_terms.items():
                if term in text and payload.get("solver") != solver:
                    issues.append(
                        _issue(
                            "SOLVER_CLAIM_MISMATCH",
                            f"paper names {solver}, evidence names {payload.get('solver')}",
                            claim.claim_id,
                        )
                    )
            if any(term in text for term in self._optimal_terms) and not (
                payload.get("status") == "OPTIMAL" and payload.get("is_optimal") is True
            ):
                issues.append(
                    _issue(
                        "OPTIMALITY_CLAIM_MISMATCH",
                        "paper claims optimality without an OPTIMAL verified result",
                        claim.claim_id,
                    )
                )
            if any(term in text for term in self._exact_terms) and not (
                payload.get("status") == "OPTIMAL" and payload.get("is_optimal") is True
            ):
                issues.append(
                    _issue(
                        "OPTIMALITY_CLAIM_MISMATCH",
                        "paper claims exact/global optimum without verified optimal status",
                        claim.claim_id,
                    )
                )
            version_match = re.search(r"(?:gurobi|scipy|or-?tools)\s+v?([0-9]+(?:\.[0-9]+)*)", text)
            if version_match and version_match.group(1) != payload.get("solver_version"):
                issues.append(
                    _issue(
                        "SOLVER_VERSION_CLAIM_MISMATCH",
                        "paper solver version differs from execution evidence",
                        claim.claim_id,
                    )
                )
            mentioned_families = {
                family for term, family in self._model_terms.items() if term in text
            }
            if mentioned_families and mentioned_families != {payload.get("model_family")}:
                issues.append(
                    _issue(
                        "MODEL_FAMILY_CLAIM_MISMATCH",
                        "paper model-family description differs from verified model",
                        claim.claim_id,
                    )
                )
        for claim in graph.claims:
            linked = graph.evidence_for(claim.claim_id)
            text = claim.text.casefold()
            sensitivity_payloads = [
                item.structured_payload
                for _, item in linked
                if item.evidence_type.value == "SENSITIVITY"
            ]
            if sensitivity_payloads and any(
                term in text for term in ("insensitive", "not sensitive")
            ):
                change = sensitivity_payloads[0].get("maximum_absolute_relative_change")
                if not isinstance(change, int | float) or change > 0.10:
                    issues.append(
                        _issue(
                            "SENSITIVITY_CLAIM_MISMATCH",
                            "insensitivity claim conflicts with measured response",
                            claim.claim_id,
                        )
                    )
            robustness_payloads = [
                item.structured_payload
                for _, item in linked
                if item.evidence_type.value == "ROBUSTNESS"
            ]
            if robustness_payloads and any(
                term in text for term in ("strong robustness", "highly robust")
            ):
                summary = robustness_payloads[0].get("summary", {})
                rate = summary.get("feasibility_rate") if isinstance(summary, dict) else None
                failed = summary.get("failed_runs") if isinstance(summary, dict) else None
                if not isinstance(rate, int | float) or rate < 0.95 or failed != 0:
                    issues.append(
                        _issue(
                            "ROBUSTNESS_CLAIM_MISMATCH",
                            "strong-robustness claim conflicts with stress evidence",
                            claim.claim_id,
                        )
                    )
            if re.search(r"\b(?:best|superior|excellent performance|significantly)\b", text):
                has_support = any(
                    isinstance(item.structured_payload.get("p_value"), int | float)
                    or item.structured_payload.get("comparative_test") is True
                    for _, item in linked
                )
                if not has_support:
                    issues.append(
                        _issue(
                            "UNSUPPORTED_CLAIM",
                            "superlative/statistical-significance language lacks test evidence",
                            claim.claim_id,
                        )
                    )
            if "who" in text and not any(
                "who" in str(item.model_dump(mode="json")).casefold() for _, item in linked
            ):
                issues.append(
                    _issue(
                        "UNSUPPORTED_CLAIM",
                        "named data source is absent from linked evidence",
                        claim.claim_id,
                    )
                )
        red_team_records = {
            item.evidence_id: item
            for claim in graph.claims
            for _, item in graph.evidence_for(claim.claim_id)
            if item.evidence_type.value == "RED_TEAM"
        }
        for record in red_team_records.values():
            findings = record.structured_payload.get("findings", [])
            for finding in findings if isinstance(findings, list) else []:
                if not isinstance(finding, dict) or finding.get("resolved") is True:
                    continue
                finding_id = finding.get("finding_id")
                severity = finding.get("severity")
                if severity == "CRITICAL":
                    issues.append(
                        _issue(
                            "UNRESOLVED_RED_TEAM_CRITICAL",
                            "unresolved critical red-team finding blocks final paper",
                            str(finding_id),
                        )
                    )
                elif severity == "MAJOR" and not any(
                    record.evidence_id in claim.evidence_refs
                    and isinstance(claim.structured_value, dict)
                    and finding_id in claim.structured_value.get("finding_ids", [])
                    for claim in graph.claims
                ):
                    issues.append(
                        _issue(
                            "UNRESOLVED_RED_TEAM_MAJOR",
                            "unresolved major finding is absent from limitations",
                            str(finding_id),
                        )
                    )
        return issues


class CitationQualityValidator:
    def validate(
        self,
        claims: list[Claim],
        metadata_checks: list[CitationMetadataCheck],
        support_checks: list[CitationSupportCheck],
    ) -> list[PaperValidationIssue]:
        metadata = {item.reference_id: item for item in metadata_checks}
        support = {(item.claim_id, item.reference_id): item for item in support_checks}
        issues: list[PaperValidationIssue] = []
        for claim in claims:
            for reference_id in claim.citation_refs:
                metadata_check = metadata.get(reference_id)
                if (
                    metadata_check is None
                    or metadata_check.status is not ReferenceMetadataStatus.VERIFIED
                ):
                    issues.append(
                        _issue(
                            "REFERENCE_NOT_FOUND", "citation metadata is not verified", reference_id
                        )
                    )
                    continue
                support_check = support.get((claim.claim_id, reference_id))
                if (
                    support_check is None
                    or support_check.status is not CitationSupportStatus.SUPPORTED
                ):
                    issues.append(
                        _issue(
                            "REFERENCE_NOT_SUPPORTED",
                            "verified reference does not directly support this claim",
                            claim.claim_id,
                        )
                    )
                elif support_check.reviewer_is_mock:
                    issues.append(
                        _issue("MOCK_EVIDENCE", "Mock citation review cannot pass", claim.claim_id)
                    )
        return issues


class PaperQualityGate:
    def evaluate(
        self,
        *,
        paper: PaperIR,
        graph: ClaimEvidenceGraph,
        model: MathematicalModel,
        equations: EquationRegistry,
        figures: FigureRegistry,
        tables: TableRegistry,
        citations: CitationRegistry,
        metadata_checks: list[CitationMetadataCheck],
        support_checks: list[CitationSupportCheck],
        compile_succeeded: bool,
        artifact_integrity_issues: list[PaperValidationIssue] | None = None,
        declared_symbols: dict[str, tuple[str, str | None]] | None = None,
        independent_review_passed: bool = False,
        independent_review_is_mock: bool = True,
    ) -> PaperQualityReport:
        issues = ClaimEvidenceValidator().validate(graph)
        issues.extend(NumberConsistencyValidator().validate(graph.claims))
        issues.extend(FactualIntegrityValidator().validate(graph))
        issues.extend(
            CitationQualityValidator().validate(graph.claims, metadata_checks, support_checks)
        )
        issues.extend(
            CitationFreshnessValidator().validate(
                graph.claims, citations.records(), metadata_checks, support_checks
            )
        )
        issues.extend(
            CrossReferenceValidator().validate(
                paper, equations=equations, figures=figures, tables=tables, citations=citations
            )
        )
        issues.extend(SymbolConsistencyValidator().validate(model, declared_symbols))
        issues.extend(SymbolNarrativeValidator().validate(paper, model))
        issues.extend(HallucinatedNumberDetector().validate(paper, graph.claims))
        issues.extend(UnsupportedFactualLanguageValidator().validate(paper))
        issues.extend(DocumentCompletenessValidator().validate(paper))
        issues.extend(artifact_integrity_issues or [])
        if not compile_succeeded:
            issues.append(_issue("PDF_COMPILE_ERROR", "a real safe PDF compile did not succeed"))
        if not independent_review_passed or independent_review_is_mock:
            issues.append(
                _issue(
                    "INDEPENDENT_REVIEW_REQUIRED",
                    "final-ready paper requires a non-Mock independent factual review",
                )
            )
        errors = [item for item in issues if item.severity is PaperValidationSeverity.ERROR]
        status = (
            PaperQualityStatus.READY_FOR_FINAL_JURY
            if not errors
            else PaperQualityStatus.HUMAN_REVIEW
            if any(item.code in {"MOCK_EVIDENCE", "INDEPENDENT_REVIEW_REQUIRED"} for item in errors)
            else PaperQualityStatus.FAILED
        )
        supported = sum(
            not any(issue.object_ref == claim.claim_id for issue in errors)
            for claim in graph.claims
        )
        checks: defaultdict[str, bool] = defaultdict(lambda: True)
        for item in errors:
            checks[item.code] = False
        checks["pdf_compile"] = compile_succeeded
        checks["independent_review"] = independent_review_passed and not independent_review_is_mock
        return PaperQualityReport(
            paper_id=paper.paper_id,
            paper_version=paper.version,
            status=status,
            checks=dict(checks),
            issues=issues,
            claim_count=len(graph.claims),
            supported_claim_count=supported,
            unsupported_claim_count=len(graph.claims) - supported,
            reference_count=len(citations.records()),
            verified_reference_count=sum(
                item.status is ReferenceMetadataStatus.VERIFIED
                and (reference := citations.get(item.reference_id)) is not None
                and item.reference_digest == reference_digest(reference)
                for item in metadata_checks
            ),
            figure_count=len(figures.records()),
            table_count=len(tables.records()),
        )
