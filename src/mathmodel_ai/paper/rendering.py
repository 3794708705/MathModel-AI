from __future__ import annotations

import re
from dataclasses import dataclass

from mathmodel_ai.mathematical.registry import EquationRegistry
from mathmodel_ai.paper.hashing import sha256_text
from mathmodel_ai.paper.registry import CitationRegistry, FigureRegistry, TableRegistry
from mathmodel_ai.schemas.paper import (
    Claim,
    PaperBlock,
    PaperBlockType,
    PaperIR,
    ReferenceMetadataStatus,
)

_FORBIDDEN_LATEX = re.compile(
    r"\\(?:input|include|includegraphics|write18|write|openout|read|catcode|immediate|"
    r"usepackage|documentclass)(?![A-Za-z])",
    re.IGNORECASE,
)
_SAFE_ID = re.compile(r"^(?:SEC|EQ|FIG|TAB|REF|CLAIM)-[A-Za-z0-9_-]+$")


class LatexRenderError(ValueError):
    pass


@dataclass(frozen=True)
class RenderedPaper:
    tex: str
    bibliography: str
    tex_hash: str
    bib_hash: str


def escape_latex(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in value)


def _escape_table_cell(value: str) -> str:
    return escape_latex(value).replace(r"\_", r"\_\allowbreak{}")


class LaTeXSafetyValidator:
    def validate_equation(self, latex: str) -> None:
        if _FORBIDDEN_LATEX.search(latex) or "\x00" in latex:
            raise LatexRenderError("unsafe LaTeX command in equation")
        if ".." in latex:
            raise LatexRenderError("path traversal token is forbidden in equation LaTeX")

    def validate_rendered(self, latex: str) -> None:
        if r"\write18" in latex.casefold() or "../" in latex or "..\\" in latex:
            raise LatexRenderError("rendered LaTeX contains an unsafe command or path")


class BibTeXRenderer:
    def render(self, citations: CitationRegistry) -> str:
        entries: list[str] = []
        for record in citations.records():
            if record.metadata_status is not ReferenceMetadataStatus.VERIFIED:
                raise LatexRenderError(
                    f"unverified reference cannot be rendered: {record.reference_id}"
                )
            fields = [
                f"  title = {{{escape_latex(record.title)}}}",
                f"  author = {{{' and '.join(escape_latex(item) for item in record.authors)}}}",
                f"  year = {{{record.year}}}",
                f"  journal = {{{escape_latex(record.venue)}}}",
            ]
            if record.doi:
                fields.append(f"  doi = {{{escape_latex(record.doi)}}}")
            if record.url:
                fields.append(f"  url = {{{escape_latex(record.url)}}}")
            entries.append(f"@article{{{record.reference_id},\n" + ",\n".join(fields) + "\n}")
        return "\n\n".join(entries) + ("\n" if entries else "")


class LaTeXRenderer:
    def __init__(self, safety: LaTeXSafetyValidator | None = None) -> None:
        self._safety = safety or LaTeXSafetyValidator()

    def render(
        self,
        *,
        paper: PaperIR,
        claims: list[Claim],
        equations: EquationRegistry,
        figures: FigureRegistry,
        tables: TableRegistry,
        citations: CitationRegistry,
    ) -> RenderedPaper:
        claim_map = {item.claim_id: item for item in claims}
        if len(claim_map) != len(claims):
            raise LatexRenderError("duplicate claim identifier")
        declared_citations = citations.identifiers()
        requested_citations = set(paper.bibliography)
        requested_citations.update(
            reference
            for section in [*paper.sections, *paper.appendices]
            for reference in section.citation_refs
        )
        requested_citations.update(
            reference
            for block in [
                *paper.abstract,
                *(
                    block
                    for section in [*paper.sections, *paper.appendices]
                    for block in section.blocks
                ),
            ]
            for reference in block.citation_refs
        )
        requested_citations.update(
            reference for claim in claims for reference in claim.citation_refs
        )
        unknown_citations = requested_citations - declared_citations
        if unknown_citations:
            raise LatexRenderError(f"unknown citation keys: {sorted(unknown_citations)}")
        preamble = [
            rf"\documentclass[{paper.competition_profile.paper_size}]{{{paper.competition_profile.document_class}}}",
            r"\usepackage{fontspec}",
            r"\usepackage{xeCJK}",
            r"\setCJKmainfont{Noto Serif CJK SC}",
            r"\usepackage{amsmath,amssymb,booktabs,graphicx,geometry,tabularx,url}",
            rf"\geometry{{margin={paper.competition_profile.margin}}}",
            r"\graphicspath{{figures/}}",
            rf"\title{{{escape_latex(paper.title)}}}",
            r"\author{}" if paper.competition_profile.anonymous else r"\author{MathModel AI}",
            r"\date{}",
            r"\begin{document}",
            r"\maketitle",
            r"\begin{abstract}",
        ]
        body = [
            self._render_block(block, claim_map, equations, figures, tables)
            for block in paper.abstract
        ]
        body.extend(
            [r"\end{abstract}", rf"\textbf{{Keywords:}} {escape_latex(', '.join(paper.keywords))}"]
        )
        for section in sorted([*paper.sections, *paper.appendices], key=lambda item: item.order):
            body.append(rf"\section{{{escape_latex(section.title)}}}\label{{{section.section_id}}}")
            body.extend(
                self._render_block(block, claim_map, equations, figures, tables)
                for block in section.blocks
            )
        if citations.records():
            body.extend(
                [
                    rf"\bibliographystyle{{{paper.competition_profile.reference_style}}}",
                    r"\bibliography{references}",
                ]
            )
        tex = "\n\n".join([*preamble, *body, r"\end{document}"]) + "\n"
        self._safety.validate_rendered(tex)
        bib = BibTeXRenderer().render(citations)
        return RenderedPaper(
            tex=tex,
            bibliography=bib,
            tex_hash=sha256_text(tex),
            bib_hash=sha256_text(bib),
        )

    def _render_block(
        self,
        block: PaperBlock,
        claims: dict[str, Claim],
        equations: EquationRegistry,
        figures: FigureRegistry,
        tables: TableRegistry,
    ) -> str:
        citations = self._citations(block.citation_refs)
        if block.block_type in {PaperBlockType.PARAGRAPH, PaperBlockType.SUBSECTION}:
            prefix = (
                r"\subsection*{" + escape_latex(block.text or "") + "}"
                if block.block_type is PaperBlockType.SUBSECTION
                else escape_latex(block.text or "")
            )
            return prefix + citations
        if block.block_type is PaperBlockType.LIST:
            items = "\n".join(rf"\item {escape_latex(item)}" for item in block.items)
            return "\\begin{itemize}\n" + items + "\n\\end{itemize}" + citations
        if block.block_type is PaperBlockType.CLAIM:
            claim = claims.get(block.claim_ref or "")
            if claim is None:
                raise LatexRenderError(f"unknown claim: {block.claim_ref}")
            return escape_latex(claim.text) + self._citations(
                list(dict.fromkeys([*claim.citation_refs, *block.citation_refs]))
            )
        if block.block_type is PaperBlockType.EQUATION:
            equation = equations.lookup(block.equation_ref or "")
            if equation is None:
                raise LatexRenderError(f"unknown equation: {block.equation_ref}")
            self._safety.validate_equation(equation.latex)
            return (
                "\\begin{equation}\n"
                + equation.latex
                + rf"\label{{{equation.equation_id}}}"
                + "\n\\end{equation}"
                + citations
            )
        if block.block_type is PaperBlockType.FIGURE:
            figure = next(
                (item for item in figures.records() if item.figure_id == block.figure_ref), None
            )
            if figure is None:
                raise LatexRenderError(f"unknown figure: {block.figure_ref}")
            return (
                "\\begin{figure}[htbp]\n\\centering\n"
                + rf"\includegraphics[width=0.85\linewidth]{{{figure.figure_id}.png}}"
                + "\n"
                + rf"\caption{{{escape_latex(figure.caption)}}}\label{{{figure.figure_id}}}"
                + "\n\\end{figure}"
                + citations
            )
        if block.block_type is PaperBlockType.TABLE:
            table = next(
                (item for item in tables.records() if item.table_id == block.table_ref), None
            )
            if table is None:
                raise LatexRenderError(f"unknown table: {block.table_ref}")
            alignment = r">{\raggedright\arraybackslash}X" * len(table.columns)
            header = " & ".join(_escape_table_cell(item) for item in table.columns) + r" \\"
            rows = "\n".join(
                " & ".join(_escape_table_cell(str(value)) for value in row) + r" \\"
                for row in table.rows
            )
            return (
                "\\begin{table}[htbp]\n\\centering\n"
                + rf"\caption{{{escape_latex(table.caption)}}}\label{{{table.table_id}}}"
                + "\n"
                + rf"\begin{{tabularx}}{{\linewidth}}{{{alignment}}}\toprule"
                + "\n"
                + header
                + "\n\\midrule\n"
                + rows
                + "\n\\bottomrule\n\\end{tabularx}\n\\end{table}"
                + citations
            )
        raise LatexRenderError(f"unsupported block type: {block.block_type.value}")

    @staticmethod
    def _citations(reference_ids: list[str]) -> str:
        if not reference_ids:
            return ""
        if any(
            _SAFE_ID.fullmatch(item) is None or not item.startswith("REF-")
            for item in reference_ids
        ):
            raise LatexRenderError("unsafe citation key")
        return "~\\cite{" + ",".join(reference_ids) + "}"
