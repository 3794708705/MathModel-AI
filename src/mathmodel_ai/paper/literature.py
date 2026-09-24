from __future__ import annotations

import asyncio
import html
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol
from uuid import NAMESPACE_URL, UUID, uuid5

import httpx

from mathmodel_ai.paper.hashing import sha256_json, sha256_text
from mathmodel_ai.schemas.paper import (
    CitationMetadataCheck,
    CitationSupportCheck,
    CitationSupportDraft,
    CitationSupportStatus,
    Claim,
    LiteratureSearchNeed,
    ReferenceAccessStatus,
    ReferenceMetadataOrigin,
    ReferenceMetadataStatus,
    ReferenceRecord,
    ReferenceSource,
)


def claim_digest(claim: Claim) -> str:
    """Bind citation review to immutable claim semantics, not persisted status."""

    return sha256_json(claim.model_dump(mode="json", exclude={"verification_status"}))


def reference_digest(reference: ReferenceRecord) -> str:
    """Bind verification to all citation metadata and trusted support text."""

    return sha256_json(
        reference.model_dump(mode="json", exclude={"metadata_status", "retrieved_at"})
    )


class LiteratureSource(Protocol):
    @property
    def name(self) -> str: ...

    async def search(
        self, need: LiteratureSearchNeed, project_id: UUID
    ) -> list[ReferenceRecord]: ...

    async def resolve(self, identifier: str, project_id: UUID) -> ReferenceRecord | None: ...


@dataclass(frozen=True)
class FixtureLiteratureSource:
    records: tuple[ReferenceRecord, ...]
    name: str = "fixture"

    async def search(self, need: LiteratureSearchNeed, project_id: UUID) -> list[ReferenceRecord]:
        terms = {item.casefold() for item in need.query.split() if len(item) >= 3}
        return [
            record
            for record in self.records
            if record.project_id == project_id
            and terms
            & {
                item.casefold().strip(".,;:()")
                for item in f"{record.title} {record.abstract or ''}".split()
            }
        ]

    async def resolve(self, identifier: str, project_id: UUID) -> ReferenceRecord | None:
        normalized = identifier.casefold().removeprefix("https://doi.org/")
        for record in self.records:
            if record.project_id == project_id and (
                record.reference_id.casefold() == normalized
                or (record.doi is not None and record.doi.casefold() == normalized)
            ):
                return record
        return None


class CrossrefLiteratureSource:
    """Live metadata adapter; it never fabricates missing Crossref fields."""

    name = "crossref"

    def __init__(
        self,
        client: httpx.AsyncClient | None = None,
        *,
        base_url: str = "https://api.crossref.org",
        mailto: str | None = None,
    ) -> None:
        self._client = client or httpx.AsyncClient(timeout=20.0, follow_redirects=False)
        self._owns_client = client is None
        self._base_url = base_url.rstrip("/")
        self._mailto = mailto

    async def search(self, need: LiteratureSearchNeed, project_id: UUID) -> list[ReferenceRecord]:
        params = {"query.bibliographic": need.query, "rows": "5"}
        if self._mailto:
            params["mailto"] = self._mailto
        response = await self._get(f"{self._base_url}/works", params=params)
        response.raise_for_status()
        payload = response.json()
        items = payload.get("message", {}).get("items", [])
        if not isinstance(items, list):
            return []
        records: list[ReferenceRecord] = []
        for item in items:
            if isinstance(item, dict):
                record = self._record(item, project_id)
                if record is not None:
                    records.append(record)
        return records

    async def resolve(self, identifier: str, project_id: UUID) -> ReferenceRecord | None:
        normalized = identifier.removeprefix("https://doi.org/").removeprefix("http://doi.org/")
        params = {"mailto": self._mailto} if self._mailto else None
        response = await self._get(f"{self._base_url}/works/{normalized}", params=params)
        if response.status_code == 404:
            return None
        item = response.json().get("message")
        return self._record(item, project_id) if isinstance(item, dict) else None

    async def _get(self, url: str, *, params: dict[str, str] | None) -> httpx.Response:
        """Respect bounded Crossref rate-limit backoff without inventing references."""
        for attempt in range(4):
            response = await self._client.get(url, params=params)
            if response.status_code != 429 or attempt == 3:
                if response.status_code != 404:
                    response.raise_for_status()
                return response
            retry_after = response.headers.get("Retry-After", "")
            try:
                delay = float(retry_after)
            except ValueError:
                delay = 2.0 * (2**attempt)
            await asyncio.sleep(min(max(delay, 1.0), 30.0))
        raise AssertionError("unreachable")

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    @staticmethod
    def _record(item: dict[str, object], project_id: UUID) -> ReferenceRecord | None:
        title_values = item.get("title")
        title = (
            str(title_values[0]).strip() if isinstance(title_values, list) and title_values else ""
        )
        author_values = item.get("author")
        authors = []
        if isinstance(author_values, list):
            for author in author_values:
                if not isinstance(author, dict):
                    continue
                name = " ".join(
                    str(author.get(key, "")).strip() for key in ("given", "family")
                ).strip()
                if name:
                    authors.append(name)
        year = CrossrefLiteratureSource._year(item)
        container = item.get("container-title")
        venue = str(container[0]).strip() if isinstance(container, list) and container else ""
        if not venue:
            # Crossref books and technical reports frequently omit container-title while
            # providing an authoritative publisher. Dropping those records made valid live
            # searches appear empty; the publisher is retained verbatim, never inferred.
            venue = str(item.get("publisher", "")).strip()
        doi = str(item.get("DOI", "")).strip().lower() or None
        source_id = doi or str(item.get("URL", "")).strip()
        if not title or not authors or year is None or not venue or not source_id:
            return None
        abstract_raw = item.get("abstract")
        abstract = (
            html.unescape(re.sub(r"<[^>]+>", " ", str(abstract_raw))).strip()
            if abstract_raw
            else None
        )
        raw_hash = sha256_json(item)
        return ReferenceRecord(
            reference_id=f"REF-{uuid5(NAMESPACE_URL, f'crossref:{source_id}').hex[:12]}",
            project_id=project_id,
            title=title,
            authors=authors,
            year=year,
            venue=venue,
            doi=doi,
            url=str(item.get("URL", "")).strip() or None,
            abstract=abstract,
            trusted_excerpt=abstract,
            source=ReferenceSource.CROSSREF,
            source_id=source_id,
            metadata_origin=ReferenceMetadataOrigin.RETRIEVED,
            retrieved_at=datetime.now(UTC),
            metadata_status=ReferenceMetadataStatus.PENDING,
            access_status=(
                ReferenceAccessStatus.AVAILABLE if abstract else ReferenceAccessStatus.METADATA_ONLY
            ),
            raw_metadata_hash=raw_hash,
        )

    @staticmethod
    def _year(item: dict[str, object]) -> int | None:
        for field in ("issued", "published", "published-print", "published-online"):
            value = item.get(field)
            if not isinstance(value, dict):
                continue
            date_parts = value.get("date-parts")
            if not (
                isinstance(date_parts, list)
                and date_parts
                and isinstance(date_parts[0], list)
                and date_parts[0]
            ):
                continue
            raw_year = date_parts[0][0]
            if isinstance(raw_year, int | float):
                return int(raw_year)
        return None


class CitationMetadataVerifier:
    """Compare a candidate record with independently retrieved source metadata."""

    async def verify(
        self,
        candidate: ReferenceRecord,
        source: LiteratureSource,
    ) -> CitationMetadataCheck:
        identifier = candidate.doi or candidate.source_id
        authoritative = await source.resolve(identifier, candidate.project_id)
        if authoritative is None:
            return CitationMetadataCheck(
                project_id=candidate.project_id,
                reference_id=candidate.reference_id,
                reference_digest=reference_digest(candidate),
                status=ReferenceMetadataStatus.NOT_FOUND,
                field_matches={},
                errors=["reference not found in independent literature source"],
            )
        checks = {
            "title": self._text(candidate.title) == self._text(authoritative.title),
            "authors": [self._text(item) for item in candidate.authors]
            == [self._text(item) for item in authoritative.authors],
            "year": candidate.year == authoritative.year,
            "venue": self._text(candidate.venue) == self._text(authoritative.venue),
            "doi": candidate.doi == authoritative.doi,
        }
        status = (
            ReferenceMetadataStatus.VERIFIED
            if all(checks.values())
            else ReferenceMetadataStatus.CONFLICT
        )
        return CitationMetadataCheck(
            project_id=candidate.project_id,
            reference_id=candidate.reference_id,
            reference_digest=reference_digest(candidate),
            status=status,
            field_matches=checks,
            errors=[] if status is ReferenceMetadataStatus.VERIFIED else ["metadata conflict"],
        )

    @staticmethod
    def verified_copy(record: ReferenceRecord, check: CitationMetadataCheck) -> ReferenceRecord:
        return record.model_copy(update={"metadata_status": check.status})

    @staticmethod
    def canonical_metadata_hash(record: ReferenceRecord) -> str:
        return sha256_json(
            {
                "title": record.title,
                "authors": record.authors,
                "year": record.year,
                "venue": record.venue,
                "doi": record.doi,
                "url": record.url,
                "source": record.source.value,
                "source_id": record.source_id,
            }
        )

    @staticmethod
    def _text(value: str) -> str:
        return " ".join(value.casefold().split())


class CitationSupportVerifier:
    """Bind an independent semantic review to the exact trusted source text."""

    def verify(
        self,
        *,
        claim: Claim,
        reference: ReferenceRecord,
        draft: CitationSupportDraft,
        reviewer_is_mock: bool,
    ) -> CitationSupportCheck:
        trusted_text = reference.trusted_excerpt or reference.abstract
        if reference.metadata_status is not ReferenceMetadataStatus.VERIFIED:
            return self._check(
                claim,
                reference,
                CitationSupportStatus.NOT_SUPPORTED,
                None,
                "reference metadata is not verified",
                reviewer_is_mock,
            )
        if not trusted_text:
            return self._check(
                claim,
                reference,
                CitationSupportStatus.INSUFFICIENT_TEXT,
                None,
                "trusted abstract or excerpt is unavailable",
                reviewer_is_mock,
            )
        if reviewer_is_mock and draft.status in {
            CitationSupportStatus.SUPPORTED,
            CitationSupportStatus.PARTIALLY_SUPPORTED,
        }:
            return self._check(
                claim,
                reference,
                CitationSupportStatus.INSUFFICIENT_TEXT,
                trusted_text,
                "Mock review cannot establish citation support",
                True,
            )
        excerpt = draft.supporting_excerpt
        if draft.status in {
            CitationSupportStatus.SUPPORTED,
            CitationSupportStatus.PARTIALLY_SUPPORTED,
        } and (not excerpt or self._normalize(excerpt) not in self._normalize(trusted_text)):
            return self._check(
                claim,
                reference,
                CitationSupportStatus.NOT_SUPPORTED,
                trusted_text,
                "claimed supporting excerpt is absent from trusted source text",
                reviewer_is_mock,
            )
        return CitationSupportCheck(
            project_id=reference.project_id,
            claim_id=claim.claim_id,
            reference_id=reference.reference_id,
            claim_digest=claim_digest(claim),
            reference_digest=reference_digest(reference),
            status=draft.status,
            trusted_text_hash=sha256_text(trusted_text),
            supporting_excerpt=excerpt,
            rationale=draft.rationale,
            reviewer_is_mock=reviewer_is_mock,
        )

    @staticmethod
    def _check(
        claim: Claim,
        reference: ReferenceRecord,
        status: CitationSupportStatus,
        trusted_text: str | None,
        rationale: str,
        reviewer_is_mock: bool,
    ) -> CitationSupportCheck:
        return CitationSupportCheck(
            project_id=reference.project_id,
            claim_id=claim.claim_id,
            reference_id=reference.reference_id,
            claim_digest=claim_digest(claim),
            reference_digest=reference_digest(reference),
            status=status,
            trusted_text_hash=sha256_text(trusted_text) if trusted_text else None,
            rationale=rationale,
            reviewer_is_mock=reviewer_is_mock,
        )

    @staticmethod
    def _normalize(value: str) -> str:
        return " ".join(value.casefold().split())
