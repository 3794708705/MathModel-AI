from __future__ import annotations

import os
from uuid import uuid4

import pytest

from mathmodel_ai.paper.literature import CitationMetadataVerifier, CrossrefLiteratureSource
from mathmodel_ai.schemas.paper import (
    LiteratureNeedType,
    LiteratureSearchNeed,
    ReferenceMetadataStatus,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("MM_RUN_LIVE_LITERATURE_TESTS") != "1",
        reason="SKIPPED_NO_NETWORK: set MM_RUN_LIVE_LITERATURE_TESTS=1 to opt in",
    ),
]


@pytest.mark.asyncio
async def test_live_crossref_search_and_independent_metadata_resolution() -> None:
    source = CrossrefLiteratureSource()
    try:
        records = await source.search(
            LiteratureSearchNeed(
                need_id="LITNEED-live-lp",
                need_type=LiteratureNeedType.THEORY,
                query="linear programming Dantzig",
                purpose="verify live method metadata",
            ),
            uuid4(),
        )
        assert records
        check = await CitationMetadataVerifier().verify(records[0], source)
        assert check.status is ReferenceMetadataStatus.VERIFIED
        assert all(check.field_matches.values())
    finally:
        await source.aclose()
