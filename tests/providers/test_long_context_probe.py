import re
from types import SimpleNamespace

import pytest

from mathmodel_ai.providers.probe import ProviderCompatibilityProbe
from mathmodel_ai.schemas.provider_registry import CapabilityStatus, ModelCapability


class EchoMarkerProvider:
    def __init__(self, *, token_count: int, correct: bool) -> None:
        self.token_count = token_count
        self.correct = correct
        self.request = None

    async def generate(self, request):
        self.request = request
        match = re.search(r"private marker is ([a-f0-9]{32})", request.messages[0].content)
        assert match is not None
        return SimpleNamespace(
            content=match.group(1) if self.correct else "wrong",
            usage=SimpleNamespace(input_tokens=self.token_count),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("token_count", "correct", "expected"),
    [
        (20000, True, CapabilityStatus.SUPPORTED),
        (20000, False, CapabilityStatus.UNSUPPORTED),
        (100, True, CapabilityStatus.UNSUPPORTED),
    ],
)
async def test_long_context_requires_recall_and_reported_token_evidence(
    token_count, correct, expected
):
    provider = EchoMarkerProvider(token_count=token_count, correct=correct)
    capabilities = {}
    errors = []
    await ProviderCompatibilityProbe._probe_long_context(
        provider, "test-model", capabilities, errors
    )
    assert capabilities[ModelCapability.LONG_CONTEXT].status is expected
    assert len(provider.request.messages[0].content) > 80000
    assert not errors
