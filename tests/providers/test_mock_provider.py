import pytest
from pydantic import BaseModel

from mathmodel_ai.providers.mock import MockProvider
from mathmodel_ai.providers.schemas import GenerationRequest, ModelMessage


class Answer(BaseModel):
    value: int


def request() -> GenerationRequest:
    return GenerationRequest(model="mock-model", messages=[ModelMessage(role="user", content="x")])


@pytest.mark.asyncio
async def test_mock_provider_is_explicitly_marked() -> None:
    provider = MockProvider(['{"value": 7}'])
    response = await provider.structured_generate(request(), Answer)
    assert response.parsed.value == 7
    assert response.response.is_mock is True
    assert provider.get_usage().requests == 1


@pytest.mark.asyncio
async def test_mock_stream_returns_queued_content() -> None:
    provider = MockProvider(["chunk"])
    assert [chunk async for chunk in provider.stream(request())] == ["chunk"]


@pytest.mark.asyncio
async def test_mock_provider_never_invents_an_unqueued_response() -> None:
    provider = MockProvider()
    with pytest.raises(RuntimeError, match="no queued response"):
        await provider.generate(request())
