from collections.abc import Iterator

import pytest

from mathmodel_ai.core.config import Settings


@pytest.fixture
def test_settings() -> Iterator[Settings]:
    yield Settings(
        environment="test",
        database_url="sqlite+pysqlite:///:memory:",
        default_provider="mock",
        default_provider_model="mock-foundation",
    )
