from mathmodel_ai.core.config import Settings
from mathmodel_ai.core.errors import ConfigurationError
from mathmodel_ai.core.types import Environment, ProviderName
from mathmodel_ai.providers.anthropic import AnthropicProvider
from mathmodel_ai.providers.base import BaseModelProvider
from mathmodel_ai.providers.google import GoogleProvider
from mathmodel_ai.providers.mock import MockProvider
from mathmodel_ai.providers.openai import OpenAIProvider


class ProviderRegistry:
    def __init__(self, providers: list[BaseModelProvider]) -> None:
        self._providers = {provider.name: provider for provider in providers}

    def get(self, name: ProviderName) -> BaseModelProvider:
        try:
            return self._providers[name]
        except KeyError as exc:
            raise ConfigurationError(f"provider {name.value!r} is not configured") from exc

    @property
    def available(self) -> frozenset[ProviderName]:
        return frozenset(self._providers)

    async def aclose(self) -> None:
        for provider in self._providers.values():
            await provider.aclose()


def build_provider_registry(settings: Settings) -> ProviderRegistry:
    providers: list[BaseModelProvider] = []
    if settings.environment is not Environment.PRODUCTION:
        providers.append(MockProvider())
    timeout = settings.provider_timeout_seconds
    if settings.openai_api_key is not None:
        providers.append(
            OpenAIProvider(
                api_key=settings.openai_api_key.get_secret_value(),
                base_url=settings.openai_base_url,
                timeout_seconds=timeout,
            )
        )
    if settings.google_api_key is not None:
        providers.append(
            GoogleProvider(
                api_key=settings.google_api_key.get_secret_value(),
                base_url=settings.google_base_url,
                timeout_seconds=timeout,
            )
        )
    if settings.anthropic_api_key is not None:
        providers.append(
            AnthropicProvider(
                api_key=settings.anthropic_api_key.get_secret_value(),
                base_url=settings.anthropic_base_url,
                api_version=settings.anthropic_version,
                timeout_seconds=timeout,
            )
        )
    return ProviderRegistry(providers)
