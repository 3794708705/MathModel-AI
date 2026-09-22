from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mathmodel_ai.providers.anthropic import AnthropicProvider as AnthropicProvider
    from mathmodel_ai.providers.base import BaseModelProvider as BaseModelProvider
    from mathmodel_ai.providers.compatible import (
        CustomJSONHttpProvider as CustomJSONHttpProvider,
    )
    from mathmodel_ai.providers.compatible import (
        OpenAICompatibleProvider as OpenAICompatibleProvider,
    )
    from mathmodel_ai.providers.factory import ProviderRegistry as ProviderRegistry
    from mathmodel_ai.providers.factory import build_provider_registry as build_provider_registry
    from mathmodel_ai.providers.google import GoogleProvider as GoogleProvider
    from mathmodel_ai.providers.mock import MockProvider as MockProvider
    from mathmodel_ai.providers.openai import OpenAIProvider as OpenAIProvider
    from mathmodel_ai.providers.probe import (
        ProviderCompatibilityProbe as ProviderCompatibilityProbe,
    )
    from mathmodel_ai.providers.repository import ModelRegistry as ModelRegistry
    from mathmodel_ai.providers.repository import ProviderModelRegistry as ProviderModelRegistry
    from mathmodel_ai.providers.secrets import (
        BaseSecretResolver as BaseSecretResolver,
    )
    from mathmodel_ai.providers.secrets import (
        BaseSecretStore as BaseSecretStore,
    )
    from mathmodel_ai.providers.secrets import (
        CompositeSecretResolver as CompositeSecretResolver,
    )
    from mathmodel_ai.providers.secrets import (
        EncryptedDatabaseSecretStore as EncryptedDatabaseSecretStore,
    )
    from mathmodel_ai.providers.secrets import (
        EnvironmentSecretResolver as EnvironmentSecretResolver,
    )

_EXPORTS = {
    "AnthropicProvider": ("mathmodel_ai.providers.anthropic", "AnthropicProvider"),
    "BaseModelProvider": ("mathmodel_ai.providers.base", "BaseModelProvider"),
    "BaseSecretResolver": ("mathmodel_ai.providers.secrets", "BaseSecretResolver"),
    "BaseSecretStore": ("mathmodel_ai.providers.secrets", "BaseSecretStore"),
    "CompositeSecretResolver": ("mathmodel_ai.providers.secrets", "CompositeSecretResolver"),
    "CustomJSONHttpProvider": (
        "mathmodel_ai.providers.compatible",
        "CustomJSONHttpProvider",
    ),
    "EncryptedDatabaseSecretStore": (
        "mathmodel_ai.providers.secrets",
        "EncryptedDatabaseSecretStore",
    ),
    "EnvironmentSecretResolver": (
        "mathmodel_ai.providers.secrets",
        "EnvironmentSecretResolver",
    ),
    "GoogleProvider": ("mathmodel_ai.providers.google", "GoogleProvider"),
    "MockProvider": ("mathmodel_ai.providers.mock", "MockProvider"),
    "ModelRegistry": ("mathmodel_ai.providers.repository", "ModelRegistry"),
    "OpenAICompatibleProvider": (
        "mathmodel_ai.providers.compatible",
        "OpenAICompatibleProvider",
    ),
    "OpenAIProvider": ("mathmodel_ai.providers.openai", "OpenAIProvider"),
    "ProviderCompatibilityProbe": (
        "mathmodel_ai.providers.probe",
        "ProviderCompatibilityProbe",
    ),
    "ProviderModelRegistry": (
        "mathmodel_ai.providers.repository",
        "ProviderModelRegistry",
    ),
    "ProviderRegistry": ("mathmodel_ai.providers.factory", "ProviderRegistry"),
    "build_provider_registry": (
        "mathmodel_ai.providers.factory",
        "build_provider_registry",
    ),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
