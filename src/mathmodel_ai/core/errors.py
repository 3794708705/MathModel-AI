class MathModelError(Exception):
    """Base class for expected application failures."""


class ConfigurationError(MathModelError):
    """Raised when required runtime configuration is unavailable or invalid."""


class ProviderError(MathModelError):
    """Raised for safe, normalized model-provider failures."""


class ProviderResponseError(ProviderError):
    """Raised when a provider response violates the adapter contract."""


class RoutingError(MathModelError):
    """Raised when no safe model route can be produced."""


class AgentRunError(MathModelError):
    """Raised after an agent exhausts its configured attempts."""


class InvalidStateTransitionError(MathModelError):
    """Raised when a workflow stage is skipped or executed out of order."""


class QualityGateError(MathModelError):
    """Raised when deterministic stage acceptance checks fail."""


class ResourceNotFoundError(MathModelError):
    """Raised when a requested persisted project or problem does not exist."""
