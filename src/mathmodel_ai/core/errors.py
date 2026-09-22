class MathModelError(Exception):
    """Base class for expected application failures."""


class ConfigurationError(MathModelError):
    """Raised when required runtime configuration is unavailable or invalid."""


class ProviderError(MathModelError):
    """Raised for safe, normalized model-provider failures."""


class ProviderResponseError(ProviderError):
    """Raised when a provider response violates the adapter contract."""


class ProviderAuthenticationError(ProviderError):
    """Raised when a remote endpoint rejects configured authentication."""


class ProviderTimeoutError(ProviderError):
    """Raised when a bounded provider request exceeds its deadline."""


class ProviderResponseTooLargeError(ProviderError):
    """Raised before an endpoint response can exceed the configured byte cap."""


class ProviderRedirectError(ProviderError):
    """Raised when an endpoint attempts a redirect that could leak credentials."""


class ModelDiscoveryError(ProviderError):
    """Raised with a stable, secret-safe model discovery error code."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int = 502,
        provider_reached: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.provider_reached = provider_reached


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


class FileValidationError(MathModelError):
    """Raised when uploaded bytes fail the guarded file contract."""


class FileParseError(MathModelError):
    """Raised when a validated file cannot be deterministically parsed."""


class StorageError(MathModelError):
    """Raised when the file-store boundary cannot preserve an artifact safely."""


class SandboxError(MathModelError):
    """Raised for invalid sandbox requests or local isolation failures."""


class SolverUnavailableError(MathModelError):
    """Raised when routing finds no installed, licensed, compatible solver."""


class DependencyUnavailableError(MathModelError):
    """Raised when generated code requests a dependency outside the approved runtime."""
