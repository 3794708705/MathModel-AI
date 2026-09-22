"""API route modules."""

from mathmodel_ai.api.routes.benchmarks import router as benchmarks_router
from mathmodel_ai.api.routes.providers import router as providers_router

__all__ = ["benchmarks_router", "providers_router"]
