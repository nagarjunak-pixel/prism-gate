"""FastAPI OpenAI-compatible REST & SSE streaming gateway."""

from prism_gate.gateway.app import app, create_app
from prism_gate.gateway.engine import GatewayRejectedError, PrismGateEngine

__all__ = ["GatewayRejectedError", "PrismGateEngine", "app", "create_app"]
