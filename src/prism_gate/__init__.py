"""PrismGate: Multi-Tenant LLM Inference Gateway & Statistical Control Plane."""

__version__ = "1.0.0"

from prism_gate.admission.token_bucket import HierarchicalAdmissionController
from prism_gate.concur.vegas_limiter import VegasAdaptiveLimiter
from prism_gate.drift.detectors import (
    DriftPlaneAnalyzer,
    jensen_shannon_divergence,
    population_stability_index,
)
from prism_gate.evals.faithfulness import ClaimFaithfulnessJudge
from prism_gate.gateway.engine import PrismGateEngine
from prism_gate.routing.cascade_router import (
    SpeculativeCascadeRouter,
    TenantSemanticCache,
)
from prism_gate.types import (
    ChatCompletionRequest,
    ChatMessage,
    DriftReport,
    EndpointMetrics,
    FaithfulnessReport,
    StreamChunk,
    TenantSLOProfile,
)

__all__ = [
    "ChatCompletionRequest",
    "ChatMessage",
    "ClaimFaithfulnessJudge",
    "DriftPlaneAnalyzer",
    "DriftReport",
    "EndpointMetrics",
    "FaithfulnessReport",
    "HierarchicalAdmissionController",
    "PrismGateEngine",
    "SpeculativeCascadeRouter",
    "StreamChunk",
    "TenantSLOProfile",
    "TenantSemanticCache",
    "VegasAdaptiveLimiter",
    "__version__",
    "jensen_shannon_divergence",
    "population_stability_index",
]
