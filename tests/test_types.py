"""Unit tests for frozen Pydantic v2 contracts in `prism_gate.types`."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from prism_gate.types import (
    ChatCompletionRequest,
    ChatMessage,
    ClaimVerdict,
    DriftReport,
    DriftSeverity,
    EndpointMetrics,
    FaithfulnessReport,
    ModelTier,
    PriorityTier,
    StreamChoice,
    StreamChunk,
    StreamDelta,
    TenantSLOProfile,
)


def test_models_are_frozen_and_forbid_extra_fields() -> None:
    msg = ChatMessage(role="user", content="Verify SLO bounds.")
    with pytest.raises(ValidationError):
        msg.content = "Mutated"  # type: ignore[misc]

    with pytest.raises(ValidationError):
        ChatMessage(role="user", content="hi", extra_field="bad")  # type: ignore[call-arg]

    req = ChatCompletionRequest(messages=(msg,), priority=PriorityTier.CRITICAL)
    assert req.tenant_id == "tenant-enterprise-01"
    with pytest.raises(ValidationError):
        req.stream = True  # type: ignore[misc]


def test_domain_contracts_roundtrip() -> None:
    slo = TenantSLOProfile(
        tenant_id="t-1",
        rps_limit=25.0,
        burst_rps=50.0,
        tpm_limit=100_000.0,
        burst_tokens=20_000.0,
    )
    assert slo.default_priority == PriorityTier.INTERACTIVE

    ep = EndpointMetrics(
        provider_id="gemini-2.5-flash",
        tier=ModelTier.FAST,
        inflight=2,
        max_inflight=16,
        min_limit=2,
        max_limit=48,
        rtt_min_ms=18.0,
        rtt_ewma_ms=19.2,
        last_rtt_ms=19.0,
        queue_estimate=0.5,
        gradient=0.94,
        total_requests=100,
        total_shed=1,
        total_errors_429_503=0,
        cost_per_1k_input_usd=0.00015,
        cost_per_1k_output_usd=0.0006,
    )
    assert ep.provider_id == "gemini-2.5-flash"

    chunk = StreamChunk(
        id="chatcmpl-123",
        created=1700000000,
        model="gemini-2.5-flash",
        choices=(StreamChoice(index=0, delta=StreamDelta(content="hello")),),
    )
    assert chunk.object == "chat.completion.chunk"

    faith = FaithfulnessReport(
        faithfulness_score=1.0,
        total_claims=1,
        supported_claims_count=1,
        hallucination_detected=False,
    )
    assert not faith.hallucination_detected
    assert ClaimVerdict.SUPPORTED.value == "SUPPORTED"

    drift = DriftReport(
        overall_severity=DriftSeverity.NO_DRIFT,
        max_psi=0.01,
        token_jsd=0.002,
        topic_jsd=0.001,
        feature_psi=(),
        baseline_sample_size=100,
        live_sample_size=100,
    )
    assert drift.overall_severity == DriftSeverity.NO_DRIFT
