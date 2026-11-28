"""Tests for BoundedDecisionRouter, PipeAxes slicer, SwitchAdaptedDriftTracker & receipts."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from hypothesis import given, settings
from hypothesis import strategies as st

from prism_gate.gateway.app import create_app
from prism_gate.routing.bounded_decision_router import (
    BoundedDecisionRouter,
    SignedBillingReceipt,
    SwitchAdaptedDriftTracker,
    compute_pipeaxes_slice_plan,
)


def test_bounded_decision_router_validation_and_routes() -> None:
    with pytest.raises(ValueError, match="min_margin"):
        BoundedDecisionRouter(min_margin=1.5)
    with pytest.raises(ValueError, match="max_normalized_entropy"):
        BoundedDecisionRouter(max_normalized_entropy=0.0)
    with pytest.raises(ValueError, match="temperature"):
        BoundedDecisionRouter(temperature=0.0)

    router = BoundedDecisionRouter(min_margin=0.20, max_normalized_entropy=0.65)
    with pytest.raises(ValueError, match="equal length >= 2"):
        router.evaluate(["only_one"], [1.0])
    with pytest.raises(ValueError, match="equal length >= 2"):
        router.evaluate(["a", "b"], [1.0])

    clear = router.evaluate(["billing", "technical", "cancel"], [4.5, 0.2, -1.0])
    assert clear.route == "bounded_zero_text"
    assert clear.selected_option == "billing"
    assert clear.near_tie_bf16_risk is False
    assert sum(clear.probabilities) == pytest.approx(1.0, abs=1e-9)

    # Near-tie logit gap < 2^-6 triggers bfloat16 cross-device rank-flip protection
    near_tie = router.evaluate(["opt_a", "opt_b"], [2.005, 2.000])
    assert near_tie.near_tie_bf16_risk is True
    assert near_tie.route == "looped_reasoning_tier"


def test_pipeaxes_slice_plan_and_validation() -> None:
    with pytest.raises(ValueError, match="prompt_tokens"):
        compute_pipeaxes_slice_plan(0)
    with pytest.raises(ValueError, match="ttft_slo_ms"):
        compute_pipeaxes_slice_plan(1024, ttft_slo_ms=10.0)
    with pytest.raises(ValueError, match="tokens_per_ms_throughput"):
        compute_pipeaxes_slice_plan(1024, tokens_per_ms_throughput=0.0)

    long_ctx = compute_pipeaxes_slice_plan(prompt_tokens=65_536, ttft_slo_ms=600.0)
    assert long_ctx.num_slices >= 16
    assert long_ctx.transport_compression_ratio == 3.75
    assert long_ctx.estimated_short_req_p95_ttft_ms <= 560.0

    fp16 = compute_pipeaxes_slice_plan(prompt_tokens=512, ttft_slo_ms=600.0, hadamard_int4=False)
    assert fp16.transport_compression_ratio == 1.0
    assert fp16.num_slices == 1


def test_signed_billing_receipt_and_tamper_detection() -> None:
    key = b"0123456789abcdef0123456789abcdef"
    with pytest.raises(ValueError, match="signing_key"):
        SignedBillingReceipt.issue(b"short", "r1", "m1", 10, ("hello",))
    with pytest.raises(ValueError, match="prompt_tokens"):
        SignedBillingReceipt.issue(key, "r1", "m1", -1, ("hello",))

    chunks = ("Order ", "approved ", "cleanly")
    rec = SignedBillingReceipt.issue(key, "req-42", "gemini-2.5-flash", 128, chunks)
    assert rec.verify(key, chunks) is True
    assert rec.verify(key, ("Order ", "tampered ", "cleanly")) is False
    assert rec.verify(b"wrong-key-0123456789abcdef012345", chunks) is False

    bad_count = SignedBillingReceipt(
        request_id=rec.request_id,
        model_id=rec.model_id,
        prompt_tokens=rec.prompt_tokens,
        completion_tokens=rec.completion_tokens + 5,
        stream_sha256=rec.stream_sha256,
        signature_hex=rec.signature_hex,
    )
    assert bad_count.verify(key, chunks) is False


def test_switch_adapted_drift_tracker_reacts_to_abrupt_shift() -> None:
    with pytest.raises(ValueError, match="hazard_rate"):
        SwitchAdaptedDriftTracker(hazard_rate=1.0)
    with pytest.raises(ValueError, match="prior_alpha"):
        SwitchAdaptedDriftTracker(prior_alpha=0.0)

    tracker = SwitchAdaptedDriftTracker(hazard_rate=0.10)
    low_rate = 0.0
    for _ in range(30):
        low_rate, _ = tracker.update(False)
    assert low_rate < 0.08

    high_rate = 0.0
    switch_prob = 0.0
    for _ in range(12):
        high_rate, switch_prob = tracker.update(True)
    assert high_rate > 0.60
    assert switch_prob > 0.40


@given(
    logits=st.lists(
        st.floats(min_value=-20.0, max_value=20.0, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=8,
    )
)
@settings(max_examples=50)
def test_property_bounded_router_probabilities_normalized(logits: list[float]) -> None:
    opts = [f"opt_{i}" for i in range(len(logits))]
    verdict = BoundedDecisionRouter().evaluate(opts, logits)
    assert sum(verdict.probabilities) == pytest.approx(1.0, abs=1e-9)
    assert 0.0 <= verdict.top1_top2_margin <= 1.0 + 1e-9
    assert 0.0 <= verdict.normalized_entropy <= 1.0 + 1e-9


def test_fastapi_bounded_decision_endpoint() -> None:
    client = TestClient(create_app())
    ok = client.post(
        "/api/v1/routing/bounded-decision",
        json={
            "options": ["approve", "escalate", "reject"],
            "logits": [5.2, 0.4, -1.2],
            "prompt_tokens": 65536,
            "ttft_slo_ms": 600.0,
        },
    )
    assert ok.status_code == 200
    body = ok.json()
    assert body["route"] == "bounded_zero_text"
    assert body["selected_option"] == "approve"
    assert body["num_slices"] >= 16

    mismatch = client.post(
        "/api/v1/routing/bounded-decision",
        json={"options": ["a", "b"], "logits": [1.0, 2.0, 3.0]},
    )
    assert mismatch.status_code == 422
