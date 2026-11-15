"""Unit and Hypothesis property-based tests for `VegasAdaptiveLimiter`."""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from prism_gate.concur.vegas_limiter import VegasAdaptiveLimiter
from prism_gate.types import ModelTier


def test_vegas_constructor_and_sample_validation() -> None:
    with pytest.raises(ValueError, match="min_limit"):
        VegasAdaptiveLimiter("p", min_limit=0)
    with pytest.raises(ValueError, match="initial_limit"):
        VegasAdaptiveLimiter("p", min_limit=4, initial_limit=2, max_limit=10)
    with pytest.raises(ValueError, match="alpha"):
        VegasAdaptiveLimiter("p", alpha=0.0, beta=4.0)
    with pytest.raises(ValueError, match="alpha <= beta"):
        VegasAdaptiveLimiter("p", alpha=6.0, beta=4.0)
    with pytest.raises(ValueError, match="backoff_ratio"):
        VegasAdaptiveLimiter("p", backoff_ratio=1.0)
    with pytest.raises(ValueError, match="smoothing"):
        VegasAdaptiveLimiter("p", smoothing=0.0)
    with pytest.raises(ValueError, match="additive_step"):
        VegasAdaptiveLimiter("p", additive_step=0.0)

    lim = VegasAdaptiveLimiter("p")
    with pytest.raises(ValueError, match="positive finite"):
        lim.record_sample(rtt_sample_ms=0.0, inflight_sample=1)
    with pytest.raises(ValueError, match="positive finite"):
        lim.record_sample(rtt_sample_ms=math.nan, inflight_sample=1)
    with pytest.raises(ValueError, match="inflight_sample"):
        lim.record_sample(rtt_sample_ms=10.0, inflight_sample=-1)


def test_vegas_additive_growth_equilibrium_and_queue_backoff() -> None:
    lim = VegasAdaptiveLimiter(
        "gemini-2.5-flash",
        tier=ModelTier.FAST,
        initial_limit=10,
        min_limit=2,
        max_limit=20,
        alpha=2.0,
        beta=5.0,
        additive_step=1.0,
        backoff_ratio=0.75,
        probe_interval=4,
    )

    # Initial warmup sets rtt_min = 20.0, queue_estimate = 0 < alpha -> +1 -> 11
    assert lim.record_sample(rtt_sample_ms=20.0, inflight_sample=8) == 11
    assert lim.rtt_min_ms == pytest.approx(20.0)
    assert lim.queue_estimate == pytest.approx(0.0)

    # Equilibrium band: inflight=10, rtt=28.57 -> gradient=0.7 -> queue_est = 3.0 in [2.0, 5.0]
    assert lim.record_sample(rtt_sample_ms=28.5714, inflight_sample=10) == 11

    # Severe standing queue: inflight=12, rtt=80.0 -> gradient=0.25 -> queue_est = 9.0 > beta
    backed_off = lim.record_sample(rtt_sample_ms=80.0, inflight_sample=12)
    assert backed_off < 11

    # Explicit HTTP 429 overload backoff
    after_429 = lim.record_sample(rtt_sample_ms=50.0, inflight_sample=8, status_code=429)
    assert after_429 <= backed_off
    assert lim.snapshot().total_errors_429_503 == 1

    # Probe interval upward relaxation under low concurrency
    lim.record_sample(rtt_sample_ms=25.0, inflight_sample=1, status_code=200)
    assert lim.rtt_min_ms >= 20.0


def test_vegas_lease_acquire_shed_and_idempotent_release() -> None:
    now = 100.0

    def fake_clock() -> float:
        return now

    lim = VegasAdaptiveLimiter("vllm", initial_limit=2, min_limit=2, max_limit=8, clock=fake_clock)
    lease_1 = lim.try_acquire()
    lease_2 = lim.try_acquire()
    assert lease_1 is not None and lease_2 is not None
    assert lease_1.inflight_snapshot == 1
    assert lease_2.inflight_snapshot == 2
    assert lim.available_headroom() == 0

    # Saturated -> non-blocking shed
    assert lim.try_acquire() is None
    assert lim.snapshot().total_shed == 1

    now = 100.015  # 15 ms elapsed
    rtt = lease_1.release(status_code=200)
    assert rtt == pytest.approx(15.0)
    # Idempotent second release is a no-op
    assert lease_1.release(status_code=200) == 0.0
    lease_2.release(status_code=503, override_rtt_ms=40.0)
    assert lim.inflight == 0


@settings(max_examples=75, deadline=None)
@given(
    min_lim=st.integers(min_value=1, max_value=8),
    span=st.integers(min_value=0, max_value=40),
    samples=st.lists(
        st.tuples(
            st.floats(min_value=0.1, max_value=5000.0, allow_nan=False, allow_infinity=False),
            st.integers(min_value=0, max_value=100),
            st.sampled_from([200, 200, 200, 429, 503]),
        ),
        min_size=1,
        max_size=40,
    ),
)
def test_hypothesis_vegas_bounds_invariant(
    min_lim: int,
    span: int,
    samples: list[tuple[float, int, int]],
) -> None:
    max_lim = min_lim + span
    init_lim = (min_lim + max_lim) // 2
    limiter = VegasAdaptiveLimiter(
        "prop-provider",
        initial_limit=init_lim,
        min_limit=min_lim,
        max_limit=max_lim,
    )

    for rtt_ms, inflight, code in samples:
        new_ceil = limiter.record_sample(
            rtt_sample_ms=rtt_ms,
            inflight_sample=inflight,
            status_code=code,
        )
        assert min_lim <= new_ceil <= max_lim
        assert min_lim <= limiter.max_inflight <= max_lim
        assert 0.0 <= limiter.gradient <= 1.0
        assert limiter.queue_estimate >= 0.0
