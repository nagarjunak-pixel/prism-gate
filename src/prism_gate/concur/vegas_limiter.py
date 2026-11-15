"""TCP-Vegas / Netflix Gradient2-Inspired Adaptive Concurrency Limiter.

Prevents cascading tail-latency collapse across heterogeneous LLM upstreams by
continuously estimating upstream standing queue depth from Little's Law:

    gradient       = rtt_min / rtt_sample
    queue_estimate = inflight * (1 - rtt_min / rtt_sample)

Control law:
- Additive increase (`+additive_step`) when `queue_estimate < alpha` (no standing queue).
- Steady equilibrium band when `alpha <= queue_estimate <= beta`.
- Multiplicative Gradient2 backoff when `queue_estimate > beta` or on HTTP 429/503 signals.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable

from prism_gate.types import EndpointMetrics, ModelTier


class VegasLease:
    """Single-use inflight concurrency lease bound to a `VegasAdaptiveLimiter`."""

    __slots__ = ("_clock", "_inflight_snapshot", "_limiter", "_released", "_start_ns")

    def __init__(
        self,
        limiter: VegasAdaptiveLimiter,
        inflight_snapshot: int,
        clock: Callable[[], float],
    ) -> None:
        self._limiter = limiter
        self._inflight_snapshot = inflight_snapshot
        self._clock = clock
        self._start_ns = self._clock()
        self._released = False

    @property
    def inflight_snapshot(self) -> int:
        """Observed inflight count at lease acquisition."""
        return self._inflight_snapshot

    def release(
        self,
        *,
        status_code: int = 200,
        override_rtt_ms: float | None = None,
    ) -> float:
        """Idempotently release the inflight slot and update the Vegas control loop."""
        if self._released:
            return 0.0
        self._released = True
        elapsed_ms = (
            override_rtt_ms
            if override_rtt_ms is not None
            else max(0.05, (self._clock() - self._start_ns) * 1000.0)
        )
        self._limiter._on_lease_released(
            rtt_sample_ms=elapsed_ms,
            inflight_sample=self._inflight_snapshot,
            status_code=status_code,
        )
        return elapsed_ms


class VegasAdaptiveLimiter:
    """Adaptive concurrency limiter implementing TCP-Vegas + Gradient2 control math."""

    def __init__(
        self,
        provider_id: str,
        *,
        tier: ModelTier = ModelTier.FAST,
        initial_limit: int = 16,
        min_limit: int = 2,
        max_limit: int = 128,
        alpha: float = 3.0,
        beta: float = 6.0,
        additive_step: float = 1.0,
        backoff_ratio: float = 0.75,
        smoothing: float = 0.2,
        probe_interval: int = 64,
        cost_per_1k_input_usd: float = 0.00015,
        cost_per_1k_output_usd: float = 0.00060,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if min_limit < 1:
            raise ValueError("min_limit must be >= 1")
        if not (min_limit <= initial_limit <= max_limit):
            raise ValueError("Expected min_limit <= initial_limit <= max_limit")
        if alpha <= 0.0 or beta < alpha:
            raise ValueError("Expected 0 < alpha <= beta")
        if not (0.0 < backoff_ratio < 1.0):
            raise ValueError("Expected 0 < backoff_ratio < 1")
        if not (0.0 < smoothing <= 1.0):
            raise ValueError("Expected 0 < smoothing <= 1")
        if additive_step <= 0.0:
            raise ValueError("additive_step must be > 0")

        self.provider_id = provider_id
        self.tier = tier
        self.min_limit = min_limit
        self.max_limit = max_limit
        self.alpha = alpha
        self.beta = beta
        self.additive_step = additive_step
        self.backoff_ratio = backoff_ratio
        self.smoothing = smoothing
        self.probe_interval = max(1, probe_interval)
        self.cost_per_1k_input_usd = cost_per_1k_input_usd
        self.cost_per_1k_output_usd = cost_per_1k_output_usd
        self._clock = clock

        self._limit_float = float(initial_limit)
        self._inflight = 0
        self._rtt_min_ms: float | None = None
        self._rtt_ewma_ms = 0.0
        self._last_rtt_ms = 0.0
        self._last_queue_estimate = 0.0
        self._last_gradient = 1.0

        self._sample_count = 0
        self._total_requests = 0
        self._total_shed = 0
        self._total_errors_429_503 = 0

    @property
    def inflight(self) -> int:
        """Current number of active inflight upstream requests."""
        return self._inflight

    @property
    def max_inflight(self) -> int:
        """Current integer concurrency ceiling strictly bounded in `[min_limit, max_limit]`."""
        bounded = min(float(self.max_limit), max(float(self.min_limit), self._limit_float))
        return max(self.min_limit, min(self.max_limit, round(bounded)))

    @property
    def rtt_min_ms(self) -> float:
        """Observed baseline no-queue RTT in milliseconds."""
        return self._rtt_min_ms if self._rtt_min_ms is not None else 0.0

    @property
    def queue_estimate(self) -> float:
        """Latest TCP-Vegas estimated upstream standing queue size."""
        return self._last_queue_estimate

    @property
    def gradient(self) -> float:
        """Latest `rtt_min / rtt_sample` gradient in `[0, 1]`."""
        return self._last_gradient

    def available_headroom(self) -> int:
        """Remaining concurrency slots before fail-fast load shedding engages."""
        return max(0, self.max_inflight - self._inflight)

    def try_acquire(self) -> VegasLease | None:
        """Attempt non-blocking inflight slot acquisition; returns `None` when saturated."""
        if self._inflight >= self.max_inflight:
            self._total_shed += 1
            return None
        self._inflight += 1
        self._total_requests += 1
        return VegasLease(self, inflight_snapshot=self._inflight, clock=self._clock)

    def record_sample(
        self,
        *,
        rtt_sample_ms: float,
        inflight_sample: int,
        status_code: int = 200,
    ) -> int:
        """Directly feed an RTT/status observation into the Vegas/Gradient2 controller."""
        if not math.isfinite(rtt_sample_ms) or rtt_sample_ms <= 0.0:
            raise ValueError("rtt_sample_ms must be a positive finite number")
        if inflight_sample < 0:
            raise ValueError("inflight_sample must be >= 0")

        sample_ms = max(1e-6, rtt_sample_ms)
        effective_inflight = max(1, inflight_sample)
        self._sample_count += 1
        self._last_rtt_ms = sample_ms

        if self._rtt_ewma_ms == 0.0:
            self._rtt_ewma_ms = sample_ms
        else:
            self._rtt_ewma_ms = (
                self.smoothing * sample_ms + (1.0 - self.smoothing) * self._rtt_ewma_ms
            )

        # Explicit upstream overload / throttling signal (HTTP 429 Too Many Requests or 503/5xx)
        if status_code == 429 or status_code >= 500:
            self._total_errors_429_503 += 1
            self._limit_float = max(
                float(self.min_limit),
                self._limit_float * self.backoff_ratio,
            )
            if self._rtt_min_ms is not None:
                grad = min(1.0, max(0.0, self._rtt_min_ms / max(self._rtt_min_ms, sample_ms)))
                self._last_gradient = grad
                self._last_queue_estimate = max(0.0, effective_inflight * (1.0 - grad))
            return self.max_inflight

        # Gentle baseline RTT relaxation under low concurrency to prevent stale rtt_min lock-in
        if (
            self._rtt_min_ms is not None
            and effective_inflight <= self.min_limit
            and self._sample_count % self.probe_interval == 0
        ):
            self._rtt_min_ms *= 1.02

        if self._rtt_min_ms is None or sample_ms < self._rtt_min_ms:
            self._rtt_min_ms = sample_ms

        gradient = min(1.0, max(0.0, self._rtt_min_ms / sample_ms))
        queue_estimate = max(0.0, effective_inflight * (1.0 - gradient))

        self._last_gradient = gradient
        self._last_queue_estimate = queue_estimate

        if queue_estimate < self.alpha:
            # RTT tracks rtt_min closely -> no standing queue -> additive window expansion
            self._limit_float = min(
                float(self.max_limit),
                self._limit_float + self.additive_step,
            )
        elif queue_estimate > self.beta:
            # Standing queue detected -> multiplicative Gradient2 backoff proportional to gradient
            scale = max(self.backoff_ratio, min(0.95, gradient))
            self._limit_float = max(
                float(self.min_limit),
                self._limit_float * scale,
            )
        # Else: alpha <= queue_estimate <= beta -> optimal Little's Law operating band

        self._limit_float = min(
            float(self.max_limit),
            max(float(self.min_limit), self._limit_float),
        )
        return self.max_inflight

    def _on_lease_released(
        self,
        *,
        rtt_sample_ms: float,
        inflight_sample: int,
        status_code: int,
    ) -> None:
        self._inflight = max(0, self._inflight - 1)
        self.record_sample(
            rtt_sample_ms=rtt_sample_ms,
            inflight_sample=inflight_sample,
            status_code=status_code,
        )

    def snapshot(self) -> EndpointMetrics:
        """Return an immutable `EndpointMetrics` snapshot for telemetry & Prometheus."""
        return EndpointMetrics(
            provider_id=self.provider_id,
            tier=self.tier,
            inflight=self._inflight,
            max_inflight=self.max_inflight,
            min_limit=self.min_limit,
            max_limit=self.max_limit,
            rtt_min_ms=round(self.rtt_min_ms, 3),
            rtt_ewma_ms=round(self._rtt_ewma_ms, 3),
            last_rtt_ms=round(self._last_rtt_ms, 3),
            queue_estimate=round(self._last_queue_estimate, 3),
            gradient=round(self._last_gradient, 4),
            total_requests=self._total_requests,
            total_shed=self._total_shed,
            total_errors_429_503=self._total_errors_429_503,
            cost_per_1k_input_usd=self.cost_per_1k_input_usd,
            cost_per_1k_output_usd=self.cost_per_1k_output_usd,
        )
