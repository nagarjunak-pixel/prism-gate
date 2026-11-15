"""Hierarchical Multi-Tenant Admission Controller & Priority Load Shedder.

Combines:
1. Gateway-level progressive priority load shedding starting at 85% utilization:
   - `u < 0.85`: Admit `CRITICAL`, `INTERACTIVE`, `BATCH_BACKGROUND`
   - `0.85 <= u < 0.95`: Shed `BATCH_BACKGROUND`
   - `0.95 <= u < 1.00`: Shed `BATCH_BACKGROUND` and `INTERACTIVE` (preserve `CRITICAL`)
   - `u >= 1.00`: Shed all tiers (`GATEWAY_SATURATED`)
2. Atomic per-tenant dual-rate token buckets (Requests-Per-Second + Tokens-Per-Minute).
   Shedding at the gateway tier never penalizes a tenant's own RPS/TPM quota.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from prism_gate.types import AdmissionSummary, PriorityTier, TenantSLOProfile


class AdmissionDecision(BaseModel):
    """Immutable admission verdict returned before upstream dispatch."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    allowed: bool
    reason: str
    status_code: int = Field(ge=200, le=599)
    tenant_id: str
    priority: PriorityTier
    gateway_utilization: float = Field(ge=0.0)
    retry_after_seconds: float = Field(ge=0.0)
    rps_tokens_remaining: float = Field(ge=0.0)
    tpm_tokens_remaining: float = Field(ge=0.0)


@dataclass
class DualTokenBucket:
    """Atomic dual-rate token bucket tracking RPS and TPM simultaneously."""

    profile: TenantSLOProfile
    clock: Callable[[], float]

    def __post_init__(self) -> None:
        self._rps_tokens: float = self.profile.burst_rps
        self._tpm_tokens: float = self.profile.burst_tokens
        self._last_refill: float = self.clock()

    @property
    def rps_tokens(self) -> float:
        self._refill()
        return self._rps_tokens

    @property
    def tpm_tokens(self) -> float:
        self._refill()
        return self._tpm_tokens

    def _refill(self) -> None:
        now = self.clock()
        elapsed = max(0.0, now - self._last_refill)
        self._last_refill = now
        if elapsed <= 0.0:
            return
        self._rps_tokens = min(
            self.profile.burst_rps,
            self._rps_tokens + elapsed * self.profile.rps_limit,
        )
        tpm_per_sec = self.profile.tpm_limit / 60.0
        self._tpm_tokens = min(
            self.profile.burst_tokens,
            self._tpm_tokens + elapsed * tpm_per_sec,
        )

    def try_consume(self, estimated_tokens: int) -> tuple[bool, str, float]:
        """Atomically consume 1 request token and `estimated_tokens` TPM tokens.

        Returns `(allowed, reason, retry_after_seconds)`.
        """
        if estimated_tokens < 1:
            raise ValueError("estimated_tokens must be >= 1")

        self._refill()
        needed_tokens = float(estimated_tokens)

        if self._rps_tokens < 1.0:
            deficit = 1.0 - self._rps_tokens
            retry_after = deficit / self.profile.rps_limit
            return False, "TENANT_RPS_EXCEEDED", max(0.01, round(retry_after, 4))

        if self._tpm_tokens < needed_tokens:
            if needed_tokens > self.profile.burst_tokens:
                return False, "TENANT_TPM_BURST_EXCEEDED", 1.0
            deficit = needed_tokens - self._tpm_tokens
            tpm_per_sec = self.profile.tpm_limit / 60.0
            retry_after = deficit / tpm_per_sec
            return False, "TENANT_TPM_EXCEEDED", max(0.01, round(retry_after, 4))

        # Both checks passed -> atomic deduction
        self._rps_tokens -= 1.0
        self._tpm_tokens -= needed_tokens
        return True, "ADMITTED", 0.0

    def reconcile(self, *, estimated_tokens: int, actual_tokens: int) -> None:
        """Reconcile pre-dispatch token reservation against actual upstream usage."""
        self._refill()
        delta = float(estimated_tokens - actual_tokens)
        if delta > 0.0:
            self._tpm_tokens = min(self.profile.burst_tokens, self._tpm_tokens + delta)
        elif delta < 0.0:
            self._tpm_tokens = max(0.0, self._tpm_tokens + delta)


class HierarchicalAdmissionController:
    """Multi-tenant admission plane with 85% gateway utilization priority shedding."""

    BATCH_SHED_UTILIZATION: float = 0.85
    INTERACTIVE_SHED_UTILIZATION: float = 0.95
    SATURATED_UTILIZATION: float = 1.00

    def __init__(
        self,
        *,
        profiles: list[TenantSLOProfile] | None = None,
        default_rps_limit: float = 50.0,
        default_burst_rps: float = 100.0,
        default_tpm_limit: float = 240_000.0,
        default_burst_tokens: float = 40_000.0,
        batch_shed_threshold: float = BATCH_SHED_UTILIZATION,
        interactive_shed_threshold: float = INTERACTIVE_SHED_UTILIZATION,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not (0.0 < batch_shed_threshold < interactive_shed_threshold <= 1.0):
            raise ValueError("Expected 0 < batch_shed_threshold < interactive_shed_threshold <= 1")

        self.batch_shed_threshold = batch_shed_threshold
        self.interactive_shed_threshold = interactive_shed_threshold
        self._default_rps_limit = default_rps_limit
        self._default_burst_rps = default_burst_rps
        self._default_tpm_limit = default_tpm_limit
        self._default_burst_tokens = default_burst_tokens
        self._clock = clock

        self._buckets: dict[str, DualTokenBucket] = {}
        for profile in profiles or self._builtin_profiles():
            self.register_tenant(profile)

        self._admitted_total = 0
        self._rejected_rps_total = 0
        self._rejected_tpm_total = 0
        self._shed_batch_total = 0
        self._shed_interactive_total = 0
        self._shed_saturated_total = 0
        self._shed_vegas_total = 0

    @staticmethod
    def _builtin_profiles() -> list[TenantSLOProfile]:
        return [
            TenantSLOProfile(
                tenant_id="tenant-enterprise-01",
                rps_limit=120.0,
                burst_rps=240.0,
                tpm_limit=600_000.0,
                burst_tokens=120_000.0,
                default_priority=PriorityTier.CRITICAL,
                latency_slo_ms=600.0,
            ),
            TenantSLOProfile(
                tenant_id="tenant-saas-prod",
                rps_limit=60.0,
                burst_rps=120.0,
                tpm_limit=300_000.0,
                burst_tokens=60_000.0,
                default_priority=PriorityTier.INTERACTIVE,
                latency_slo_ms=850.0,
            ),
            TenantSLOProfile(
                tenant_id="tenant-batch-analytics",
                rps_limit=30.0,
                burst_rps=60.0,
                tpm_limit=180_000.0,
                burst_tokens=40_000.0,
                default_priority=PriorityTier.BATCH_BACKGROUND,
                latency_slo_ms=2500.0,
            ),
        ]

    def register_tenant(self, profile: TenantSLOProfile) -> None:
        """Register or replace a tenant's SLO & rate-limit profile."""
        self._buckets[profile.tenant_id] = DualTokenBucket(profile=profile, clock=self._clock)

    def get_or_create_bucket(self, tenant_id: str) -> DualTokenBucket:
        """Fetch tenant bucket, creating a default SLO profile for unseen tenants."""
        bucket = self._buckets.get(tenant_id)
        if bucket is None:
            profile = TenantSLOProfile(
                tenant_id=tenant_id,
                rps_limit=self._default_rps_limit,
                burst_rps=self._default_burst_rps,
                tpm_limit=self._default_tpm_limit,
                burst_tokens=self._default_burst_tokens,
            )
            bucket = DualTokenBucket(profile=profile, clock=self._clock)
            self._buckets[tenant_id] = bucket
        return bucket

    def evaluate(
        self,
        *,
        tenant_id: str,
        priority: PriorityTier,
        estimated_tokens: int,
        gateway_utilization: float,
    ) -> AdmissionDecision:
        """Evaluate global priority load shedding followed by tenant dual token-bucket limits."""
        util = max(0.0, gateway_utilization)
        bucket = self.get_or_create_bucket(tenant_id)

        # Stage 1: Gateway-wide progressive priority load shedding (>= 85% utilization)
        if util >= self.SATURATED_UTILIZATION:
            self._shed_saturated_total += 1
            return AdmissionDecision(
                allowed=False,
                reason="GATEWAY_SATURATED",
                status_code=503,
                tenant_id=tenant_id,
                priority=priority,
                gateway_utilization=round(util, 4),
                retry_after_seconds=0.50,
                rps_tokens_remaining=round(bucket.rps_tokens, 2),
                tpm_tokens_remaining=round(bucket.tpm_tokens, 2),
            )

        if util >= self.interactive_shed_threshold and priority != PriorityTier.CRITICAL:
            if priority == PriorityTier.BATCH_BACKGROUND:
                self._shed_batch_total += 1
                reason = "PRIORITY_LOAD_SHED_BATCH"
            else:
                self._shed_interactive_total += 1
                reason = "PRIORITY_LOAD_SHED_INTERACTIVE"
            return AdmissionDecision(
                allowed=False,
                reason=reason,
                status_code=503,
                tenant_id=tenant_id,
                priority=priority,
                gateway_utilization=round(util, 4),
                retry_after_seconds=0.25,
                rps_tokens_remaining=round(bucket.rps_tokens, 2),
                tpm_tokens_remaining=round(bucket.tpm_tokens, 2),
            )

        if util >= self.batch_shed_threshold and priority == PriorityTier.BATCH_BACKGROUND:
            self._shed_batch_total += 1
            return AdmissionDecision(
                allowed=False,
                reason="PRIORITY_LOAD_SHED_BATCH",
                status_code=503,
                tenant_id=tenant_id,
                priority=priority,
                gateway_utilization=round(util, 4),
                retry_after_seconds=0.20,
                rps_tokens_remaining=round(bucket.rps_tokens, 2),
                tpm_tokens_remaining=round(bucket.tpm_tokens, 2),
            )

        # Stage 2: Atomic per-tenant RPS + TPM token-bucket enforcement
        allowed, reason, retry_after = bucket.try_consume(max(1, estimated_tokens))
        if not allowed:
            if reason == "TENANT_RPS_EXCEEDED":
                self._rejected_rps_total += 1
            else:
                self._rejected_tpm_total += 1
            return AdmissionDecision(
                allowed=False,
                reason=reason,
                status_code=429,
                tenant_id=tenant_id,
                priority=priority,
                gateway_utilization=round(util, 4),
                retry_after_seconds=retry_after,
                rps_tokens_remaining=round(bucket.rps_tokens, 2),
                tpm_tokens_remaining=round(bucket.tpm_tokens, 2),
            )

        self._admitted_total += 1
        return AdmissionDecision(
            allowed=True,
            reason="ADMITTED",
            status_code=200,
            tenant_id=tenant_id,
            priority=priority,
            gateway_utilization=round(util, 4),
            retry_after_seconds=0.0,
            rps_tokens_remaining=round(bucket.rps_tokens, 2),
            tpm_tokens_remaining=round(bucket.tpm_tokens, 2),
        )

    def record_vegas_shed(self) -> None:
        """Increment counter when upstream Vegas concurrency ceilings shed a request."""
        self._shed_vegas_total += 1

    def reconcile_usage(
        self,
        *,
        tenant_id: str,
        estimated_tokens: int,
        actual_tokens: int,
    ) -> None:
        """Reconcile actual prompt+completion tokens against initial reservation."""
        bucket = self._buckets.get(tenant_id)
        if bucket is not None:
            bucket.reconcile(estimated_tokens=estimated_tokens, actual_tokens=actual_tokens)

    def summary(self) -> AdmissionSummary:
        """Return cumulative admission & priority-shedding counters."""
        return AdmissionSummary(
            admitted_total=self._admitted_total,
            rejected_rps_total=self._rejected_rps_total,
            rejected_tpm_total=self._rejected_tpm_total,
            shed_batch_total=self._shed_batch_total,
            shed_interactive_total=self._shed_interactive_total,
            shed_saturated_total=self._shed_saturated_total,
            shed_vegas_total=self._shed_vegas_total,
        )
