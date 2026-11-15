"""Unit and Hypothesis tests for `HierarchicalAdmissionController` & `DualTokenBucket`."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from prism_gate.admission.token_bucket import DualTokenBucket, HierarchicalAdmissionController
from prism_gate.types import PriorityTier, TenantSLOProfile


def test_admission_threshold_validation_and_bucket_guards() -> None:
    with pytest.raises(ValueError, match="batch_shed_threshold"):
        HierarchicalAdmissionController(batch_shed_threshold=0.95, interactive_shed_threshold=0.85)

    profile = TenantSLOProfile(
        tenant_id="t-err",
        rps_limit=2.0,
        burst_rps=2.0,
        tpm_limit=600.0,
        burst_tokens=100.0,
    )
    bucket = DualTokenBucket(profile=profile, clock=lambda: 0.0)
    with pytest.raises(ValueError, match="estimated_tokens"):
        bucket.try_consume(0)

    # Request larger than burst_tokens fails without deducting RPS token
    allowed, reason, _ = bucket.try_consume(250)
    assert not allowed
    assert reason == "TENANT_TPM_BURST_EXCEEDED"
    assert bucket.rps_tokens == pytest.approx(2.0)


def test_dual_token_bucket_rps_tpm_refill_and_reconcile() -> None:
    now = 0.0

    def clock() -> float:
        return now

    profile = TenantSLOProfile(
        tenant_id="tenant-alpha",
        rps_limit=2.0,
        burst_rps=2.0,
        tpm_limit=600.0,  # 10 tokens/sec
        burst_tokens=50.0,
    )
    ctrl = HierarchicalAdmissionController(profiles=[profile], clock=clock)

    # Consume 2 requests (40 tokens total) -> RPS exhausted
    d1 = ctrl.evaluate(
        tenant_id="tenant-alpha",
        priority=PriorityTier.INTERACTIVE,
        estimated_tokens=20,
        gateway_utilization=0.3,
    )
    d2 = ctrl.evaluate(
        tenant_id="tenant-alpha",
        priority=PriorityTier.INTERACTIVE,
        estimated_tokens=20,
        gateway_utilization=0.3,
    )
    assert d1.allowed and d2.allowed

    d3 = ctrl.evaluate(
        tenant_id="tenant-alpha",
        priority=PriorityTier.INTERACTIVE,
        estimated_tokens=5,
        gateway_utilization=0.3,
    )
    assert not d3.allowed
    assert d3.status_code == 429
    assert d3.reason == "TENANT_RPS_EXCEEDED"

    # Advance 1.0s -> +2 RPS tokens, +10 TPM tokens (total TPM = 20). Ask for 35 TPM -> TPM fail!
    now = 1.0
    d4 = ctrl.evaluate(
        tenant_id="tenant-alpha",
        priority=PriorityTier.INTERACTIVE,
        estimated_tokens=35,
        gateway_utilization=0.3,
    )
    assert not d4.allowed
    assert d4.status_code == 429
    assert d4.reason == "TENANT_TPM_EXCEEDED"

    # Reconcile refunds unused tokens, and excess actual tokens deduct cleanly
    ctrl.reconcile_usage(tenant_id="tenant-alpha", estimated_tokens=30, actual_tokens=10)
    assert ctrl.get_or_create_bucket("tenant-alpha").tpm_tokens == pytest.approx(40.0)
    ctrl.reconcile_usage(tenant_id="tenant-alpha", estimated_tokens=10, actual_tokens=25)
    assert ctrl.get_or_create_bucket("tenant-alpha").tpm_tokens == pytest.approx(25.0)


def test_progressive_priority_load_shedding_above_85_percent() -> None:
    ctrl = HierarchicalAdmissionController()

    # At 84% utilization -> BATCH_BACKGROUND is admitted
    ok_batch = ctrl.evaluate(
        tenant_id="new-unseen-tenant",
        priority=PriorityTier.BATCH_BACKGROUND,
        estimated_tokens=32,
        gateway_utilization=0.84,
    )
    assert ok_batch.allowed

    # At 86% utilization (>= 85%) -> BATCH_BACKGROUND shed, INTERACTIVE & CRITICAL admitted
    shed_batch = ctrl.evaluate(
        tenant_id="tenant-enterprise-01",
        priority=PriorityTier.BATCH_BACKGROUND,
        estimated_tokens=32,
        gateway_utilization=0.86,
    )
    assert not shed_batch.allowed
    assert shed_batch.status_code == 503
    assert shed_batch.reason == "PRIORITY_LOAD_SHED_BATCH"

    ok_inter = ctrl.evaluate(
        tenant_id="tenant-enterprise-01",
        priority=PriorityTier.INTERACTIVE,
        estimated_tokens=32,
        gateway_utilization=0.88,
    )
    assert ok_inter.allowed

    # At 96% utilization (>= 95%) -> INTERACTIVE & BATCH shed, CRITICAL preserved!
    shed_inter = ctrl.evaluate(
        tenant_id="tenant-enterprise-01",
        priority=PriorityTier.INTERACTIVE,
        estimated_tokens=32,
        gateway_utilization=0.96,
    )
    shed_batch_high = ctrl.evaluate(
        tenant_id="tenant-enterprise-01",
        priority=PriorityTier.BATCH_BACKGROUND,
        estimated_tokens=32,
        gateway_utilization=0.96,
    )
    ok_crit = ctrl.evaluate(
        tenant_id="tenant-enterprise-01",
        priority=PriorityTier.CRITICAL,
        estimated_tokens=32,
        gateway_utilization=0.98,
    )
    assert not shed_inter.allowed and shed_inter.reason == "PRIORITY_LOAD_SHED_INTERACTIVE"
    assert not shed_batch_high.allowed and shed_batch_high.reason == "PRIORITY_LOAD_SHED_BATCH"
    assert ok_crit.allowed

    # At 100% utilization -> all tiers shed
    saturated = ctrl.evaluate(
        tenant_id="tenant-enterprise-01",
        priority=PriorityTier.CRITICAL,
        estimated_tokens=32,
        gateway_utilization=1.0,
    )
    assert not saturated.allowed and saturated.reason == "GATEWAY_SATURATED"

    summary = ctrl.summary()
    assert summary.shed_batch_total == 2
    assert summary.shed_interactive_total == 1
    assert summary.shed_saturated_total == 1


@settings(max_examples=60, deadline=None)
@given(
    steps=st.lists(
        st.tuples(
            st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False),
            st.integers(min_value=1, max_value=500),
        ),
        min_size=1,
        max_size=30,
    )
)
def test_hypothesis_dual_bucket_conservation_bounds(steps: list[tuple[float, int]]) -> None:
    t = 0.0

    def clock() -> float:
        return t

    profile = TenantSLOProfile(
        tenant_id="prop-t",
        rps_limit=10.0,
        burst_rps=25.0,
        tpm_limit=12_000.0,
        burst_tokens=300.0,
    )
    bucket = DualTokenBucket(profile=profile, clock=clock)
    for dt, req_tokens in steps:
        t += dt
        bucket.try_consume(req_tokens)
        assert 0.0 <= bucket.rps_tokens <= profile.burst_rps + 1e-9
        assert 0.0 <= bucket.tpm_tokens <= profile.burst_tokens + 1e-9
