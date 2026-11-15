# ADR 0001: TCP-Vegas & Netflix Gradient2 Adaptive Concurrency Control

- **Status**: Accepted
- **Date**: 2026-11-15

## Context
Static semaphore limits per LLM upstream fail under real-world batching and KV-cache contention: an upstream `vLLM` replica or managed commercial API (`gemini-2.5-flash`, `claude-sonnet-4`) experiences dynamic throughput knees depending on prompt sequence lengths and co-tenant load. Unbounded gateway queueing turns transient upstream slowdowns into multi-second `p99` tail-latency collapse.

## Decision
Implement non-blocking, fail-fast adaptive concurrency limiters (`VegasAdaptiveLimiter`) governed by TCP-Vegas standing-queue estimation (`queue_estimate = inflight * (1 - rtt_min / rtt_sample)`) combined with Netflix Gradient2 multiplicative backoff on queue buildup (`queue_estimate > beta`) and explicit HTTP `429`/`503` signals, paired with gateway-level progressive priority load shedding (`BATCH_BACKGROUND` shed at `>= 85%` utilization, `INTERACTIVE` shed at `>= 95%` utilization).

## Consequences
- **Positive**: Zero manual concurrency retuning across heterogeneous providers; `CRITICAL` tier traffic maintains >97% admission under 32-way burst load while standing queues remain bounded.
- **Trade-off**: Periodic `rtt_min` upward probing (`* 1.02` every `probe_interval` low-load samples) is required so transient ultra-fast responses do not permanently bias the baseline RTT downward.
