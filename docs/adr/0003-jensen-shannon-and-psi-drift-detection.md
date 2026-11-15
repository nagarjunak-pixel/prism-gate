# ADR 0003: Dual PSI & Base-2 Jensen-Shannon Distribution-Shift Plane

- **Status**: Accepted
- **Date**: 2026-11-15

## Context
LLM gateway regressions rarely surface first as HTTP 500s: prompt-injection surges, upstream prompt-template bugs, and silent tenant workload migrations manifest as shifts in prompt token length, complexity score, latency quantiles, and vocabulary/topic mix. Raw Kullback-Leibler divergence (`KL(P || Q)`) diverges to `+\infty` whenever a live token is absent from reference traffic.

## Decision
Combine two complementary information-theoretic detectors in `src/prism_gate/drift/detectors.py`:
1. **Population Stability Index (`PSI`)** over decile-binned continuous telemetry (`prompt_length_tokens`, `complexity_score`, `upstream_latency_ms`) with symmetric proportion smoothing (`0.10` Moderate, `0.25` Critical).
2. **Base-2 Jensen-Shannon Divergence (`JSD`)** over discrete token and topic distributions, guaranteeing exact symmetry (`JSD(P, Q) == JSD(Q, P)`), zero smoothing bias (`JSD(P, P) == 0.0`), and strict `[0.0, 1.0]` bit bounds even on disjoint vocabularies.

## Consequences
- **Positive**: Parameter-free, bounded `[0, 1]` discrete shift metric paired with industry-standard continuous PSI thresholds, verified across hundreds of randomized distributions via `Hypothesis`.
- **Trade-off**: Sliding windows require bounded memory (`deque(maxlen=400)`), trading infinite historical retention for predictable $O(1)$ per-request memory footprint.
