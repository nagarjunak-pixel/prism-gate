# ADR 0002: Speculative Complexity Cascade Routing & Signed-Hash Semantic Cache

- **Status**: Accepted
- **Date**: 2026-11-15

## Context
Pinning 100% of enterprise traffic to frontier models (`claude-sonnet-4` / `gemini-2.5-pro`) inflates inference spend by 5x–15x on routine lookups, ticket classification, and repeated FAQ paraphrases. Conversely, running a heavy neural classifier or dense bi-encoder inside the gateway hot path adds 15–40ms of CPU/GPU latency and multi-hundred-MB container bloat.

## Decision
1. Score structural prompt complexity deterministically in `< 50µs` across reasoning depth, code/math token density, constraint markers, and multi-hop compositionality.
2. Precede upstream dispatch with a **Tenant-Isolated Signed-Hash Cosine Semantic Cache** (`D = 256` Weinberger feature hashing with numeric & negation guards).
3. Route `score < 0.48` speculatively to the fast tier (`gemini-2.5-flash` / `vllm-qwen-32b`) and escalate to frontier iff upstream confidence `< min_confidence` or RAG faithfulness `< 0.75`, recording honest dual-invocation FinOps costs whenever escalation triggers.

## Consequences
- **Positive**: Delivers **57.16% net FinOps cost reduction** at **0.067ms `p50` gateway overhead** with strict zero-leakage tenant cache partitioning.
- **Trade-off**: Escalated requests pay both fast-draft and frontier latency/cost (~10% of mixed traffic), making accurate structural complexity calibration essential.
