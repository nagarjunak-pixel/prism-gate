# ADR 0004: Online Atomic Claim-Faithfulness Verification for RAG

- **Status**: Accepted
- **Date**: 2026-11-15

## Context
In financial and healthcare RAG pipelines, a single hallucinated dosage (`500mg` instead of `5mg` Warfarin) or misquoted filing figure (`$18.9M` instead of `$12.4M` Q3 revenue) creates severe compliance and safety liability—even when 90% of surrounding prose matches retrieved chunks so coarse document-level cosine similarity scores remain deceptively high.

## Decision
Implement an online `ClaimFaithfulnessJudge` (`src/prism_gate/evals/faithfulness.py`) that:
1. Splits completions into atomic factual clauses while preserving abbreviations and decimal literals.
2. Canonicalizes numeric/currency/dosage anchors (`$1,250,000` == `$1250000`, `5 mg` == `5mg`, `500mg` != `5mg`) and verifies exact quantitative grounding against retrieved chunks.
3. Checks domain named-entity grounding and clinical/financial antonym polarity (`recommended` vs `contraindicated`, `increased` vs `decreased`).
4. Blocks any completion with `hallucination_detected == True` from entering the semantic cache and triggers speculative frontier escalation when fast-tier RAG faithfulness `< 0.75`.

## Consequences
- **Positive**: Deterministic sub-millisecond detection of quantitative, entity, and polarity hallucinations on the live gateway path without secondary LLM judge latency.
- **Trade-off**: Lexical + anchor verification is intentionally conservative on abstract abstractive synthesis that introduces novel terminology absent from context chunks.
