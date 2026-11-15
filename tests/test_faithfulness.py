"""Unit tests for `ClaimFaithfulnessJudge` (financial & clinical RAG hallucination detection)."""

from __future__ import annotations

import pytest

from prism_gate.evals.faithfulness import ClaimFaithfulnessJudge
from prism_gate.types import ClaimVerdict


def test_faithfulness_threshold_validation_and_empty_inputs() -> None:
    with pytest.raises(ValueError, match="support_threshold"):
        ClaimFaithfulnessJudge(support_threshold=0.02)

    judge = ClaimFaithfulnessJudge()
    empty_rep = judge.evaluate(completion="Sure, here is that!", context_chunks=["chunk"])
    assert empty_rep.total_claims == 0
    assert empty_rep.faithfulness_score == 1.0
    assert not empty_rep.hallucination_detected

    no_ctx = judge.evaluate(
        completion="Acme Corp reported $12.4M revenue in Q3 2025.",
        context_chunks=[],
    )
    assert no_ctx.faithfulness_score == 0.0
    assert no_ctx.hallucination_detected
    assert "$12.4M" in no_ctx.flagged_numbers
    assert "Acme Corp" in no_ctx.flagged_entities


def test_grounded_financial_and_formatted_numeric_alignment() -> None:
    judge = ClaimFaithfulnessJudge()
    chunks = [
        (
            "Dr. Patel at Acme Corp reported Q3 2025 revenue of $1,250,000 "
            "and 14.20% operating margin."
        ),
        "Warfarin maintenance therapy is recommended at 5 mg daily oral dosage.",
    ]
    completion = (
        "Dr. Patel at Acme Corp reported Q3 2025 revenue of $1250000 with 14.2% operating margin. "
        "Warfarin maintenance therapy is recommended at 5mg daily oral dosage."
    )
    report = judge.evaluate(completion=completion, context_chunks=chunks)
    assert not report.hallucination_detected
    assert report.supported_claims_count == 2
    assert report.faithfulness_score >= 0.85


def test_catches_numeric_dosage_entity_and_polarity_hallucinations() -> None:
    judge = ClaimFaithfulnessJudge()
    chunks = [
        "Acme Corp reported Q3 2025 revenue of $12.4M, while cloud revenue increased YoY.",
        "Warfarin maintenance therapy is recommended at 5mg daily oral dosage.",
    ]

    # 1. Numeric contradiction ($18.9M vs $12.4M and 500mg vs 5mg)
    num_halluc = (
        "Acme Corp reported Q3 2025 revenue of $18.9M, while Warfarin maintenance "
        "therapy is recommended at 500mg daily oral dosage."
    )
    rep_num = judge.evaluate(completion=num_halluc, context_chunks=chunks)
    assert rep_num.hallucination_detected
    assert "$18.9M" in rep_num.flagged_numbers
    assert "500mg" in rep_num.flagged_numbers
    assert len(rep_num.contradicted_claims) == 2

    # 2. Polarity / antonym contradiction (contraindicated vs recommended; decreased vs increased)
    pol_halluc = "Warfarin maintenance therapy is contraindicated at 5mg daily oral dosage."
    rep_pol = judge.evaluate(completion=pol_halluc, context_chunks=chunks)
    assert rep_pol.hallucination_detected
    assert rep_pol.claim_verifications[0].verdict == ClaimVerdict.CONTRADICTED

    # 3. Unverified entity & completely unsupported quantitative claim
    ent_halluc = (
        "Hyperion BioLabs acquired BRCA99 therapeutics during fiscal expansion. "
        "Unrelated satellite launches reached 42 orbits."
    )
    rep_ent = judge.evaluate(completion=ent_halluc, context_chunks=chunks)
    assert rep_ent.hallucination_detected
    assert rep_ent.supported_claims_count == 0
    assert any("Hyperion" in e for e in rep_ent.flagged_entities)
    assert "42" in rep_ent.flagged_numbers


def test_partial_faithfulness_and_suffix_stemming() -> None:
    judge = ClaimFaithfulnessJudge()
    chunks = [
        "Cloud platforms processed streaming transactions during Q4.",
    ]
    mixed = (
        "Cloud platform processing streamed transaction during Q4. "
        "Completely unrelated moonbase construction started yesterday."
    )
    rep = judge.evaluate(completion=mixed, context_chunks=chunks)
    assert rep.total_claims == 2
    assert rep.supported_claims_count == 1
    assert 0.25 < rep.faithfulness_score < 0.85
    assert rep.hallucination_detected
