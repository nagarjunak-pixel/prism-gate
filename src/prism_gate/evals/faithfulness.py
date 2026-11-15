"""Online Claim-Level Groundedness & Hallucination Judge.

Decomposes model completions into atomic factual claims and verifies each claim
against retrieved RAG context chunks along four axes:
1. Content-token & lexical n-gram entailment recall.
2. Canonicalized numeric/unit/dosage/currency alignment (`$12.4M`, `500mg`, `14.2%`).
3. Domain named-entity & acronym grounding (`Warfarin`, `GAAP`, `EBITDA`, `BRCA1`).
4. Polarity & clinical/financial antonym contradiction detection.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from prism_gate.types import ClaimVerdict, ClaimVerification, FaithfulnessReport

_ABBREV_PLACEHOLDER = "\x00"
_ABBREV_RE = re.compile(r"\b(Dr|Mr|Mrs|Ms|Prof|Inc|Corp|Ltd|vs|etc|approx|est)\.", re.IGNORECASE)
_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+\s+|\n+|;\s+")
_CLAUSE_SPLIT_RE = re.compile(
    r",\s+(?:while|whereas|however|although|and\s+reported|and\s+prescribed)\s+",
    re.IGNORECASE,
)

# Captures numbers with optional $, commas, decimals, ratios, and financial/clinical units
_NUMERIC_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9])\$?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?:\s?(?:%|mg|mcg|g|kg|ml|m|b|k|x|bps))?(?![A-Za-z0-9])"
    r"|(?<![A-Za-z0-9])\$?\d+(?:\.\d+)?(?:\s?(?:%|mg|mcg|g|kg|ml|m|b|k|x|bps))?(?![A-Za-z0-9])"
    r"|\bQ[1-4]\b",
    re.IGNORECASE,
)

_ENTITY_TOKEN_RE = re.compile(r"\b(?:[A-Z]{2,}[0-9]*|[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*)\b")
_WORD_RE = re.compile(r"[a-z0-9]+")

_FILLER_PREFIXES: tuple[str, ...] = (
    "sure,",
    "certainly,",
    "here is",
    "here's",
    "let me help",
    "i would be happy",
)

_GENERIC_CAPITALIZED: frozenset[str] = frozenset(
    {
        "the",
        "this",
        "that",
        "these",
        "those",
        "based",
        "according",
        "overall",
        "however",
        "during",
        "therefore",
        "moreover",
        "furthermore",
        "summary",
        "total",
        "revenue",
        "income",
        "margin",
        "quarter",
        "annual",
        "fiscal",
        "patient",
        "patients",
        "clinical",
        "dosage",
        "treatment",
        "therapy",
        "daily",
        "oral",
        "table",
        "section",
        "context",
        "report",
        "company",
        "yes",
        "no",
        "for",
        "with",
        "from",
        "after",
        "before",
        "while",
        "when",
        "where",
    }
)

_CONTENT_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "to",
        "of",
        "in",
        "on",
        "at",
        "by",
        "for",
        "with",
        "as",
        "from",
        "that",
        "this",
        "it",
        "its",
        "and",
        "or",
        "but",
        "into",
        "per",
        "their",
        "has",
        "have",
        "had",
        "achieved",
        "reported",
        "recorded",
        "showed",
        "stated",
        "noted",
    }
)

_NEGATION_WORDS: frozenset[str] = frozenset(
    {"not", "never", "no", "none", "neither", "without", "contraindicated", "prohibited"}
)

_ANTONYM_PAIRS: tuple[tuple[str, str], ...] = (
    ("increased", "decreased"),
    ("increase", "decrease"),
    ("grew", "declined"),
    ("growth", "decline"),
    ("rose", "fell"),
    ("higher", "lower"),
    ("approved", "rejected"),
    ("approved", "unapproved"),
    ("recommended", "contraindicated"),
    ("indicated", "contraindicated"),
    ("safe", "unsafe"),
    ("benign", "malignant"),
    ("exceeded", "missed"),
    ("surplus", "deficit"),
)


def _stem_light(token: str) -> str:
    """Lightweight deterministic suffix normalization for lexical entailment."""
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 4 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


class ClaimFaithfulnessJudge:
    """Deterministic sub-millisecond RAG groundedness & hallucination verifier."""

    def __init__(self, *, support_threshold: float = 0.50) -> None:
        if not (0.1 <= support_threshold <= 1.0):
            raise ValueError("support_threshold must be in [0.1, 1.0]")
        self.support_threshold = support_threshold

    def split_atomic_claims(self, completion: str) -> list[str]:
        """Split a model completion into atomic factual assertion clauses."""
        cleaned = re.sub(r"```[\s\S]*?```", " ", completion)
        cleaned = _ABBREV_RE.sub(lambda m: f"{m.group(1)}{_ABBREV_PLACEHOLDER}", cleaned)

        raw_sentences = _SENTENCE_SPLIT_RE.split(cleaned)
        claims: list[str] = []
        for sent in raw_sentences:
            restored = sent.replace(_ABBREV_PLACEHOLDER, ".").strip()
            restored = re.sub(r"^[-*•\d.)\s]+", "", restored).strip()
            if not restored:
                continue
            lower = restored.lower()
            if (
                any(lower.startswith(prefix) for prefix in _FILLER_PREFIXES)
                and len(restored.split()) < 8
            ):
                continue

            sub_clauses = _CLAUSE_SPLIT_RE.split(restored)
            for clause in sub_clauses:
                item = clause.strip(" ,.;:-")
                if len(_WORD_RE.findall(item.lower())) >= 3:
                    claims.append(item)
        return claims

    @staticmethod
    def extract_numbers(text: str) -> list[tuple[str, str]]:
        """Extract `(raw_surface, canonical_key)` numeric & unit anchors."""
        results: list[tuple[str, str]] = []
        for match in _NUMERIC_TOKEN_RE.finditer(text):
            raw = match.group(0).strip()
            norm = raw.lower().replace(",", "").replace("$", "").replace(" ", "")
            # Normalize trailing decimal zeros like 12.40m -> 12.4m or 12.0 -> 12
            num_unit = re.match(r"^(\d+\.\d+)([a-z%]*)$", norm)
            if num_unit:
                val_str = num_unit.group(1).rstrip("0").rstrip(".")
                norm = f"{val_str}{num_unit.group(2)}"
            results.append((raw, norm))
        return results

    @staticmethod
    def extract_entities(text: str) -> list[str]:
        """Extract domain named entities, drug names, tickers, and proper nouns."""
        entities: list[str] = []
        seen: set[str] = set()
        stripped_text = text.lstrip()
        lead_offset = len(text) - len(stripped_text)
        for match in _ENTITY_TOKEN_RE.finditer(text):
            raw = match.group(0).strip()
            parts = [p for p in raw.split() if p.lower() not in _GENERIC_CAPITALIZED]
            if not parts:
                continue
            cleaned = " ".join(parts)
            # Ignore clause-leading capitalized adverbs (e.g. "Completely ...")
            if (
                match.start() == lead_offset
                and len(parts) == 1
                and not cleaned.isupper()
                and cleaned.endswith("ly")
            ):
                continue
            # Ignore pure quarter markers handled by numeric extractor
            if re.fullmatch(r"Q[1-4]", cleaned):
                continue
            key = cleaned.lower()
            if len(key) >= 2 and key not in seen:
                seen.add(key)
                entities.append(cleaned)
        return entities

    @staticmethod
    def _content_tokens(text: str) -> set[str]:
        return {
            _stem_light(tok)
            for tok in _WORD_RE.findall(text.lower())
            if tok not in _CONTENT_STOPWORDS and len(tok) > 1
        }

    @staticmethod
    def _bigrams(tokens_ordered: list[str]) -> set[tuple[str, str]]:
        if len(tokens_ordered) < 2:
            return set()
        return {(tokens_ordered[i], tokens_ordered[i + 1]) for i in range(len(tokens_ordered) - 1)}

    @staticmethod
    def _has_polarity_conflict(claim_text: str, chunk_text: str) -> bool:
        claim_words = set(_WORD_RE.findall(claim_text.lower()))
        chunk_words = set(_WORD_RE.findall(chunk_text.lower()))

        for pos, neg in _ANTONYM_PAIRS:
            if (pos in claim_words and neg in chunk_words and pos not in chunk_words) or (
                neg in claim_words and pos in chunk_words and neg not in chunk_words
            ):
                return True

        claim_neg = bool(claim_words & _NEGATION_WORDS)
        chunk_neg = bool(chunk_words & _NEGATION_WORDS)
        return claim_neg != chunk_neg

    def evaluate(
        self,
        *,
        completion: str,
        context_chunks: Sequence[str],
    ) -> FaithfulnessReport:
        """Verify every atomic claim in `completion` against `context_chunks`."""
        claims = self.split_atomic_claims(completion)
        if not claims:
            return FaithfulnessReport(
                faithfulness_score=1.0,
                total_claims=0,
                supported_claims_count=0,
                hallucination_detected=False,
            )

        if not context_chunks or all(not c.strip() for c in context_chunks):
            verifications = tuple(
                ClaimVerification(
                    claim_text=c,
                    verdict=ClaimVerdict.UNSUPPORTED,
                    entailment_score=0.0,
                    unmatched_numbers=tuple(raw for raw, _ in self.extract_numbers(c)),
                    unmatched_entities=tuple(self.extract_entities(c)),
                    reason="No retrieved RAG context chunks supplied.",
                )
                for c in claims
            )
            all_nums = tuple(dict.fromkeys(n for v in verifications for n in v.unmatched_numbers))
            all_ents = tuple(dict.fromkeys(e for v in verifications for e in v.unmatched_entities))
            return FaithfulnessReport(
                faithfulness_score=0.0,
                total_claims=len(claims),
                supported_claims_count=0,
                unsupported_claims=tuple(claims),
                contradicted_claims=(),
                flagged_numbers=all_nums,
                flagged_entities=all_ents,
                claim_verifications=verifications,
                hallucination_detected=True,
            )

        chunk_tokens = [self._content_tokens(ch) for ch in context_chunks]
        chunk_ordered = [
            [_stem_light(w) for w in _WORD_RE.findall(ch.lower()) if w not in _CONTENT_STOPWORDS]
            for ch in context_chunks
        ]
        chunk_bigrams = [self._bigrams(seq) for seq in chunk_ordered]
        chunk_numbers = [{norm for _, norm in self.extract_numbers(ch)} for ch in context_chunks]
        all_context_numbers: set[str] = set().union(*chunk_numbers)
        all_context_lower = " ".join(context_chunks).lower()

        verifications_list: list[ClaimVerification] = []
        unsupported_list: list[str] = []
        contradicted_list: list[str] = []
        flagged_nums_list: list[str] = []
        flagged_ents_list: list[str] = []
        supported_count = 0

        for claim in claims:
            c_tokens = self._content_tokens(claim)
            c_ordered = [
                _stem_light(w)
                for w in _WORD_RE.findall(claim.lower())
                if w not in _CONTENT_STOPWORDS
            ]
            c_bigrams = self._bigrams(c_ordered)

            best_idx = 0
            best_recall = 0.0
            best_bigram_recall = 0.0

            for idx, ch_toks in enumerate(chunk_tokens):
                recall = len(c_tokens & ch_toks) / max(1, len(c_tokens))
                bg_recall = (
                    len(c_bigrams & chunk_bigrams[idx]) / len(c_bigrams) if c_bigrams else recall
                )
                combined = 0.8 * recall + 0.2 * bg_recall
                if combined > (0.8 * best_recall + 0.2 * best_bigram_recall):
                    best_idx = idx
                    best_recall = recall
                    best_bigram_recall = bg_recall

            c_nums = self.extract_numbers(claim)
            matched_nums: list[str] = []
            unmatched_nums: list[str] = []
            for raw_num, norm_num in c_nums:
                if norm_num in all_context_numbers:
                    matched_nums.append(raw_num)
                else:
                    unmatched_nums.append(raw_num)
                    if raw_num not in flagged_nums_list:
                        flagged_nums_list.append(raw_num)

            c_ents = self.extract_entities(claim)
            unmatched_ents: list[str] = []
            for ent in c_ents:
                if ent.lower() not in all_context_lower:
                    unmatched_ents.append(ent)
                    if ent not in flagged_ents_list:
                        flagged_ents_list.append(ent)

            entity_recall = (len(c_ents) - len(unmatched_ents)) / len(c_ents) if c_ents else 1.0

            # Hierarchy: Numeric hallucination -> Polarity conflict -> Entity/Lexical entailment
            if unmatched_nums:
                non_numeric_tokens = {t for t in c_tokens if not any(ch.isdigit() for ch in t)}
                topic_overlap = len(non_numeric_tokens & chunk_tokens[best_idx]) / max(
                    1, len(non_numeric_tokens)
                )
                if topic_overlap >= 0.30 and chunk_numbers[best_idx]:
                    verdict = ClaimVerdict.CONTRADICTED
                    score = round(min(0.15, best_recall * 0.2), 4)
                    reason = (
                        f"Numeric contradiction against chunk #{best_idx}: "
                        f"unverified value(s) {', '.join(unmatched_nums)}."
                    )
                    contradicted_list.append(claim)
                else:
                    verdict = ClaimVerdict.UNSUPPORTED
                    score = round(min(0.20, best_recall * 0.25), 4)
                    reason = (
                        f"Unsupported quantitative claim: {', '.join(unmatched_nums)} "
                        "absent from context."
                    )
                unsupported_list.append(claim)
            elif best_recall >= 0.38 and self._has_polarity_conflict(
                claim, context_chunks[best_idx]
            ):
                verdict = ClaimVerdict.CONTRADICTED
                score = round(min(0.15, best_recall * 0.2), 4)
                reason = (
                    f"Semantic polarity / clinical antonym contradiction with chunk #{best_idx}."
                )
                contradicted_list.append(claim)
                unsupported_list.append(claim)
            elif unmatched_ents and best_recall < 0.70:
                verdict = ClaimVerdict.UNSUPPORTED
                score = round(min(0.30, best_recall * 0.45), 4)
                reason = f"Unverified named entity/identifier(s): {', '.join(unmatched_ents)}."
                unsupported_list.append(claim)
            else:
                raw_score = 0.70 * best_recall + 0.15 * best_bigram_recall + 0.15 * entity_recall
                if matched_nums and not unmatched_nums and not unmatched_ents:
                    raw_score = min(1.0, raw_score + 0.12)
                score = round(min(1.0, max(0.0, raw_score)), 4)
                if score >= self.support_threshold:
                    verdict = ClaimVerdict.SUPPORTED
                    supported_count += 1
                    reason = f"Entailed by context chunk #{best_idx}."
                else:
                    verdict = ClaimVerdict.UNSUPPORTED
                    reason = "Insufficient lexical/n-gram entailment in retrieved context."
                    unsupported_list.append(claim)

            verifications_list.append(
                ClaimVerification(
                    claim_text=claim,
                    verdict=verdict,
                    entailment_score=score,
                    matched_numbers=tuple(matched_nums),
                    unmatched_numbers=tuple(unmatched_nums),
                    unmatched_entities=tuple(unmatched_ents),
                    supporting_chunk_index=best_idx if verdict == ClaimVerdict.SUPPORTED else None,
                    reason=reason,
                )
            )

        support_ratio = supported_count / len(claims)
        mean_entailment = sum(v.entailment_score for v in verifications_list) / len(
            verifications_list
        )
        if supported_count == len(claims):
            overall_score = round(
                min(1.0, max(0.80, 0.65 * support_ratio + 0.35 * mean_entailment)), 4
            )
        elif supported_count == 0:
            overall_score = round(min(0.25, mean_entailment * 0.5), 4)
        else:
            overall_score = round(
                min(1.0, max(0.0, 0.75 * support_ratio + 0.25 * mean_entailment)), 4
            )

        return FaithfulnessReport(
            faithfulness_score=overall_score,
            total_claims=len(claims),
            supported_claims_count=supported_count,
            unsupported_claims=tuple(unsupported_list),
            contradicted_claims=tuple(contradicted_list),
            flagged_numbers=tuple(flagged_nums_list),
            flagged_entities=tuple(flagged_ents_list),
            claim_verifications=tuple(verifications_list),
            hallucination_detected=bool(unsupported_list),
        )
