"""
Contradiction Detection Agent.

Runs actual NLI comparisons between pairs of RELEVANT/CONTRADICTORY evidence
items grouped by sub-question, and for confirmed contradictions attempts a
deterministic, evidence-grounded guess at *why* the sources disagree (e.g.
different years, different reported metrics) by inspecting their metadata
and text — never inventing a reason unrelated to what's actually in the
evidence.
"""

import re
from itertools import combinations
from typing import List, Optional

from src.config import agents as agents_cfg
from src.state import Contradiction, EvidenceItem

_METRIC_WORDS = ("accuracy", "f1", "bleu", "rouge", "precision", "recall", "score")
_LANGUAGE_HINT_RE = re.compile(
    r"\b(english|chinese|spanish|french|german|hindi|arabic|multilingual|low-resource|cross-lingual)\b",
    re.I,
)


def _infer_disagreement_reason(a: EvidenceItem, b: EvidenceItem) -> str:
    reasons = []

    year_a, year_b = a.get("year"), b.get("year")
    if year_a and year_b and year_a != year_b:
        reasons.append(f"published in different years ({year_a} vs {year_b})")

    metrics_a = {w for w in _METRIC_WORDS if w in a.get("text", "").lower()}
    metrics_b = {w for w in _METRIC_WORDS if w in b.get("text", "").lower()}
    if metrics_a and metrics_b and metrics_a != metrics_b:
        reasons.append(
            f"different evaluation metrics referenced ({', '.join(metrics_a)} vs {', '.join(metrics_b)})"
        )

    langs_a = set(m.lower() for m in _LANGUAGE_HINT_RE.findall(a.get("text", "")))
    langs_b = set(m.lower() for m in _LANGUAGE_HINT_RE.findall(b.get("text", "")))
    if langs_a and langs_b and langs_a != langs_b:
        reasons.append(f"different language settings discussed ({', '.join(langs_a)} vs {', '.join(langs_b)})")

    if not reasons:
        reasons.append("differing experimental settings or scope (not further specified in the retrieved text)")

    return "; ".join(reasons)


def detect_contradictions(
    evidence_items: List[EvidenceItem],
    nli_model=None,
    contradiction_threshold: float = agents_cfg.contradiction_score_threshold,
    relevance_floor: float = agents_cfg.evidence_relevance_threshold,
    max_total_pairs: int = 40,
) -> List[Contradiction]:
    """
    Compares pairs of evidence whose underlying relevance_score clears
    `relevance_floor`, across the WHOLE evidence set — not just within a
    single sub-question, and not gated by the keyword-overlap-influenced
    RELEVANT/WEAKLY_RELEVANT label. A passage that flatly contradicts
    accepted evidence (e.g. "data is abundant" vs "data is scarce") can
    share very few literal keywords with the sub-question wording and would
    otherwise be filtered out as IRRELEVANT before ever being compared —
    using the numeric relevance_score directly avoids that failure mode.
    Different papers only (same-doc_id pairs are never inter-source
    contradictions). Pair count is capped for cost. Returns only pairs the
    NLI model actually scores as CONTRADICTION above threshold — never
    merges or guesses at contradictions without a real model check.
    """
    if nli_model is None:
        return []

    usable = [
        item for item in evidence_items
        if item.get("relevance_score", 0.0) >= relevance_floor
        or item.get("classification") == "CONTRADICTORY"
    ]
    if len(usable) < 2:
        return []

    pairs = [
        (a, b) for a, b in combinations(usable, 2)
        if a.get("doc_id") != b.get("doc_id")
    ][:max_total_pairs]

    contradictions: List[Contradiction] = []
    for a, b in pairs:
        try:
            scores = nli_model.predict(a["text"], b["text"])
        except Exception:
            continue
        contradiction_score = scores.get("CONTRADICTION", 0.0)
        if contradiction_score >= contradiction_threshold:
            contradictions.append(
                Contradiction(
                    statement_a=a["text"][:300],
                    source_a=a.get("title", ""),
                    statement_b=b["text"][:300],
                    source_b=b.get("title", ""),
                    nli_label="CONTRADICTION",
                    nli_score=round(contradiction_score, 3),
                    likely_reason=_infer_disagreement_reason(a, b),
                )
            )

    return contradictions
