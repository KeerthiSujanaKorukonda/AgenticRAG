"""
Evidence Agent.

Classifies each retrieved chunk as RELEVANT, WEAKLY_RELEVANT, IRRELEVANT, or
CONTRADICTORY. Deliberately does NOT rely on cosine similarity alone:
combines the hybrid retrieval score (semantic + BM25) with lexical keyword
overlap against the sub-question, and — when an NLI model is available —
checks the candidate against evidence already accepted for the same
sub-question to catch direct contradictions at the per-passage level.
"""

import re
from typing import Dict, List, Optional

from src.config import agents as agents_cfg
from src.state import EvidenceItem, RetrievedChunk

_STOPWORDS = {
    "what", "are", "the", "is", "of", "in", "for", "to", "a", "an", "and",
    "or", "how", "why", "does", "do", "which", "on", "with", "that", "this",
}


def _keywords(text: str) -> set:
    words = re.findall(r"[a-zA-Z][a-zA-Z\-]+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def _keyword_overlap_ratio(sub_question: str, chunk_text: str) -> float:
    q_kw = _keywords(sub_question)
    if not q_kw:
        return 0.0
    c_kw = _keywords(chunk_text)
    overlap = q_kw & c_kw
    return len(overlap) / len(q_kw)


def evaluate_chunk(
    chunk: RetrievedChunk,
    accepted_for_subquestion: List[EvidenceItem],
    nli_model=None,
    relevance_threshold: float = agents_cfg.evidence_relevance_threshold,
    contradiction_threshold: float = agents_cfg.contradiction_score_threshold,
) -> EvidenceItem:
    """
    Classify a single retrieved chunk. `accepted_for_subquestion` is the list
    of EvidenceItems already accepted (RELEVANT) for the same sub-question in
    this session — used only for the optional per-passage contradiction check.
    """
    sub_question = chunk.get("sub_question", "")
    overlap_ratio = _keyword_overlap_ratio(sub_question, chunk.get("text", ""))
    hybrid_score = chunk.get("hybrid_score", 0.0)

    # Combine hybrid retrieval score with lexical completeness — a passage
    # that scores well semantically but shares almost no keywords with the
    # sub-question is treated more skeptically than cosine similarity alone
    # would suggest.
    combined_score = 0.7 * hybrid_score + 0.3 * overlap_ratio

    reasons = [
        f"hybrid_score={hybrid_score:.2f}",
        f"keyword_overlap={overlap_ratio:.2f}",
    ]

    classification = "IRRELEVANT"
    if combined_score >= relevance_threshold * 1.5:
        classification = "RELEVANT"
    elif combined_score >= relevance_threshold:
        classification = "WEAKLY_RELEVANT"

    # Optional per-passage contradiction check against already-accepted
    # evidence for the same sub-question. This runs based on the semantic
    # (hybrid) score alone, independent of the keyword-overlap-adjusted
    # classification above — a passage that directly contradicts accepted
    # evidence can otherwise share very few literal keywords with the
    # sub-question (e.g. "abundant" vs "scarcity") and would wrongly get
    # filtered out as IRRELEVANT before ever being compared.
    if nli_model is not None and hybrid_score >= relevance_threshold and accepted_for_subquestion:
        for existing in accepted_for_subquestion[:3]:  # cap NLI calls for cost
            try:
                scores = nli_model.predict(existing["text"], chunk.get("text", ""))
                contradiction_score = scores.get("CONTRADICTION", 0.0)
                if contradiction_score >= contradiction_threshold:
                    classification = "CONTRADICTORY"
                    reasons.append(
                        f"NLI contradiction with existing evidence from {existing.get('title', 'another source')} "
                        f"(score={contradiction_score:.2f})"
                    )
                    break
            except Exception as exc:
                reasons.append(f"NLI check unavailable: {exc}")
                break

    return EvidenceItem(
        chunk_id=chunk.get("chunk_id", ""),
        doc_id=chunk.get("doc_id", ""),
        title=chunk.get("title", ""),
        url=chunk.get("url", ""),
        text=chunk.get("text", ""),
        year=chunk.get("year"),
        sub_question=sub_question,
        classification=classification,
        relevance_score=round(combined_score, 4),
        reasons=reasons,
    )


def evaluate_chunks(
    chunks: List[RetrievedChunk],
    nli_model=None,
) -> List[EvidenceItem]:
    """
    Evaluate all chunks, processed sub-question by sub-question so the
    per-passage contradiction check has something to compare against as
    RELEVANT evidence accumulates for that sub-question.
    """
    accepted_by_subquestion: Dict[str, List[EvidenceItem]] = {}
    results: List[EvidenceItem] = []

    for chunk in chunks:
        sq = chunk.get("sub_question", "")
        accepted = accepted_by_subquestion.get(sq, [])
        item = evaluate_chunk(chunk, accepted, nli_model=nli_model)
        results.append(item)
        if item["classification"] == "RELEVANT":
            accepted_by_subquestion.setdefault(sq, []).append(item)

    return results


def compute_evidence_coverage(
    sub_questions: List[str],
    evidence_items: List[EvidenceItem],
    sufficiency_threshold: float = agents_cfg.evidence_sufficiency_threshold,
) -> Dict:
    """
    Real coverage computation — never fabricated. For each sub-question,
    coverage = (# RELEVANT evidence items) / (a target count), capped at 1.0.
    The target count (3) is a simple, documented heuristic: "at least a
    few independent relevant passages" rather than a single hit.
    """
    TARGET_RELEVANT_PER_SUBQ = 3
    per_subquestion = {}
    missing = []

    for sq in sub_questions:
        relevant = [
            e for e in evidence_items
            if e.get("sub_question") == sq and e.get("classification") == "RELEVANT"
        ]
        weak = [
            e for e in evidence_items
            if e.get("sub_question") == sq and e.get("classification") == "WEAKLY_RELEVANT"
        ]
        coverage_score = min(1.0, len(relevant) / TARGET_RELEVANT_PER_SUBQ)
        per_subquestion[sq] = {
            "relevant_count": len(relevant),
            "weakly_relevant_count": len(weak),
            "coverage_score": round(coverage_score, 2),
            "sufficient": coverage_score >= sufficiency_threshold,
        }
        if coverage_score < sufficiency_threshold:
            missing.append(sq)

    overall = (
        round(sum(v["coverage_score"] for v in per_subquestion.values()) / len(per_subquestion), 2)
        if per_subquestion else 0.0
    )

    return {
        "per_subquestion": per_subquestion,
        "overall_coverage_score": overall,
        "missing_subquestions": missing,
        "is_sufficient": len(missing) == 0,
    }
