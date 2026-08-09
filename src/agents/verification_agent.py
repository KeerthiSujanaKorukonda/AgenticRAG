"""
Verification Agent.

Checks the final synthesized claims against their cited evidence using the
NLI model: for each claim/citation pair, verifies that the citation's text
actually entails (supports) the claim, rather than merely being topically
related. Also re-checks whether any contradictory evidence undermines the
claim. This is the last line of defense against hallucination before a
report is shown to the user.
"""

from typing import List, Optional

from src.state import Citation, EvidenceItem, VerificationResult


def verify_claim(
    claim: str,
    supporting_evidence: List[EvidenceItem],
    nli_model=None,
) -> VerificationResult:
    """
    A claim is considered "supported" only if:
      - it has at least one supporting evidence item with a real doc_id, AND
      - if an NLI model is available, at least one of those items is scored
        ENTAILMENT (not just NEUTRAL) against the claim.

    Without an NLI model, verification falls back to the weaker but still
    real check of "does at least one citable evidence item exist" — it never
    fabricates an entailment score it didn't compute.
    """
    doc_ids = [e.get("doc_id") for e in supporting_evidence if e.get("doc_id")]
    if not doc_ids:
        return VerificationResult(
            claim=claim,
            supported=False,
            supporting_doc_ids=[],
            nli_label="NONE",
            nli_score=0.0,
            notes="No citable evidence was provided for this claim.",
        )

    if nli_model is None:
        return VerificationResult(
            claim=claim,
            supported=True,
            supporting_doc_ids=doc_ids,
            nli_label="NOT_CHECKED",
            nli_score=0.0,
            notes="NLI model unavailable — verified only that citable evidence exists, entailment not checked.",
        )

    best_score = 0.0
    for item in supporting_evidence:
        try:
            scores = nli_model.predict(item.get("text", ""), claim)
        except Exception:
            continue
        entailment_score = scores.get("ENTAILMENT", 0.0)
        if entailment_score > best_score:
            best_score = entailment_score

    supported = best_score >= 0.4
    best_label = "ENTAILMENT" if supported else "NO_STRONG_ENTAILMENT"
    notes = (
        f"Best entailment score across {len(supporting_evidence)} evidence item(s): {best_score:.2f}."
        if supported
        else f"No evidence item entailed this claim strongly enough (best score {best_score:.2f})."
    )

    return VerificationResult(
        claim=claim,
        supported=supported,
        supporting_doc_ids=doc_ids,
        nli_label=best_label,
        nli_score=round(best_score, 3),
        notes=notes,
    )


def verify_gaps_and_answer(
    final_answer: str,
    research_gaps: List[dict],
    evidence_items: List[EvidenceItem],
    nli_model=None,
) -> List[VerificationResult]:
    """
    Verifies the final answer text and every accepted research gap's
    description as separate claims, each checked against the evidence items
    that actually support it (gaps already carry their own supporting_doc_ids
    from gap_agent's validation step).
    """
    results: List[VerificationResult] = []

    if final_answer and final_answer.strip():
        # Verify the answer against all usable evidence, since it may
        # synthesize across multiple sub-questions.
        usable = [e for e in evidence_items if e.get("classification") in ("RELEVANT", "WEAKLY_RELEVANT")]
        results.append(verify_claim(final_answer[:500], usable, nli_model=nli_model))

    for gap in research_gaps:
        supporting = [
            e for e in evidence_items
            if e.get("doc_id") in set(gap.get("supporting_doc_ids", []))
        ]
        results.append(verify_claim(gap.get("description", gap.get("title", "")), supporting, nli_model=nli_model))

    return results


def needs_another_cycle(verification_results: List[VerificationResult]) -> bool:
    """Real decision: another verification/retrieval cycle is needed only if
    something failed verification."""
    return any(not r.get("supported", False) for r in verification_results)
