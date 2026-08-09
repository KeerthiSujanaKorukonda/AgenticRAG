"""
Reasoning Agent.

Synthesizes accepted evidence into a grounded answer for a specific question
(the overall research question, or a follow-up). Strictly evidence-bound:
if there is no RELEVANT/WEAKLY_RELEVANT evidence to reason over, it returns
an explicit "insufficient evidence" message rather than ever calling the
generation model without grounding.
"""

from typing import List, Optional

from src.models.llm import build_grounded_prompt
from src.state import EvidenceItem
from src.utils.text import truncate

INSUFFICIENT_EVIDENCE_MESSAGE = (
    "Insufficient evidence to provide a reliable research conclusion for this question."
)


def _evidence_to_blocks(evidence_items: List[EvidenceItem], max_items: int = 8) -> List[str]:
    blocks = []
    for item in evidence_items[:max_items]:
        title = item.get("title", "unknown source")
        year = item.get("year")
        year_str = f", {year}" if year else ""
        blocks.append(f"{title}{year_str}: {truncate(item.get('text', ''), 350)}")
    return blocks


def synthesize_answer(
    question: str,
    evidence_items: List[EvidenceItem],
    contradictions: List[dict],
    generation_model=None,
) -> str:
    """
    Returns a grounded natural-language answer, or the fixed insufficient-
    evidence message if there's nothing usable to reason over.
    """
    usable_evidence = [
        e for e in evidence_items
        if e.get("classification") in ("RELEVANT", "WEAKLY_RELEVANT")
    ]

    if not usable_evidence:
        return INSUFFICIENT_EVIDENCE_MESSAGE

    if generation_model is None:
        # No generation model available (e.g. not yet loaded, or disabled for
        # a fast/offline test run) — fall back to a deterministic extractive
        # synthesis: list the strongest evidence verbatim-cited rather than
        # attempt any generation at all.
        lines = [f"Based on {len(usable_evidence)} retrieved passages:"]
        for e in sorted(usable_evidence, key=lambda x: x["relevance_score"], reverse=True)[:5]:
            lines.append(f"- {e['title']}: {truncate(e['text'], 220)}")
        if contradictions:
            lines.append(
                f"\nNote: {len(contradictions)} conflicting evidence pair(s) were detected on this topic; "
                "see the Conflicting Evidence section."
            )
        return "\n".join(lines)

    blocks = _evidence_to_blocks(usable_evidence)
    prompt = build_grounded_prompt(
        instruction="You are a research assistant answering strictly from the evidence provided.",
        evidence_blocks=blocks,
        question=question,
    )

    try:
        answer = generation_model.generate(prompt)
    except Exception:
        answer = None

    if not answer or len(answer.strip()) < 5:
        return INSUFFICIENT_EVIDENCE_MESSAGE

    if contradictions:
        answer += (
            f"\n\nNote: the available sources disagree on at least one point "
            f"({len(contradictions)} conflicting evidence pair(s) detected)."
        )

    return answer
