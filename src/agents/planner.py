"""
Planner Agent.

Decomposes a research question into sub-questions dynamically. Uses the
generation model to produce candidate sub-questions from the research
question itself, then deterministically cleans/deduplicates/caps them —
the LLM proposes, deterministic code disposes, so a bad/short model output
degrades gracefully instead of breaking the pipeline.
"""

import re
from typing import Dict, List

from src.config import agents as agents_cfg

_FALLBACK_DIMENSIONS = [
    "What existing approaches or methods address this topic?",
    "What datasets or evaluation benchmarks are commonly used?",
    "What limitations or challenges have been reported?",
    "Are there conflicting findings or disagreements in the literature?",
    "What open problems or unresolved questions remain?",
]


def _clean_subquestions(raw_text: str, max_items: int) -> List[str]:
    """
    Parse a numbered/bulleted list out of raw LLM text. Robust to the model
    not perfectly following formatting instructions — falls back to
    splitting on newlines/sentence boundaries if no numbering is found.
    """
    lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
    cleaned = []
    for line in lines:
        line = re.sub(r"^\s*[\d]+[\.\)]\s*", "", line)
        line = re.sub(r"^\s*[-•*]\s*", "", line)
        line = line.strip()
        if len(line) < 8:
            continue
        cleaned.append(line)

    if not cleaned:
        # Model returned an unstructured blob; split on sentence boundaries.
        cleaned = [s.strip() for s in re.split(r"(?<=[.?])\s+", raw_text) if len(s.strip()) > 8]

    # Deduplicate while preserving order.
    seen = set()
    deduped = []
    for q in cleaned:
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(q if q.endswith("?") else q.rstrip(".") + "?")

    return deduped[:max_items]


def create_research_plan(
    research_question: str,
    generation_model=None,
    max_sub_questions: int = agents_cfg.max_sub_questions,
) -> Dict:
    """
    Returns a research plan dict:
        {
            "research_question": str,
            "sub_questions": [str, ...],
            "comparison_criteria": [str, ...],
            "used_model": bool,
        }

    `generation_model` is any object with a `.generate(prompt) -> str`
    method (see src/models/llm.py). If None or generation fails, falls back
    to a fixed set of research dimensions applied to the question — still
    dynamically phrased around the actual question text, not hardcoded gap
    content.
    """
    used_model = False
    sub_questions: List[str] = []

    if generation_model is not None:
        prompt = (
            "You are a research planning assistant. Break the following research "
            "question into distinct, specific sub-questions that together would "
            "let someone investigate it thoroughly. Cover: existing approaches, "
            "datasets/benchmarks used, known limitations, disagreements in the "
            "literature, and unresolved problems. Number each sub-question.\n\n"
            f"Research question: {research_question}\n\nSub-questions:"
        )
        try:
            raw = generation_model.generate(prompt, max_new_tokens=200)
            sub_questions = _clean_subquestions(raw, max_sub_questions)
            used_model = bool(sub_questions)
        except Exception:
            sub_questions = []

    if not sub_questions:
        # Deterministic fallback: apply the fixed research dimensions to the
        # actual question topic, rather than a single hardcoded plan.
        topic = research_question.strip().rstrip("?")
        sub_questions = [
            dim.replace("this topic", f"'{topic}'") for dim in _FALLBACK_DIMENSIONS
        ][:max_sub_questions]

    comparison_criteria = [
        "methodology", "dataset/benchmark used", "reported results",
        "publication year", "stated limitations",
    ]

    return {
        "research_question": research_question,
        "sub_questions": sub_questions,
        "comparison_criteria": comparison_criteria,
        "used_model": used_model,
    }
