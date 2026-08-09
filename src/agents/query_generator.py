"""
Query Generation Agent.

For every sub-question, produces multiple retrieval queries with varied
phrasing/emphasis, so hybrid retrieval sees more than one lexical/semantic
angle on the same underlying information need.
"""

import re
from typing import List

from src.config import agents as agents_cfg

_STOPWORDS = {
    "what", "are", "the", "is", "of", "in", "for", "to", "a", "an", "and",
    "or", "how", "why", "does", "do", "which", "on", "with", "that", "this",
}


def _keyword_core(sub_question: str) -> str:
    words = re.findall(r"[a-zA-Z][a-zA-Z\-]+", sub_question.lower())
    core = [w for w in words if w not in _STOPWORDS]
    return " ".join(core)


def generate_queries_for_subquestion(
    sub_question: str,
    research_question: str,
    max_queries: int = agents_cfg.max_queries_per_subquestion,
) -> List[str]:
    """
    Deterministically derive several distinct retrieval queries from a
    sub-question: the sub-question itself, a keyword-stripped core version,
    a version anchored to the overall research topic, and a "limitations"-
    angled version. This is intentionally rule-based (not another LLM call)
    so query generation is fast, dependency-free, and fully reproducible.
    """
    core = _keyword_core(sub_question)
    topic_core = _keyword_core(research_question)

    candidates = [
        sub_question.rstrip("?"),
        core,
        f"{core} {topic_core}".strip() if topic_core and topic_core not in core else core,
        f"{core} limitations challenges" if core else sub_question,
    ]

    seen = set()
    queries = []
    for q in candidates:
        q = re.sub(r"\s+", " ", q).strip()
        if not q or q.lower() in seen:
            continue
        seen.add(q.lower())
        queries.append(q)

    return queries[:max_queries]


def generate_all_queries(sub_questions: List[str], research_question: str) -> List[str]:
    all_queries: List[str] = []
    for sq in sub_questions:
        all_queries.extend(generate_queries_for_subquestion(sq, research_question))
    return all_queries
