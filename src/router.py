"""
Intent router — the mandatory first component of every request.

Deterministic rules run first and handle the overwhelming majority of
casual input (greetings, thanks, "what can you do") without ever touching a
model. A lightweight zero-shot classifier is used only as a fallback for
genuinely ambiguous input, and its output is never trusted blindly — it's
just one more signal combined with simple heuristics (question length,
presence of a prior research state for follow-up detection, etc.).

Intents:
    GREETING, CAPABILITIES, RESEARCH_QUERY, FOLLOW_UP, CLARIFICATION,
    UNSUPPORTED_REQUEST
"""

import re
from typing import Dict, Optional

from src.config import router as router_cfg

INTENTS = {
    "GREETING",
    "CAPABILITIES",
    "RESEARCH_QUERY",
    "FOLLOW_UP",
    "CLARIFICATION",
    "UNSUPPORTED_REQUEST",
}

# Deterministic signals that something is a research-style question, used to
# avoid over-triggering CLARIFICATION/UNSUPPORTED_REQUEST on legitimate short
# research asks like "LLM safety gaps?".
_RESEARCH_KEYWORDS = (
    "research", "gap", "gaps", "evidence", "paper", "papers", "study", "studies",
    "dataset", "datasets", "method", "methods", "approach", "survey", "literature",
    "findings", "limitation", "limitations", "compare", "comparison", "evaluate",
    "evaluation", "benchmark", "model", "models", "nlp", "llm", "machine learning",
    "contradict", "disagree", "cite", "citation",
)

_UNSUPPORTED_PATTERNS = (
    re.compile(r"\bwrite (me )?a (poem|song|story|joke)\b", re.I),
    re.compile(r"\btell me a joke\b", re.I),
    re.compile(r"\bwhat'?s the weather\b", re.I),
    re.compile(r"\bwho won the\b.*\bgame\b", re.I),
)


def _normalize(text: str) -> str:
    return re.sub(r"[^\w\s?]", "", text.strip().lower())


def _matches_any(text_norm: str, phrases) -> bool:
    """
    Word-boundary phrase matching. Plain substring matching would let short
    phrases like "hi" or "yo" match inside unrelated words ("this", "you"),
    which is exactly the kind of false positive this router must avoid.
    """
    return any(re.search(rf"\b{re.escape(phrase)}\b", text_norm) for phrase in phrases)


def _looks_like_follow_up_reference(text_norm: str) -> bool:
    """Heuristics for references back to prior research state."""
    patterns = (
        r"\bgap\s*\d+\b", r"\bthis gap\b", r"\bthat gap\b", r"\bthe gap\b",
        r"\bmost important\b", r"\bwhich (one|gap|theme|paper)\b",
        r"\btell me more\b", r"\bexplain (that|this|it)\b", r"\bwhat about\b",
        r"\bcompare (gap|theme)\b", r"\belaborate\b", r"\bmore detail\b",
        r"\bdisagree with\b", r"\bconflict(ing)? (with|evidence)\b",
    )
    return any(re.search(p, text_norm) for p in patterns)


def route_intent(
    user_input: str,
    has_prior_research_state: bool,
    zero_shot_classifier=None,
) -> Dict:
    """
    Returns {"intent": str, "confidence": float, "reason": str}.

    `zero_shot_classifier` is an optional callable(text, candidate_labels) ->
    {"labels": [...], "scores": [...]} (matching a HF zero-shot-classification
    pipeline's call signature) used only for the ambiguous fallback case.
    Passing None disables the model fallback entirely and the router still
    functions correctly using deterministic rules alone.
    """
    if not user_input or not user_input.strip():
        return {"intent": "CLARIFICATION", "confidence": 1.0, "reason": "Empty input."}

    text_norm = _normalize(user_input)
    word_count = len(text_norm.split())

    # 1. Greeting — deterministic, exact/near-exact short match.
    if word_count <= 4 and (
        _matches_any(text_norm, router_cfg.greeting_phrases)
        or _matches_any(text_norm, router_cfg.gratitude_phrases)
    ):
        return {
            "intent": "GREETING",
            "confidence": 0.98,
            "reason": "Matched a deterministic greeting/gratitude phrase.",
        }

    # 2. Capabilities — deterministic phrase match.
    if _matches_any(text_norm, router_cfg.capability_phrases):
        return {
            "intent": "CAPABILITIES",
            "confidence": 0.95,
            "reason": "Matched a deterministic capabilities-question phrase.",
        }

    # 3. Follow-up — only possible if there IS prior research state, and the
    #    text either references it explicitly or is short/pronoun-heavy.
    references_prior_item = _looks_like_follow_up_reference(text_norm)

    if has_prior_research_state and (references_prior_item or word_count <= 6):
        return {
            "intent": "FOLLOW_UP",
            "confidence": 0.85,
            "reason": "Prior research state exists and input references or is short enough to plausibly continue it.",
        }

    # 3b. A follow-up-shaped reference ("gap 2", "that gap", "explain this")
    #     with NO prior state to resolve it against — do not silently start
    #     an unrelated fresh research query. Flag it explicitly so the graph
    #     can respond with "please start a research question first" instead
    #     of inventing an answer about a gap that was never identified.
    if not has_prior_research_state and references_prior_item:
        return {
            "intent": "CLARIFICATION",
            "confidence": 0.9,
            "reason": "dangling_follow_up_reference: refers to prior research state that doesn't exist yet.",
        }

    # 4. Unsupported — deterministic pattern match for clearly off-topic asks.
    if any(p.search(user_input) for p in _UNSUPPORTED_PATTERNS):
        return {
            "intent": "UNSUPPORTED_REQUEST",
            "confidence": 0.9,
            "reason": "Matched a deterministic off-topic request pattern.",
        }

    # 5. Research query — deterministic keyword signal + sufficient length.
    has_research_keyword = _matches_any(text_norm, _RESEARCH_KEYWORDS)
    is_question_like = "?" in user_input or text_norm.startswith(
        ("what", "how", "why", "which", "are there", "is there", "do", "does", "can")
    )

    if word_count >= router_cfg.min_research_query_words and (has_research_keyword or is_question_like):
        return {
            "intent": "RESEARCH_QUERY",
            "confidence": 0.9 if has_research_keyword else 0.7,
            "reason": (
                "Contains research-topic keywords and sufficient length."
                if has_research_keyword
                else "Question-like phrasing with sufficient length; no explicit research keyword."
            ),
        }

    # 6. Ambiguous — try the zero-shot fallback if one was provided.
    if zero_shot_classifier is not None:
        try:
            candidate_labels = [
                "a research question",
                "small talk or a greeting",
                "a request unrelated to research",
            ]
            result = zero_shot_classifier(user_input, candidate_labels)
            top_label = result["labels"][0]
            top_score = float(result["scores"][0])

            if top_score < router_cfg.zero_shot_confidence_floor:
                return {
                    "intent": "CLARIFICATION",
                    "confidence": top_score,
                    "reason": f"Zero-shot classifier confidence too low ({top_score:.2f}) to trust; asking for clarification.",
                }

            mapped = {
                "a research question": "RESEARCH_QUERY",
                "small talk or a greeting": "GREETING",
                "a request unrelated to research": "UNSUPPORTED_REQUEST",
            }[top_label]
            return {
                "intent": mapped,
                "confidence": top_score,
                "reason": f"Zero-shot fallback classified as '{top_label}' (score={top_score:.2f}).",
            }
        except Exception as exc:  # model unavailable, etc. — degrade gracefully
            return {
                "intent": "CLARIFICATION",
                "confidence": 0.5,
                "reason": f"Zero-shot fallback unavailable ({exc}); defaulting to clarification.",
            }

    # 7. No model fallback available — default to clarification rather than
    #    guessing, per the "never blindly trust confidence" requirement.
    return {
        "intent": "CLARIFICATION",
        "confidence": 0.5,
        "reason": "Ambiguous input with no research keywords and no zero-shot fallback available.",
    }
