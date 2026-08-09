"""
Real tests for src/router.py — no mocking needed since the router is
deterministic by design. These are the exact scenarios from the project's
own Test Plan (Tests 1, 2, 6, 7) plus the follow-up/dangling-reference cases.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.router import route_intent


def test_greeting():
    assert route_intent("Hi", has_prior_research_state=False)["intent"] == "GREETING"
    assert route_intent("Hello there", has_prior_research_state=False)["intent"] == "GREETING"
    assert route_intent("Thanks", has_prior_research_state=False)["intent"] == "GREETING"


def test_greeting_does_not_false_positive_on_substrings():
    """Regression test for the 'yo' inside 'you' bug found during development."""
    result = route_intent("What can you do?", has_prior_research_state=False)
    assert result["intent"] == "CAPABILITIES"

    result2 = route_intent("Who are you?", has_prior_research_state=False)
    assert result2["intent"] == "CAPABILITIES"

    result3 = route_intent("You are wrong about that", has_prior_research_state=False)
    assert result3["intent"] != "GREETING"


def test_capabilities():
    for text in ("What can you do?", "Who are you?", "Help"):
        assert route_intent(text, has_prior_research_state=False)["intent"] == "CAPABILITIES"


def test_research_query():
    result = route_intent(
        "What are the research gaps in low-resource multilingual NLP?",
        has_prior_research_state=False,
    )
    assert result["intent"] == "RESEARCH_QUERY"
    assert result["confidence"] > 0.5


def test_follow_up_with_prior_state():
    for text in (
        "Tell me more about the dataset gap.",
        "Which gap is the most important?",
        "Compare gap 2 and gap 4.",
        "Are there papers that disagree with gap 2?",
    ):
        result = route_intent(text, has_prior_research_state=True)
        assert result["intent"] == "FOLLOW_UP", f"{text!r} -> {result}"


def test_dangling_follow_up_reference_without_prior_state():
    """Spec section 36: 'Explain gap 2.' with no prior research must NOT
    silently start a new research query about a nonexistent gap."""
    result = route_intent("Explain gap 2.", has_prior_research_state=False)
    assert result["intent"] == "CLARIFICATION"
    assert "dangling_follow_up_reference" in result["reason"]


def test_unsupported_request():
    result = route_intent("Write me a poem about love.", has_prior_research_state=False)
    assert result["intent"] == "UNSUPPORTED_REQUEST"


def test_empty_input():
    result = route_intent("", has_prior_research_state=False)
    assert result["intent"] == "CLARIFICATION"
    assert result["confidence"] == 1.0


def test_ambiguous_input_without_classifier_defaults_to_clarification():
    result = route_intent("asdkj", has_prior_research_state=False)
    assert result["intent"] == "CLARIFICATION"


def test_zero_shot_fallback_low_confidence_is_not_trusted():
    """The router must not blindly trust a low-confidence model score."""

    def low_confidence_classifier(text, labels):
        return {"labels": labels, "scores": [0.3, 0.3, 0.4]}

    result = route_intent(
        "asdkj zzz qqq",
        has_prior_research_state=False,
        zero_shot_classifier=low_confidence_classifier,
    )
    assert result["intent"] == "CLARIFICATION"


def test_zero_shot_fallback_high_confidence_is_used():
    def high_confidence_classifier(text, labels):
        return {"labels": ["a research question", "small talk or a greeting", "a request unrelated to research"], "scores": [0.9, 0.05, 0.05]}

    result = route_intent(
        "asdkj zzz qqq",
        has_prior_research_state=False,
        zero_shot_classifier=high_confidence_classifier,
    )
    assert result["intent"] == "RESEARCH_QUERY"
