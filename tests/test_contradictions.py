"""
Real tests for contradiction detection, including the regression case for
the cross-sub-question contradiction bug found during development.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.contradiction_agent import detect_contradictions


class FakeNLI:
    """Contradicts only when one text says 'scarcity'/'improves' and the
    other says the opposite — deterministic and inspectable."""

    def predict(self, premise, hypothesis):
        p, h = premise.lower(), hypothesis.lower()
        opposites = [("scarcity", "abundant"), ("improves", "does not improve")]
        for a, b in opposites:
            if (a in p and b in h) or (b in p and a in h):
                return {"CONTRADICTION": 0.85, "ENTAILMENT": 0.05, "NEUTRAL": 0.10}
        return {"CONTRADICTION": 0.1, "ENTAILMENT": 0.2, "NEUTRAL": 0.7}


def test_no_nli_model_returns_no_contradictions():
    evidence = [
        {"chunk_id": "1", "doc_id": "d1", "title": "A", "text": "x", "classification": "RELEVANT", "relevance_score": 0.9},
        {"chunk_id": "2", "doc_id": "d2", "title": "B", "text": "y", "classification": "RELEVANT", "relevance_score": 0.9},
    ]
    assert detect_contradictions(evidence, nli_model=None) == []


def test_contradiction_detected_across_different_sub_questions():
    """Regression test: two contradicting papers surfaced under DIFFERENT
    sub-questions must still be compared against each other."""
    evidence = [
        {"chunk_id": "c1", "doc_id": "d1", "title": "Paper A",
         "text": "Cross-lingual transfer improves accuracy on low-resource languages.",
         "sub_question": "q1", "classification": "RELEVANT", "relevance_score": 0.7, "year": 2022},
        {"chunk_id": "c2", "doc_id": "d2", "title": "Paper B",
         "text": "Cross-lingual transfer does not improve accuracy for distant language pairs.",
         "sub_question": "q2", "classification": "RELEVANT", "relevance_score": 0.7, "year": 2024},
    ]
    contradictions = detect_contradictions(evidence, nli_model=FakeNLI())
    assert len(contradictions) == 1
    assert contradictions[0]["nli_label"] == "CONTRADICTION"
    assert "2022" in contradictions[0]["likely_reason"] or "2024" in contradictions[0]["likely_reason"]


def test_same_document_pairs_are_never_compared():
    evidence = [
        {"chunk_id": "c1", "doc_id": "d1", "title": "Paper A", "text": "Data is scarce.",
         "sub_question": "q1", "classification": "RELEVANT", "relevance_score": 0.7},
        {"chunk_id": "c2", "doc_id": "d1", "title": "Paper A", "text": "Data is abundant.",
         "sub_question": "q1", "classification": "RELEVANT", "relevance_score": 0.7},
    ]
    # Same doc_id ('d1') on both sides — must never be reported as an
    # inter-source contradiction, even though the NLI model would flag it.
    contradictions = detect_contradictions(evidence, nli_model=FakeNLI())
    assert contradictions == []


def test_low_relevance_items_excluded_unless_flagged_contradictory():
    evidence = [
        {"chunk_id": "c1", "doc_id": "d1", "title": "A", "text": "x", "classification": "IRRELEVANT", "relevance_score": 0.1},
        {"chunk_id": "c2", "doc_id": "d2", "title": "B", "text": "y", "classification": "IRRELEVANT", "relevance_score": 0.1},
    ]
    assert detect_contradictions(evidence, nli_model=FakeNLI()) == []

    # But an item explicitly flagged CONTRADICTORY by the evidence agent
    # must still be considered even if its raw relevance_score is low.
    evidence2 = [
        {"chunk_id": "c1", "doc_id": "d1", "title": "A", "text": "Data scarcity is a known issue.",
         "classification": "RELEVANT", "relevance_score": 0.7},
        {"chunk_id": "c2", "doc_id": "d2", "title": "B", "text": "Data is abundant now.",
         "classification": "CONTRADICTORY", "relevance_score": 0.1},
    ]
    contradictions = detect_contradictions(evidence2, nli_model=FakeNLI())
    assert len(contradictions) == 1
