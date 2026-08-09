"""
Real tests for the evidence, gap, and verification agents. These use small
hand-built evidence lists rather than a live retrieval index, since the
agents' logic is independent of where the evidence came from.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.evidence_agent import compute_evidence_coverage, evaluate_chunks
from src.agents.gap_agent import detect_gaps
from src.agents.reasoning_agent import INSUFFICIENT_EVIDENCE_MESSAGE, synthesize_answer
from src.agents.verification_agent import needs_another_cycle, verify_claim


def test_evidence_agent_classifies_relevant_and_irrelevant():
    chunks = [
        {
            "chunk_id": "c1", "doc_id": "d1", "title": "Multilingual Survey", "url": "u1",
            "text": "Multilingual NLP datasets remain scarce for low-resource languages, limiting model evaluation.",
            "sub_question": "What datasets are commonly used?", "hybrid_score": 0.8,
        },
        {
            "chunk_id": "c2", "doc_id": "d2", "title": "Unrelated Paper", "url": "u2",
            "text": "Cats and dogs behavior study in urban households.",
            "sub_question": "What datasets are commonly used?", "hybrid_score": 0.05,
        },
    ]
    evidence = evaluate_chunks(chunks)
    by_id = {e["chunk_id"]: e for e in evidence}
    assert by_id["c1"]["classification"] == "RELEVANT"
    assert by_id["c2"]["classification"] == "IRRELEVANT"


def test_evidence_coverage_is_computed_not_fabricated():
    evidence = [
        {"sub_question": "q1", "classification": "RELEVANT"},
        {"sub_question": "q1", "classification": "RELEVANT"},
        {"sub_question": "q2", "classification": "IRRELEVANT"},
    ]
    coverage = compute_evidence_coverage(["q1", "q2"], evidence)
    assert coverage["per_subquestion"]["q1"]["relevant_count"] == 2
    assert coverage["per_subquestion"]["q2"]["relevant_count"] == 0
    assert coverage["is_sufficient"] is False  # q2 has zero relevant evidence


def test_gap_agent_rejects_irrelevant_only_candidates():
    evidence = [
        {"chunk_id": "1", "doc_id": "d1", "title": "X", "text": "This has limitation words but is irrelevant.",
         "sub_question": "q", "classification": "IRRELEVANT"},
    ]
    gaps = detect_gaps(evidence, {"per_subquestion": {}})
    assert gaps == []


def test_gap_agent_accepts_validated_candidate():
    evidence = [
        {"chunk_id": "1", "doc_id": "d1", "title": "Survey A",
         "text": "Dataset scarcity for low-resource languages remains an open challenge.",
         "sub_question": "q", "classification": "RELEVANT"},
    ]
    gaps = detect_gaps(evidence, {"per_subquestion": {}})
    assert len(gaps) == 1
    assert gaps[0]["supporting_doc_ids"] == ["d1"]
    assert gaps[0]["validation_passed"] is True


def test_reasoning_agent_never_hallucinates_without_evidence():
    answer = synthesize_answer("What are the gaps?", [], [], generation_model=None)
    assert answer == INSUFFICIENT_EVIDENCE_MESSAGE


def test_reasoning_agent_extractive_fallback_cites_real_evidence():
    evidence = [
        {"chunk_id": "1", "doc_id": "d1", "title": "Paper A",
         "text": "Dataset scarcity remains unresolved.", "classification": "RELEVANT", "relevance_score": 0.8},
    ]
    answer = synthesize_answer("What are dataset gaps?", evidence, [], generation_model=None)
    assert "Paper A" in answer


def test_verification_without_nli_requires_citable_evidence():
    evidence = [{"chunk_id": "1", "doc_id": "d1", "title": "A", "text": "x"}]
    result = verify_claim("A claim.", evidence, nli_model=None)
    assert result["supported"] is True

    result2 = verify_claim("A claim.", [], nli_model=None)
    assert result2["supported"] is False


def test_verification_with_nli_rejects_unsupported_claims():
    class NeverEntailNLI:
        def predict(self, premise, hypothesis):
            return {"ENTAILMENT": 0.1, "NEUTRAL": 0.8, "CONTRADICTION": 0.1}

    evidence = [{"chunk_id": "1", "doc_id": "d1", "title": "A", "text": "Something unrelated."}]
    result = verify_claim("A totally different claim.", evidence, nli_model=NeverEntailNLI())
    assert result["supported"] is False
    assert needs_another_cycle([result]) is True
