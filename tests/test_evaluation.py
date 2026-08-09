"""
Real, executable tests of the full agentic pipeline and the evaluation/
ablation framework, using stub embedding/generation/NLI models so they run
fast and offline (no torch/HF network access needed). These are the same
stubs used during interactive development to find and fix real bugs
(LangGraph dropping undeclared state keys, cross-sub-question
contradictions, the ablation config override being clobbered).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.benchmark import run_ablation, run_comparison
from src.graph.workflow import WorkflowDependencies, build_graph
from src.rag.embeddings import EmbeddingModel
from src.rag.hybrid_retriever import HybridRetriever


class FakeEmbeddingModel(EmbeddingModel):
    VOCAB = ["multilingual", "nlp", "dataset", "scarcity", "evaluation", "transfer", "hallucination", "retrieval", "gap", "low", "resource", "abundant"]

    def load(self):
        self._model = "fake"

    def embed(self, texts, batch_size=32, normalize=True):
        vecs = []
        for t in texts:
            tl = t.lower()
            v = np.array([1.0 if w in tl else 0.0 for w in self.VOCAB], dtype="float32")
            n = np.linalg.norm(v)
            vecs.append(v / n if n > 0 else v)
        return np.stack(vecs)

    @property
    def dimension(self):
        return len(self.VOCAB)


class FakeGenModel:
    def generate(self, prompt, max_new_tokens=200, temperature=0.3):
        if "Sub-questions:" in prompt:
            return "1. What datasets are used?\n2. What are the limitations?\n3. Are there disagreements?"
        return "Based on the evidence, low-resource multilingual NLP faces dataset scarcity and evaluation limitations."


class FakeNLI:
    def predict(self, premise, hypothesis):
        p, h = premise.lower(), hypothesis.lower()
        if ("scarcity" in p and "abundant" in h) or ("abundant" in p and "scarcity" in h):
            return {"CONTRADICTION": 0.8, "ENTAILMENT": 0.1, "NEUTRAL": 0.1}
        if any(w in p for w in h.split() if len(w) > 5):
            return {"ENTAILMENT": 0.7, "NEUTRAL": 0.2, "CONTRADICTION": 0.1}
        return {"ENTAILMENT": 0.1, "NEUTRAL": 0.8, "CONTRADICTION": 0.1}


CHUNKS = [
    {"doc_id": "d1", "chunk_id": "d1-0", "title": "Multilingual NLP Survey",
     "text": "Dataset scarcity for low-resource multilingual nlp remains a major unresolved challenge.",
     "authors": ["A"], "year": 2024, "url": "http://d1", "source": "arXiv"},
    {"doc_id": "d2", "chunk_id": "d2-0", "title": "Evaluation Survey",
     "text": "Evaluation methodology for multilingual nlp models varies widely, causing inconsistent benchmarking.",
     "authors": ["B"], "year": 2023, "url": "http://d2", "source": "arXiv"},
    {"doc_id": "d3", "chunk_id": "d3-0", "title": "Contradicting Paper",
     "text": "Training data for multilingual nlp is now abundant thanks to web-scale corpora.",
     "authors": ["C"], "year": 2025, "url": "http://d3", "source": "arXiv"},
    {"doc_id": "d4", "chunk_id": "d4-0", "title": "Unrelated Paper",
     "text": "A study of cat and dog behavior in urban households.",
     "authors": ["D"], "year": 2020, "url": "http://d4", "source": "arXiv"},
]


@pytest.fixture
def deps():
    retriever = HybridRetriever(embedding_model=FakeEmbeddingModel())
    retriever.build(CHUNKS)
    return WorkflowDependencies(
        retriever=retriever,
        embedding_model=FakeEmbeddingModel(),
        generation_model=FakeGenModel(),
        nli_model=FakeNLI(),
    )


def test_full_pipeline_runs_end_to_end_and_terminates(deps):
    graph = build_graph(deps)
    result = graph.invoke({
        "user_input": "What are the research gaps in low-resource multilingual NLP?",
        "agent_events": [],
    })

    assert result["intent"] == "RESEARCH_QUERY"
    assert len(result["sub_questions"]) > 0
    assert len(result["retrieved_documents"]) > 0
    # Loop termination guarantee: must not exceed configured caps.
    assert result["iteration"] <= result["max_iterations"]
    assert result["verification_cycle"] <= result["max_verification_cycles"]


def test_full_pipeline_detects_the_planted_contradiction(deps):
    """Regression test for the contradiction-detection bug fixed during
    development (cross-sub-question comparisons were being skipped)."""
    graph = build_graph(deps)
    result = graph.invoke({
        "user_input": "What are the research gaps in low-resource multilingual NLP?",
        "agent_events": [],
    })
    assert len(result["contradictions"]) >= 1


def test_full_pipeline_produces_only_citable_gaps(deps):
    graph = build_graph(deps)
    result = graph.invoke({
        "user_input": "What are the research gaps in low-resource multilingual NLP?",
        "agent_events": [],
    })
    for gap in result["research_gaps"]:
        assert len(gap["supporting_doc_ids"]) >= 1


def test_agent_activity_log_reflects_real_events(deps):
    graph = build_graph(deps)
    result = graph.invoke({"user_input": "Hi", "agent_events": []})
    assert len(result["agent_events"]) >= 1
    assert result["agent_events"][0]["agent"] == "Intent Router"


def test_run_comparison_produces_all_three_systems(deps):
    results = run_comparison(["What are multilingual nlp dataset gaps?"], deps)
    assert len(results) == 1
    row = results[0]
    assert "baseline_vector_rag" in row
    assert "baseline_hybrid_rag" in row
    assert "agentic_rag" in row


def test_ablation_without_adaptive_retrieval_actually_caps_iterations(deps):
    """Regression test for the bug where the ablation's max_iterations
    override was silently overwritten by the planner node."""
    result = run_ablation("What are multilingual nlp dataset gaps?", deps, "without_adaptive_retrieval")
    assert result["final_state"]["iteration"] == 0


def test_ablation_full_variant_is_unaffected():
    retriever = HybridRetriever(embedding_model=FakeEmbeddingModel())
    retriever.build(CHUNKS)
    deps = WorkflowDependencies(
        retriever=retriever, embedding_model=FakeEmbeddingModel(),
        generation_model=FakeGenModel(), nli_model=FakeNLI(),
    )
    result = run_ablation("What are multilingual nlp dataset gaps?", deps, "full")
    assert result["final_state"]["iteration"] > 0


def test_unknown_ablation_variant_raises():
    retriever = HybridRetriever(embedding_model=FakeEmbeddingModel())
    retriever.build(CHUNKS)
    deps = WorkflowDependencies(retriever=retriever, embedding_model=FakeEmbeddingModel())
    with pytest.raises(ValueError):
        run_ablation("q", deps, "not_a_real_variant")
