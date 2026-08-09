"""
Benchmark comparison: Baseline 1 (vector-only RAG), Baseline 2 (hybrid RAG,
no agents), and the full Agentic RAG system — run against the same set of
benchmark questions, with the same underlying retriever/index, so
differences reflect the pipeline architecture rather than data differences.

Also implements the ablation study (full system minus one component at a
time). All results are measured from real runs; nothing here is a
precomputed/fabricated number. If you haven't run scripts/evaluate.py in an
environment with the real models/index available, there ARE no results —
this module produces them, it doesn't ship them.
"""

import time
from typing import Callable, Dict, List, Optional

from src.agents import evidence_agent, reasoning_agent
from src.evaluation.metrics import summarize_run_metrics
from src.graph.workflow import WorkflowDependencies, build_graph
from src.state import ResearchState

DEFAULT_BENCHMARK_QUESTIONS = [
    "What are the research gaps in low-resource multilingual NLP?",
    "What are the main limitations of retrieval-augmented generation?",
    "How is hallucination detected in large language models?",
    "What datasets are commonly used to evaluate multilingual NLP models?",
    "Are there disagreements in the literature about cross-lingual transfer?",
]


def run_baseline_vector_rag(question: str, retriever, embedding_model, top_k: int = 8) -> Dict:
    """
    Baseline 1: Question -> semantic retrieval only -> extractive "answer"
    (top passages concatenated). No planning, no BM25, no agents.
    """
    start = time.time()
    if retriever is None or not retriever.is_ready:
        return {"answer": "", "evidence_items": [], "latency_ms": 0.0, "retrieved_count": 0}

    query_embedding = embedding_model.embed_query(question)
    hits = retriever.vector_store.search(query_embedding, top_k=top_k)
    chunks = [retriever.chunks[i] for i, _ in hits]
    answer = " ".join(c["text"] for c in chunks[:3])
    latency_ms = (time.time() - start) * 1000
    return {
        "answer": answer,
        "evidence_items": chunks,
        "latency_ms": round(latency_ms, 1),
        "retrieved_count": len(chunks),
    }


def run_baseline_hybrid_rag(question: str, retriever, top_k: int = 8) -> Dict:
    """
    Baseline 2: Question -> BM25 + semantic (hybrid) retrieval -> extractive
    "answer". Still no planning/multi-query/agents.
    """
    start = time.time()
    if retriever is None or not retriever.is_ready:
        return {"answer": "", "evidence_items": [], "latency_ms": 0.0, "retrieved_count": 0}

    hits = retriever.search(question, top_k=top_k)
    answer = " ".join(h["text"] for h in hits[:3])
    latency_ms = (time.time() - start) * 1000
    return {
        "answer": answer,
        "evidence_items": hits,
        "latency_ms": round(latency_ms, 1),
        "retrieved_count": len(hits),
    }


def run_agentic_rag(question: str, deps: WorkflowDependencies) -> Dict:
    """System: the full compiled LangGraph pipeline."""
    start = time.time()
    graph = build_graph(deps)
    final_state: ResearchState = graph.invoke({"user_input": question, "agent_events": []})
    latency_ms = (time.time() - start) * 1000

    known_doc_ids = {c.get("doc_id") for c in final_state.get("retrieved_documents", [])}
    metrics = summarize_run_metrics(final_state, known_doc_ids)
    metrics["latency_ms"] = round(latency_ms, 1)
    metrics["retrieval_iterations"] = final_state.get("iteration", 0)

    return {"final_state": final_state, "metrics": metrics}


def run_comparison(
    questions: List[str],
    deps: WorkflowDependencies,
) -> List[Dict]:
    """
    Runs all three systems on the same benchmark questions using the same
    retriever/models. Returns one result row per question with each
    system's real measured outputs.
    """
    results = []
    for question in questions:
        row = {"question": question}

        b1 = run_baseline_vector_rag(question, deps.retriever, deps.embedding_model)
        row["baseline_vector_rag"] = {
            "answer_preview": b1["answer"][:200],
            "retrieved_count": b1["retrieved_count"],
            "latency_ms": b1["latency_ms"],
        }

        b2 = run_baseline_hybrid_rag(question, deps.retriever)
        row["baseline_hybrid_rag"] = {
            "answer_preview": b2["answer"][:200],
            "retrieved_count": b2["retrieved_count"],
            "latency_ms": b2["latency_ms"],
        }

        system = run_agentic_rag(question, deps)
        row["agentic_rag"] = {
            "answer_preview": system["final_state"].get("final_answer", "")[:200],
            "retrieved_count": len(system["final_state"].get("retrieved_documents", [])),
            "metrics": system["metrics"],
        }

        results.append(row)
    return results


ABLATION_VARIANTS = (
    "full",
    "without_planner",
    "without_evidence_verification",
    "without_adaptive_retrieval",
    "without_contradiction_detection",
    "without_final_verification",
)


def run_ablation(question: str, deps: WorkflowDependencies, variant: str) -> Dict:
    """
    Runs the agentic pipeline with one component disabled by swapping out
    the relevant dependency/config for that run only. Each variant is a
    REAL, separately-executed graph run — not a simulated toggle.
    """
    if variant not in ABLATION_VARIANTS:
        raise ValueError(f"Unknown ablation variant: {variant}")

    ablated_deps = WorkflowDependencies(
        retriever=deps.retriever,
        embedding_model=deps.embedding_model,
        generation_model=deps.generation_model if variant != "without_planner" else None,
        nli_model=(
            None if variant in ("without_evidence_verification", "without_contradiction_detection", "without_final_verification")
            else deps.nli_model
        ),
        reranker=deps.reranker,
        zero_shot_classifier=deps.zero_shot_classifier,
    )

    # "without_adaptive_retrieval" is enforced by capping iterations to 0 at
    # invoke time (see max_iterations below) rather than a separate graph.
    max_iterations = 0 if variant == "without_adaptive_retrieval" else None

    start = time.time()
    graph = build_graph(ablated_deps)
    initial_state = {"user_input": question, "agent_events": []}
    if max_iterations is not None:
        initial_state["max_iterations"] = max_iterations
    final_state = graph.invoke(initial_state)
    latency_ms = (time.time() - start) * 1000

    known_doc_ids = {c.get("doc_id") for c in final_state.get("retrieved_documents", [])}
    metrics = summarize_run_metrics(final_state, known_doc_ids)
    metrics["latency_ms"] = round(latency_ms, 1)

    return {"variant": variant, "final_state": final_state, "metrics": metrics}
