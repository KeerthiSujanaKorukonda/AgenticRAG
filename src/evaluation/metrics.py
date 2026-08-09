"""
Evaluation metrics.

Every metric here is computed from real pipeline outputs — nothing in this
module invents a number. Where a metric would require ground-truth
relevance judgments this project does not have (e.g. true retrieval
recall against a labeled qrels set), the function returns None and callers
must render that as "Not evaluated" rather than a fabricated figure.
"""

import time
from typing import Callable, Dict, List, Optional


def measure_latency(fn: Callable, *args, **kwargs) -> Dict:
    """Runs fn and returns {"result": ..., "latency_ms": float} using a real
    wall-clock measurement — never a guessed/typical number."""
    start = time.time()
    result = fn(*args, **kwargs)
    latency_ms = (time.time() - start) * 1000
    return {"result": result, "latency_ms": round(latency_ms, 1)}


def citation_correctness(citations: List[dict], known_doc_ids: set) -> Optional[float]:
    """
    Fraction of citations whose doc_id actually corresponds to a document
    that was really retrieved in this run. Real and computable without
    ground truth — a citation either maps to a real retrieved doc or not.
    """
    if not citations:
        return None
    correct = sum(1 for c in citations if c.get("doc_id") in known_doc_ids)
    return round(correct / len(citations), 3)


def evidence_coverage_score(evidence_coverage: Dict) -> Optional[float]:
    if not evidence_coverage:
        return None
    return evidence_coverage.get("overall_coverage_score")


def faithfulness_rate(verification_results: List[dict]) -> Optional[float]:
    """
    Real, computed from the Verification Agent's actual NLI-based checks:
    fraction of checked claims that were found to be supported by their
    cited evidence. Requires verification to have actually run.
    """
    if not verification_results:
        return None
    supported = sum(1 for r in verification_results if r.get("supported"))
    return round(supported / len(verification_results), 3)


def hallucination_rate(verification_results: List[dict]) -> Optional[float]:
    rate = faithfulness_rate(verification_results)
    return None if rate is None else round(1 - rate, 3)


def retrieval_iterations(state: dict) -> Optional[int]:
    return state.get("iteration")


def context_relevance(evidence_items: List[dict]) -> Optional[float]:
    """
    Fraction of ALL retrieved evidence that the Evidence Agent classified
    as RELEVANT (a real, computed precision-style metric over retrieved
    context — not retrieval recall, which would need ground truth).
    """
    if not evidence_items:
        return None
    relevant = sum(1 for e in evidence_items if e.get("classification") == "RELEVANT")
    return round(relevant / len(evidence_items), 3)


def answer_correctness() -> Optional[float]:
    """
    Real answer correctness against a gold answer would require a labeled
    benchmark with reference answers, which this project does not have.
    Always returns None; callers must display "Not evaluated".
    """
    return None


def retrieval_recall() -> Optional[float]:
    """Same caveat as answer_correctness — no labeled qrels available."""
    return None


def summarize_run_metrics(final_state: dict, known_doc_ids: set) -> Dict[str, Optional[float]]:
    """Compute every metric that IS computable from a single pipeline run's
    final state, leaving the rest explicitly "Not evaluated" (None)."""
    return {
        "context_relevance": context_relevance(final_state.get("evidence_items", [])),
        "evidence_coverage": evidence_coverage_score(final_state.get("evidence_coverage", {})),
        "citation_correctness": citation_correctness(final_state.get("citations", []), known_doc_ids),
        "faithfulness": faithfulness_rate(final_state.get("verification_results", [])),
        "hallucination_rate": hallucination_rate(final_state.get("verification_results", [])),
        "retrieval_iterations": retrieval_iterations(final_state),
        "retrieval_recall": retrieval_recall(),
        "answer_correctness": answer_correctness(),
    }


def format_metric(value: Optional[float]) -> str:
    return "Not evaluated" if value is None else str(value)
