"""
Retrieval Agent.

Runs the hybrid retriever for every generated query, tags each hit with the
sub-question it was retrieved for, deduplicates across queries/sub-questions
by chunk_id, and optionally reranks the merged result.
"""

from typing import List, Optional

from src.config import retrieval as retrieval_cfg
from src.rag.hybrid_retriever import HybridRetriever
from src.state import RetrievedChunk


def retrieve_for_sub_questions(
    retriever: HybridRetriever,
    sub_questions: List[str],
    queries_by_subquestion: dict,
    reranker=None,
    enable_reranking: bool = retrieval_cfg.enable_reranking,
) -> List[RetrievedChunk]:
    """
    queries_by_subquestion: {sub_question: [query, ...]}

    Returns a deduplicated, hybrid-score-sorted list of RetrievedChunk.
    """
    if not retriever.is_ready:
        return []

    seen_chunk_ids = set()
    merged: List[RetrievedChunk] = []

    for sq in sub_questions:
        queries = queries_by_subquestion.get(sq, [])
        for query in queries:
            hits = retriever.search(query, sub_question=sq)
            for hit in hits:
                if hit["chunk_id"] in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(hit["chunk_id"])
                merged.append(hit)

    merged.sort(key=lambda c: c["hybrid_score"], reverse=True)

    if enable_reranking and reranker is not None and merged:
        # Rerank against the overall research question context implicitly
        # captured by re-scoring against each chunk's own sub_question/query
        # would require per-chunk queries; instead we rerank the merged list
        # against the concatenation of sub-questions as a single relevance
        # anchor, which is a reasonable approximation for a flat re-score.
        combined_query = " ".join(sub_questions)
        merged = reranker.rerank(combined_query, merged)

    return merged[: retrieval_cfg.max_papers]


def retrieve_additional(
    retriever: HybridRetriever,
    missing_queries: List[str],
    already_seen_chunk_ids: set,
) -> List[RetrievedChunk]:
    """Used by adaptive retrieval to fetch new evidence for missing coverage."""
    if not retriever.is_ready:
        return []

    new_hits: List[RetrievedChunk] = []
    for query in missing_queries:
        for hit in retriever.search(query):
            if hit["chunk_id"] in already_seen_chunk_ids:
                continue
            already_seen_chunk_ids.add(hit["chunk_id"])
            new_hits.append(hit)

    new_hits.sort(key=lambda c: c["hybrid_score"], reverse=True)
    return new_hits
