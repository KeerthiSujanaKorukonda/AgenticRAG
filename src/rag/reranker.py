"""
Optional cross-encoder reranker.

Disabled by default (see config.retrieval.enable_reranking) because a
cross-encoder forward pass per candidate is meaningfully more CPU-expensive
than the hybrid retrieval score fusion. The app and retrieval agent both
work correctly with this disabled — reranking only refines an already
-reasonable hybrid ranking.
"""

from typing import List

from src.config import models, retrieval as retrieval_cfg
from src.state import RetrievedChunk


class CrossEncoderReranker:
    def __init__(self, model_name: str = models.reranker_model):
        self.model_name = model_name
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(self.model_name)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def rerank(self, query: str, chunks: List[RetrievedChunk], top_n: int = retrieval_cfg.rerank_top_n) -> List[RetrievedChunk]:
        if not chunks:
            return chunks
        if self._model is None:
            self.load()

        candidates = chunks[:top_n]
        pairs = [(query, c["text"]) for c in candidates]
        scores = self._model.predict(pairs)

        reranked = []
        for chunk, score in zip(candidates, scores):
            updated = dict(chunk)
            updated["rerank_score"] = float(score)
            reranked.append(updated)

        reranked.sort(key=lambda c: c["rerank_score"], reverse=True)
        # Append any chunks beyond top_n unchanged, preserving their original order.
        return reranked + chunks[top_n:]


_shared_instance = None


def get_reranker() -> CrossEncoderReranker:
    global _shared_instance
    if _shared_instance is None:
        _shared_instance = CrossEncoderReranker()
    return _shared_instance
