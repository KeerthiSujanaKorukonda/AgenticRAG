"""
Hybrid retrieval: combines semantic (FAISS) and lexical (BM25) search over
the same chunk corpus, normalizes each score independently, and fuses them
with configurable weights.

This module holds the actual corpus (chunk metadata + text) in memory and
is the single object both `prepare_data.py` (to build/save) and the app (to
load/query) interact with.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.config import retrieval as retrieval_cfg
from src.rag.bm25 import BM25Index
from src.rag.embeddings import EmbeddingModel
from src.rag.vector_store import FaissVectorStore
from src.state import RetrievedChunk


def _min_max_normalize(scores: List[float]) -> List[float]:
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        return [1.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


class HybridRetriever:
    def __init__(self, embedding_model: Optional[EmbeddingModel] = None):
        self.embedding_model = embedding_model or EmbeddingModel()
        self.bm25 = BM25Index()
        self.vector_store: Optional[FaissVectorStore] = None
        self.chunks: List[Dict] = []  # parallel to embeddings/BM25 corpus order

    # --- Index construction -------------------------------------------------

    def build(self, chunks: List[Dict]) -> None:
        """
        chunks: list of dicts each with at least {doc_id, title, text, ...}.
        Builds both the BM25 index and the FAISS semantic index over the
        same ordered list of chunks.
        """
        self.chunks = chunks
        texts = [c["text"] for c in chunks]

        self.bm25.build(texts)

        self.embedding_model.load()
        embeddings = self.embedding_model.embed(texts)
        self.vector_store = FaissVectorStore(dimension=embeddings.shape[1] if len(embeddings) else 384)
        self.vector_store.build(embeddings)

    @property
    def is_ready(self) -> bool:
        return bool(self.chunks) and self.bm25.is_built and self.vector_store is not None and self.vector_store.is_built

    # --- Persistence ----------------------------------------------------

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / "chunks.jsonl", "w") as f:
            for chunk in self.chunks:
                f.write(json.dumps(chunk) + "\n")
        if self.vector_store is not None:
            self.vector_store.save(directory / "faiss.index")

    def load(self, directory: Path) -> None:
        chunks_path = directory / "chunks.jsonl"
        if not chunks_path.exists():
            raise FileNotFoundError(f"No prepared index found at {directory}")

        chunks = []
        with open(chunks_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
        self.chunks = chunks

        texts = [c["text"] for c in chunks]
        self.bm25.build(texts)

        self.embedding_model.load()
        self.vector_store = FaissVectorStore(dimension=self.embedding_model.dimension)
        index_path = directory / "faiss.index"
        if index_path.exists():
            self.vector_store.load(index_path)
        else:
            embeddings = self.embedding_model.embed(texts)
            self.vector_store.build(embeddings)

    # --- Querying ---------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = retrieval_cfg.top_k_hybrid,
        semantic_weight: float = retrieval_cfg.semantic_weight,
        bm25_weight: float = retrieval_cfg.bm25_weight,
        sub_question: str = "",
    ) -> List[RetrievedChunk]:
        if not self.is_ready:
            return []

        query_embedding = self.embedding_model.embed_query(query)
        semantic_hits = self.vector_store.search(query_embedding, top_k=retrieval_cfg.top_k_semantic)
        bm25_hits = self.bm25.search(query, top_k=retrieval_cfg.top_k_bm25)

        semantic_scores_by_idx = dict(semantic_hits)
        bm25_scores_by_idx = dict(bm25_hits)

        candidate_indices = set(semantic_scores_by_idx) | set(bm25_scores_by_idx)
        if not candidate_indices:
            return []

        # Normalize each score type independently across this query's candidates.
        sem_vals = [semantic_scores_by_idx.get(i, 0.0) for i in candidate_indices]
        bm25_vals = [bm25_scores_by_idx.get(i, 0.0) for i in candidate_indices]
        sem_norm = dict(zip(candidate_indices, _min_max_normalize(sem_vals)))
        bm25_norm = dict(zip(candidate_indices, _min_max_normalize(bm25_vals)))

        results: List[RetrievedChunk] = []
        for idx in candidate_indices:
            chunk = self.chunks[idx]
            sem_score = semantic_scores_by_idx.get(idx, 0.0)
            bm25_score = bm25_scores_by_idx.get(idx, 0.0)
            hybrid_score = semantic_weight * sem_norm.get(idx, 0.0) + bm25_weight * bm25_norm.get(idx, 0.0)

            results.append(
                RetrievedChunk(
                    doc_id=chunk.get("doc_id", ""),
                    chunk_id=chunk.get("chunk_id", f"{chunk.get('doc_id','')}-{idx}"),
                    title=chunk.get("title", ""),
                    authors=chunk.get("authors", []),
                    year=chunk.get("year"),
                    url=chunk.get("url", ""),
                    source=chunk.get("source", ""),
                    text=chunk.get("text", ""),
                    semantic_score=sem_score,
                    bm25_score=bm25_score,
                    hybrid_score=hybrid_score,
                    rerank_score=None,
                    sub_question=sub_question,
                    query=query,
                )
            )

        results.sort(key=lambda r: r["hybrid_score"], reverse=True)
        return results[:top_k]
