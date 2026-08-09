"""FAISS-backed semantic vector index over the chunk corpus."""

from pathlib import Path
from typing import List, Tuple

import numpy as np


class FaissVectorStore:
    """
    Flat inner-product FAISS index over normalized embeddings (equivalent to
    cosine similarity). Simple and exact — appropriate for the small corpus
    sizes (tens to low hundreds of chunks) this app targets on Streamlit
    Cloud's free CPU tier; an approximate index would be unnecessary
    complexity at this scale.
    """

    def __init__(self, dimension: int):
        self.dimension = dimension
        self._index = None

    def build(self, embeddings: np.ndarray) -> None:
        import faiss

        if embeddings.shape[0] == 0:
            self._index = None
            return
        index = faiss.IndexFlatIP(self.dimension)
        index.add(embeddings.astype("float32"))
        self._index = index

    @property
    def is_built(self) -> bool:
        return self._index is not None

    def search(self, query_embedding: np.ndarray, top_k: int = 8) -> List[Tuple[int, float]]:
        """Return [(corpus_index, cosine_score), ...] sorted by score descending."""
        if self._index is None:
            return []
        query = query_embedding.reshape(1, -1).astype("float32")
        scores, indices = self._index.search(query, top_k)
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            results.append((int(idx), float(score)))
        return results

    def save(self, path: Path) -> None:
        import faiss

        if self._index is not None:
            faiss.write_index(self._index, str(path))

    def load(self, path: Path) -> None:
        import faiss

        self._index = faiss.read_index(str(path))
