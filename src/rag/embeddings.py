"""
Sentence embedding wrapper (BAAI/bge-small-en-v1.5 by default).

Loaded lazily and cached by Streamlit's @st.cache_resource at the app layer
(see app.py); this module itself stays framework-agnostic so it's also
usable from scripts/prepare_data.py and from tests.
"""

from functools import lru_cache
from typing import List

import numpy as np

from src.config import models


class EmbeddingModel:
    """
    Thin wrapper around sentence-transformers. Import of the heavy
    `sentence_transformers` package is deferred to `load()` so importing
    this module (e.g. for type checking or tests that mock it) never
    requires torch/sentence-transformers to be installed.
    """

    def __init__(self, model_name: str = models.embedding_model):
        self.model_name = model_name
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.model_name)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def embed(self, texts: List[str], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
        """Return an (N, D) float32 array of embeddings for the given texts."""
        if not texts:
            return np.zeros((0, self.dimension), dtype="float32")
        if self._model is None:
            self.load()
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.astype("float32")

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed([query])[0]

    @property
    def dimension(self) -> int:
        # bge-small-en-v1.5 produces 384-dim embeddings; if a different
        # model is configured, infer it lazily on first real embed call.
        if self._model is not None:
            return self._model.get_sentence_embedding_dimension()
        return 384


_shared_instance = None


def get_embedding_model() -> EmbeddingModel:
    """Process-wide singleton, mirroring how it's cached at the app layer."""
    global _shared_instance
    if _shared_instance is None:
        _shared_instance = EmbeddingModel()
    return _shared_instance
