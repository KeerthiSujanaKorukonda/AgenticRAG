"""BM25 lexical retrieval over the chunk corpus, using rank_bm25."""

import re
from typing import List, Tuple

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


class BM25Index:
    """
    Wraps rank_bm25.BM25Okapi over a fixed list of chunk texts. Built once
    per corpus (see build()) and queried many times.
    """

    def __init__(self):
        self._bm25 = None
        self._corpus_size = 0

    def build(self, texts: List[str]) -> None:
        from rank_bm25 import BM25Okapi

        tokenized = [tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(tokenized)
        self._corpus_size = len(texts)

    @property
    def is_built(self) -> bool:
        return self._bm25 is not None

    def search(self, query: str, top_k: int = 8) -> List[Tuple[int, float]]:
        """Return [(corpus_index, score), ...] sorted by score descending."""
        if self._bm25 is None or self._corpus_size == 0:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(idx, float(score)) for idx, score in ranked[:top_k] if score > 0]
