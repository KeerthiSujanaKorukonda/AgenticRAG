"""
Real tests for the hybrid retriever, using a deterministic fake embedding
model (word-overlap vectors) instead of a real downloaded model, so these
tests run fast and offline. BM25 and FAISS are the REAL libraries — only
the embedding *model* is faked.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.embeddings import EmbeddingModel
from src.rag.hybrid_retriever import HybridRetriever


class FakeEmbeddingModel(EmbeddingModel):
    VOCAB = ["cat", "dog", "multilingual", "nlp", "retrieval", "hallucination", "gap", "dataset"]

    def load(self):
        self._model = "fake"

    def embed(self, texts, batch_size=32, normalize=True):
        vecs = []
        for t in texts:
            t_low = t.lower()
            v = np.array([1.0 if w in t_low else 0.0 for w in self.VOCAB], dtype="float32")
            n = np.linalg.norm(v)
            vecs.append(v / n if n > 0 else v)
        return np.stack(vecs)

    @property
    def dimension(self):
        return len(self.VOCAB)


CHUNKS = [
    {
        "doc_id": "d1", "chunk_id": "d1-0", "title": "Multilingual NLP Survey",
        "text": "This paper surveys multilingual nlp datasets and gaps.",
        "authors": ["A"], "year": 2024, "url": "http://d1", "source": "arXiv",
    },
    {
        "doc_id": "d2", "chunk_id": "d2-0", "title": "Hallucination Survey",
        "text": "This paper surveys hallucination detection in retrieval systems.",
        "authors": ["B"], "year": 2023, "url": "http://d2", "source": "arXiv",
    },
    {
        "doc_id": "d3", "chunk_id": "d3-0", "title": "Cats and Dogs",
        "text": "A paper about cat and dog behavior, unrelated to research.",
        "authors": ["C"], "year": 2022, "url": "http://d3", "source": "arXiv",
    },
]


@pytest.fixture
def retriever():
    r = HybridRetriever(embedding_model=FakeEmbeddingModel())
    r.build(CHUNKS)
    return r


def test_build_and_is_ready(retriever):
    assert retriever.is_ready


def test_relevant_result_ranks_highest(retriever):
    results = retriever.search("multilingual nlp dataset gap", top_k=3)
    assert results[0]["title"] == "Multilingual NLP Survey"
    assert results[0]["hybrid_score"] >= results[-1]["hybrid_score"]


def test_unrelated_query_scores_low(retriever):
    results = retriever.search("multilingual nlp dataset gap", top_k=3)
    cats_result = next(r for r in results if r["title"] == "Cats and Dogs")
    top_result = results[0]
    assert cats_result["hybrid_score"] < top_result["hybrid_score"]


def test_save_and_load_round_trip(tmp_path, retriever):
    save_dir = tmp_path / "index"
    retriever.save(save_dir)

    reloaded = HybridRetriever(embedding_model=FakeEmbeddingModel())
    reloaded.load(save_dir)

    assert reloaded.is_ready
    results = reloaded.search("multilingual nlp")
    assert results[0]["title"] == "Multilingual NLP Survey"


def test_empty_retriever_returns_no_results():
    r = HybridRetriever(embedding_model=FakeEmbeddingModel())
    assert not r.is_ready
    assert r.search("anything") == []
