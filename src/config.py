"""
Centralized configuration for ResearchGapPilot.

Every model name, path, threshold, and limit referenced elsewhere in the
codebase should come from here, so the whole system's behavior can be tuned
(or ported to different models) from a single file.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = DATA_DIR / "index"
SEED_PAPERS_PATH = DATA_DIR / "seed_papers.jsonl"


@dataclass(frozen=True)
class ModelConfig:
    """
    All models are Hugging Face models runnable on CPU. No paid APIs are used
    anywhere in this project.
    """

    # Sentence embedding model for semantic retrieval (small, CPU-friendly).
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # Natural Language Inference model used for contradiction detection.
    # A small, widely-used MNLI-finetuned model that runs reasonably on CPU.
    nli_model: str = "cross-encoder/nli-deberta-v3-xsmall"

    # Optional cross-encoder reranker (see src/rag/reranker.py). Disabled by
    # default via RetrievalConfig.enable_reranking; only loaded if enabled.
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Text-generation model used for the strictly-grounded final report and
    # for lightweight zero-shot intent classification fallback.
    # flan-t5-small is instruction-tuned, CPU-friendly, and small enough for
    # a free Streamlit Cloud instance.
    generation_model: str = "google/flan-t5-small"

    # Zero-shot classification fallback for ambiguous router inputs.
    zero_shot_model: str = "typeform/mobilebert-uncased-mnli"

    generation_max_new_tokens: int = 256
    generation_temperature: float = 0.3


@dataclass(frozen=True)
class RetrievalConfig:
    top_k_semantic: int = 8
    top_k_bm25: int = 8
    top_k_hybrid: int = 8

    # Weight given to the semantic score vs. BM25 score when combining
    # normalized scores into a single hybrid score. Must sum to 1.0 with
    # bm25_weight below; kept configurable per spec.
    semantic_weight: float = 0.6
    bm25_weight: float = 0.4

    enable_reranking: bool = False
    rerank_top_n: int = 20  # only rerank the top N hybrid candidates

    chunk_size_words: int = 180
    chunk_overlap_words: int = 40

    max_papers: int = 40


@dataclass(frozen=True)
class AgentConfig:
    max_sub_questions: int = 7
    max_queries_per_subquestion: int = 4

    evidence_relevance_threshold: float = 0.35
    evidence_sufficiency_threshold: float = 0.6  # coverage score, 0-1

    max_retrieval_iterations: int = 3
    max_verification_cycles: int = 2

    contradiction_score_threshold: float = 0.55  # NLI "contradiction" label prob


@dataclass(frozen=True)
class RouterConfig:
    # Deterministic keyword sets used before ever invoking a model. These
    # cheaply and reliably catch the overwhelming majority of casual input
    # so the expensive research pipeline is never triggered for "hi".
    greeting_phrases: tuple = (
        "hi", "hello", "hey", "yo", "hiya", "good morning", "good afternoon",
        "good evening", "greetings", "howdy",
    )
    gratitude_phrases: tuple = (
        "thanks", "thank you", "thx", "ty", "appreciate it", "cheers",
    )
    capability_phrases: tuple = (
        "what can you do", "who are you", "what are you", "help",
        "what is this", "how do you work", "what do you do",
    )
    # Very short inputs (<= this many words) skip the research pipeline and
    # are routed to clarification unless they match a research-like pattern.
    min_research_query_words: int = 3

    zero_shot_confidence_floor: float = 0.55


@dataclass(frozen=True)
class AppConfig:
    app_title: str = "🔬 ResearchGapPilot"
    app_subtitle: str = "Agentic RAG for Evidence-Based Research Discovery"
    max_conversation_turns: int = 30


# Allow overriding key model choices via environment variables without
# touching code (e.g. to swap in a smaller/mocked model for testing).
_embedding_override = os.environ.get("RGP_EMBEDDING_MODEL")
_generation_override = os.environ.get("RGP_GENERATION_MODEL")

models = ModelConfig(
    embedding_model=_embedding_override or ModelConfig.embedding_model,
    generation_model=_generation_override or ModelConfig.generation_model,
)
retrieval = RetrievalConfig()
agents = AgentConfig()
router = RouterConfig()
app = AppConfig()
