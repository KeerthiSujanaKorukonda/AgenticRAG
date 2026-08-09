"""Text cleaning, normalization, and chunking utilities."""

import re
from typing import List


def clean_text(text: str) -> str:
    """Collapse whitespace and strip control characters from raw text."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    return text.strip()


def chunk_text(text: str, chunk_size_words: int = 180, overlap_words: int = 40) -> List[str]:
    """
    Split text into overlapping word-count-based chunks.

    Overlap helps avoid losing context at chunk boundaries during retrieval.
    """
    words = clean_text(text).split(" ")
    if not words or words == [""]:
        return []

    if len(words) <= chunk_size_words:
        return [" ".join(words)]

    chunks = []
    step = max(1, chunk_size_words - overlap_words)
    for start in range(0, len(words), step):
        chunk_words = words[start : start + chunk_size_words]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        if start + chunk_size_words >= len(words):
            break
    return chunks


def truncate(text: str, max_chars: int = 400) -> str:
    text = text or ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def word_count(text: str) -> int:
    return len((text or "").split())


def normalize_query(text: str) -> str:
    return clean_text(text).lower()
