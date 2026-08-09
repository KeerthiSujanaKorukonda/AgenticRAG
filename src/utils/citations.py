"""
Citation construction and validation.

Per the project spec, no citation should ever be shown unless it maps to a
real retrieved document with real metadata. This module is the single choke
point through which citations are created, so nothing downstream can
fabricate one.
"""

from typing import Dict, List, Optional

from src.state import Citation, RetrievedChunk


def build_citation(chunk: RetrievedChunk, index: int) -> Optional[Citation]:
    """
    Build a Citation from a retrieved chunk, or return None if the chunk is
    missing the minimum required metadata (doc_id + title). A citation with
    no verifiable source is worse than no citation at all.
    """
    doc_id = chunk.get("doc_id")
    title = chunk.get("title")
    if not doc_id or not title:
        return None

    return Citation(
        citation_id=f"C{index}",
        doc_id=doc_id,
        title=title,
        url=chunk.get("url", ""),
        authors=chunk.get("authors", []) or [],
        year=chunk.get("year"),
    )


def build_citations(chunks: List[RetrievedChunk]) -> List[Citation]:
    """Build a deduplicated citation list (one citation per unique doc_id)."""
    seen_doc_ids = set()
    citations: List[Citation] = []
    idx = 1
    for chunk in chunks:
        doc_id = chunk.get("doc_id")
        if not doc_id or doc_id in seen_doc_ids:
            continue
        citation = build_citation(chunk, idx)
        if citation is None:
            continue
        seen_doc_ids.add(doc_id)
        citations.append(citation)
        idx += 1
    return citations


def validate_citation_map(citations: List[Citation], known_doc_ids: set) -> List[Citation]:
    """
    Drop any citation whose doc_id doesn't correspond to a document that was
    actually retrieved in this session. Defends against a citation ID being
    referenced without a backing document (e.g. from a stale/edited state).
    """
    return [c for c in citations if c.get("doc_id") in known_doc_ids]


def format_citation_line(citation: Citation) -> str:
    authors = citation.get("authors") or []
    author_str = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
    year = citation.get("year")
    year_str = f" ({year})" if year else ""
    url = citation.get("url", "")
    url_str = f" — {url}" if url else ""
    return f"[{citation['citation_id']}] {citation['title']}{year_str} — {author_str}{url_str}"
