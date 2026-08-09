"""
Research Theme Detection Agent.

Groups accepted evidence into themes using unsupervised clustering over the
same embeddings already computed for retrieval (no separate/duplicate model
needed). Falls back to grouping by sub_question if too few evidence items
exist for meaningful clustering — themes are always derived from the actual
retrieved evidence, never a fixed hardcoded list.
"""

from collections import defaultdict
from typing import List, Optional

import numpy as np

from src.state import EvidenceItem, ResearchTheme
from src.utils.text import truncate


def _cluster_labels(embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
    from sklearn.cluster import KMeans

    n_clusters = max(1, min(n_clusters, len(embeddings)))
    if n_clusters == 1:
        return np.zeros(len(embeddings), dtype=int)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    return km.fit_predict(embeddings)


def detect_themes(
    evidence_items: List[EvidenceItem],
    embedding_model=None,
    target_theme_count: int = 4,
) -> List[ResearchTheme]:
    usable = [e for e in evidence_items if e.get("classification") in ("RELEVANT", "WEAKLY_RELEVANT")]
    if not usable:
        return []

    if embedding_model is not None and len(usable) >= 3:
        try:
            texts = [e["text"] for e in usable]
            embeddings = embedding_model.embed(texts)
            n_clusters = min(target_theme_count, len(usable))
            labels = _cluster_labels(embeddings, n_clusters)
        except Exception:
            labels = None
    else:
        labels = None

    groups = defaultdict(list)
    if labels is not None:
        for item, label in zip(usable, labels):
            groups[int(label)].append(item)
    else:
        # Fallback: group by sub_question — still dynamic, just coarser.
        for item in usable:
            groups[item.get("sub_question", "general")].append(item)

    themes: List[ResearchTheme] = []
    for group_items in groups.values():
        if not group_items:
            continue
        # Theme title: the most frequent meaningful words across the group's
        # evidence text, rather than a hardcoded label.
        title = _summarize_group_title(group_items)
        doc_ids = sorted({item["doc_id"] for item in group_items if item.get("doc_id")})
        description = truncate(
            " ".join(item["text"] for item in group_items[:2]), 280
        )
        themes.append(
            ResearchTheme(
                title=title,
                description=description,
                supporting_doc_ids=doc_ids,
                evidence_count=len(group_items),
            )
        )

    themes.sort(key=lambda t: t["evidence_count"], reverse=True)
    return themes


def _summarize_group_title(items: List[EvidenceItem], top_n_words: int = 3) -> str:
    import re
    from collections import Counter

    stop = {
        "the", "and", "for", "with", "that", "this", "from", "have", "are",
        "was", "were", "been", "into", "such", "these", "those", "which",
        "their", "also", "than", "when", "while", "over", "more", "most",
    }
    words = []
    for item in items:
        words.extend(w for w in re.findall(r"[a-zA-Z]{4,}", item["text"].lower()) if w not in stop)

    if not words:
        return "Emerging Theme"

    counts = Counter(words)
    top_words = [w for w, _ in counts.most_common(top_n_words)]
    return " / ".join(w.capitalize() for w in top_words)
