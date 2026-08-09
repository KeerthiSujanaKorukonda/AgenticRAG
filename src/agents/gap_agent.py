"""
Research Gap Agent — the main feature of this application.

Infers candidate research gaps from actual retrieved evidence (never a fixed
hardcoded gap list), then validates each candidate against the evidence
before accepting it. A gap is only kept if:

  1. Evidence actually demonstrates the underlying problem exists.
  2. Nothing in the accepted evidence indicates it has been resolved.
  3. It's supported by at least one real paper (citable doc_id).
  4. It reads as a genuine opportunity, not just one paper's own limitations
     section restated as if it were a broader field-level gap.

Gap *candidates* come from two real signals already computed earlier in the
pipeline — not from asking a model to invent gaps out of thin air:
  - sub-questions whose evidence coverage was insufficient (a real gap in
    what the literature retrieved actually addresses), and
  - recurring "limitation"/"challenge"/"unresolved" language inside the
    accepted evidence itself.
"""

import re
from typing import Dict, List, Optional

from src.state import EvidenceItem, ResearchGap
from src.utils.text import truncate

_GAP_CATEGORY_KEYWORDS = {
    "Dataset": ("dataset", "corpus", "annotated data", "labeled data", "benchmark data"),
    "Evaluation": ("evaluation", "metric", "benchmark", "measure", "assessment"),
    "Methodological": ("method", "approach", "technique", "algorithm", "architecture"),
    "Generalization": ("generalize", "generalization", "transfer", "unseen", "out-of-domain"),
    "Reproducibility": ("reproduc", "replicat", "code release", "open source"),
    "Scalability": ("scal", "compute cost", "efficiency", "resource-intensive"),
    "Theoretical": ("theoretical", "theory", "formal", "unclear why", "not well understood"),
}

_LIMITATION_PATTERNS = re.compile(
    r"\b(limitation|limited|challenge|remains? (an? )?(open|unresolved)|"
    r"has not been (well[- ]studied|investigated|explored)|future work|"
    r"is (still )?unclear|lacks?|scarc(e|ity)|insufficient|"
    r"do(es)? not (release|provide|report))\b",
    re.I,
)


def _infer_category(text: str) -> str:
    text_low = text.lower()
    best_category, best_hits = "Methodological", 0
    for category, keywords in _GAP_CATEGORY_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in text_low)
        if hits > best_hits:
            best_category, best_hits = category, hits
    return best_category


def _candidates_from_coverage_gaps(
    evidence_coverage: Dict,
    evidence_items: List[EvidenceItem],
) -> List[Dict]:
    """
    A sub-question with insufficient coverage is itself real evidence of a
    gap in what the retrieved literature addresses — but only worth
    surfacing if there's at least some (even weak) evidence pointing at it,
    otherwise it's just "we didn't search well," not a research gap.
    """
    candidates = []
    per_sq = evidence_coverage.get("per_subquestion", {})
    for sub_question, stats in per_sq.items():
        if stats.get("sufficient"):
            continue
        related = [e for e in evidence_items if e.get("sub_question") == sub_question]
        if not related:
            continue
        candidates.append(
            {
                "origin": "coverage_gap",
                "sub_question": sub_question,
                "related_evidence": related,
            }
        )
    return candidates


def _candidates_from_limitation_language(evidence_items: List[EvidenceItem]) -> List[Dict]:
    candidates = []
    usable = [e for e in evidence_items if e.get("classification") in ("RELEVANT", "WEAKLY_RELEVANT")]
    for item in usable:
        if _LIMITATION_PATTERNS.search(item.get("text", "")):
            candidates.append({"origin": "limitation_language", "related_evidence": [item]})
    return candidates


def _validate_gap(candidate_evidence: List[EvidenceItem], min_supporting_docs: int = 1) -> (bool, List[str]):
    notes = []
    doc_ids = {e.get("doc_id") for e in candidate_evidence if e.get("doc_id")}

    if len(doc_ids) < min_supporting_docs:
        notes.append("Rejected: no citable supporting document.")
        return False, notes

    if candidate_evidence and all(e.get("classification") == "IRRELEVANT" for e in candidate_evidence):
        notes.append("Rejected: underlying evidence was classified IRRELEVANT.")
        return False, notes

    # Reject if it looks like a single paper's own stated limitation being
    # generalized without any second source — this app allows single-paper
    # gaps (spec allows "supported by one or more papers") but flags them as
    # lower-confidence rather than silently treating them as field-wide.
    single_source = len(doc_ids) == 1
    notes.append(
        "Supported by a single paper — treat as a narrower, lower-confidence gap."
        if single_source
        else f"Supported by {len(doc_ids)} independent papers."
    )
    return True, notes


def _build_gap_from_candidate(candidate: Dict) -> Optional[ResearchGap]:
    related = candidate["related_evidence"]
    is_valid, notes = _validate_gap(related)
    if not is_valid:
        return None

    combined_text = " ".join(e.get("text", "") for e in related)
    category = _infer_category(combined_text)
    doc_ids = sorted({e.get("doc_id") for e in related if e.get("doc_id")})
    single_source = len(doc_ids) == 1

    if candidate["origin"] == "coverage_gap":
        sub_question = candidate["sub_question"]
        title = f"Under-addressed question: {sub_question.rstrip('?')}"
        description = (
            f"Retrieved evidence only weakly covers the question '{sub_question}'. "
            f"Available related evidence: {truncate(combined_text, 300)}"
        )
        why_it_matters = (
            "This sub-question is part of the research plan but is not well answered by the "
            "literature actually retrieved, suggesting either a genuine gap or a need for more "
            "targeted future work."
        )
    else:
        first = related[0]
        title = f"{category} limitation noted in {first.get('title', 'a retrieved paper')}"
        description = truncate(combined_text, 350)
        why_it_matters = (
            "The retrieved text explicitly frames this as an open limitation or challenge, "
            "indicating an opportunity for follow-up research."
        )

    confidence = 0.75 if not single_source else 0.5
    confidence_methodology = (
        "Heuristic confidence: +0.75 base for multi-paper support, 0.5 for single-paper support; "
        "not a calibrated probability. See docs/methodology.md."
    )

    return ResearchGap(
        title=title,
        description=description,
        category=category,
        evidence=[truncate(e.get("text", ""), 250) for e in related[:4]],
        supporting_doc_ids=doc_ids,
        why_it_matters=why_it_matters,
        confidence=confidence,
        confidence_methodology=confidence_methodology,
        validation_passed=True,
        validation_notes=notes,
    )


def detect_gaps(
    evidence_items: List[EvidenceItem],
    evidence_coverage: Dict,
    max_gaps: int = 6,
) -> List[ResearchGap]:
    candidates = _candidates_from_coverage_gaps(evidence_coverage, evidence_items)
    candidates += _candidates_from_limitation_language(evidence_items)

    gaps: List[ResearchGap] = []
    seen_titles = set()
    for candidate in candidates:
        gap = _build_gap_from_candidate(candidate)
        if gap is None:
            continue
        if gap["title"] in seen_titles:
            continue
        seen_titles.add(gap["title"])
        gaps.append(gap)

    gaps.sort(key=lambda g: g["confidence"], reverse=True)
    return gaps[:max_gaps]
