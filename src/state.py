"""
Shared state definitions for ResearchGapPilot.

`ResearchState` is the single object threaded through every LangGraph node.
Nodes read what they need and return a partial dict of updates (LangGraph
merges partial updates into the running state), so no node needs to know
about fields it doesn't use.
"""

from typing import Any, Dict, List, Optional, TypedDict


class Document(TypedDict, total=False):
    doc_id: str
    title: str
    authors: List[str]
    year: Optional[int]
    source: str
    url: str
    text: str  # the chunk's text
    topic: Optional[str]


class RetrievedChunk(TypedDict, total=False):
    doc_id: str
    chunk_id: str
    title: str
    authors: List[str]
    year: Optional[int]
    url: str
    source: str
    text: str
    semantic_score: float
    bm25_score: float
    hybrid_score: float
    rerank_score: Optional[float]
    sub_question: str
    query: str


class EvidenceItem(TypedDict, total=False):
    chunk_id: str
    doc_id: str
    title: str
    url: str
    text: str
    year: Optional[int]
    sub_question: str
    classification: str  # RELEVANT | WEAKLY_RELEVANT | IRRELEVANT | CONTRADICTORY
    relevance_score: float
    reasons: List[str]


class Contradiction(TypedDict, total=False):
    statement_a: str
    source_a: str
    statement_b: str
    source_b: str
    nli_label: str  # CONTRADICTION
    nli_score: float
    likely_reason: str


class ResearchTheme(TypedDict, total=False):
    title: str
    description: str
    supporting_doc_ids: List[str]
    evidence_count: int


class ResearchGap(TypedDict, total=False):
    title: str
    description: str
    category: str  # Methodological | Dataset | Evaluation | ...
    evidence: List[str]
    supporting_doc_ids: List[str]
    why_it_matters: str
    confidence: float
    confidence_methodology: str
    validation_passed: bool
    validation_notes: List[str]


class VerificationResult(TypedDict, total=False):
    claim: str
    supported: bool
    supporting_doc_ids: List[str]
    nli_label: str
    nli_score: float
    notes: str


class Citation(TypedDict, total=False):
    citation_id: str
    doc_id: str
    title: str
    url: str
    authors: List[str]
    year: Optional[int]


class AgentEvent(TypedDict, total=False):
    agent: str
    icon: str
    message: str
    timestamp: float


class ResearchState(TypedDict, total=False):
    # --- Routing / conversation ---
    user_input: str
    intent: str
    intent_confidence: float
    intent_reason: str
    conversation_history: List[Dict[str, str]]
    conversational_reply: Optional[str]

    # --- Research question & plan ---
    research_question: str
    research_plan: Dict[str, Any]
    sub_questions: List[str]

    # --- Retrieval ---
    retrieval_queries: List[str]
    queries_by_subquestion: Dict[str, List[str]]
    retrieved_documents: List[RetrievedChunk]

    # --- Evidence ---
    evidence_items: List[EvidenceItem]
    contradictions: List[Contradiction]
    evidence_coverage: Dict[str, Any]

    # --- Reasoning outputs ---
    research_themes: List[ResearchTheme]
    research_gaps: List[ResearchGap]
    verification_results: List[VerificationResult]
    citations: List[Citation]

    # --- Control flow ---
    iteration: int
    max_iterations: int
    verification_cycle: int
    max_verification_cycles: int
    needs_more_evidence: bool
    missing_evidence_notes: List[str]

    # --- Output ---
    final_answer: str
    final_report_sections: Dict[str, str]
    agent_events: List[AgentEvent]

    # --- Timing ---
    node_timings_ms: Dict[str, float]
