"""
LangGraph workflow — wires every agent into a real state graph with
conditional edges, exactly matching the architecture in the project spec:

    intent_router -> planner -> query_generator -> retriever ->
    evidence_evaluator -> contradiction_detector -> sufficiency_decision ->
    [adaptive_retrieval loop] -> reasoning -> theme_detector -> gap_detector
    -> verification -> [adaptive_retrieval loop] -> final_generator

A separate, cheaper path handles GREETING/CAPABILITIES/CLARIFICATION/
UNSUPPORTED_REQUEST (conversational_reply) and FOLLOW_UP (follow_up_handler)
without ever touching retrieval/embeddings/generation unless genuinely
needed.

Every node is a plain function of (state) -> partial_state_update, which is
what makes this unit-testable: any node can be called directly with a hand-
built state dict in tests without spinning up the whole graph or any model.
"""

import time
from typing import Callable, Optional

from langgraph.graph import END, StateGraph

from src.agents import (
    contradiction_agent,
    evidence_agent,
    gap_agent,
    planner,
    query_generator,
    reasoning_agent,
    retrieval_agent,
    theme_agent,
    verification_agent,
)
from src.config import agents as agents_cfg
from src.graph.routing import (
    route_after_follow_up,
    route_after_intent,
    route_after_sufficiency,
    route_after_verification,
)
from src.router import route_intent
from src.state import ResearchState
from src.utils.citations import build_citations
from src.utils.logging import append_event

CAPABILITIES_TEXT = (
    "I can investigate research questions by:\n\n"
    "• planning a research strategy\n"
    "• searching real research documents\n"
    "• combining semantic and keyword retrieval\n"
    "• evaluating evidence\n"
    "• detecting contradictory findings\n"
    "• performing additional retrieval when evidence is insufficient\n"
    "• identifying research themes\n"
    "• discovering evidence-backed research gaps\n"
    "• verifying claims and citations"
)

GREETING_TEXT = (
    "Hi! I'm ResearchGapPilot.\n\n"
    "I can investigate research questions, compare papers, identify "
    "evidence-backed research gaps, detect conflicting findings, and "
    "generate cited research reports.\n\n"
    "What research topic would you like me to investigate?"
)

UNSUPPORTED_TEXT = (
    "I'm designed primarily for research discovery and evidence analysis. "
    "Please give me a research question or research topic."
)

DANGLING_FOLLOW_UP_TEXT = (
    "Please start a research question first so I can identify the research gaps."
)


class WorkflowDependencies:
    """
    Bundles every model/retriever the graph's nodes need. Passing this in
    explicitly (rather than importing global singletons inside each node)
    is what makes it possible to build the graph once in app.py with real,
    cached models, and separately build it in tests with lightweight stubs.
    """

    def __init__(
        self,
        retriever=None,
        embedding_model=None,
        generation_model=None,
        nli_model=None,
        reranker=None,
        zero_shot_classifier: Optional[Callable] = None,
    ):
        self.retriever = retriever
        self.embedding_model = embedding_model
        self.generation_model = generation_model
        self.nli_model = nli_model
        self.reranker = reranker
        self.zero_shot_classifier = zero_shot_classifier


def _timed(state: ResearchState, node_name: str, start: float) -> dict:
    timings = dict(state.get("node_timings_ms", {}))
    timings[node_name] = round((time.time() - start) * 1000, 1)
    return {"node_timings_ms": timings}


def build_graph(deps: WorkflowDependencies):
    """Build and compile the LangGraph StateGraph. Called once and cached
    by the app layer (see app.py's @st.cache_resource)."""

    # ---- Node implementations -------------------------------------------

    def intent_router_node(state: ResearchState) -> dict:
        start = time.time()
        result = route_intent(
            state["user_input"],
            has_prior_research_state=bool(state.get("research_question")),
            zero_shot_classifier=deps.zero_shot_classifier,
        )
        events = append_event(
            state.get("agent_events", []), "Intent Router", "🧭",
            f"Classified as {result['intent']} (confidence {result['confidence']:.2f}): {result['reason']}",
        )
        return {
            "intent": result["intent"],
            "intent_confidence": result["confidence"],
            "intent_reason": result["reason"],
            "agent_events": events,
            **_timed(state, "intent_router", start),
        }

    def conversational_reply_node(state: ResearchState) -> dict:
        intent = state.get("intent")
        if intent == "GREETING":
            reply = GREETING_TEXT
        elif intent == "CAPABILITIES":
            reply = CAPABILITIES_TEXT
        elif intent == "UNSUPPORTED_REQUEST":
            reply = UNSUPPORTED_TEXT
        elif state.get("intent_reason", "").startswith("dangling_follow_up_reference"):
            reply = DANGLING_FOLLOW_UP_TEXT
        else:
            reply = (
                "I didn't quite catch a research question there. Could you rephrase, "
                "or ask me a specific research question to investigate?"
            )
        return {"conversational_reply": reply, "final_answer": reply}

    def follow_up_handler_node(state: ResearchState) -> dict:
        """
        Uses the existing research state. Decides whether current evidence
        already answers the follow-up, or whether targeted additional
        retrieval is needed, based on real keyword overlap between the
        follow-up text and existing evidence — not a fixed guess.
        """
        follow_up_text = state["user_input"].lower()
        existing_evidence = state.get("evidence_items", [])

        overlapping = [
            e for e in existing_evidence
            if e.get("classification") in ("RELEVANT", "WEAKLY_RELEVANT")
            and any(w in e.get("text", "").lower() for w in follow_up_text.split() if len(w) > 4)
        ]

        needs_more = len(overlapping) == 0 and bool(existing_evidence)
        events = append_event(
            state.get("agent_events", []), "Follow-up Handler", "🔁",
            f"Found {len(overlapping)} existing evidence item(s) relevant to the follow-up."
            + (" Will retrieve additional evidence." if needs_more else ""),
        )
        return {
            "needs_more_evidence": needs_more,
            "research_question": follow_up_text,  # drives reasoning/retrieval below
            "agent_events": events,
        }

    def planner_node(state: ResearchState) -> dict:
        start = time.time()
        question = state["user_input"]
        plan = planner.create_research_plan(question, generation_model=deps.generation_model)
        events = append_event(
            state.get("agent_events", []), "Planner", "🧠",
            f"Created {len(plan['sub_questions'])} sub-questions"
            + (" (model-generated)" if plan["used_model"] else " (deterministic fallback)"),
        )
        return {
            "research_question": question,
            "research_plan": plan,
            "sub_questions": plan["sub_questions"],
            "iteration": 0,
            "max_iterations": state.get("max_iterations", agents_cfg.max_retrieval_iterations),
            "verification_cycle": 0,
            "max_verification_cycles": agents_cfg.max_verification_cycles,
            "agent_events": events,
            **_timed(state, "planner", start),
        }

    def query_generator_node(state: ResearchState) -> dict:
        start = time.time()
        sub_questions = state.get("sub_questions", [])
        research_question = state.get("research_question", "")
        queries_by_sq = {
            sq: query_generator.generate_queries_for_subquestion(sq, research_question)
            for sq in sub_questions
        }
        all_queries = [q for qs in queries_by_sq.values() for q in qs]
        events = append_event(
            state.get("agent_events", []), "Query Generator", "🔎",
            f"Generated {len(all_queries)} research queries across {len(sub_questions)} sub-questions",
        )
        return {
            "retrieval_queries": all_queries,
            "queries_by_subquestion": queries_by_sq,
            "agent_events": events,
            **_timed(state, "query_generator", start),
        }

    def retriever_node(state: ResearchState) -> dict:
        start = time.time()
        queries_by_sq = state.get("queries_by_subquestion", {})
        sub_questions = state.get("sub_questions", [])

        if deps.retriever is None or not deps.retriever.is_ready:
            events = append_event(
                state.get("agent_events", []), "Retriever", "📚",
                "No prepared retrieval index is available — 0 candidate passages retrieved.",
            )
            return {"retrieved_documents": [], "agent_events": events, **_timed(state, "retriever", start)}

        hits = retrieval_agent.retrieve_for_sub_questions(
            deps.retriever, sub_questions, queries_by_sq,
            reranker=deps.reranker,
        )
        events = append_event(
            state.get("agent_events", []), "Retriever", "📚",
            f"Retrieved {len(hits)} candidate passages",
        )
        return {"retrieved_documents": hits, "agent_events": events, **_timed(state, "retriever", start)}

    def evidence_evaluator_node(state: ResearchState) -> dict:
        start = time.time()
        chunks = state.get("retrieved_documents", [])
        new_evidence = evidence_agent.evaluate_chunks(chunks, nli_model=deps.nli_model)

        # Merge with any evidence already accumulated from prior iterations
        # (dedup by chunk_id) rather than discarding it.
        existing = state.get("evidence_items", [])
        seen = {e["chunk_id"] for e in existing}
        merged = existing + [e for e in new_evidence if e["chunk_id"] not in seen]

        coverage = evidence_agent.compute_evidence_coverage(state.get("sub_questions", []), merged)

        events = append_event(
            state.get("agent_events", []), "Evidence Agent", "🧪",
            f"Evaluated {len(chunks)} passages "
            f"({sum(1 for e in new_evidence if e['classification']=='RELEVANT')} newly relevant); "
            f"overall coverage {coverage['overall_coverage_score']:.2f}",
        )
        return {
            "evidence_items": merged,
            "evidence_coverage": coverage,
            "agent_events": events,
            **_timed(state, "evidence_evaluator", start),
        }

    def contradiction_detector_node(state: ResearchState) -> dict:
        start = time.time()
        evidence = state.get("evidence_items", [])
        contradictions = contradiction_agent.detect_contradictions(evidence, nli_model=deps.nli_model)
        events = append_event(
            state.get("agent_events", []), "Contradiction Detector", "⚠️",
            f"Detected {len(contradictions)} conflicting evidence group(s)"
            if contradictions else "No conflicting evidence groups detected",
        )
        return {"contradictions": contradictions, "agent_events": events, **_timed(state, "contradiction_detector", start)}

    def adaptive_retrieval_node(state: ResearchState) -> dict:
        start = time.time()
        coverage = state.get("evidence_coverage", {}) or {}
        missing_sqs = coverage.get("missing_subquestions", [])
        iteration = state.get("iteration", 0) + 1

        missing_queries = []
        for sq in missing_sqs:
            missing_queries.extend(query_generator.generate_queries_for_subquestion(sq, state.get("research_question", "")))

        already_seen = {c["chunk_id"] for c in state.get("retrieved_documents", [])}
        new_hits = []
        if deps.retriever is not None and deps.retriever.is_ready and missing_queries:
            new_hits = retrieval_agent.retrieve_additional(deps.retriever, missing_queries, already_seen)

        events = append_event(
            state.get("agent_events", []), "Adaptive Retrieval", "🔄",
            f"Evidence insufficient for {len(missing_sqs)} sub-question(s); retrieved {len(new_hits)} additional passage(s) (iteration {iteration})",
        )
        merged_docs = state.get("retrieved_documents", []) + new_hits
        return {
            "retrieved_documents": merged_docs,
            "iteration": iteration,
            "verification_cycle": state.get("verification_cycle", 0),  # unchanged here
            "agent_events": events,
            **_timed(state, f"adaptive_retrieval_{iteration}", start),
        }

    def reasoning_node(state: ResearchState) -> dict:
        start = time.time()
        evidence = state.get("evidence_items", [])
        contradictions = state.get("contradictions", [])
        answer = reasoning_agent.synthesize_answer(
            state.get("research_question", state.get("user_input", "")),
            evidence, contradictions,
            generation_model=deps.generation_model,
        )
        events = append_event(
            state.get("agent_events", []), "Reasoning", "🧠",
            f"Synthesized evidence across {len({e['doc_id'] for e in evidence if e.get('doc_id')})} papers",
        )
        return {"final_answer": answer, "agent_events": events, **_timed(state, "reasoning", start)}

    def theme_detector_node(state: ResearchState) -> dict:
        start = time.time()
        themes = theme_agent.detect_themes(state.get("evidence_items", []), embedding_model=deps.embedding_model)
        events = append_event(
            state.get("agent_events", []), "Theme Detector", "🗂️",
            f"Identified {len(themes)} research theme(s)",
        )
        return {"research_themes": themes, "agent_events": events, **_timed(state, "theme_detector", start)}

    def gap_detector_node(state: ResearchState) -> dict:
        start = time.time()
        gaps = gap_agent.detect_gaps(state.get("evidence_items", []), state.get("evidence_coverage", {}))
        events = append_event(
            state.get("agent_events", []), "Gap Agent", "🔬",
            f"Generated {len(gaps)} validated candidate gap(s)",
        )
        return {"research_gaps": gaps, "agent_events": events, **_timed(state, "gap_detector", start)}

    def verification_node(state: ResearchState) -> dict:
        start = time.time()
        results = verification_agent.verify_gaps_and_answer(
            state.get("final_answer", ""),
            state.get("research_gaps", []),
            state.get("evidence_items", []),
            nli_model=deps.nli_model,
        )
        cycle = state.get("verification_cycle", 0) + 1
        events = append_event(
            state.get("agent_events", []), "Verification", "✅",
            f"Verified {len(results)} claim(s); "
            f"{sum(1 for r in results if r.get('supported'))} supported / {len(results)} total (cycle {cycle})",
        )
        return {
            "verification_results": results,
            "verification_cycle": cycle,
            "agent_events": events,
            **_timed(state, f"verification_{cycle}", start),
        }

    def final_generator_node(state: ResearchState) -> dict:
        start = time.time()
        evidence = state.get("evidence_items", [])
        relevant_chunks = [
            {"doc_id": e["doc_id"], "title": e["title"], "url": e.get("url", ""), "authors": [], "year": None}
            for e in evidence if e.get("classification") in ("RELEVANT", "WEAKLY_RELEVANT")
        ]
        citations = build_citations(relevant_chunks)

        coverage = state.get("evidence_coverage", {})
        verification_results = state.get("verification_results", [])
        unsupported = [r for r in verification_results if not r.get("supported")]

        sections = {
            "executive_summary": state.get("final_answer", ""),
            "research_plan": "\n".join(f"- {sq}" for sq in state.get("sub_questions", [])),
            "research_themes": "\n".join(f"- **{t['title']}** ({t['evidence_count']} evidence items)" for t in state.get("research_themes", [])),
            "conflicting_evidence": "\n".join(
                f"- {c['source_a']} vs {c['source_b']} (score {c['nli_score']}): {c['likely_reason']}"
                for c in state.get("contradictions", [])
            ) or "No conflicting evidence detected.",
            "research_gaps": "\n".join(f"- **{g['title']}** ({g['category']}, confidence {g['confidence']})" for g in state.get("research_gaps", [])),
            "evidence_coverage": f"Overall coverage score: {coverage.get('overall_coverage_score', 'Not evaluated')}",
            "retrieval_iterations": str(state.get("iteration", 0)),
            "verification": (
                f"{len(verification_results) - len(unsupported)}/{len(verification_results)} claims verified as supported."
                if verification_results else "Not evaluated"
            ),
            "sources": "\n".join(f"[{c['citation_id']}] {c['title']}" for c in citations) or "No citable sources.",
        }

        events = append_event(state.get("agent_events", []), "Final Generator", "📄", "Assembled final grounded report")
        return {
            "final_report_sections": sections,
            "citations": citations,
            "agent_events": events,
            **_timed(state, "final_generator", start),
        }

    # ---- Graph wiring ----------------------------------------------------

    graph = StateGraph(ResearchState)

    graph.add_node("intent_router", intent_router_node)
    graph.add_node("conversational_reply", conversational_reply_node)
    graph.add_node("follow_up_handler", follow_up_handler_node)
    graph.add_node("planner", planner_node)
    graph.add_node("query_generator", query_generator_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("evidence_evaluator", evidence_evaluator_node)
    graph.add_node("contradiction_detector", contradiction_detector_node)
    graph.add_node("adaptive_retrieval", adaptive_retrieval_node)
    graph.add_node("reasoning", reasoning_node)
    graph.add_node("theme_detector", theme_detector_node)
    graph.add_node("gap_detector", gap_detector_node)
    graph.add_node("verification", verification_node)
    graph.add_node("final_generator", final_generator_node)

    graph.set_entry_point("intent_router")

    graph.add_conditional_edges(
        "intent_router",
        route_after_intent,
        {
            "planner": "planner",
            "follow_up_handler": "follow_up_handler",
            "conversational_reply": "conversational_reply",
        },
    )
    graph.add_edge("conversational_reply", END)

    graph.add_conditional_edges(
        "follow_up_handler",
        route_after_follow_up,
        {"adaptive_retrieval": "adaptive_retrieval", "reasoning": "reasoning"},
    )

    graph.add_edge("planner", "query_generator")
    graph.add_edge("query_generator", "retriever")
    graph.add_edge("retriever", "evidence_evaluator")
    graph.add_edge("evidence_evaluator", "contradiction_detector")

    graph.add_conditional_edges(
        "contradiction_detector",
        route_after_sufficiency,
        {"adaptive_retrieval": "adaptive_retrieval", "reasoning": "reasoning"},
    )
    graph.add_edge("adaptive_retrieval", "evidence_evaluator")

    graph.add_edge("reasoning", "theme_detector")
    graph.add_edge("theme_detector", "gap_detector")
    graph.add_edge("gap_detector", "verification")

    graph.add_conditional_edges(
        "verification",
        route_after_verification,
        {"adaptive_retrieval": "adaptive_retrieval", "final_generator": "final_generator"},
    )
    graph.add_edge("final_generator", END)

    return graph.compile()
