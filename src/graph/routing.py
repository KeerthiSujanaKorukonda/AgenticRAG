"""
Conditional edge logic for the LangGraph workflow.

Kept separate from workflow.py so the branching decisions themselves — the
part most likely to need tuning — are easy to find and unit-test in
isolation from the (heavier, model-touching) node implementations.
"""

from src.config import agents as agents_cfg
from src.state import ResearchState


def route_after_intent(state: ResearchState) -> str:
    intent = state.get("intent", "CLARIFICATION")
    if intent == "RESEARCH_QUERY":
        return "planner"
    if intent == "FOLLOW_UP":
        return "follow_up_handler"
    # GREETING, CAPABILITIES, CLARIFICATION, UNSUPPORTED_REQUEST all resolve
    # to a direct conversational reply with no pipeline execution.
    return "conversational_reply"


def route_after_sufficiency(state: ResearchState) -> str:
    """After evidence_evaluator + sufficiency check: continue reasoning, or
    loop back for another retrieval iteration (bounded)."""
    coverage = state.get("evidence_coverage", {}) or {}
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", agents_cfg.max_retrieval_iterations)

    if coverage.get("is_sufficient", False):
        return "reasoning"
    if iteration >= max_iterations:
        # Out of iterations — proceed anyway with whatever evidence exists;
        # the reasoning/verification agents will honestly report
        # insufficient evidence where it matters rather than block forever.
        return "reasoning"
    return "adaptive_retrieval"


def route_after_verification(state: ResearchState) -> str:
    """After verification: accept the report, or loop back for more
    evidence/reasoning (bounded)."""
    results = state.get("verification_results", [])
    cycle = state.get("verification_cycle", 0)
    max_cycles = state.get("max_verification_cycles", agents_cfg.max_verification_cycles)

    any_failed = any(not r.get("supported", False) for r in results)

    if not any_failed:
        return "final_generator"
    if cycle >= max_cycles:
        return "final_generator"  # give up gracefully; final report will flag unsupported claims
    return "adaptive_retrieval"


def route_after_follow_up(state: ResearchState) -> str:
    """After follow-up handling determines whether existing evidence
    suffices or more retrieval is needed."""
    if state.get("needs_more_evidence", False):
        return "adaptive_retrieval"
    return "reasoning"
