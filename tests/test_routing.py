"""Unit tests for the conditional-edge routing functions in src/graph/routing.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.graph.routing import (
    route_after_follow_up,
    route_after_intent,
    route_after_sufficiency,
    route_after_verification,
)


def test_route_after_intent():
    assert route_after_intent({"intent": "RESEARCH_QUERY"}) == "planner"
    assert route_after_intent({"intent": "FOLLOW_UP"}) == "follow_up_handler"
    assert route_after_intent({"intent": "GREETING"}) == "conversational_reply"
    assert route_after_intent({"intent": "CLARIFICATION"}) == "conversational_reply"


def test_route_after_sufficiency_proceeds_when_sufficient():
    state = {"evidence_coverage": {"is_sufficient": True}, "iteration": 0, "max_iterations": 3}
    assert route_after_sufficiency(state) == "reasoning"


def test_route_after_sufficiency_loops_when_insufficient_and_budget_remains():
    state = {"evidence_coverage": {"is_sufficient": False}, "iteration": 1, "max_iterations": 3}
    assert route_after_sufficiency(state) == "adaptive_retrieval"


def test_route_after_sufficiency_never_loops_forever():
    """This is the loop-termination guarantee: once iteration reaches the
    cap, it MUST proceed regardless of coverage, or the graph would spin."""
    state = {"evidence_coverage": {"is_sufficient": False}, "iteration": 3, "max_iterations": 3}
    assert route_after_sufficiency(state) == "reasoning"


def test_route_after_verification_accepts_when_all_supported():
    state = {"verification_results": [{"supported": True}], "verification_cycle": 1, "max_verification_cycles": 2}
    assert route_after_verification(state) == "final_generator"


def test_route_after_verification_loops_when_unsupported_and_budget_remains():
    state = {"verification_results": [{"supported": False}], "verification_cycle": 0, "max_verification_cycles": 2}
    assert route_after_verification(state) == "adaptive_retrieval"


def test_route_after_verification_never_loops_forever():
    state = {"verification_results": [{"supported": False}], "verification_cycle": 2, "max_verification_cycles": 2}
    assert route_after_verification(state) == "final_generator"


def test_route_after_follow_up():
    assert route_after_follow_up({"needs_more_evidence": True}) == "adaptive_retrieval"
    assert route_after_follow_up({"needs_more_evidence": False}) == "reasoning"
