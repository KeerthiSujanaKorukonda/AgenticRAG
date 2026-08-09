"""
Real tests exercising the compiled LangGraph workflow end-to-end, including
the conversational paths (no models needed) and the dangling-follow-up
case. The full research-pipeline path with real models is exercised
separately in test_evaluation.py using stub models (no network required).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.graph.workflow import WorkflowDependencies, build_graph


def test_greeting_requires_no_models():
    """A WorkflowDependencies() with every model set to None must still
    correctly answer a greeting — this is the whole point of the intent
    router existing before any retrieval/generation step."""
    deps = WorkflowDependencies()
    graph = build_graph(deps)
    result = graph.invoke({"user_input": "Hi", "agent_events": []})
    assert result["intent"] == "GREETING"
    assert "ResearchGapPilot" in result["final_answer"]


def test_capabilities_requires_no_models():
    deps = WorkflowDependencies()
    graph = build_graph(deps)
    result = graph.invoke({"user_input": "What can you do?", "agent_events": []})
    assert result["intent"] == "CAPABILITIES"
    assert "research" in result["final_answer"].lower()


def test_unsupported_request():
    deps = WorkflowDependencies()
    graph = build_graph(deps)
    result = graph.invoke({"user_input": "Write me a poem about love.", "agent_events": []})
    assert result["intent"] == "UNSUPPORTED_REQUEST"


def test_dangling_follow_up_reference_returns_helpful_message():
    """Spec section 36: must not invent an answer about 'gap 2' when no
    research has happened yet in this session."""
    deps = WorkflowDependencies()
    graph = build_graph(deps)
    result = graph.invoke({"user_input": "Explain gap 2.", "agent_events": []})
    assert result["intent"] == "CLARIFICATION"
    assert "start a research question" in result["final_answer"].lower()


def test_empty_input_does_not_crash():
    deps = WorkflowDependencies()
    graph = build_graph(deps)
    result = graph.invoke({"user_input": "", "agent_events": []})
    assert result["intent"] == "CLARIFICATION"
