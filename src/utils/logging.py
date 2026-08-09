"""
Agent event logging.

Every entry appended here corresponds to a real state transition that
actually happened in the graph — never a scripted/fake animation. The
Streamlit UI renders these directly as the "Agent Activity" feed.
"""

import time
from typing import List

from src.state import AgentEvent


def make_event(agent: str, icon: str, message: str) -> AgentEvent:
    return AgentEvent(agent=agent, icon=icon, message=message, timestamp=time.time())


def append_event(events: List[AgentEvent], agent: str, icon: str, message: str) -> List[AgentEvent]:
    """Return a new events list with one more real event appended."""
    return [*(events or []), make_event(agent, icon, message)]


def format_event_line(event: AgentEvent) -> str:
    return f"{event.get('icon', '•')} **{event.get('agent', '')}**\n{event.get('message', '')}"
