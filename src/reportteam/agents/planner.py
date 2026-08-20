"""The planner: turns a topic into an outline and a research agenda."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from ..claude_cli import for_role
from ..config import Settings
from ..models import Outline
from ..state import ReportState

PLANNER_PROMPT = """\
You are the planner for a research team. You turn a topic into a report \
structure and a research agenda. You do not research and you do not write the \
report.

Two things matter:

1. The sections must form an argument, not a list of nouns. "Background", \
"Current state", "Challenges", "Outlook" is the generic shape every weak report \
takes; prefer sections that say something specific about this topic.

2. The sub-questions are dispatched to independent researchers who work \
concurrently and see nothing but the single question you wrote. So each one \
must be self-contained -- no "the topic", no "as above", no dependency on \
another sub-question's answer -- and each must be narrow enough to answer from \
a handful of web sources.

Cover the whole outline with the sub-questions: anything you do not ask about, \
the writer will have no sourced material for, and the critic will reject the \
draft for it.
"""


def plan(state: ReportState, settings: Settings) -> dict:
    """Produce the outline. Writes `outline`."""
    outline = (
        for_role("planner", settings)
        .with_structured_output(Outline)
        .invoke(
            [
                SystemMessage(PLANNER_PROMPT),
                HumanMessage(f"Topic: {state['topic']}"),
            ]
        )
    )
    return {"outline": outline}
