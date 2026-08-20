"""The shared state every agent reads and writes.

Phase 1 got by on a bare topic and a note list. From Phase 2 the team needs
real shared memory, because each specialist consumes what the previous one
produced: the writer needs the planner's outline *and* the researchers' notes,
and the critic needs all three plus the draft in order to judge whether the
draft is grounded in them.

The reducers are the interesting part. Most fields overwrite (last writer
wins), but `notes` accumulates — which is what lets Phase 5 fan researchers out
concurrently and have their results merge with no merge code at all.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from .models import Outline, ResearchNote, Verdict


class ReportState(TypedDict, total=False):
    """The team's working memory for one report job."""

    # -- input ------------------------------------------------------------
    topic: str

    # -- planner ----------------------------------------------------------
    outline: Outline | None

    # -- researchers ------------------------------------------------------
    # Appended, never replaced. Two researchers finishing at the same time on
    # different sub-questions must both survive; with the default
    # last-write-wins channel one would silently overwrite the other.
    notes: Annotated[list[ResearchNote], operator.add]

    # -- writer -----------------------------------------------------------
    draft: str
    # The critic's instructions for the next revision. Cleared when a draft is
    # approved so a later revision never acts on stale complaints.
    revision_notes: list[str]

    # -- critic -----------------------------------------------------------
    verdict: Verdict | None
    # How many times the critic has sent the draft back. The brief asks for an
    # explicit cap here in addition to LangGraph's recursion_limit: the two
    # guard different failures, a stubborn critic versus a routing loop.
    revisions: int

    # -- supervisor -------------------------------------------------------
    next: str
    reason: str

    # -- output -----------------------------------------------------------
    final_report: str


def initial_state(topic: str) -> ReportState:
    """A fresh job. Every channel is seeded so nodes never guard against None."""
    return {
        "topic": topic,
        "outline": None,
        "notes": [],
        "draft": "",
        "revision_notes": [],
        "verdict": None,
        "revisions": 0,
        "next": "",
        "reason": "",
        "final_report": "",
    }


def pending_sub_questions(state: ReportState) -> list[str]:
    """Sub-questions from the outline that no researcher has answered yet.

    Comparing on the question text rather than an index keeps this correct when
    notes come back out of order, which under Phase 5's concurrent fan-out they
    routinely do.
    """
    outline = state.get("outline")
    if outline is None:
        return []
    answered = {note.sub_question.strip() for note in state.get("notes") or []}
    return [
        sub.question
        for sub in outline.sub_questions
        if sub.question.strip() not in answered
    ]
