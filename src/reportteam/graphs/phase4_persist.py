"""Phase 4 — persistence and human-in-the-loop.

Everything so far has been a single `invoke()` that either finished or died.
This phase makes a report job *durable*: it lives on a `thread_id`, it can be
stopped and picked up later, and it pauses at a gate for a human to approve the
outline before any research is paid for.

The gate is placed after the planner deliberately. That is the cheapest moment
at which a human can still change the outcome -- the outline determines every
sub-question, and therefore every research call and most of the run's cost.
Approving a report after it is written is a rubber stamp; editing the outline
before the researchers start is a real decision.

On the checkpointer: the brief suggests `InMemorySaver` to start and Postgres
for production. SQLite is the honest middle. `InMemorySaver` cannot demonstrate
the thing Phase 4 claims -- its state dies with the process, so "resume" only
ever means "resume inside the same script". With SQLite you can stop a run in
one terminal invocation and finish it in a genuinely separate one, which is the
actual behaviour being taught. Pass `--in-memory` to see the difference.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from ..agents import critic, planner, researcher, supervisor, writer
from ..config import Settings
from ..state import ReportState
from .phase3_reflection import finalize, route_from_critic, route_from_supervisor

WORKERS = ("planner", "researcher", "writer", "critic")


def approve_outline(state: ReportState) -> dict:
    """Pause for a human to approve or edit the outline.

    `interrupt()` throws a special signal that LangGraph catches: the graph
    stops here and the checkpointer persists everything up to this point. The
    call *re-runs from the top of the node* when resumed, and returns whatever
    the resume payload carried -- so this node must stay free of side effects.

    Accepted resume payloads:
      True / {"approved": True}          -- ship the outline as planned
      {"sections": [...]}                -- replace the section list
      {"feedback": "..."}                -- send it back to the planner
    """
    outline = state.get("outline")
    if outline is None:
        return {}

    answer = interrupt(
        {
            "gate": "approve_outline",
            "title": outline.title,
            "angle": outline.angle,
            "sections": list(outline.sections),
            "sub_questions": [q.question for q in outline.sub_questions],
        }
    )

    if answer is True or answer is None:
        return {}
    if not isinstance(answer, dict):
        return {}

    if answer.get("feedback"):
        # Rejecting the outline discards it so the supervisor routes back to
        # the planner, rather than researching a plan a human just refused.
        return {"outline": None, "revision_notes": [answer["feedback"]]}

    edited = outline.model_copy(
        update={
            k: v
            for k, v in answer.items()
            if k in {"title", "angle", "sections"} and v
        }
    )
    return {"outline": edited}


def build(settings: Settings | None = None, *, checkpointer: Any = None):
    settings = settings or Settings.load()

    graph = StateGraph(ReportState)
    graph.add_node("supervisor", lambda s: supervisor.route(s, settings))
    graph.add_node("planner", lambda s: planner.plan(s, settings))
    graph.add_node("approve_outline", approve_outline)
    graph.add_node("researcher", lambda s: researcher.research(s, settings))
    graph.add_node("writer", lambda s: writer.write(s, settings))
    graph.add_node("critic", lambda s: critic.review(s, settings))
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {**{w: w for w in WORKERS}, END: "finalize"},
    )

    # The gate sits between the planner and the supervisor, so the human sees
    # the outline the moment it exists and before anything is spent on it.
    graph.add_edge("planner", "approve_outline")
    graph.add_edge("approve_outline", "supervisor")
    graph.add_edge("researcher", "supervisor")
    graph.add_edge("writer", "supervisor")
    graph.add_conditional_edges(
        "critic",
        lambda s: route_from_critic(s, settings),
        {"writer": "writer", "finalize": "finalize"},
    )
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=checkpointer)


def open_checkpointer(settings: Settings, in_memory: bool = False):
    """Build a checkpointer and return it with the connection to close.

    The sqlite3 connection is created here rather than via
    `SqliteSaver.from_conn_string`, which is a context manager that would close
    the database as soon as the `with` block ends. A CLI that starts a run,
    exits, and is re-invoked later to resume needs to own that lifetime itself.
    """
    if in_memory:
        return InMemorySaver(), None

    settings.checkpoint_db.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False because LangGraph may touch the connection from
    # a worker thread during fan-out.
    conn = sqlite3.connect(str(settings.checkpoint_db), check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return saver, conn
