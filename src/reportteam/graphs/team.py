"""Phase 5 — the finished team, with researchers running in parallel.

Phase 2 researched one sub-question per supervisor turn: five sub-questions
meant five researcher calls, five supervisor calls, and five round-trips one
after another. The questions were independent the whole time -- the planner is
told to make them so -- which means that was latency spent for nothing.

`Send` fixes it. `dispatch_research` returns a *list* of `Send` objects, one per
outstanding sub-question, and LangGraph runs them as concurrent branches of the
same superstep. Their results merge back through the `operator.add` reducer on
`notes`, which has been in the state since Phase 1 precisely so that this change
would need no state rewrite. That is the whole diff: one node, one edge, no new
plumbing.

Two details that matter:

* The supervisor no longer routes each research call, so its per-question turns
  disappear from the route as well -- the saving is more than just wall-clock.
* Each researcher is a separate CLI subprocess, so concurrency is bounded by the
  account's rate limits rather than local CPU. `max_concurrent_researchers`
  keeps the fan-out from stampeding.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from ..agents import critic, planner, researcher, supervisor, writer
from ..config import Settings
from ..state import ReportState, pending_sub_questions
from .phase3_reflection import finalize, route_from_critic
from .phase4_persist import approve_outline

# "research" replaces "researcher": the supervisor now asks for the whole
# research phase at once rather than one question at a time.
WORKERS = ("planner", "research", "writer", "critic")


def route_from_supervisor(state: ReportState) -> str:
    nxt = state.get("next", "FINISH")
    # The supervisor's vocabulary still says "researcher" (its prompt describes
    # the role, not the graph topology), so accept both names for the node that
    # now fans out.
    if nxt == "researcher":
        return "research"
    return nxt if nxt in WORKERS else END


def dispatch_research(state: ReportState) -> list[Send] | str:
    """Fan every outstanding sub-question out to its own researcher.

    Returning a list of `Send`s from a conditional edge is the map half of
    map-reduce: each carries its own private input, so the workers never see or
    contend for each other's state.
    """
    pending = pending_sub_questions(state)
    if not pending:
        return "supervisor"

    return [
        Send("researcher", {"topic": state["topic"], "sub_question": question})
        for question in pending
    ]


def research_gate(state: ReportState) -> dict:
    """A no-op node that exists purely to hang the fan-out edge on.

    LangGraph dispatches `Send`s from a conditional edge, and an edge needs a
    source node. Keeping it empty makes the graph diagram read correctly:
    control arrives at "research", fans out, and reconvenes.
    """
    return {}


def build(settings: Settings | None = None, *, checkpointer: Any = None, gated: bool = False):
    """Compile the full team.

    `gated=True` adds the Phase 4 human approval gate after the planner.
    """
    settings = settings or Settings.load()

    graph = StateGraph(ReportState)
    graph.add_node("supervisor", lambda s: supervisor.route(s, settings))
    graph.add_node("planner", lambda s: planner.plan(s, settings))
    graph.add_node("research", research_gate)
    graph.add_node("researcher", lambda s: researcher.research(s, settings))
    graph.add_node("writer", lambda s: writer.write(s, settings))
    graph.add_node("critic", lambda s: critic.review(s, settings))
    graph.add_node("finalize", finalize)
    if gated:
        graph.add_node("approve_outline", approve_outline)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {**{w: w for w in WORKERS}, END: "finalize"},
    )

    if gated:
        graph.add_edge("planner", "approve_outline")
        graph.add_edge("approve_outline", "supervisor")
    else:
        graph.add_edge("planner", "supervisor")

    # The map step. The edge lists "researcher" as a possible destination so
    # LangGraph can validate the graph, even though the Send objects are what
    # actually carry control there.
    graph.add_conditional_edges(
        "research", dispatch_research, ["researcher", "supervisor"]
    )
    # The reduce step: every branch lands back on the supervisor, which runs
    # once after all of them have merged into `notes`.
    graph.add_edge("researcher", "supervisor")

    graph.add_edge("writer", "supervisor")
    graph.add_conditional_edges(
        "critic",
        lambda s: route_from_critic(s, settings),
        {"writer": "writer", "finalize": "finalize"},
    )
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=checkpointer)
