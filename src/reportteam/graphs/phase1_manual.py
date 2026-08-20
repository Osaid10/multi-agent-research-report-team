"""Phase 1 — a supervisor and one worker, built by hand.

The brief is insistent that this comes before `create_supervisor`, and running
Phase 0 next to this shows why. The prebuilt helper routes by giving the
supervisor a `transfer_to_<worker>` tool per worker and letting it make a tool
call; you can see the machinery in the Phase 0 route
(`supervisor -> agent -> tools -> counter -> model -> supervisor`) and in its
price — three model calls and a growing message history to count eight words.

What that helper is doing underneath is the four pieces assembled here:

1. a state channel (`next`) saying who should run,
2. a supervisor node that writes to it,
3. worker nodes that do one job and report back,
4. conditional edges that read `next` and jump, with FINISH wired to END.

Routing here is a *structured output*, not a tool call: the supervisor is
launched with no tools at all, so delegation cannot be confused with doing the
work. That is the brief's "keep the supervisor dumb on tools" guardrail
expressed as architecture.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from ..claude_cli import for_role
from ..config import Settings
from ..models import ResearchNote, RouteDecision

# Phase 1 deliberately keeps a thin state. Phase 2 replaces it with the real
# `ReportState`; seeing the bare version first is what makes the richer one
# feel necessary rather than arbitrary.
class Phase1State(TypedDict):
    topic: str
    # `operator.add` appends instead of overwriting. Introduced here with one
    # researcher so that Phase 5's concurrent fan-in needs no state change.
    notes: Annotated[list[ResearchNote], operator.add]
    next: str
    reason: str


SUPERVISOR_PROMPT = """\
You are the supervisor of a research team. You do not do research yourself and \
you have no tools; your only job is to decide who works next.

Your team:
- researcher: searches the web and reports findings with sources.

Rules:
- If there are no research notes yet, route to the researcher.
- Once the researcher has reported findings on the topic, the work is done: \
answer FINISH.
- Never route to the researcher twice for the same topic. Repeating a worker \
that has already reported is the failure this system is designed to avoid.
"""

RESEARCHER_PROMPT = """\
You are a researcher. Search the web for the question you are given and report \
what you actually found.

- Use the WebSearch tool. Do not answer from memory.
- Every finding must come from a source you list. Never invent a URL.
- If you cannot establish something, put it in `gaps` rather than guessing.
"""


def supervisor_node(state: Phase1State, settings: Settings) -> dict:
    """Decide who works next and write it to `next`."""
    notes = state.get("notes") or []
    status = (
        "No research has been done yet."
        if not notes
        else f"The researcher has reported on {len(notes)} question(s): "
        + "; ".join(n.sub_question for n in notes)
    )

    decision = (
        for_role("supervisor", settings)
        .with_structured_output(RouteDecision)
        .invoke(
            [
                SystemMessage(SUPERVISOR_PROMPT),
                HumanMessage(f"Topic: {state['topic']}\n\nStatus: {status}"),
            ]
        )
    )
    return {"next": decision.next, "reason": decision.reason}


def researcher_node(state: Phase1State, settings: Settings) -> dict:
    """Research the topic and append a note."""
    note = (
        for_role("researcher", settings)
        .with_structured_output(ResearchNote)
        .invoke(
            [
                SystemMessage(RESEARCHER_PROMPT),
                HumanMessage(f"Research this question: {state['topic']}"),
            ]
        )
    )
    return {"notes": [note]}


def route_from_supervisor(state: Phase1State) -> str:
    """The conditional edge: read `next`, map FINISH to the terminal node.

    Kept as a plain function of state rather than folded into the supervisor
    node, because that separation is the whole lesson: the node *decides*, the
    edge *moves*. Merging them is what makes a graph hard to reason about.
    """
    nxt = state.get("next", "FINISH")
    return END if nxt == "FINISH" else "researcher"


def build(settings: Settings | None = None):
    settings = settings or Settings.load()

    graph = StateGraph(Phase1State)
    graph.add_node("supervisor", lambda s: supervisor_node(s, settings))
    graph.add_node("researcher", lambda s: researcher_node(s, settings))

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        # The explicit mapping is what makes the graph diagram readable and
        # lets LangGraph validate that every branch has somewhere to land.
        {"researcher": "researcher", END: END},
    )
    # The worker reports back to the supervisor rather than deciding for itself
    # what happens next. This is the handoff: control always returns to the
    # orchestrator, which is what keeps routing in one auditable place.
    graph.add_edge("researcher", "supervisor")

    return graph.compile()


def main() -> None:
    import sys

    from ..trace import (
        RunLog,
        RunLogger,
        configure_langsmith,
        enable_utf8_console,
        new_run_id,
        totals,
    )

    enable_utf8_console()
    settings = Settings.load()
    traced = configure_langsmith()
    print(f"LangSmith tracing: {'on' if traced else 'off (no LANGSMITH_API_KEY)'}")

    topic = " ".join(sys.argv[1:]) or "What is the current state of sodium-ion batteries?"
    run_id = new_run_id()
    run_log = RunLog(settings.run_log, run_id, graph="phase1_manual", topic=topic)

    result = build(settings).invoke(
        {"topic": topic, "notes": [], "next": "", "reason": ""},
        config={
            "callbacks": [RunLogger(run_log)],
            "recursion_limit": settings.recursion_limit,
        },
    )

    print(f"\n--- findings for: {topic} ---")
    for note in result["notes"]:
        for finding in note.findings:
            print(f"  * {finding}")
        for source in note.sources:
            print(f"    [{source.title}]({source.url})")
        if note.gaps:
            print(f"    gaps: {note.gaps}")

    print(f"\nsupervisor's last call: {result['next']} -- {result['reason']}")

    summary = totals(RunLog.read(settings.run_log, run_id))
    print("\n--- run summary ---")
    print(f"route:  {' -> '.join(summary['route'])}")
    print(f"calls:  {summary['calls']}   cost: ${summary['cost_usd']:.4f}")


if __name__ == "__main__":
    main()
