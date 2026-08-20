"""Phase 2 — planner, researcher and writer under one supervisor.

Phase 1 proved the routing machinery with a single worker. This adds the two
specialists that make it a pipeline, and swaps the thin state for the real
`ReportState`, because now each agent genuinely consumes what the last one
produced: the researcher needs the planner's sub-questions, the writer needs
both the outline and every note.

Research is still sequential here -- the supervisor sends the researcher back
once per sub-question -- which is exactly the inefficiency Phase 5 removes.
Running the two and comparing wall-clock is the point of keeping both.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from ..agents import planner, researcher, supervisor, writer
from ..config import Settings
from ..state import ReportState, initial_state

# The workers the supervisor may name, and the node each maps to. Phase 3 adds
# the critic to this table; nothing else about the wiring changes.
WORKERS = ("planner", "researcher", "writer")


def route_from_supervisor(state: ReportState) -> str:
    """Conditional edge: send control wherever `next` points.

    Anything unrecognised terminates rather than raising. A supervisor that
    invents a worker name is a prompt bug worth seeing in the trace as a clean
    stop, not as a stack trace halfway through a paid run.
    """
    nxt = state.get("next", "FINISH")
    return nxt if nxt in WORKERS else END


def finalize(state: ReportState) -> dict:
    """Promote the accepted draft to the final report."""
    return {"final_report": state.get("draft", "")}


def build(settings: Settings | None = None):
    settings = settings or Settings.load()

    graph = StateGraph(ReportState)
    graph.add_node("supervisor", lambda s: supervisor.route(s, settings))
    graph.add_node("planner", lambda s: planner.plan(s, settings))
    graph.add_node("researcher", lambda s: researcher.research(s, settings))
    graph.add_node("writer", lambda s: writer.write(s, settings))
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {**{w: w for w in WORKERS}, END: "finalize"},
    )
    # Every worker hands control back to the supervisor. Workers never decide
    # who runs next -- that is what keeps routing in one auditable place.
    for worker in WORKERS:
        graph.add_edge(worker, "supervisor")
    graph.add_edge("finalize", END)

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

    topic = " ".join(sys.argv[1:]) or "The state of sodium-ion batteries for grid storage"
    run_id = new_run_id()
    run_log = RunLog(settings.run_log, run_id, graph="phase2_pipeline", topic=topic)

    result = build(settings).invoke(
        initial_state(topic),
        config={
            "callbacks": [RunLogger(run_log)],
            "recursion_limit": settings.recursion_limit,
        },
    )

    outline = result.get("outline")
    if outline is not None:
        print(f"\n--- outline: {outline.title} ---")
        for section in outline.sections:
            print(f"  # {section}")

    report = result.get("final_report") or result.get("draft") or ""
    out_path = settings.out_dir / f"report_{run_id}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")

    print(f"\n--- report ({len(report.split())} words) -> {out_path} ---")
    print(report[:1500])

    summary = totals(RunLog.read(settings.run_log, run_id))
    print("\n--- run summary ---")
    print(f"route:  {' -> '.join(summary['route'])}")
    print(
        f"calls:  {summary['calls']}   cost: ${summary['cost_usd']:.4f}   "
        f"searches: {summary['web_searches']}"
    )


if __name__ == "__main__":
    main()
