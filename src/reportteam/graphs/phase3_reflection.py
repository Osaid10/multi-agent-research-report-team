"""Phase 3 — the critic and the reflection loop.

This is where the system stops being a pipeline. Everything up to Phase 2 runs
forward once and could have been a plain LangChain chain; adding a reviewer that
can send work *backwards* introduces a cycle, and a cycle is the thing a graph
gives you that a chain cannot.

There are two conditional edges now, and they answer different questions --
which is exactly the distinction the brief draws:

* `route_from_supervisor` answers "who works next?"
* `route_from_critic` answers "approved, or send it back to revise?"

The second one owns the loop guard. A critic with an unbounded appetite for
revisions is a runaway cost, so the cycle terminates on the revision cap even
when the critic is still unhappy -- and the run says so in its output rather
than quietly shipping a draft that failed review.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from ..agents import critic, planner, researcher, supervisor, writer
from ..config import Settings
from ..state import ReportState, initial_state

WORKERS = ("planner", "researcher", "writer", "critic")


def route_from_supervisor(state: ReportState) -> str:
    """Who works next. Unrecognised names terminate rather than raise."""
    nxt = state.get("next", "FINISH")
    return nxt if nxt in WORKERS else END


def route_from_critic(state: ReportState, settings: Settings) -> str:
    """Approved, or back to the writer?

    The loop guard lives here rather than in the critic's prompt because a cap
    that depends on the model honouring it is not a cap. Hitting it is a real
    outcome worth reporting, not a silent fallback.
    """
    verdict = state.get("verdict")
    if verdict is not None and verdict.approved:
        return "finalize"
    if int(state.get("revisions") or 0) >= settings.max_revisions:
        return "finalize"
    return "writer"


def finalize(state: ReportState) -> dict:
    """Promote the draft, recording whether it actually passed review."""
    verdict = state.get("verdict")
    report = state.get("draft", "")

    if verdict is not None and not verdict.approved:
        # Ship the best draft we have, but never let an unapproved report
        # masquerade as an approved one.
        report += (
            "\n\n---\n\n> **Note:** this report reached the revision limit "
            f"({state.get('revisions')}) without passing review. "
            f"Outstanding issues: {'; '.join(verdict.issues) or 'unspecified'}\n"
        )

    return {"final_report": report}


def build(settings: Settings | None = None):
    settings = settings or Settings.load()

    graph = StateGraph(ReportState)
    graph.add_node("supervisor", lambda s: supervisor.route(s, settings))
    graph.add_node("planner", lambda s: planner.plan(s, settings))
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

    # Workers report back to the supervisor -- except the critic, which owns
    # its own branch. That is the reflection cycle: critic -> writer -> ... ->
    # critic, until it passes or the cap stops it.
    for worker in ("planner", "researcher", "writer"):
        graph.add_edge(worker, "supervisor")

    graph.add_conditional_edges(
        "critic",
        lambda s: route_from_critic(s, settings),
        {"writer": "writer", "finalize": "finalize"},
    )
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

    args = [a for a in sys.argv[1:] if a != "--sabotage-draft"]
    sabotage = "--sabotage-draft" in sys.argv[1:]
    topic = " ".join(args) or "The state of sodium-ion batteries for grid storage"

    run_id = new_run_id()
    run_log = RunLog(settings.run_log, run_id, graph="phase3_reflection", topic=topic)

    state = initial_state(topic)
    state["sabotage"] = sabotage
    if sabotage:
        print("sabotage: ON -- the first draft will be deliberately weak\n")

    result = build(settings).invoke(
        state,
        config={
            "callbacks": [RunLogger(run_log)],
            "recursion_limit": settings.recursion_limit,
        },
    )

    verdict = result.get("verdict")
    print("\n--- review ---")
    if verdict is not None:
        s = verdict.scores
        print(f"approved:     {verdict.approved}")
        print(
            f"scores:       coverage={s.coverage} groundedness={s.groundedness} "
            f"structure={s.structure} support={s.support}"
        )
        print(f"revisions:    {result.get('revisions')} of {settings.max_revisions}")
        for issue in verdict.issues[:5]:
            print(f"  issue: {issue}")

    report = result.get("final_report") or ""
    out_path = settings.out_dir / f"report_{run_id}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"\n--- report ({len(report.split())} words) -> {out_path} ---")

    summary = totals(RunLog.read(settings.run_log, run_id))
    print("\n--- run summary ---")
    print(f"route:  {' -> '.join(summary['route'])}")
    print(
        f"calls:  {summary['calls']}   cost: ${summary['cost_usd']:.4f}   "
        f"searches: {summary['web_searches']}"
    )


if __name__ == "__main__":
    main()
