"""One entry point for the whole team.

    reportteam run "topic"                  the full parallel team
    reportteam run "topic" --gated          pause for outline approval
    reportteam approve --thread-id X        resume a paused run
    reportteam run "topic" --sabotage       force a revision cycle
    reportteam baseline "topic"             the single-agent comparison
    reportteam graph --which team           print the graph structure
"""

from __future__ import annotations

import json
import time
from typing import Optional

import typer

from .config import Settings
from .graphs import phase1_manual, phase2_pipeline, phase3_reflection, phase4_persist, team
from .state import initial_state
from .trace import (
    RunLog,
    RunLogger,
    configure_langsmith,
    enable_utf8_console,
    new_run_id,
    totals,
)

app = typer.Typer(add_completion=False, help=__doc__)

# Every phase stays runnable. The progression is part of the deliverable, and
# the eval needs `phase3` (sequential) and `team` (parallel) side by side to
# evidence the parallelism claim.
GRAPHS = {
    "phase1": phase1_manual,
    "phase2": phase2_pipeline,
    "phase3": phase3_reflection,
    "phase4": phase4_persist,
    "team": team,
}


def _report_summary(result: dict, settings: Settings, run_id: str, wall_s: float) -> None:
    verdict = result.get("verdict")
    if verdict is not None:
        s = verdict.scores
        typer.echo("\n--- review ---")
        typer.echo(f"approved:   {verdict.approved}")
        typer.echo(
            f"scores:     coverage={s.coverage} groundedness={s.groundedness} "
            f"structure={s.structure} support={s.support}"
        )
        typer.echo(f"revisions:  {result.get('revisions', 0)} of {settings.max_revisions}")
        for issue in verdict.issues[:5]:
            typer.echo(f"  issue: {issue}")

    report = result.get("final_report") or result.get("draft") or ""
    out_path = settings.out_dir / f"report_{run_id}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")

    summary = totals(RunLog.read(settings.run_log, run_id))
    typer.echo(f"\n--- report ({len(report.split())} words) -> {out_path} ---")
    typer.echo("\n--- run summary ---")
    typer.echo(f"route:      {' -> '.join(summary['route'])}")
    typer.echo(
        f"calls:      {summary['calls']}   cost: ${summary['cost_usd']:.4f}   "
        f"searches: {summary['web_searches']}"
    )
    # Wall-clock and summed model time diverge exactly as much as the fan-out
    # parallelises, so printing both makes the Phase 5 speedup self-evident.
    typer.echo(
        f"wall clock: {wall_s:.1f}s   model time: {summary['model_ms'] / 1000:.1f}s"
    )


@app.command()
def run(
    topic: list[str] = typer.Argument(..., help="The report topic."),
    graph: str = typer.Option("team", help=f"Which graph: {', '.join(GRAPHS)}"),
    gated: bool = typer.Option(False, help="Pause after the planner for approval."),
    sabotage: bool = typer.Option(False, help="Force a weak first draft to trigger a revision."),
    thread_id: Optional[str] = typer.Option(None, help="Resume/identify this job."),
    in_memory: bool = typer.Option(
        False,
        "--in-memory",
        help=(
            "Use InMemorySaver instead of SQLite for --gated. State dies with "
            "the process, so `approve` in a new process cannot find the thread "
            "-- which is the point: it shows why SQLite is needed."
        ),
    ),
) -> None:
    """Research a topic and write a cited report."""
    enable_utf8_console()
    settings = Settings.load()
    traced = configure_langsmith()
    typer.echo(f"LangSmith tracing: {'on' if traced else 'off (no LANGSMITH_API_KEY)'}")

    if graph not in GRAPHS:
        raise typer.BadParameter(f"unknown graph {graph!r}; choose from {list(GRAPHS)}")

    subject = " ".join(topic)
    run_id = new_run_id()
    tid = thread_id or run_id
    run_log = RunLog(settings.run_log, run_id, graph=graph, topic=subject, thread_id=tid)

    module = GRAPHS[graph]
    checkpointer = conn = None
    if gated:
        if graph not in {"phase4", "team"}:
            raise typer.BadParameter("--gated needs --graph phase4 or --graph team")
        checkpointer, conn = phase4_persist.open_checkpointer(settings, in_memory)
        compiled = (
            module.build(settings, checkpointer=checkpointer, gated=True)
            if graph == "team"
            else module.build(settings, checkpointer=checkpointer)
        )
    else:
        compiled = module.build(settings)

    # Phase 1 predates the shared ReportState and declares only four channels.
    # Handing it the full state would push updates at channels its schema does
    # not define.
    if graph == "phase1":
        state = {"topic": subject, "notes": [], "next": "", "reason": ""}
    else:
        state = initial_state(subject)
        state["sabotage"] = sabotage
    if sabotage:
        typer.echo("sabotage: ON -- the first draft will be deliberately weak\n")

    config = {
        "callbacks": [RunLogger(run_log)],
        "recursion_limit": settings.recursion_limit,
        "configurable": {"thread_id": tid},
    }

    started = time.monotonic()
    try:
        result = compiled.invoke(state, config=config)

        if "__interrupt__" in result:
            payload = result["__interrupt__"][0].value
            typer.echo("\n=== PAUSED: the outline needs a human ===")
            typer.echo(f"title:    {payload.get('title')}")
            typer.echo(f"angle:    {payload.get('angle')}")
            for section in payload.get("sections", []):
                typer.echo(f"  # {section}")
            for question in payload.get("sub_questions", []):
                typer.echo(f"  ? {question}")
            typer.echo(f"\nthread_id: {tid}")
            typer.echo(f"resume with:  reportteam approve --thread-id {tid}")
            typer.echo(f"or reject:    reportteam approve --thread-id {tid} --feedback '...'")
            return

        _report_summary(result, settings, run_id, time.monotonic() - started)
    finally:
        if conn is not None:
            conn.close()


@app.command()
def approve(
    thread_id: str = typer.Option(..., help="The paused job's thread id."),
    feedback: Optional[str] = typer.Option(None, help="Reject the outline and re-plan."),
    sections: Optional[str] = typer.Option(None, help="Replace sections (JSON list)."),
    graph: str = typer.Option("team", help="The graph the job was started on."),
) -> None:
    """Resume a paused job. Runs as a separate process from `run`, which is the point."""
    enable_utf8_console()
    settings = Settings.load()
    configure_langsmith()

    from langgraph.types import Command

    run_id = new_run_id()
    run_log = RunLog(
        settings.run_log, run_id, graph=graph, thread_id=thread_id, resumed=True
    )

    checkpointer, conn = phase4_persist.open_checkpointer(settings)
    module = GRAPHS[graph]
    compiled = (
        module.build(settings, checkpointer=checkpointer, gated=True)
        if graph == "team"
        else module.build(settings, checkpointer=checkpointer)
    )

    if feedback:
        payload: object = {"feedback": feedback}
        typer.echo(f"rejecting the outline: {feedback}")
    elif sections:
        payload = {"sections": json.loads(sections)}
        typer.echo(f"replacing sections with {payload['sections']}")
    else:
        payload = True
        typer.echo("approving the outline as planned")

    config = {
        "callbacks": [RunLogger(run_log)],
        "recursion_limit": settings.recursion_limit,
        "configurable": {"thread_id": thread_id},
    }

    started = time.monotonic()
    try:
        result = compiled.invoke(Command(resume=payload), config=config)
        _report_summary(result, settings, run_id, time.monotonic() - started)
    finally:
        conn.close()


@app.command()
def baseline(topic: list[str] = typer.Argument(..., help="The report topic.")) -> None:
    """Write the same report with one agent in one pass, for comparison."""
    enable_utf8_console()
    settings = Settings.load()
    from . import baseline as baseline_module

    subject = " ".join(topic)
    run_id = new_run_id()

    started = time.monotonic()
    result = baseline_module.run(subject, settings)
    wall = time.monotonic() - started

    out_path = settings.out_dir / f"baseline_{run_id}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(result["report"], encoding="utf-8")

    typer.echo(f"--- baseline ({len(result['report'].split())} words) -> {out_path} ---")
    typer.echo(
        f"cost: ${result['cost_usd']:.4f}   wall clock: {wall:.1f}s   "
        f"searches: {result['web_searches']}   turns: {result['num_turns']}"
    )


@app.command("trace-url")
def trace_url(
    limit: int = typer.Option(5, help="How many recent root runs to list."),
    project: Optional[str] = typer.Option(None, help="LangSmith project name."),
) -> None:
    """Print shareable LangSmith links for recent runs.

    The brief asks for a trace link as a deliverable, and hunting for one in the
    LangSmith UI after the fact is tedious. Root runs only -- a single team run
    produces dozens of nested spans, and the one worth sharing is the top.
    """
    enable_utf8_console()
    Settings.load()
    if not configure_langsmith():
        typer.echo("tracing is off -- set LANGSMITH_API_KEY in .env first")
        raise typer.Exit(1)

    import os

    from langsmith import Client

    client = Client()
    name = project or os.environ.get("LANGSMITH_PROJECT", "multi-agent-report-team")

    # `is_root` filters server-side, which matters: a single team run produces
    # dozens of nested spans, and the API refuses a limit above 100, so
    # over-fetching and filtering here would miss roots on a busy project.
    try:
        candidates = client.list_runs(project_name=name, is_root=True, limit=limit)
    except TypeError:  # older SDK without the filter
        candidates = client.list_runs(project_name=name, limit=100)

    shown = 0
    for run in candidates:
        if run.parent_run_id is not None:
            continue
        started = run.start_time.strftime("%Y-%m-%d %H:%M") if run.start_time else "?"
        typer.echo(f"{started}  {str(run.name)[:28]:28}  {run.url}")
        shown += 1
        if shown >= limit:
            break

    if not shown:
        typer.echo(f"no root runs found in project {name!r} yet")


@app.command()
def graph(which: str = typer.Option("team", help=f"One of: {', '.join(GRAPHS)}")) -> None:
    """Print a graph's structure as ASCII -- useful for the README."""
    enable_utf8_console()
    settings = Settings.load()
    module = GRAPHS[which]
    compiled = module.build(settings, gated=True) if which == "team" else module.build(settings)
    typer.echo(compiled.get_graph().draw_ascii())


if __name__ == "__main__":
    app()
