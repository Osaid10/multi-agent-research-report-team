"""Phase 0 — the throwaway supervisor.

The brief's first milestone: stand up one supervisor and one trivial worker
using the *prebuilt* helper, watch it route and finish, and confirm the run
reaches LangSmith. Nothing here survives into the real system — Phase 1 rebuilds
the supervisor by hand, which is the point of the exercise. Keeping this module
around is what lets the README compare the two honestly.

Two things worth noticing when you run it:

* `create_supervisor` delegates through *handoff tools* (`transfer_to_...`), so
  it needs a tool-calling model. This backend has no tool-calling API, which is
  why `ChatClaudeCLI.bind_tools` emulates one through the output schema.
* `langgraph-supervisor` is soft-deprecated — LangChain now recommends building
  the supervisor pattern directly. That is exactly what Phase 1 does, so the
  brief's "build it by hand first" instruction and the library's own advice
  happen to agree.
"""

from __future__ import annotations

from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph_supervisor import create_supervisor

from ..claude_cli import for_role
from ..config import Settings


@tool
def word_count(text: str) -> str:
    """Count the words in a piece of text."""
    return f"{len(text.split())} words"


def build(settings: Settings | None = None):
    """Compile the trivial two-node team."""
    settings = settings or Settings.load()

    counter = create_agent(
        model=for_role("planner", settings),
        tools=[word_count],
        system_prompt=(
            "You count words. Call the word_count tool exactly once, then report "
            "the number it returns and stop."
        ),
        name="counter",
    )

    supervisor = create_supervisor(
        [counter],
        model=for_role("supervisor", settings),
        prompt=(
            "You are a supervisor managing one worker: 'counter', which counts "
            "words in text.\n"
            "Delegate the user's request to counter. When counter has reported "
            "back, reply with its answer and finish. Do not do the work "
            "yourself and do not delegate twice."
        ),
    )
    return supervisor.compile()


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

    run_id = new_run_id()
    run_log = RunLog(settings.run_log, run_id, graph="phase0_hello")
    text = " ".join(sys.argv[1:]) or "the quick brown fox jumps over the lazy dog"

    graph = build(settings)
    result = graph.invoke(
        {"messages": [{"role": "user", "content": f"How many words: {text!r}"}]},
        config={
            "callbacks": [RunLogger(run_log)],
            "recursion_limit": settings.recursion_limit,
        },
    )

    print("\n--- final message ---")
    print(result["messages"][-1].content.strip()[:500])

    summary = totals(RunLog.read(settings.run_log, run_id))
    print("\n--- run summary ---")
    print(f"route:  {' -> '.join(summary['route'])}")
    print(f"calls:  {summary['calls']}   cost: ${summary['cost_usd']:.4f}")
    print(f"log:    {settings.run_log}  (run_id={run_id})")


if __name__ == "__main__":
    main()
