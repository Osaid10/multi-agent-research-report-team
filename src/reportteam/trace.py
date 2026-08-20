"""Observability: LangSmith tracing plus a local, append-only run log.

Two recorders, because they answer different questions.

LangSmith is for *debugging* — when the supervisor misroutes or the critic loops
forever, the trace shows the prompts and the order things ran in, which is
nearly impossible to reconstruct from code. The brief is emphatic about turning
it on from day one.

The JSONL log is for *measuring*. Phase 6 needs route accuracy, cost and latency
over a whole eval sweep, and computing those by querying the LangSmith API would
make the eval depend on a network service and an account. Every fact the eval
needs is written locally as it happens, so `run_eval.py` is a pure file reader.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


def configure_langsmith() -> bool:
    """Turn on tracing if the environment supplies a key. Returns whether it is on.

    Deliberately silent-but-honest: a missing key is not an error (the graph
    runs fine without tracing), but the CLI prints the returned flag so you are
    never left wondering whether a run was recorded.
    """
    if not os.environ.get("LANGSMITH_API_KEY"):
        return False

    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", "multi-agent-report-team")
    # langchain-core still reads the older names on some paths; set both so
    # tracing does not depend on which one wins.
    os.environ.setdefault("LANGCHAIN_TRACING_V2", os.environ["LANGSMITH_TRACING"])
    os.environ.setdefault("LANGCHAIN_PROJECT", os.environ["LANGSMITH_PROJECT"])
    return os.environ["LANGSMITH_TRACING"].lower() in {"1", "true", "yes", "on"}


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]


class RunLog:
    """Append-only JSONL writer, one line per event.

    Append-only and flushed per write on purpose: a run that dies to a routing
    loop or a killed process must still leave behind everything that happened
    up to the failure, since that is exactly the run you want to inspect.
    """

    def __init__(self, path: Path, run_id: str, **context: Any) -> None:
        self.path = Path(path)
        self.run_id = run_id
        self.context = context
        self._lock = threading.Lock()  # Phase 5 fans out across threads
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def event(self, kind: str, **fields: Any) -> None:
        record = {
            "run_id": self.run_id,
            "ts": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            **self.context,
            **fields,
        }
        line = json.dumps(record, default=str)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    # -- readers ----------------------------------------------------------

    @staticmethod
    def read(path: Path, run_id: str | None = None) -> list[dict]:
        """Load events, optionally for one run. Tolerates a truncated last line."""
        path = Path(path)
        if not path.is_file():
            return []
        events: list[dict] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # A killed process can leave a half-written final line. Losing
                # it is fine; refusing to read the rest of the log is not.
                continue
            if run_id is None or record.get("run_id") == run_id:
                events.append(record)
        return events


class RunLogger(BaseCallbackHandler):
    """Records every model call and node transition into a `RunLog`.

    Attaches to the graph as a callback so nodes never have to remember to log.
    What it captures is chosen to serve the three Phase 6 metrics: `node` events
    give the route sequence, `llm` events give cost and latency.
    """

    def __init__(self, run_log: RunLog) -> None:
        self.log = run_log
        self._last_step: tuple[str, Any] | None = None

    def on_chain_start(
        self, serialized: dict, inputs: Any, *, metadata: dict | None = None, **kwargs: Any
    ) -> None:
        # LangGraph tags each node's chain run with its node name. Everything
        # else flowing through here is internal plumbing we do not want.
        node = (metadata or {}).get("langgraph_node")
        if not node:
            return

        # One node execution nests several chain runs (the node, the agent
        # inside it, the model call), and every one of them inherits the same
        # `langgraph_node` metadata. Logging each would report a route like
        # "agent -> agent -> agent" and wreck Phase 6's route accuracy, so key
        # on (node, step) and record only the first sighting of each pair.
        step = (metadata or {}).get("langgraph_step")
        if (node, step) == self._last_step:
            return
        self._last_step = (node, step)
        self.log.event("node", node=node, step=step)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        for generation_list in response.generations:
            for generation in generation_list:
                message = getattr(generation, "message", None)
                if message is None:
                    continue
                meta = getattr(message, "response_metadata", {}) or {}
                usage = getattr(message, "usage_metadata", {}) or {}
                self.log.event(
                    "llm",
                    role=meta.get("role"),
                    model=meta.get("model"),
                    cost_usd=meta.get("cost_usd", 0.0),
                    duration_ms=meta.get("duration_ms", 0),
                    num_turns=meta.get("num_turns", 0),
                    web_search_requests=meta.get("web_search_requests", 0),
                    input_tokens=usage.get("input_tokens", 0),
                    output_tokens=usage.get("output_tokens", 0),
                    session_id=meta.get("session_id"),
                )

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        self.log.event("llm_error", error=f"{type(error).__name__}: {error}")


def totals(events: list[dict]) -> dict[str, Any]:
    """Roll a run's events up into the numbers Phase 6 reports."""
    llm_events = [e for e in events if e.get("kind") == "llm"]
    nodes = [e["node"] for e in events if e.get("kind") == "node"]
    return {
        "calls": len(llm_events),
        "cost_usd": round(sum(float(e.get("cost_usd") or 0) for e in llm_events), 6),
        # Summed model time, not wall-clock: with Phase 5's fan-out the two
        # diverge sharply, and that gap is precisely the parallelism payoff.
        "model_ms": sum(int(e.get("duration_ms") or 0) for e in llm_events),
        "input_tokens": sum(int(e.get("input_tokens") or 0) for e in llm_events),
        "output_tokens": sum(int(e.get("output_tokens") or 0) for e in llm_events),
        "web_searches": sum(int(e.get("web_search_requests") or 0) for e in llm_events),
        "route": nodes,
        "errors": sum(1 for e in events if e.get("kind") == "llm_error"),
    }
