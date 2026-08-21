"""Did the supervisor delegate sensibly, and stop at the right time?

The brief asks for route accuracy as a first-class metric, and it is the one
that most needs a definition rather than a vibe. "Sensible" here is expressed
as a set of preconditions -- a worker may only run once its inputs exist -- plus
two liveness rules: don't repeat finished work, and do stop.

Deliberately computed from the local JSONL run log rather than from LangSmith,
so the eval is a pure file reader and does not depend on a network service.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# The nodes that represent real delegated work, as opposed to the orchestrator
# and the terminal bookkeeping node.
WORKERS = {"planner", "researcher", "research", "writer", "critic"}


@dataclass
class RouteReport:
    route: list[str]
    violations: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations

    @property
    def steps(self) -> int:
        return len(self.route)

    @property
    def worker_calls(self) -> int:
        return sum(1 for node in self.route if node in WORKERS)


def check(route: list[str], *, max_revisions: int = 2, finished: bool = True) -> RouteReport:
    """Score one run's node sequence against the routing rules."""
    report = RouteReport(route=list(route))
    add = report.violations.append

    if not route:
        add("empty route: the graph never ran")
        return report

    seen: set[str] = set()
    research_done = False
    drafts = 0
    reviews = 0

    for position, node in enumerate(route):
        if node == "planner" and "planner" in seen:
            # Re-planning is legitimate only when a human rejected the outline,
            # which this eval never does.
            add(f"step {position}: planner ran twice -- the outline already existed")

        if node in {"researcher", "research"} and "planner" not in seen:
            add(f"step {position}: researched before an outline existed")

        if node == "writer":
            drafts += 1
            if "planner" not in seen:
                add(f"step {position}: wrote a draft with no outline")
            elif not research_done:
                add(f"step {position}: wrote a draft before research finished")

        if node == "critic":
            reviews += 1
            if drafts == 0:
                add(f"step {position}: reviewed before any draft existed")

        if node in {"researcher", "research"}:
            research_done = True

        seen.add(node)

    # Liveness. A revision cycle is writer -> critic, so the number of drafts
    # beyond the first is the number of revisions taken.
    revisions = max(0, drafts - 1)
    if revisions > max_revisions:
        add(f"took {revisions} revisions, over the cap of {max_revisions}")

    if reviews and drafts and reviews > drafts:
        add(f"reviewed {reviews} times for {drafts} drafts -- re-reviewed unchanged work")

    if not finished:
        add("did not terminate: hit the recursion limit")
    elif route[-1] != "finalize":
        add(f"ended on {route[-1]!r} rather than finalize")

    if "planner" not in seen:
        add("never planned")
    if "writer" not in seen:
        add("never produced a draft")

    return report


def summarise(reports: list[RouteReport]) -> dict:
    """Aggregate across an eval sweep."""
    if not reports:
        return {"runs": 0, "accuracy": 0.0, "clean": 0, "violations": {}}

    tally: dict[str, int] = {}
    for report in reports:
        for violation in report.violations:
            # Strip the "step N: " prefix so like failures group together.
            key = violation.split(": ", 1)[-1] if violation.startswith("step ") else violation
            tally[key] = tally.get(key, 0) + 1

    clean = sum(1 for r in reports if r.ok)
    return {
        "runs": len(reports),
        "clean": clean,
        "accuracy": round(clean / len(reports), 3),
        "mean_steps": round(sum(r.steps for r in reports) / len(reports), 1),
        "mean_worker_calls": round(sum(r.worker_calls for r in reports) / len(reports), 1),
        "violations": dict(sorted(tally.items(), key=lambda kv: -kv[1])),
    }
