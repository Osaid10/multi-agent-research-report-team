"""Phase 6 — the capstone: team versus single-agent baseline.

Runs both systems over the topic set, scores route accuracy, report quality and
cost/latency, and writes a results table plus the raw rows.

Every completed topic is cached to disk. A sweep over 20 topics runs both
systems and calls the judge twice per topic; losing all of that to a crash on
topic 18, or re-paying for it to change one line of the output table, would make
the eval something you run once and never touch. With the cache it is a file
reader on the second run.

    python -m eval.run_eval                 # the whole set
    python -m eval.run_eval --limit 2       # smoke test
    python -m eval.run_eval --bucket ambiguous
    python -m eval.run_eval --refresh       # ignore the cache
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml

from eval.judge import judge
from eval.route_accuracy import check, summarise
from reportteam import baseline as baseline_module
from reportteam.claude_cli import ClaudeSessionLimitError
from reportteam.config import Settings
from reportteam.graphs import team
from reportteam.state import initial_state
from reportteam.trace import (
    RunLog,
    RunLogger,
    enable_utf8_console,
    new_run_id,
    recent_spend,
    totals,
)

class _SpendCapReached(RuntimeError):
    """The trailing-window spend cap was reached; stop before the account is."""


ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "out" / "eval_cache"


def _cache_path(topic: str, system: str) -> Path:
    # Hash-free, human-readable filenames: when a row looks wrong you want to
    # open the artefact without grepping for a checksum.
    slug = "".join(c if c.isalnum() else "_" for c in topic.lower())[:70].strip("_")
    return CACHE_DIR / f"{system}__{slug}.json"


def run_team(topic: str, settings: Settings) -> dict:
    """One full team run, with its metrics."""
    run_id = new_run_id()
    run_log = RunLog(settings.run_log, run_id, graph="team", topic=topic, eval=True)

    started = time.monotonic()
    finished = True
    try:
        result = team.build(settings).invoke(
            initial_state(topic),
            config={
                "callbacks": [RunLogger(run_log)],
                "recursion_limit": settings.recursion_limit,
                "configurable": {"thread_id": run_id},
            },
        )
    except ClaudeSessionLimitError:
        # Not a data point -- nothing will succeed until the allowance resets.
        raise
    except Exception as exc:  # a run that dies IS a data point, not a crash
        finished = False
        result = {"final_report": "", "error": f"{type(exc).__name__}: {exc}"}

    wall = time.monotonic() - started
    summary = totals(RunLog.read(settings.run_log, run_id))
    verdict = result.get("verdict")

    return {
        "system": "team",
        "topic": topic,
        "report": result.get("final_report") or result.get("draft") or "",
        "cost_usd": summary["cost_usd"],
        "wall_s": round(wall, 1),
        "model_s": round(summary["model_ms"] / 1000, 1),
        "calls": summary["calls"],
        "web_searches": summary["web_searches"],
        "route": summary["route"],
        "finished": finished,
        "approved": bool(verdict and verdict.approved),
        "revisions": int(result.get("revisions") or 0),
        "error": result.get("error"),
    }


def run_baseline(topic: str, settings: Settings) -> dict:
    started = time.monotonic()
    try:
        result = baseline_module.run(topic, settings)
        error = None
    except ClaudeSessionLimitError:
        raise
    except Exception as exc:
        result = {"report": "", "cost_usd": 0.0, "web_searches": 0}
        error = f"{type(exc).__name__}: {exc}"

    wall = time.monotonic() - started
    return {
        "system": "baseline",
        "topic": topic,
        "report": result["report"],
        "cost_usd": round(result["cost_usd"], 6),
        "wall_s": round(wall, 1),
        "model_s": round(result.get("duration_ms", 0) / 1000, 1),
        "calls": 1,
        "web_searches": result.get("web_searches", 0),
        "route": ["baseline"],
        "finished": error is None,
        "error": error,
    }


def evaluate(topic: str, bucket: str, system: str, settings: Settings, refresh: bool) -> dict:
    path = _cache_path(topic, system)
    if path.is_file() and not refresh:
        return json.loads(path.read_text(encoding="utf-8"))

    row = run_team(topic, settings) if system == "team" else run_baseline(topic, settings)
    row["bucket"] = bucket

    scores = judge(topic, row["report"], settings)
    row["scores"] = scores.model_dump()
    row["quality"] = scores.mean
    row["words"] = len(row["report"].split())

    # Only cache a row that actually produced a report. Caching a failure
    # bakes it in permanently -- the next sweep reads it straight back and the
    # results table quietly reports a crash as a score of 1.00.
    if not row.get("error") and row.get("words", 0) > 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(row, indent=1, default=str), encoding="utf-8")
    return row


def _mean(rows: list[dict], key: str) -> float:
    values = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    return round(sum(values) / len(values), 3) if values else 0.0


def _score_mean(rows: list[dict], dimension: str) -> float:
    """Mean of one judge dimension across rows."""
    values = [
        r["scores"][dimension] for r in rows if isinstance(r.get("scores"), dict)
    ]
    return round(sum(values) / len(values), 2) if values else 0.0


def _table(rows: list[dict], buckets: list[str]) -> str:
    lines = [
        "| Bucket | System | n | Quality | Cite | Cover | Cost | Wall | Words | Searches |",
        "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for bucket in buckets + ["ALL"]:
        for system in ("team", "baseline"):
            subset = [
                r
                for r in rows
                if r["system"] == system and (bucket == "ALL" or r["bucket"] == bucket)
            ]
            if not subset:
                continue
            lines.append(
                f"| {bucket} | {system} | {len(subset)} | "
                f"{_mean(subset, 'quality'):.2f} | "
                f"{_score_mean(subset, 'citation_integrity'):.2f} | "
                f"{_score_mean(subset, 'coverage'):.2f} | "
                f"${_mean(subset, 'cost_usd'):.3f} | "
                f"{_mean(subset, 'wall_s'):.0f}s | "
                f"{_mean(subset, 'words'):.0f} | "
                f"{_mean(subset, 'web_searches'):.1f} |"
            )
    return "\n".join(lines)


def main() -> None:
    enable_utf8_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="Topics per bucket (0 = all).")
    parser.add_argument("--bucket", default="", help="Only this bucket.")
    parser.add_argument("--refresh", action="store_true", help="Ignore the cache.")
    parser.add_argument("--team-only", action="store_true", help="Skip the baseline.")
    parser.add_argument(
        "--spend-window-hours",
        type=float,
        default=5.0,
        help=(
            "Trailing window the spend cap measures over. Set this to the time "
            "since your allowance last reset, so pre-reset spend is not counted "
            "against the current session."
        ),
    )
    parser.add_argument(
        "--max-spend",
        type=float,
        default=6.0,
        help=(
            "Stop once notional spend in the trailing 5h window reaches this. "
            "A proxy for the session allowance, which the CLI does not expose."
        ),
    )
    args = parser.parse_args()

    settings = Settings.load()
    topics: dict[str, list[str]] = yaml.safe_load(
        (ROOT / "eval" / "topics.yaml").read_text(encoding="utf-8")
    )
    if args.bucket:
        topics = {args.bucket: topics[args.bucket]}
    if args.limit:
        topics = {b: t[: args.limit] for b, t in topics.items()}

    systems = ["team"] if args.team_only else ["team", "baseline"]
    total = sum(len(t) for t in topics.values()) * len(systems)
    rows: list[dict] = []
    done = 0

    # Round-robin across buckets rather than finishing one bucket at a time.
    # A full sweep takes hours; if it is interrupted, an interleaved order
    # leaves a partial result that still spans easy/broad/ambiguous, which is
    # the comparison the whole eval exists to make. Bucket-at-a-time would
    # leave a prefix that says nothing about where the team earns its cost.
    ordered: list[tuple[str, str]] = []
    for index in range(max(len(t) for t in topics.values())):
        for bucket, bucket_topics in topics.items():
            if index < len(bucket_topics):
                ordered.append((bucket, bucket_topics[index]))

    stopped_early = ""
    try:
        for bucket, topic in ordered:
            for system in systems:
                done += 1
                print(
                    f"[{done}/{total}] {system:8} | {bucket:16} | {topic[:60]}",
                    flush=True,
                )
                spent = recent_spend(settings.run_log, hours=args.spend_window_hours)
                if args.max_spend and spent >= args.max_spend:
                    raise _SpendCapReached(
                        f"${spent:.2f} of notional spend in the trailing "
                        f"{args.spend_window_hours:g}h window, "
                        f"at or over the --max-spend cap of ${args.max_spend:.2f}"
                    )
                row = evaluate(topic, bucket, system, settings, args.refresh)
                rows.append(row)
                print(
                    f"          quality={row['quality']:.2f} "
                    f"cost=${row['cost_usd']:.3f} wall={row['wall_s']:.0f}s "
                    f"words={row['words']}"
                    + (f"  ERROR: {row['error']}" if row.get("error") else ""),
                    flush=True,
                )
    except (ClaudeSessionLimitError, _SpendCapReached) as exc:
        # Stop the sweep, keep what completed. Everything already cached
        # survives, so re-running after the reset resumes rather than restarts.
        stopped_early = str(exc)
        print(f"\n!! STOPPED: {exc}", flush=True)
        print(
            f"!! {len(rows)} of {total} evaluations completed. Cached results are "
            f"kept -- re-run after the reset to continue where this left off.",
            flush=True,
        )

    # Scored rows only. A run that failed has no report to judge, and folding
    # its floor score of 1.00 into the means would misreport an outage as poor
    # quality -- which is exactly what an earlier sweep did.
    scored = [r for r in rows if not r.get("error") and r.get("words", 0) > 0]
    failed = [r for r in rows if r not in scored]

    team_rows = [r for r in scored if r["system"] == "team"]
    routes = summarise(
        [check(r["route"], max_revisions=settings.max_revisions, finished=r["finished"])
         for r in team_rows]
    )

    out_dir = settings.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval_rows.json").write_text(
        json.dumps(rows, indent=1, default=str), encoding="utf-8"
    )

    table = _table(scored, list(topics))
    topics_done = len({r["topic"] for r in scored})
    report_lines = [
        "# Evaluation: multi-agent team vs single-agent baseline",
        "",
        f"Topics scored: **{topics_done}** of {sum(len(t) for t in topics.values())} "
        f"across {len(topics)} buckets. "
        f"Quality is the mean of five 1-5 judge scores; Cite is citation integrity.",
        "",
    ]
    if stopped_early:
        report_lines += [
            f"> **Partial sweep.** Stopped early: {stopped_early}",
            ">",
            "> Completed results are cached; re-running resumes from here.",
            "",
        ]
    if failed:
        report_lines += [
            f"> {len(failed)} run(s) failed and are excluded from the scores below.",
            "",
        ]
    report_lines += [
        table,
        "",
        "## Route accuracy (team only)",
        "",
        f"- Clean runs: **{routes['clean']}/{routes['runs']}** "
        f"(accuracy {routes['accuracy']:.0%})",
        f"- Mean supersteps: {routes.get('mean_steps')}, "
        f"mean worker calls: {routes.get('mean_worker_calls')}",
    ]
    if routes["violations"]:
        report_lines.append("- Violations observed:")
        report_lines += [f"  - {v} x{n}" for v, n in routes["violations"].items()]
    else:
        report_lines.append("- No routing violations observed.")

    approved = sum(1 for r in team_rows if r.get("approved"))
    revised = sum(1 for r in team_rows if int(r.get("revisions") or 0) > 0)
    report_lines += [
        "",
        "## Critic behaviour",
        "",
        f"- Drafts approved by the critic: {approved}/{len(team_rows)}",
        f"- Runs that needed at least one revision: {revised}/{len(team_rows)}",
        "",
        f"_Raw rows: `out/eval_rows.json`. Cached artefacts: `out/eval_cache/`._",
    ]

    text = "\n".join(report_lines)
    (out_dir / "eval_results.md").write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"\nwritten -> {out_dir / 'eval_results.md'}")


if __name__ == "__main__":
    main()
