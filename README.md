# Multi-Agent Research & Report Team

A supervisor delegates to a planner, a fan-out of researchers, a writer and a
critic. The critic is the gate: it grades every draft against a rubric and sends
weak ones back to the writer until they pass. What comes out is a cited report;
what the project is actually about is the orchestration.

Built for the Mercurial Minds Agentic AI internship, against a LangGraph brief.

**Headline result:** across 7 paired topics the team scored 4.03 against a
single agent's 4.09 — at **4.9x the cost**. The average hides the finding: the
team wins on broad and ambiguous topics (**+0.50 citation integrity**) and
loses on easy factual ones (**-0.67**). Route accuracy was 8/8. Full analysis
in [FINDINGS.md](FINDINGS.md).

---

## Demo

**[3-minute walkthrough video](demo/demo_3min.mp4)** — one full run: the graph,
the parallel fan-out, and the critic rejecting a draft and passing the revision.

- `demo/demo.html` — the animated source; open it in a browser and it plays on
  the same clock. `?t=95` seeks, `?speed=2` runs double, `?paused=1&t=95` freezes a frame.
- `demo/VOICEOVER.md` — the narration script, cued to each beat.

---

## The system

```
                                 topic
                                   │
                                   ▼
                          ┌────────────────┐
                    ┌────▶│   SUPERVISOR   │──── FINISH ────┐
                    │     │  routes work   │                │
                    │     └────────────────┘                │
                    │        │    │     │                   ▼
                    │        │    │     │             ┌──────────┐
        ┌───────────┴──┐     │    │     │             │ finalize │
        │              │     │    │     │             └──────────┘
   ┌────────┐   ┌──────────┐ │ ┌──────┐ │ ┌────────┐
   │ writer │   │ planner  │ │ │research│ │ critic │
   └────────┘   └──────────┘ │ └──────┘ │ └────────┘
        ▲            │        │    │           │
        │            ▼        │    │ Send      │ verdict
        │   ┌─────────────────┐│   ├──────────┐│
        │   │ approve_outline ││   ▼          ▼▼
        │   │  (human gate)   ││ researcher ×N  approved? ──▶ finalize
        │   └─────────────────┘│   │          │
        │                      │   └──────────┤ rejected
        └──────────────────────┴──────────────┘  (revise)
```

Two conditional edges, answering different questions — the distinction the brief
draws:

- **`route_from_supervisor`** — *who works next?* Reads the `next` state field.
- **`route_from_critic`** — *approved, or send it back?* Reads the verdict, and
  owns the revision cap.

### Shared state

Defined in [state.py](src/reportteam/state.py). Most channels overwrite; one does not.

| Channel | Written by | Reducer |
|---|---|---|
| `topic` | input | overwrite |
| `outline` | planner | overwrite |
| `notes` | researcher(s) | **`operator.add`** — appends |
| `draft` | writer | overwrite |
| `verdict` | critic (cleared by writer) | overwrite |
| `revision_notes` | critic | overwrite |
| `revisions` | critic | overwrite |
| `next` / `reason` | supervisor | overwrite |
| `final_report` | finalize | overwrite |

`notes` is the one that matters. It has used an appending reducer since Phase 1,
which is why Phase 5's concurrent fan-in needed no merge code and no state
rewrite — parallelism became a routing change.

### Routing rules

The supervisor is handed a **status summary** — what exists, what is missing —
never the outline text, the notes or the draft. That keeps its prompt cheap and
means a worker's output can never be distorted by passing through it (the
brief's "do not re-paraphrase finished work").

| Worker | May run when |
|---|---|
| `planner` | always, if no outline exists |
| `research` / `researcher` | an outline exists and sub-questions are unanswered |
| `writer` | outline exists and all sub-questions are answered |
| `critic` | a draft exists that it has not yet reviewed |
| `FINISH` | the critic approved, or the revision cap is spent |

Two guards, because they catch different failures: `max_revisions` (a stubborn
critic) and LangGraph's `recursion_limit` (a routing loop).

---

## Quick start

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows
pip install -r requirements.txt && pip install -e .
claude                                              # sign in once, then quit
cp .env.example .env                                # optional: add LANGSMITH_API_KEY
```

There is **no model API key**. Every agent runs through the `claude` CLI on your
logged-in session — see below.

```bash
reportteam run "How are grid operators handling negative electricity prices?"
reportteam run "..." --sabotage            # force a revision cycle
reportteam run "..." --gated               # pause for outline approval
reportteam approve --thread-id <id>        # resume, in a separate process
reportteam baseline "..."                  # single-agent comparison
reportteam graph --which team              # print the graph
python -m eval.run_eval --limit 2          # smoke-test the eval
```

### The 3-minute demo

[demo/demo_3min.mp4](demo/demo_3min.mp4) walks one sabotaged run end to end —
the critic rejecting a thin draft, the writer revising, and the run terminating
on the revision cap — then the parallel fan-out and the eval. Every number and
every quoted critic issue in it is read off `out/runs.jsonl` and the run logs
for run `0fccd2c95218`; the frames are a replay of that log, not a live capture,
because the CLI prints only at the end of a run.
[demo/demo.html](demo/demo.html) is the source and plays on the same clock
(`?t=95` to seek, `?speed=2` to skim); [demo/VOICEOVER.md](demo/VOICEOVER.md) is
the narration script with its cue times.

---

## The one big decision: the CLI as the model backend

This project does not call the Anthropic API. Every worker is a
`claude -p --output-format json` subprocess, wrapped as a LangChain
`BaseChatModel` in [claude_cli.py](src/reportteam/claude_cli.py). That bought four
things:

1. **`--json-schema` gives validated structured output.** The envelope returns
   `structured_output` already parsed and schema-checked, including nested
   `$defs`. The planner's `Outline` and the critic's `Verdict` need no
   JSON-repair fallback at all.
2. **`--tools ""` makes "the supervisor is dumb on tools" structural.** The
   brief lists this as a guardrail to be prompted for and checked afterwards.
   Here the supervisor process has no tools, so the violation is impossible.
3. **Cost and latency arrive per call** — `total_cost_usd`, `modelUsage`,
   `duration_ms`, `num_turns`. Phase 6's whole measurement story, with no price
   table to maintain and no token accounting to get wrong.
4. **Built-in `WebSearch`** — no Tavily key, no second signup.

Wrapping it as a `BaseChatModel` rather than shelling out from nodes keeps
LangSmith tracing automatic and `with_structured_output()` idiomatic.
`bind_tools` **emulates** tool calling through the output schema — the model
picks `tool_call` or `final_answer` inside a generated envelope — which is what
lets the prebuilt `create_supervisor` helper run against a backend that has no
tool-calling API.

### Deviations from the brief, stated plainly

- **No Tavily.** Web search is Claude's built-in server-side tool, so the search
  loop happens *inside* the researcher process rather than as a LangChain tool
  bound to a graph node. This fits the brief's own framing (every worker is an
  agent loop; the graph coordinates them), but it is a real gap against the
  brief's "Tools" row: **no LangChain tool is ever bound to a graph node in this
  project.** A Tavily backend would close it and is not implemented — stated
  here rather than left as an implied feature.
- **`langgraph-supervisor` appears only in Phase 0.** It is soft-deprecated —
  LangChain now recommends building the supervisor pattern directly — and it
  needs a tool-calling model. The brief's "build it by hand first" instruction
  and the library's own advice happen to agree.

### Four gotchas, each of which cost a debugging session

| Symptom | Cause | Fix |
|---|---|---|
| Every call ~5x too expensive | `--append-system-prompt` keeps Claude Code's full harness prompt (7,375 cache-creation tokens on a two-line request) | `--system-prompt` replaces it — 940 tokens |
| Researcher burns turns explaining it can't search (one $0.43 call) | `--tools WebSearch` grants existence, not permission | add `--allowedTools` + `--permission-mode dontAsk` |
| `"Not logged in"` | `--bare` forces API-key-only auth and never reads the OAuth session | never pass `--bare` |
| Researcher reaches for `mcp__playwright__browser_navigate` | workers inherit the *user's* global MCP servers | `--strict-mcp-config` |

Two more, found while building:

- **Windows can't `CreateProcess` a `.cmd`**, and going via `cmd.exe /c` re-parses
  the JSON schema argument. The npm shim execs a real `claude.exe` — target that.
  Prompts go in on **stdin**, because Windows caps a command line at ~32K chars
  and the writer's prompt carries every research note.
- **`usage.input_tokens` under-reports** (9 tokens for a 10K-token prompt). Cost
  accounting must read `modelUsage[*]`, which is the per-call total.

---

## The phases, and what each one actually taught

Every phase stays runnable (`--graph phase1` … `--graph team`); the progression
is part of the deliverable.

| Phase | Graph | Concept | Verified result |
|---|---|---|---|
| 0 | `phase0_hello` | prebuilt supervisor, tracing | routes and finishes; 3 calls to count 8 words |
| 1 | `phase1_manual` | `StateGraph`, `next`, conditional edges | `supervisor → researcher → supervisor → FINISH` |
| 2 | `phase2_pipeline` | shared state, multi-step delegation | 2,592-word report, 97 citations, 42 searches |
| 3 | `phase3_reflection` | reflection cycle, structured grading, loop guard | sabotaged draft → 2 revisions → clean termination |
| 4 | `phase4_persist` | checkpointer, threads, `interrupt()` | paused, process exited, **separate** process resumed and finished |
| 5 | `team` | `Send` fan-out, map-reduce | 270.6s of research compressed into 104.4s — **2.6×** |
| 6 | `eval/` | route accuracy, quality rubric, cost | 100% route accuracy; team wins on hard topics only — [FINDINGS.md](FINDINGS.md) |

### Phase 3: the critic earns its keep

**[Traced run: critic rejects a draft, writer fixes it, critic approves](https://smith.langchain.com/o/a1acb5c0-351f-4784-b723-e90a6c1a4661/projects/p/3d3f5ef1-3e0a-4e18-b002-593b30b272da/r/01a047a9-34f7-74b0-8b39-960f075ae076?trace_id=01a047a9-34f7-74b0-8b39-960f075ae076&start_time=2026-08-28T09:17:46.615693)**

Run `--sabotage` and the writer deliberately produces a thin, uncited first
draft. The critic rejected it, the writer revised, and on the *second* pass the
critic still caught two real defects — not stylistic ones:

> Section 1: *'at a 4:1 ratio, a heat pump needs to comfortably clear an SPF of
> 4.0'* is attached to the Think House citation, but the notes only state the
> 3:1/SPF-3.0 threshold — this is the writer's own extrapolation.

> Section 4: the £465 figure does not appear in the Findings; it appears only
> embedded in two source *titles*.

Both are the failure the brief names: claims the research does not support. The
report shipped with an explicit note that it hit the revision limit without
passing review, rather than quietly presenting itself as approved.

### Phase 5: what the fan-out actually bought

Five sub-questions, from the run log:

```
researcher ends +  0.0s  duration 15.8s  -> started -15.8s
researcher ends + 27.2s  duration 43.4s  -> started -16.1s
researcher ends + 34.6s  duration 50.8s  -> started -16.1s
researcher ends + 60.0s  duration 75.9s  -> started -15.9s
researcher ends + 88.6s  duration 84.8s  -> started + 3.8s   <- waited for a slot
```

Four start together (the concurrency cap); the fifth starts the moment the first
releases its slot. **270.6s of model work inside a 104.4s window.** Supervisor
calls also fell from 7 to 3, because it no longer routes each question
individually — the saving is more than wall-clock.

A whole-run check: the parallel graph finished in **299s wall against 438s of
model time**. Model time exceeding wall-clock is only possible if work
overlapped.

**Phase 0 vs Phase 1 is worth running back to back.** The prebuilt helper routes
through `transfer_to_<worker>` tool calls; you can see the machinery in its route
(`supervisor → agent → tools → counter → model → supervisor`). The hand-built
version routes on a state field, with no tools anywhere near the supervisor.

**On the checkpointer:** the brief suggests `InMemorySaver`, and Phase 4 starts
there — but in-memory state dies with the process, so "resume" can only ever mean
"resume inside the same script". SQLite makes `reportteam run --gated` and
`reportteam approve --thread-id` genuinely separate invocations, which is the
behaviour being taught. `--in-memory` shows the difference.

**On the recursion limit:** the brief warns that hitting it is almost always a
routing loop to fix in the prompt, not to raise away. That warning earned its
place — an early run spun `writer → supervisor → writer` until it tripped,
because the writer produced a new draft while the *old* rejected verdict was
still in state, so the supervisor kept reading "the writer must revise". The fix
was for a new draft to invalidate its review, not to move the limit. The limit
then moved from 25 to 40 for a different and legitimate reason: the sequential
graphs spend two supersteps per sub-question, so a five-question topic with two
revisions genuinely reaches ~23.

---

## Layout

```
src/reportteam/
  claude_cli.py     the model backend: claude -p as a BaseChatModel
  config.py         per-role model, tools and budget ceilings
  models.py         the typed contracts between agents
  state.py          ReportState and its reducers
  trace.py          LangSmith + the local JSONL run log
  baseline.py       the single-agent comparison
  cli.py            entry point
  agents/           planner, researcher, writer, critic, supervisor
  graphs/           phase0 … phase5, each still runnable
eval/
  topics.yaml       20 topics in 3 buckets
  route_accuracy.py did it delegate sensibly and stop?
  judge.py          LLM-as-judge quality rubric
  run_eval.py       the sweep, with per-topic caching
```

Observability is split on purpose: **LangSmith for debugging** (a misrouting
supervisor is near-impossible to read from code), **a local JSONL log for
measuring** (so the eval is a pure file reader with no network dependency).
