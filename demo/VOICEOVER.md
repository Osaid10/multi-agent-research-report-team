# 3-minute demo — narration script

For `demo/demo_3min.mp4` — 3:01.8, 1920×1080, 30 fps, no audio.

Target pace **145 words per minute** (≈2.4 words/second). Total script: **441
words**, which lands at 3:02 with the pauses written in. If you naturally read
faster, take the pauses longer rather than slowing the words down.

**Sync:** cue times match the timecode burned into the bottom-right of the frame
and the beat ticks on the progress bar, so you can scrub to any cue and check.
The video's own position runs about half a second ahead of that timecode —
irrelevant at this pace, but that's the discrepancy if you notice it.

**Rehearsing:** open `demo/demo.html` in a browser and it plays on the same
clock. `?t=95` starts at 1:35, `?speed=2` skims, `?paused=1&t=95` freezes a
frame.

Four lines are marked **↷ carries over the cut** — they finish a few seconds
into the next scene on purpose. That's normal narration and it stops the joins
from sounding clipped. Don't rush to beat the transition.

---

## The script

### 1 — Title `0:00 → 0:11`

**0:00** · *(title card, roles appearing)* — 28 words, 11s

> This is a multi-agent research and report team, built on LangGraph. A
> supervisor delegates to a planner, researchers, a writer, and a critic that
> acts as the gate.

*Beat until the graph appears.*

---

### 2 — Architecture `0:11 → 0:34`

**0:12** · *(graph drawing itself)* — 24 words, 10s

> It's one graph, with two conditional edges answering different questions: who
> works next — and, is this draft approved, or does it go back?

**0:22** · *(state table, `notes` highlighted)* — 11 words, 5s

> Most state channels overwrite. Notes appends, and has since phase one.

**0:27** · *(the `--tools ""` panel)* — 25 words, 10s ↷ carries over the cut

> Every worker is a Claude CLI subprocess. The supervisor's runs with no tools
> at all — so a tool violation is impossible, not just discouraged.

---

### 3 — The run `0:34 → 2:09`

**0:37** · *(command finishing, `sabotage: ON` in amber)* — 20 words, 8s

> This is one full run, replayed from its log. Sabotage is on, which forces a
> deliberately weak first draft.

**0:46** · *(planner line, five sub-questions listing out)* — 15 words, 6s

> The supervisor routes to the planner, and the planner breaks the topic into
> five sub-questions.

**0:52** · *(researchers streaming, cost and search counters climbing)* — 50 words, 21s

> Each one goes out to its own researcher, which runs a web-search loop and
> appends a note. Five researchers, thirty-two searches. The supervisor never
> sees those notes — it's handed a status summary of what exists and what's
> missing, so a worker's output can't be distorted by passing through it.

**1:15** · *(writer line, in pink — 379 tokens)* — 18 words, 7s

> Every question answered, so the writer runs — and produces three hundred and
> seventy-nine tokens. Thin, barely cited. That's the sabotage.

**1:22** · *(critic grading, then the NEEDS WORK card)* — 23 words, 10s

> The critic grades it against the rubric — coverage, groundedness, structure,
> support — rejects it, and sends specific revision notes back to the writer.

**1:32** · *(writer redrafting, 8,310 tokens)* — 30 words, 12s

> The new draft clears the old verdict from state. That matters: when it didn't,
> the supervisor kept reading a stale rejection, and spun writer, supervisor,
> writer, into the recursion limit.

**1:44** · *(second critic pass, scores landing)* — 10 words, 4s

> The rewrite is a real report this time. Second pass.

**1:48** · *(NOT APPROVED card with the two quoted issues)* — 43 words, 18s

> And the critic still finds two things: a threshold the writer extrapolated and
> hung on someone else's citation, and a figure that only ever appeared inside
> two source titles. Both are claims the research doesn't support — exactly the
> failure the brief names.

**2:01** · *(run summary: route, calls, cost)* — 29 words, 12s ↷ carries over the cut

> The cap ends the run, not the critic — so the report ships saying it never
> passed review, rather than presenting itself as approved. Nineteen calls, one
> dollar sixty-seven.

---

### 4 — Parallel fan-out `2:09 → 2:30`

**2:13** · *(gantt bars growing, then the four figures)* — 44 words, 18s

> In phase five the researchers fan out with Send, and merge back through that
> same reducer. Two hundred and seventy seconds of model work, inside a
> hundred-and-four-second window — two point six times faster. The fifth waits
> for a slot: the concurrency cap is a real semaphore.

---

### 5 — The eval `2:30 → 2:51`

**2:31** · *(paired results table)* — 28 words, 12s

> Phase six: does the team earn its cost? Against a single-agent baseline, across
> seven paired topics — on average, no. The same quality, for four point nine
> times the money.

**2:43** · *(per-topic rows, route accuracy, the finding)* — 27 words, 11s ↷ carries over the cut

> But the average hides the story. The team wins on broad and ambiguous topics,
> and loses on easy factual ones. Routing was clean — eight out of eight.

---

### 6 — Close `2:51 → 3:01`

**2:54** · *(limits and deliverables)* — 16 words, 7s

> The limits are stated too, along with five bugs that only running the thing
> could find.

*Let it land. Two seconds of silence to the end of the video.*

---

## Clean read-through

No annotations — read straight down, one paragraph per cue, pausing where the
paragraphs break.

> This is a multi-agent research and report team, built on LangGraph. A supervisor delegates to a planner, researchers, a writer, and a critic that acts as the gate.
>
> It's one graph, with two conditional edges answering different questions: who works next — and, is this draft approved, or does it go back?
>
> Most state channels overwrite. Notes appends, and has since phase one.
>
> Every worker is a Claude CLI subprocess. The supervisor's runs with no tools at all — so a tool violation is impossible, not just discouraged.
>
> This is one full run, replayed from its log. Sabotage is on, which forces a deliberately weak first draft.
>
> The supervisor routes to the planner, and the planner breaks the topic into five sub-questions.
>
> Each one goes out to its own researcher, which runs a web-search loop and appends a note. Five researchers, thirty-two searches. The supervisor never sees those notes — it's handed a status summary of what exists and what's missing, so a worker's output can't be distorted by passing through it.
>
> Every question answered, so the writer runs — and produces three hundred and seventy-nine tokens. Thin, barely cited. That's the sabotage.
>
> The critic grades it against the rubric — coverage, groundedness, structure, support — rejects it, and sends specific revision notes back to the writer.
>
> The new draft clears the old verdict from state. That matters: when it didn't, the supervisor kept reading a stale rejection, and spun writer, supervisor, writer, into the recursion limit.
>
> The rewrite is a real report this time. Second pass.
>
> And the critic still finds two things: a threshold the writer extrapolated and hung on someone else's citation, and a figure that only ever appeared inside two source titles. Both are claims the research doesn't support — exactly the failure the brief names.
>
> The cap ends the run, not the critic — so the report ships saying it never passed review, rather than presenting itself as approved. Nineteen calls, one dollar sixty-seven.
>
> In phase five the researchers fan out with Send, and merge back through that same reducer. Two hundred and seventy seconds of model work, inside a hundred-and-four-second window — two point six times faster. The fifth waits for a slot: the concurrency cap is a real semaphore.
>
> Phase six: does the team earn its cost? Against a single-agent baseline, across seven paired topics — on average, no. The same quality, for four point nine times the money.
>
> But the average hides the story. The team wins on broad and ambiguous topics, and loses on easy factual ones. Routing was clean — eight out of eight.
>
> The limits are stated too, along with five bugs that only running the thing could find.

---

## Delivery notes

- **Numbers are spelled out where they're spoken** — "three hundred and seventy
  nine", "one dollar sixty-seven", "four point nine". Read them as written; they
  are the figures on screen.
- **Two lines carry the whole demo.** "That's the sabotage" at 1:15, and "The cap
  ends the run, not the critic" at 2:01. Slow down on both — everything else is
  setup for them.
- **Don't sell it.** The eval finding is that the team *loses* on average. Read
  2:31 flat and let the table do the work; hedging it sounds worse than stating
  it.
- **"Send" is a proper noun** (the LangGraph primitive), not the verb. A small
  stress on it at 2:13 stops it reading as "sent out".
- **If you run long**, cut the 0:22 line ("Most state channels overwrite…")
  first — the highlighted row says it — then the second half of 1:32, from
  "That matters". That buys 17 seconds without losing a beat.
- **If you run short**, hold the pause after 2:54 rather than adding words; the
  close card is on screen for another five seconds.
