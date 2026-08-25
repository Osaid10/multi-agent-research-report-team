# Findings: does the team earn its cost?

The brief's capstone question is not "is the report good?" but "when do five
agents beat one?" Multi-agent is expensive by construction — a planner, N
researchers, a writer, a critic, and a routing call between each — so the
deliverable is evidence about *when* that expense buys something.

## The verdict

**On this evidence the team does not beat a single agent on average, and the
average hides the whole story.** Across 7 paired topics the team scored 4.03
against the baseline's 4.09 — a rounding error — at **4.9× the cost** and
**4.1× the latency**. But the direction reverses cleanly by topic shape: the
team wins on broad and ambiguous topics (+0.10 quality, **+0.50 citation
integrity** on both), and loses on easy factual ones (−0.27 quality, −0.67
citation integrity). The one thing the team was built to protect — every claim
tied to a source — is the thing it improves where the topic is genuinely hard,
and degrades where it is not. If you run this in production, route simple
questions to one agent and reserve the team for questions that need
decomposing; running the team on everything pays five times over for a worse
answer most of the time.

## Method

- **Systems.** The full parallel team (`graphs/team.py`) versus `baseline.py` —
  one agent, one pass, the same web-search tool. Only the orchestration differs.
- **Scoring.** An LLM judge (`eval/judge.py`) rates five dimensions 1–5. It sees
  the topic and the report only. The team produces research notes and the
  baseline does not, so showing the judge those notes would hand the team an
  advantage unrelated to the artefact a reader receives.
- **Pairing.** Only topics where *both* systems produced a report are compared.
  The auto-generated `out/eval_results.md` has a team n=8 against a baseline
  n=7; averaging across different topic sets is not a comparison, so every
  number below is paired.

## Results (7 paired topics)

| Bucket | n | Quality (team → base) | Citation integrity | Cost | Latency |
|---|--:|---|---|--:|--:|
| easy_factual | 3 | 3.93 → 4.20 (**−0.27**) | 3.00 → 3.67 (**−0.67**) | 5.0× | 4.1× |
| broad_multipart | 2 | 4.20 → 4.10 (**+0.10**) | 3.50 → 3.00 (**+0.50**) | 4.2× | 3.7× |
| ambiguous | 2 | 4.00 → 3.90 (**+0.10**) | 3.00 → 2.50 (**+0.50**) | 5.9× | 4.6× |
| **ALL** | **7** | **4.03 → 4.09 (−0.06)** | **3.14 → 3.14 (0.00)** | **4.9×** | **4.1×** |

Per topic:

| Bucket | Team | Base | Δ | Team $ | Base $ |
|---|--:|--:|--:|--:|--:|
| ambiguous — remote work & productivity | 3.80 | 3.80 | +0.00 | $1.99 | $0.31 |
| ambiguous — is nuclear cost-competitive | 4.20 | 4.00 | +0.20 | $2.29 | $0.42 |
| broad — European negative pricing | 4.00 | 4.00 | +0.00 | $1.93 | $0.46 |
| broad — solid-state commercialisation | 4.40 | 4.20 | +0.20 | $2.43 | $0.59 |
| easy — data centre electricity | 3.80 | 4.20 | −0.40 | $1.72 | $0.27 |
| easy — LFP energy density | 4.00 | 4.00 | +0.00 | $1.96 | $0.29 |
| easy — DMA enforcement status | 4.00 | 4.40 | −0.40 | $1.50 | $0.48 |

## Why the team loses on easy topics

Not randomness — a structural failure mode worth naming. The planner is
required to decompose a topic into 2–5 independent sub-questions, and it does
that *whether or not the topic has 2–5 parts*. "What is the current energy
density of commercial LFP cells?" has one answer. Split into five research
threads and handed to a writer told to cover every outline section, it becomes
a 2,017-word report where the baseline wrote 1,466 — and the extra 550 words
are padding around a fact that needed a paragraph. The judge penalises exactly
that under `specificity`.

The team also ran **43.5 web searches per report against the baseline's 8.9**.
On a broad topic that breadth is the point. On a narrow one it is five agents
finding five framings of the same fact.

The obvious fix, untested here: let the planner emit a single sub-question — or
route past the team entirely — when the topic does not decompose. That is a
cheap change and the eval already exists to measure whether it works.

## Orchestration results

| Metric | Result |
|---|---|
| Route accuracy | **8/8 clean (100%)** — no violations |
| Mean supersteps per run | 11.1 |
| Drafts approved by the critic | 8/8 |
| Runs needing ≥1 revision | 3/8 |
| Parallel speedup (research phase) | **2.6×** — 270.6s of work in 104.4s |

Route accuracy is the number I trust most here, because it is computed from the
local run log against explicit preconditions (`eval/route_accuracy.py`) rather
than judged. No run researched before planning, wrote before research finished,
reviewed before a draft existed, repeated finished work, or failed to
terminate.

The critic approved 8/8 *after* revisions, with 3 runs needing at least one —
so the reflection loop is doing work rather than rubber-stamping. That took a
correction: an earlier prompt said "approve if the draft holds up", and the
critic kept finding smaller objections until the revision cap ran out, shipping
an *unapproved* report after paying for two rewrites. An explicit bar (every
dimension ≥4 and no fabricated citations) fixed it.

## What these numbers cannot support

- **n=7.** Below the brief's 15–25. Two topics per bucket for broad and
  ambiguous is enough to see a direction, not to claim significance. A ±0.10
  quality gap on n=2 is not a result; the citation-integrity gap (+0.50 on both
  hard buckets, −0.67 on easy) is more consistent and more likely real.
- **One judge, one pass.** No inter-rater check, no self-consistency runs. The
  judge is Sonnet grading Sonnet's output.
- **`citation_integrity` does not verify liveness.** It checks that claims carry
  sources and that publishers plausibly support them. It cannot confirm a URL
  resolves. A fabricated-but-plausible citation would pass.
- **Cost is notional.** The figures are API-equivalent from the CLI envelope,
  not billed spend.
- **The sweep stopped at a spend cap**, not at a natural end. 8 of 9 topics
  scored; results are cached, so `python -m eval.run_eval --limit 3` resumes.

## Engineering findings

Five bugs that only appeared by running the system, each worth more than the
report it produced:

1. **A routing loop.** The writer produced a new draft while the *old* rejected
   verdict was still in state, so the supervisor kept reading "the writer must
   revise" and spun `writer → supervisor → writer` into the recursion limit.
   Fixed where the brief says to fix it — a new draft invalidates its review —
   not by raising the limit.
2. **Tool isolation leak.** Workers inherited the machine's global MCP servers;
   a researcher reached for a Playwright browser it was never granted. Now
   `--strict-mcp-config`, so a worker's tool surface is what this project says
   it is on any machine.
3. **An unenforced concurrency cap.** `max_concurrent_researchers` was config
   decoration — LangGraph dispatches every `Send` in a superstep concurrently,
   so a five-question plan launched five processes regardless.
4. **A measurement bug that produced confident nonsense.** The first full sweep
   hit the account's session limit, treated it as one more failed row, raced
   through 18 remaining topics in under two minutes, scored every empty report
   1.00, and cached them. The results table reported an outage as uniformly
   terrible quality. Failures are no longer cached, and a spent allowance stops
   the sweep instead of being averaged into it.
5. **Unlogged baseline cost.** The baseline bypasses the graph, so no callback
   was attached and its spend never reached the run log — leaving the spend cap
   blind to roughly a third of each topic-pair's cost.

Number 4 is the one worth remembering. The others produced crashes or wrong
behaviour that announced itself. That one produced a *plausible-looking table
of numbers that meant nothing*, which is the failure mode an evaluation harness
exists to prevent and is uniquely capable of causing.
