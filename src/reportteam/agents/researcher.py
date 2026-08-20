"""The researcher: answers one sub-question from the web, with sources.

Written so the *same* node function serves both the sequential pipeline
(Phase 2, which picks the next unanswered question off the outline) and the
concurrent fan-out (Phase 5, where `Send` hands each worker its question
directly). Phase 5 then becomes a routing change rather than a rewrite.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from ..claude_cli import for_role
from ..config import Settings
from ..models import ResearchNote
from ..state import ReportState, pending_sub_questions

RESEARCHER_PROMPT = """\
You are a researcher. You answer exactly one question using the web, and you \
report only what you actually found.

- Use the WebSearch tool. Do not answer from memory: your training data is \
older than the web, and this report will be judged on whether its claims hold up.
- Every finding must be supported by a source you list, and every source must \
be a page you actually retrieved. A fabricated URL is the single worst thing \
you can produce here -- the critic checks them, the draft gets rejected, and \
the whole team pays for the revision.
- Prefer specifics: numbers, dates, named organisations. "Adoption is growing" \
is worthless to the writer; "shipments reached 1 GWh by end-2026 (CATL)" is not.
- If you cannot establish something, put it in `gaps` and move on. An honest \
gap is useful; a confident guess is a liability.
"""


def research(state: ReportState, settings: Settings) -> dict:
    """Answer one sub-question. Appends to `notes`.

    Takes its question from `sub_question` when the caller supplied one (the
    Phase 5 `Send` payload), otherwise claims the next unanswered question off
    the outline (the Phase 2 sequential loop).
    """
    question = (state.get("sub_question") or "").strip()
    if not question:
        remaining = pending_sub_questions(state)
        if not remaining:
            # Nothing to do. Returning an empty update is important: raising
            # here would kill a run whose supervisor merely routed one step too
            # many, which is a recoverable mistake.
            return {}
        question = remaining[0]

    note = (
        for_role("researcher", settings)
        .with_structured_output(ResearchNote)
        .invoke(
            [
                SystemMessage(RESEARCHER_PROMPT),
                HumanMessage(
                    f"Overall report topic (context only): {state['topic']}\n\n"
                    f"The question you must answer: {question}\n\n"
                    f"Copy that question verbatim into the `sub_question` field."
                ),
            ]
        )
    )

    # The model occasionally paraphrases the question despite being told not
    # to, which would break `pending_sub_questions` and send the team round the
    # same question forever. Overwrite it with the question we actually asked.
    note.sub_question = question
    return {"notes": [note]}
