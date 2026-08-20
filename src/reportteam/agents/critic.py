"""The critic: the gate. Grades a draft against the rubric and decides.

This is the agent that makes the whole thing worth building. Everything before
it is a pipeline that could have been a chain; the critic is what introduces a
*cycle*, and the brief is explicit that grounding and citations are its job --
the report must not contain claims the research does not support.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from ..claude_cli import for_role
from ..config import Settings
from ..models import Verdict
from ..state import ReportState
from .writer import _render_notes, _render_outline

CRITIC_PROMPT = """\
You are the critic. You review a draft report against the research it was \
written from, and you decide whether it ships.

Grade each dimension 1-5:

- **coverage**: does the draft address every section of the outline?
- **groundedness**: is every factual claim traceable to a finding in the \
research notes, with the source cited inline? This is the one that matters. \
Check the citations against the notes -- a citation to a URL that is not in the \
notes is a fabrication, and it is the most serious defect a draft can have.
- **structure**: does it follow the outline, read as prose, and carry a \
complete Sources section?
- **support**: is it free of unsupported assertions, weasel words and padding?

Approve only if the draft genuinely holds up. Be strict on the first pass: a \
draft you wave through with a 3 for groundedness is a report that misleads \
whoever reads it. But do not manufacture objections to a draft that is sound -- \
an endless loop is its own failure.

When you reject, your revision notes must be things the writer can act on \
without guessing. Name the section, quote the offending sentence, and say what \
would fix it. "Improve the analysis" is useless; "the 2026 deployment figure in \
section 2 cites no source -- cite the CATL finding from the notes or cut the \
sentence" is actionable.
"""


def review(state: ReportState, settings: Settings) -> dict:
    """Grade the draft. Writes `verdict`, `revision_notes` and `revisions`."""
    prompt = "\n".join(
        [
            f"Report topic: {state['topic']}",
            "",
            "## The outline the draft was meant to follow",
            _render_outline(state.get("outline")),
            "",
            "## The research notes the draft must be grounded in",
            _render_notes(state.get("notes") or []),
            "",
            "## The draft under review",
            state.get("draft", "(empty draft)"),
        ]
    )

    verdict = (
        for_role("critic", settings)
        .with_structured_output(Verdict)
        .invoke([SystemMessage(CRITIC_PROMPT), HumanMessage(prompt)])
    )

    revisions = int(state.get("revisions") or 0)
    if verdict.approved:
        # Clear the notes on approval so a later pass can never act on
        # complaints that have already been addressed.
        return {"verdict": verdict, "revision_notes": [], "revisions": revisions}

    return {
        "verdict": verdict,
        "revision_notes": verdict.revision_notes or verdict.issues,
        "revisions": revisions + 1,
    }
