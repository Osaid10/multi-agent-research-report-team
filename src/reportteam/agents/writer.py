"""The writer: turns the outline and the research notes into a cited draft."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from ..claude_cli import for_role
from ..config import Settings
from ..models import Outline, ResearchNote
from ..state import ReportState

WRITER_PROMPT = """\
You are the writer. You turn an outline and a set of research notes into a \
finished report in Markdown.

The single rule that matters: **every factual claim must come from the research \
notes, and must carry its source inline** as [Title](url). You have no other \
source of truth. If the notes do not support something, you cannot say it -- not \
softened, not hedged, not "it is widely believed". Where the notes record a gap, \
say plainly that the evidence does not settle it.

Structure:
- Follow the outline's sections, in order, using them as `##` headings.
- Open with a short paragraph stating what the report establishes.
- Close with a `## Sources` section listing every source you cited, once each.

Write in prose, not bullet fragments. Be specific and concrete; prefer the \
numbers and named actors in the notes over general statements. Do not pad -- a \
short report that is entirely grounded beats a long one that is not.
"""

REVISION_PROMPT = """\
A reviewer rejected your previous draft. Revise it.

Address every point below specifically. Do not rewrite from scratch and do not \
restructure what was already working -- fix what was named, keep the rest. If a \
point asks you to support a claim you cannot support from the notes, delete the \
claim rather than inventing a citation for it.
"""


def _render_notes(notes: list[ResearchNote]) -> str:
    """Lay the research out so every finding sits next to its citation."""
    if not notes:
        return "(no research notes -- you cannot write a grounded report)"

    blocks: list[str] = []
    for note in notes:
        lines = [f"### Research on: {note.sub_question}", "", "Findings:"]
        lines += [f"- {finding}" for finding in note.findings]
        lines += ["", "Sources available for these findings:"]
        lines += [f"- [{source.title}]({source.url})" for source in note.sources]
        if note.gaps:
            lines += ["", f"Known gaps (do not paper over these): {note.gaps}"]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _render_outline(outline: Outline | None) -> str:
    if outline is None:
        return "(no outline)"
    sections = "\n".join(f"{i}. {s}" for i, s in enumerate(outline.sections, 1))
    return f"Title: {outline.title}\nAngle: {outline.angle}\n\nSections:\n{sections}"


def write(state: ReportState, settings: Settings) -> dict:
    """Produce or revise the draft. Writes `draft`."""
    revision_notes = state.get("revision_notes") or []
    parts = [
        f"Report topic: {state['topic']}",
        "",
        "## Outline",
        _render_outline(state.get("outline")),
        "",
        "## Research notes",
        _render_notes(state.get("notes") or []),
    ]

    if revision_notes:
        numbered = "\n".join(f"{i}. {n}" for i, n in enumerate(revision_notes, 1))
        parts += [
            "",
            "## Your previous draft",
            state.get("draft", ""),
            "",
            "## Reviewer's required changes",
            numbered,
        ]

    system = WRITER_PROMPT + ("\n\n" + REVISION_PROMPT if revision_notes else "")
    message = for_role("writer", settings).invoke(
        [SystemMessage(system), HumanMessage("\n".join(parts))]
    )
    return {"draft": message.text.strip()}
