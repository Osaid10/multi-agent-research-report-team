"""The supervisor: routes work and decides when the job is done.

Two design commitments here, both from the brief's guardrails.

*Dumb on tools.* The supervisor is launched with `--tools ""`, so it cannot
search, write or grade even if a prompt tells it to. Delegation is the only
thing it can physically do.

*Never re-paraphrase finished work.* The supervisor is handed a compact
*status summary* — what exists, what is missing — not the outline text, the
research notes or the draft. That keeps its prompt small and cheap, and means
a worker's output can never be distorted by passing through it.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from ..claude_cli import for_role
from ..config import Settings
from ..models import RouteDecision
from ..state import ReportState, pending_sub_questions

SUPERVISOR_PROMPT = """\
You are the supervisor of a research and report team. You have no tools and you \
never do the work yourself. Your only output is who works next.

Your team, and what each needs before it can run:
- **planner**: needs only the topic. Produces the outline and the sub-questions.
- **researcher**: needs the outline. Answers one outstanding sub-question per turn.
- **writer**: needs the outline and all research complete. Produces the draft.
- **critic**: needs a draft. Approves it or sends it back with revision notes.

The normal order is planner -> researcher (once per sub-question) -> writer -> \
critic. After the critic rejects a draft, the writer revises and the critic \
reviews again.

Answer FINISH when the critic has approved the draft, or when the revision \
limit has been reached and no further progress is possible.

Route strictly on what the status says is missing. Do not route to a worker \
whose inputs are not ready, and do not re-run a worker whose output already \
exists and has not been invalidated -- repeating finished work is the most \
expensive mistake you can make.
"""


def _status(state: ReportState, settings: Settings) -> str:
    """A compact description of what exists and what is missing.

    Deliberately a summary rather than the artefacts themselves. Feeding the
    supervisor the full draft on every routing decision would multiply its cost
    by the length of the report and tempt it to start editing.
    """
    lines: list[str] = [f"Topic: {state['topic']}"]

    outline = state.get("outline")
    if outline is None:
        lines.append("Outline: MISSING. The planner has not run.")
    else:
        lines.append(
            f"Outline: present ({len(outline.sections)} sections, "
            f"{len(outline.sub_questions)} sub-questions)."
        )
        pending = pending_sub_questions(state)
        if pending:
            lines.append(
                f"Research: {len(state.get('notes') or [])} of "
                f"{len(outline.sub_questions)} sub-questions answered. "
                f"STILL UNANSWERED: {'; '.join(pending)}"
            )
        else:
            lines.append(
                f"Research: COMPLETE -- all {len(outline.sub_questions)} "
                f"sub-questions answered."
            )

    draft = state.get("draft") or ""
    lines.append(
        f"Draft: {'present (' + str(len(draft.split())) + ' words)' if draft else 'MISSING'}."
    )

    verdict = state.get("verdict")
    revisions = int(state.get("revisions") or 0)
    if verdict is None:
        lines.append("Review: the critic has not seen this draft.")
    elif verdict.approved:
        lines.append("Review: APPROVED by the critic. The report is finished.")
    else:
        lines.append(
            f"Review: REJECTED by the critic ({revisions} of "
            f"{settings.max_revisions} revisions used). "
            f"The writer must revise. Issues: {'; '.join(verdict.issues[:3])}"
        )

    if revisions >= settings.max_revisions and not (verdict and verdict.approved):
        lines.append(
            "NOTE: the revision limit is reached. No further revision is "
            "allowed -- answer FINISH."
        )

    return "\n".join(lines)


def route(state: ReportState, settings: Settings) -> dict:
    """Decide who works next. Writes `next` and `reason`."""
    decision = (
        for_role("supervisor", settings)
        .with_structured_output(RouteDecision)
        .invoke(
            [
                SystemMessage(SUPERVISOR_PROMPT),
                HumanMessage(_status(state, settings)),
            ]
        )
    )
    return {"next": decision.next, "reason": decision.reason}
