"""LLM-as-judge: scores a finished report on the qualities the brief names.

The judge sees the topic and the report, and nothing else. That is a deliberate
constraint: the team produces research notes and the baseline does not, so
showing the judge the notes would hand the team an advantage that has nothing
to do with report quality. Both systems are graded on the artefact a reader
would actually receive.

`citation_integrity` is the closest thing to a grounding check available under
that constraint -- claims carrying inline sources, sources that resolve to a
plausible publisher for the claim, and a Sources section that matches the body.
It cannot verify that a URL is live, and the results table says so rather than
implying a stronger check than was made.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from reportteam.claude_cli import for_role
from reportteam.config import Settings

JUDGE_PROMPT = """\
You are an exacting editor scoring a research report. Score each dimension 1-5, \
where 3 means "acceptable but unremarkable" and 5 is reserved for genuinely \
excellent work. Do not inflate: a report where most scores are 5 tells the \
reader nothing.

- **coverage**: does the report address the topic fully, including the parts a \
knowledgeable reader would expect and would notice were missing?
- **citation_integrity**: does every substantive factual claim carry an inline \
source? Do the cited publishers plausibly support the claims attached to them? \
Is there a complete Sources section consistent with the body? Penalise heavily \
any confident factual claim with no source at all.
- **structure**: is it organised, readable, and does it build an argument rather \
than list disconnected facts?
- **specificity**: does it use concrete numbers, dates and named organisations, \
rather than hedged generalities like "adoption is growing"?

Also judge `usefulness`: would a smart reader who knew nothing about this topic \
come away genuinely informed and able to act on it?

In `notable_weakness`, name the single biggest problem in one sentence. If the \
report is genuinely strong, say what would have to improve to reach a 5.
"""


class JudgeScores(BaseModel):
    coverage: int = Field(ge=1, le=5)
    citation_integrity: int = Field(ge=1, le=5)
    structure: int = Field(ge=1, le=5)
    specificity: int = Field(ge=1, le=5)
    usefulness: int = Field(ge=1, le=5)
    notable_weakness: str

    @property
    def mean(self) -> float:
        return round(
            (
                self.coverage
                + self.citation_integrity
                + self.structure
                + self.specificity
                + self.usefulness
            )
            / 5,
            2,
        )


def judge(topic: str, report: str, settings: Settings | None = None) -> JudgeScores:
    """Score one report."""
    settings = settings or Settings.load()

    if not report.strip():
        return JudgeScores(
            coverage=1,
            citation_integrity=1,
            structure=1,
            specificity=1,
            usefulness=1,
            notable_weakness="The report is empty.",
        )

    return (
        for_role("judge", settings)
        .with_structured_output(JudgeScores)
        .invoke(
            [
                SystemMessage(JUDGE_PROMPT),
                HumanMessage(f"Topic: {topic}\n\n--- REPORT ---\n\n{report}"),
            ]
        )
    )
