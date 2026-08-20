"""The typed contracts between agents.

Every hand-off in this team is a schema, not a paragraph. That is deliberate:
the failure mode of a multi-agent system is one agent misreading another's
prose, and the cheapest fix is to stop passing prose. These models are also
literally the prompt — the CLI receives `model_json_schema()` and validates
against it, so each `description=` is instruction to the model, not just
documentation for the reader. Write them accordingly.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Source(BaseModel):
    """One citable web source."""

    url: str = Field(description="The full URL. Must be a real page you actually read.")
    title: str = Field(description="The page or article title.")


class SubQuestion(BaseModel):
    """One focused, independently researchable question."""

    question: str = Field(
        description=(
            "A specific question answerable from a handful of web sources. Must "
            "stand alone: a researcher will receive this and nothing else, so it "
            "cannot refer to 'the topic' or to other sub-questions."
        )
    )
    why: str = Field(
        description="One sentence on what this contributes to the final report."
    )


class Outline(BaseModel):
    """The planner's output: how the report will be structured and researched."""

    title: str = Field(description="A specific, non-generic title for the report.")
    angle: str = Field(
        description=(
            "One or two sentences on what this report argues or establishes, so "
            "the writer knows what it is for."
        )
    )
    sections: list[str] = Field(
        description="Section headings, in order, for the final report.",
        min_length=2,
        max_length=6,
    )
    sub_questions: list[SubQuestion] = Field(
        description=(
            "The research agenda. Each is dispatched to its own researcher and "
            "they run concurrently, so they must not depend on one another."
        ),
        min_length=2,
        max_length=5,
    )


class ResearchNote(BaseModel):
    """One researcher's findings for one sub-question."""

    sub_question: str = Field(description="The question this answers, copied verbatim.")
    findings: list[str] = Field(
        description=(
            "Specific, factual findings. Prefer concrete numbers, dates and named "
            "actors over general statements. Each finding must be supported by "
            "one of the sources you list."
        ),
        min_length=1,
    )
    sources: list[Source] = Field(
        description=(
            "The sources these findings came from. Never invent a URL: an "
            "unciteable finding is worth less than no finding, because the "
            "critic will reject the draft that uses it."
        )
    )
    gaps: str = Field(
        default="",
        description=(
            "Anything you could not establish. Say so plainly rather than "
            "filling the hole with a plausible guess."
        ),
    )


class RubricScores(BaseModel):
    """The critic's scores, 1-5, against the brief's four rubric dimensions."""

    coverage: int = Field(
        ge=1, le=5, description="Does the draft cover every section of the outline?"
    )
    groundedness: int = Field(
        ge=1,
        le=5,
        description=(
            "Is every factual claim traceable to a cited source in the research "
            "notes? This is the one that matters most."
        ),
    )
    structure: int = Field(
        ge=1, le=5, description="Is it well organised and readable, following the outline?"
    )
    support: int = Field(
        ge=1,
        le=5,
        description="Is it free of unsupported assertions, hedging and filler?",
    )


class Verdict(BaseModel):
    """The critic's gate decision on a draft."""

    approved: bool = Field(
        description=(
            "True only if the draft meets every rubric dimension. Approving a "
            "weak draft defeats the purpose of the review; be strict on the "
            "first pass."
        )
    )
    scores: RubricScores
    issues: list[str] = Field(
        default_factory=list,
        description="Specific problems found. Quote the offending text where you can.",
    )
    revision_notes: list[str] = Field(
        default_factory=list,
        description=(
            "Concrete instructions the writer can act on directly. Not 'improve "
            "the analysis' but 'the claim about 2026 grid deployments in section "
            "2 cites no source; either cite one from the notes or remove it'."
        ),
    )


# The workers a supervisor may delegate to, plus the terminal branch. Kept as a
# module-level constant because it appears in three places that must agree: the
# routing schema, the supervisor's prompt, and the graph's conditional edges.
WorkerName = Literal["planner", "researcher", "writer", "critic", "FINISH"]


class RouteDecision(BaseModel):
    """The supervisor's only output: who works next."""

    next: WorkerName = Field(
        description=(
            "The worker to run next, or FINISH when the critic has approved a "
            "draft and the report is complete."
        )
    )
    reason: str = Field(
        description=(
            "One sentence justifying the choice. Recorded for the route-accuracy "
            "evaluation, so be honest about why rather than restating the rule."
        )
    )
