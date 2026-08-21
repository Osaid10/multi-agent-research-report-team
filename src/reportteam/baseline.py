"""The single-agent baseline: one agent, one shot, same topic.

The brief's sharpest instruction is to always keep this around. A multi-agent
team is expensive -- five specialists, a routing call between each, and a critic
that can send the whole thing round again -- and the only way to know whether
that expense bought anything is to have one agent do the same job in one pass
and compare.

It is given the same web access the researchers get, so the comparison isolates
*orchestration* rather than capability. If the team cannot beat this on
groundedness and coverage, the team is not worth its cost, and saying so is a
more valuable result than a report.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from .claude_cli import for_role
from .config import Settings

BASELINE_PROMPT = """\
You are a research analyst. Given a topic, research it on the web and write a \
complete, cited report in Markdown -- all in one pass.

- Use the WebSearch tool to gather current information. Do not answer from memory.
- Every factual claim must carry its source inline as [Title](url).
- Never invent a URL.
- Structure the report with `##` section headings and finish with a `## Sources` \
section listing every source you cited.
- Be specific: prefer numbers, dates and named organisations over generalities.
"""


def run(topic: str, settings: Settings | None = None) -> dict:
    """Write a report in a single agent call. Returns the report and its metrics."""
    settings = settings or Settings.load()

    message = for_role("baseline", settings).invoke(
        [
            SystemMessage(BASELINE_PROMPT),
            HumanMessage(f"Research and write a report on: {topic}"),
        ]
    )
    meta = message.response_metadata

    return {
        "report": message.text.strip(),
        "cost_usd": meta.get("cost_usd", 0.0),
        "duration_ms": meta.get("duration_ms", 0),
        "web_searches": meta.get("web_search_requests", 0),
        "num_turns": meta.get("num_turns", 0),
    }
