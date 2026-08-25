"""Unit tests for the parts that must not need a model call to verify.

Everything here is pure logic: routing rules, state bookkeeping, prompt
flattening, and the tool-call emulation envelope. The graph's behaviour costs
money to exercise, so the pieces that *can* be pinned down for free should be.

    .venv/Scripts/python.exe -m pytest tests/ -q
"""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from eval.route_accuracy import check, summarise
from reportteam.claude_cli import (
    ChatClaudeCLI,
    ClaudeCLIError,
    ClaudeSessionLimitError,
    _action_schema,
    _split_prompt,
)
from reportteam.config import DEFAULT_ROLES, Settings
from reportteam.models import Outline, ResearchNote, Source, SubQuestion
from reportteam.state import initial_state, pending_sub_questions


# --- state ---------------------------------------------------------------


def _outline(*questions: str) -> Outline:
    return Outline(
        title="t",
        angle="a",
        sections=["s1", "s2"],
        sub_questions=[SubQuestion(question=q, why="w") for q in questions],
    )


def _note(question: str) -> ResearchNote:
    return ResearchNote(
        sub_question=question,
        findings=["f"],
        sources=[Source(url="https://example.com", title="ex")],
    )


def test_pending_tracks_answered_questions():
    state = initial_state("topic")
    state["outline"] = _outline("q1", "q2", "q3")
    assert pending_sub_questions(state) == ["q1", "q2", "q3"]

    state["notes"] = [_note("q2")]
    assert pending_sub_questions(state) == ["q1", "q3"]


def test_pending_matches_on_text_not_order():
    """Under Phase 5 fan-out, notes come back in arbitrary order."""
    state = initial_state("topic")
    state["outline"] = _outline("q1", "q2", "q3")
    state["notes"] = [_note("q3"), _note("q1")]
    assert pending_sub_questions(state) == ["q2"]


def test_pending_is_empty_without_an_outline():
    assert pending_sub_questions(initial_state("topic")) == []


# --- routing rules -------------------------------------------------------

GOOD_ROUTE = [
    "supervisor", "planner", "supervisor", "research", "researcher",
    "researcher", "supervisor", "writer", "supervisor", "critic",
    "writer", "supervisor", "critic", "finalize",
]


def test_clean_route_passes():
    assert check(GOOD_ROUTE).ok


def test_catches_the_writer_supervisor_loop():
    """The real bug this project hit: a stale verdict spun writer -> supervisor."""
    route = ["supervisor", "planner", "supervisor", "writer", "supervisor",
             "critic", "writer", "supervisor", "writer", "supervisor", "writer"]
    report = check(route, finished=False)
    assert not report.ok
    assert any("recursion limit" in v for v in report.violations)
    assert any("over the cap" in v for v in report.violations)


def test_catches_out_of_order_work():
    report = check(["supervisor", "researcher", "writer", "finalize"])
    assert any("before an outline existed" in v for v in report.violations)
    assert any("no outline" in v for v in report.violations)


def test_catches_review_before_draft():
    report = check(["supervisor", "planner", "supervisor", "critic", "finalize"])
    assert any("reviewed before any draft" in v for v in report.violations)


def test_catches_replanning():
    route = ["supervisor", "planner", "supervisor", "planner", "supervisor",
             "research", "researcher", "supervisor", "writer", "finalize"]
    assert any("planner ran twice" in v for v in check(route).violations)


def test_empty_route_is_a_violation_not_a_crash():
    assert not check([]).ok


def test_summarise_groups_like_violations():
    stats = summarise([check(GOOD_ROUTE), check([]), check([])])
    assert stats["runs"] == 3
    assert stats["clean"] == 1
    assert stats["accuracy"] == pytest.approx(0.333, abs=0.01)


# --- prompt flattening ---------------------------------------------------


def test_split_prompt_separates_system_from_body():
    system, body = _split_prompt(
        [SystemMessage("be terse"), HumanMessage("the question")]
    )
    assert system == "be terse"
    assert body == "the question"


def test_split_prompt_labels_tool_results():
    _, body = _split_prompt(
        [
            HumanMessage("q"),
            AIMessage(
                content="",
                tool_calls=[{"name": "search", "args": {"q": "x"}, "id": "1", "type": "tool_call"}],
            ),
            ToolMessage(content="the answer", name="search", tool_call_id="1"),
        ]
    )
    assert "[result of tool `search`]" in body
    assert "the answer" in body
    assert "you previously called" in body


def test_split_prompt_joins_multiple_system_messages():
    system, _ = _split_prompt([SystemMessage("one"), SystemMessage("two"), HumanMessage("q")])
    assert "one" in system and "two" in system


# --- tool-call emulation -------------------------------------------------


def test_action_schema_constrains_tool_names():
    schema = _action_schema(["alpha", "beta"])
    assert schema["properties"]["tool_name"]["enum"] == ["alpha", "beta"]
    assert schema["properties"]["action"]["enum"] == ["tool_call", "final_answer"]
    assert schema["required"] == ["action"]


def _envelope_message(payload: dict) -> AIMessage:
    return AIMessage(content="raw", additional_kwargs={"structured_output": payload})


def test_finalize_converts_an_envelope_into_a_real_tool_call():
    model = ChatClaudeCLI(role="t")
    out = model._finalize(
        _envelope_message(
            {
                "action": "tool_call",
                "tool_name": "search",
                "tool_arguments": json.dumps({"query": "sodium"}),
            }
        ),
        "tools",
    )
    assert out.tool_calls[0]["name"] == "search"
    assert out.tool_calls[0]["args"] == {"query": "sodium"}
    assert out.content == ""


def test_finalize_returns_the_final_answer_when_no_tool_is_called():
    model = ChatClaudeCLI(role="t")
    out = model._finalize(
        _envelope_message({"action": "final_answer", "final_answer": "hello"}), "tools"
    )
    assert out.content == "hello"
    assert not out.tool_calls


def test_finalize_rejects_malformed_tool_arguments():
    """Arguments travel as a string, so the schema cannot police them."""
    model = ChatClaudeCLI(role="t")
    with pytest.raises(ClaudeCLIError, match="not valid JSON"):
        model._finalize(
            _envelope_message(
                {"action": "tool_call", "tool_name": "s", "tool_arguments": "{not json"}
            ),
            "tools",
        )


def test_finalize_is_a_passthrough_outside_tool_mode():
    model = ChatClaudeCLI(role="t")
    message = _envelope_message({"anything": 1})
    assert model._finalize(message, "schema") is message


# --- CLI invocation ------------------------------------------------------


def test_no_tools_role_is_launched_with_an_empty_tool_list():
    """The supervisor's 'dumb on tools' guarantee is structural."""
    argv = ChatClaudeCLI(role="supervisor", tools=())._argv("sys", None)
    assert "--tools" in argv
    assert argv[argv.index("--tools") + 1] == ""
    assert "--allowedTools" not in argv


def test_tooled_role_gets_both_flags():
    """--tools grants existence; --allowedTools grants permission. Both needed."""
    argv = ChatClaudeCLI(role="researcher", tools=("WebSearch",))._argv("sys", None)
    assert argv[argv.index("--tools") + 1] == "WebSearch"
    assert argv[argv.index("--allowedTools") + 1] == "WebSearch"
    assert "dontAsk" in argv


def test_workers_never_inherit_user_mcp_servers():
    assert "--strict-mcp-config" in ChatClaudeCLI(role="x")._argv("sys", None)


def test_system_prompt_replaces_rather_than_appends():
    """--append-system-prompt costs ~5x by keeping the full harness prompt."""
    argv = ChatClaudeCLI(role="x")._argv("my system prompt", None)
    assert "--append-system-prompt" not in argv
    assert argv[argv.index("--system-prompt") + 1] == "my system prompt"


def test_bare_is_never_passed():
    """--bare forces API-key-only auth and ignores the OAuth session."""
    assert "--bare" not in ChatClaudeCLI(role="x")._argv("s", None)


def test_json_schema_is_serialised_into_argv():
    argv = ChatClaudeCLI(role="x")._argv("s", {"type": "object"})
    assert json.loads(argv[argv.index("--json-schema") + 1]) == {"type": "object"}


# --- human-in-the-loop gate ----------------------------------------------


def _resume_with(answer, outline):
    """Run approve_outline as if a human resumed the graph with `answer`.

    `interrupt()` needs a live LangGraph execution context, so it is patched to
    return the resume payload directly -- which is exactly what it does when a
    real run resumes. That keeps the three payload branches testable without
    standing up a checkpointer and paying for a planner call.
    """
    import reportteam.graphs.phase4_persist as phase4

    state = initial_state("topic")
    state["outline"] = outline

    original = phase4.interrupt
    phase4.interrupt = lambda payload: answer
    try:
        return phase4.approve_outline(state)
    finally:
        phase4.interrupt = original


def test_approving_leaves_the_outline_untouched():
    outline = _outline("q1", "q2")
    assert _resume_with(True, outline) == {}


def test_editing_sections_replaces_only_what_was_sent():
    outline = _outline("q1", "q2")
    result = _resume_with({"sections": ["new one", "new two"]}, outline)
    assert result["outline"].sections == ["new one", "new two"]
    # Everything the human did not touch survives.
    assert result["outline"].title == outline.title
    assert result["outline"].sub_questions == outline.sub_questions


def test_rejecting_discards_the_outline_so_the_planner_reruns():
    """A rejected plan must not be researched anyway."""
    outline = _outline("q1", "q2")
    result = _resume_with({"feedback": "too UK-specific, widen it"}, outline)
    assert result["outline"] is None
    assert "too UK-specific, widen it" in result["revision_notes"]


def test_gate_is_a_noop_when_there_is_no_outline_yet():
    import reportteam.graphs.phase4_persist as phase4

    assert phase4.approve_outline(initial_state("topic")) == {}


# --- failure handling ----------------------------------------------------


def test_session_limit_raises_its_own_error_type():
    """A spent allowance must be distinguishable from an ordinary failure.

    An eval that treats it as one more failed row races through every
    remaining topic in seconds, scoring each empty report 1.00.
    """
    model = ChatClaudeCLI(role="x")
    envelope = json.dumps(
        {
            "is_error": True,
            "result": "You've hit your session limit · resets 9:30pm (Asia/Karachi)",
            "terminal_reason": "api_error",
        }
    )
    with pytest.raises(ClaudeSessionLimitError, match="session limit"):
        model._parse(envelope, "")


def test_budget_exhausted_names_the_setting_to_change():
    model = ChatClaudeCLI(role="critic", max_budget_usd=0.5)
    envelope = json.dumps(
        {"is_error": True, "result": "", "terminal_reason": "budget_exhausted"}
    )
    with pytest.raises(ClaudeCLIError, match="max_budget_usd"):
        model._parse(envelope, "")


def test_not_logged_in_mentions_bare():
    model = ChatClaudeCLI(role="x")
    envelope = json.dumps(
        {"is_error": True, "result": "Not logged in · Please run /login"}
    )
    with pytest.raises(ClaudeCLIError, match="--bare"):
        model._parse(envelope, "")


def test_failed_rows_are_never_cached(tmp_path, monkeypatch):
    """Caching a failure bakes an outage into the results table forever."""
    import eval.run_eval as run_eval

    monkeypatch.setattr(run_eval, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(run_eval, "_cache_path", lambda t, s: tmp_path / f"{s}.json")
    monkeypatch.setattr(
        run_eval,
        "run_baseline",
        lambda topic, settings: {
            "system": "baseline", "topic": topic, "report": "",
            "cost_usd": 0.0, "wall_s": 1.0, "web_searches": 0,
            "route": ["baseline"], "finished": False, "error": "boom",
        },
    )
    monkeypatch.setattr(run_eval, "judge", lambda topic, report, settings: _StubScores())

    row = run_eval.evaluate("t", "easy_factual", "baseline", Settings.load(), False)
    assert row["error"] == "boom"
    assert list(tmp_path.glob("*.json")) == []


class _StubScores:
    coverage = citation_integrity = structure = specificity = usefulness = 1
    notable_weakness = "empty"
    mean = 1.0

    def model_dump(self) -> dict:
        return {
            "coverage": 1, "citation_integrity": 1, "structure": 1,
            "specificity": 1, "usefulness": 1, "notable_weakness": "empty",
        }


# --- config --------------------------------------------------------------


def test_role_models_are_overridable_by_env(monkeypatch):
    monkeypatch.setenv("REPORTTEAM_MODEL_WRITER", "opus")
    settings = Settings.load()
    assert settings.role("writer").model == "opus"
    # Overriding one role must not disturb the others.
    assert settings.role("supervisor").model == DEFAULT_ROLES["supervisor"].model


def test_unknown_role_names_the_known_ones():
    with pytest.raises(KeyError, match="known roles"):
        Settings.load().role("nope")


def test_only_research_roles_hold_tools():
    """A supervisor or critic with web access would break the guardrail."""
    for name in ("supervisor", "planner", "writer", "critic", "judge"):
        assert DEFAULT_ROLES[name].tools == ()
    assert DEFAULT_ROLES["researcher"].tools == ("WebSearch",)
