"""The model backend: every worker is a `claude` CLI process.

This project deliberately does not call the Anthropic API. Each specialist runs
as `claude -p --output-format json` against the machine's logged-in session,
which buys four things the API path does not give for free:

* `--json-schema` returns validated, already-parsed structured output, so the
  planner's outline and the critic's verdict need no JSON-repair fallback.
* `--tools ""` makes "the supervisor has no tools" a property of the process,
  not a hope about the prompt.
* The JSON envelope reports cost, token usage and latency per call, which is
  the whole of the Phase 6 measurement story with no price table to maintain.
* Web search is built in, so there is no second API key to provision.

It is wrapped as a LangChain `BaseChatModel` rather than called inline from the
graph nodes so that LangSmith traces every call automatically and
`with_structured_output()` keeps its usual shape.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any, Sequence, TypeVar

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class ClaudeCLIError(RuntimeError):
    """The CLI could not produce a usable answer.

    Always carries the fix in the message, not just the symptom -- the failure
    modes here are nearly all configuration (not logged in, tool not allowed,
    budget too low) and each has an obvious remedy worth stating.
    """


def _find_claude() -> list[str]:
    """Locate the CLI, preferring the native executable over the npm shims.

    `shutil.which` on Windows finds `claude.cmd`, and Windows cannot hand a
    `.cmd` to CreateProcess directly -- you would have to go through
    `cmd.exe /c`, which then re-parses our JSON schema argument and mangles the
    quoting. The npm shim just execs a real `claude.exe`; targeting that
    directly sidesteps the whole problem.
    """
    override = os.environ.get("REPORTTEAM_CLAUDE_BIN", "").strip()
    if override:
        return [override]

    found = shutil.which("claude")
    if found:
        shim = Path(found)
        native = (
            shim.parent
            / "node_modules"
            / "@anthropic-ai"
            / "claude-code"
            / "bin"
            / ("claude.exe" if os.name == "nt" else "claude")
        )
        if native.is_file():
            return [str(native)]
        if os.name == "nt" and shim.suffix.lower() in {".cmd", ".bat"}:
            return ["cmd.exe", "/c", str(shim)]
        return [str(shim)]

    raise ClaudeCLIError(
        "the `claude` CLI is not on PATH. Install it with "
        "`npm install -g @anthropic-ai/claude-code`, or set REPORTTEAM_CLAUDE_BIN "
        "to its full path."
    )


def _text(message: BaseMessage) -> str:
    """Message text, tolerating the langchain-core 1.x property/method change.

    `.text` was a method and became a property; accepting either keeps this
    working across the versions a reader might have installed.
    """
    raw = getattr(message, "text", None)
    # Order matters: in langchain-core 1.x `.text` is a str subclass that is
    # still callable for back-compat, and calling it emits a deprecation
    # warning. Test for str first so the modern path wins.
    if isinstance(raw, str):
        return raw
    if callable(raw):
        return raw()
    return str(message.content)


def _split_prompt(messages: Sequence[BaseMessage]) -> tuple[str, str]:
    """Flatten a LangChain message list into (system prompt, user prompt).

    Every node in this project makes a single-shot call -- the graph, not the
    message list, is what carries state between turns -- so this is a flatten
    rather than a real transcript encoder. Non-system messages are joined in
    order and role-labelled so a multi-message prompt still reads correctly.
    """
    system_parts: list[str] = []
    body_parts: list[str] = []

    for message in messages:
        text = _text(message)
        if isinstance(message, SystemMessage):
            system_parts.append(text)
        elif isinstance(message, ToolMessage):
            # Emulated tool calling (see `bind_tools`) means results come back
            # as plain prose rather than through a real tool_result block, so
            # label them clearly enough that the model treats them as fact.
            body_parts.append(f"[result of tool `{message.name}`]\n{text}")
        elif isinstance(message, AIMessage):
            calls = getattr(message, "tool_calls", None) or []
            if calls:
                rendered = ", ".join(
                    f"{c.get('name')}({json.dumps(c.get('args', {}))})" for c in calls
                )
                body_parts.append(f"[you previously called]\n{rendered}")
            if text:
                body_parts.append(f"[previous assistant output]\n{text}")
        else:
            body_parts.append(text)

    return "\n\n".join(system_parts).strip(), "\n\n".join(body_parts).strip()


def _tool_catalogue(tool_schemas: list[dict]) -> str:
    """Describe the bound tools in the system prompt.

    With a real tool-calling API the tool list travels in its own request field.
    Here it has to be prose, because the only structured channel the CLI exposes
    is the output schema.
    """
    lines = ["You have access to these tools:"]
    for schema in tool_schemas:
        fn = schema.get("function", schema)
        lines.append(
            f"- {fn.get('name')}: {fn.get('description', '').strip()}\n"
            f"  arguments JSON Schema: {json.dumps(fn.get('parameters', {}))}"
        )
    lines.append(
        "\nTo use a tool, set action='tool_call' with tool_name and "
        "tool_arguments (a JSON object encoded as a string). To answer the user "
        "directly instead, set action='final_answer' and fill in final_answer."
    )
    return "\n".join(lines)


def _action_schema(tool_names: list[str]) -> dict:
    """The envelope that stands in for a tool-call response.

    `tool_arguments` is a *string* holding JSON rather than a nested object:
    an open-ended object cannot be expressed in a strict schema (every property
    would have to be declared up front), so the arguments are carried as text
    and parsed on this side.
    """
    return {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["tool_call", "final_answer"]},
            "tool_name": {"type": "string", "enum": tool_names},
            "tool_arguments": {
                "type": "string",
                "description": "A JSON object, encoded as a string.",
            },
            "final_answer": {"type": "string"},
        },
        "required": ["action"],
    }


class ChatClaudeCLI(BaseChatModel):
    """A chat model whose inference happens in a `claude -p` subprocess."""

    role: str = "worker"
    model: str = "haiku"
    # Empty means `--tools ""`: the process gets no tools at all.
    tools: tuple[str, ...] = ()
    max_budget_usd: float = 0.75
    timeout_seconds: int = 300
    # Workers run from a scratch directory so the *project's* own CLAUDE.md,
    # skills and hooks never leak into a worker's context and quietly change
    # its behaviour. A research agent that has read this repo's instructions is
    # not the agent we think we are evaluating.
    workdir: Path | None = None

    @property
    def _llm_type(self) -> str:
        return "claude-cli"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        # Shows up in LangSmith, so make the interesting knobs visible.
        return {
            "role": self.role,
            "model": self.model,
            "tools": list(self.tools),
            "max_budget_usd": self.max_budget_usd,
        }

    # -- argv -------------------------------------------------------------

    def _argv(self, system_prompt: str, json_schema: dict | None) -> list[str]:
        argv = [
            *_find_claude(),
            "-p",
            "--model",
            self.model,
            "--output-format",
            "json",
            # Replace the default system prompt rather than appending to it.
            # Appending keeps Claude Code's full harness prompt -- measured at
            # 7,375 cache-creation tokens on a two-line request, roughly 5x the
            # cost of the same call with a replaced prompt.
            "--system-prompt",
            system_prompt or "You are a helpful assistant.",
            # A runaway worker stops here instead of on a billing alert.
            "--max-budget-usd",
            str(self.max_budget_usd),
            # These runs are driven by the graph and replayed from the
            # checkpointer, never resumed through the CLI's own session store.
            "--no-session-persistence",
            "--disable-slash-commands",
            # Without this a worker inherits the *user's* MCP servers. A
            # researcher run on this machine reached for
            # `mcp__playwright__browser_navigate` — a browser it was never
            # granted, from a server this project knows nothing about. That is
            # a reproducibility hole (the team behaves differently per machine)
            # and a cost one (denied calls still burn turns).
            "--strict-mcp-config",
        ]

        if self.tools:
            # Both flags are needed. `--tools` decides which tools exist;
            # `--allowedTools` decides which may run without a prompt. Give
            # only the first and the worker burns turns explaining that it is
            # not allowed to search -- a failure that cost $0.43 to discover.
            argv += ["--tools", ",".join(self.tools)]
            argv += ["--allowedTools", ",".join(self.tools)]
            argv += ["--permission-mode", "dontAsk"]
        else:
            argv += ["--tools", ""]

        if json_schema is not None:
            argv += ["--json-schema", json.dumps(json_schema)]

        return argv

    def _prepare(self, messages: list[BaseMessage], kwargs: dict) -> tuple[list[str], str, str]:
        """Turn a call into (argv, stdin prompt, response mode).

        Shared by the sync and async paths so the two can never drift.
        """
        system_prompt, user_prompt = _split_prompt(messages)
        tool_schemas = kwargs.get("tool_schemas")
        json_schema = kwargs.get("json_schema")

        if tool_schemas and json_schema is None:
            names = [s.get("function", s).get("name") for s in tool_schemas]
            system_prompt = f"{system_prompt}\n\n{_tool_catalogue(tool_schemas)}".strip()
            json_schema = _action_schema(names)
            mode = "tools"
        else:
            mode = "schema" if json_schema is not None else "plain"

        return self._argv(system_prompt, json_schema), user_prompt, mode

    def _finalize(self, message: AIMessage, mode: str) -> AIMessage:
        """Translate an emulated tool-call envelope back into a real AIMessage."""
        if mode != "tools":
            return message

        payload = message.additional_kwargs.get("structured_output") or {}
        if payload.get("action") == "tool_call" and payload.get("tool_name"):
            try:
                args = json.loads(payload.get("tool_arguments") or "{}")
            except json.JSONDecodeError:
                # The arguments are the one part of the envelope the schema
                # cannot police, since they travel as a string. A malformed
                # object is worth surfacing rather than silently calling the
                # tool with no arguments.
                raise ClaudeCLIError(
                    f"{self.role}: tool_arguments for {payload['tool_name']!r} was "
                    f"not valid JSON: {payload.get('tool_arguments')!r}"
                ) from None
            if not isinstance(args, dict):
                args = {"__value__": args}
            return message.model_copy(
                update={
                    "content": "",
                    "tool_calls": [
                        {
                            "name": payload["tool_name"],
                            "args": args,
                            "id": f"call_{uuid.uuid4().hex[:12]}",
                            "type": "tool_call",
                        }
                    ],
                }
            )

        return message.model_copy(
            update={"content": payload.get("final_answer") or message.content}
        )

    def _cwd(self) -> str:
        target = self.workdir or (Path(__file__).resolve().parents[2] / "out" / "workdir")
        target.mkdir(parents=True, exist_ok=True)
        return str(target)

    # -- envelope ---------------------------------------------------------

    def _parse(self, stdout: str, stderr: str) -> AIMessage:
        if not stdout.strip():
            raise ClaudeCLIError(
                f"the CLI returned no output. stderr: {stderr.strip()[:400] or '(empty)'}"
            )

        try:
            envelope = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise ClaudeCLIError(
                f"could not parse the CLI's JSON envelope: {exc}. "
                f"First 400 chars of stdout: {stdout[:400]!r}"
            ) from exc

        result_text = str(envelope.get("result") or "")

        if envelope.get("is_error"):
            if "not logged in" in result_text.lower():
                raise ClaudeCLIError(
                    "the CLI is not logged in. Run `claude` once interactively and "
                    "sign in. (If you passed --bare anywhere, remove it: it forces "
                    "API-key-only auth and never reads the OAuth session.)"
                )
            reason = envelope.get("terminal_reason") or envelope.get("subtype")
            if reason == "budget_exhausted":
                # Worth its own message: the fix is a config value, and the
                # roles that read the whole research set (writer, critic) need
                # a much larger ceiling than the routing roles.
                raise ClaudeCLIError(
                    f"{self.role}: hit its --max-budget-usd ceiling of "
                    f"${self.max_budget_usd}. Raise `max_budget_usd` for this "
                    f"role in config.DEFAULT_ROLES -- roles that read the full "
                    f"research notes need more headroom than routing roles."
                )
            raise ClaudeCLIError(
                f"the CLI reported an error [{reason}]: {result_text[:400]}"
            )

        denials = envelope.get("permission_denials") or []
        if denials:
            # Not fatal -- the worker may have completed anyway -- but it means a
            # tool it wanted was blocked, which usually shows up later as a
            # thin, sourceless draft. Surface it now.
            log.warning(
                "%s: %d tool permission denial(s): %s",
                self.role,
                len(denials),
                json.dumps(denials)[:300],
            )

        # `usage` reports only the final iteration -- a 10k-token prompt can show
        # `input_tokens: 9`. `modelUsage` is the per-model total for the whole
        # call, so cost and token accounting must come from there.
        model_usage: dict[str, dict] = envelope.get("modelUsage") or {}
        input_tokens = sum(int(m.get("inputTokens", 0)) for m in model_usage.values())
        output_tokens = sum(int(m.get("outputTokens", 0)) for m in model_usage.values())
        cache_read = sum(
            int(m.get("cacheReadInputTokens", 0)) for m in model_usage.values()
        )
        web_searches = sum(
            int(m.get("webSearchRequests", 0)) for m in model_usage.values()
        )

        return AIMessage(
            content=result_text,
            additional_kwargs={"structured_output": envelope.get("structured_output")},
            usage_metadata={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "input_token_details": {"cache_read": cache_read},
            },
            response_metadata={
                "role": self.role,
                "model": self.model,
                "models_used": sorted(model_usage),
                # Phase 6 reads these three straight off the envelope.
                "cost_usd": float(envelope.get("total_cost_usd") or 0.0),
                "duration_ms": int(envelope.get("duration_ms") or 0),
                "duration_api_ms": int(envelope.get("duration_api_ms") or 0),
                "num_turns": int(envelope.get("num_turns") or 0),
                "web_search_requests": web_searches,
                "permission_denials": denials,
                "session_id": envelope.get("session_id"),
                "terminal_reason": envelope.get("terminal_reason"),
            },
        )

    # -- sync -------------------------------------------------------------

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        argv, user_prompt, mode = self._prepare(messages, kwargs)

        try:
            proc = subprocess.run(
                argv,
                # The prompt goes in on stdin, never as an argv element: Windows
                # caps a command line at ~32k characters and the writer's prompt
                # carries every research note.
                input=user_prompt,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_seconds,
                cwd=self._cwd(),
            )
        except subprocess.TimeoutExpired as exc:
            raise ClaudeCLIError(
                f"{self.role}: the CLI did not finish within "
                f"{self.timeout_seconds}s. Raise REPORTTEAM_CLI_TIMEOUT if this "
                f"role legitimately needs longer."
            ) from exc

        message = self._finalize(self._parse(proc.stdout, proc.stderr), mode)
        return ChatResult(generations=[ChatGeneration(message=message)])

    # -- async (Phase 5 fan-out) -----------------------------------------

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        argv, user_prompt, mode = self._prepare(messages, kwargs)

        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self._cwd(),
        )
        try:
            raw_out, raw_err = await asyncio.wait_for(
                proc.communicate(user_prompt.encode("utf-8")),
                timeout=self.timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            proc.kill()
            await proc.wait()
            raise ClaudeCLIError(
                f"{self.role}: the CLI did not finish within {self.timeout_seconds}s."
            ) from exc

        message = self._finalize(
            self._parse(
                raw_out.decode("utf-8", "replace"), raw_err.decode("utf-8", "replace")
            ),
            mode,
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    # -- structured output ------------------------------------------------

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Runnable:
        """Emulate tool calling on a backend that has no tool-calling API.

        The CLI exposes exactly one structured channel — the output schema — so
        a tool call is modelled as a *choice* the model makes inside that
        schema: either `final_answer`, or `tool_call` naming one of the bound
        tools. `_finalize` turns the latter back into a genuine `AIMessage`
        with `.tool_calls`, which is all any LangChain/LangGraph consumer
        actually inspects.

        This exists so the prebuilt helpers the brief names — `create_supervisor`
        and `create_agent`, both of which require a tool-calling model — work
        against this backend. The hand-built graph from Phase 1 onward does not
        need it: it routes on a state field instead, which is cheaper and far
        easier to debug.
        """
        schemas = [convert_to_openai_tool(tool) for tool in tools]
        if not schemas:
            return self
        return self.bind(tool_schemas=schemas, **kwargs)

    def with_structured_output(
        self, schema: type[T], *, include_raw: bool = False, **kwargs: Any
    ) -> Runnable:
        """Bind a Pydantic schema to `--json-schema`.

        The CLI validates the model's output against the schema itself and
        returns it pre-parsed in `structured_output`, so there is nothing to
        repair here -- a marked contrast with providers whose tool-calling is a
        text convention the model can simply break.
        """
        json_schema = schema.model_json_schema()

        def _parse(message: AIMessage) -> Any:
            payload = message.additional_kwargs.get("structured_output")
            if payload is None:
                # The CLI only omits this when the model ended its turn without
                # emitting the structured block at all.
                raise ClaudeCLIError(
                    f"{self.role}: expected {schema.__name__} but the CLI returned no "
                    f"structured output. Raw text began: {_text(message)[:200]!r}"
                )
            parsed = schema.model_validate(payload)
            if include_raw:
                return {"raw": message, "parsed": parsed, "parsing_error": None}
            return parsed

        return self.bind(json_schema=json_schema) | RunnableLambda(_parse)


def for_role(role: str, settings: Any = None) -> ChatClaudeCLI:
    """Build the model for one specialist from settings."""
    from .config import Settings

    settings = settings or Settings.load()
    cfg = settings.role(role)
    return ChatClaudeCLI(
        role=role,
        model=cfg.model,
        tools=cfg.tools,
        max_budget_usd=cfg.max_budget_usd,
        timeout_seconds=settings.cli_timeout_seconds,
    )
