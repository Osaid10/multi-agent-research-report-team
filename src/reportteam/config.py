"""Settings, loaded from `.env` / environment. No key ever lives in source.

Unlike a normal LangChain project there is no model API key here at all: every
worker runs through the `claude` CLI on the machine's logged-in session. What
this file configures instead is *how each role is allowed to run* — which model
alias, which tools it may touch, and how much it may spend before the CLI cuts
it off.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Repo root = three levels up from this file (src/reportteam/config.py)
ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader so the project runs even without python-dotenv.

    Existing environment variables always win, which is what you want when
    running in CI or with a value exported in the shell.
    """
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _as_bool(value: str, default: bool = False) -> bool:
    if not value:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class RoleConfig:
    """How one specialist is launched.

    `tools` is the whole security story for this project. The brief's first
    guardrail is "keep the supervisor dumb on tools" — give it only route and
    finish, and forbid tool use in its prompt. Because each worker is a separate
    CLI process, an empty `tools` is enforced by the process boundary rather
    than by asking the model nicely, so a supervisor *cannot* search the web
    even if a prompt injection tells it to.
    """

    model: str  # "haiku" | "sonnet" | "opus", or a full model id
    tools: tuple[str, ...] = ()
    # A per-call ceiling, not a per-run one. Sized generously enough that a
    # legitimate call never trips it, and tight enough that a runaway worker
    # stops before it costs real money.
    #
    # The spread is wide on purpose. Routing roles see a short status summary;
    # the writer and critic both read the entire research set, so their prompts
    # grow with the number of sub-questions. A $0.50 critic ceiling looked
    # ample until a five-sub-question topic blew straight through it.
    max_budget_usd: float = 0.75


# Models are assigned by role, not globally: routing and planning are cheap
# classification work that Haiku does well, while research, drafting and
# grading carry the report's quality and get Sonnet. This split is also the
# brief's Stretch Goal #2 (small/fast model in the supervisor and critic,
# stronger model in the writer) — keeping it configurable means the eval can
# re-run any mix by changing env vars alone.
DEFAULT_ROLES: dict[str, RoleConfig] = {
    "supervisor": RoleConfig(model="haiku", tools=(), max_budget_usd=0.25),
    "planner": RoleConfig(model="haiku", tools=(), max_budget_usd=0.25),
    "researcher": RoleConfig(model="haiku", tools=("WebSearch",), max_budget_usd=0.75),
    "writer": RoleConfig(model="sonnet", tools=(), max_budget_usd=2.00),
    "critic": RoleConfig(model="sonnet", tools=(), max_budget_usd=2.00),
    # The Phase 6 comparison point: one agent, one shot, same web access the
    # team gets, so the only variable is the orchestration.
    "baseline": RoleConfig(model="sonnet", tools=("WebSearch",), max_budget_usd=1.50),
    # Phase 6's LLM-as-judge. Deliberately not the same model as the critic —
    # a grader that shares the writer's blind spots inflates the scores.
    "judge": RoleConfig(model="sonnet", tools=(), max_budget_usd=1.00),
}


@dataclass(frozen=True)
class Settings:
    roles: dict[str, RoleConfig] = field(default_factory=lambda: dict(DEFAULT_ROLES))

    # -- loop guards ------------------------------------------------------
    # The brief asks for both: a domain-level cap on how many times the critic
    # may bounce a draft, and LangGraph's own recursion_limit as the backstop.
    # Hitting the recursion limit is almost always a routing bug — fix the
    # prompt, not the limit.
    max_revisions: int = 2
    # The brief suggests ~25, with the warning that hitting it is almost always
    # a routing loop to be fixed in the prompt rather than raised away. That
    # warning earned its place here: an early run spun
    # writer -> supervisor -> writer until it tripped, and the fix was to
    # invalidate the stale verdict, not to move the limit.
    #
    # 40 is still a real guard. The sequential graphs cost two supersteps per
    # sub-question (researcher, then supervisor), so a five-question topic with
    # two revisions legitimately reaches ~23 -- uncomfortably close to 25 for a
    # limit whose whole job is to distinguish "busy" from "broken". The
    # parallel team graph collapses all that research into one superstep and
    # never comes near it.
    recursion_limit: int = 40

    # -- execution --------------------------------------------------------
    # Phase 5 fans researchers out as concurrent subprocesses. Capped because
    # the ceiling here is the account's rate limit, not the local CPU.
    max_concurrent_researchers: int = 4
    cli_timeout_seconds: int = 300

    # -- paths ------------------------------------------------------------
    out_dir: Path = ROOT / "out"
    checkpoint_db: Path = ROOT / "out" / "checkpoints.sqlite"
    run_log: Path = ROOT / "out" / "runs.jsonl"

    @classmethod
    def load(cls, env_file: Path | None = None) -> "Settings":
        _load_dotenv(env_file or (ROOT / ".env"))

        roles = dict(DEFAULT_ROLES)
        # REPORTTEAM_MODEL_WRITER=opus overrides one role without touching the
        # rest — this is how the eval sweeps model mixes.
        for name, cfg in list(roles.items()):
            override = os.environ.get(f"REPORTTEAM_MODEL_{name.upper()}", "").strip()
            if override:
                roles[name] = RoleConfig(
                    model=override, tools=cfg.tools, max_budget_usd=cfg.max_budget_usd
                )

        return cls(
            roles=roles,
            max_revisions=int(os.environ.get("REPORTTEAM_MAX_REVISIONS", "2")),
            recursion_limit=int(os.environ.get("REPORTTEAM_RECURSION_LIMIT", "40")),
            max_concurrent_researchers=int(
                os.environ.get("REPORTTEAM_CONCURRENCY", "4")
            ),
            cli_timeout_seconds=int(os.environ.get("REPORTTEAM_CLI_TIMEOUT", "300")),
        )

    def role(self, name: str) -> RoleConfig:
        try:
            return self.roles[name]
        except KeyError:
            raise KeyError(
                f"unknown role {name!r}; known roles: {sorted(self.roles)}"
            ) from None
