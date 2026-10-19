"""Actions only the *user* can take (via the UI). They are deliberately not exposed as model tools,
so cleaning approvals can never be granted by the LLM itself."""
from __future__ import annotations

from .cleaning import describe
from .context import AgentContext
from .workflow import sync


def record_cleaning_decision(ctx: AgentContext, issue_id: str, approve: bool, alternative: int | None = None) -> dict:
    """Approve the recommended treatment (or a numbered alternative), or reject it (keep data as is)."""
    m = ctx.memory
    issue = next((i for i in m.cleaning_issues if i["id"] == issue_id), None)
    if issue is None or not issue.get("requires_approval"):
        raise ValueError("Unknown or informational issue.")
    if approve:
        choice = issue["recommended"] if alternative is None else issue["alternatives"][alternative]
        issue["decision"], issue["status"] = choice["op"], "approved"
    else:
        issue["decision"], issue["status"] = {"op": "keep", "column": issue.get("column"), "params": {}}, "rejected"
    m.cleaning_applied = False  # decisions changed: they must be (re)applied
    m.log_event("user_decision", {"issue": issue["title"], "decision": describe(issue["decision"]), "approved": approve})
    sync(m)
    ctx.save()
    return issue


def decisions_summary(ctx: AgentContext) -> str:
    parts = []
    for i in ctx.memory.cleaning_issues:
        if i.get("requires_approval") and i["status"] in ("approved", "rejected"):
            verb = "approved" if i["status"] == "approved" else "kept as is"
            parts.append(f"{i['title']}: {verb} ({describe(i['decision'])})")
    return "; ".join(parts)


def all_decided(ctx: AgentContext) -> bool:
    return not any(i.get("requires_approval") and i["status"] == "pending" for i in ctx.memory.cleaning_issues)


def set_assistant(ctx: AgentContext, provider: str | None, model: str | None) -> None:
    """The user picks who assists: None = offline guided assistant, otherwise an AI provider + model.
    Picking an AI provider is the user's consent to share column summaries and a few example rows with it."""
    m = ctx.memory
    m.llm_choice = {"provider": provider, "model": model} if provider else {}
    m.consent = bool(provider)
    m.mode = "llm" if provider else "guided"
    m.log_event("assistant", {"provider": provider, "model": model})
    ctx.save()


def set_consent(ctx: AgentContext, consent: bool) -> None:
    ctx.memory.consent = consent
    ctx.memory.mode = "llm" if consent else "guided"
    ctx.memory.log_event("consent", {"consent": consent})
    ctx.save()
