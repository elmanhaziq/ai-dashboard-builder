"""LLM agent loop with controlled tool calling.

* The model only sees tools allowed at the current workflow stage (later steps are impossible until the
  earlier ones are complete).
* Each user turn allows at most ``MAX_TOOL_CALLS`` tool calls; asking the user a question or requesting
  cleaning approval ends the turn.
* The transcript sent to the model is append-only (thinking blocks are passed back unchanged) and saved
  after every step, so an interrupted turn is repaired on the next message instead of corrupting history.
* If the provider is unavailable, the turn is handled by the offline guided agent instead.
"""
from __future__ import annotations

import json

from . import guided
from .actions import all_decided, decisions_summary
from .context import AgentContext
from .llm import LLMClient, LLMUnavailable
from .tools import TOOLS, available_tools, run_tool
from .workflow import STEP_LABELS, Stage, step_index, unmet_requirement

MAX_TOOL_CALLS = 12
MAX_TRANSCRIPT_MESSAGES = 120

SYSTEM_PROMPT = """You are a friendly, careful data analyst working with a person who may know nothing about Excel, \
statistics or dashboards. You help them understand a spreadsheet they uploaded and build a useful dashboard.

How you work:
- The app runs a fixed workflow: understand the data → learn the user's goal → clarify unclear columns → review data \
quality → plan the dashboard → validate → build → explore. Tools for later steps only become available once earlier \
steps are done; the <app_state> block in each user message tells you the current step and what is still missing.
- Use tools to look at the data. You receive profiles and at most a few example rows, never the whole file.
- Whenever you need something from the user, call ask_user_clarification (never ask only in plain text — the tool \
shows the question with clickable answers). Ask ONE question at a time, about ONE column, most important first, and \
offer short suggested answers. \
Don't ask about columns whose meaning is clear from strong evidence unless the meaning changes calculations. Never ask \
something the user already answered — the tools tell you when that happened.
- Never assume what an unclear column means (for example UAG_2, REV_NET, QTY_SLD). Ask, then save the user's answer \
with save_column_interpretation (based_on_user_answer=true). Distinguish meaning, unit, whether it should be added up \
or averaged, and whether it is a good headline number. If the user doesn't know, set business_role='ignore'.
- For data quality: call propose_cleaning_operations, explain the important issues simply (what you found, why it \
matters, what you recommend, the alternatives), then call request_cleaning_approval. Only the user can approve \
changes; after they decide, call apply_approved_transformations. Never fill missing numbers with zero and never delete \
records just because they look unusual.
- To build the dashboard: suggest_dashboard_plan → adapt it to the user's goal and confirmed meanings → \
create_dashboard_plan → validate_dashboard_plan (fix any errors) → execute_dashboard_plan.
- Numbers: never calculate, estimate or invent numbers yourself. Only quote numbers that appear in tool results \
(for questions about the data use answer_data_question). If a tool can't answer, say so and suggest a rephrasing.
- Language: short, warm, plain English (or the user's language). Avoid jargon such as "cardinality", "null", \
"aggregation", "schema" — say "too many different values", "missing", "added up / averaged", "column".
- After the dashboard exists, help the user explore: answer questions with answer_data_question and change the \
dashboard with update_dashboard when they ask.
"""


def _state_block(ctx: AgentContext) -> str:
    m = ctx.memory
    stage = Stage(m.stage)
    nxt = next((s for s in list(Stage)[list(Stage).index(stage) + 1:]), None)
    state = {
        "current_step": STEP_LABELS[step_index(stage)][0], "stage": stage.value,
        "blocked_because": unmet_requirement(m, nxt) if nxt else None,
        "objective": m.objective or None,
        "pending_question": (m.pending() or {}).get("question"),
        "answered_questions": [{"column": q["column"], "topic": q["kind"], "answer": q["answer"]}
                               for q in m.questions if q["status"] == "answered"][-12:],
        "confirmed_columns": {c: {k: it[k] for k in ("meaning", "unit", "business_role", "aggregation")}
                              for c, it in m.interpretations.items()},
        "unclear_columns_remaining": m.unresolved_columns(),
        "cleaning_decisions": decisions_summary(ctx) or None,
        "dashboard_built": bool(m.dashboard_id),
        "sample_rows_allowed": ctx.sample_rows_allowed,
    }
    return "<app_state>\n" + json.dumps(state, default=str) + "\n</app_state>"


class Orchestrator:
    def __init__(self, ctx: AgentContext, llm: LLMClient, max_tool_calls: int = MAX_TOOL_CALLS):
        self.ctx, self.llm, self.max_tool_calls = ctx, llm, max_tool_calls

    @property
    def transcript(self) -> list[dict]:
        return self.ctx.memory.llm.setdefault("transcript", [])

    # ------------------------------------------------------------------ public API
    def handle(self, user_text: str | None = None, event: str | None = None) -> None:
        m = self.ctx.memory
        self._repair_interrupted_turn()
        parts = []
        if user_text:
            m.add_message("user", user_text)
            q = m.answer_pending(user_text)
            parts.append(user_text if not q else f"(Answer to your question “{q['question']}”): {user_text}")
        if event == "start":
            parts.append(f"(App event) The user just uploaded “{m.dataset_name}”. Inspect it, briefly say what you found "
                         "in plain language, then ask what they want to understand.")
        if event == "cleaning_decided" and all_decided(self.ctx):
            note = f"(App event) The user reviewed the clean-up suggestions: {decisions_summary(self.ctx) or 'no changes'}."
            m.add_message("user", note, kind="event")
            parts.append(note + " Apply the approved changes and continue.")
        if not parts:
            return
        self._maybe_start_new_episode()
        self._append({"role": "user", "content": [{"type": "text", "text": "\n\n".join(parts) + "\n\n" + _state_block(self.ctx)}]})
        try:
            self._run()
        except LLMUnavailable as exc:
            m.add_message("assistant", f"I couldn't reach the AI service ({exc}). I'll continue in guided mode for now.", kind="notice")
            m.log_event("llm_unavailable", {"error": str(exc)}, ok=False)
            self.ctx.save()
            guided.handle(self.ctx, None, event)

    # ------------------------------------------------------------------ loop
    def _run(self) -> None:
        m = self.ctx.memory
        calls = 0
        while True:
            tools = [t.api_definition() for t in available_tools(m.stage)]
            limit_hit = calls >= self.max_tool_calls
            reply = self.llm.create(SYSTEM_PROMPT, self.transcript, tools, allow_tools=not limit_hit)
            self._append({"role": "assistant", "content": reply.content})
            texts = [b["text"] for b in reply.content if b.get("type") == "text" and b.get("text", "").strip()]
            tool_uses = [b for b in reply.content if b.get("type") == "tool_use"]
            if texts:
                m.add_message("assistant", "\n\n".join(texts))
            if reply.stop_reason == "refusal":
                m.add_message("assistant", "I can't help with that request. Let's continue with the analysis.", kind="notice")
                self.ctx.save()
                return
            if not tool_uses:
                self.ctx.save()
                return
            results, stop = [], False
            for tu in tool_uses:
                if calls >= self.max_tool_calls:
                    result, is_error = {"error": "Tool limit for this turn reached. Summarise progress for the user and ask how to continue."}, True
                    stop = stop or False
                else:
                    result, is_error = run_tool(self.ctx, tu["name"], tu.get("input") or {})
                    calls += 1
                    if TOOLS.get(tu["name"]) and TOOLS[tu["name"]].interactive and result.get("waiting_for_user"):
                        stop = True
                results.append({"type": "tool_result", "tool_use_id": tu["id"], "content": json.dumps(result, default=str),
                                **({"is_error": True} if is_error else {})})
            self._append({"role": "user", "content": results})
            self.ctx.save()
            if stop:
                return
            if calls >= self.max_tool_calls and limit_hit:
                return

    # ------------------------------------------------------------------ transcript management
    def _append(self, message: dict) -> None:
        self.transcript.append(message)
        self.ctx.save()

    def _repair_interrupted_turn(self) -> None:
        """If the app stopped between a tool request and its result, close the open tool calls."""
        t = self.transcript
        if not t or t[-1]["role"] != "assistant":
            return
        open_ids = [b["id"] for b in t[-1]["content"] if isinstance(b, dict) and b.get("type") == "tool_use"]
        if open_ids:
            t.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": i, "is_error": True,
                                                   "content": "Interrupted before this tool ran. Check the current app state and continue."}
                                                  for i in open_ids]})
            self.ctx.memory.log_event("repaired_interrupted_turn", {"tool_calls": len(open_ids)})
            self.ctx.save()

    def _maybe_start_new_episode(self) -> None:
        """Keep the request size bounded without editing history: start a fresh transcript with a summary."""
        m = self.ctx.memory
        if len(self.transcript) < MAX_TRANSCRIPT_MESSAGES:
            return
        recent = [f"{c['role']}: {c['text'][:300]}" for c in m.conversation[-10:] if c["role"] in ("user", "assistant")]
        summary = ("(Context from earlier in this analysis — the full history was archived.)\n"
                   f"Objective: {m.objective or 'not set'}\nRecent conversation:\n" + "\n".join(recent))
        m.llm["episode"] = m.llm.get("episode", 1) + 1
        m.llm["summary"] = summary
        m.llm["transcript"] = [{"role": "user", "content": [{"type": "text", "text": summary}]},
                               {"role": "assistant", "content": [{"type": "text", "text": "Understood — I have the context."}]}]


def run_turn(ctx: AgentContext, llm: LLMClient | None, user_text: str | None = None, event: str | None = None) -> str:
    """Entry point used by the UI. Returns which engine handled the turn ('llm' or 'guided')."""
    try:
        if llm is not None and ctx.memory.consent:
            Orchestrator(ctx, llm).handle(user_text, event)
            return "llm"
        if event == "start":
            guided.start(ctx)
        else:
            guided.handle(ctx, user_text, event)
        return "guided"
    finally:
        ctx.save()  # messages added after the last tool call must survive a reload
