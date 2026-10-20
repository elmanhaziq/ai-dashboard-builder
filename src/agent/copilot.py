"""The dashboard assistant ("Dash"): a chat that lives on the dashboard page.

It answers questions about the dashboard's data, suggests ideas (charts it drafts and validates, which the user adds
with one click), changes colours, and learns from the conversation (see ``knowledge``). Like the AI analyst:

* every number comes from pandas through tools — the model only decides what to calculate and explains it;
* chart ideas are validated against the data before the user sees them;
* without an AI provider (or if it fails) a rule-based assistant handles the same requests offline.

The class is independent of Streamlit: the UI passes the dashboard config, the rows currently shown and a plain dict
that holds the chat state, so everything here is testable.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from ..chart_generator import ChartConfigError, describe_chart, recommend_charts, validate_chart_spec
from ..data_profiler import DatasetProfile
from ..insights import generate_insights
from ..query_engine import answer_question
from .. import themes
from . import knowledge
from .llm import LLMClient, LLMUnavailable
from .planner import validate_plan
from .tools import (TOOLS, ToolError, _arr, _enum, _normalise_plan, _obj, _PLAN_CHART, B, S, t_get_insights,
                    t_query_data, validate_args)

MAX_TOOL_CALLS = 8
MAX_TRANSCRIPT = 60
MAX_MESSAGES = 80
MAX_SUGGESTIONS = 12

SYSTEM_PROMPT = """You are Dash, a friendly, curious data-analyst buddy who lives inside the user's dashboard. The user \
may not know statistics or Excel. You help them understand their data, spot what matters, and improve the dashboard.

How you think and work:
- Before answering, think about what the user really wants and which calculations would answer it. Then use tools; \
several calls are fine. Check the results make sense before you reply.
- query_data also does share of total, growth vs the previous period, two-way tables (pivot) and A-vs-B comparisons.
- Translate the user's words to real columns yourself ("sales" → the closest money column, "department" → the closest \
category). Say which columns you used. The <knowledge> block lists word meanings and facts learned earlier: use them.
- Numbers: never calculate, estimate or invent numbers. Only quote numbers from tool results. Answers use the rows \
currently shown on the dashboard (filters applied) unless the tool result says otherwise.
- Reply format: lead with the finding in one sentence, then 2–4 supporting numbers, then ONE idea — a chart worth \
adding, a filter to try, or a question to explore next. Keep it short, warm and in plain language (or the user's \
language). Avoid jargon such as "aggregation", "cardinality", "null".
- Ideas: when you suggest a chart, draft it with propose_chart (add_now=false) so the user can add it with one click. \
Use add_now=true only when the user explicitly asks you to add or create a chart. When asked for ideas, look at the \
dashboard (describe_dashboard) and the data (get_insights) first, then propose 2–3 charts that are NOT already on it, \
each with a one-line reason tied to what the data shows.
- Colours: use set_theme or set_chart_color when the user asks for different colours, a mood ("make it calm", "more \
corporate", "dark mode") or brand colours. Pick a preset when one fits; otherwise choose a readable palette.
- Learning: when the user corrects you, explains what a column or word means, or states a preference ("I prefer \
horizontal bars", "always answer in Malay"), call remember so you handle it better next time. Don't remember numbers \
or one-off questions.
- Only remove charts when the user clearly asks.
- Greetings and small talk: reply briefly and warmly, and offer two things you could do — no tool call needed.
"""

GUIDED_HELP = ("I can answer questions about this dashboard's data, suggest charts, and change the colours. "
               "Try “Which region has the highest sales?”, “Give me ideas”, or “Make it ocean blue”.")


class DashboardData:
    """The minimal context the shared query tools need (the dashboard's current rows and profile)."""

    def __init__(self, df: pd.DataFrame, profile: DatasetProfile):
        self.df, self.profile, self.memory = df, profile, None

    def working(self) -> tuple[pd.DataFrame, DatasetProfile]:
        return self.df, self.profile

    def column_name(self, name: str) -> str:
        return name

    def original_name(self, name: str) -> str:
        return name


def new_state() -> dict:
    return {"messages": [], "transcript": [], "suggestions": {}, "mood": "idle", "greeted": False}


class Copilot:
    def __init__(self, cfg: dict, df: pd.DataFrame, profile: DatasetProfile, state: dict, knowledge_dir: Path,
                 llm: LLMClient | None = None, total_rows: int | None = None):
        self.cfg, self.df, self.profile, self.state, self.kdir, self.llm = cfg, df, profile, state, Path(knowledge_dir), llm
        self.total_rows = total_rows if total_rows is not None else len(df)
        self.data = DashboardData(df, profile)
        self.changed = False
        self._turn_suggestions: list[str] = []
        for k, v in new_state().items():
            state.setdefault(k, v)

    # ------------------------------------------------------------------ public API
    @property
    def columns(self) -> list[str]:
        return list(self.profile.columns)

    def knowledge(self) -> dict:
        return knowledge.for_dataset(self.kdir, self.columns)

    def greet(self, ai: bool) -> None:
        """First message when the chat opens for this dashboard."""
        if self.state["greeted"]:
            return
        self.state["greeted"] = True
        seen = knowledge.touch(self.kdir, self.columns, self.cfg.get("dataset_name", ""))
        kn = self.knowledge()
        if kn["from_similar_file"]:
            knowledge.adopt_similar(self.kdir, self.columns, self.cfg.get("dataset_name", ""))
            hello = (f"Hi! This file looks like **{kn['similar_name'] or 'one I’ve seen before'}**, so I’ll reuse what I "
                     f"learned there ({len(kn['notes']) + len(kn['aliases'])} notes). ")
        elif seen > 1 and (kn["notes"] or kn["aliases"]):
            hello = f"Welcome back! I remember {len(kn['notes']) + len(kn['aliases'])} things about this data. "
        else:
            hello = "Hi, I’m Dash, your data buddy! "
        hello += ("Ask me anything about this dashboard, ask for ideas, or tell me to change the colours."
                  if ai else GUIDED_HELP)
        self._say(hello, mood="idle")

    def ask(self, text: str) -> str:
        """Handle one user message. Returns which engine answered ('llm' or 'guided')."""
        text = (text or "").strip()[:2000]
        if not text:
            return "none"
        self.state["messages"].append({"role": "user", "text": text})
        if self.llm is not None:
            try:
                self._llm_turn(text)
                return "llm"
            except LLMUnavailable as exc:
                self._say(f"I couldn't reach the AI service ({exc}). Here's what I can work out with the built-in rules:",
                          mood="idle", kind="notice")
        self._guided_turn(text)
        return "guided"

    def add_suggestion(self, sid: str) -> str:
        """The user clicked 'Add to dashboard' on a drafted chart."""
        sug = self.state["suggestions"].get(sid)
        if not sug or sug["status"] != "open":
            return "That idea is no longer available."
        try:
            spec = validate_chart_spec(sug["spec"], self.profile)
        except ChartConfigError as exc:
            sug["status"] = "failed"
            return f"That chart no longer fits the data: {exc}"
        spec["description"] = spec.get("description") or describe_chart(spec, self.profile)
        self.cfg["charts"].append(spec)
        sug["status"] = "added"
        self.changed = True
        self.state["mood"] = "celebrate"
        return f"Added “{spec['title']}” to the dashboard."

    def dismiss_suggestion(self, sid: str) -> None:
        if sid in self.state["suggestions"]:
            self.state["suggestions"][sid]["status"] = "dismissed"

    # ------------------------------------------------------------------ shared helpers
    def _say(self, text: str, mood: str = "aha", steps: list[str] | None = None, kind: str = "answer",
             suggestions: list[str] | None = None, chart: dict | None = None) -> None:
        self.state["messages"].append({"role": "assistant", "text": text, "steps": steps or [], "kind": kind,
                                       "suggestions": suggestions or [], "chart": chart})
        del self.state["messages"][:-MAX_MESSAGES]
        self.state["mood"] = mood

    def _context_block(self) -> str:
        kn = self.knowledge()
        filters = self.cfg.get("filters") or {}
        active = {**(filters.get("categorical") or {}), **{k: v for k, v in (filters.get("date") or {}).items()}}
        ctx = {
            "dashboard": self.cfg.get("name"),
            "rows_shown": len(self.df), "rows_total": self.total_rows,
            "active_filters": active or None,
            "columns": {c: {"type": cp.role, "usually": cp.aggregation} for c, cp in list(self.profile.columns.items())[:60]},
            "column_notes": self.cfg.get("column_notes") or None,
        }
        learned = {"word_meanings": kn["aliases"] or None, "facts_about_this_data": kn["notes"] or None,
                   "user_preferences": kn["general"] or None}
        return ("<dashboard_state>\n" + json.dumps(ctx, default=str) + "\n</dashboard_state>\n<knowledge>\n"
                + json.dumps(learned, ensure_ascii=False) + "\n</knowledge>")

    def _add_suggestion(self, spec: dict, reason: str) -> str:
        sid = spec["id"]
        self.state["suggestions"][sid] = {"spec": spec, "reason": reason, "status": "open"}
        open_ids = [k for k, v in self.state["suggestions"].items() if v["status"] == "open"]
        for old in open_ids[:-MAX_SUGGESTIONS]:
            self.state["suggestions"].pop(old, None)
        return sid

    def _validated_chart(self, chart: dict) -> dict:
        mini = _normalise_plan({"title": self.cfg.get("name") or "Dashboard", "summary": "", "business_questions": [],
                                "kpis": [], "charts": [chart], "filters": []})
        clean, errors, _ = validate_plan(mini, self.profile, {})
        if errors or not clean["charts"]:
            raise ToolError("; ".join(errors) or "That chart doesn't fit the data.")
        spec = clean["charts"][0]
        spec["description"] = describe_chart(spec, self.profile)
        dup = next((c for c in self.cfg["charts"] if (c["type"], c.get("x"), c.get("y"), c.get("agg"), c.get("color"))
                    == (spec["type"], spec.get("x"), spec.get("y"), spec.get("agg"), spec.get("color"))), None)
        if dup:
            raise ToolError(f"The dashboard already has this chart (“{dup['title']}”). Suggest something different.")
        return spec

    def _apply_theme(self, preset: str = "", **colors) -> str:
        if preset in themes.PRESETS:
            base = themes.from_preset(preset)
        else:
            base = dict(themes.clean_theme(self.cfg.get("theme")) or themes.from_preset(themes.DEFAULT))
        for k in ("page_bg", "card_bg", "accent"):
            if themes.is_hex(colors.get(k)):
                base[k], base["preset"] = colors[k], "Custom"
        pal = [c for c in (colors.get("palette") or []) if themes.is_hex(c)]
        if len(pal) >= 2:
            base["palette"], base["preset"] = pal, "Custom"
        self.cfg["theme"] = themes.clean_theme(base)
        self.changed = True
        return self.cfg["theme"]["preset"]

    # ------------------------------------------------------------------ LLM tools
    def _t_describe(self, args: dict) -> dict:
        t = themes.resolve(self.cfg.get("theme"))
        return {
            "name": self.cfg.get("name"),
            "kpis": [k.get("label") for k in self.cfg.get("kpis", [])],
            "charts": [{"id": c["id"], "title": c["title"], "type": c["type"], "x": c.get("x"), "y": c.get("y"),
                        "aggregation": c.get("agg"), "legend": c.get("color")} for c in self.cfg.get("charts", [])],
            "filters_available": self.cfg.get("filter_columns", []),
            "rows_shown": len(self.df), "rows_total": self.total_rows,
            "colours": {"preset": t["preset"], "accent": t["accent"], "page": t["page_bg"], "cards": t["card_bg"],
                        "palette": t["palette"]},
            "colour_presets": list(themes.PRESETS),
        }

    def _t_propose(self, args: dict) -> dict:
        spec = self._validated_chart(args["chart"])
        if args["add_now"]:
            self.cfg["charts"].append(spec)
            self.changed = True
            return {"added": True, "chart_id": spec["id"], "title": spec["title"]}
        sid = self._add_suggestion(spec, args["reason"])
        self._turn_suggestions.append(sid)
        return {"drafted": True, "title": spec["title"],
                "note": "Shown to the user with an 'Add to dashboard' button. Mention it briefly in your reply."}

    def _t_remove(self, args: dict) -> dict:
        before = len(self.cfg["charts"])
        self.cfg["charts"] = [c for c in self.cfg["charts"] if c["id"] != args["chart_id"]]
        if len(self.cfg["charts"]) == before:
            raise ToolError("No chart with that id. Charts: " + "; ".join(f"{c['id']}={c['title']}" for c in self.cfg["charts"]))
        self.changed = True
        return {"removed": True}

    def _t_theme(self, args: dict) -> dict:
        bad = [k for k in ("accent", "page_bg", "card_bg") if args[k] and not themes.is_hex(args[k])]
        bad += [c for c in args["palette"] if not themes.is_hex(c)]
        if bad:
            raise ToolError(f"Colours must be hex codes like #1f5fbf. Not valid: {', '.join(bad[:5])}")
        name = self._apply_theme(args["preset"], accent=args["accent"], page_bg=args["page_bg"],
                                 card_bg=args["card_bg"], palette=args["palette"])
        return {"applied": True, "theme": name, "colours": self.cfg["theme"]}

    def _t_chart_color(self, args: dict) -> dict:
        if not themes.is_hex(args["color"]):
            raise ToolError("The colour must be a hex code like #e8590c.")
        for c in self.cfg["charts"]:
            if c["id"] == args["chart_id"]:
                c["color_hex"] = args["color"].lower()
                self.changed = True
                return {"applied": True, "chart": c["title"]}
        raise ToolError("No chart with that id. Charts: " + "; ".join(f"{c['id']}={c['title']}" for c in self.cfg["charts"]))

    def _t_remember(self, args: dict) -> dict:
        msg = knowledge.remember(self.kdir, self.columns, args["note"], args["scope"], args["term"], args["column"],
                                 self.cfg.get("dataset_name", ""))
        if msg.startswith("Not saved"):
            raise ToolError(msg)
        return {"saved": msg}

    def _tools(self) -> dict[str, tuple[dict, str, callable]]:
        presets = list(themes.PRESETS)
        return {
            "query_data": (TOOLS["query_data"].schema, TOOLS["query_data"].description, lambda a: t_query_data(self.data, a)),
            "get_insights": (TOOLS["get_insights"].schema, TOOLS["get_insights"].description,
                             lambda a: t_get_insights(self.data, a)),
            "describe_dashboard": (_obj({}), "See the dashboard's KPIs, charts (with ids), filters and colours.", self._t_describe),
            "propose_chart": (_obj({"chart": _PLAN_CHART, "reason": S, "add_now": B}),
                              "Draft a chart (validated against the data). add_now=false shows it as an idea with an 'Add' "
                              "button; add_now=true adds it immediately (only when the user asked). Use '' for unused fields.",
                              self._t_propose),
            "remove_chart": (_obj({"chart_id": S}), "Remove a chart, only when the user clearly asks.", self._t_remove),
            "set_theme": (_obj({"preset": _enum(presets + [""]), "accent": S, "page_bg": S, "card_bg": S, "palette": _arr(S)}),
                          f"Change the dashboard colours. preset: one of {', '.join(presets)} (Midnight is dark mode), or '' to "
                          "keep the current one. Optional hex overrides: accent, page_bg, card_bg, palette (2–8 chart colours, "
                          "first = main). Use '' / [] for no change.", self._t_theme),
            "set_chart_color": (_obj({"chart_id": S, "color": S}), "Set the main colour (hex) of one chart.", self._t_chart_color),
            "remember": (_obj({"note": S, "scope": _enum(["dataset", "general"]), "term": S, "column": S}),
                         "Save something you learned for next time. scope='dataset' for facts about this data, 'general' for "
                         "the user's preferences. To save a word meaning, set term (the user's word) and column (the real "
                         "column name); otherwise leave them ''.", self._t_remember),
        }

    @staticmethod
    def _step_label(name: str, args: dict, result: dict, ok: bool) -> str:
        if not ok:
            return f"Tried {name.replace('_', ' ')} — {str(result.get('error', ''))[:120]}"
        if name == "query_data":
            what = args.get("metric") or "rows"
            by = f" by {args['group_by']}" if args.get("group_by") else ""
            return f"Calculated {args.get('aggregation', '')} of {what}{by}".replace("  ", " ")
        return {"get_insights": "Looked for patterns, changes and outliers",
                "describe_dashboard": "Looked at what's on your dashboard",
                "propose_chart": f"Drafted a chart: {result.get('title', '')}",
                "remove_chart": "Removed a chart",
                "set_theme": f"Changed the colours ({result.get('theme', '')})",
                "set_chart_color": f"Recoloured “{result.get('chart', '')}”",
                "remember": f"Saved to memory: {args.get('term') + ' → ' + args.get('column') if args.get('term') else args.get('note', '')[:80]}",
                }.get(name, name)

    def _llm_turn(self, text: str) -> None:
        tools = self._tools()
        defs = [{"name": n, "description": d, "input_schema": s, "strict": True} for n, (s, d, _) in tools.items()]
        t = self.state["transcript"]
        self._trim_transcript()
        t.append({"role": "user", "content": [{"type": "text", "text": f"{text}\n\n{self._context_block()}"}]})
        steps: list[str] = []
        self._turn_suggestions: list[str] = []
        calls = 0
        mood = "aha"
        while True:
            reply = self.llm.create(SYSTEM_PROMPT, t, defs, allow_tools=calls < MAX_TOOL_CALLS)
            t.append({"role": "assistant", "content": reply.content})
            texts = [b["text"] for b in reply.content if b.get("type") == "text" and b.get("text", "").strip()]
            uses = [b for b in reply.content if b.get("type") == "tool_use"]
            if not uses or calls >= MAX_TOOL_CALLS:
                if reply.stop_reason == "refusal":
                    texts = ["I can't help with that one. Ask me something about this dashboard instead."]
                if self.changed:
                    mood = "celebrate"
                elif self._turn_suggestions:
                    mood = "present"
                self._say("\n\n".join(texts) or "Done.", mood=mood, steps=steps, suggestions=self._turn_suggestions)
                return
            results = []
            for tu in uses:
                name, args = tu["name"], tu.get("input") or {}
                spec = tools.get(name)
                if spec is None:
                    result, ok = {"error": f"Unknown tool '{name}'."}, False
                else:
                    try:
                        clean = validate_args(spec[0], args)
                        result, ok = spec[2](clean), True
                    except ToolError as exc:
                        result, ok = {"error": str(exc)}, False
                    except Exception as exc:  # noqa: BLE001 - a tool bug must not break the chat
                        result, ok = {"error": f"The tool failed ({type(exc).__name__})."}, False
                calls += 1
                steps.append(self._step_label(name, args if isinstance(args, dict) else {}, result, ok))
                content = json.dumps(result, default=str)[:7000]
                results.append({"type": "tool_result", "tool_use_id": tu["id"], "content": content,
                                **({} if ok else {"is_error": True})})
            t.append({"role": "user", "content": results})

    def _trim_transcript(self) -> None:
        """Keep the request small: drop the oldest exchanges, cutting only before a plain user message."""
        t = self.state["transcript"]
        # close tool calls left open by an interrupted run
        if t and t[-1]["role"] == "assistant":
            open_ids = [b["id"] for b in t[-1]["content"] if isinstance(b, dict) and b.get("type") == "tool_use"]
            if open_ids:
                t.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": i, "is_error": True,
                                                       "content": "Interrupted."} for i in open_ids]})
                t.append({"role": "assistant", "content": [{"type": "text", "text": "(interrupted)"}]})
        if len(t) <= MAX_TRANSCRIPT:
            return
        for i in range(len(t) - MAX_TRANSCRIPT, len(t)):
            m = t[i]
            if m["role"] == "user" and all(b.get("type") == "text" for b in m["content"]):
                del t[:i]
                return
        t.clear()

    # ------------------------------------------------------------------ offline assistant
    _COLOR_WORDS = {"ocean": "Ocean", "blue": "Ocean", "sea": "Ocean", "calm": "Ocean", "sunset": "Sunset",
                    "orange": "Sunset", "warm": "Sunset", "red": "Sunset", "pink": "Sunset", "forest": "Forest",
                    "green": "Forest", "nature": "Forest", "malaysia": "Malaysia", "jalur": "Malaysia",
                    "pastel": "Pastel", "soft": "Pastel", "cute": "Pastel", "dark": "Midnight", "night": "Midnight",
                    "midnight": "Midnight", "black": "Mono", "grey": "Mono", "gray": "Mono", "mono": "Mono",
                    "corporate": "Mono", "default": "Classic", "classic": "Classic", "original": "Classic", "reset": "Classic"}
    _MEANS = re.compile(r"^(?:please\s+)?(?:remember(?:\s+that)?\s+)?[\"'“]?(?P<term>[\w &/-]{2,40}?)[\"'”]?\s+(?:means|is|refers to)\s+"
                        r"(?:the\s+)?[\"'“]?(?P<col>[^\"'”]{2,120}?)[\"'”]?(?:\s+column)?\.?$", re.I)

    def _guided_turn(self, text: str) -> None:
        low = text.casefold()
        m = self._MEANS.match(text.strip())
        if m:
            cols = {c.casefold(): c for c in self.columns}
            col = cols.get(m.group("col").strip().casefold())
            if col:
                msg = knowledge.remember(self.kdir, self.columns, "", "dataset", m.group("term").strip(), col,
                                         self.cfg.get("dataset_name", ""))
                self._say(msg + " I'll understand that word in questions from now on.", mood="aha",
                          steps=["Saved to memory"])
                return
        if re.search(r"\b(colou?rs?|theme|palette|dark mode|light mode)\b", low):
            preset = next((p for w, p in self._COLOR_WORDS.items() if re.search(rf"\b{w}\b", low)), None)
            if preset:
                self._apply_theme(preset)
                self._say(f"Done — I switched the dashboard to the **{preset}** colours. You can fine-tune every colour "
                          "with the **Colours** button at the top.", mood="celebrate", steps=["Changed the colours"])
            else:
                self._say("Which style would you like? Options: " + ", ".join(themes.PRESETS) +
                          ". You can also pick exact colours with the **Colours** button at the top.", mood="idle")
            return
        if re.search(r"\b(ideas?|suggest|recommend|improve|what else|add (a )?(chart|graph|visual))\b", low):
            self._guided_ideas()
            return
        if re.search(r"\b(summary|summari[sz]e|overview|insights?|highlights?|good news|bad news|what is this data|tell me about|look into|check next|what next)\b", low):
            items = generate_insights(self.df, self.profile, max_items=5)
            body = "\n".join(f"- **{i.title}:** {i.text}" for i in items) or "I couldn't find notable patterns."
            self._say("Here's what stands out in the rows shown:\n\n" + body, mood="aha", steps=["Looked for patterns"])
            return
        if re.fullmatch(r"(hi|hello|hey|hai|helo|salam|thanks|thank you|terima kasih)[!. ]*", low):
            self._say("Hello! " + GUIDED_HELP, mood="idle")
            return
        kn = self.knowledge()
        res = answer_question(text, self.df, self.profile, aliases=knowledge.engine_aliases(kn))
        if res.ok:
            chart = None
            if res.chart and res.table is not None:
                chart = {"kind": res.chart.get("kind"), "x": res.chart.get("x"), "y": res.chart.get("y"),
                         "rows": json.loads(res.table.head(25).to_json(orient="records", date_format="iso", default_handler=str))}
            self._say(res.answer, mood="aha", steps=[f"Calculated with pandas: {res.method}"] if res.method else [], chart=chart)
        else:
            self._say(res.answer + "\n\nTip: if I don't know a word you use, teach me — e.g. “dept means Product "
                      "Category”. An AI model (choose one with the assistant button) understands free-form questions better.",
                      mood="idle")

    def _guided_ideas(self) -> None:
        have = {(c["type"], c.get("x"), c.get("y"), c.get("color")) for c in self.cfg["charts"]}
        ids = []
        for spec in recommend_charts(self.profile, max_charts=12):
            key = (spec["type"], spec.get("x"), spec.get("y"), spec.get("color"))
            if key in have:
                continue
            try:
                clean = validate_chart_spec(spec, self.profile)
            except ChartConfigError:
                continue
            clean["description"] = describe_chart(clean, self.profile)
            ids.append(self._add_suggestion(clean, clean["description"]))
            if len(ids) == 3:
                break
        if not ids:
            self._say("Your dashboard already covers the charts I'd recommend. Try asking a question about the data, "
                      "or change the colours.", mood="idle")
            return
        self._say("Here are a few charts your dashboard doesn't have yet. Add any you like:", mood="present",
                  steps=["Compared recommended charts with your dashboard"], suggestions=ids)
