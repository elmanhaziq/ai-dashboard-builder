"""Offline guided analyst: the same workflow and tools, driven by deterministic rules instead of an LLM.

Used when no AI provider is configured, the user hasn't consented to sending data to one, or the provider
is unavailable. It calls tools through ``run_tool`` exactly like the LLM, so it gets the same validation.
"""
from __future__ import annotations

import re

from ..formatting import format_value
from .actions import all_decided, decisions_summary
from .ambiguity import readable
from .context import AgentContext
from .planner import OBJECTIVES, classify_objective
from .tools import run_tool
from .workflow import Stage

OBJECTIVE_SUGGESTIONS = [
    "Sales performance", "Revenue and profitability", "Marketing performance", "Customer analysis",
    "Operational efficiency", "Inventory management", "Just explore the data",
]
_SKIP = re.compile(r"not sure|don.?t know|no idea|leave (it )?out|skip|ignore|unknown|tak (pasti|tahu)", re.IGNORECASE)
_MONEY = re.compile(r"\brm\b|ringgit|\$|usd|dollar|money|sales|revenue|amount|income|value|spend|cost|price|profit|payment|currency", re.IGNORECASE)
_COUNT = re.compile(r"quantit|count|units?\b|pieces|items?\b|number of|qty|volume|orders?\b|sold", re.IGNORECASE)
_RATE = re.compile(r"percent|%|rate|ratio|average|avg|score|rating|margin|price", re.IGNORECASE)
_GENERIC_SUGGESTION = re.compile(r"^(a money amount|a quantity|a percentage|not sure)", re.IGNORECASE)


def _say(ctx: AgentContext, text: str, kind: str = "text", data: dict | None = None) -> None:
    ctx.memory.add_message("assistant", text, kind=kind, data=data)


def _ask(ctx: AgentContext, question: str, column: str | None, topic: str, suggestions: list[str]) -> bool:
    result, err = run_tool(ctx, "ask_user_clarification",
                           {"question": question, "column": column or "", "topic": topic, "suggested_answers": suggestions})
    return not err and result.get("asked", False)


def _interpret_answer(ctx: AgentContext, column: str, answer: str, amb: dict | None, prior: dict | None) -> tuple[dict, bool]:
    """Turn a free-text answer into an interpretation. Returns (args for save tool, needs unit follow-up)."""
    cp = ctx.original_profile.columns[column]
    if _SKIP.search(answer):
        return {"column": column, "meaning": "Unknown (user wasn't sure)", "display_name": "", "data_type": "number" if cp.role == "metric" else "text",
                "unit": "", "business_role": "ignore", "aggregation": "none", "kpi_suitable": False, "based_on_user_answer": True}, False
    is_measure = cp.role == "metric"
    confirmed = bool(re.fullmatch(r"(yes|ya|yup|correct|right|ok(ay)?|betul)\b.*", answer.strip(), re.I))
    typed = not _GENERIC_SUGGESTION.search(answer)
    if confirmed and amb:
        display, meaning = readable(amb["expansion"])[:50], readable(amb["expansion"])
    else:
        meaning = answer.strip()[:300]
        display = answer.strip().rstrip(".")[:50] if typed and len(answer.strip()) <= 50 else ""
        if display and amb and display.casefold() == amb["expansion"].casefold():
            display = readable(amb["expansion"])
    text = f"{answer} {meaning}"  # detect the unit from the confirmed meaning too ("quantity sold" → a count)
    unit, agg, unit_clear = (prior or {}).get("unit", ""), "sum", False
    if is_measure:
        if _RATE.search(text):
            unit, agg, unit_clear = ("%" if re.search(r"percent|%", text, re.I) else unit or ""), "mean", True
        elif _MONEY.search(text) or cp.is_currency:
            unit, unit_clear = ("RM" if re.search(r"\brm\b|ringgit", text, re.I) or cp.is_currency else (unit or "currency")), True
        elif _COUNT.search(text):
            unit, unit_clear = "units", True
        if cp.is_percent:
            unit, agg, unit_clear = "%", "mean", True
    args = {"column": column, "meaning": meaning, "display_name": display if display.casefold() != column.casefold() else "",
            "data_type": "number" if is_measure else ("date" if cp.role == "date" else "text"), "unit": unit,
            "business_role": "measure" if is_measure else ("identifier" if cp.role == "identifier" else "category"),
            "aggregation": agg if is_measure else "none", "kpi_suitable": is_measure, "based_on_user_answer": True}
    needs_unit = is_measure and not unit_clear and not (amb or {}).get("unit_known")
    return args, needs_unit


def _apply_unit_answer(ctx: AgentContext, column: str, answer: str) -> None:
    it = dict(ctx.memory.interpretations.get(column, {}))
    if not it:
        return
    if _RATE.search(answer):
        it.update(unit="%" if re.search(r"percent|%", answer, re.I) else it.get("unit", ""), aggregation="mean")
    elif _MONEY.search(answer):
        it.update(unit="RM" if re.search(r"\brm\b|ringgit", answer, re.I) else "currency", aggregation="sum")
    elif _COUNT.search(answer):
        it.update(unit="units", aggregation="sum")
    run_tool(ctx, "save_column_interpretation", {
        "column": column, "meaning": it.get("meaning", ""), "display_name": it.get("display_name", ""),
        "data_type": it.get("data_type", "number"), "unit": it.get("unit", ""), "business_role": it.get("business_role", "measure"),
        "aggregation": it.get("aggregation", "sum"), "kpi_suitable": it.get("kpi_suitable", True), "based_on_user_answer": True})


def start(ctx: AgentContext) -> None:
    """First message after upload: what was found, then the objective question."""
    m = ctx.memory
    if m.conversation:
        return
    insp, _ = run_tool(ctx, "inspect_dataset", {})
    unclear = insp.get("unclear_columns", [])
    lines = [f"I've read **{m.dataset_name}**" + (f" (sheet “{m.sheet}”)" if m.sheet and m.sheet != "CSV data" else "")
             + f": {insp['rows']:,} rows and {insp['columns']} columns."]
    if insp.get("loading_notes"):
        lines.append(f"While reading it I tidied up a few layout issues ({len(insp['loading_notes'])} notes); I'll explain them when we review data quality.")
    if unclear:
        lines.append(f"A few column names aren't clear to me ({', '.join(unclear[:4])}{'…' if len(unclear) > 4 else ''}). I'll ask about those one at a time.")
    _say(ctx, " ".join(lines))
    _ask(ctx, "First, what would you like to understand from this data?", None, "objective", OBJECTIVE_SUGGESTIONS)


def handle(ctx: AgentContext, user_text: str | None = None, event: str | None = None) -> None:
    """Process one user turn (or UI event) and advance the workflow as far as possible."""
    m = ctx.memory
    answered = None
    if user_text:
        m.add_message("user", user_text)
        answered = m.answer_pending(user_text)
    if event == "cleaning_decided" and all_decided(ctx):
        m.add_message("user", f"(Reviewed the clean-up suggestions: {decisions_summary(ctx) or 'no changes'})", kind="event")
    if not m.conversation or (len(m.conversation) == 1 and m.conversation[0]["role"] == "user"):
        start(ctx)
        return

    for _ in range(12):
        stage = Stage(m.stage)
        if stage == Stage.OBJECTIVE:
            if answered and answered["kind"] == "objective":
                text = answered["answer"]
                cat = next((k for k, (label, _) in OBJECTIVES.items() if label.lower() == text.lower()), None) or classify_objective(text)
                run_tool(ctx, "set_business_objective", {"objective": text, "category": cat})
                _say(ctx, f"Great — I'll focus on **{OBJECTIVES[cat][0].lower()}**.")
                answered = None
                continue
            if not m.pending():
                _ask(ctx, "What would you like to understand from this data?", None, "objective", OBJECTIVE_SUGGESTIONS)
            return

        if stage == Stage.CLARIFICATION:
            if answered and answered["column"]:
                col = answered["column"]
                amb = next((a for a in m.ambiguous if a["column"] == col), None)
                if answered["kind"] == "unit":
                    _apply_unit_answer(ctx, col, answered["answer"])
                else:
                    args, needs_unit = _interpret_answer(ctx, col, answered["answer"], amb, m.interpretations.get(col))
                    result, err = run_tool(ctx, "save_column_interpretation", args)
                    if err:
                        _say(ctx, f"I couldn't save that: {result['error']}")
                    elif needs_unit and not m.find_question(col, "unit"):
                        _ask(ctx, f"Thanks. Is “{args['display_name'] or col}” a money amount, a count of items, or a percentage?",
                             col, "unit", ["A money amount (RM)", "A count of items", "A percentage or rate"])
                        return
                answered = None
                continue
            nxt = next((a for a in m.ambiguous if a.get("blocking") and a["column"] not in m.interpretations), None)
            if nxt and not m.pending():
                reason = nxt["reasons"][0]
                _ask(ctx, f"I found a column named **{nxt['column']}**. {reason} Could you tell me what it represents?",
                     nxt["column"], "meaning", nxt["suggestions"])
                return
            if m.pending():
                return
            continue  # nothing unclear: the workflow moves on

        if stage == Stage.CLEANING_REVIEW:
            result, _ = run_tool(ctx, "propose_cleaning_operations", {})
            needing = [i for i in m.cleaning_issues if i["requires_approval"] and i["status"] == "pending"]
            if needing:
                top = needing[:3]
                bullet = "\n".join(f"- **{i['title']}** — {i['detail']} {i['why']}" for i in top)
                more = f"\n\n…and {len(needing) - 3} more." if len(needing) > 3 else ""
                run_tool(ctx, "request_cleaning_approval",
                         {"message": "I checked the data quality. Here's what I recommend — nothing changes until you approve, "
                                     f"and your original file is always kept as it is:\n\n{bullet}{more}"})
                return
            continue

        if stage == Stage.CLEANING_APPROVAL:
            if not all_decided(ctx):
                return
            result, err = run_tool(ctx, "apply_approved_transformations", {})
            if not err:
                changes = result.get("applied", [])
                _say(ctx, "Done. Changes applied to a working copy (your original file is untouched):\n\n" + "\n".join(f"- {c}" for c in changes))
            continue

        if stage == Stage.DASHBOARD_PLANNING:
            sug, err = run_tool(ctx, "suggest_dashboard_plan", {})
            if err:
                _say(ctx, f"I couldn't plan the dashboard: {sug.get('error')}", kind="notice")
                return
            plan = sug["suggested_plan"]
            plan_args = {"title": plan["title"], "summary": plan["summary"], "business_questions": plan["business_questions"],
                         "kpis": [{"kind": k["kind"], "column": k.get("column") or "", "denominator": k.get("denominator") or "",
                                   "aggregation": k["aggregation"], "label": k["label"]} for k in plan["kpis"]],
                         "charts": [{"type": c["type"], "x": c.get("x") or "", "y": c.get("y") or "", "aggregation": c.get("agg") or "",
                                     "color": c.get("color") or "", "x2": c.get("x2") or "", "title": c.get("title") or "", "why": ""}
                                    for c in plan["charts"]],
                         "filters": plan["filters"]}
            run_tool(ctx, "create_dashboard_plan", {"plan": plan_args})
            val, _ = run_tool(ctx, "validate_dashboard_plan", {})
            if not val.get("valid"):
                # Keep only the parts that passed validation and try once more.
                clean_titles = val.get("errors", [])
                plan_args["charts"] = plan_args["charts"][: max(1, val.get("charts_ok", 1))]
                run_tool(ctx, "create_dashboard_plan", {"plan": plan_args})
                val, _ = run_tool(ctx, "validate_dashboard_plan", {})
                if not val.get("valid"):
                    _say(ctx, "I couldn't build a valid dashboard automatically: " + "; ".join(clean_titles[:3]))
                    return
            continue

        if stage in (Stage.PLAN_VALIDATION, Stage.DASHBOARD_GENERATION):
            res, err = run_tool(ctx, "execute_dashboard_plan", {})
            if err:
                _say(ctx, f"Something went wrong building the dashboard: {res['error']}")
                return
            ins = "\n".join(f"- **{i['title']}:** {i['finding']}" for i in res["insights_calculated_by_python"][:4])
            _say(ctx, f"Your dashboard **{res['name']}** is ready, with {len(res['kpis'])} key numbers and {len(res['charts'])} charts. "
                      f"Here's what stands out (calculated directly from your data):\n\n{ins}\n\n"
                      "Open it with the button on the right, or ask me questions — for example "
                      "“Which region has the highest sales?”.", kind="dashboard_ready", data={"dashboard_id": res["dashboard_id"]})
            continue

        if stage == Stage.READY:
            if user_text and not answered:
                res, _ = run_tool(ctx, "answer_data_question", {"question": user_text})
                _say(ctx, res.get("answer", "I couldn't work that out."), kind="answer", data={"table": res.get("table"), "method": res.get("method")})
            return
        return


__all__ = ["handle", "start", "format_value"]
