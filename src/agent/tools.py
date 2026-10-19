"""Agent tools: validated, stage-gated application functions the LLM (or guided agent) may call.

Each tool has a strict JSON schema (sent to the model) and a Python handler. Inputs are validated here
again before anything runs; tools only read profiles, record decisions, or call the existing engine.
No tool executes model-written code, and numbers in results are always computed with pandas.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

from ..chart_generator import CHART_TYPES
from ..data_profiler import COORDINATE, DATE, DIMENSION, IDENTIFIER, METRIC, TEXT, ROLE_LABELS, allowed_roles
from ..formatting import format_value
from ..insights import generate_insights
from ..query_engine import answer_question
from . import cleaning as cl
from .context import AgentContext
from .planner import OBJECTIVES, PlanError, build_dashboard_config, classify_objective, suggest_plan, validate_plan
from .quality import chosen_ops, detect_issues
from .workflow import Stage, WorkflowError, go_back, sync, tools_for

MAX_RESULT_CHARS = 7000
BUSINESS_ROLES = ["measure", "category", "date", "identifier", "text", "ignore"]
ROLE_TO_PROFILE = {"measure": METRIC, "category": DIMENSION, "date": DATE, "identifier": IDENTIFIER, "text": TEXT, "ignore": "hidden"}
TOPICS = ["meaning", "unit", "aggregation", "role", "objective", "cleaning", "other"]
AGG_ENUM = ["sum", "mean", "median", "min", "max", "count", "nunique"]


class ToolError(Exception):
    """A tool call that can't be carried out; the message is returned to the model as an error result."""


@dataclass
class Tool:
    name: str
    description: str
    schema: dict
    handler: Callable[[AgentContext, dict], dict]
    interactive: bool = False  # ends the agent turn and waits for the user

    def api_definition(self) -> dict:
        return {"name": self.name, "description": self.description, "input_schema": self.schema, "strict": True}


# --------------------------------------------------------------------------- schema helpers

def _obj(props: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": props, "required": list(props) if required is None else required,
            "additionalProperties": False}


S = {"type": "string"}
B = {"type": "boolean"}
I = {"type": "integer"}


def _arr(items: dict) -> dict:
    return {"type": "array", "items": items}


def _enum(values: list[str]) -> dict:
    return {"type": "string", "enum": values}


def validate_args(schema: dict, value: Any, path: str = "input") -> Any:
    """Validate ``value`` against the subset of JSON Schema used by these tools."""
    t = schema.get("type")
    if "enum" in schema:
        if value not in schema["enum"]:
            raise ToolError(f"{path} must be one of {schema['enum']}.")
        return value
    if t == "object":
        if not isinstance(value, dict):
            raise ToolError(f"{path} must be an object.")
        props = schema.get("properties", {})
        extra = set(value) - set(props)
        if extra and schema.get("additionalProperties") is False:
            raise ToolError(f"{path} has unexpected fields: {', '.join(sorted(extra))}.")
        missing = [r for r in schema.get("required", []) if r not in value]
        if missing:
            raise ToolError(f"{path} is missing: {', '.join(missing)}.")
        return {k: validate_args(props[k], v, f"{path}.{k}") for k, v in value.items() if k in props}
    if t == "array":
        if not isinstance(value, list):
            raise ToolError(f"{path} must be a list.")
        return [validate_args(schema["items"], v, f"{path}[{i}]") for i, v in enumerate(value[:50])]
    if t == "string":
        if not isinstance(value, str):
            raise ToolError(f"{path} must be text.")
        return value[:2000]
    if t == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ToolError(f"{path} must be a whole number.")
        return value
    if t == "boolean":
        if not isinstance(value, bool):
            raise ToolError(f"{path} must be true or false.")
        return value
    raise ToolError(f"Unsupported schema at {path}.")


def _none(v: str | None) -> str | None:
    return v.strip() or None if isinstance(v, str) else v


def _short(v, n: int = 60) -> str:
    s = "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)
    return s if len(s) <= n else s[: n - 1] + "…"


# --------------------------------------------------------------------------- stage A: discovery

def _column_summary(ctx: AgentContext, name: str) -> dict:
    df, profile = ctx.working()
    cp = profile.columns[name]
    orig = ctx.original_name(name)
    info = {"name": name, "type": ROLE_LABELS.get(cp.role, cp.role), "missing_pct": round(cp.missing_pct, 1),
            "distinct_values": cp.unique}
    if orig != name:
        info["original_name"] = orig
    if cp.role == METRIC and cp.stats:
        info.update(min=format_value(cp.stats["min"]), max=format_value(cp.stats["max"]),
                    default_calculation="average" if cp.aggregation == "mean" else "total")
    elif cp.role == DATE and cp.stats:
        info.update(first=str(cp.stats["min"])[:10], last=str(cp.stats["max"])[:10])
    elif cp.top_values:
        info["examples"] = [_short(v, 30) for v, _ in cp.top_values[:4]]
    it = ctx.memory.interpretations.get(orig)
    if it:
        info["confirmed_meaning"] = it.get("meaning")
    return info


def t_inspect_dataset(ctx: AgentContext, args: dict) -> dict:
    df, profile = ctx.working()
    r = ctx.original
    m = ctx.memory
    return {
        "dataset": m.dataset_name, "sheet": m.sheet, "rows": profile.n_rows, "columns": profile.n_cols,
        "duplicate_rows": profile.duplicate_rows, "missing_cells_pct": round(profile.missing_pct, 1),
        "column_summaries": [_column_summary(ctx, c) for c in profile.columns][:60],
        "loading_notes": r.warnings[:10],
        "unclear_columns": [a["column"] for a in m.ambiguous if a["column"] not in m.interpretations],
        "objective": m.objective or None, "stage": m.stage,
    }


def t_get_column_profile(ctx: AgentContext, args: dict) -> dict:
    df, profile = ctx.working()
    name = args["column"]
    if name not in profile.columns:
        name = ctx.column_name(name)
    if name not in profile.columns:
        raise ToolError(f"There is no column called '{args['column']}'. Columns: {', '.join(list(profile.columns)[:40])}.")
    cp = profile.columns[name]
    out = _column_summary(ctx, name)
    out.update(notes=cp.notes[:5], outliers=cp.stats.get("outliers"), top_values=[[_short(v, 40), c] for v, c in cp.top_values[:10]])
    if cp.role == METRIC and cp.stats:
        out.update(average=format_value(cp.stats["mean"]), median=format_value(cp.stats["median"]), total=format_value(cp.stats["sum"]))
    amb = next((a for a in ctx.memory.ambiguous if a["column"] == ctx.original_name(name)), None)
    if amb:
        out["why_unclear"] = amb["reasons"]
        out["possible_reading"] = amb["expansion"]
    return out


def t_get_sample_records(ctx: AgentContext, args: dict) -> dict:
    if not ctx.sample_rows_allowed:
        raise ToolError("The user hasn't allowed example rows to be shared. Use column profiles instead.")
    n = max(1, min(5, args["count"]))
    df, _ = ctx.working()
    rows = df.sample(min(n, len(df)), random_state=7) if len(df) > n else df.head(n)
    return {"rows": [{c: _short(v, 40) for c, v in row.items()} for row in rows.to_dict(orient="records")]}


def t_identify_ambiguous_columns(ctx: AgentContext, args: dict) -> dict:
    m = ctx.memory
    out = []
    for a in m.ambiguous:
        out.append({"column": a["column"], "why_unclear": a["reasons"], "possible_reading": a["expansion"],
                    "matters_for_totals": a["blocking"], "unit_known": a.get("unit_known", False),
                    "suggested_answers": a["suggestions"],
                    "status": "resolved" if a["column"] in m.interpretations else
                              ("answered" if m.answers_for(a["column"]) else "open")})
    return {"unclear_columns": out, "note": "Only ask about columns whose meaning affects the dashboard. Ask one question at a time."}


def t_ask_user_clarification(ctx: AgentContext, args: dict) -> dict:
    m = ctx.memory
    column = _none(args["column"])
    if column:
        _, profile = ctx.working()
        if column not in profile.columns and ctx.column_name(column) not in profile.columns and column not in ctx.original_profile.columns:
            raise ToolError(f"There is no column called '{column}'.")
        column = ctx.original_name(column)
    previous = m.find_question(column, args["topic"])
    if previous and previous["status"] == "answered":
        return {"already_answered": True, "question": previous["question"], "answer": previous["answer"],
                "note": "The user already answered this. Don't ask again; use the answer."}
    if m.pending():
        raise ToolError("A question is already waiting for the user. Wait for their answer before asking another.")
    q, is_new = m.ask(args["question"].strip()[:500], column, args["topic"], [s[:120] for s in args["suggested_answers"]][:5])
    if not is_new:
        return {"already_answered": True, "question": q["question"], "answer": q["answer"],
                "note": "The user already answered this. Don't ask again; use the answer."}
    m.add_message("assistant", q["question"], kind="question", data={"question_id": q["id"], "suggestions": q["suggestions"], "column": column})
    return {"asked": True, "question_id": q["id"], "waiting_for_user": True}


def t_set_business_objective(ctx: AgentContext, args: dict) -> dict:
    m = ctx.memory
    if not any(msg["role"] == "user" for msg in m.conversation):
        raise ToolError("The objective must come from the user. Ask the user what they want to understand first.")
    text = args["objective"].strip()[:500]
    if not text:
        raise ToolError("The objective is empty.")
    category = args["category"] if args["category"] in OBJECTIVES else classify_objective(text)
    m.objective, m.objective_category = text, category
    m.log_event("objective", {"objective": text, "category": category})
    return {"saved": True, "objective": text, "focus": OBJECTIVES[category][0]}


def t_save_column_interpretation(ctx: AgentContext, args: dict) -> dict:
    m = ctx.memory
    col = ctx.original_name(args["column"])
    profile = ctx.original_profile
    if col not in profile.columns:
        raise ToolError(f"There is no column called '{args['column']}'.")
    answers = m.answers_for(col)
    amb = next((a for a in m.ambiguous if a["column"] == col), None)
    if args["based_on_user_answer"] and not answers:
        raise ToolError(f"There's no user answer about '{col}' yet. Ask the user first, or save it as your own inference.")
    if amb and amb.get("blocking") and not answers and args["business_role"] != "ignore":
        raise ToolError(f"'{col}' is ambiguous and affects totals. Ask the user what it means before saving an interpretation.")
    role = args["business_role"]
    cp = profile.columns[col]
    profile_role = ROLE_TO_PROFILE[role]
    if profile_role != "hidden" and profile_role not in allowed_roles(cp):
        raise ToolError(f"'{col}' holds {cp.dtype} data, so it can't be used as a {role}. "
                        f"Allowed: {', '.join(ROLE_LABELS[r] for r in allowed_roles(cp))}.")
    agg = args["aggregation"]
    if role != "measure":
        agg = "none"
    elif agg == "none":
        agg = cp.aggregation if cp.aggregation in ("sum", "mean") else "sum"
    display = args["display_name"].strip()[:60]
    if display:
        others = {c.casefold() for c in profile.columns if c != col} | {
            (it.get("display_name") or "").casefold() for c, it in m.interpretations.items() if c != col}
        if display.casefold() in others:
            raise ToolError(f"Another column is already called '{display}'. Choose a different display name.")
    m.interpretations[col] = {
        "meaning": args["meaning"].strip()[:300], "display_name": display, "data_type": args["data_type"],
        "unit": args["unit"].strip()[:30], "business_role": role, "aggregation": agg, "kpi_suitable": bool(args["kpi_suitable"]) and role == "measure",
        "source": "user" if answers else "inferred", "confirmed": bool(answers),
        "evidence": [q["answer"] for q in answers][-3:],
    }
    ctx.invalidate()
    m.log_event("interpretation", {"column": col, **{k: m.interpretations[col][k] for k in ("business_role", "aggregation", "unit")}})
    return {"saved": True, "column": col, "shown_as": display or col, "used_as": role,
            "calculation": {"sum": "added up", "mean": "averaged"}.get(agg, "not calculated"),
            "remaining_unclear": m.unresolved_columns()}


def t_go_to_stage(ctx: AgentContext, args: dict) -> dict:
    try:
        stage = go_back(ctx.memory, args["stage"], args["reason"])
    except WorkflowError as exc:
        raise ToolError(str(exc)) from exc
    return {"stage": stage.value}


# --------------------------------------------------------------------------- stage D: cleaning

def _issue_view(issue: dict) -> dict:
    return {"id": issue["id"], "title": issue["title"], "what": issue["detail"], "why_it_matters": issue["why"],
            "recommended": issue["recommended"]["label"], "alternatives": [a["label"] for a in issue["alternatives"]],
            "needs_approval": issue["requires_approval"], "status": issue["status"],
            "decision": issue["decision"] and cl.describe(issue["decision"])}


def t_propose_cleaning_operations(ctx: AgentContext, args: dict) -> dict:
    m, r = ctx.memory, ctx.original
    issues = detect_issues(r.df, ctx.original_profile, r.warnings, len(r.removed_rows), m.interpretations)
    previous = {i["id"]: i for i in m.cleaning_issues}
    for i in issues:  # keep decisions the user already made
        if i["id"] in previous and previous[i["id"]]["status"] in ("approved", "rejected"):
            i["status"], i["decision"] = previous[i["id"]]["status"], previous[i["id"]]["decision"]
    m.cleaning_issues = issues
    needing = [i for i in issues if i["requires_approval"]]
    if not needing:
        m.cleaning_reviewed = True
    return {"issues": [_issue_view(i) for i in issues][:20], "needing_approval": len(needing),
            "note": "Explain the important ones simply, then call request_cleaning_approval. Never fill missing numbers with zero."}


def t_preview_cleaning_operations(ctx: AgentContext, args: dict) -> dict:
    m, df = ctx.memory, ctx.original.df
    out = []
    for iid in args["issue_ids"][:10]:
        issue = next((i for i in m.cleaning_issues if i["id"] == iid), None)
        if issue is None:
            raise ToolError(f"Unknown issue id '{iid}'. Call propose_cleaning_operations first.")
        op = cl.validate_op(issue["recommended"]["op"], list(df.columns), set(df.select_dtypes("number").columns))
        out.append({"id": iid, **cl.preview(df, op)})
    return {"previews": out}


def t_request_cleaning_approval(ctx: AgentContext, args: dict) -> dict:
    m = ctx.memory
    if not m.cleaning_issues:
        raise ToolError("Call propose_cleaning_operations first.")
    pending = [i for i in m.cleaning_issues if i["requires_approval"] and i["status"] == "pending"]
    m.cleaning_reviewed = True
    if not pending:
        return {"waiting_for_user": False, "note": "All proposals already have a decision."}
    m.add_message("assistant", args["message"].strip()[:1500] or "Please review these clean-up suggestions.",
                  kind="cleaning_review", data={"issue_ids": [i["id"] for i in pending]})
    return {"waiting_for_user": True, "proposals": len(pending)}


def t_apply_approved_transformations(ctx: AgentContext, args: dict) -> dict:
    m = ctx.memory
    pending = [i for i in m.cleaning_issues if i["requires_approval"] and i["status"] == "pending"]
    if pending:
        raise ToolError(f"{len(pending)} proposal(s) still need the user's decision. Only the user can approve changes.")
    df = ctx.original.df
    numeric = set(df.select_dtypes("number").columns)
    ops = [cl.validate_op(op, list(df.columns), numeric) for op in chosen_ops(m.cleaning_issues)]
    _, log = cl.apply_ops(df, ops)
    m.applied_ops = ops
    m.cleaning_applied = True
    stamp = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
    reasons = {i["decision"]["op"] + str(i["decision"].get("column")): i["title"] for i in m.cleaning_issues if i.get("decision")}
    m.transformations = [dict(entry, approved_by_user=True, reason=reasons.get(entry["op"] + str(entry["column"]), ""), at=stamp)
                         for entry in log]
    for i in m.cleaning_issues:
        if i["status"] == "rejected":
            m.transformations.append({"op": "keep", "column": i.get("column"), "rows_affected": 0, "description": f"Kept as is: {i['title']}",
                                      "approved_by_user": True, "reason": "User chose not to change this.", "at": stamp})
    ctx.invalidate()
    return {"applied": [e["description"] + f" ({e['rows_affected']:,} rows)" for e in log] or ["No changes — data kept as is."],
            "original_file_unchanged": True}


# --------------------------------------------------------------------------- stage E/F: planning and dashboard

def _kpi_exclusions(ctx: AgentContext) -> set[str]:
    return {ctx.column_name(c) for c, it in ctx.memory.interpretations.items() if it.get("kpi_suitable") is False}


def _interp_by_name(ctx: AgentContext) -> dict:
    return {ctx.column_name(c): it for c, it in ctx.memory.interpretations.items()}


def t_suggest_dashboard_plan(ctx: AgentContext, args: dict) -> dict:
    df, profile = ctx.working()
    m = ctx.memory
    plan = suggest_plan(profile, df, m.objective_category or "exploratory", m.objective, _interp_by_name(ctx), _kpi_exclusions(ctx))
    return {"suggested_plan": plan, "note": "Adapt it to the user's objective if needed, then pass it to create_dashboard_plan."}


_PLAN_CHART = _obj({
    "type": _enum(list(CHART_TYPES)), "x": S, "y": S, "aggregation": _enum(AGG_ENUM + [""]), "color": S, "x2": S,
    "title": S, "why": S,
})
_PLAN_KPI = _obj({"kind": _enum(["agg", "count", "ratio"]), "column": S, "denominator": S,
                  "aggregation": _enum(AGG_ENUM + ["ratio"]), "label": S})
_PLAN = _obj({"title": S, "summary": S, "business_questions": _arr(S), "kpis": _arr(_PLAN_KPI), "charts": _arr(_PLAN_CHART), "filters": _arr(S)})


def _normalise_plan(raw: dict) -> dict:
    plan = dict(raw)
    plan["kpis"] = [{**k, "column": _none(k.get("column")), "denominator": _none(k.get("denominator"))} for k in raw.get("kpis", [])]
    plan["charts"] = [{**{k: _none(v) if isinstance(v, str) else v for k, v in c.items()},
                       "agg": _none(c.get("aggregation"))} for c in raw.get("charts", [])]
    return plan


def t_create_dashboard_plan(ctx: AgentContext, args: dict) -> dict:
    m = ctx.memory
    m.plan = {"raw": _normalise_plan(args["plan"]), "clean": None}
    m.plan_valid, m.plan_errors = False, []
    return {"saved": True, "charts": len(args["plan"]["charts"]), "kpis": len(args["plan"]["kpis"]),
            "next": "Call validate_dashboard_plan."}


def t_validate_dashboard_plan(ctx: AgentContext, args: dict) -> dict:
    m = ctx.memory
    if not m.plan:
        raise ToolError("Create a plan first.")
    _, profile = ctx.working()
    try:
        clean, errors, warnings = validate_plan(m.plan["raw"], profile, _interp_by_name(ctx))
    except PlanError as exc:
        raise ToolError(str(exc)) from exc
    m.plan["clean"] = clean if not errors else None
    m.plan_valid, m.plan_errors = not errors, errors
    return {"valid": not errors, "errors": errors, "warnings": warnings,
            "charts_ok": len(clean["charts"]), "kpis_ok": len(clean["kpis"])}


def t_execute_dashboard_plan(ctx: AgentContext, args: dict) -> dict:
    m = ctx.memory
    if not m.plan_valid or not m.plan or not m.plan.get("clean"):
        raise ToolError("The plan must pass validate_dashboard_plan first.")
    df, profile = ctx.working()
    cfg = build_dashboard_config(m.plan["clean"], profile, df, m.dataset_id, m.dataset_name, m.sheet, m.header_row, m)
    if m.dashboard_id:
        cfg["id"] = m.dashboard_id  # rebuilding keeps the same dashboard
    saved = ctx.storage.save_dashboard(cfg)
    m.dashboard_id = saved["id"]
    insights = [{"title": i.title, "finding": i.text} for i in generate_insights(df, profile)][:6]
    return {"dashboard_id": saved["id"], "name": saved["name"], "kpis": [k["label"] for k in saved["kpis"]],
            "charts": [c["title"] for c in saved["charts"]], "insights_calculated_by_python": insights,
            "note": "Explain what was built and the insights in plain language. Only quote numbers from these results."}


def column_aliases(ctx: AgentContext) -> dict[str, list[str]]:
    """Other words the user may use for each column: original name, confirmed meaning, readable expansion."""
    from .ambiguity import expand

    _, profile = ctx.working()
    out: dict[str, list[str]] = {}
    for name in profile.columns:
        orig = ctx.original_name(name)
        it = ctx.memory.interpretations.get(orig, {})
        names = [orig if orig != name else "", it.get("meaning", "") if it.get("source") == "user" else "", expand(orig)[0]]
        out[name] = [n for n in dict.fromkeys(names) if n and n.casefold() != name.casefold() and len(n) <= 60]
    return out


def t_answer_data_question(ctx: AgentContext, args: dict) -> dict:
    df, profile = ctx.working()
    res = answer_question(args["question"], df, profile, aliases=column_aliases(ctx))
    out = {"answered": res.ok, "answer": res.answer, "method": res.method}
    if res.table is not None and len(res.table):
        out["table"] = json.loads(res.table.head(10).to_json(orient="records", date_format="iso", default_handler=str))
    if not res.ok:
        out["note"] = "The calculation engine couldn't answer this. Ask the user to rephrase; never estimate numbers yourself."
    return out


QUERY_INTENTS = ["group_agg", "rank", "overall_agg", "period_rank", "trend", "count_rows", "count_distinct",
                 "correlation", "missing", "outliers", "extreme_row", "duplicates"]


def _resolve_column(ctx: AgentContext, profile, name: str) -> str | None:
    if not name:
        return None
    for candidate in (name, ctx.column_name(name)):
        if candidate in profile.columns:
            return candidate
    folded = {c.casefold(): c for c in profile.columns}
    if name.casefold() in folded:
        return folded[name.casefold()]
    raise ToolError(f"There is no column called '{name}'. Available columns: {', '.join(list(profile.columns)[:40])}.")


def t_query_data(ctx: AgentContext, args: dict) -> dict:
    """Structured calculation: the model chooses intent/columns, pandas computes the numbers."""
    from ..query_engine import NeedsClarification, QueryPlan, execute_plan

    df, profile = ctx.working()
    metric = _resolve_column(ctx, profile, args["metric"])
    metric2 = _resolve_column(ctx, profile, args["metric2"])
    group = _resolve_column(ctx, profile, args["group_by"])
    date_col = _resolve_column(ctx, profile, args["date_column"])
    grain = args["time_grain"] or None
    if (grain or args["intent"] in ("trend", "period_rank")) and not date_col:
        date_col = next(iter(profile.date_columns), None)
        if date_col is None:
            raise ToolError("This dataset has no date column, so time-based questions can't be answered.")
    if args["intent"] == "period_rank" and not grain:
        grain = "M"
    filters: dict[str, list[str]] = {}
    for f in args["filters"][:5]:
        col = _resolve_column(ctx, profile, f["column"])
        if col:
            filters[col] = [str(v) for v in f["values"][:20]]
    agg = args["aggregation"]
    if metric is None and agg not in ("count",):
        agg = "count"
    if args["intent"] == "count_distinct" and not group:
        group = metric
    plan = QueryPlan(intent=args["intent"], metric=metric, metric2=metric2, group=group, agg=agg,
                     n=max(1, min(50, args["top_n"] or 5)), order=args["order"], time_grain=grain, date_col=date_col,
                     filters=filters)
    try:
        res = execute_plan(plan, df, profile)
    except NeedsClarification as exc:
        raise ToolError(f"{exc} Columns: {', '.join(list(profile.columns)[:40])}.") from exc
    out = {"answered": res.ok, "result": res.answer, "method": res.method}
    if res.table is not None and len(res.table):
        out["table"] = json.loads(res.table.head(15).to_json(orient="records", date_format="iso", default_handler=str))
    return out


def t_get_insights(ctx: AgentContext, args: dict) -> dict:
    """Calculated findings (trend, concentration, outliers, correlation, data quality) plus recent period changes."""
    from ..dashboard_service import _period_compare

    df, profile = ctx.working()
    findings = [{"title": i.title, "finding": i.text, "kind": i.kind} for i in generate_insights(df, profile, max_items=10)]
    changes = []
    dates = [d for d in profile.date_columns if profile.columns[d].stats.get("span_days", 0) >= 60]
    if dates:
        for m in [c for c in profile.metric_columns if profile.columns[c].aggregation == "sum"][:5]:
            pct, label = _period_compare(df, m, "sum", dates[0])
            if pct is not None:
                changes.append({"measure": m, "change_pct": round(pct, 1), "comparison": label,
                                "direction": "up" if pct > 0 else "down" if pct < 0 else "flat"})
    return {"findings": findings, "recent_changes": changes,
            "note": "Good news = rising totals, strong categories; bad news = falling totals, heavy concentration, "
                    "data gaps. Quote only these numbers."}


def t_update_dashboard(ctx: AgentContext, args: dict) -> dict:
    m = ctx.memory
    if not m.dashboard_id:
        raise ToolError("There's no dashboard yet.")
    cfg = ctx.storage.load_dashboard(m.dashboard_id)
    _, profile = ctx.working()
    action = args["action"]
    if action in ("add_chart", "update_chart", "add_kpi"):
        mini = {"title": cfg["name"], "charts": [args["chart"]] if action != "add_kpi" else [],
                "kpis": [args["kpi"]] if action == "add_kpi" else []}
        mini = _normalise_plan(mini)
        clean, errors, _ = validate_plan(mini, profile, _interp_by_name(ctx))
        errors = [e for e in errors if e != "The plan has no valid charts." or action != "add_kpi"]
        if errors:
            raise ToolError("; ".join(errors))
        if action == "add_kpi":
            cfg["kpis"] = (cfg["kpis"] + clean["kpis"])[:8]
        elif action == "add_chart":
            cfg["charts"].append(clean["charts"][0])
        else:
            idx = next((i for i, c in enumerate(cfg["charts"]) if c["id"] == args["chart_id"]), None)
            if idx is None:
                raise ToolError(f"No chart with id '{args['chart_id']}'. Charts: " + "; ".join(f"{c['id']}={c['title']}" for c in cfg["charts"]))
            cfg["charts"][idx] = dict(clean["charts"][0], id=args["chart_id"])
    elif action == "remove_chart":
        before = len(cfg["charts"])
        cfg["charts"] = [c for c in cfg["charts"] if c["id"] != args["chart_id"]]
        if len(cfg["charts"]) == before:
            raise ToolError(f"No chart with id '{args['chart_id']}'. Charts: " + "; ".join(f"{c['id']}={c['title']}" for c in cfg["charts"]))
    elif action == "remove_kpi":
        before = len(cfg["kpis"])
        cfg["kpis"] = [k for k in cfg["kpis"] if k["label"].casefold() != args["kpi"]["label"].casefold()]
        if len(cfg["kpis"]) == before:
            raise ToolError("No KPI with that label. KPIs: " + ", ".join(k["label"] for k in cfg["kpis"]))
    elif action == "rename_dashboard":
        if not args["name"].strip():
            raise ToolError("The new name is empty.")
        cfg["name"] = args["name"].strip()[:120]
    saved = ctx.storage.save_dashboard(cfg)
    return {"updated": True, "charts": [{"id": c["id"], "title": c["title"]} for c in saved["charts"]],
            "kpis": [k["label"] for k in saved["kpis"]]}


# --------------------------------------------------------------------------- registry

TOOLS: dict[str, Tool] = {t.name: t for t in [
    Tool("inspect_dataset", "Get a structured overview of the dataset: size, columns with their detected type, missing values, "
         "examples, loading notes and which columns are unclear. Use this first. Never returns the full data.",
         _obj({}), t_inspect_dataset),
    Tool("get_column_profile", "Get detailed statistics and the most common values for one column.",
         _obj({"column": S}), t_get_column_profile),
    Tool("get_sample_records", "Get up to 5 example rows (values truncated). Only use when column profiles aren't enough.",
         _obj({"count": I}), t_get_sample_records),
    Tool("identify_ambiguous_columns", "List columns whose names can't be interpreted reliably and whether each affects totals.",
         _obj({}), t_identify_ambiguous_columns),
    Tool("ask_user_clarification", "Ask the user ONE question and wait for the answer (ends your turn). Use plain language and "
         "offer short suggested answers. Use column='' when the question isn't about a column. Already-answered questions are not re-asked.",
         _obj({"question": S, "column": S, "topic": _enum(TOPICS), "suggested_answers": _arr(S)}), t_ask_user_clarification, interactive=True),
    Tool("set_business_objective", "Record what the user wants to understand, in their words, plus the closest category.",
         _obj({"objective": S, "category": _enum(list(OBJECTIVES))}), t_set_business_objective),
    Tool("save_column_interpretation", "Save what a column means and how to use it. Set based_on_user_answer=true only when the user "
         "explained it. Use business_role='ignore' when the user doesn't know or wants it left out. aggregation: 'sum' for amounts "
         "and counts, 'mean' for rates, prices, scores and percentages, 'none' for non-measures. display_name='' keeps the original name.",
         _obj({"column": S, "meaning": S, "display_name": S, "data_type": _enum(["number", "text", "date"]), "unit": S,
               "business_role": _enum(BUSINESS_ROLES), "aggregation": _enum(["sum", "mean", "none"]), "kpi_suitable": B,
               "based_on_user_answer": B}), t_save_column_interpretation),
    Tool("go_to_stage", "Go back to an earlier stage when new information changes an earlier decision.",
         _obj({"stage": _enum([s.value for s in Stage if s not in (Stage.UPLOAD, Stage.PROFILE)]), "reason": S}), t_go_to_stage),
    Tool("propose_cleaning_operations", "Detect data-quality issues and build cleaning proposals (what, why, recommendation, alternatives).",
         _obj({}), t_propose_cleaning_operations),
    Tool("preview_cleaning_operations", "Show exactly what the recommended treatment would change for the given issues, without changing data.",
         _obj({"issue_ids": _arr(S)}), t_preview_cleaning_operations),
    Tool("request_cleaning_approval", "Show the proposals to the user with approve/reject controls and wait (ends your turn). "
         "The message should summarise the most important issues in simple language.",
         _obj({"message": S}), t_request_cleaning_approval, interactive=True),
    Tool("apply_approved_transformations", "Apply only the treatments the user approved. The original file is never changed.",
         _obj({}), t_apply_approved_transformations),
    Tool("suggest_dashboard_plan", "Get a rule-based dashboard plan tailored to the objective and confirmed column meanings.",
         _obj({}), t_suggest_dashboard_plan),
    Tool("create_dashboard_plan", "Save a dashboard plan: KPIs (column + aggregation), charts (type, x, y, aggregation, color, x2), "
         "filters and the business questions it answers. Use '' for unused fields. Titles must not contain numbers or findings.",
         _obj({"plan": _PLAN}), t_create_dashboard_plan),
    Tool("validate_dashboard_plan", "Check the saved plan against the data and the confirmed interpretations.",
         _obj({}), t_validate_dashboard_plan),
    Tool("execute_dashboard_plan", "Build and save the dashboard from the validated plan. Returns calculated insights.",
         _obj({}), t_execute_dashboard_plan),
    Tool("answer_data_question", "Answer a simple, clearly worded question with the rule-based question engine. "
         "Prefer query_data, which lets you choose the exact columns.",
         _obj({"question": S}), t_answer_data_question),
    Tool("query_data", "Calculate an answer with pandas. YOU translate the user's words to real column names "
         "(e.g. 'department' → the closest category column such as 'Product Category'; 'sales' → 'Net Sales (RM)'), "
         "and say which columns you used. intent: group_agg (value per group), rank (top/bottom N groups), "
         "overall_agg (one total/average), period_rank (best/worst month etc.), trend, count_rows, count_distinct, "
         "correlation (metric vs metric2), missing, outliers, extreme_row, duplicates. Use '' for unused text fields and 0 "
         "for top_n when not ranking. filters: [{column, values}] to restrict rows.",
         _obj({"intent": _enum(QUERY_INTENTS), "metric": S, "metric2": S, "group_by": S,
               "aggregation": _enum(["sum", "mean", "median", "min", "max", "count", "nunique"]),
               "top_n": I, "order": _enum(["desc", "asc"]), "date_column": S, "time_grain": _enum(["", "D", "W", "M", "Q", "Y"]),
               "filters": _arr(_obj({"column": S, "values": _arr(S)}))}), t_query_data),
    Tool("get_insights", "Calculated findings for summaries and 'good news / bad news': trends, the biggest groups, "
         "outliers, related measures, data gaps, and the latest month-over-month change of each total.",
         _obj({}), t_get_insights),
    Tool("update_dashboard", "Change the dashboard: add/update/remove a chart, add/remove a KPI, or rename it. "
         "Fill only the fields the action needs; use '' or empty objects elsewhere.",
         _obj({"action": _enum(["add_chart", "update_chart", "remove_chart", "add_kpi", "remove_kpi", "rename_dashboard"]),
               "chart_id": S, "chart": _PLAN_CHART, "kpi": _PLAN_KPI, "name": S}), t_update_dashboard),
]}


def available_tools(stage: str) -> list[Tool]:
    return [TOOLS[n] for n in tools_for(stage)]


def run_tool(ctx: AgentContext, name: str, args: Any) -> tuple[dict, bool]:
    """Validate and execute a tool call. Returns (result, is_error). Never raises for bad model input."""
    m = ctx.memory
    tool = TOOLS.get(name)
    if tool is None:
        return {"error": f"Unknown tool '{name}'."}, True
    if name not in tools_for(m.stage):
        return {"error": f"'{name}' isn't available at the current stage ({m.stage}). Finish the current step first."}, True
    try:
        clean = validate_args(tool.schema, args if args is not None else {})
        result = tool.handler(ctx, clean)
        ok = True
    except ToolError as exc:
        result, ok = {"error": str(exc)}, False
    except Exception as exc:  # noqa: BLE001 - a tool bug must not crash the conversation
        result, ok = {"error": f"The tool failed ({type(exc).__name__}). Try a different approach."}, False
    m.log_event(f"tool:{name}", {"args": _short(json.dumps(args, default=str), 300), "ok": ok}, ok=ok)
    before = m.stage
    sync(m)
    if m.stage != before:
        result = dict(result, stage_now=m.stage)
    ctx.save()
    text = json.dumps(result, default=str)
    if len(text) > MAX_RESULT_CHARS:
        result = {"truncated": True, "partial": text[:MAX_RESULT_CHARS]}
    return result, not ok
