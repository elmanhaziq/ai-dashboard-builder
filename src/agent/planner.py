"""Objective-aware dashboard planning, plan validation, and plan execution.

A *plan* is plain JSON: KPIs (column + aggregation), charts (the existing chart-spec fields), filters and
the business questions it answers. Plans may come from the LLM or from the deterministic suggester, but
every plan is validated against the interpreted schema and built by the existing chart engine.
"""
from __future__ import annotations

import copy
import re
from typing import Any

from ..chart_generator import (
    AGGREGATIONS, CHART_TYPES, ChartConfigError, agg_label, attach_correlations, describe_chart,
    recommend_charts, recommend_kpis, validate_chart_spec, validate_kpi_spec,
)
from ..dashboard_service import build_default_dashboard, default_filter_columns, filter_candidates
from ..data_profiler import DATE, DIMENSION, HIDDEN, IDENTIFIER, METRIC, DatasetProfile

OBJECTIVES = {
    "sales": ("Sales performance", r"sales|revenue|turnover|amount|order|quantity|qty|units|sold|value|price"),
    "profitability": ("Revenue and profitability", r"profit|margin|revenue|sales|cost|cogs|expense|discount|net|gross"),
    "marketing": ("Marketing performance", r"click|impression|conversion|lead|spend|ctr|cpc|campaign|visit|session|reach"),
    "customer": ("Customer analysis", r"customer|client|member|account|order|sales|revenue|visit|segment"),
    "operations": ("Operational efficiency", r"duration|time|hours|days|lead|delay|count|volume|throughput|cost|ticket"),
    "inventory": ("Inventory management", r"stock|inventory|on hand|reorder|quantity|qty|units|warehouse|cost"),
    "finance": ("Spending and finance", r"expense|cost|spend|budget|amount|tax|payment|balance"),
    "hr": ("People and HR", r"salary|headcount|employee|tenure|hours|age|rating|bonus"),
    "exploratory": ("General exploration", r"."),
}
OBJECTIVE_HINTS = {
    "sales": r"sale|sell|revenue|turnover|order",
    "profitability": r"profit|margin|cost|profitab",
    "marketing": r"market|campaign|ad |ads|click|lead|conversion",
    "customer": r"customer|client|member|churn|retention|segment",
    "operations": r"operation|efficien|process|delivery|ticket|delay|throughput",
    "inventory": r"inventory|stock|warehouse|reorder",
    "finance": r"expense|spend|budget|financ|cost control",
    "hr": r"hr\b|people|staff|employee|headcount|salary",
}
QUESTION_BANK = {
    "sales": ["How are sales trending over time?", "Which regions, branches or products sell the most?", "Where is performance falling behind?"],
    "profitability": ["Which products or regions are most profitable?", "How do costs compare with revenue?", "Where are margins shrinking?"],
    "marketing": ["Which channels or campaigns perform best?", "How is spend converting into results?", "How is performance changing over time?"],
    "customer": ["Who are the most valuable customers?", "How are customers distributed across segments?", "How is customer activity changing?"],
    "operations": ["Where do delays or bottlenecks occur?", "How is volume changing over time?", "Which teams or locations perform best?"],
    "inventory": ["Which items have the most stock?", "Where is stock running low?", "How is stock split across locations?"],
    "finance": ["Where is the money being spent?", "How is spending changing over time?", "Which categories or vendors cost the most?"],
    "hr": ["How is headcount distributed?", "How do pay and tenure compare across teams?", "Where are the biggest differences?"],
    "exploratory": ["What are the main totals?", "How do the key measures change over time?", "Which groups stand out?"],
}
# Text in plans must not state numeric findings — numbers come only from calculations on the data.
_NUMERIC_CLAIM = re.compile(
    r"\d[\d,.]*\s*(?:%|percent\b|k\b|m\b|million\b|billion\b|rm\b)"  # 20%, 5k, 3 million, 100 RM
    r"|(?:\brm|\$|€|£)\s*\d"  # RM 1,200 / $50
    r"|\b\d{1,3}(?:,\d{3})+\b"  # 1,234,567
    r"|\b\d{5,}\b",  # 12345 (years like 2024 are fine)
    re.IGNORECASE,
)


class PlanError(ValueError):
    pass


def classify_objective(text: str) -> str:
    t = (text or "").lower()
    for key, pattern in OBJECTIVE_HINTS.items():
        if re.search(pattern, t):
            return key
    return "exploratory"


def _rank_for_objective(profile: DatasetProfile, objective: str, interpretations: dict, objective_text: str = "") -> DatasetProfile:
    """Copy of the profile with measure/category priorities nudged toward the user's objective."""
    from .ambiguity import expand

    p = copy.deepcopy(profile)
    pattern = OBJECTIVES.get(objective, OBJECTIVES["exploratory"])[1]
    meanings = {name: (it.get("meaning") or "") for name, it in interpretations.items()}
    # Words the user used when describing their goal ("by branch and region") pull those columns forward.
    category_words = {w.rstrip("s") for w in re.findall(r"[a-z]{4,}", OBJECTIVES.get(objective, OBJECTIVES["exploratory"])[0].lower())}
    goal_words = {w.rstrip("s") for w in re.findall(r"[a-z]{4,}", objective_text.lower())} - category_words - {
        "want", "show", "understand", "performance", "data", "with", "from", "that", "this", "what", "which", "about"}
    for cp in p.columns.values():
        text = f"{cp.name} {meanings.get(cp.name, '')} {expand(cp.name)[0]}".lower()
        if cp.role == METRIC and re.search(pattern, text, re.IGNORECASE):  # objective keywords rank measures only
            cp.priority += 4
        if cp.role in (METRIC, DIMENSION) and any(w in text for w in goal_words):
            cp.priority += 5
    return p


def suggest_plan(profile: DatasetProfile, df, objective_category: str, objective_text: str,
                 interpretations: dict, kpi_exclude: set[str] | None = None) -> dict:
    """Deterministic plan from the existing recommendation rules, biased toward the objective."""
    ranked = _rank_for_objective(profile, objective_category, interpretations, objective_text or "")
    attach_correlations(ranked, df)
    charts = recommend_charts(ranked, max_charts=8)
    kpis = [k for k in recommend_kpis(ranked, max_kpis=6)
            if not (k.get("column") in (kpi_exclude or set()) and k.get("kind") == "agg")]
    label = OBJECTIVES.get(objective_category, OBJECTIVES["exploratory"])[0]
    return {
        "title": f"{label}",
        "summary": f"A dashboard focused on {label.lower()}" + (f": {objective_text.strip()[:140]}" if objective_text else "."),
        "business_questions": QUESTION_BANK.get(objective_category, QUESTION_BANK["exploratory"]),
        "kpis": [_kpi_to_plan(k) for k in kpis],
        "charts": [{k: c.get(k) for k in ("type", "x", "y", "agg", "color", "x2", "time_grain", "sort", "top_n", "title")}
                   | {"why": c.get("description", "")} for c in charts],
        "filters": default_filter_columns(ranked),
    }


def _kpi_to_plan(k: dict) -> dict:
    if k["kind"] == "count":
        return {"kind": "count", "column": "", "aggregation": "count", "label": k["label"]}
    if k["kind"] == "ratio":
        return {"kind": "ratio", "column": k["numerator"], "denominator": k["denominator"], "aggregation": "ratio",
                "label": k["label"], "format": k.get("format", "number")}
    return {"kind": "agg", "column": k["column"], "aggregation": k["agg"], "label": k["label"]}


def _check_text(text: Any, where: str, errors: list[str]) -> str:
    s = str(text or "").strip()[:200]
    if _NUMERIC_CLAIM.search(s):
        errors.append(f"{where} states a number (“{s}”). Numbers must come from calculations, not from the plan text.")
    return s


def validate_plan(plan: Any, profile: DatasetProfile, interpretations: dict) -> tuple[dict, list[str], list[str]]:
    """Check a plan against the interpreted schema. Returns (clean plan, errors, warnings)."""
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(plan, dict):
        raise PlanError("The dashboard plan must be an object.")
    title = _check_text(plan.get("title") or "Dashboard", "The title", errors) or "Dashboard"
    summary = _check_text(plan.get("summary") or "", "The summary", errors)
    questions = [_check_text(q, "A business question", errors) for q in (plan.get("business_questions") or [])[:6]]

    meaning = {c: it for c, it in interpretations.items()}
    def agg_conflict(col: str, agg: str) -> str | None:
        cp = profile.columns.get(col)
        if cp is None:
            return None
        if cp.role == IDENTIFIER and agg not in ("nunique", "count"):
            return f"'{col}' is an identifier (like an order number), so it can only be counted, not added up or averaged."
        if cp.role == HIDDEN:
            return f"'{col}' was set to be left out of the dashboard."
        if cp.role == METRIC and agg == "sum" and cp.aggregation == "mean":
            return f"'{col}' is an average-type measure (a rate, price or score); adding it up would be misleading — use the average."
        it = meaning.get(col) or {}
        if it.get("kpi_suitable") is False and agg in ("sum", "mean"):
            return f"You said '{col}' shouldn't be used as a headline number."
        return None

    kpis = []
    for i, k in enumerate((plan.get("kpis") or [])[:8]):
        if not isinstance(k, dict):
            errors.append(f"KPI {i + 1} is malformed.")
            continue
        kind = k.get("kind") or ("count" if k.get("aggregation") == "count" else "agg")
        label = _check_text(k.get("label") or "", f"KPI {i + 1}'s label", errors)
        try:
            if kind == "count":
                spec = validate_kpi_spec({"kind": "count", "label": label or "Records"}, profile)
            elif kind == "ratio":
                spec = validate_kpi_spec({"kind": "ratio", "numerator": k.get("column"), "denominator": k.get("denominator"),
                                          "label": label, "format": k.get("format", "number")}, profile)
            else:
                col, agg = k.get("column"), k.get("aggregation")
                problem = agg_conflict(col, agg)
                if problem:
                    raise ChartConfigError(problem)
                date_col = next((d for d in profile.date_columns if profile.columns[d].stats.get("span_days", 0) >= 14), None)
                spec = validate_kpi_spec({"kind": "agg", "column": col, "agg": agg,
                                          "label": label or (agg_label(agg, col) if agg != "nunique" else f"Distinct {col}"),
                                          "compare_date": date_col if agg == "sum" else None,
                                          "description": f"{AGGREGATIONS.get(agg, agg)} of '{col}'."}, profile)
            kpis.append(spec)
        except ChartConfigError as exc:
            errors.append(f"KPI {i + 1} ({label or k.get('column')}): {exc}")

    charts = []
    for i, c in enumerate((plan.get("charts") or [])[:12]):
        if not isinstance(c, dict):
            errors.append(f"Chart {i + 1} is malformed.")
            continue
        if c.get("type") not in CHART_TYPES:
            errors.append(f"Chart {i + 1}: unsupported chart type {c.get('type')!r}.")
            continue
        chart_title = _check_text(c.get("title") or "", f"Chart {i + 1}'s title", errors)
        why = _check_text(c.get("why") or "", f"Chart {i + 1}'s explanation", errors)
        raw = {k: c.get(k) for k in ("type", "x", "y", "color", "x2", "size", "sort", "top_n", "time_grain", "barmode")}
        raw["agg"] = c.get("agg") or c.get("aggregation") or ("count" if not c.get("y") else None)
        if raw["agg"] is None and raw.get("y") in profile.columns:
            raw["agg"] = profile.columns[raw["y"]].aggregation if profile.columns[raw["y"]].aggregation in AGGREGATIONS else "sum"
        raw = {k: v for k, v in raw.items() if v not in (None, "")}
        if chart_title:
            raw["title"], raw["title_custom"] = chart_title, True
        y, agg = raw.get("y"), raw.get("agg", "sum")
        if y and agg not in ("count", "nunique") and c.get("type") not in ("scatter", "map", "box", "histogram"):
            problem = agg_conflict(y, agg)
            if problem:
                errors.append(f"Chart {i + 1}: {problem}")
                continue
        for col in (raw.get("x"), raw.get("color"), raw.get("x2")):
            if col and profile.columns.get(col) and profile.columns[col].role == HIDDEN:
                errors.append(f"Chart {i + 1}: '{col}' was set to be left out of the dashboard.")
                break
        else:
            try:
                spec = validate_chart_spec(raw, profile)
            except ChartConfigError as exc:
                errors.append(f"Chart {i + 1} ({chart_title or c.get('type')}): {exc}")
                continue
            spec["description"] = (why + " " if why else "") + describe_chart(spec, profile)
            spec["auto"] = False
            charts.append(spec)

    if not charts:
        errors.append("The plan has no valid charts.")
    candidates = set(filter_candidates(profile))
    filters = [f for f in (plan.get("filters") or []) if f in candidates][:5]
    dropped = [f for f in (plan.get("filters") or []) if f not in candidates]
    if dropped:
        warnings.append(f"Filters that can't be used were skipped: {', '.join(map(str, dropped[:5]))}.")
    clean = {"title": title, "summary": summary, "business_questions": [q for q in questions if q],
             "kpis": kpis, "charts": charts, "filters": filters or default_filter_columns(profile)}
    return clean, errors, warnings


def build_dashboard_config(clean_plan: dict, profile: DatasetProfile, df, dataset_id: str, dataset_name: str,
                           sheet: str | None, header_row: int | None, memory) -> dict:
    """Turn a validated plan into a normal dashboard config (same format as every other dashboard)."""
    from .pipeline import column_notes, interpretation_overrides

    cfg = build_default_dashboard(profile, df, dataset_id, dataset_name, sheet or "CSV data", header_row)
    renames = memory.renames()
    cfg.update(
        name=clean_plan["title"][:120] or cfg["name"],
        kpis=clean_plan["kpis"], charts=clean_plan["charts"], filter_columns=clean_plan["filters"],
        cleaning_ops=list(memory.applied_ops), renames=renames,
        column_overrides=interpretation_overrides(memory.interpretations, renames),
        column_notes=column_notes(memory.interpretations, renames),
        analysis={"dataset_id": dataset_id, "sheet": sheet, "objective": memory.objective[:300],
                  "questions": clean_plan.get("business_questions", []), "summary": clean_plan.get("summary", "")},
    )
    return cfg


__all__ = ["OBJECTIVES", "PlanError", "build_dashboard_config", "classify_objective", "suggest_plan", "validate_plan", "DATE"]
