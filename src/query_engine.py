"""Natural-language questions answered with deterministic pandas calculations.

Architecture (designed so an LLM can be plugged in later without touching the maths):

    question ──► Interpreter ──► QueryPlan (validated, JSON-able) ──► execute_plan() ──► QueryResult

``RuleBasedInterpreter`` is the built-in, offline interpreter. A future LLM-backed
interpreter would only need to emit a ``QueryPlan``; the plan is validated against
the dataset profile and executed by the same pandas code, so the LLM never produces
the numbers itself.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from typing import Protocol

import numpy as np
import pandas as pd

from .chart_generator import AGGREGATIONS, agg_label
from .data_profiler import DATE, DIMENSION, IDENTIFIER, METRIC, TEXT, DatasetProfile
from .formatting import format_value
from .insights import generate_insights, outlier_insight, period_label, period_series, trend_insight

INTENTS = {
    "missing", "duplicates", "summary", "outliers", "trend", "correlation", "period_rank",
    "rank", "group_agg", "trends_outliers", "overall_agg", "count_rows", "count_distinct", "columns", "extreme_row",
    "share", "growth", "pivot", "compare",
}
_NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
              "nine": 9, "ten": 10, "fifteen": 15, "twenty": 20}
_STOP = {"the", "a", "an", "of", "by", "for", "per", "in", "on", "and", "or", "to", "is", "are", "what", "which",
         "who", "how", "many", "much", "each", "with", "had", "has", "have", "was", "were", "me", "show", "give",
         "list", "top", "bottom", "highest", "lowest", "most", "least", "total", "sum", "average", "avg", "mean",
         "median", "value", "values", "amount", "number", "count", "data", "dataset", "column", "columns", "all",
         "do", "does", "did", "there", "this", "that", "it", "be", "across", "between", "max", "min", "maximum",
         "minimum", "biggest", "largest", "smallest", "best", "worst", "name", "date", "type", "id"}
_TIME_WORDS = {"day": "D", "daily": "D", "date": "D", "week": "W", "weekly": "W", "month": "M", "monthly": "M",
               "quarter": "Q", "quarterly": "Q", "year": "Y", "yearly": "Y", "annual": "Y", "annually": "Y"}


@dataclass
class QueryPlan:
    intent: str
    metric: str | None = None
    metric2: str | None = None
    group: str | None = None
    agg: str = "sum"
    n: int = 5
    order: str = "desc"
    time_grain: str | None = None
    date_col: str | None = None
    filters: dict[str, list[str]] = field(default_factory=dict)
    assumptions: list[str] = field(default_factory=list)
    group2: str | None = None  # second category for two-way tables (pivot)
    values: list[str] = field(default_factory=list)  # category values to compare (compare)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class QueryResult:
    ok: bool
    answer: str
    table: pd.DataFrame | None = None
    chart: dict | None = None  # {"kind": "bar"|"line", "x": col, "y": col, "title": str}
    method: str = ""
    plan: dict | None = None


class Interpreter(Protocol):
    def interpret(self, question: str, df: pd.DataFrame, profile: DatasetProfile) -> QueryPlan | QueryResult: ...


class NeedsClarification(Exception):
    pass


# --------------------------------------------------------------------------- text helpers

def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9%]+", " ", text.lower()).strip()


def _stem(word: str) -> str:
    for suf in ("ies", "es", "s"):
        if word.endswith(suf) and len(word) > len(suf) + 2:
            return word[: -len(suf)] + ("y" if suf == "ies" else "")
    return word


def _stems(text: str) -> list[str]:
    return [_stem(w) for w in _norm(text).split()]


def _contains_seq(hay: list[str], needle: list[str]) -> int:
    """Index of the first occurrence of ``needle`` in ``hay`` or -1."""
    k = len(needle)
    for i in range(len(hay) - k + 1):
        if hay[i:i + k] == needle:
            return i
    return -1


def find_columns(question: str, profile: DatasetProfile) -> list[tuple[str, int, float]]:
    """Columns mentioned in the question as (name, position, score), best first, non-overlapping."""
    q = _stems(question)
    candidates: list[tuple[float, int, int, str]] = []  # score, pos, length, name
    # Optional aliases (confirmed meanings / readable expansions, e.g. REV_NET → "net revenue") set by the analyst.
    aliases: dict[str, list[str]] = profile.__dict__.get("_aliases") or {}
    all_tokens: dict[str, int] = {}
    for col in profile.columns:
        for t in set(_stems(col)) | {t for a in aliases.get(col, []) for t in _stems(a)}:
            all_tokens[t] = all_tokens.get(t, 0) + 1
    for col in profile.columns:
        toks = _stems(col)
        if not toks:
            continue
        pos = _contains_seq(q, toks)
        if pos >= 0:
            candidates.append((10.0 + len(toks), pos, len(toks), col))
            continue
        alias_hit = None
        for alias in aliases.get(col, []):
            atoks = [t for t in _stems(alias) if t not in _STOP or len(_stems(alias)) == 1]
            if atoks and all(t in q for t in atoks):  # any word order: "net revenue" ↔ "revenue net"
                idx = [q.index(t) for t in atoks]
                cand = (9.0 + len(atoks), min(idx), max(idx) - min(idx) + 1, col)
                alias_hit = max(alias_hit, cand) if alias_hit else cand
        if alias_hit:
            candidates.append(alias_hit)
            continue
        toks = toks + [t for a in aliases.get(col, []) for t in _stems(a)]
        # Partial: a distinctive token of the column name appears in the question.
        best = None
        for t in toks:
            if len(t) >= 3 and t not in _STOP and all_tokens.get(t, 0) == 1 and t in q:
                best = (5.0 + len(t) / 10, q.index(t), 1, col)
                break
        if best is None:
            # Fuzzy single-token match for typos (e.g. "revenu" -> "revenue").
            for t in toks:
                if len(t) >= 5 and t not in _STOP and all_tokens.get(t, 0) == 1:
                    for i, w in enumerate(q):
                        if len(w) >= 4 and SequenceMatcher(None, t, w).ratio() >= 0.85:
                            best = (3.0, i, 1, col)
                            break
                if best:
                    break
        if best:
            candidates.append(best)
    candidates.sort(key=lambda c: (-c[0], c[1]))
    used: set[int] = set()
    out = []
    for score, pos, length, col in candidates:
        span = set(range(pos, pos + length))
        if span & used:
            continue
        used |= span
        out.append((col, pos, score))
    return out


def _find_value_filters(question: str, df: pd.DataFrame, profile: DatasetProfile, mentioned: set[str]) -> dict[str, list[str]]:
    """Category values quoted or named in the question (e.g. 'in the North region')."""
    qn = f" {_norm(question)} "
    quoted = re.findall(r"['\"“‘]([^'\"”’]+)['\"”’]", question)
    found: dict[str, list[str]] = {}
    col_words = {w for c in profile.columns for w in _norm(c).split()}
    for col in profile.dimension_columns:
        cp = profile.columns[col]
        if cp.unique > 300:
            continue
        for val in df[col].dropna().astype(str).unique():
            v = _norm(val)
            if not v or len(v) < 2 or v.isdigit():
                continue
            if any(q.strip().lower() == val.strip().lower() for q in quoted) or (
                len(v) >= 3 and f" {v} " in qn and v not in col_words and v not in _STOP
            ):
                found.setdefault(col, []).append(val)
    # Drop ambiguous values that matched in several columns.
    return {c: v for c, v in found.items() if len(v) <= 5}


def _extract_n(question: str) -> int | None:
    q = question.lower()
    m = re.search(r"\b(?:top|bottom|first|last|best|worst|highest|lowest|largest|smallest)\s+(\d{1,3}|" + "|".join(_NUM_WORDS) + r")\b", q)
    if not m:
        m = re.search(r"\b(\d{1,3}|" + "|".join(_NUM_WORDS) + r")\s+(?:most|least|highest|lowest|top|best|worst|largest|biggest|smallest)\b", q)
    if m:
        tok = m.group(1)
        return int(tok) if tok.isdigit() else _NUM_WORDS[tok]
    return None


def _detect_agg(q: str) -> str | None:
    if re.search(r"\b(average|avg|mean)\b", q):
        return "mean"
    if re.search(r"\bmedian\b", q):
        return "median"
    if re.search(r"\b(total|sum|overall)\b", q):
        return "sum"
    if re.search(r"\b(distinct|unique|different)\b", q):
        return "nunique"
    if re.search(r"\b(how many|number of|count)\b", q):
        return "count"
    return None


# --------------------------------------------------------------------------- interpreter

class RuleBasedInterpreter:
    """Pattern-based interpreter for common analytical questions."""

    def interpret(self, question: str, df: pd.DataFrame, profile: DatasetProfile) -> QueryPlan:
        q = " " + question.lower().strip() + " "
        if len(question.strip()) < 3:
            raise NeedsClarification("Please type a question about the data.")
        mentions = find_columns(question, profile)
        names = [m[0] for m in mentions]
        metrics = [c for c in names if profile.columns[c].role == METRIC]
        groups = [c for c in names if profile.columns[c].role in (DIMENSION, IDENTIFIER)]
        dates_mentioned = [c for c in names if profile.columns[c].role == DATE]
        filters = _find_value_filters(question, df, profile, set(names))
        groups = [g for g in groups if g not in filters] or groups
        agg = _detect_agg(q)
        n = _extract_n(question)
        order = "asc" if re.search(r"\b(lowest|least|bottom|smallest|worst|fewest|minimum|min)\b", q) else "desc"
        plan = QueryPlan(intent="", filters=filters, n=n or 5, order=order)
        time_word = next((w for w in re.findall(r"[a-z]+", q) if w in _TIME_WORDS), None)
        date_col = dates_mentioned[0] if dates_mentioned else (profile.date_columns[0] if profile.date_columns else None)

        if re.search(r"\b(what|which|list|show)\b.*\bcolumns?\b|\bschema\b|\bfields\b", q) and not metrics and not re.search(r"missing|null|blank", q):
            plan.intent = "columns"
            return plan
        if re.search(r"\b(missing|null|nulls|blank|blanks|empty|nan|incomplete)\b", q):
            plan.intent = "missing"
            return plan
        if re.search(r"\bduplicat", q):
            plan.intent = "duplicates"
            return plan
        if re.search(r"summar|key findings|overview|insight|tell me about|describe the data|what stands out", q):
            plan.intent = "summary"
            return plan
        if re.search(r"outlier|anomal|unusual|extreme values", q) and re.search(r"\btrends?\b", q) and date_col:
            plan.intent, plan.metric, plan.date_col = "trends_outliers", metrics[0] if metrics else None, date_col
            return plan
        if re.search(r"outlier|anomal|unusual|extreme values", q):
            plan.intent, plan.metric = "outliers", metrics[0] if metrics else None
            return plan
        if re.search(r"correlat|relationship between|related to|relationship of", q):
            if len(metrics) < 2:
                raise NeedsClarification("Name the two numeric columns to compare, e.g. “correlation between Clicks and Conversions”.")
            plan.intent, plan.metric, plan.metric2 = "correlation", metrics[0], metrics[1]
            return plan
        if re.search(r"\b(share|proportion|contribut\w*|percent(age)? of (the )?total|what percent)\b|% of", q) and groups:
            plan.intent, plan.group = "share", groups[0]
            plan.metric = self._pick_metric(metrics, profile, plan, allow_default=True)
            plan.agg = "sum" if plan.metric else "count"
            return plan
        if re.search(r"\b(compare|versus|vs\.?)\b", q) and any(len(v) >= 2 for v in filters.values()):
            col = next(c for c, v in filters.items() if len(v) >= 2)
            plan.intent, plan.group, plan.values = "compare", col, filters.pop(col)
            plan.metric = self._pick_metric(metrics, profile, plan, allow_default=True)
            plan.agg = (agg if agg in ("sum", "mean", "median", "count") else None) or (
                profile.columns[plan.metric].aggregation if plan.metric else "count")
            return plan
        if re.search(r"\b(growth rate|grew|month[- ]over[- ]month|year[- ]over[- ]year|mom|yoy|last month|"
                     r"previous (month|quarter|year)|vs\.? last|compared (to|with) (the )?(last|previous|prior))\b", q):
            if not date_col:
                raise NeedsClarification("This dataset has no date column, so growth can't be calculated.")
            plan.intent, plan.date_col = "growth", date_col
            plan.time_grain = _TIME_WORDS.get(time_word or "", None) or ("Y" if re.search(r"year|yoy", q) else "M")
            plan.metric = self._pick_metric(metrics, profile, plan, allow_default=True)
            plan.agg = (agg if agg in ("sum", "mean", "median", "count") else None) or (
                profile.columns[plan.metric].aggregation if plan.metric else "count")
            plan.group = groups[0] if groups else None
            return plan
        if len(groups) >= 2 and not re.search(r"\b(top|bottom|highest|lowest|most|least|rank)\b", q) \
                and re.search(r"\b(by|per|across)\b.*\b(and|by|per)\b", q):
            plan.intent, plan.group, plan.group2 = "pivot", groups[0], groups[1]
            plan.metric = self._pick_metric(metrics, profile, plan, allow_default=True)
            plan.agg = (agg if agg in ("sum", "mean", "median", "count") else None) or (
                profile.columns[plan.metric].aggregation if plan.metric else "count")
            return plan
        if re.search(r"\btrends?\b|over time|grow|growth|declin|increas|decreas|change over", q):
            if not date_col:
                raise NeedsClarification("This dataset has no date column, so trends over time cannot be calculated.")
            plan.intent, plan.date_col = "trend", date_col
            plan.metric = self._pick_metric(metrics, profile, plan, allow_default=True)
            plan.agg = (agg if agg in AGGREGATIONS and agg not in ("nunique",) else None) or (profile.columns[plan.metric].aggregation if plan.metric else "count")
            return plan

        ranking = re.search(r"\b(top|bottom|highest|lowest|most|least|best|worst|largest|biggest|smallest|fewest|max|maximum|min|minimum|rank|ranking)\b", q)
        if ranking and time_word and date_col and not groups and re.search(r"\b(which|what)\b", q):
            plan.intent, plan.date_col, plan.time_grain = "period_rank", date_col, _TIME_WORDS[time_word]
            if plan.time_grain == "D" and time_word == "date":
                plan.time_grain = "D"
            plan.metric = self._pick_metric(metrics, profile, plan, allow_default=agg == "count" or bool(re.search(r"\b(most|fewest)\b.*\b(records|rows|orders|transactions|entries)\b", q)))
            plan.agg = agg if agg in ("sum", "mean", "median", "count") else (profile.columns[plan.metric].aggregation if plan.metric else "count")
            if plan.metric is None:
                plan.agg = "count"
            plan.n = n or 1
            return plan

        if re.search(r"\bhow many\b", q) and not metrics:
            if groups and not re.search(r"\b(rows|records|entries)\b", q):
                plan.intent, plan.group, plan.agg = "count_distinct", groups[0], "nunique"
                return plan
            if not groups:
                plan.intent, plan.agg = "count_rows", "count"
                return plan

        group = groups[0] if groups else None
        if group is None and time_word and date_col and re.search(r"\b(by|per|each|every)\b", q):
            plan.date_col, plan.time_grain = date_col, _TIME_WORDS[time_word]
        if ranking and (group or plan.time_grain):
            plan.intent = "rank" if group else "period_rank"
            plan.group = group
            plan.metric = self._pick_metric(metrics, profile, plan, allow_default=agg == "count")
            plan.agg = agg if agg in ("sum", "mean", "median", "count", "min", "max") else (
                profile.columns[plan.metric].aggregation if plan.metric else "count")
            if plan.metric is None:
                plan.agg = "count"
            plan.n = n or (1 if re.search(r"\bwhich\b|\bwhat\b", q) and not re.search(r"\b(products|items|categories)\b", q) and not n else 5)
            if plan.intent == "rank" and n is None and re.search(r"\b(which|what)\b", q):
                plan.n = 1 if not re.search(r"\b(top|bottom)\b", q) else 5
            return plan
        if group or plan.time_grain:
            if re.search(r"\b(by|per|each|every|for|across|breakdown|split|group)\b", q) or agg:
                plan.intent, plan.group = "group_agg", group
                plan.metric = self._pick_metric(metrics, profile, plan, allow_default=agg == "count")
                if plan.metric is None:
                    plan.agg = "count"
                else:
                    plan.agg = agg if agg in ("sum", "mean", "median", "min", "max", "count") else profile.columns[plan.metric].aggregation
                    if agg is None:
                        plan.assumptions.append(f"Used the default aggregation for '{plan.metric}' ({AGGREGATIONS[plan.agg].lower()}).")
                return plan
        if metrics:
            if ranking and not agg:
                plan.intent, plan.metric = "extreme_row", metrics[0]
                return plan
            plan.intent, plan.metric = "overall_agg", metrics[0]
            if ranking and re.search(r"\b(max|maximum|highest|largest|biggest)\b", q):
                plan.agg = "max"
            elif ranking:
                plan.agg = "min"
            else:
                plan.agg = agg if agg in AGGREGATIONS else profile.columns[metrics[0]].aggregation
                if agg is None:
                    plan.assumptions.append(f"Used the default aggregation for '{metrics[0]}' ({AGGREGATIONS[plan.agg].lower()}).")
                if plan.agg == "nunique":
                    plan.intent, plan.group = "count_distinct", metrics[0]
            return plan
        if agg == "count" or re.search(r"\b(rows|records)\b", q):
            plan.intent, plan.agg = "count_rows", "count"
            return plan
        raise NeedsClarification(_rephrase_message(profile))

    @staticmethod
    def _pick_metric(metrics: list[str], profile: DatasetProfile, plan: QueryPlan, allow_default: bool) -> str | None:
        if metrics:
            return metrics[0]
        if allow_default:
            return None
        available = [m for m in profile.metric_columns if not profile.columns[m].constant]
        if len(available) == 1:
            plan.assumptions.append(f"No measure was named, so '{available[0]}' (the only numeric measure) was used.")
            return available[0]
        if not available:
            plan.assumptions.append("The dataset has no numeric measures, so rows were counted.")
            return None
        raise NeedsClarification(
            "Which measure should I use? Mention one of: " + ", ".join(f"'{m}'" for m in available[:8])
            + ". For example: “top 5 " + (profile.chartable_dimensions[0] if profile.chartable_dimensions else "categories")
            + f" by {available[0]}”."
        )


def _rephrase_message(profile: DatasetProfile) -> str:
    m = profile.metric_columns[:1]
    d = profile.chartable_dimensions[:1]
    examples = []
    if m and d:
        examples.append(f"“Total {m[0]} by {d[0]}”")
        examples.append(f"“Top 5 {d[0]} by {m[0]}”")
    if m and profile.date_columns:
        examples.append(f"“Which month had the highest {m[0]}?”")
    examples += ["“Which columns have the most missing values?”", "“Summarize the key findings”"]
    return ("I couldn't interpret that question reliably, so I haven't guessed an answer. "
            "Try mentioning a column name and what to calculate, for example: " + "; ".join(examples[:4]) + ".")


# --------------------------------------------------------------------------- executor

def _validate_plan(plan: QueryPlan, profile: DatasetProfile) -> None:
    if plan.intent not in INTENTS:
        raise NeedsClarification("Unsupported question type.")
    for col in (plan.metric, plan.metric2):
        if col is not None and (col not in profile.columns or profile.columns[col].role != METRIC):
            raise NeedsClarification(f"'{col}' is not a numeric column.")
    for col in (plan.group, plan.group2):
        if col is not None and col not in profile.columns:
            raise NeedsClarification(f"Unknown column '{col}'.")
    if plan.intent in ("share", "pivot", "compare") and plan.group is None:
        raise NeedsClarification("Name the category to split by, e.g. “share of sales by region”.")
    if plan.intent == "pivot" and (plan.group2 is None or plan.group2 == plan.group):
        raise NeedsClarification("A two-way table needs two different categories, e.g. “sales by region and product”.")
    if plan.intent == "compare" and len(plan.values) < 2:
        raise NeedsClarification(f"Name at least two values of '{plan.group}' to compare.")
    if plan.intent == "growth" and plan.date_col is None:
        raise NeedsClarification("Growth needs a date column, and this dataset has none.")
    if plan.date_col is not None and (plan.date_col not in profile.columns or profile.columns[plan.date_col].role != DATE):
        raise NeedsClarification(f"'{plan.date_col}' is not a date column.")
    if plan.agg not in AGGREGATIONS:
        raise NeedsClarification("Unsupported calculation.")
    plan.n = max(1, min(50, int(plan.n)))
    for col in plan.filters:
        if col not in profile.columns:
            raise NeedsClarification(f"Unknown column '{col}'.")


def _f(v, col: str | None = None, profile: DatasetProfile | None = None, agg: str = "sum") -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return format_value(v, decimals=2)


def execute_plan(plan: QueryPlan, df: pd.DataFrame, profile: DatasetProfile) -> QueryResult:
    _validate_plan(plan, profile)
    data = df
    filter_note = ""
    if plan.filters:
        mask = pd.Series(True, index=df.index)
        for col, vals in plan.filters.items():
            mask &= df[col].astype(str).isin(vals)
        data = df[mask]
        filter_note = " Filtered to " + "; ".join(f"{c} = {', '.join(v)}" for c, v in plan.filters.items()) + f" ({len(data):,} rows)."
    if data.empty:
        return QueryResult(False, "No rows match that question's conditions.", plan=plan.to_dict())
    notes = (" " + " ".join(plan.assumptions)) if plan.assumptions else ""
    measure = agg_label(plan.agg, plan.metric)
    i = plan.intent

    if i == "columns":
        t = profile.summary_frame()[["Column", "Role", "Missing %", "Unique"]]
        return QueryResult(True, f"The dataset has {profile.n_cols} columns.", table=t, method="Read from the data profile.")

    if i == "missing":
        miss = data.isna().sum()
        t = pd.DataFrame({"Column": miss.index, "Missing": miss.values, "Missing %": (miss.values / len(data) * 100).round(1)})
        t = t[t["Missing"] > 0].sort_values("Missing", ascending=False)
        if t.empty:
            return QueryResult(True, f"No column has missing values ({len(data):,} rows checked).{filter_note}", method="data.isna().sum()")
        top = t.iloc[0]
        ans = f"**{top['Column']}** has the most missing values: {int(top['Missing']):,} of {len(data):,} rows ({top['Missing %']}%). {len(t)} column(s) have gaps.{filter_note}"
        return QueryResult(True, ans, table=t.reset_index(drop=True), method="Counted empty cells per column.",
                           chart={"kind": "hbar", "x": "Column", "y": "Missing", "title": "Missing values by column"}, plan=plan.to_dict())

    if i == "duplicates":
        dup = int(data.duplicated().sum())
        return QueryResult(True, f"{dup:,} row(s) are exact duplicates of an earlier row ({dup / len(data):.1%} of {len(data):,}).{filter_note}",
                           method="data.duplicated().sum()", plan=plan.to_dict())

    if i == "summary":
        ins = generate_insights(data, profile)
        text = "\n".join(f"- **{x.title}:** {x.text}" for x in ins)
        return QueryResult(True, text + (f"\n\n{filter_note.strip()}" if filter_note else ""), method="Deterministic insight rules over the data.", plan=plan.to_dict())

    if i == "outliers":
        cols = [plan.metric] if plan.metric else profile.metric_columns[:5]
        found = [o for o in (outlier_insight(data, c) for c in cols) if o]
        if not found:
            return QueryResult(True, "No outliers were found using the 1.5 × IQR rule" + (f" in '{plan.metric}'." if plan.metric else " in the main measures.") + filter_note,
                               method="1.5 × interquartile range rule.", plan=plan.to_dict())
        return QueryResult(True, "\n".join(f"- {o.text}" for o in found) + filter_note, method="1.5 × interquartile range rule.", plan=plan.to_dict())

    if i == "trends_outliers":
        cols = [plan.metric] if plan.metric else [m for m in profile.metric_columns if not profile.columns[m].constant][:3]
        lines = []
        for c in cols or [None]:
            t = trend_insight(data, profile, c, plan.date_col)
            if t:
                lines.append(f"- **{t.title}:** {t.text}")
        for c in cols:
            o = outlier_insight(data, c) if c else None
            if o:
                lines.append(f"- **{o.title}:** {o.text}")
        if not lines:
            return QueryResult(True, "No clear trends or outliers were found." + filter_note, plan=plan.to_dict())
        return QueryResult(True, "\n".join(lines) + filter_note, method="Linear trend over periods and the 1.5 × IQR outlier rule.", plan=plan.to_dict())

    if i == "correlation":
        a, b = plan.metric, plan.metric2
        pair = data[[a, b]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(pair) < 3:
            return QueryResult(False, "Not enough rows with both values to measure a correlation.", plan=plan.to_dict())
        r = float(pair[a].corr(pair[b]))
        strength = "strong" if abs(r) >= 0.7 else "moderate" if abs(r) >= 0.4 else "weak"
        return QueryResult(True, f"The correlation between **{a}** and **{b}** is **r = {r:.2f}** ({strength}, {'positive' if r >= 0 else 'negative'}), based on {len(pair):,} rows.{filter_note}",
                           method="Pearson correlation (pandas Series.corr).", plan=plan.to_dict())

    if i == "trend":
        ins = trend_insight(data, profile, plan.metric, plan.date_col)
        s, grain, partial = period_series(data, plan.date_col, plan.metric, plan.agg if plan.metric else "count")
        t = pd.DataFrame({"Period": [period_label(p) for p in s.index], measure: s.values})
        if ins is None:
            return QueryResult(True, f"There are too few periods to describe a trend.{filter_note}", table=t, plan=plan.to_dict())
        return QueryResult(True, ins.text + filter_note + notes, table=t, method=f"{measure} grouped by period of '{plan.date_col}'.",
                           chart={"kind": "line", "x": "Period", "y": measure, "title": f"{measure} over time"}, plan=plan.to_dict())

    if i == "period_rank":
        s, _, _ = _by_period(data, plan)
        s = s.dropna()
        if s.empty:
            return QueryResult(False, "No values to rank.", plan=plan.to_dict())
        ranked = s.sort_values(ascending=plan.order == "asc")
        word = {"D": "day", "W": "week", "M": "month", "Q": "quarter", "Y": "year"}[plan.time_grain]
        best = ranked.index[0]
        adj = "lowest" if plan.order == "asc" else "highest"
        t = pd.DataFrame({word.title(): [period_label(p) for p in ranked.index[:max(plan.n, 10)]], measure: ranked.values[:max(plan.n, 10)]})
        ans = f"**{period_label(best)}** had the {adj} {measure.lower() if plan.agg == 'count' else measure}: **{_f(ranked.iloc[0])}**"
        if len(ranked) > 1:
            ans += f" (next: {period_label(ranked.index[1])} with {_f(ranked.iloc[1])})"
        ans += f". Compared across {len(s)} {word}s of '{plan.date_col}'.{filter_note}{notes}"
        chrono = s.sort_index()
        chart_t = pd.DataFrame({word.title(): [period_label(p) for p in chrono.index], measure: chrono.values})
        return QueryResult(True, ans, table=t, method=f"{measure} grouped by {word} of '{plan.date_col}'.",
                           chart={"kind": "bar", "x": word.title(), "y": measure, "title": f"{measure} by {word}", "data": chart_t}, plan=plan.to_dict())

    if i in ("rank", "group_agg"):
        if plan.group:
            keys = data[plan.group].map(lambda v: "(blank)" if pd.isna(v) else str(v))
            label = plan.group
            g = data.groupby(keys, observed=True)
        else:
            s, _, _ = _by_period(data, plan)
            word = {"D": "Day", "W": "Week", "M": "Month", "Q": "Quarter", "Y": "Year"}[plan.time_grain]
            t = pd.DataFrame({word: [period_label(p) for p in s.index], measure: s.values})
            return QueryResult(True, f"{measure} by {word.lower()} of '{plan.date_col}' across {len(s)} periods.{filter_note}{notes}", table=t,
                               chart={"kind": "line", "x": word, "y": measure, "title": f"{measure} by {word.lower()}"},
                               method=f"Grouped by {word.lower()} of '{plan.date_col}'.", plan=plan.to_dict())
        s = g.size().astype(float) if plan.agg == "count" or not plan.metric else g[plan.metric].agg(plan.agg)
        s = s.dropna()
        if s.empty:
            return QueryResult(False, f"No values of '{plan.metric}' to aggregate.", plan=plan.to_dict())
        ranked = s.sort_values(ascending=plan.order == "asc")
        t = ranked.rename(measure).reset_index().rename(columns={ranked.index.name or "index": label})
        if i == "rank":
            n = min(plan.n, len(ranked))
            adj = "lowest" if plan.order == "asc" else "highest"
            if n == 1:
                ans = f"**{ranked.index[0]}** has the {adj} {measure}: **{_f(ranked.iloc[0])}**"
                if len(ranked) > 1:
                    ans += f" (next: {ranked.index[1]} with {_f(ranked.iloc[1])})"
                ans += f". Compared across {len(ranked)} '{label}' values."
            else:
                items = ", ".join(f"{k} ({_f(v)})" for k, v in ranked.iloc[:n].items())
                ans = f"{'Bottom' if plan.order == 'asc' else 'Top'} {n} '{label}' by {measure}: {items}."
            t = t.head(max(n, 10))
        else:
            top = ranked.index[0]
            ans = f"{measure} for each of {len(ranked)} '{label}' values. Highest: **{top}** ({_f(ranked.iloc[0])}); lowest: **{ranked.index[-1]}** ({_f(ranked.iloc[-1])})."
            t = t.head(50)
        chart_data = t.head(min(plan.n, len(t))) if i == "rank" and plan.n > 1 else t.head(25)
        return QueryResult(True, ans + filter_note + notes, table=t, method=f"{measure} grouped by '{label}' with pandas groupby.",
                           chart={"kind": "hbar", "x": label, "y": measure, "title": f"{measure} by {label}", "data": chart_data}, plan=plan.to_dict())

    if i in ("share", "growth", "pivot", "compare"):
        return _analytic(i, plan, data, profile, measure, filter_note + notes)

    if i == "overall_agg":
        v = pd.to_numeric(data[plan.metric], errors="coerce")
        val = float(getattr(v.dropna(), plan.agg)()) if plan.agg != "nunique" else float(v.nunique())
        missing = int(v.isna().sum())
        ans = f"The {AGGREGATIONS[plan.agg].lower()} of **{plan.metric}** is **{_f(val)}** across {int(v.notna().sum()):,} rows"
        ans += f" ({missing:,} missing values excluded)." if missing else "."
        return QueryResult(True, ans + filter_note + notes, method=f"{plan.agg}() over '{plan.metric}'.", plan=plan.to_dict())

    if i == "extreme_row":
        v = pd.to_numeric(data[plan.metric], errors="coerce")
        if v.notna().sum() == 0:
            return QueryResult(False, f"'{plan.metric}' has no values.", plan=plan.to_dict())
        idx = v.idxmin() if plan.order == "asc" else v.idxmax()
        row = data.loc[[idx]]
        adj = "lowest" if plan.order == "asc" else "highest"
        return QueryResult(True, f"The {adj} '{plan.metric}' is **{_f(v[idx])}** (row shown below).{filter_note}", table=row,
                           method=f"Row with the {adj} '{plan.metric}'.", plan=plan.to_dict())

    if i == "count_rows":
        return QueryResult(True, f"There are **{len(data):,}** rows.{filter_note}", method="len(data)", plan=plan.to_dict())

    if i == "count_distinct":
        col = plan.group
        k = int(data[col].nunique(dropna=True))
        t = data[col].astype(str).value_counts().head(20).rename_axis(col).reset_index(name="Rows")
        return QueryResult(True, f"There are **{k:,}** distinct '{col}' values.{filter_note}", table=t, method=f"nunique() of '{col}'.", plan=plan.to_dict())

    return QueryResult(False, _rephrase_message(profile), plan=plan.to_dict())


def _keys(data: pd.DataFrame, col: str) -> pd.Series:
    return data[col].map(lambda v: "(blank)" if pd.isna(v) else str(v))


def _grouped(data: pd.DataFrame, by, plan: QueryPlan) -> pd.Series:
    g = data.groupby(by, observed=True)
    return g.size().astype(float) if plan.agg == "count" or not plan.metric else g[plan.metric].agg(plan.agg)


def _pct(a: float, b: float) -> float | None:
    return None if not b else (a - b) / abs(b) * 100


_GRAIN_WORD = {"D": "day", "W": "week", "M": "month", "Q": "quarter", "Y": "year"}
_PER_YEAR = {"M": 12, "Q": 4, "W": 52}


def _analytic(i: str, plan: QueryPlan, data: pd.DataFrame, profile: DatasetProfile, measure: str, tail: str) -> QueryResult:
    """Share of total, growth vs the previous period, two-way tables and A-vs-B comparisons."""
    if i == "share":
        if plan.metric and profile.columns[plan.metric].aggregation == "mean" and plan.agg != "count":
            raise NeedsClarification(f"'{plan.metric}' is a rate or average, so its values don't add up to a total. "
                                     "Ask for its average by group instead.")
        if plan.agg not in ("sum", "count"):
            plan.agg = "sum" if plan.metric else "count"
            measure = agg_label(plan.agg, plan.metric)
        s = _grouped(data, _keys(data, plan.group), plan).dropna().sort_values(ascending=False)
        total = float(s.sum())
        if s.empty or total == 0:
            return QueryResult(False, "There is nothing to divide into shares.", plan=plan.to_dict())
        pct = s / total * 100
        t = pd.DataFrame({plan.group: s.index, measure: s.values, "Share %": pct.round(1).values}).head(50)
        ans = (f"**{s.index[0]}** has the largest share of {measure}: **{pct.iloc[0]:.1f}%** "
               f"({_f(s.iloc[0])} of {_f(total)}).")
        if len(s) > 3:
            k80 = int((pct.cumsum() < 80).sum()) + 1
            ans += f" The top {k80} of {len(s)} '{plan.group}' values make up 80% of the total."
        return QueryResult(True, ans + tail, table=t, method=f"{measure} per '{plan.group}' divided by the overall total.",
                           chart={"kind": "hbar", "x": plan.group, "y": "Share %", "title": f"Share of {measure} by {plan.group}",
                                  "data": t.head(15)}, plan=plan.to_dict())

    if i == "growth":
        grain = plan.time_grain or "M"
        word = _GRAIN_WORD[grain]
        d = data.dropna(subset=[plan.date_col])
        per = d[plan.date_col].dt.to_period(grain)
        s = _grouped(d, per, plan).sort_index()
        if len(s) < 2:
            return QueryResult(False, f"There is only one {word} of data, so growth can't be measured.", plan=plan.to_dict())
        last, prev = s.index[-1], s.index[-2]
        ch = _pct(float(s[last]), float(s[prev]))
        diff = float(s[last] - s[prev])
        direction = "up" if diff > 0 else "down" if diff < 0 else "unchanged"
        ans = (f"{measure} in **{period_label(last)}** was **{_f(s[last])}**, {direction} "
               + (f"**{abs(ch):.1f}%** " if ch is not None and diff else "")
               + f"from {period_label(prev)} ({_f(s[prev])}).")
        if grain in _PER_YEAR and (last - _PER_YEAR[grain]) in s.index:
            ly = last - _PER_YEAR[grain]
            yoy = _pct(float(s[last]), float(s[ly]))
            if yoy is not None:
                ans += f" Compared with the same {word} last year ({period_label(ly)}): {yoy:+.1f}%."
        if d[plan.date_col].max() < last.end_time.normalize():
            ans += (f" Note: the data ends on {d[plan.date_col].max():%Y-%m-%d}, so {period_label(last)} may be "
                    "incomplete.")
        t = pd.DataFrame({word.title(): [period_label(p) for p in s.index], measure: s.values,
                          "Change": s.diff().values, "Change %": (s.pct_change() * 100).round(1).values})
        if plan.group:
            g = _grouped(d, [per, _keys(d, plan.group)], plan).unstack(fill_value=0)
            if prev in g.index and last in g.index:
                mv = (g.loc[last] - g.loc[prev]).sort_values()
                ups = [f"{k} ({_f(v)})" for k, v in mv[::-1].items() if v > 0][:3]
                downs = [f"{k} ({_f(v)})" for k, v in mv.items() if v < 0][:3]
                if ups:
                    ans += f" Biggest increases by {plan.group}: {', '.join(ups)}."
                if downs:
                    ans += f" Biggest drops: {', '.join(downs)}."
        return QueryResult(True, ans + tail, table=t.tail(24), method=f"{measure} per {word} of '{plan.date_col}', "
                           "latest period compared with the one before.",
                           chart={"kind": "line", "x": word.title(), "y": measure, "title": f"{measure} by {word}"},
                           plan=plan.to_dict())

    if i == "pivot":
        k1, k2 = _keys(data, plan.group), _keys(data, plan.group2)
        top1 = _grouped(data, k1, plan).sort_values(ascending=False).index[:12]
        top2 = _grouped(data, k2, plan).sort_values(ascending=False).index[:8]
        sub = data[k1.isin(top1) & k2.isin(top2)]
        p = _grouped(sub, [_keys(sub, plan.group), _keys(sub, plan.group2)], plan).unstack()
        p = p.reindex(index=[x for x in top1 if x in p.index], columns=[x for x in top2 if x in p.columns])
        if p.empty:
            return QueryResult(False, "No combinations to show.", plan=plan.to_dict())
        stacked = p.stack().dropna()
        (a, b), v = stacked.idxmax(), stacked.max()
        ans = (f"{measure} by **{plan.group}** and **{plan.group2}**. The largest combination is **{a} × {b}** "
               f"({_f(v)}).")
        if len(top1) == 12 or len(top2) == 8:
            ans += " Showing the biggest groups only."
        p.index.name, p.columns.name = plan.group, None
        t = p.round(2).reset_index()
        return QueryResult(True, ans + tail, table=t, method=f"{measure} grouped by '{plan.group}' and '{plan.group2}' "
                           "(a two-way table).", plan=plan.to_dict())

    # compare
    keys = _keys(data, plan.group)
    wanted = {v.casefold() for v in plan.values}
    found = {k for k in keys.unique() if k.casefold() in wanted}
    if len(found) < 2:
        return QueryResult(False, f"I couldn't find at least two of {', '.join(plan.values)} in '{plan.group}'.",
                           plan=plan.to_dict())
    sub = data[keys.isin(found)]
    s = _grouped(sub, _keys(sub, plan.group), plan).sort_values(ascending=False)
    rows = _keys(sub, plan.group).value_counts()
    t = pd.DataFrame({plan.group: s.index, measure: s.values, "Rows": [int(rows.get(k, 0)) for k in s.index]})
    a, b = s.index[0], s.index[1]
    ch = _pct(float(s[a]), float(s[b]))
    ans = f"**{a}**: {_f(s[a])} vs **{b}**: {_f(s[b])} — {a} is higher by {_f(s[a] - s[b])}"
    ans += f" ({ch:.1f}% more)." if ch is not None else "."
    missing = [v for v in plan.values if v.casefold() not in {f.casefold() for f in found}]
    if missing:
        ans += f" Not found: {', '.join(missing)}."
    return QueryResult(True, ans + tail, table=t, method=f"{measure} for each chosen '{plan.group}' value.",
                       chart={"kind": "hbar", "x": plan.group, "y": measure, "title": f"{measure}: comparison"},
                       plan=plan.to_dict())


def _by_period(data: pd.DataFrame, plan: QueryPlan) -> tuple[pd.Series, str, bool]:
    d = data.dropna(subset=[plan.date_col])
    periods = d[plan.date_col].dt.to_period(plan.time_grain)
    if plan.metric and plan.agg != "count":
        s = d.groupby(periods)[plan.metric].agg(plan.agg)
    else:
        s = d.groupby(periods).size().astype(float)
    return s.sort_index(), plan.time_grain, False


def answer_question(question: str, df: pd.DataFrame, profile: DatasetProfile, interpreter: Interpreter | None = None,
                    aliases: dict[str, list[str]] | None = None) -> QueryResult:
    """Interpret and answer ``question``. Never fabricates numbers; asks to rephrase when unsure.

    ``aliases`` maps column names to other words the user may use for them (e.g. confirmed meanings).
    """
    interpreter = interpreter or RuleBasedInterpreter()
    if aliases is not None:
        profile.__dict__["_aliases"] = {c: [a for a in v if a] for c, v in aliases.items() if c in profile.columns}
    try:
        plan = interpreter.interpret(question, df, profile)
        if isinstance(plan, QueryResult):
            return plan
        return execute_plan(plan, df, profile)
    except NeedsClarification as exc:
        return QueryResult(False, str(exc))
    except (KeyError, ValueError, TypeError) as exc:
        return QueryResult(False, f"That question couldn't be calculated on this data ({type(exc).__name__}). Try rephrasing it.")


SUGGESTED_QUESTIONS = [
    "Summarize the key findings",
    "Which columns have the most missing values?",
]


def suggested_questions(profile: DatasetProfile) -> list[str]:
    out = []
    m = [x for x in profile.metric_columns if not profile.columns[x].constant]
    d = profile.chartable_dimensions
    if m and d:
        out.append(f"Which {d[0]} has the highest {m[0]}?")
        out.append(f"Average {m[0]} by {d[0]}")
    if m and profile.date_columns:
        out.append(f"Which month had the highest {m[0]}?")
        out.append(f"What is the trend in {m[0]}?")
    if len(d) > 1 and m:
        out.append(f"Top 5 {d[1]} by {m[-1] if len(m) > 1 else m[0]}")
    if m:
        out.append(f"Are there outliers in {m[0]}?")
    return (out + SUGGESTED_QUESTIONS)[:7]
