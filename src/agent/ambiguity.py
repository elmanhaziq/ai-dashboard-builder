"""Find column names whose meaning can't be inferred reliably.

The detector only flags columns where the meaning is unclear *and* it matters: measures (which drive KPIs
and totals) get asked about first; clear names (``Order Date``, ``Region``) are never asked about.
"""
from __future__ import annotations

import re

from ..data_profiler import COORDINATE, DATE, DIMENSION, EMPTY, IDENTIFIER, METRIC, TEXT, DatasetProfile

# Common abbreviations in business spreadsheets → their usual meaning (used for *suggested* answers only;
# the user always confirms).
ABBREVIATIONS = {
    "rev": "revenue", "qty": "quantity", "sld": "sold", "br": "branch", "cd": "code", "dt": "date",
    "trx": "transaction", "txn": "transaction", "tx": "transaction", "amt": "amount", "disc": "discount",
    "cust": "customer", "prod": "product", "cat": "category", "dept": "department", "emp": "employee",
    "inv": "invoice", "uom": "unit of measure", "pct": "percent", "avg": "average", "yr": "year",
    "mth": "month", "mo": "month", "no": "number", "num": "number", "desc": "description", "grp": "group",
    "val": "value", "bal": "balance", "cnt": "count", "rcv": "received", "shp": "shipped", "sls": "sales",
    "tot": "total", "mgr": "manager", "loc": "location", "reg": "region", "st": "state", "pc": "pieces",
    "pcs": "pieces", "wk": "week", "ytd": "year to date", "mtd": "month to date", "gp": "gross profit",
    "cogs": "cost of goods sold", "sku": "stock keeping unit", "kpi": "key performance indicator",
    "rm": "ringgit", "usd": "US dollars", "sst": "sales and service tax", "gst": "goods and services tax",
    "net": "net", "gross": "gross", "ref": "reference", "addr": "address", "tel": "telephone", "id": "identifier",
    "acc": "account", "acct": "account", "bu": "business unit", "cc": "cost centre", "pos": "point of sale",
    "promo": "promotion", "mkt": "marketing", "ops": "operations", "hr": "human resources",
}

# Words that are clear on their own (lower-case, singular).
VOCABULARY = set("""
id date time day week month quarter year period region area zone state city country branch store outlet shop
location address postcode product item sku category subcategory group type class segment channel source medium
campaign customer client account user member employee staff rep representative salesperson manager team department
division company supplier vendor brand model status stage priority level grade gender age name title description
note comment remark sales sale revenue income turnover amount value price cost costs expense expenses spend budget
profit margin discount tax fee charge total subtotal net gross quantity unit units volume count number orders
order invoice transaction payment method currency rate ratio percent percentage share score rating rank target
actual forecast variance growth clicks impressions conversions leads visits sessions views ctr cpc cpm roi roas
stock inventory reorder hand warehouse shipped received returns return delivered delivery days hours duration
salary wage bonus headcount tenure hire start end created updated modified latitude longitude lat lon email phone %
code key reference sold purchased bought weight height bmi distance speed temperature
""".split())

GENERIC = {"value", "amount", "total", "number", "data", "field", "metric", "measure", "figure", "col", "column", "unnamed", "misc", "other"}


# Abbreviations so standard that confirming them would only annoy the user.
CONFIDENT = {"cogs", "sku", "ytd", "mtd", "kpi", "gst", "sst", "id", "no", "num", "pct", "avg", "uom", "pos", "rm", "usd",
             "hr", "roi", "ctr", "cpc", "cpm", "roas", "yr", "wk", "mth"}
_UNIT_RE = re.compile(r"\(\s*(rm|myr|usd|\$|€|£|%|kg|g|km|m|pcs|units?|hrs?|hours?|days?)\s*\)", re.IGNORECASE)


def tokens(name: str) -> list[str]:
    core = re.sub(r"\([^)]*\)", " ", name)  # "(RM)" or "(units)" is a unit, not part of the meaning
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", core)
    return [t for t in re.split(r"[^A-Za-z0-9%]+", spaced.lower()) if t]


def _stem(word: str) -> str:
    return word[:-1] if word.endswith("s") and len(word) > 3 else word


def _looks_like_code(t: str) -> bool:
    """Short codes (uag), letter+digit mixes (q3x), or vowel-less clusters (trx) — not ordinary words."""
    if t.isdigit() or t == "%":
        return False
    if any(ch.isdigit() for ch in t) and any(ch.isalpha() for ch in t):
        return True
    if len(t) <= 4:
        return True
    return not re.search(r"[aeiouy]", t)


def expand(name: str) -> tuple[str, list[str]]:
    """Best-effort reading of a column name. Returns (expansion, unknown tokens).

    Ordinary longer words ("shipping", "recognized") are assumed readable; only code-like tokens are unknown.
    """
    out, unknown = [], []
    for t in tokens(name):
        if t.isdigit() or t == "%":
            out.append(t)
        elif t in VOCABULARY or _stem(t) in VOCABULARY:
            out.append(t)
        elif t in ABBREVIATIONS:
            out.append(ABBREVIATIONS[t])
        else:
            out.append(t)
            if _looks_like_code(t):
                unknown.append(t)
    return " ".join(out), unknown


_LEADING = {"net", "gross", "total", "average", "unit", "monthly", "daily", "annual", "year to date", "month to date"}


def readable(expansion: str) -> str:
    """'revenue net' → 'Net revenue'; 'date transaction' → 'Transaction date'; 'quantity sold' → 'Quantity sold'."""
    words = expansion.split()
    if len(words) >= 2 and words[-1] in _LEADING:
        words = [words[-1]] + words[:-1]
    elif len(words) == 2 and words[0] in ("date", "code", "number", "identifier", "name"):
        words = [words[1], words[0]]
    text = " ".join(words)
    return text[:1].upper() + text[1:]


def assess_column(cp, currency: bool = False) -> dict | None:
    """Return an ambiguity record for a column, or None when its meaning is clear enough."""
    if cp.role in (EMPTY, COORDINATE):
        return None
    name = cp.name
    toks = tokens(name)
    expansion, unknown = expand(name)
    abbreviated = [t for t in toks if t in ABBREVIATIONS and t not in VOCABULARY and t not in CONFIDENT]
    currency = currency or bool(_UNIT_RE.search(name))
    generic = bool(toks) and all(t in GENERIC or t.isdigit() for t in toks)
    auto_named = bool(re.fullmatch(r"(column|unnamed)[ _:]*\d+", name.strip(), re.IGNORECASE))
    has_code_suffix = bool(re.search(r"[_\-]\d+$", name)) and bool(unknown)

    reasons = []
    if auto_named:
        reasons.append("The column had no header, so its name was generated automatically.")
    if unknown:
        reasons.append(f"'{name}' contains a term I can't interpret reliably ({', '.join(unknown[:3])}).")
    if generic and not auto_named:
        reasons.append(f"'{name}' is a very general name, so it isn't clear what is being measured.")
    # A percentage column with an otherwise readable name (e.g. "Disc %") is clear enough: it is averaged anyway.
    if abbreviated and not unknown and cp.role == METRIC and not cp.is_percent:
        reasons.append(f"'{name}' looks like an abbreviation for “{expansion}”, which I should confirm before using it in totals.")
    if not reasons:
        return None

    role = cp.role
    # Only ask when the meaning materially affects calculations or the dashboard.
    if role == METRIC:
        priority = 3 if (unknown or generic or auto_named) else 2
    elif role in (DIMENSION,) and (unknown or auto_named):
        priority = 1
    elif role in (DATE, IDENTIFIER, TEXT):
        return None  # dates, IDs and free text don't change any totals
    else:
        return None

    suggestions = []
    if expansion != " ".join(toks):  # only when at least one part was actually expanded (not "Uag 2")
        suggestions.append(readable(expansion))
    if role == METRIC:
        suggestions += ["A money amount (sales or revenue)", "A quantity / count of items", "A percentage or rate"]
    suggestions.append("Not sure — leave it out of the dashboard")
    unit_known = currency or cp.is_currency or cp.is_percent
    return {
        "column": name, "role": role, "priority": priority, "reasons": reasons, "expansion": expansion,
        "unknown_terms": unknown, "suggestions": list(dict.fromkeys(suggestions))[:5],
        "blocking": priority >= 2, "unit_known": unit_known,
    }


def find_ambiguous(profile: DatasetProfile, currency_columns: list[str] | None = None) -> list[dict]:
    cur = set(currency_columns or [])
    found = [r for r in (assess_column(cp, cp.name in cur) for cp in profile.columns.values()) if r]
    return sorted(found, key=lambda r: (-r["priority"], list(profile.columns).index(r["column"])))
