"""Dataset profiling: column roles, data quality and summary statistics.

The resulting :class:`DatasetProfile` is the shared internal representation that
the chart recommender, KPI builder, insights and query engine all consume.
"""
from __future__ import annotations

import copy
import re
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# Column roles
DATE, METRIC, DIMENSION, IDENTIFIER, TEXT, EMPTY, COORDINATE, HIDDEN = (
    "date", "metric", "dimension", "identifier", "text", "empty", "coordinate", "hidden")
ROLE_LABELS = {DATE: "Date", METRIC: "Measure", DIMENSION: "Category", IDENTIFIER: "Identifier", TEXT: "Text",
               COORDINATE: "Coordinate", EMPTY: "Empty", HIDDEN: "Hidden"}

_ID_RE = re.compile(
    r"(?:^|[^a-z])id$|^id(?:[^a-z]|$)|[a-z]ID$|[a-z]Id$|\buuid\b|\bguid\b|\bsku\b|\bzip\b|zip ?code|postal|post ?code"
    r"|phone|\bref(erence)?\b|\bkey$|\bcode$|(?<!number of )\bnumber$|\bno\.?$|\bnum$|#$",
)
_NOT_ID_RE = re.compile(r"^(number|no\.?|num)\s+of\b|count", re.IGNORECASE)
_MEAN_RE = re.compile(
    r"rate|ratio|percent|pct|%|\bavg\b|average|\bmean\b|median|price|per\s|/\s*unit|unit\s*(cost|price)|margin"
    r"|score|rating|\bage\b|temperature|\btemp\b|\bctr\b|\bcpc\b|\bcpm\b|\bcpa\b|\broas\b|\broi\b|probability"
    r"|index|satisfaction|\bnps\b|tenure|duration|lead\s*time|latitude|longitude|\blat\b|\blon\b|\blng\b|share"
    r"|height|weight|\bbmi\b|blood\s*pressure|heart\s*rate|cholesterol|glucose|humidity|speed|\bgpa\b",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"\byear\b|^yr$|fiscal\s*year|\bfy\b", re.IGNORECASE)
_MONTH_RE = re.compile(r"^month$|month\s*(no|num|number)?$", re.IGNORECASE)
_QUARTER_RE = re.compile(r"^(quarter|qtr|q)$", re.IGNORECASE)

# Higher score = more likely to be the "headline" measure. These only influence
# ordering; labels always use the actual column name.
_METRIC_PRIORITY = [
    (r"revenue|sales|turnover|income|amount|gmv|value|spend|cost|expense|profit|budget|salary|payroll|price", 6),
    (r"conversions?|orders?|purchases?|transactions?|leads?|signups?", 5),
    (r"quantity|qty|units|volume|clicks|impressions|visits|sessions|views|stock|inventory|on\s*hand|headcount|hours", 4),
    (r"total", 2),
]
_DIM_PRIORITY = [
    (r"category|segment|region|channel|department|dept|product|campaign|type|status|country|state|city|platform|source"
     r"|brand|store|branch|team|division|vendor|supplier|warehouse|location|class|group|tier|gender|level|role|title|method|account", 5),
    (r"name|customer|client|manager|owner|rep|agent|employee|item", 2),
]


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    role: str
    missing: int
    missing_pct: float
    unique: int
    unique_pct: float
    aggregation: str = "none"  # "sum", "mean" or "none" — the defensible default for this column
    is_percent: bool = False
    is_currency: bool = False
    is_time_part: bool = False  # Year / Month / Quarter numbers used as dimensions
    geo: str | None = None  # "lat" / "lon" when coordinates are detected
    high_cardinality: bool = False
    constant: bool = False
    stats: dict[str, Any] = field(default_factory=dict)
    top_values: list[tuple[str, int]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    priority: float = 0.0
    detected_role: str = ""
    detected_aggregation: str = "none"


@dataclass
class DatasetProfile:
    n_rows: int
    n_cols: int
    duplicate_rows: int
    total_missing: int
    missing_pct: float
    columns: dict[str, ColumnProfile]
    warnings: list[str] = field(default_factory=list)

    def by_role(self, role: str) -> list[str]:
        cols = [c for c in self.columns.values() if c.role == role]
        return [c.name for c in sorted(cols, key=lambda c: -c.priority)]

    @property
    def date_columns(self) -> list[str]:
        return self.by_role(DATE)

    @property
    def metric_columns(self) -> list[str]:
        return self.by_role(METRIC)

    @property
    def dimension_columns(self) -> list[str]:
        return self.by_role(DIMENSION)

    @property
    def identifier_columns(self) -> list[str]:
        return self.by_role(IDENTIFIER)

    @property
    def text_columns(self) -> list[str]:
        return self.by_role(TEXT)

    @property
    def chartable_dimensions(self) -> list[str]:
        """Dimensions with 2–50 categories, ordered by usefulness."""
        return [c for c in self.dimension_columns if 2 <= self.columns[c].unique <= 50 and not self.columns[c].constant]

    def coordinate_columns(self) -> tuple[str, str] | None:
        lat = next((c.name for c in self.columns.values() if c.geo == "lat"), None)
        lon = next((c.name for c in self.columns.values() if c.geo == "lon"), None)
        return (lat, lon) if lat and lon else None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["roles"] = {r: self.by_role(r) for r in (DATE, METRIC, DIMENSION, IDENTIFIER, TEXT, COORDINATE, EMPTY)}
        return d

    def summary_frame(self) -> pd.DataFrame:
        rows = []
        for c in self.columns.values():
            rows.append({
                "Column": c.name,
                "Role": c.role.title(),
                "Type": c.dtype,
                "Missing": c.missing,
                "Missing %": round(c.missing_pct, 1),
                "Unique": c.unique,
                "Default aggregation": c.aggregation if c.role == METRIC else "—",
                "Notes": "; ".join(c.notes),
            })
        return pd.DataFrame(rows)


def _keyword_score(name: str, table) -> float:
    words = re.sub(r"([a-z])([A-Z])", r" ", name).replace("_", " ")
    for pattern, score in table:
        if re.search(rf"\b(?:{pattern})\b", words, re.IGNORECASE):
            return score
    return 0.0


def _is_identifier_name(name: str) -> bool:
    if _NOT_ID_RE.search(name):
        return False
    return bool(_ID_RE.search(name) or _ID_RE.search(name.lower()))


def _profile_numeric(s: pd.Series, cp: ColumnProfile, n_rows: int, percent_hint: bool, currency_hint: bool) -> None:
    v = s.dropna().astype(float)
    name = cp.name
    integer_valued = bool(len(v) and np.allclose(v, v.round()))
    cp.is_percent = percent_hint or bool(re.search(r"%|percent|pct|\brate\b", name, re.IGNORECASE))
    cp.is_currency = currency_hint or bool(re.search(r"revenue|sales|cost|spend|price|amount|profit|expense|salary|budget|usd|\$|€|£|rm\b", name, re.IGNORECASE))

    if len(v):
        q1, q3 = np.percentile(v, [25, 75])
        iqr = q3 - q1
        outliers = int(((v < q1 - 1.5 * iqr) | (v > q3 + 1.5 * iqr)).sum()) if iqr > 0 else 0
        cp.stats = {
            "min": float(v.min()), "max": float(v.max()), "mean": float(v.mean()), "median": float(v.median()),
            "std": float(v.std()) if len(v) > 1 else 0.0, "sum": float(v.sum()), "q1": float(q1), "q3": float(q3),
            "zeros": int((v == 0).sum()), "negatives": int((v < 0).sum()), "outliers": outliers,
            "integer_valued": integer_valued,
        }

    # Coordinates
    lname = name.lower().strip()
    if len(v) and re.fullmatch(r"lat|latitude|.*[ _]lat(itude)?", lname) and v.between(-90, 90).all():
        cp.geo = "lat"
    elif len(v) and re.fullmatch(r"lon|lng|long|longitude|.*[ _](lon|lng|longitude)", lname) and v.between(-180, 180).all():
        cp.geo = "lon"

    sequential = integer_valued and cp.unique == len(v) and len(v) > 5 and bool((np.diff(np.sort(v.values)) == 1).all())
    if cp.geo:
        cp.role = COORDINATE
        cp.notes.append(f"Detected as {'latitude' if cp.geo == 'lat' else 'longitude'}; not summed")
    elif _is_identifier_name(name) and (integer_valued or cp.unique_pct > 50):
        cp.role = IDENTIFIER
        cp.notes.append("Looks like an identifier; not aggregated")
    elif sequential and cp.unique_pct > 95:
        cp.role = IDENTIFIER
        cp.notes.append("Sequential whole numbers; treated as a row identifier")
    elif _YEAR_RE.search(name) and integer_valued and len(v) and v.between(1900, 2100).all():
        cp.role, cp.is_time_part = DIMENSION, True
        cp.notes.append("Year number; used as a time dimension, not summed")
    elif (_MONTH_RE.search(name) and integer_valued and len(v) and v.between(1, 12).all()) or (
        _QUARTER_RE.search(name) and integer_valued and len(v) and v.between(1, 4).all()
    ):
        cp.role, cp.is_time_part = DIMENSION, True
        cp.notes.append("Period number; used as a dimension, not summed")
    else:
        cp.role = METRIC
        ratio_like = len(v) > 0 and not integer_valued and v.between(0, 1).all() and not re.search(r"qty|quantity|count|units", name, re.IGNORECASE)
        if cp.is_percent or _MEAN_RE.search(name) or ratio_like:
            cp.aggregation = "mean"
            cp.notes.append("Rate/price/score-like; averaged rather than summed")
        else:
            cp.aggregation = "sum"
        cp.priority = _keyword_score(name, _METRIC_PRIORITY) + (3 if cp.aggregation == "sum" else 0) - cp.missing_pct / 50
        if cp.constant:
            cp.priority -= 10


def _profile_object(s: pd.Series, cp: ColumnProfile, n_rows: int) -> None:
    v = s.dropna().astype(str)
    counts = v.value_counts()
    cp.top_values = [(str(k), int(c)) for k, c in counts.head(10).items()]
    avg_len = float(v.str.len().mean()) if len(v) else 0.0
    cp.stats = {"avg_length": round(avg_len, 1)}
    codes = v.str.fullmatch(r"[A-Za-z]{0,6}[-_/#]?\d[\w\-/#.]*").mean() if len(v) else 0
    all_unique = cp.unique == len(v)
    if len(v) > 10 and (cp.unique_pct >= 95 or all_unique) and avg_len <= 40 and (_is_identifier_name(cp.name) or codes > 0.8):
        cp.role = IDENTIFIER
        cp.notes.append("Unique codes; treated as an identifier")
    elif avg_len > 50 or (cp.unique > 100 and cp.unique_pct > 80 and v.str.contains(" ").mean() > 0.5 and avg_len > 25):
        cp.role = TEXT
        cp.notes.append("Free text; not used for grouping")
    elif _is_identifier_name(cp.name) and cp.unique_pct > 50 and len(v) > 10:
        cp.role = IDENTIFIER
        cp.notes.append("Looks like an identifier")
    else:
        cp.role = DIMENSION
        if cp.unique > 50:
            cp.high_cardinality = True
            cp.notes.append(f"{cp.unique:,} categories; charts show the top values only")
        card_score = 3 if 2 <= cp.unique <= 12 else 2 if cp.unique <= 30 else 0 if cp.unique <= 50 else -3
        cp.priority = _keyword_score(cp.name, _DIM_PRIORITY) + card_score - cp.missing_pct / 25
        if cp.constant:
            cp.priority -= 10


def profile_dataset(df: pd.DataFrame, percent_columns: list[str] | None = None, currency_columns: list[str] | None = None) -> DatasetProfile:
    percent_columns = set(percent_columns or [])
    currency_columns = set(currency_columns or [])
    n_rows, n_cols = df.shape
    columns: dict[str, ColumnProfile] = {}
    try:
        dup = int(df.duplicated().sum()) if n_rows else 0
    except TypeError:  # unhashable cells
        dup = int(df.astype(str).duplicated().sum())

    for name in df.columns:
        s = df[name]
        missing = int(s.isna().sum())
        nonnull = n_rows - missing
        try:
            unique = int(s.nunique(dropna=True))
        except TypeError:
            unique = int(s.astype(str).nunique(dropna=True))
        cp = ColumnProfile(
            name=str(name), dtype=str(s.dtype), role=EMPTY, missing=missing,
            missing_pct=(missing / n_rows * 100) if n_rows else 0.0, unique=unique,
            unique_pct=(unique / nonnull * 100) if nonnull else 0.0, constant=unique == 1,
        )
        if nonnull == 0:
            cp.role = EMPTY
            cp.notes.append("No values")
        elif pd.api.types.is_datetime64_any_dtype(s):
            cp.role = DATE
            v = s.dropna()
            cp.stats = {"min": v.min(), "max": v.max(), "span_days": int((v.max() - v.min()).days)}
            cp.priority = 5 - cp.missing_pct / 20 + (1 if re.search(r"date|day|time|period|month", cp.name, re.IGNORECASE) else 0)
            if cp.constant:
                cp.notes.append("Single date value")
                cp.priority -= 10
        elif pd.api.types.is_bool_dtype(s):
            cp.role = DIMENSION
            cp.top_values = [(str(k), int(c)) for k, c in s.value_counts().items()]
            cp.priority = 1
        elif pd.api.types.is_numeric_dtype(s):
            _profile_numeric(s, cp, n_rows, cp.name in percent_columns, cp.name in currency_columns)
        else:
            _profile_object(s, cp, n_rows)
        if cp.constant and cp.role not in (EMPTY,):
            cp.notes.append("Same value in every row")
        if cp.missing_pct >= 50 and cp.role != EMPTY:
            cp.notes.append(f"{cp.missing_pct:.0f}% missing")
        cp.detected_role = cp.role
        cp.detected_aggregation = cp.aggregation
        columns[cp.name] = cp

    total_missing = int(df.isna().sum().sum())
    cells = n_rows * n_cols
    prof = DatasetProfile(
        n_rows=n_rows, n_cols=n_cols, duplicate_rows=dup, total_missing=total_missing,
        missing_pct=(total_missing / cells * 100) if cells else 0.0, columns=columns,
    )
    if dup:
        prof.warnings.append(
            f"{dup:,} row(s) are exact duplicates of another row. They are kept, so totals include them; "
            "check whether they are genuine repeats."
        )
    if not prof.metric_columns:
        prof.warnings.append("No numeric measures were detected; the dashboard will use record counts.")
    heavy = [c.name for c in columns.values() if c.missing_pct >= 50 and c.role != EMPTY]
    if heavy:
        prof.warnings.append(f"Columns with more than half the values missing: {', '.join(heavy[:6])}.")
    empty = [c.name for c in columns.values() if c.role == EMPTY]
    if empty:
        prof.warnings.append(f"Columns with no values: {', '.join(empty[:6])}.")
    return prof


# --------------------------------------------------------------------------- user corrections

def allowed_roles(cp: ColumnProfile) -> list[str]:
    """Roles a user may assign to a column, given its underlying data type."""
    if (cp.detected_role or cp.role) == EMPTY:
        return [EMPTY, HIDDEN]
    if cp.dtype.startswith("datetime"):
        return [DATE, HIDDEN]
    if cp.detected_role in (METRIC, COORDINATE) or cp.dtype.startswith(("int", "float", "Int", "Float")):
        base = [METRIC, DIMENSION, IDENTIFIER, HIDDEN]
        return ([COORDINATE] + base) if cp.geo else base
    return [DIMENSION, IDENTIFIER, TEXT, HIDDEN]


def apply_overrides(profile: DatasetProfile, overrides: dict | None) -> tuple[DatasetProfile, list[str]]:
    """Return a copy of ``profile`` with the user's role/aggregation corrections applied.

    Invalid overrides (unknown columns, roles that don't fit the data type) are ignored with a warning.
    """
    out = copy.deepcopy(profile)
    warnings: list[str] = []
    for name, ov in (overrides or {}).items():
        cp = out.columns.get(name)
        if cp is None or not isinstance(ov, dict):
            warnings.append(f"Column setting for '{name}' was ignored: the column no longer exists.")
            continue
        role = ov.get("role")
        if role and role != cp.role:
            if role not in allowed_roles(cp):
                warnings.append(f"'{name}' can't be used as {ROLE_LABELS.get(role, role)}; kept as {ROLE_LABELS[cp.role]}.")
            else:
                cp.role = role
                cp.notes = [n for n in cp.notes if "identifier" not in n.lower()] + [f"Set to {ROLE_LABELS[role]} by you"]
                if role == METRIC and cp.aggregation == "none":
                    cp.aggregation = "sum"
                    cp.priority = max(cp.priority, 1.0)
                if role == DIMENSION:
                    cp.high_cardinality = cp.unique > 50
                    cp.priority = max(cp.priority, 1.0) if cp.unique <= 50 else -3
        agg = ov.get("aggregation")
        if agg in ("sum", "mean") and cp.role == METRIC:
            cp.aggregation = agg
    return out, warnings
