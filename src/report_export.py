"""Export a dashboard as one self-contained HTML file that opens in any browser, offline, with no setup.

The page is a snapshot: KPIs, charts and insights are calculated when it is exported (using the dashboard's
current filters). Charts stay interactive (hover, zoom, legend toggles) because Plotly is embedded in the file.
"""
from __future__ import annotations

import html
from datetime import datetime, timezone

import pandas as pd

from .chart_generator import ChartConfigError, validate_chart_spec, validate_kpi_spec
from .chart_renderer import ChartDataError, build_figure, prepare_chart_data
from .dashboard_service import compute_kpi, describe_filters
from .data_profiler import DatasetProfile
from .formatting import format_value
from .insights import generate_insights

_CSS = """
:root { --ink:#1c1f24; --ink2:#5b616b; --ink3:#8a9099; --line:#e3e6ea; --surface:#fff; --canvas:#f6f7f9; --accent:#1f5fbf; }
* { box-sizing: border-box; }
body { margin:0; background:var(--canvas); color:var(--ink); font-family: Inter, -apple-system, "Segoe UI", Roboto, sans-serif; }
.wrap { max-width: 1280px; margin: 0 auto; padding: 32px 24px 48px; }
.eyebrow { font-size:12px; font-weight:600; letter-spacing:.06em; text-transform:uppercase; color:var(--ink3); }
h1 { font-size:26px; margin:4px 0 8px; letter-spacing:-.01em; }
h2 { font-size:17px; margin:28px 0 12px; }
.meta { display:flex; flex-wrap:wrap; gap:6px; margin-bottom:6px; }
.chip { font-size:12px; color:var(--ink2); border:1px solid var(--line); background:var(--surface); border-radius:999px; padding:2px 10px; }
.chip.accent { color:var(--accent); border-color:#c9d8f1; background:#e8effa; }
.kpis { display:grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap:12px; }
.kpi, .card { background:var(--surface); border:1px solid var(--line); border-radius:10px; }
.kpi { padding:12px 14px; }
.kpi .l { font-size:12px; color:var(--ink2); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.kpi .v { font-size:23px; font-weight:650; margin-top:3px; font-variant-numeric: tabular-nums; }
.kpi .d { font-size:12px; color:var(--ink3); margin-top:3px; }
.kpi .d b.up { color:#0f7b3f; } .kpi .d b.down { color:#b42318; }
.grid { display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap:14px; }
.card { padding:14px 16px 10px; min-width:0; }
.card.full { grid-column: 1 / -1; }
.card .t { font-size:15px; font-weight:600; }
.card .s { font-size:12px; color:var(--ink3); margin:3px 0 6px; }
.card .n { font-size:12px; color:var(--ink3); margin-top:4px; }
table.data { border-collapse:collapse; width:100%; font-size:13px; }
table.data th, table.data td { border-bottom:1px solid var(--line); padding:6px 8px; text-align:left; }
table.data td.num, table.data th.num { text-align:right; font-variant-numeric: tabular-nums; }
.insight { padding:9px 0; border-bottom:1px solid var(--line); }
.insight:last-child { border-bottom:none; }
.insight b { font-size:13px; } .insight div { font-size:14px; color:var(--ink2); line-height:1.45; }
.foot { margin-top:28px; font-size:12px; color:var(--ink3); line-height:1.5; }
@media (max-width: 860px) { .grid { grid-template-columns: 1fr; } .wrap { padding: 20px 16px 36px; } }
"""


def _e(text) -> str:
    return html.escape(str(text), quote=True)


def _table_html(frame: pd.DataFrame) -> str:
    head = "".join(f'<th class="{"num" if pd.api.types.is_numeric_dtype(frame[c]) else ""}">{_e(c)}</th>' for c in frame.columns)
    rows = []
    for row in frame.itertuples(index=False):
        cells = []
        for c, v in zip(frame.columns, row):
            if pd.api.types.is_numeric_dtype(frame[c]) and pd.notna(v):
                cells.append(f'<td class="num">{_e(format_value(v, decimals=2))}</td>')
            else:
                cells.append(f"<td>{_e('' if pd.isna(v) else v)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f'<table class="data"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


def build_html_report(cfg: dict, data: pd.DataFrame, profile: DatasetProfile, total_rows: int) -> str:
    """Render the dashboard to a standalone HTML document. ``data`` is the (filtered) rows to report on."""
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    filters = describe_filters(cfg.get("filters"))
    chips = [f'<span class="chip">{_e(cfg.get("dataset_name") or "Dataset")}</span>']
    if cfg.get("sheet") and cfg["sheet"] != "CSV data":
        chips.append(f'<span class="chip">Sheet: {_e(cfg["sheet"])}</span>')
    chips.append(f'<span class="chip">{len(data):,} of {total_rows:,} rows</span>')
    if filters != "No filters":
        chips.append(f'<span class="chip accent">Filters: {_e(filters)}</span>')

    kpi_html = []
    for k in cfg.get("kpis", []):
        try:
            validate_kpi_spec(k, profile)
        except ChartConfigError:
            continue
        out = compute_kpi(data, k, profile)
        delta = ""
        if out.get("delta"):
            cls = "up" if out["delta"].startswith("+") else "down"
            delta = f'<div class="d"><b class="{cls}">{"▲" if cls == "up" else "▼"} {_e(out["delta"])}</b> {_e(out.get("delta_label") or "")}</div>'
        kpi_html.append(f'<div class="kpi" title="{_e(out.get("help", ""))}"><div class="l">{_e(out["label"])}</div>'
                        f'<div class="v">{_e(out["display"])}</div>{delta}</div>')

    cards, plotly_included = [], False
    for spec in cfg.get("charts", []):
        try:
            spec = dict(validate_chart_spec(spec, profile), description=spec.get("description", ""))
            agg, notes = prepare_chart_data(data, spec, profile)
        except (ChartConfigError, ChartDataError) as exc:
            body, notes = f'<div class="n">{_e(exc)}</div>', []
        else:
            if spec["type"] == "table":
                body = _table_html(agg)
            else:
                fig = build_figure(agg, spec, profile)
                body = fig.to_html(full_html=False, include_plotlyjs="inline" if not plotly_included else False,
                                   config={"displaylogo": False, "responsive": True})
                plotly_included = True
        full = spec.get("type") in ("table", "map") or spec.get("width") == "full"
        note = f'<div class="n">{_e(" ".join(notes))}</div>' if notes else ""
        cards.append(f'<section class="card{" full" if full else ""}"><div class="t">{_e(spec.get("title", ""))}</div>'
                     f'<div class="s">{_e(spec.get("description", ""))}</div>{body}{note}</section>')

    insights = "".join(f'<div class="insight"><b>{_e(i.title)}</b><div>{_e(i.text)}</div></div>'
                       for i in generate_insights(data, profile))

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_e(cfg.get("name", "Dashboard"))}</title><style>{_CSS}</style></head>
<body><main class="wrap">
<div class="eyebrow">Dashboard snapshot</div>
<h1>{_e(cfg.get("name", "Dashboard"))}</h1>
<div class="meta">{"".join(chips)}</div>
<h2>Key metrics</h2><div class="kpis">{"".join(kpi_html) or '<div class="n">No KPIs.</div>'}</div>
<h2>Charts</h2><div class="grid">{"".join(cards) or '<div class="n">No charts.</div>'}</div>
<h2>Insights</h2><section class="card">{insights}</section>
<p class="foot">Snapshot exported {generated} from AI Dashboard Builder. Numbers were calculated from the data when the file
was exported; filters can't be changed here, but you can hover, zoom and toggle legend items on the charts.
This file contains the chart data shown (totals, and individual points for scatter, box and histogram charts),
not the full original spreadsheet.</p>
</main></body></html>"""
