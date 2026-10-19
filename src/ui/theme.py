"""Visual styling and small HTML components for the Streamlit UI."""
from __future__ import annotations

import html
import re
from pathlib import Path

import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root {
  --ink: #1c1f24; --ink-2: #5b616b; --ink-3: #8a9099;
  --line: #e3e6ea; --surface: #ffffff; --canvas: #f6f7f9;
  --accent: #1f5fbf; --accent-soft: #e8effa;
  --good: #0f7b3f; --bad: #b42318;
}
html, body, .stApp, .stMarkdown p, .stMarkdown li, h1, h2, h3, label, input, textarea { font-family: "Inter", -apple-system, "Segoe UI", Roboto, sans-serif; }
[data-testid="stAppDeployButton"], footer { display: none !important; }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1400px; }
h1, h2, h3 { letter-spacing: -0.01em; color: var(--ink); }
h1 { font-size: 1.55rem !important; font-weight: 650 !important; }
h2 { font-size: 1.2rem !important; font-weight: 600 !important; }
h3 { font-size: 1.0rem !important; font-weight: 600 !important; }
[data-testid="stSidebar"] { border-right: 1px solid var(--line); }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 10px; }
[data-testid="stExpander"] details { border-radius: 10px; border-color: var(--line); background: var(--surface); }

.app-brand { display:flex; align-items:center; gap:10px; font-weight:650; font-size:0.98rem; color:var(--ink); padding: 2px 0 6px 0; }
.app-brand .mark { width:22px; height:22px; border-radius:6px; background: var(--accent); display:inline-flex; align-items:center; justify-content:center; }
.app-brand .mark span { width:10px; height:10px; border:2px solid #fff; border-radius:2px; border-top-color: transparent; }

.page-eyebrow { font-size: 0.75rem; font-weight: 600; color: var(--ink-3); text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 2px; }
.page-sub { color: var(--ink-2); font-size: 0.9rem; margin-top: -6px; }
.meta-row { display:flex; flex-wrap:wrap; gap:6px; margin: 4px 0 2px 0; }
.chip { display:inline-flex; align-items:center; gap:4px; padding: 2px 9px; border:1px solid var(--line); border-radius: 999px; font-size: 0.78rem; color: var(--ink-2); background: var(--surface); }
.chip.accent { border-color: #c9d8f1; background: var(--accent-soft); color: var(--accent); }
.chip.warn { border-color: #f1d9a8; background: #fdf6e7; color: #8a5a00; }

.kpi-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin: 6px 0 14px 0; }
.kpi { background: var(--surface); border:1px solid var(--line); border-radius: 10px; padding: 12px 14px 11px 14px; min-width: 0; }
.kpi-label { font-size: 0.76rem; color: var(--ink-2); font-weight: 500; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.kpi-value { font-size: 1.42rem; font-weight: 650; color: var(--ink); margin-top: 3px; letter-spacing: -0.015em; font-variant-numeric: tabular-nums; }
.kpi-delta { font-size: 0.74rem; margin-top: 3px; color: var(--ink-3); white-space: nowrap; overflow:hidden; text-overflow: ellipsis; }
.kpi-delta b { font-weight: 600; }
.kpi-delta.up b { color: var(--good); }
.kpi-delta.down b { color: var(--bad); }

.chart-title { font-size: 0.93rem; font-weight: 600; color: var(--ink); margin: 0; line-height: 1.3; }
.chart-desc { font-size: 0.77rem; color: var(--ink-3); margin: 3px 0 0 0; line-height: 1.4; }
[class*="st-key-card_"] { background: var(--surface); }
[class*="st-key-card_qa"] p, [class*="st-key-card_qa"] li { font-size: 0.9rem; line-height: 1.5; }
[data-testid="stHeaderActionElements"] { display: none !important; }
[data-testid="stMetricValue"] { font-size: 1.45rem; font-weight: 600; }
[data-testid="stMetricLabel"] p { font-size: 0.8rem; color: var(--ink-2); }
.chart-note { font-size: 0.74rem; color: var(--ink-3); margin-top: -4px; }

.insight { padding: 9px 0; border-bottom: 1px solid var(--line); }
.insight:last-child { border-bottom: none; }
.insight-title { font-size: 0.8rem; font-weight: 600; color: var(--ink); }
.insight-text { font-size: 0.86rem; color: var(--ink-2); line-height: 1.45; }

.stepper { display:flex; flex-wrap:wrap; gap:6px; margin: 10px 0 14px 0; }
.step { display:flex; align-items:center; gap:6px; font-size:0.78rem; color: var(--ink-3); padding: 4px 10px 4px 4px;
        border:1px solid var(--line); border-radius:999px; background: var(--surface); }
.step span { width:20px; height:20px; border-radius:50%; display:inline-flex; align-items:center; justify-content:center;
             font-size:0.7rem; font-weight:600; background:#eef0f3; color: var(--ink-2); }
.step.done { color: var(--ink-2); }
.step.done span { background: #d6f0e0; color: #0f7b3f; }
.step.now { color: var(--accent); border-color: #c9d8f1; background: var(--accent-soft); font-weight:600; }
.step.now span { background: var(--accent); color:#fff; }
.ai-note { font-size:0.85rem; color: var(--ink-2); line-height:1.5; border:1px solid var(--line); border-left: 3px solid var(--accent);
           background: var(--surface); border-radius: 8px; padding: 10px 14px; }
.panel-label { font-size:0.72rem; font-weight:600; text-transform:uppercase; letter-spacing:0.06em; color: var(--accent); margin-bottom:4px; }
.issue-text { font-size:0.84rem; color: var(--ink-2); line-height:1.45; margin-bottom: 4px; }
.empty { border: 1px dashed #cfd4da; border-radius: 10px; padding: 28px 20px; text-align:center; color: var(--ink-2); background: var(--surface); }
.empty b { color: var(--ink); display:block; margin-bottom: 4px; }
.privacy { font-size: 0.75rem; color: var(--ink-3); line-height: 1.4; }
.stat-line { font-size: 0.82rem; color: var(--ink-2); }
@media (max-width: 640px) {
  .block-container { padding-left: 1rem; padding-right: 1rem; padding-top: 1.2rem; }
  .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }
  .kpi-value { font-size: 1.15rem; }
}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def esc(text) -> str:
    return html.escape(str(text), quote=True)


def brand() -> None:
    logo = Path(__file__).resolve().parent.parent.parent / "assets" / "logo.svg"
    if logo.is_file():
        st.logo(str(logo), size="large")


def page_header(eyebrow: str, title: str, subtitle: str | None = None) -> None:
    st.markdown(f'<div class="page-eyebrow">{esc(eyebrow)}</div>', unsafe_allow_html=True)
    st.markdown(f"# {esc(title)}", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="page-sub">{esc(subtitle)}</div>', unsafe_allow_html=True)


def chips(items: list[tuple[str, str]]) -> None:
    """items: (text, style) where style is '', 'accent' or 'warn'."""
    inner = "".join(f'<span class="chip {esc(s)}">{esc(t)}</span>' for t, s in items)
    st.markdown(f'<div class="meta-row">{inner}</div>', unsafe_allow_html=True)


def kpi_grid(cards: list[dict]) -> None:
    parts = []
    for c in cards:
        delta = ""
        if c.get("delta"):
            cls = "up" if c["delta"].startswith("+") else "down"
            arrow = "▲" if cls == "up" else "▼"
            delta = f'<div class="kpi-delta {cls}"><b>{arrow} {esc(c["delta"])}</b> {esc(c.get("delta_label") or "")}</div>'
        elif c.get("subtitle"):
            delta = f'<div class="kpi-delta">{esc(c["subtitle"])}</div>'
        parts.append(
            f'<div class="kpi" title="{esc(c.get("help", ""))}"><div class="kpi-label">{esc(c["label"])}</div>'
            f'<div class="kpi-value">{esc(c["display"])}</div>{delta}</div>'
        )
    st.markdown(f'<div class="kpi-grid">{"".join(parts)}</div>', unsafe_allow_html=True)


def chart_heading(title: str, description: str = "") -> None:
    desc = f'<div class="chart-desc">{esc(description)}</div>' if description else ""
    st.markdown(f'<div class="chart-title">{esc(title)}</div>{desc}', unsafe_allow_html=True)


def empty_state(title: str, text: str) -> None:
    st.markdown(f'<div class="empty"><b>{esc(title)}</b>{esc(text)}</div>', unsafe_allow_html=True)


def insight_list(items) -> None:
    body = "".join(
        f'<div class="insight"><div class="insight-title">{esc(i.title)}</div><div class="insight-text">{esc(i.text)}</div></div>'
        for i in items
    )
    st.markdown(body, unsafe_allow_html=True)


def card(key: str):
    """A bordered white card container (styled via its key class)."""
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", str(key))
    return st.container(border=True, key=f"card_{safe}")


def date_column_config(df) -> dict:
    """Show datetime columns without a time part when every time is midnight."""
    import pandas as pd
    cfg = {}
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            v = s.dropna()
            if len(v) and (v == v.dt.normalize()).all():
                cfg[c] = st.column_config.DateColumn(str(c), format="YYYY-MM-DD")
    return cfg


def human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"
