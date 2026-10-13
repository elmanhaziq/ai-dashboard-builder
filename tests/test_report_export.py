"""Standalone HTML export and Power BI-style bar highlighting."""
import pandas as pd

from src.chart_generator import validate_chart_spec
from src.chart_renderer import build_figure, prepare_chart_data
from src.dashboard_service import apply_filters, build_default_dashboard
from src.report_export import build_html_report
from tests.conftest import load_and_profile
from tests import synthetic as syn


def test_html_report_is_self_contained(sales_loaded):
    res, p = sales_loaded
    cfg = build_default_dashboard(p, res.df, "a" * 16, "sales.csv", "CSV data")
    doc = build_html_report(cfg, res.df, p, total_rows=len(res.df))
    assert doc.startswith("<!doctype html>")
    assert "Sum of Sales" in doc and "500 of 500 rows" in doc
    assert len(doc) > 1_000_000  # Plotly library embedded in the file, so it works offline
    assert '<script src="http' not in doc and "<script src='http" not in doc  # nothing loaded from the internet
    assert doc.count('class="card') >= len(cfg["charts"])


def test_html_report_uses_filters(sales_loaded):
    res, p = sales_loaded
    cfg = build_default_dashboard(p, res.df, "a" * 16, "sales.csv", "CSV data")
    cfg["filters"] = {"categorical": {"Region": ["North"]}, "date": {}}
    north = apply_filters(res.df, cfg["filters"], p)
    doc = build_html_report(cfg, north, p, total_rows=len(res.df))
    assert f"{len(north):,} of 500 rows" in doc and "Filters: Region: North" in doc


def test_html_report_escapes_user_text():
    df = pd.DataFrame({"<script>alert(1)</script>": ["a", "b", "c"] * 5, "Amount": [1.0, 2.0, 3.5] * 5})
    res, p = load_and_profile(df)
    cfg = build_default_dashboard(p, res.df, "a" * 16, '<img src=x onerror="alert(2)">.csv', "CSV data")
    cfg["name"] = "<b onmouseover=alert(3)>Report</b>"
    doc = build_html_report(cfg, res.df, p, total_rows=len(res.df))
    assert "<script>alert(1)</script>" not in doc
    assert '<img src=x onerror="alert(2)">' not in doc
    assert "<b onmouseover=alert(3)>" not in doc
    assert "&lt;b onmouseover=alert(3)&gt;Report&lt;/b&gt;" in doc


def test_bar_highlight_fades_unselected(sales_loaded):
    res, p = sales_loaded
    spec = validate_chart_spec({"type": "bar", "x": "Region", "y": "Sales", "agg": "sum"}, p)
    data, _ = prepare_chart_data(res.df, spec, p)
    fig = build_figure(data, spec, p, highlight=["North"])
    tr = fig.data[0]
    opac = dict(zip(tr.x, tr.marker.opacity))
    assert opac["North"] == 1.0 and all(v < 1 for k, v in opac.items() if k != "North")
    assert len(tr.x) == 4  # the clicked chart still shows every region
    plain = build_figure(data, spec, p)
    assert plain.data[0].marker.opacity is None


def test_inventory_report_builds():
    res, p = load_and_profile(syn.inventory())
    cfg = build_default_dashboard(p, res.df, "b" * 16, "inv.csv", "CSV data")
    assert "Stock On Hand" in build_html_report(cfg, res.df, p, total_rows=len(res.df))
