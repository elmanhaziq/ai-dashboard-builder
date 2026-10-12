import pandas as pd
import pytest

from src.chart_generator import (
    ChartConfigError, adapt_spec, attach_correlations, compatible_types, recommend_charts, recommend_kpis,
    validate_chart_spec, validate_kpi_spec,
)
from src.chart_renderer import ChartDataError, build_figure, prepare_chart_data
from src.dashboard_service import compute_kpi
from tests import synthetic as syn
from tests.conftest import load_and_profile


ALL = [syn.sales, syn.marketing, syn.expenses, syn.inventory]


@pytest.mark.parametrize("make", ALL)
def test_recommendations_are_valid_unique_and_renderable(make):
    res, p = load_and_profile(make())
    attach_correlations(p, res.df)
    charts = recommend_charts(p)
    assert len(charts) >= 4
    keys = [(c["type"], c["x"], c["y"], c["color"]) for c in charts]
    assert len(keys) == len(set(keys)), "duplicate charts"
    for c in charts:
        assert validate_chart_spec(c, p)["id"] == c["id"]
        assert c["description"]
        for col in (c["x"], c["y"], c["color"]):
            assert col is None or p.columns[col].role != "identifier" or c["type"] in ("bar", "hbar", "table")
        data, _ = prepare_chart_data(res.df, c, p)
        build_figure(data, c, p)


def test_sales_recommendations_content(sales_loaded):
    res, p = sales_loaded
    attach_correlations(p, res.df)
    charts = recommend_charts(p)
    first = charts[0]
    assert first["type"] == "line" and first["x"] == "Order Date" and first["y"] == "Sales" and first["agg"] == "sum"
    assert any(c["type"] in ("bar", "hbar") and c["x"] == "Region" and c["y"] == "Sales" for c in charts)
    # IDs are never used as measures
    assert all(c["y"] not in ("Order ID", "Customer ID") for c in charts)
    # Donut only for an additive measure
    for c in charts:
        if c["type"] == "donut" and c["y"]:
            assert p.columns[c["y"]].aggregation == "sum"


def test_validate_chart_spec_rejects_bad_configs(sales_loaded):
    _, p = sales_loaded
    bad = [
        {"type": "pie3d", "x": "Region"},
        {"type": "bar", "x": "Nope", "y": "Sales"},
        {"type": "bar", "x": "Region", "y": "Region", "agg": "sum"},
        {"type": "line", "x": "Region", "y": "Sales"},
        {"type": "scatter", "x": "Sales", "y": "Sales"},
        {"type": "donut", "x": "Region", "y": "Sales", "agg": "mean"},
        {"type": "bar", "x": "Region", "y": "Sales", "agg": "__import__('os').system('x')"},
        "not a dict",
    ]
    for spec in bad:
        with pytest.raises(ChartConfigError):
            validate_chart_spec(spec, p)


def test_validate_clamps_and_sanitises(sales_loaded):
    _, p = sales_loaded
    clean = validate_chart_spec({"type": "bar", "x": "Region", "y": "Sales", "top_n": 9999, "sort": "weird",
                                 "title": "x" * 500, "unknown_key": "ignored"}, p)
    assert clean["top_n"] == 50 and clean["sort"] == "none" and len(clean["title"]) == 120 and "unknown_key" not in clean


def test_adapt_spec_switches_types_compatibly(sales_loaded):
    _, p = sales_loaded
    base = validate_chart_spec({"type": "bar", "x": "Region", "y": "Sales", "agg": "sum"}, p)
    for t in compatible_types(p):
        validate_chart_spec(adapt_spec(base, t, p), p)


def test_kpis_use_column_names_and_safe_ratios(marketing_loaded):
    res, p = marketing_loaded
    kpis = recommend_kpis(p, max_kpis=8)
    labels = [k["label"] for k in kpis]
    assert labels[0] == "Records"
    assert "Sum of Spend" in labels
    assert not any(l.lower().startswith(("revenue", "profit", "total spend")) for l in labels)
    ratios = {k["label"]: k for k in kpis if k["kind"] == "ratio"}
    assert "Click-through rate" in ratios
    ctr = compute_kpi(res.df, ratios["Click-through rate"], p)
    assert ctr["value"] == pytest.approx(res.df["Clicks"].sum() / res.df["Impressions"].sum())
    for k in kpis:
        validate_kpi_spec(k, p)


def test_no_ratio_when_names_are_ambiguous():
    df = pd.DataFrame({"Clicks A": [1, 2, 3] * 5, "Clicks B": [2, 3, 4] * 5, "Impressions": [10, 20, 30] * 5})
    _, p = load_and_profile(df)
    assert not [k for k in recommend_kpis(p) if k["kind"] == "ratio"]


def test_kpi_values_match_pandas(sales_loaded):
    res, p = sales_loaded
    df = res.df
    by_label = {k["label"]: k for k in recommend_kpis(p)}
    assert compute_kpi(df, by_label["Records"], p)["value"] == len(df)
    assert compute_kpi(df, by_label["Sum of Sales"], p)["value"] == pytest.approx(df["Sales"].sum())
    assert compute_kpi(df, by_label["Average of Unit Price"], p)["value"] == pytest.approx(df["Unit Price"].mean())


def test_bar_aggregation_top_n_and_other_folding(sales_loaded):
    res, p = sales_loaded
    spec = validate_chart_spec({"type": "hbar", "x": "Product", "y": "Sales", "agg": "sum", "top_n": 5, "sort": "value_desc"}, p)
    data, notes = prepare_chart_data(res.df, spec, p)
    expected = res.df.groupby("Product")["Sales"].sum().sort_values(ascending=False).head(5)
    assert list(data["Product"].astype(str)) == list(expected.index)
    assert data["value"].tolist() == pytest.approx(expected.tolist())
    assert any("Showing 5 of 12" in n for n in notes)

    donut = validate_chart_spec({"type": "donut", "x": "Product", "y": "Sales", "agg": "sum"}, p)
    d, notes = prepare_chart_data(res.df, donut, p)
    assert "Other" in d["Product"].tolist()
    assert d["value"].sum() == pytest.approx(res.df["Sales"].sum())  # nothing lost or double counted


def test_line_chart_monthly_totals(sales_loaded):
    res, p = sales_loaded
    spec = validate_chart_spec({"type": "line", "x": "Order Date", "y": "Sales", "agg": "sum", "time_grain": "M"}, p)
    data, _ = prepare_chart_data(res.df, spec, p)
    assert len(data) == 12
    assert data["value"].sum() == pytest.approx(res.df["Sales"].sum())


def test_empty_data_gives_chart_data_error(sales_loaded):
    res, p = sales_loaded
    spec = validate_chart_spec({"type": "bar", "x": "Region", "y": "Sales"}, p)
    with pytest.raises(ChartDataError):
        prepare_chart_data(res.df.iloc[0:0], spec, p)
