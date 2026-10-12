"""Power BI-style customisation: multi-value visuals, stacking, heatmaps, visual filters, field overrides."""
import pandas as pd
import pytest

from src.chart_generator import (
    ChartConfigError, adapt_spec, compatible_types, describe_chart, recommend_charts, validate_chart_spec,
)
from src.chart_renderer import SERIES, apply_chart_filters, build_figure, prepare_chart_data
from src.dashboard_service import build_default_dashboard, validate_dashboard_config
from src.data_profiler import DIMENSION, HIDDEN, IDENTIFIER, METRIC, allowed_roles, apply_overrides
from tests import synthetic as syn
from tests.conftest import load_and_profile


def test_multiple_values_share_one_axis(sales_loaded):
    res, p = sales_loaded
    df = res.df
    spec = validate_chart_spec({"type": "bar", "x": "Region", "y": "Sales", "agg": "sum",
                                "extra": [{"y": "Quantity", "agg": "sum"}]}, p)
    assert spec["extra"] == [{"y": "Quantity", "agg": "sum"}]
    data, notes = prepare_chart_data(df, spec, p)
    assert set(data[SERIES]) == {"Sum of Sales", "Sum of Quantity"}
    sales = data[data[SERIES] == "Sum of Sales"].set_index("Region")["value"]
    qty = data[data[SERIES] == "Sum of Quantity"].set_index("Region")["value"]
    expected = df.groupby("Region")[["Sales", "Quantity"]].sum()
    for r in expected.index:
        assert sales[r] == pytest.approx(expected.loc[r, "Sales"])
        assert qty[r] == pytest.approx(expected.loc[r, "Quantity"])
    assert any("different scales" in n for n in notes)  # sales ≫ quantity
    fig = build_figure(data, spec, p)
    assert len(fig.data) == 2 and all(getattr(tr, "yaxis", "y") in (None, "y") for tr in fig.data)  # no second axis
    assert "Sales and Quantity" in spec["title"]


def test_multi_value_line_over_time(marketing_loaded):
    res, p = marketing_loaded
    spec = validate_chart_spec({"type": "line", "x": "Date", "y": "Clicks", "agg": "sum", "time_grain": "W",
                                "extra": [{"y": "Conversions", "agg": "sum"}]}, p)
    data, _ = prepare_chart_data(res.df, spec, p)
    assert data.groupby(SERIES)["value"].sum()["Sum of Clicks"] == pytest.approx(res.df["Clicks"].sum())
    build_figure(data, dict(spec, labels=True), p)


@pytest.mark.parametrize("bad", [
    {"extra": "Quantity"},
    {"extra": [{"y": "Region", "agg": "sum"}]},
    {"extra": [{"y": "Quantity", "agg": "eval"}]},
    {"extra": [{"y": "Quantity", "agg": "sum"}], "color": "Product"},
])
def test_invalid_extra_values_rejected(sales_loaded, bad):
    _, p = sales_loaded
    with pytest.raises(ChartConfigError):
        validate_chart_spec(dict({"type": "bar", "x": "Region", "y": "Sales", "agg": "sum"}, **bad), p)


def test_extra_values_dropped_for_types_that_cannot_show_them(sales_loaded):
    _, p = sales_loaded
    spec = validate_chart_spec({"type": "donut", "x": "Region", "y": "Sales", "agg": "sum",
                                "extra": [{"y": "Quantity", "agg": "sum"}]}, p)
    assert spec["extra"] == []


def test_stacked_and_percent_bars(sales_loaded):
    res, p = sales_loaded
    spec = validate_chart_spec({"type": "bar", "x": "Region", "y": "Sales", "agg": "sum", "color": "Product",
                                "barmode": "percent"}, p)
    data, _ = prepare_chart_data(res.df, spec, p)
    fig = build_figure(data, spec, p)
    assert fig.layout.barmode == "relative"
    # each region's segments add up to 100%
    per_region: dict = {}
    for tr in fig.data:
        for x, y in zip(tr.x, tr.y):
            per_region[x] = per_region.get(x, 0) + y
    assert all(v == pytest.approx(100) for v in per_region.values())
    with pytest.raises(ChartConfigError, match="100%"):
        validate_chart_spec(dict(spec, agg="mean"), p)


def test_heatmap_matrix(sales_loaded):
    res, p = sales_loaded
    df = res.df
    spec = validate_chart_spec({"type": "heatmap", "x": "Region", "x2": "Product", "y": "Sales", "agg": "sum"}, p)
    data, _ = prepare_chart_data(df, spec, p)
    cell = data[(data["Region"] == "North") & (data["Product"] == "Product 1")]["value"].iloc[0]
    assert cell == pytest.approx(df[(df["Region"] == "North") & (df["Product"] == "Product 1")]["Sales"].sum())
    fig = build_figure(data, spec, p)
    assert fig.data[0].type == "heatmap"
    with pytest.raises(ChartConfigError):
        validate_chart_spec({"type": "heatmap", "x": "Region", "x2": "Region", "y": "Sales"}, p)
    assert "heatmap" in compatible_types(p)


def test_heatmap_recommended_for_two_small_categories():
    res, p = load_and_profile(syn.expenses())
    from src.chart_generator import attach_correlations
    attach_correlations(p, res.df)
    assert any(c["type"] == "heatmap" for c in recommend_charts(p, max_charts=12))


def test_visual_level_filters_only_affect_that_visual(sales_loaded):
    res, p = sales_loaded
    df = res.df
    spec = validate_chart_spec({"type": "bar", "x": "Product", "y": "Sales", "agg": "sum",
                                "chart_filters": {"Region": ["North"]}}, p)
    data, _ = prepare_chart_data(df, spec, p)
    expected = df[df["Region"] == "North"].groupby("Product")["Sales"].sum()
    got = data.set_index("Product")["value"]
    for prod, v in got.items():
        assert v == pytest.approx(expected[str(prod)])
    assert len(apply_chart_filters(df, spec)) == int((df["Region"] == "North").sum())
    assert len(df) == 500  # original untouched
    assert "Visual filter: Region = North" in describe_chart(spec, p)
    with pytest.raises(ChartConfigError):
        validate_chart_spec(dict(spec, chart_filters={"Sales": ["1"]}), p)


def test_bubble_size_and_width(sales_loaded):
    res, p = sales_loaded
    spec = validate_chart_spec({"type": "scatter", "x": "Unit Price", "y": "Sales", "size": "Quantity", "width": "full"}, p)
    assert spec["size"] == "Quantity" and spec["width"] == "full"
    data, _ = prepare_chart_data(res.df, spec, p)
    build_figure(data, spec, p)
    assert validate_chart_spec(dict(spec, width="huge"), p)["width"] == "half"


def test_adapt_spec_keeps_specs_valid_across_all_types(sales_loaded):
    _, p = sales_loaded
    base = validate_chart_spec({"type": "bar", "x": "Region", "y": "Sales", "agg": "sum",
                                "extra": [{"y": "Quantity", "agg": "sum"}], "chart_filters": {"Product": ["Product 1"]}}, p)
    for t in compatible_types(p):
        spec = validate_chart_spec(adapt_spec(base, t, p), p)
        assert spec["chart_filters"] == {"Product": ["Product 1"]}


def test_old_saved_charts_still_load(sales_loaded):
    """Configs saved before the customisation fields existed must keep working."""
    _, p = sales_loaded
    legacy = {"id": "abc123", "type": "bar", "title": "Sales by region", "x": "Region", "y": "Sales", "agg": "sum",
              "sort": "value_desc", "top_n": 15, "time_grain": "auto", "auto": True}
    spec = validate_chart_spec(legacy, p)
    assert spec["extra"] == [] and spec["chart_filters"] == {} and spec["barmode"] == "group" and spec["width"] == "half"


# ------------------------------------------------------------------ field overrides

def test_override_identifier_to_category_and_metric_calculation(sales_loaded):
    res, p = sales_loaded
    assert p.columns["Customer ID"].role == IDENTIFIER
    q, warnings = apply_overrides(p, {"Customer ID": {"role": "dimension"}, "Sales": {"aggregation": "mean"}})
    assert not warnings
    assert q.columns["Customer ID"].role == DIMENSION
    assert q.columns["Sales"].aggregation == "mean"
    assert p.columns["Customer ID"].role == IDENTIFIER  # original profile untouched
    spec = validate_chart_spec({"type": "bar", "x": "Customer ID", "y": "Sales", "agg": "sum"}, q)
    data, _ = prepare_chart_data(res.df, spec, q)
    assert len(data) <= 15


def test_override_hide_and_invalid_roles(sales_loaded):
    _, p = sales_loaded
    q, warnings = apply_overrides(p, {"Region": {"role": "hidden"}, "Product": {"role": "metric"}, "Nope": {"role": "metric"}})
    assert q.columns["Region"].role == HIDDEN
    assert q.columns["Product"].role == DIMENSION  # text can't become a measure
    assert len(warnings) == 2
    assert METRIC not in allowed_roles(p.columns["Product"])
    with pytest.raises(ChartConfigError):
        validate_chart_spec({"type": "bar", "x": "Region", "y": "Sales"}, q)


def test_overrides_persist_in_dashboard_config(sales_loaded):
    res, p = sales_loaded
    cfg = build_default_dashboard(p, res.df, "d" * 16, "s.csv", "CSV data")
    cfg["column_overrides"] = {"Customer ID": {"role": "dimension"}, "Bad": {"role": "rm -rf"}, "Sales": "x"}
    clean, _ = validate_dashboard_config(cfg)
    assert clean["column_overrides"] == {"Customer ID": {"role": "dimension"}}


def test_numeric_category_filters_and_labels():
    df = pd.DataFrame({"Store": [1, 2, 3, 1, 2, 3] * 5, "Sales": [10.0, 20.0, 30.0, 5.0, 5.0, 5.0] * 5})
    res, p = load_and_profile(df)
    q, _ = apply_overrides(p, {"Store": {"role": "dimension"}})
    spec = validate_chart_spec({"type": "bar", "x": "Store", "y": "Sales", "agg": "sum", "chart_filters": {"Store": ["1"]}}, q)
    data, _ = prepare_chart_data(res.df, spec, q)
    assert list(data["Store"].astype(str)) == ["1"] and data["value"].iloc[0] == pytest.approx(75.0)
