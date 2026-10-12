import pandas as pd
import pytest

from src.chart_generator import ChartConfigError
from src.dashboard_service import (
    apply_filters, build_default_dashboard, compute_kpi, validate_dashboard_config,
)
from tests import synthetic as syn
from tests.conftest import load_and_profile


def test_filters_do_not_mutate_original(sales_loaded):
    res, p = sales_loaded
    df = res.df
    before = df.copy()
    out = apply_filters(df, {"categorical": {"Region": ["North", "East"]}, "date": {}}, p)
    assert set(out["Region"]) == {"North", "East"}
    assert len(out) == int(df["Region"].isin(["North", "East"]).sum())
    pd.testing.assert_frame_equal(df, before)


def test_date_filter_is_inclusive(sales_loaded):
    res, p = sales_loaded
    df = res.df
    out = apply_filters(df, {"date": {"Order Date": ["2024-03-01", "2024-03-31"]}}, p)
    expected = df[(df["Order Date"] >= "2024-03-01") & (df["Order Date"] < "2024-04-01")]
    assert len(out) == len(expected) and len(out) > 0


def test_filters_update_kpis_consistently(sales_loaded):
    res, p = sales_loaded
    df = res.df
    cfg = build_default_dashboard(p, df, "a" * 16, "sales.csv", "CSV data")
    filtered = apply_filters(df, {"categorical": {"Region": ["West"]}}, p)
    kpi = next(k for k in cfg["kpis"] if k.get("column") == "Sales")
    assert compute_kpi(filtered, kpi, p)["value"] == pytest.approx(df.loc[df["Region"] == "West", "Sales"].sum())
    count = next(k for k in cfg["kpis"] if k["kind"] == "count")
    assert compute_kpi(filtered, count, p)["value"] == len(filtered)


def test_blank_category_filter():
    df = pd.DataFrame({"Region": ["A", None, "B", None], "Sales": [1, 2, 3, 4]})
    res, p = load_and_profile(df)
    out = apply_filters(res.df, {"categorical": {"Region": ["(blank)"]}}, p)
    assert out["Sales"].sum() == 6


def test_period_over_period_uses_complete_periods():
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2024-01-01", "2024-01-20", "2024-02-10", "2024-02-25", "2024-03-03"]),
        "Amount": [100, 100, 150, 150, 999],
        "Group": ["a", "b", "a", "b", "a"],
    })
    res, p = load_and_profile(df)
    kpi = {"kind": "agg", "column": "Amount", "agg": "sum", "compare_date": "Date", "label": "Sum of Amount"}
    out = compute_kpi(res.df, kpi, p)
    # March is incomplete, so Feb (300) is compared with Jan (200): +50%
    assert out["delta"] == "+50.0%"
    assert "Feb 2024 vs Jan 2024" in out["delta_label"]


def test_dashboard_config_validation_drops_invalid_parts(sales_loaded):
    res, p = sales_loaded
    cfg = build_default_dashboard(p, res.df, "b" * 16, "sales.csv", "CSV data")
    cfg["charts"].append({"type": "bar", "x": "DeletedColumn", "y": "Sales", "title": "Broken"})
    cfg["charts"].append({"type": "exec", "code": "import os"})
    cfg["kpis"].append({"kind": "agg", "column": "Region", "agg": "sum"})
    clean, warnings = validate_dashboard_config(cfg, p)
    assert len(clean["charts"]) == len(cfg["charts"]) - 2
    assert len(clean["kpis"]) == len(cfg["kpis"]) - 1
    assert len(warnings) == 3


@pytest.mark.parametrize("bad", [None, [], {"id": "../../etc"}, {"id": "abcdef12", "dataset_id": "..\\x"}])
def test_structurally_invalid_configs_raise(bad):
    with pytest.raises(ChartConfigError):
        validate_dashboard_config(bad)


def test_default_dashboard_for_every_synthetic_dataset():
    for make in (syn.sales, syn.marketing, syn.expenses, syn.inventory):
        res, p = load_and_profile(make())
        cfg = build_default_dashboard(p, res.df, "c" * 16, "x.csv", "CSV data")
        assert len(cfg["kpis"]) >= 3 and len(cfg["charts"]) >= 4
        assert cfg["filter_columns"]
