import numpy as np
import pandas as pd

from src.data_profiler import COORDINATE, DATE, DIMENSION, EMPTY, IDENTIFIER, METRIC, TEXT, profile_dataset
from tests import synthetic as syn
from tests.conftest import load_and_profile


def roles(profile):
    return {c.name: c.role for c in profile.columns.values()}


def test_sales_roles_and_aggregations(sales_loaded):
    _, p = sales_loaded
    r = roles(p)
    assert r["Order ID"] == IDENTIFIER
    assert r["Customer ID"] == IDENTIFIER
    assert r["Order Date"] == DATE
    assert r["Region"] == DIMENSION and r["Product"] == DIMENSION
    assert r["Sales"] == METRIC and r["Quantity"] == METRIC
    assert p.columns["Sales"].aggregation == "sum"
    assert p.columns["Unit Price"].aggregation == "mean"  # prices are averaged, never summed
    assert p.metric_columns[0] == "Sales"


def test_marketing_rates_are_averaged(marketing_loaded):
    _, p = marketing_loaded
    assert roles(p)["Campaign ID"] == IDENTIFIER
    assert p.columns["CTR %"].aggregation == "mean"
    assert p.columns["CTR %"].is_percent
    assert p.columns["Clicks"].aggregation == "sum"


def test_inventory_and_expenses_profiles():
    _, inv = load_and_profile(syn.inventory())
    assert roles(inv)["SKU"] == IDENTIFIER
    assert inv.columns["Unit Cost"].aggregation == "mean"
    assert inv.metric_columns[0] == "Stock On Hand"
    _, exp = load_and_profile(syn.expenses())
    assert exp.columns["Tax Rate"].aggregation == "mean"
    assert "Department" in exp.chartable_dimensions


def test_missing_duplicates_and_stats():
    df = pd.DataFrame({
        "a": [1.0, 2.0, np.nan, 4.0, 4.0],
        "b": ["x", "y", None, "z", "z"],
        "c": [None] * 5,
    })
    p = profile_dataset(df)
    assert p.n_rows == 5 and p.n_cols == 3
    assert p.duplicate_rows == 1
    assert p.columns["a"].missing == 1 and p.columns["a"].missing_pct == 20.0
    assert p.columns["a"].stats["sum"] == 11.0 and p.columns["a"].stats["median"] == 3.0
    assert p.columns["c"].role == EMPTY
    assert p.columns["b"].unique == 3
    assert any("duplicates" in w for w in p.warnings)


def test_sequential_index_and_year_columns_are_not_metrics():
    df = pd.DataFrame({
        "Row": range(1, 101),
        "Year": [2020 + i % 4 for i in range(100)],
        "Month": [1 + i % 12 for i in range(100)],
        "Value": np.linspace(1, 50, 100),
    })
    p = profile_dataset(df)
    assert p.columns["Row"].role == IDENTIFIER
    assert p.columns["Year"].role == DIMENSION and p.columns["Year"].is_time_part
    assert p.columns["Month"].role == DIMENSION
    assert p.metric_columns == ["Value"]


def test_number_of_columns_are_metrics_not_ids():
    df = pd.DataFrame({"Number of Units": [1, 5, 3, 3, 9, 2, 8, 7, 1, 4, 6, 2], "Invoice Number": range(12)})
    p = profile_dataset(df)
    assert p.columns["Number of Units"].role == METRIC
    assert p.columns["Invoice Number"].role == IDENTIFIER


def test_free_text_and_high_cardinality():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "Comment": [f"This is a fairly long free text comment number {i} about the service quality" for i in range(200)],
        "Customer": [f"Customer {i}" for i in rng.integers(0, 120, 200)],
        "Amount": rng.normal(100, 10, 200),
    })
    p = profile_dataset(df)
    assert p.columns["Comment"].role == TEXT
    assert p.columns["Customer"].role == DIMENSION and p.columns["Customer"].high_cardinality
    assert "Customer" not in p.chartable_dimensions


def test_coordinates_detected():
    df = pd.DataFrame({"Latitude": [3.1, 5.4, 1.5] * 5, "Longitude": [101.7, 100.3, 103.8] * 5, "Sales": [5, 9, 2, 7, 7, 3, 8, 1, 4, 6, 2, 9, 5, 3, 8]})
    p = profile_dataset(df)
    assert p.columns["Latitude"].role == COORDINATE and p.coordinate_columns() == ("Latitude", "Longitude")
    assert p.metric_columns == ["Sales"]


def test_profile_to_dict_is_serialisable(sales_loaded):
    import json
    _, p = sales_loaded
    json.dumps(p.to_dict(), default=str)
    assert set(p.summary_frame().columns) >= {"Column", "Role", "Missing", "Unique"}


def test_body_measurements_are_averaged_not_summed():
    df = pd.DataFrame({"Gender": ["F", "M"] * 10, "Height": np.linspace(1.5, 1.9, 20), "Weight": np.linspace(50, 95, 20),
                       "BMI": np.linspace(18, 32, 20), "Units Sold": [3, 5, 2, 8, 1] * 4})
    p = profile_dataset(df)
    assert p.columns["Height"].aggregation == "mean"
    assert p.columns["Weight"].aggregation == "mean"
    assert p.columns["BMI"].aggregation == "mean"
    assert p.columns["Units Sold"].aggregation == "sum"
