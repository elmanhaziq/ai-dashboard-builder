import pandas as pd
import pytest

from src.query_engine import QueryPlan, answer_question, execute_plan, NeedsClarification, suggested_questions
from tests import synthetic as syn
from tests.conftest import load_and_profile


def fmt(v):
    return f"{v:,.2f}"


def test_category_with_highest_total(sales_loaded):
    res, p = sales_loaded
    df = res.df
    r = answer_question("Which region has the highest total sales?", df, p)
    g = df.groupby("Region")["Sales"].sum().sort_values(ascending=False)
    assert r.ok
    assert f"**{g.index[0]}**" in r.answer and fmt(g.iloc[0]) in r.answer
    assert r.plan["intent"] == "rank" and r.plan["agg"] == "sum"


def test_average_per_group(sales_loaded):
    res, p = sales_loaded
    df = res.df
    r = answer_question("What is the average sales for each region?", df, p)
    g = df.groupby("Region")["Sales"].mean()
    assert r.ok and r.plan["agg"] == "mean"
    table = r.table.set_index("Region")["Average of Sales"]
    for region, v in g.items():
        assert table[region] == pytest.approx(v)


def test_month_with_highest_sales(sales_loaded):
    res, p = sales_loaded
    df = res.df
    r = answer_question("Which month had the highest sales?", df, p)
    m = df.groupby(df["Order Date"].dt.to_period("M"))["Sales"].sum()
    best = m.idxmax()
    assert r.ok and best.strftime("%b %Y") in r.answer and fmt(m.max()) in r.answer


def test_top_five_products_by_quantity(sales_loaded):
    res, p = sales_loaded
    df = res.df
    r = answer_question("What are the top five products by quantity?", df, p)
    top = df.groupby("Product")["Quantity"].sum().sort_values(ascending=False).head(5)
    assert r.ok and r.plan["n"] == 5
    for prod in top.index:
        assert prod in r.answer


def test_bottom_and_lowest(sales_loaded):
    res, p = sales_loaded
    df = res.df
    r = answer_question("Which region has the lowest total sales?", df, p)
    g = df.groupby("Region")["Sales"].sum()
    assert r.ok and f"**{g.idxmin()}**" in r.answer


def test_missing_values_question():
    df = pd.DataFrame({"a": [1, None, 3, None], "b": ["x", None, "y", "z"], "c": [1, 2, 3, 4]})
    res, p = load_and_profile(df)
    r = answer_question("Which columns have the most missing values?", res.df, p)
    assert r.ok and "**a**" in r.answer and "2 of 4" in r.answer


def test_overall_aggregates_and_value_filter(sales_loaded):
    res, p = sales_loaded
    df = res.df
    r = answer_question("total sales in North", df, p)
    assert r.ok and fmt(df.loc[df["Region"] == "North", "Sales"].sum()) in r.answer
    assert "Region = North" in r.answer
    r = answer_question("average quantity", df, p)
    assert fmt(df["Quantity"].mean()) in r.answer


def test_counts(sales_loaded):
    res, p = sales_loaded
    assert "**500**" in answer_question("How many rows are there?", res.df, p).answer
    r = answer_question("How many products are there?", res.df, p)
    assert r.ok and "**12**" in r.answer


def test_trends_outliers_and_summary(sales_loaded):
    res, p = sales_loaded
    for q in ["What trends or outliers are visible?", "Summarize the key findings in this dataset.",
              "Are there outliers in sales?", "What is the trend in sales?"]:
        r = answer_question(q, res.df, p)
        assert r.ok and len(r.answer) > 30, q


def test_correlation(marketing_loaded):
    res, p = marketing_loaded
    r = answer_question("What is the correlation between clicks and spend?", res.df, p)
    v = res.df["Clicks"].corr(res.df["Spend"])
    assert r.ok and f"r = {v:.2f}" in r.answer


@pytest.mark.parametrize("q", ["what is the weather today", "tell me a joke", "?", "asdf qwerty"])
def test_unsupported_questions_ask_to_rephrase(sales_loaded, q):
    res, p = sales_loaded
    r = answer_question(q, res.df, p)
    assert not r.ok
    assert any(w in r.answer.lower() for w in ("rephrase", "couldn't", "please", "which measure"))


def test_ambiguous_measure_asks_for_clarification():
    df = pd.DataFrame({"Category": list("ABCAB") * 4, "Cost": [3, 7, 2, 9, 4] * 4, "Revenue": [30, 12, 55, 41, 18] * 4})
    res, p = load_and_profile(df)
    r = answer_question("Which category has the highest total value?", res.df, p)
    assert not r.ok and "Which measure" in r.answer


def test_plans_are_validated(sales_loaded):
    res, p = sales_loaded
    with pytest.raises(NeedsClarification):
        execute_plan(QueryPlan(intent="rank", metric="Region", group="Product"), res.df, p)
    with pytest.raises(NeedsClarification):
        execute_plan(QueryPlan(intent="drop_table"), res.df, p)


def test_suggested_questions_are_answerable():
    for make in (syn.sales, syn.marketing, syn.expenses, syn.inventory):
        res, p = load_and_profile(make())
        for q in suggested_questions(p):
            assert answer_question(q, res.df, p).ok, q
