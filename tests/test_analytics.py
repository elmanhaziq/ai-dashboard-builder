"""Share, growth, two-way tables and comparisons — every number checked against pandas directly."""
import pandas as pd
import pytest

from src.query_engine import NeedsClarification, QueryPlan, answer_question, execute_plan


def test_share_of_total(sales_loaded):
    res, profile = sales_loaded
    df = res.df
    out = answer_question("What share of Sales does each Region contribute?", df, profile)
    assert out.ok and out.plan["intent"] == "share"
    expected = (df.groupby("Region")["Sales"].sum() / df["Sales"].sum() * 100).round(1)
    t = out.table.set_index("Region")["Share %"]
    assert t.to_dict() == expected.to_dict()
    assert abs(t.sum() - 100) < 0.5
    assert f"**{expected.idxmax()}** has the largest share" in out.answer


def test_share_refuses_averages(sales_loaded):
    res, profile = sales_loaded
    with pytest.raises(NeedsClarification):
        execute_plan(QueryPlan(intent="share", metric="Unit Price", group="Region", agg="mean"), res.df, profile)


def test_growth_month_over_month(sales_loaded):
    res, profile = sales_loaded
    df = res.df
    out = answer_question("How did Sales change compared to the previous month?", df, profile)
    assert out.ok and out.plan["intent"] == "growth"
    monthly = df.groupby(df["Order Date"].dt.to_period("M"))["Sales"].sum()
    last, prev = monthly.iloc[-1], monthly.iloc[-2]
    pct = abs((last - prev) / prev * 100)
    assert f"{pct:.1f}%" in out.answer
    assert list(out.table.columns) == ["Month", "Sum of Sales", "Change", "Change %"]


def test_growth_with_biggest_movers(sales_loaded):
    res, profile = sales_loaded
    plan = QueryPlan(intent="growth", metric="Sales", agg="sum", date_col="Order Date", time_grain="M", group="Region")
    out = execute_plan(plan, res.df, profile)
    assert out.ok and ("Biggest increases" in out.answer or "Biggest drops" in out.answer)


def test_two_way_table(sales_loaded):
    res, profile = sales_loaded
    df = res.df
    out = answer_question("Total Sales by Region and Product", df, profile)
    assert out.ok and out.plan["intent"] == "pivot"
    t = out.table.set_index("Region")
    pair = df.groupby(["Region", "Product"])["Sales"].sum()
    a, b = t.stack().idxmax()
    assert round(pair[(a, b)], 2) == round(t.loc[a, b], 2)
    assert f"**{a} × {b}**" in out.answer


def test_compare_two_regions(sales_loaded):
    res, profile = sales_loaded
    df = res.df
    out = answer_question("Compare Sales for North vs South", df, profile)
    assert out.ok and out.plan["intent"] == "compare"
    totals = df.groupby("Region")["Sales"].sum()
    hi, lo = ("North", "South") if totals["North"] >= totals["South"] else ("South", "North")
    assert out.answer.startswith(f"**{hi}**") and f"**{lo}**" in out.answer
    assert set(out.table["Region"]) == {"North", "South"}


def test_query_tool_runs_new_intents(sales_loaded):
    from src.agent.copilot import DashboardData
    from src.agent.tools import TOOLS, t_query_data, validate_args

    res, profile = sales_loaded
    ctx = DashboardData(res.df, profile)
    base = {"intent": "pivot", "metric": "Sales", "metric2": "", "group_by": "Region", "group_by_2": "Product",
            "aggregation": "sum", "top_n": 0, "order": "desc", "date_column": "", "time_grain": "", "filters": [],
            "compare_values": []}
    out = t_query_data(ctx, validate_args(TOOLS["query_data"].schema, base))
    assert out["answered"] and out["table"]
    out = t_query_data(ctx, validate_args(TOOLS["query_data"].schema, dict(base, intent="compare", group_by_2="",
                                                                            compare_values=["West", "East"])))
    assert out["answered"] and "vs" in out["result"]
    out = t_query_data(ctx, validate_args(TOOLS["query_data"].schema, dict(base, intent="growth", group_by="", group_by_2="")))
    assert out["answered"] and "from" in out["result"]


def test_existing_questions_still_route_the_same(sales_loaded):
    res, profile = sales_loaded
    assert answer_question("Top 5 Product by Sales", res.df, profile).plan["intent"] == "rank"
    assert answer_question("Sales trend over time", res.df, profile).plan["intent"] == "trend"
    assert answer_question("Total Sales by Region", res.df, profile).plan["intent"] == "group_agg"
