"""Synthetic datasets used ONLY by tests and the optional sample-data script."""
from __future__ import annotations

import numpy as np
import pandas as pd


def marketing(n: int = 240, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n // 4, freq="D").repeat(4)[:n]
    channels = rng.choice(["Search", "Social", "Email", "Display"], n)
    impressions = rng.integers(1_000, 20_000, n)
    clicks = (impressions * rng.uniform(0.01, 0.06, n)).astype(int)
    conversions = (clicks * rng.uniform(0.02, 0.12, n)).astype(int)
    spend = np.round(clicks * rng.uniform(0.4, 1.6, n), 2)
    return pd.DataFrame({
        "Campaign ID": [f"CMP-{i:04d}" for i in range(n)],
        "Date": dates,
        "Channel": channels,
        "Campaign": rng.choice([f"Campaign {c}" for c in "ABCDEFGH"], n),
        "Impressions": impressions,
        "Clicks": clicks,
        "Conversions": conversions,
        "Spend": spend,
        "CTR %": np.round(clicks / impressions * 100, 2),
    })


def sales(n: int = 500, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    qty = rng.integers(1, 20, n)
    price = np.round(rng.uniform(5, 250, n), 2)
    return pd.DataFrame({
        "Order ID": np.arange(100001, 100001 + n),
        "Order Date": pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D"),
        "Region": rng.choice(["North", "South", "East", "West"], n, p=[0.4, 0.2, 0.2, 0.2]),
        "Product": rng.choice([f"Product {i}" for i in range(1, 13)], n),
        "Customer ID": rng.integers(1, 120, n),
        "Quantity": qty,
        "Unit Price": price,
        "Sales": np.round(qty * price, 2),
    })


def expenses(n: int = 200, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "Expense Date": pd.date_range("2024-01-01", periods=n, freq="2D"),
        "Department": rng.choice(["Finance", "HR", "IT", "Marketing", "Operations"], n),
        "Category": rng.choice(["Travel", "Software", "Hardware", "Office", "Training"], n),
        "Vendor": rng.choice([f"Vendor {c}" for c in "PQRSTUVW"], n),
        "Amount": np.round(rng.gamma(2.0, 400.0, n), 2),
        "Tax Rate": rng.choice([0.0, 0.06, 0.08], n),
    })


def inventory(n: int = 150, seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "SKU": [f"SKU{i:05d}" for i in range(n)],
        "Warehouse": rng.choice(["KL", "Penang", "Johor"], n),
        "Category": rng.choice(["Electronics", "Apparel", "Home", "Toys"], n),
        "Stock On Hand": rng.integers(0, 500, n),
        "Reorder Level": rng.integers(20, 100, n),
        "Unit Cost": np.round(rng.uniform(1, 80, n), 2),
        "Last Restock": pd.to_datetime("2025-06-01") + pd.to_timedelta(rng.integers(0, 120, n), unit="D"),
    })


def messy_rows() -> list[list]:
    """A messy sheet as a list of rows: title rows, blank rows/columns, duplicate and blank
    headers, currency strings, mixed types, a TOTAL row, duplicates and odd dates."""
    rows = [
        ["Quarterly Report (CONFIDENTIAL)", None, None, None, None, None, None],
        [None, None, None, None, None, None, None],
        ["Date", "Region", "Amount ($)", "Amount ($)", None, "Notes", "Discount %"],
    ]
    regions = ["North", "South", "East", None]
    for i in range(40):
        rows.append([
            f"{(i % 28) + 1:02d}/0{(i % 9) + 1}/2024",
            regions[i % 4],
            f"${1000 + i * 37:,}.50" if i % 7 else "n/a",
            str(10 + i),
            None,
            "ok" if i % 3 else None,
            f"{(i % 5) * 5}%",
        ])
    rows.append(rows[5][:])  # exact duplicate row
    rows.append([None] * 7)  # blank row inside data
    rows.append(["TOTAL", None, "$99,999.00", "999", None, None, None])
    return rows


def messy_frame() -> pd.DataFrame:
    return pd.DataFrame(messy_rows())
