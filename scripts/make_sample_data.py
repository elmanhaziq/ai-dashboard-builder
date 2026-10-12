"""Write synthetic sample files to ./sample_data for trying the app.

These are clearly synthetic test files. They are never loaded automatically and are
never mixed into a dashboard built from your own data.

    python scripts/make_sample_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests import synthetic as syn  # noqa: E402


def main() -> None:
    out = ROOT / "sample_data"
    out.mkdir(exist_ok=True)
    with pd.ExcelWriter(out / "synthetic_business_workbook.xlsx") as xw:
        pd.DataFrame({"About": ["Synthetic sample data for testing the dashboard builder."]}).to_excel(xw, sheet_name="README", index=False)
        syn.sales().to_excel(xw, sheet_name="Sales", index=False)
        syn.marketing().to_excel(xw, sheet_name="Marketing", index=False)
        syn.expenses().to_excel(xw, sheet_name="Expenses", index=False)
        syn.inventory().to_excel(xw, sheet_name="Inventory", index=False)
        pd.DataFrame(syn.messy_rows()).to_excel(xw, sheet_name="Messy", index=False, header=False)
    syn.sales().to_csv(out / "synthetic_sales.csv", index=False)
    pd.DataFrame(syn.messy_rows()).to_csv(out / "synthetic_messy.csv", index=False, header=False)
    try:
        import xlwt  # optional, only needed to produce a legacy .xls sample

        book = xlwt.Workbook()
        sheet = book.add_sheet("Expenses")
        df = syn.expenses()
        for j, c in enumerate(df.columns):
            sheet.write(0, j, c)
        for i, row in enumerate(df.itertuples(index=False), start=1):
            for j, v in enumerate(row):
                sheet.write(i, j, v.strftime("%Y-%m-%d") if hasattr(v, "strftime") else (v.item() if hasattr(v, "item") else v))
        book.save(str(out / "synthetic_expenses.xls"))
    except ImportError:
        pass
    print(f"Sample files written to {out}")


if __name__ == "__main__":
    main()
