"""Write sample_data/extremely_messy_workbook.xlsx (titles, two tables on a sheet, merged headers, a month-per-column
report and a sideways table). Usage: python scripts/make_messy_workbook.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests.messy_workbook import build  # noqa: E402

out = ROOT / "sample_data" / "extremely_messy_workbook.xlsx"
out.write_bytes(build())
print(f"Wrote {out}")
