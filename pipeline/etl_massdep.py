"""
ETL for the MassDEP Municipal Solid Waste & Recycling Survey, CY2021-CY2025.

Reads the five annual survey Excel files (all 351 Massachusetts
municipalities), locates the key columns by header name — the five years are
formatted differently (different sheet names, header rows, column counts and
spellings) — applies quality screens, and writes a clean town-by-year panel.

Inputs (same folder as this script; links in the paper's Appendix A):
    MuniData21.xlsx           CY2021 survey
    MuniData22.xlsx           CY2022 survey
    MuniData2023.xlsx         CY2023 survey
    MuniData2024_0.xlsx       CY2024 survey
    MuniData2025-a11y.xlsx    CY2025 survey

Output:
    panel_town_year.csv       one row per municipality-year that passes the
                              quality screens, with PAYT status, households,
                              trash and recycling tonnage, and lbs/hh/year.

Quality screens (documented in docs/evidence_ledger.csv, claim E1):
    * households served by the municipal trash program > 0 and trash
      disposal tonnage > 0 (a town that reports neither has no municipal
      program we can measure);
    * 200 < annual lbs per served household < 8,000 (outside this range the
      denominator and numerator almost certainly cover different populations
      — e.g. transfer-station tonnage divided by a stale household count);
    * coverage ratio (households served / total households) between 30% and
      150% (screens out partial-service towns and obvious misreports).
"""
import os
import re

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_FILE = os.path.join(SCRIPT_DIR, "panel_town_year.csv")

FILES = {
    2021: "MuniData21.xlsx",
    2022: "MuniData22.xlsx",
    2023: "MuniData2023.xlsx",
    2024: "MuniData2024_0.xlsx",
    2025: "MuniData2025-a11y.xlsx",
}

# canonical name -> list of lowercase substrings that identify the column.
# First pattern that matches a header wins; patterns are tried in order.
COLUMN_PATTERNS = {
    "hh_total":     ["total number of households"],
    "hh_trash":     ["households served by municipal trash"],
    "hh_recycling": ["households served by municipal recycling"],
    "svc_type":     ["trash service type"],
    "payt":         ["payt/ smart", "payt"],          # NOT the "funded by" column
    "trash_tons":   ["trash disposal tonnage"],
    "bulky_incl":   ["tonnage include bulky"],
    "recyc_tons":   ["single stream recyclables tons"],
}

SCREEN_LBS_LO, SCREEN_LBS_HI = 200, 8000     # annual lbs per served household
SCREEN_COV_LO, SCREEN_COV_HI = 0.30, 1.50    # served / total households


def find_data_sheet(xl: pd.ExcelFile) -> str:
    """The data sheet is whichever sheet isn't the Key/READ ME sheet."""
    candidates = [s for s in xl.sheet_names
                  if not re.match(r"^(key|read\s*me)$", s.strip(), re.I)]
    if len(candidates) != 1:
        raise ValueError(f"ambiguous data sheet among {xl.sheet_names}")
    return candidates[0]


def find_header_row(raw: pd.DataFrame) -> int:
    """Header row is the one containing 'Total Number of Households'."""
    for i in range(min(8, len(raw))):
        if raw.iloc[i].astype(str).str.contains(
                "total number of households", case=False).any():
            return i
    raise ValueError("header row not found")


def locate_columns(headers) -> dict:
    """Map canonical names to column positions by fuzzy header match."""
    cols = {}
    lower = [str(h).strip().lower() if pd.notna(h) else "" for h in headers]
    for name, patterns in COLUMN_PATTERNS.items():
        for pat in patterns:
            hits = [j for j, h in enumerate(lower)
                    if pat in h and "funded by" not in h]
            if hits:
                cols[name] = hits[0]
                break
        else:
            raise ValueError(f"column not found: {name}")
    return cols


def yes_no(v):
    s = str(v).strip().lower()
    if s.startswith("y"):
        return 1
    if s.startswith("n"):
        return 0
    return None


def load_year(path: str, year: int) -> pd.DataFrame:
    xl = pd.ExcelFile(path)
    sheet = find_data_sheet(xl)
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    hr = find_header_row(raw)
    cols = locate_columns(raw.iloc[hr])

    body = raw.iloc[hr + 1:]
    df = pd.DataFrame({
        "municipality": body.iloc[:, 0].astype(str).str.strip(),
        "year": year,
    })
    for name, j in cols.items():
        df[name] = body.iloc[:, j].values

    # drop footer/notes rows: municipality must be a plausible town name
    df = df[df["municipality"].str.len().between(2, 40)]
    df = df[~df["municipality"].str.lower().isin(["nan", "municipality",
                                                  "municipality name", "total"])]

    for c in ["hh_total", "hh_trash", "hh_recycling", "trash_tons", "recyc_tons"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["payt"] = df["payt"].map(yes_no)
    df["bulky_incl"] = df["bulky_incl"].map(yes_no)
    df["svc_type"] = df["svc_type"].astype(str).str.strip()
    return df


def main():
    frames = []
    for year, fname in FILES.items():
        path = os.path.join(SCRIPT_DIR, fname)
        df = load_year(path, year)
        print(f"{year}: {len(df)} municipalities read from {fname}")
        frames.append(df)
    panel = pd.concat(frames, ignore_index=True)

    # derived measures
    panel["lbs_hh_yr"] = panel["trash_tons"] * 2000.0 / panel["hh_trash"]
    panel["coverage"] = panel["hh_trash"] / panel["hh_total"]

    # quality screens
    ok = (
        panel["payt"].notna()
        & (panel["hh_trash"] > 0)
        & (panel["trash_tons"] > 0)
        & panel["lbs_hh_yr"].between(SCREEN_LBS_LO, SCREEN_LBS_HI, inclusive="neither")
        & panel["coverage"].between(SCREEN_COV_LO, SCREEN_COV_HI)
    )
    usable = panel[ok].copy()
    usable["payt"] = usable["payt"].astype(int)

    usable = usable[["municipality", "year", "payt", "svc_type",
                     "hh_total", "hh_trash", "hh_recycling",
                     "trash_tons", "recyc_tons", "bulky_incl",
                     "lbs_hh_yr", "coverage"]]
    usable.to_csv(OUT_FILE, index=False)

    print(f"\nPanel: {len(panel)} town-years read, {len(usable)} usable "
          f"after screens -> {OUT_FILE}")
    print("\nMean annual lbs per served household, by PAYT status:")
    print(usable.groupby(["year", "payt"])["lbs_hh_yr"]
          .agg(["count", "mean", "median"]).round(0))


if __name__ == "__main__":
    main()
