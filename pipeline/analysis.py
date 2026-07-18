"""
Analysis for the Massachusetts PAYT panel.

Merges the town-by-year panel from etl_massdep.py with Census ACS 2023
5-year demographics (median household income, household size, population,
single-family share, owner-occupancy share; pulled via the free Census API
and cached in data_external/acs5.json) and land area (Census 2023 Gazetteer
county-subdivision file, auto-downloaded on first run), then runs the four
comparisons in the paper:

    1. Pooled OLS of log annual lbs per served household on PAYT status
       with demographic controls and year fixed effects, standard errors
       clustered by municipality.
    2. Nearest-neighbor matching: each PAYT town-year matched to the most
       demographically similar non-PAYT town in the same year.
    3. Alternative-denominator check: same regression with total households
       (not households served) as the denominator.
    4. Within-town switcher analysis: two-way fixed-effects regression on
       the towns that adopted or dropped PAYT inside the window.

Outputs:
    analysis_results.txt   full printed results
    regression_pooled.csv  coefficient table for the headline regression
"""
import io
import json
import os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_EXT = os.path.join(os.path.dirname(SCRIPT_DIR), "data_external")
PANEL_FILE = os.path.join(SCRIPT_DIR, "panel_town_year.csv")
ACS_CACHE = os.path.join(DATA_EXT, "acs5.json")
GAZ_CACHE = os.path.join(DATA_EXT, "2023_gaz_cousubs_25.txt")
OUT_TXT = os.path.join(SCRIPT_DIR, "analysis_results.txt")
OUT_REG = os.path.join(SCRIPT_DIR, "regression_pooled.csv")

ACS_VARS = ["B19013_001E",  # median household income
            "B25010_001E",  # average household size
            "B01003_001E",  # total population
            "B25024_001E", "B25024_002E", "B25024_003E",  # units in structure
            "B25003_001E", "B25003_002E"]                 # tenure / owner-occ
ACS_URL = ("https://api.census.gov/data/2023/acs/acs5?get=NAME,"
           + ",".join(ACS_VARS)
           + "&for=county%20subdivision:*&in=state:25")
GAZ_URL = ("https://www2.census.gov/geo/docs/maps-data/data/gazetteer/"
           "2023_Gazetteer/2023_gaz_cousubs_25.txt")

CONTROLS = ["log_income", "hh_size", "single_family_share",
            "owner_share", "log_density"]

# MassDEP name -> Census name, where the two disagree
ALIASES = {"manchester": "manchesterbythesea"}


def norm_town(name: str) -> str:
    """'Barnstable Town city, Barnstable County, MA' -> 'barnstable'."""
    n = str(name).split(",")[0].strip()
    for suffix in (" city", " town", " Town"):
        if n.endswith(suffix):
            n = n[: -len(suffix)]
    return "".join(ch for ch in n.lower() if ch.isalpha())


def load_acs() -> pd.DataFrame:
    if os.path.exists(ACS_CACHE):
        with open(ACS_CACHE, encoding="utf-8") as f:
            raw = json.load(f)
    else:
        import requests
        raw = requests.get(ACS_URL, timeout=60).json()
        os.makedirs(DATA_EXT, exist_ok=True)
        with open(ACS_CACHE, "w", encoding="utf-8") as f:
            json.dump(raw, f)
    df = pd.DataFrame(raw[1:], columns=raw[0])
    for v in ACS_VARS:
        df[v] = pd.to_numeric(df[v], errors="coerce")
        df.loc[df[v] <= -666666, v] = np.nan  # Census sentinel for N/A
    df = df[~df["NAME"].str.startswith("County subdivisions not defined")]
    out = pd.DataFrame({
        "town_key": df["NAME"].map(norm_town),
        "geoid": df["state"] + df["county"] + df["county subdivision"],
        "income": df["B19013_001E"],
        "hh_size": df["B25010_001E"],
        "population": df["B01003_001E"],
        "single_family_share": (df["B25024_002E"] + df["B25024_003E"])
                                / df["B25024_001E"],
        "owner_share": df["B25003_002E"] / df["B25003_001E"],
    })
    # a handful of towns appear in two counties; keep the larger piece
    return (out.sort_values("population", ascending=False)
               .drop_duplicates("town_key"))


def load_land_area() -> pd.DataFrame:
    if not os.path.exists(GAZ_CACHE):
        import requests
        r = requests.get(GAZ_URL, timeout=60)
        r.raise_for_status()
        os.makedirs(DATA_EXT, exist_ok=True)
        with open(GAZ_CACHE, "wb") as f:
            f.write(r.content)
    gaz = pd.read_csv(GAZ_CACHE, sep="\t", dtype={"GEOID": str})
    gaz.columns = [c.strip() for c in gaz.columns]
    out = pd.DataFrame({
        "geoid": gaz["GEOID"],
        "land_sqmi": pd.to_numeric(gaz["ALAND_SQMI"], errors="coerce"),
    })
    return out[out["land_sqmi"] > 0]


def nearest_neighbor_att(df: pd.DataFrame) -> tuple:
    """Match each PAYT town-year to the nearest same-year non-PAYT town on
    standardized controls (with replacement); return per-match log diffs."""
    z = df[CONTROLS].apply(lambda c: (c - c.mean()) / c.std())
    diffs = []
    for year, grp_idx in df.groupby("year").groups.items():
        grp = df.loc[grp_idx]
        zg = z.loc[grp_idx]
        treated = grp[grp["payt"] == 1]
        control = grp[grp["payt"] == 0]
        if treated.empty or control.empty:
            continue
        zc = zg.loc[control.index].to_numpy()
        for i in treated.index:
            d = np.linalg.norm(zc - zg.loc[i].to_numpy(), axis=1)
            j = control.index[int(np.argmin(d))]
            diffs.append(df.loc[i, "log_lbs"] - df.loc[j, "log_lbs"])
    return np.array(diffs)


def main():
    buf = io.StringIO()

    def log(*args):
        print(*args)
        print(*args, file=buf)

    panel = pd.read_csv(PANEL_FILE)
    panel["town_key"] = panel["municipality"].map(norm_town)
    panel["town_key"] = panel["town_key"].replace(ALIASES)

    acs = load_acs()
    land = load_land_area()
    acs = acs.merge(land, on="geoid", how="left")

    df = panel.merge(acs, on="town_key", how="left")
    unmatched = df[df["income"].isna()]["municipality"].unique()
    if len(unmatched):
        log(f"WARNING: no ACS match for {len(unmatched)} towns: "
            f"{sorted(unmatched)[:10]}")

    df["log_lbs"] = np.log(df["lbs_hh_yr"])
    df["log_income"] = np.log(df["income"])
    df["log_density"] = np.log(df["population"] / df["land_sqmi"])
    est = df.dropna(subset=["log_lbs"] + CONTROLS).copy()
    log(f"Estimation sample: {len(est)} town-years, "
        f"{est['municipality'].nunique()} towns "
        f"({int(est['payt'].sum())} PAYT town-years)\n")

    # ---- 0. raw gap -------------------------------------------------------
    raw = est.groupby("payt")["lbs_hh_yr"].mean()
    log("Raw means (lbs/hh/yr): non-PAYT "
        f"{raw[0]:.0f}, PAYT {raw[1]:.0f} "
        f"({raw[1] / raw[0] - 1:+.1%} raw gap)\n")

    # ---- 1. pooled OLS with controls, year FE, clustered SEs -------------
    formula = "log_lbs ~ payt + " + " + ".join(CONTROLS) + " + C(year)"
    m1 = smf.ols(formula, data=est).fit(
        cov_type="cluster", cov_kwds={"groups": est["municipality"]})
    log("=" * 72)
    log("1. Pooled OLS, log(lbs/hh/yr), year FE, SEs clustered by town")
    log(m1.summary().as_text())
    log(f"\nPAYT coefficient: {m1.params['payt']:+.4f} "
        f"(SE {m1.bse['payt']:.4f}) -> "
        f"{np.exp(m1.params['payt']) - 1:+.1%} with controls\n")
    pd.DataFrame({"coef": m1.params, "se": m1.bse,
                  "t": m1.tvalues, "p": m1.pvalues}).to_csv(OUT_REG)

    # ---- 2. nearest-neighbor matching ------------------------------------
    diffs = nearest_neighbor_att(est.reset_index(drop=True))
    t, p = stats.ttest_1samp(diffs, 0.0)
    log("=" * 72)
    log("2. Nearest-neighbor matching (same year, standardized controls)")
    log(f"   {len(diffs)} matches; ATT in logs = {diffs.mean():+.4f} "
        f"({np.exp(diffs.mean()) - 1:+.1%}), t = {t:.2f}, p = {p:.4f}\n")

    # ---- 3. alternative denominator: total households --------------------
    est["log_lbs_alt"] = np.log(est["trash_tons"] * 2000.0 / est["hh_total"])
    m3 = smf.ols("log_lbs_alt ~ payt + " + " + ".join(CONTROLS) + " + C(year)",
                 data=est).fit(cov_type="cluster",
                               cov_kwds={"groups": est["municipality"]})
    log("=" * 72)
    log("3. Alternative denominator (total households, not households served)")
    log(f"   PAYT coefficient: {m3.params['payt']:+.4f} "
        f"(SE {m3.bse['payt']:.4f}) -> {np.exp(m3.params['payt']) - 1:+.1%}\n")

    # ---- 4. within-town switchers ----------------------------------------
    var = est.groupby("municipality")["payt"].agg(["min", "max", "count"])
    switchers = var[(var["min"] == 0) & (var["max"] == 1)].index.tolist()
    log("=" * 72)
    log(f"4. Within-town switcher analysis: {len(switchers)} towns changed "
        f"PAYT status in-window:")
    for town in switchers:
        seq = est[est["municipality"] == town].sort_values("year")
        path = ", ".join(f"{int(y)}:{int(p)}"
                         for y, p in zip(seq["year"], seq["payt"]))
        log(f"     {town}: {path}")
    sw = est[est["municipality"].isin(switchers)]
    if len(switchers) >= 3:
        m4 = smf.ols("log_lbs ~ payt + C(municipality) + C(year)",
                     data=sw).fit(cov_type="cluster",
                                  cov_kwds={"groups": sw["municipality"]})
        log(f"\n   Two-way FE on switchers only ({len(sw)} town-years): "
            f"PAYT = {m4.params['payt']:+.4f} (SE {m4.bse['payt']:.4f}) "
            f"-> {np.exp(m4.params['payt']) - 1:+.1%}")

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    print(f"\nSaved {OUT_TXT} and {OUT_REG}")


if __name__ == "__main__":
    main()
