"""
Basket analysis — per-use cost and packaging-waste comparison for matched
product pairs (wasteful format vs. low-waste format of the same brand).

Inputs:
    basket_completed_42.csv  — 42 products in 20 matched pairs, with prices
                               and source URLs (captured 2026-07-17/18).
Outputs:
    basket_results.csv       — per-row cost per use, packaging waste (grams)
                               per use, and tax-inclusive cost per use under
                               three illustrative waste-tax rates.
    basket_breakeven.csv     — per-pair cost delta, waste delta, and the
                               break-even waste tax tau* ($/kg) at which the
                               low-waste option becomes the cheaper one.

Method notes:
  * uses_per_package is normalized to a number (loads, cups, fills, shaves,
    servings...) in the ASSUMPTIONS table below, because the raw column is
    free text ("~34 6-fl-oz cups", "label-dependent yield").
  * package_waste_g is the estimated mass of packaging discarded over the
    life of the package. Durable components (razor handle, Brita pitcher,
    Wild case, reusable trigger sprayers) are excluded; consumable
    components (pods, cartridges, refill pouches, used paper-towel sheets)
    are included. These are hand estimates from label data and typical
    packaging masses; they are sensitivity-checked in the paper.
  * The two 20-fl-oz single-bottle soda rows are scaled to a 12-fl-oz
    serving basis (scale = 12/20) so they compare against the can rows.
  * Waste-tax scenarios: $0.11/kg, $0.30/kg, and $1.00/kg.
"""
import csv
import os

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IN_FILE = os.path.join(SCRIPT_DIR, "basket_completed_42.csv")
OUT_RESULTS = os.path.join(SCRIPT_DIR, "basket_results.csv")
OUT_BREAKEVEN = os.path.join(SCRIPT_DIR, "basket_breakeven.csv")

# Waste-tax scenarios in $/kg and the column suffix used for each.
TAUS = [(0.11, "011"), (0.30, "030"), (1.00, "100")]

# row index (file order) -> (uses_per_package, package_waste_g, scale, note)
#   uses_per_package : numeric uses over the life of the package
#   package_waste_g  : grams of packaging discarded over those uses
#   scale            : price/waste rescaling factor (1.0 except the 20-oz
#                      soda bottles, scaled to a 12-oz serving basis)
ASSUMPTIONS = {
    0:  (60,    140,  1.0, "HDPE jug 90oz ~140g; per load"),
    1:  (66,    55,   1.0, "compact auto-dose bottle ~55g; per load"),
    2:  (22,    107,  1.0, "pods 3.5g each + carton 30g; per cup"),
    3:  (34,    18,   1.0, "coffee bag ~18g; per cup"),
    4:  (1,     100,  1.0, "PET trigger bottle ~100g; per 26oz fill"),
    5:  (2.6,   75,   1.0, "refill bottle ~75g, trigger reused; per 26oz fill"),
    6:  (12,    322,  1.0, "cans 13.5g + 12pk carton 160g; per 12oz serving"),
    7:  (1,     26,   0.6, "PET 20oz scaled to 12oz-serving basis (special)"),
    8:  (4,     908,  1.0, "glass 355ml ~210g + cap + 4pk carton; per serving"),
    9:  (144,   120,  1.0, "disposable razors 10g each, 12 shaves each; per shave"),
    10: (150,   45,   1.0, "cartridges 4.5g each (handle durable); per shave"),
    11: (1,     70,   1.0, "pump bottle ~70g; per 11.25oz bottle"),
    12: (4.44,  45,   1.0, "refill jug ~45g, pump reused; per bottle-fill"),
    13: (1,     40,   1.0, "Dawn 18oz bottle ~40g; per bottle"),
    14: (3.72,  80,   1.0, "refill jug 67oz ~80g; per bottle-fill"),
    15: (984,   2629, 1.0, "cores+wrap; used sheet 2.5g each becomes trash; per sheet-use"),
    16: (750,   198,  1.0, "cloths 18g ea, ~75 uses each then discarded; per use"),
    17: (24,    240,  1.0, "PET water 16.9oz ~10g; per bottle-serving"),
    18: (303,   80,   1.0, "Brita filter ~80g per 40gal (pitcher durable, excluded); per 16.9oz"),
    19: (5.3,   10,   1.0, "PP cup+foil ~10g; per oz"),
    20: (32,    33,   1.0, "PP tub ~33g; per oz"),
    21: (1,     65,   1.0, "Method pump 10oz ~65g; per bottle"),
    22: (2.8,   35,   1.0, "refill ~35g; per bottle-fill"),
    23: (1,     90,   1.0, "Aveeno 33oz pump ~90g; per bottle"),
    24: (1.09,  30,   1.0, "refill pouch ~30g; per bottle-equiv"),
    25: (1,     90,   1.0, "CleanCult spray 16oz ~90g; per fill"),
    26: (2,     30,   1.0, "CleanCult refill carton ~30g; per fill"),
    27: (1,     100,  1.0, "Everspring spray 28oz ~100g; per fill"),
    28: (4,     45,   1.0, "ASSUME 1:3 dilution -> 4 fills (label-dependent; FLAGGED); per fill"),
    29: (1,     95,   1.0, "Powerwash starter w/ sprayer ~95g; per 16oz"),
    30: (1,     45,   1.0, "Powerwash refill ~45g, sprayer reused; per 16oz"),
    31: (1,     15,   1.0, "Wild starter: case durable (excluded), refill tube ~15g; per refill-cycle"),
    32: (1,     15,   1.0, "Wild refill tube ~15g; per refill-cycle"),
    33: (1,     75,   1.0, "Honest 18oz pump ~75g; per bottle"),
    34: (1.78,  35,   1.0, "refill ~35g; per bottle-equiv"),
    35: (12,    322,  1.0, "Sprite cans + carton; per 12oz serving"),
    36: (1,     26,   0.6, "Sprite PET 20oz scaled to 12oz basis (special)"),
    37: (8,     214,  1.0, "Pellegrino 11.15oz cans + 8pk carton; per serving"),
    38: (12,    268,  1.0, "Pellegrino PET 16.9oz ~19g + wrap; per serving"),
    39: (8,     2496, 1.0, "Pellegrino glass 16.9oz ~300g + cap + carton; per serving"),
    40: (1,     100,  1.0, "Windex spray (dup row 4); per fill"),
    41: (10,    20,   1.0, "Blueland 10 tablets, wrappers+carton ~20g; per fill"),
}


def fmt(x, nd=4):
    """Round to nd decimals and drop a trailing .0 on whole numbers."""
    r = float(np.round(x, nd))
    return str(int(r)) if r == int(r) else str(r)


def main():
    with open(IN_FILE, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != len(ASSUMPTIONS):
        raise SystemExit(f"expected {len(ASSUMPTIONS)} rows, got {len(rows)}")

    # ---- per-row results -------------------------------------------------
    results = []
    for idx, row in enumerate(rows):
        uses, waste_g, scale, note = ASSUMPTIONS[idx]
        price = float(row["price_usd"])
        cost_per_use = price * scale / uses          # $ per use
        waste_per_use = waste_g * scale / uses       # grams per use
        taxed = [cost_per_use + tau * waste_per_use / 1000.0 for tau, _ in TAUS]
        results.append({
            "idx": idx,
            "pair_id": int(row["pair_id"]),
            "category": row["category"],
            "product": row["product_name"][:55],
            "flag": int(row["refill_or_concentrate_flag"]),
            "cost": cost_per_use,
            "waste": waste_per_use,
            "note": note,
            "taxed": taxed,
        })

    with open(OUT_RESULTS, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "pair_id", "category", "product", "flag",
                    "cost_per_unit", "waste_g_per_unit", "note"]
                   + [f"tau_{suf}" for _, suf in TAUS])
        for r in results:
            w.writerow([r["idx"], r["pair_id"], r["category"], r["product"],
                        r["flag"], fmt(r["cost"]), fmt(r["waste"]), r["note"]]
                       + [fmt(t) for t in r["taxed"]])

    # ---- per-pair break-even ---------------------------------------------
    # Only pairs with exactly one designated low-waste row (flag == 1) allow a
    # clean baseline-vs-alternative comparison; all-flag-0 pairs (the soda and
    # Pellegrino packaging-arithmetic pairs) are excluded.
    n_cheaper = 0
    pairs = sorted({r["pair_id"] for r in results})
    with open(OUT_BREAKEVEN, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pair", "category", "dC_per_use", "dWaste_g", "tau_star"])
        for p in pairs:
            grp = [r for r in results if r["pair_id"] == p]
            low = [r for r in grp if r["flag"] == 1]
            base = [r for r in grp if r["flag"] == 0]
            if len(low) != 1 or not base:
                continue
            low, base = low[0], base[0]
            dC = low["cost"] - base["cost"]          # <0: low-waste already cheaper
            dW = base["waste"] - low["waste"]        # grams avoided per use
            if dW == 0:
                tau_star = ""                        # tax can never flip the choice
            elif dC <= 0:
                tau_star = fmt(0)                    # already cheaper: break-even at 0
                n_cheaper += 1
            else:
                tau_star = fmt(dC / dW * 1000.0)     # $/kg needed to flip
            w.writerow([p, grp[0]["category"], fmt(dC), fmt(dW), tau_star])

    print(f"Wrote {OUT_RESULTS} ({len(results)} rows) and {OUT_BREAKEVEN}.")
    print(f"Low-waste option already cheaper per use (tau*=0) in {n_cheaper} pairs.")


if __name__ == "__main__":
    main()
