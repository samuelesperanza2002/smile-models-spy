"""
Robustness to the data filters (Section 5.6.1 of the thesis): main filters of Chapter 4 against
the minimal filters used in the 0DTE literature (Andersen, Fusari and Todorov 2017; Bandi,
Fusari and Reno 2025): bid > 0, ask/bid <= 10, -15 < m < 5.

On the sample periods it compares the number of options per smile and the price-based measures
(price RMSE in dollars, share of model prices inside the bid-ask spread), which are the ones
that can be compared with the 0DTE literature. The IV-RMSE is also shown, but with the minimal
filters it is dominated by the tick size of very cheap options.

How to run (from the src folder):  python3 compare_filters.py
"""

import pathlib
import numpy as np
import pandas as pd

HOME = pathlib.Path.home() / "thesis_spy"
RESULTS = HOME / "results"
pd.set_option("display.width", 220)
BUCKETS = ["0", "1", "2", "3-7", "30"]
MODELS = ["BS", "SABR:P1", "VV:P1", "SABR(B)", "StickyStrike"]


def load(tag):
    """Results of one run (regular periods, main pillar set), with maturity group and model name."""
    r = pd.read_csv(RESULTS / f"model_results{tag}.csv", dtype={"period_id": str}, low_memory=False)
    r = r[(r.group == "regular") & (r.pillar_set.isin(["P1", "-", "odd/even"]))].copy()
    r["bucket"] = pd.Categorical(np.where(r.control, "30", np.where(r.dte <= 2, r.dte.astype(str), "3-7")), BUCKETS, ordered=True)
    r["model_key"] = r.model + np.where(r.design == "B", "(B)", "") + \
        np.where((r.model.isin(["SABR", "VV"])) & (r.design == "A"), ":" + r.pillar_set, "")
    return r


main, aft = load(""), load("_aft")
out = RESULTS / "tables"; out.mkdir(exist_ok=True)

print("SMILE SIZE at calibration (mean number of OTM options per smile, t 10:30):")
sz = pd.DataFrame({"main filters": main[(main.stage == "insample") & (main.model_key == "BS") & (main.region == "all")]
                   .groupby("bucket", observed=False).n_options.mean(),
                   "AFT filters": aft[(aft.stage == "insample") & (aft.model_key == "BS") & (aft.region == "all")]
                   .groupby("bucket", observed=False).n_options.mean()}).round(1)
print(sz.to_string()); sz.to_csv(out / "T9b_filters_smile_size.csv")

rows = []
for name, r in (("main", main), ("AFT", aft)):
    for stage in ("insample", "intraday", "nextday"):
        region = "all" if stage == "insample" else "common"
        d = r[(r.stage == stage) & (r.region == region)]
        for mk in MODELS:
            dd = d[d.model_key == mk]
            if not len(dd):
                continue
            g = dd.groupby("bucket", observed=False)
            for b, grp in g:
                if not len(grp):
                    continue
                rows.append(dict(filters=name, stage=stage, model=mk, bucket=b, n_smiles=len(grp),
                                 price_rmse=grp.price_rmse.mean(), price_mae=grp.price_mae.mean(),
                                 inside_spread=grp.inside_spread.mean(), iv_rmse=grp.iv_rmse.mean(),
                                 fail_rate=grp.fail_rate.mean()))
T = pd.DataFrame(rows)

for metric, label in [("inside_spread", "SHARE OF MODEL PRICES INSIDE THE BID-ASK SPREAD"),
                      ("price_rmse", "PRICE RMSE ($ per share)"), ("iv_rmse", "IV-RMSE (vol points)")]:
    print(f"\n{label} - main filters vs AFT filters, by stage, model and maturity:")
    piv = T.pivot_table(index=["stage", "model"], columns=["filters", "bucket"], values=metric, observed=False).round(3)
    piv = piv.reindex(columns=pd.MultiIndex.from_product([["main", "AFT"], BUCKETS]))
    print(piv.to_string())
    piv.to_csv(out / f"T9c_filters_{metric}.csv")
    with open(out / f"T9c_filters_{metric}.tex", "w") as f:
        f.write(piv.to_latex(float_format="%.3f", caption=f"Robustness to data filters: {label.lower()}",
                             label=f"tab:T9c_{metric}", escape=True))

# main numbers for the text: 0-7 days, in sample, share inside the spread and price RMSE
print("\nHEADLINE (in-sample, maturities 0-7 days, all options):")
h = T[(T.stage == "insample") & (T.bucket != "30")].groupby(["filters", "model"]).apply(
    lambda g: pd.Series(dict(inside_spread=np.average(g.inside_spread, weights=g.n_smiles),
                             price_rmse=np.average(g.price_rmse, weights=g.n_smiles),
                             iv_rmse=np.average(g.iv_rmse, weights=g.n_smiles))), include_groups=False).round(3)
print(h.unstack("filters").to_string())
print(f"\nSaved results/tables/T9b_filters_smile_size.csv and T9c_filters_*.csv")
