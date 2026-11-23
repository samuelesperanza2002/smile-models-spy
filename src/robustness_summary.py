"""
Robustness table of Chapter 5 (working name T9_robustness). It collects the main numbers of the
main run and of each robustness run (results/model_results<tag>.csv) in one table: mean IV-RMSE
by stage and model, for all maturities from 0 to 7 days and for the 0DTE alone, plus the share
of smiles where SABR beats Vanna-Volga (Design A, P1).

How to run (from the src folder):  python3 robustness_summary.py
"""

import pathlib
import numpy as np
import pandas as pd

HOME = pathlib.Path.home() / "thesis_spy"
RESULTS = HOME / "results"
pd.set_option("display.width", 220)

SPECS = [("main", ""), ("beta=0.5", "_beta0.5"), ("vega-weighted", "_vega"), ("business time", "_bustime"),
         ("calib 11:30", "_calib1130"), ("spread<=30%", "_spread30")]
MODELS = ["BS", "SABR:P1", "VV:P1", "SABR(B)", "StickyStrike"]

blocks = {}
for name, tag in SPECS:
    f = RESULTS / f"model_results{tag}.csv"
    if not f.exists():
        print(f"[{name}] file missing: {f.name}")
        continue
    r = pd.read_csv(f, dtype={"period_id": str}, low_memory=False)
    r = r[(r.group == "regular") & (r.region == "all") & (r.pillar_set.isin(["P1", "-", "odd/even"]))].copy()
    r["model_key"] = r.model + np.where(r.design == "B", "(B)", "") + \
        np.where((r.model.isin(["SABR", "VV"])) & (r.design == "A"), ":" + r.pillar_set, "")
    short = r[~r.control]
    col = {}
    for stage in ("insample", "intraday", "nextday"):
        for mk in MODELS:
            sub = short[(short.stage == stage) & (short.model_key == mk)]
            if len(sub):
                col[(stage, mk, "0-7d")] = sub.iv_rmse.mean()
                col[(stage, mk, "0-7d rel%")] = sub.iv_rmse_rel.mean()
                z = sub[sub.dte == 0]
                col[(stage, mk, "0DTE")] = z.iv_rmse.mean() if len(z) else np.nan
                col[(stage, mk, "0DTE rel%")] = z.iv_rmse_rel.mean() if len(z) else np.nan
    # share of smiles where SABR beats VV in sample (Design A, P1), and number of smiles
    piv = short[short.stage == "insample"].pivot_table(index=["period_id", "expiration"], columns="model_key", values="iv_rmse")
    if "SABR:P1" in piv.columns and "VV:P1" in piv.columns:
        d = (piv["SABR:P1"] - piv["VV:P1"]).dropna()
        col[("insample", "SABR vs VV", "share SABR wins")] = (d < 0).mean()
        col[("insample", "n smiles", "0-7d")] = len(d)
    blocks[name] = pd.Series(col)

T9 = pd.DataFrame(blocks)
T9.index = pd.MultiIndex.from_tuples(T9.index, names=["stage", "model", "maturity"])
T9 = T9.sort_index()
print("TABLE 9 - robustness: mean IV-RMSE (vol points), maturities 0-7 days and 0DTE only")
print(T9.round(2).to_string())
out = RESULTS / "tables"; out.mkdir(exist_ok=True)
T9.round(3).to_csv(out / "T9_robustness.csv")
with open(out / "T9_robustness.tex", "w") as f:
    f.write(T9.round(2).to_latex(caption="Robustness checks", label="tab:T9_robustness", escape=True))
print(f"\nSaved {out / 'T9_robustness.csv'}")
