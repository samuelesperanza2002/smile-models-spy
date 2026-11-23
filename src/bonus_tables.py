"""
Chapter 7 of the thesis: the two tables for the other underlyings (descriptive statistics, and
IV-RMSE by maturity), from data_<symbol>/results/smile_summary.csv and
model_results_<symbol>.csv. Regular periods only.
It writes results/tables/T11_bonus_desc.csv and T11_bonus_results.csv (aggregates only).

How to run (from the src folder):  python3 bonus_tables.py USO GLD
"""

import sys, pathlib
import numpy as np
import pandas as pd

HOME = pathlib.Path.home() / "thesis_spy"
OUT = HOME / "results" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
BUCKETS = ["0", "1", "2", "3-7", "30"]
symbols = [s.upper() for s in sys.argv[1:]] or ["USO"]
sample = pd.read_csv(HOME / "data" / "sample_dates.csv")
regular_t = set(pd.to_datetime(sample[sample.group == "regular"].t).dt.date)


def bucket(df):
    """Maturity group of each row: 0, 1, 2, 3-7 days, or 30 for the control expiry."""
    return pd.Categorical(np.where(df.control, "30", np.where(df.dte <= 2, df.dte.astype(str), "3-7")),
                          BUCKETS, ordered=True)


desc_rows, res_rows = [], []
for sym in symbols:
    root = HOME / f"data_{sym.lower()}" / "results"
    # ------------------------------------------------ descriptives: 10:30 on day t, regular periods
    s = pd.read_csv(root / "smile_summary.csv")
    s["day"] = pd.to_datetime(s.day).dt.date
    s = s[(s.snap.astype(str) == "1030") & (s.day.isin(regular_t))].copy()
    s["bucket"] = bucket(s)
    g = s.groupby("bucket", observed=True)
    d = pd.DataFrame(dict(smiles=g.size(), calibratable=g.calib_ok.sum(),
                          options_mean=g.n_options.mean(), puts_share=(g.n_puts.sum() / g.n_options.sum()),
                          sigma_atm_median=g.sigma_atm.median() * 100,
                          m_min_median=g.m_min.median(), m_max_median=g.m_max.median())).round(2)
    d.insert(0, "symbol", sym)
    desc_rows.append(d.reset_index())

    # ------------------------------------------------ errors: in sample (all test options) and intraday (common strikes)
    r = pd.read_csv(root / f"model_results_{sym.lower()}.csv")
    r = r[r.group == "regular"].copy()
    r["bucket"] = bucket(r)
    spec = [("BS", ["P1", "-"], "Black-Scholes"), ("VV", ["P1"], "Vanna-Volga (P1)"),
            ("SABR", ["P1"], "SABR (P1)"), ("SABR", ["odd/even"], "SABR (B)"), ("StickyStrike", ["-"], "Sticky strike")]
    for stage, region in (("insample", "all"), ("intraday", "common"), ("nextday", "common")):
        for model, psets, label in spec:
            sub = r[(r.stage == stage) & (r.region == region) & (r.model == model) & (r.pillar_set.isin(psets))]
            if not len(sub):
                continue
            g = sub.groupby("bucket", observed=True)
            row = dict(symbol=sym, stage=stage, model=label)
            for b in BUCKETS:
                if b in g.groups:
                    gb = g.get_group(b)
                    row[f"iv_rmse_{b}"] = round(gb.iv_rmse.mean(), 2)
                    row[f"n_{b}"] = int(len(gb))
                else:
                    row[f"iv_rmse_{b}"] = np.nan
                    row[f"n_{b}"] = 0
            res_rows.append(row)

desc = pd.concat(desc_rows, ignore_index=True)
res = pd.DataFrame(res_rows)
desc.to_csv(OUT / "T11_bonus_desc.csv", index=False)
res.to_csv(OUT / "T11_bonus_results.csv", index=False)
pd.set_option("display.width", 220)
print("TABLE A - cleaned smiles at 10:30 on day t, regular periods:")
print(desc.to_string(index=False))
print("\nTABLE B - mean IV-RMSE (volatility points) by maturity; in sample on all test options, "
      "out of sample on the strikes quoted at both snapshots; n = number of smiles:")
print(res.to_string(index=False))
print(f"\nSaved {OUT / 'T11_bonus_desc.csv'} and {OUT / 'T11_bonus_results.csv'}")
