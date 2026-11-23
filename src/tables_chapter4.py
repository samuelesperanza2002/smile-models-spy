"""
Tables of Chapter 4 of the thesis (data and methodology), built from aggregated files only:
  T1   sample periods: dates, regime, VIX, number of smiles and of OTM options at t 10:30
  T2   filter funnel on the OTM options by maturity, 10:30 snapshot (sample days and all days)
  T2b  descriptive statistics of the cleaned smiles at 10:30 (sample days)
Each table is saved in results/tables/ as CSV and as LaTeX.

How to run (from the src folder):  python3 tables_chapter4.py
"""

import pathlib
import numpy as np
import pandas as pd

HOME = pathlib.Path.home() / "thesis_spy"
RESULTS, DATA = HOME / "results", HOME / "data"
OUT = RESULTS / "tables"; OUT.mkdir(exist_ok=True)
pd.set_option("display.width", 220)
BUCKETS = ["0", "1", "2", "3-7", "30"]

cnt = pd.read_csv(RESULTS / "filter_counts.csv", parse_dates=["day", "expiration"])
sm = pd.read_csv(RESULTS / "smile_summary.csv", parse_dates=["day", "expiration"])
sample = pd.read_csv(DATA / "sample_dates.csv", parse_dates=["t", "t1"])
for df in (cnt, sm):
    df["bucket"] = pd.Categorical(np.where(df.control, "30", np.where(df.dte <= 2, df.dte.astype(str), "3-7")), BUCKETS, ordered=True)
sample_days = set(sample.t) | set(sample.t1)

# ------------------------------------------------------------ T1: sample periods
rows = []
for _, s in sample.iterrows():
    z = sm[(sm.day == s.t) & (sm.snap == 1030)]
    z = z[(z.dte <= 7) | z.control]
    calib = z[z.calib_ok]
    rows.append(dict(period=s.period_id, t=s.t.date(), t1=s.t1.date(), weekday=s.weekday_t, regime=s.regime,
                     vix_tm1=s.vix_tm1, vix_t=s.vix_t, vix_t1=s.vix_t1,
                     n_smiles_calib=len(calib), n_options_1030=int(z.n_options.sum()),
                     sigma_atm_1dte=float(z[z.dte == 1].sigma_atm.iloc[0] * 100) if (z.dte == 1).any() else np.nan,
                     n_0dte_1030=int(z[z.dte == 0].n_options.sum()) if (z.dte == 0).any() else 0))
T1 = pd.DataFrame(rows)
print("TABLE 1 - sample periods (t = calibration day; options and smiles at t 10:30 after all filters):")
print(T1.round(1).to_string(index=False))
T1.round(2).to_csv(OUT / "T1_sample.csv", index=False)
with open(OUT / "T1_sample.tex", "w") as f:
    f.write(T1.round(1).to_latex(index=False, caption="Sample periods", label="tab:T1_sample", escape=True))
print("\nTotals: smiles calibrated at t 10:30 =", T1.n_smiles_calib.sum(), "| OTM options at t 10:30 =", T1.n_options_1030.sum())
print(T1.groupby("regime")[["n_smiles_calib", "n_options_1030", "n_0dte_1030"]].agg(["sum", "mean"]).round(1).to_string())

# ------------------------------------------------------------ T2: OTM filter funnel, 10:30
funnel = ["otm_n0", "otm_f1_bid_ask", "otm_f2_mid", "otm_f3_spread", "otm_f4_liquidity", "f6_moneyness", "f7_noarb", "f8_iv", "f9_final", "f9b_calib"]
labels = ["OTM quotes", "1. bid > 0, ask > bid", "2. mid >= 0.05", "3. spread <= 50%", "4. volume > 0 or OI >= 100",
          "6. -6 <= m <= 3", "7. no-arbitrage", "8. IV exists", "9. >= 4 strikes (kept)", "9b. >= 8 strikes (calibratable)"]
c10 = cnt[cnt.snap == 1030].copy()
c10["in_sample"] = c10.day.isin(sample_days)
for name, d in (("sample days (40)", c10[c10.in_sample]), ("all trading days (977)", c10)):
    tab = d.groupby("bucket", observed=False)[funnel].sum()
    tab.loc["total"] = tab.sum()
    tab.columns = labels
    print(f"\nTABLE 2 - OTM filter funnel at 10:30, {name} (number of options surviving each filter):")
    print(tab.T.to_string())
    pct = (tab.div(tab["OTM quotes"], axis=0) * 100).round(1)
    print("as % of OTM quotes:"); print(pct.T.to_string())
    tag = "sample" if "sample" in name else "all"
    tab.T.to_csv(OUT / f"T2_filters_{tag}.csv"); pct.T.to_csv(OUT / f"T2_filters_{tag}_pct.csv")
    with open(OUT / f"T2_filters_{tag}.tex", "w") as f:
        f.write(tab.T.to_latex(float_format="%.0f", caption=f"Filter funnel on OTM options, 10:30 snapshot, {name}", label=f"tab:T2_filters_{tag}", escape=True))
print("\nSmiles dropped (10:30, all days):", cnt[(cnt.snap == 1030) & (cnt.f9_final == 0)].note.fillna("").str.replace(r"only \d+ strikes", "few strikes", regex=True).value_counts().to_dict())

# ------------------------------------------------------------ T2b: descriptive statistics, sample days, 10:30
s10 = sm[(sm.snap == 1030) & sm.day.isin(sample_days)]
T2b = s10.groupby("bucket", observed=False).agg(n_smiles=("n_options", "size"), calibratable=("calib_ok", "mean"),
                                              options_mean=("n_options", "mean"), options_min=("n_options", "min"), options_max=("n_options", "max"),
                                              puts_share=("n_puts", lambda x: x.sum() / s10.loc[x.index, "n_options"].sum()),
                                              sigma_atm_median=("sigma_atm", lambda x: x.median() * 100),
                                              m_min_median=("m_min", "median"), m_max_median=("m_max", "median"),
                                              tau_days_median=("tau", lambda x: x.median() * 365),
                                              F_disp_median=("F_disp", "median"), noarb_removed=("n_noarb_removed", "sum")).round(2)
print("\nTABLE 2b - cleaned smiles at 10:30, sample days (t and t+1):"); print(T2b.to_string())
T2b.to_csv(OUT / "T2b_descriptives.csv")
with open(OUT / "T2b_descriptives.tex", "w") as f:
    f.write(T2b.to_latex(float_format="%.2f", caption="Descriptive statistics of the cleaned smiles", label="tab:T2b_descriptives", escape=True))
print("\nSaved T1_sample, T2_filters_sample/all, T2b_descriptives (csv + tex) in results/tables/")
