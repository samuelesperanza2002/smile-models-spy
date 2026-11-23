"""
Quality checks on the output of clean.py --all. It reads only aggregated files (results/*.csv)
and the VIX, and prints: the status of each snapshot, the filter funnel, the availability of the
smiles, the consistency of the forward across expiries, the ATM volatility by regime and the
effect of the no-arbitrage filter.

How to run (from the src folder):  python3 summarize_clean.py
"""

import pathlib
import numpy as np
import pandas as pd

HOME = pathlib.Path.home() / "thesis_spy"
RES, DATA = HOME / "results", HOME / "data"
pd.set_option("display.width", 200)

cnt = pd.read_csv(RES / "filter_counts.csv", parse_dates=["day", "expiration"])
sm = pd.read_csv(RES / "smile_summary.csv", parse_dates=["day", "expiration"])
log = pd.read_csv(RES / "clean_log.csv")

# maturity buckets: 0, 1, 2, 3-7, 30 (control)
def bucket(df):
    """Maturity group of each row: 0, 1, 2, 3-7 days, or 30 for the control expiry."""
    b = np.where(df.control, "30", np.where(df.dte <= 2, df.dte.astype(str), "3-7"))
    return pd.Categorical(b, ["0", "1", "2", "3-7", "30"], ordered=True)
cnt["bucket"], sm["bucket"] = bucket(cnt), bucket(sm)

# regime of each day from the VIX close of the day before (same thresholds as select_sample.py)
vix = pd.read_csv(DATA / "fred_VIXCLS.csv"); vix.columns = ["date", "v"]
vix["date"] = pd.to_datetime(vix["date"]); vix["v"] = pd.to_numeric(vix["v"], errors="coerce")
vix = vix.dropna().set_index("date")["v"]
def regime(day):
    """VIX regime of a day, from the VIX close of the day before."""
    prev = vix[vix.index < day]
    if not len(prev): return "n.a."
    v = prev.iloc[-1]
    return "calm" if v < 15 else "normal" if v <= 20 else "stressed" if v > 25 else "20-25"
reg = {d: regime(d) for d in sm.day.unique()}
sm["regime"] = sm.day.map(reg); cnt["regime"] = cnt.day.map(reg)

print("=" * 90); print("1) SNAPSHOT STATUS"); print("=" * 90)
print(log.groupby(["snap", log.status.str.split(" \\(").str[0]]).size().to_string())

print("\n" + "=" * 90); print("2) FILTER FUNNEL (sum over all 10:30 and 14:30 snapshots), by maturity"); print("=" * 90)
fcols = ["n0_quote", "f1_bid_ask", "f2_mid", "f3_spread", "f4_liquidity", "f5_otm", "f6_moneyness",
         "f7_noarb", "f8_iv", "f9_final", "f9b_calib"]
tab = cnt.groupby("bucket", observed=True)[fcols].sum()
tab.loc["total"] = tab.sum()
print(tab.to_string())
print("\nShare remaining after each filter (relative to the quotes with bid>0 and ask>bid):")
print((tab[fcols[1:]].div(tab["f1_bid_ask"], axis=0) * 100).round(1).to_string())
print("\nReasons for dropping smiles (non-empty notes):")
print(cnt.note.fillna("").replace("", "ok").str.replace(r"only \\d+ strikes", "few strikes", regex=True)
      .value_counts().to_string())

print("\n" + "=" * 90); print("3) AVAILABILITY: smiles by maturity and snapshot, and share of calibratable smiles (>= 8 strikes)"); print("=" * 90)
avail = sm.groupby(["snap", "bucket"], observed=True).agg(n_smiles=("n_options", "size"),
        share_calib=("calib_ok", "mean"), mean_options=("n_options", "mean"), median_strikes=("n_strikes", "median"))
print(avail.round(2).to_string())
print("\nShare of calibratable smiles by regime (VIX of the day before) x maturity, snapshot 14:30:")
print(sm[sm.snap == 1430].pivot_table(index="regime", columns="bucket", values="calib_ok", aggfunc="mean", observed=True).round(2).to_string())
print("\nMean options per smile, regime x maturity, snapshot 10:30:")
print(sm[sm.snap == 1030].pivot_table(index="regime", columns="bucket", values="n_options", aggfunc="mean", observed=True).round(1).to_string())

print("\n" + "=" * 90); print("4) FORWARD: consistency across expiries (F_dte0 - F_dte1, expected about -0.07 $ of carry)"); print("=" * 90)
w = sm.pivot_table(index=["day", "snap"], columns="dte", values="F")
for a, b in [(0, 1), (1, 2), (0, 2)]:
    if a in w.columns and b in w.columns:
        d = (w[a] - w[b]).dropna()
        print(f"F(dte {a}) - F(dte {b}): n={len(d)}, median={d.median():.3f}, 10%={d.quantile(.1):.3f}, "
              f"90%={d.quantile(.9):.3f}  | by snapshot: " +
              ", ".join(f"{s}: {d.xs(s, level='snap').median():.3f}" for s in [1030, 1430] if s in d.index.get_level_values('snap')))
print("\nDispersion of the 8 estimates of F (F_disp, $) by maturity: median and 90th percentile")
print(sm.groupby("bucket", observed=True).F_disp.quantile([.5, .9]).unstack().round(3).to_string())
print("\n|F_reg - F| median by maturity ($):", (sm.F_reg - sm.F).abs().groupby(sm.bucket, observed=True).median().round(4).to_dict())
print("D_reg: 10%/50%/90% =", sm.D_reg.quantile([.1, .5, .9]).round(4).tolist(), "| mean D (T-bill) by maturity:",
      sm.groupby("bucket", observed=True).D.mean().round(5).to_dict())

print("\n" + "=" * 90); print("5) Median ATM VOLATILITY (%) by regime x maturity, 10:30"); print("=" * 90)
print((sm[sm.snap == 1030].pivot_table(index="regime", columns="bucket", values="sigma_atm", aggfunc="median", observed=True) * 100).round(1).to_string())

print("\n" + "=" * 90); print("6) NO-ARBITRAGE FILTER and QUOTE AGE"); print("=" * 90)
print("Options removed by filter 7 in total:", int(sm.n_noarb_removed.sum()),
      "| smiles with at least one removal:", int((sm.n_noarb_removed > 0).sum()), "of", len(sm))
print("Median quote age (sec), by maturity:", sm.groupby("bucket", observed=True).age_median_sec.median().round(1).to_dict())
print("Mean share of quotes older than 60 s:", round(sm.share_age_gt60s.mean(), 4))
print("\nEND")
