"""
Closer look at the candidate outliers found by contact_sheet.py (local check only, nothing saved).
For each flagged option it prints bid, ask, mid, relative spread, its position from the right end
of the smile (0 = last strike) and the implied volatility of the two strikes just below it, to see
whether the rise of the smile is gradual.

How to run (from the src folder):  python3 check_outliers.py
"""

import pathlib
import pandas as pd

HOME = pathlib.Path.home() / "thesis_spy"
CLEAN = HOME / "data" / "clean"
pd.set_option("display.width", 220)

o = pd.read_csv(CLEAN / "plots" / "check" / "outlier_candidates.csv", parse_dates=["day", "expiration"])
rows = []
for (day, snap), g in o.groupby(["day", "snap"]):
    df = pd.read_parquet(CLEAN / f"clean_{day:%Y%m%d}_{snap}.parquet")
    df["expiration"] = pd.to_datetime(df.expiration)
    for _, r in g.iterrows():
        sm = df[df.expiration == r.expiration].sort_values("strike").reset_index(drop=True)
        i = sm.index[(sm.strike == r.strike) & (sm.right == r.right)]
        if not len(i):
            continue
        i = int(i[0]); q = sm.loc[i]
        prev = sm.iv.iloc[max(i - 2, 0):i].to_numpy() * 100
        rows.append(dict(day=day.date(), snap=snap, dte=int(q.dte), strike=q.strike, m=round(q.m, 2), iv=round(q.iv * 100, 2),
                         iv_prev2=" / ".join(f"{v:.1f}" for v in prev), bid=q.bid, ask=q.ask, mid=q.mid,
                         spread_rel=round(q.spread_rel, 2), volume=int(q.volume), oi=int(q.oi), pos_from_right=len(sm) - 1 - i,
                         n_calls_m_gt_1_5=int(((sm.right == "CALL") & (sm.m > 1.5)).sum())))
res = pd.DataFrame(rows).sort_values(["day", "snap", "dte", "strike"])
print(res.to_string(index=False))
print("\nmid <= 0.10 $:", int((res.mid <= 0.10).sum()), "of", len(res), "| last or second-last strike:", int((res.pos_from_right <= 1).sum()),
      "| spread_rel >= 0.30:", int((res.spread_rel >= 0.30).sum()))
