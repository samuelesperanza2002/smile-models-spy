"""
Visual check of the sample smiles (Chapter 4 of the thesis). For each snapshot it draws contact
sheets with all the smiles of the 40 sample days (t and t+1), grouped by maturity, and it lists
candidate outliers: points far from a smooth curve fitted to their smile.
It reads the cleaned files of the sample days. The sheets were for local use only and are not in
the thesis.

How to run (from the src folder):  python3 contact_sheet.py
"""

import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HOME = pathlib.Path.home() / "thesis_spy"
CLEAN, DATA = HOME / "data" / "clean", HOME / "data"
OUT = CLEAN / "plots" / "check"; OUT.mkdir(parents=True, exist_ok=True)
BUCKETS = ["0", "1", "2", "3-7", "30"]

sample = pd.read_csv(DATA / "sample_dates.csv", parse_dates=["t", "t1"])
days = sorted(set(sample.t) | set(sample.t1))
regime = {}
for _, s in sample.iterrows():
    regime[s.t] = s.regime; regime[s.t1] = s.regime

outliers = []
for snap in ("1030", "1430"):
    panels = []                                      # (day, expiration, dte, bucket, df)
    for day in days:
        f = CLEAN / f"clean_{day:%Y%m%d}_{snap}.parquet"
        if not f.exists():
            continue
        df = pd.read_parquet(f)
        for exp, g in df.groupby("expiration"):
            dte = int(g.dte.iloc[0])
            b = "30" if g.is_control.iloc[0] else ("0" if dte == 0 else "1" if dte == 1 else "2" if dte == 2 else "3-7")
            panels.append((day, pd.Timestamp(exp), dte, b, g.sort_values("strike")))
            # outlier test: distance of each IV from a cubic curve in m, compared with the typical distance
            g = g.sort_values("strike")
            if len(g) >= 8:
                x, y = g.m.to_numpy(), g.iv.to_numpy() * 100
                coef = np.polyfit(x, y, 3)
                res = y - np.polyval(coef, x)
                mad = np.median(np.abs(res - np.median(res))) * 1.4826 + 1e-9
                bad = np.abs(res) > max(4 * mad, 1.0)        # > 4 robust sigma AND > 1 vol point
                for i in np.where(bad)[0]:
                    outliers.append(dict(day=day.date(), snap=snap, expiration=exp.date() if hasattr(exp, "date") else exp,
                                         dte=dte, strike=float(g.strike.iloc[i]), right=g.right.iloc[i], m=round(float(x[i]), 2),
                                         iv=round(float(y[i]), 2), residual=round(float(res[i]), 2), n=len(g)))
    for b in BUCKETS:
        ps = [p for p in panels if p[3] == b]
        if not ps:
            continue
        n = len(ps); cols = 6; rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.4 * rows), squeeze=False)
        for ax, (day, exp, dte, _, g) in zip(axes.ravel(), ps):
            p, c = g[g.right == "PUT"], g[g.right == "CALL"]
            ax.plot(p.m, p.iv * 100, "o", ms=2, color="#1f77b4"); ax.plot(c.m, c.iv * 100, "^", ms=2, color="#ff7f0e")
            ax.axvline(0, color="grey", lw=0.4)
            ax.set_title(f"{day.date()} {regime.get(day, '')[:3]} | {exp.date()} d{dte} | n={len(g)}", fontsize=6)
            ax.tick_params(labelsize=5)
        for ax in axes.ravel()[n:]:
            ax.axis("off")
        fig.suptitle(f"Sample smiles, snapshot {snap[:2]}:{snap[2:]}, maturity bucket {b} days (x: standardized moneyness, y: IV %)", fontsize=9)
        fig.tight_layout()
        fig.savefig(OUT / f"sheet_{snap}_dte{b.replace('-', '_')}.png", dpi=110); plt.close(fig)
        print(f"sheet_{snap}_dte{b.replace('-', '_')}.png: {n} smiles")

o = pd.DataFrame(outliers)
print(f"\nCandidate outliers (residual from a cubic fit in m > 4 robust sigma and > 1 vol point): {len(o)}")
if len(o):
    print(o.sort_values("residual", key=abs, ascending=False).head(40).to_string(index=False))
    o.to_csv(OUT / "outlier_candidates.csv", index=False)
print(f"\nSheets saved in {OUT}")
