"""
Diagnostic used to choose the filters of clean.py (counts only, sample days).

For the expiries with 0, 1 and 2 days to expiry it counts how many out-of-the-money options
remain under four versions of filters 2-3, and it compares two estimates of the forward:
the median used in clean.py and a regression of C - P on K.

How to run (from the src folder):  python3 diag_filters.py
"""

import pathlib
import numpy as np
import pandas as pd

HOME = pathlib.Path.home() / "thesis_spy"
RAW, DATA = HOME / "data" / "raw", HOME / "data"
OI_MIN = 100
SNAPS = {"1030": (10, 30), "1430": (14, 30)}
VOLCOL = {"1030": "vol_1030", "1430": "vol_1430"}

tb = pd.read_csv(DATA / "fred_DTB4WK.csv"); tb.columns = ["date", "value"]
tb["date"] = pd.to_datetime(tb["date"]); tb["value"] = pd.to_numeric(tb["value"], errors="coerce")
tb = tb.dropna().set_index("date")["value"] / 100

VARIANTS = {                                   # name: (minimum mid, maximum relative spread, maximum ask / bid)
    "A: mid>=0.05, spread<=50%": (0.05, 0.50, np.inf),
    "B: mid>=0.05, spread<=100%": (0.05, 1.00, np.inf),
    "C: mid>=0.03, spread<=100%": (0.03, 1.00, np.inf),
    "D: bid>0, ask/bid<=10 (AFT)": (0.00, np.inf, 10.0),
}

sample = pd.read_csv(DATA / "sample_dates.csv")
rows, fwd = [], []
for _, s in sample.iterrows():
    for d in (s.t, s.t1):
        day = pd.Timestamp(d)
        oif = RAW / f"oi_{day:%Y%m%d}.parquet"
        if not oif.exists():
            print("missing", oif.name, "- day skipped")
            continue
        oi = pd.read_parquet(oif)
        oi["expiration"] = pd.to_datetime(oi["expiration"].astype(str))
        r = float(tb[tb.index <= day].iloc[-1])
        for snap, hm in SNAPS.items():
            f = RAW / f"quotes_{day:%Y%m%d}_{snap}.parquet"
            if not f.exists():
                continue
            q = pd.read_parquet(f).drop_duplicates()
            q["expiration"] = pd.to_datetime(q["expiration"].astype(str))
            q["dte"] = (q.expiration - day).dt.days
            q = q[q.dte.between(0, 2)]
            q = q.merge(oi[["expiration", "strike", "right", "open_interest"]], on=["expiration", "strike", "right"], how="left")
            q["oi"] = q.open_interest.fillna(0)
            q["mid"] = (q.bid + q.ask) / 2
            q["spread_rel"] = np.where(q.mid > 0, (q.ask - q.bid) / q.mid, np.inf)
            q["ask_bid"] = np.where(q.bid > 0, q.ask / q.bid, np.inf)
            for exp, g in q.groupby("expiration"):
                dte = int(g.dte.iloc[0])
                vf = RAW / f"vol_{exp:%Y%m%d}.parquet"
                g = g.copy(); g["volume"] = 0
                if vf.exists():
                    v = pd.read_parquet(vf); v = v[v["day"].astype(str) == str(day.date())]
                    g = g.merge(v[["strike", "right", VOLCOL[snap]]], on=["strike", "right"], how="left")
                    g["volume"] = g[VOLCOL[snap]].fillna(0)
                tau = ((exp - day).days * 1440 + 960 - hm[0] * 60 - hm[1]) / 525600
                base = g[(g.bid > 0) & (g.ask > g.bid) & (g.mid >= 0.05)]
                c = base[base.right == "CALL"][["strike", "mid"]].rename(columns={"mid": "C"})
                p = base[base.right == "PUT"][["strike", "mid"]].rename(columns={"mid": "P"})
                pairs = c.merge(p, on="strike")
                if len(pairs) < 4:
                    continue
                pairs["diff"] = pairs.C - pairs.P
                k0 = pairs.loc[pairs["diff"].abs().idxmin(), "strike"]
                sel = pairs.assign(dist=(pairs.strike - k0).abs()).sort_values(["dist", "strike"]).head(8)
                D_tb = np.exp(-r * tau)
                est = sel.strike + sel["diff"] / D_tb
                F_med = float(np.median(est))
                b, a = np.polyfit(sel.strike, sel["diff"], 1)
                F_reg, D_reg = (a / -b if b != 0 else np.nan), -b
                fwd.append(dict(regime=s.regime, snap=snap, dte=dte, F_med=F_med, F_reg=F_reg, D_reg=D_reg, D_tb=D_tb,
                                disp=float(est.max() - est.min()), diff_F=abs(F_reg - F_med)))
                otm = g[(((g.right == "PUT") & (g.strike < F_med)) | ((g.right == "CALL") & (g.strike >= F_med)))
                        & (g.bid > 0) & (g.ask > g.bid) & ((g.volume > 0) | (g.oi >= OI_MIN))]
                rec = dict(regime=s.regime, period=s.period_id, day=day.date(), snap=snap, dte=dte, n_otm_bid=len(otm))
                for name, (mmin, smax, abmax) in VARIANTS.items():
                    sub = otm[(otm.mid >= mmin) & (otm.spread_rel <= smax) & (otm.ask_bid <= abmax)]
                    rec[name] = len(sub)
                rows.append(rec)

df = pd.DataFrame(rows)
fw = pd.DataFrame(fwd)
pd.set_option("display.width", 200)
print(f"Smiles (day x snapshot x expiry 0-2 days) examined: {len(df)}\n")
print("MEAN NUMBER of OTM options per smile, by regime / snapshot / dte (after bid>0 and liquidity):")
cols = ["n_otm_bid"] + list(VARIANTS)
print(df.groupby(["regime", "snap", "dte"])[cols].mean().round(1).to_string())
print("\nSHARE of smiles with at least 8 OTM options, by snapshot / dte:")
print(df.groupby(["snap", "dte"])[cols].agg(lambda x: (x >= 8).mean()).round(2).to_string())
print("\nSHARE of smiles with at least 8 OTM options, by regime (all expiries 0-2 days):")
print(df.groupby(["regime"])[cols].agg(lambda x: (x >= 8).mean()).round(2).to_string())
print("\nFORWARD: regression vs median (8 pairs around the money), by dte:")
print(fw.groupby("dte").agg(diff_F_median=("diff_F", "median"), diff_F_max=("diff_F", "max"),
                            disp_median=("disp", "median"), D_reg_min=("D_reg", "min"),
                            D_reg_max=("D_reg", "max"), D_tb_mean=("D_tb", "mean")).round(4).to_string())
df.to_csv(HOME / "results" / "diag_filters_sample.csv", index=False)
print("\nSaved results/diag_filters_sample.csv (counts only).")
