"""
Figure 5.1 of the thesis: market smiles and the fitted models on three sample days (calm,
stressed, extreme), for three maturities (0, 2 and about 30 days), at 10:30.

Theta Data approved this type of chart for the thesis (9 November 2026): implied volatility
points against standardized moneyness, with the model curves, and no strikes, prices, bid/ask
or labels on single points. The dates appear only in the LaTeX caption.
Rows = maturities, columns = days. The figure is saved on the Desktop and in results/figures.
It needs the cleaned files in data/clean, which are licensed data and are not in this repository.

How to run (from the src folder):
  python3 make_fig1_smiles.py 2024-08-20 2024-08-06 2025-04-08     calm, stressed, extreme day
  python3 make_fig1_smiles.py 2024-08-20 2024-08-06                two columns only
"""

import sys, pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import black, sabr, vv

HOME = pathlib.Path.home() / "thesis_spy"
CLEAN, FIG, DESK = HOME / "data" / "clean", HOME / "results" / "figures", pathlib.Path.home() / "Desktop"
FIG.mkdir(parents=True, exist_ok=True)
SNAP = "1030"
days = [pd.Timestamp(d) for d in (sys.argv[1:] if len(sys.argv) >= 3 else ["2024-08-20", "2024-08-06", "2025-04-08"])]
col_names = ["Calm day", "Stressed day", "Extreme day"][:len(days)]
MATS = [(0, "0 days to expiry"), (2, "2 days to expiry"), ("ctrl", "about 30 days to expiry")]
COL = {"BS": "#7f7f7f", "VV": "#d62728", "SABR": "#1f77b4", "SABRB": "#17becf"}
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "legend.fontsize": 8, "figure.dpi": 150})


def pick(df, which):
    """The smile to plot: which = 0 or 2 (days to expiry) or 'ctrl' (the control expiry at about 30 days).
    Returns None when that smile cannot be calibrated."""
    sel = df[df.is_control] if which == "ctrl" else df[(df.dte == which) & (~df.is_control)]
    if not len(sel):
        return None
    exp = sel.expiration.iloc[0]
    sm = sel[sel.expiration == exp].sort_values("strike").reset_index(drop=True)
    return sm if bool(sm.calib_ok.iloc[0]) else None


ncol = len(days)
fig, axes = plt.subplots(3, ncol, figsize=(2.6 * ncol + 0.6, 8.8), squeeze=False)
report = []
for j, (day, cname) in enumerate(zip(days, col_names)):
    f = CLEAN / f"clean_{day:%Y%m%d}_{SNAP}.parquet"
    if not f.exists():
        sys.exit(f"missing {f}")
    df = pd.read_parquet(f)
    for i, (which, rname) in enumerate(MATS):
        ax = axes[i, j]
        sm = pick(df, which)
        if sm is None:
            ax.set_title("no calibratable smile"); continue
        F, D, tau, s_atm = float(sm.F.iloc[0]), float(sm.D.iloc[0]), float(sm.tau.iloc[0]), float(sm.sigma_atm.iloc[0])
        K = sm.strike.to_numpy(); ivm = sm.iv.to_numpy()
        Kg = np.linspace(K.min(), K.max(), 300)
        mg = np.log(Kg / F) / (s_atm * np.sqrt(tau))
        pil = vv.select_pillars(sm, "P1")
        calA = sabr.calibrate(pil.strike.to_numpy(), pil.iv.to_numpy(), F, tau, beta=1.0, sigma_atm=s_atm)
        iv_sabrA = sabr.hagan_iv(Kg, F, tau, calA["alpha"], calA["rho"], calA["nu"], 1.0)
        model = vv.VannaVolga(F, tau, D, pil.strike.to_numpy(), pil.iv.to_numpy(), vv.ref_index("P1"))
        iv_vv = model.iv(Kg, Kg >= F)
        odd = sm.iloc[1::2]
        calB = sabr.calibrate(odd.strike.to_numpy(), odd.iv.to_numpy(), F, tau, beta=1.0, sigma_atm=s_atm)
        iv_sabrB = sabr.hagan_iv(Kg, F, tau, calB["alpha"], calB["rho"], calB["nu"], 1.0)
        ax.scatter(sm.m, ivm * 100, s=11, facecolors="none", edgecolors="black", linewidths=0.7, label="Market", zorder=3)
        ax.axhline(s_atm * 100, color=COL["BS"], ls=":", lw=1.1, label="Black-Scholes")
        ax.plot(mg, iv_vv * 100, color=COL["VV"], ls="--", lw=1.2, label="Vanna-Volga (P1)")
        ax.plot(mg, iv_sabrA * 100, color=COL["SABR"], lw=1.2, label="SABR (P1)")
        ax.plot(mg, iv_sabrB * 100, color=COL["SABRB"], lw=1.2, label="SABR (B)")
        if i == 0:
            ax.set_title(cname)
        if j == 0:
            ax.set_ylabel(f"{rname}\nimplied volatility (%)")
        if i == 2:
            ax.set_xlabel("standardized moneyness $m$")
        ax.grid(alpha=0.3)
        lo, hi = np.nanmin(ivm) * 100, np.nanmax(ivm) * 100
        ax.set_ylim(max(0, lo - 0.15 * (hi - lo)), hi + 0.25 * (hi - lo))
        report.append(dict(day=day.date(), panel=rname, n_options=len(sm), sigma_atm=round(s_atm * 100, 1),
                           m_min=round(float(sm.m.min()), 2), m_max=round(float(sm.m.max()), 2),
                           sabr_P1_nu=round(calA["nu"], 1), sabr_B_nu=round(calB["nu"], 1),
                           vv_fail_grid=int(np.isnan(iv_vv).sum())))
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.96))
for folder in (FIG, DESK):
    fig.savefig(folder / "fig1_smiles.pdf"); fig.savefig(folder / "fig1_smiles.png")
plt.close(fig)
pd.set_option("display.width", 200)
print(pd.DataFrame(report).to_string(index=False))
print(f"\nSaved fig1_smiles.pdf and .png on the Desktop and in {FIG}")
