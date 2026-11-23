"""
Figure 5.5 of the thesis: smiles out of sample. The models are fitted at 10:30 on day t and moved,
without a new fit, to the same expiry at 14:30 (row 1: the 0DTE expiry) and at 10:30 on day t+1
(row 2: the 1-day expiry, which is then a 0DTE).

The models move as in run_models.apply_oos: Black-Scholes keeps its ATM volatility, SABR keeps
its parameters and Vanna-Volga moves its pillars with the sticky-delta rule. The market points at
10:30 are also the prediction of the sticky-strike benchmark. The x axis is the standardized
moneyness computed at 10:30, so that a strike keeps its position in both snapshots.
Same type of chart as make_fig1_smiles.py (approved by Theta Data on 9 November 2026).

How to run (from the src folder):
  python3 make_fig6_oos.py 2024-08-20 2024-08-06 2025-04-08
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
days = [pd.Timestamp(d) for d in (sys.argv[1:] if len(sys.argv) >= 3 else ["2024-08-20", "2024-08-06", "2025-04-08"])]
col_names = ["Calm day", "Stressed day", "Extreme day"][:len(days)]
sample = pd.read_csv(HOME / "data" / "sample_dates.csv")
next_day = {pd.Timestamp(r.t): pd.Timestamp(r.t1) for _, r in sample.iterrows()}
COL = {"BS": "#7f7f7f", "VV": "#d62728", "SABR": "#1f77b4", "SABRB": "#17becf"}
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "legend.fontsize": 7.5, "figure.dpi": 150})


def load(day, snap):
    """Cleaned options of one day and snapshot (None if the file does not exist)."""
    f = CLEAN / f"clean_{day:%Y%m%d}_{snap}.parquet"
    return pd.read_parquet(f) if f.exists() else None


def smile(df, expiration):
    """The options of one expiry, sorted by strike (None if there are none)."""
    sm = df[df.expiration == expiration].sort_values("strike").reset_index(drop=True)
    return sm if len(sm) else None


ncol = len(days)
fig, axes = plt.subplots(2, ncol, figsize=(2.6 * ncol + 0.6, 6.4), squeeze=False)
report = []
for j, (day, cname) in enumerate(zip(days, col_names)):
    cal = load(day, "1030")
    if cal is None:
        sys.exit(f"missing 10:30 file for {day.date()}")
    tests = [(0, "0DTE expiry, 10:30 to 14:30", load(day, "1430")),
             (1, "1-day expiry, 10:30 to 10:30 next day", load(next_day.get(day, day + pd.Timedelta(days=1)), "1030"))]
    for i, (dte, rname, test_df) in enumerate(tests):
        ax = axes[i, j]
        sel = cal[(cal.dte == dte) & (~cal.is_control)]
        if not len(sel) or not bool(sel.calib_ok.iloc[0]) or test_df is None:
            ax.set_title("no smile"); continue
        exp = sel.expiration.iloc[0]
        sm_cal, sm_test = smile(cal, exp), smile(test_df, exp)
        if sm_test is None:
            ax.set_title("no test smile"); continue
        F0, D0, tau0, s0 = float(sm_cal.F.iloc[0]), float(sm_cal.D.iloc[0]), float(sm_cal.tau.iloc[0]), float(sm_cal.sigma_atm.iloc[0])
        F1, D1, tau1, s1 = float(sm_test.F.iloc[0]), float(sm_test.D.iloc[0]), float(sm_test.tau.iloc[0]), float(sm_test.sigma_atm.iloc[0])
        m_of = lambda K: np.log(np.asarray(K, float) / F0) / (s0 * np.sqrt(tau0))   # position of a strike, fixed at 10:30
        # fit at 10:30 (Design A with P1, Design B), as in evaluate.eval_smile
        pil = vv.select_pillars(sm_cal, "P1")
        calA = sabr.calibrate(pil.strike.to_numpy(), pil.iv.to_numpy(), F0, tau0, beta=1.0, sigma_atm=s0)
        model = vv.VannaVolga(F0, tau0, D0, pil.strike.to_numpy(), pil.iv.to_numpy(), vv.ref_index("P1"))
        odd = sm_cal.iloc[1::2]
        calB = sabr.calibrate(odd.strike.to_numpy(), odd.iv.to_numpy(), F0, tau0, beta=1.0, sigma_atm=s0)
        # move to the test snapshot, as in run_models.apply_oos
        Kg = np.linspace(sm_test.strike.min(), sm_test.strike.max(), 300)
        iv_sabrA = sabr.hagan_iv(Kg, F1, tau1, calA["alpha"], calA["rho"], calA["nu"], 1.0)
        iv_sabrB = sabr.hagan_iv(Kg, F1, tau1, calB["alpha"], calB["rho"], calB["nu"], 1.0)
        sh = model.shifted(F1, tau1, D1, pil.delta_call.to_numpy())
        iv_vv = sh.iv(Kg, Kg >= F1)
        # plot
        ax.scatter(m_of(sm_cal.strike), sm_cal.iv * 100, s=11, facecolors="none", edgecolors="#9a9a9a", linewidths=0.7,
                   label="Market at 10:30 (= sticky strike)", zorder=3)
        ax.scatter(m_of(sm_test.strike), sm_test.iv * 100, s=12, color="black", label="Market at test time", zorder=4)
        ax.axhline(s0 * 100, color=COL["BS"], ls=":", lw=1.1, label="Black-Scholes")
        ax.plot(m_of(Kg), iv_vv * 100, color=COL["VV"], ls="--", lw=1.2, label="Vanna-Volga (P1)")
        ax.plot(m_of(Kg), iv_sabrA * 100, color=COL["SABR"], lw=1.2, label="SABR (P1)")
        ax.plot(m_of(Kg), iv_sabrB * 100, color=COL["SABRB"], lw=1.2, label="SABR (B)")
        if i == 0:
            ax.set_title(cname)
        if j == 0:
            ax.set_ylabel(f"{rname}\nimplied volatility (%)")
        if i == 1:
            ax.set_xlabel("standardized moneyness $m$ at 10:30")
        ax.grid(alpha=0.3)
        allv = np.concatenate([sm_cal.iv.to_numpy(), sm_test.iv.to_numpy()]) * 100
        lo, hi = np.nanmin(allv), np.nanmax(allv)
        ax.set_ylim(max(0, lo - 0.15 * (hi - lo)), hi + 0.30 * (hi - lo))
        # short report on screen: mean absolute error of each method on the common strikes (vol points)
        mm = sm_test[["strike", "right", "iv"]].merge(sm_cal[["strike", "right", "iv"]], on=["strike", "right"], suffixes=("_t", "_c"))
        Kt = sm_test.strike.to_numpy(); ct = (sm_test.right == "CALL").to_numpy()
        e_sabrB = np.nanmean(np.abs(sabr.hagan_iv(Kt, F1, tau1, calB["alpha"], calB["rho"], calB["nu"], 1.0) - sm_test.iv.to_numpy())) * 100
        e_vv = np.nanmean(np.abs(sh.iv(Kt, ct) - sm_test.iv.to_numpy())) * 100
        e_ss = np.nanmean(np.abs(mm.iv_c - mm.iv_t)) * 100 if len(mm) else np.nan
        report.append(dict(day=day.date(), row=rname, n_cal=len(sm_cal), n_test=len(sm_test), n_common=len(mm),
                           sigma_atm_cal=round(s0 * 100, 1), sigma_atm_test=round(s1 * 100, 1),
                           mae_sticky=round(e_ss, 1), mae_sabrB=round(e_sabrB, 1), mae_vv=round(e_vv, 1)))
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.93))
for folder in (FIG, DESK):
    fig.savefig(folder / "fig6_oos_smiles.pdf"); fig.savefig(folder / "fig6_oos_smiles.png")
plt.close(fig)
pd.set_option("display.width", 220)
print(pd.DataFrame(report).to_string(index=False))
print(f"\nSaved fig6_oos_smiles.pdf and .png on the Desktop and in {FIG}")
