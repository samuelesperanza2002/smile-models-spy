"""
Four figures of Chapters 4 and 5 of the thesis, from aggregated results only (results/model_results.csv,
results/smile_summary.csv and the VIX). No option-level data is used.
  fig2  IV-RMSE against days to expiry, in sample and out of sample (mean and interquartile band)
  fig3  SABR nu, nu * sqrt(tau) and rho against the time to expiry, by regime (log scale)
  fig4  heat map of the in-sample IV-RMSE by moneyness group and maturity, one panel per model
  fig5  share of 0DTE smiles that can be calibrated, and smile size, by regime and snapshot
The figures are saved in results/figures/ as PNG and PDF.

How to run (from the src folder):  python3 make_figures.py
"""

import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HOME = pathlib.Path.home() / "thesis_spy"
RESULTS, FIG = HOME / "results", HOME / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
BUCKETS = ["0", "1", "2", "3-7", "30"]
COLORS = {"BS": "#7f7f7f", "SABR:P1": "#1f77b4", "VV:P1": "#d62728", "SABR(B)": "#17becf", "StickyStrike": "#2ca02c",
          "VV:P2": "#ff7f0e", "VV:P3": "#9467bd"}
LABELS = {"BS": "Black-Scholes", "SABR:P1": "SABR (P1)", "VV:P1": "Vanna-Volga (P1)", "SABR(B)": "SABR (B)",
          "StickyStrike": "Sticky strike", "VV:P2": "Vanna-Volga (P2)", "VV:P3": "Vanna-Volga (P3)"}
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "legend.fontsize": 8, "figure.dpi": 150})

res = pd.read_csv(RESULTS / "model_results.csv", dtype={"period_id": str}, low_memory=False)
res["bucket"] = pd.Categorical(np.where(res.control, "30", np.where(res.dte <= 2, res.dte.astype(str), "3-7")), BUCKETS, ordered=True)
res["model_key"] = res.model + np.where(res.design == "B", "(B)", "") + \
    np.where((res.model.isin(["SABR", "VV"])) & (res.design == "A"), ":" + res.pillar_set, "")
reg = res[res.group == "regular"]


def band_plot(ax, df, models, title, ylabel="IV-RMSE (vol points)", logy=True):
    """Mean error of each model by maturity, with a shaded band from the 25th to the 75th percentile."""
    x = np.arange(len(BUCKETS))
    for mk in models:
        d = df[df.model_key == mk]
        if not len(d):
            continue
        g = d.groupby("bucket", observed=False).iv_rmse
        mean, q1, q3 = g.mean().reindex(BUCKETS), g.quantile(.25).reindex(BUCKETS), g.quantile(.75).reindex(BUCKETS)
        ax.plot(x, mean.values, "-o", ms=4, color=COLORS[mk], label=LABELS[mk])
        ax.fill_between(x, q1.values, q3.values, color=COLORS[mk], alpha=0.12)
    ax.set_xticks(x); ax.set_xticklabels(["0", "1", "2", "3-7", "~30"]); ax.set_xlabel("days to expiry")
    ax.set_ylabel(ylabel); ax.set_title(title)
    if logy:
        ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    if ax.get_legend_handles_labels()[0]:
        ax.legend()


# ------------------------------------------------------------------ fig2: IV-RMSE by days to expiry
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
ins = reg[(reg.stage == "insample") & (reg.region == "all") & (reg.pillar_set.isin(["P1", "odd/even"]))]
band_plot(axes[0], ins, ["BS", "SABR:P1", "VV:P1", "SABR(B)"], "In-sample")
intra = reg[(reg.stage == "intraday") & (reg.region == "common") & (reg.pillar_set.isin(["P1", "-", "odd/even"]))]
band_plot(axes[1], intra, ["BS", "SABR:P1", "VV:P1", "SABR(B)", "StickyStrike"], "Out-of-sample: 10:30 -> 14:30")
nxt = reg[(reg.stage == "nextday") & (reg.region == "common") & (reg.pillar_set.isin(["P1", "-", "odd/even"]))]
band_plot(axes[2], nxt, ["BS", "SABR:P1", "VV:P1", "SABR(B)", "StickyStrike"], "Out-of-sample: next day 10:30")
fig.tight_layout(); fig.savefig(FIG / "fig2_ivrmse_vs_dte.png"); fig.savefig(FIG / "fig2_ivrmse_vs_dte.pdf"); plt.close(fig)

# ------------------------------------------------------------------ fig3: SABR parameters against the time to expiry
sB = reg[(reg.stage == "insample") & (reg.model == "SABR") & (reg.design == "B") & (reg.region == "all")].copy()
sB["nu_sqrt_tau"] = sB.nu * np.sqrt(sB.tau)
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
rc = {"calm": "#2ca02c", "normal": "#1f77b4", "stressed": "#d62728"}
for regime, d in sB.groupby("regime"):
    axes[0].scatter(d.tau * 365, d.nu, s=14, color=rc.get(regime, "k"), label=regime, alpha=0.8)
    axes[1].scatter(d.tau * 365, d.nu_sqrt_tau, s=14, color=rc.get(regime, "k"), label=regime, alpha=0.8)
    axes[2].scatter(d.tau * 365, d.rho, s=14, color=rc.get(regime, "k"), label=regime, alpha=0.8)
# power-law fit nu ~ tau^b (a straight line in log-log scale)
lx, ly = np.log(sB.tau), np.log(sB.nu)
b, a = np.polyfit(lx, ly, 1)
xx = np.linspace(sB.tau.min(), sB.tau.max(), 100)
axes[0].plot(xx * 365, np.exp(a) * xx ** b, "k--", lw=1, label=f"fit: ν ∝ τ^{b:.2f}")
axes[0].set_xscale("log"); axes[0].set_yscale("log"); axes[0].set_xlabel("τ (days)"); axes[0].set_ylabel("ν (vol-of-vol)")
axes[0].set_title("SABR ν vs time to expiry"); axes[0].grid(alpha=0.3, which="both"); axes[0].legend()
axes[1].set_xscale("log"); axes[1].set_xlabel("τ (days)"); axes[1].set_ylabel("ν·√τ"); axes[1].set_title("Scaled vol-of-vol ν√τ")
axes[1].grid(alpha=0.3, which="both"); axes[1].legend()
axes[2].set_xscale("log"); axes[2].set_xlabel("τ (days)"); axes[2].set_ylabel("ρ"); axes[2].set_title("SABR ρ vs time to expiry")
axes[2].axhline(0, color="grey", lw=0.5); axes[2].grid(alpha=0.3, which="both"); axes[2].legend()
fig.tight_layout(); fig.savefig(FIG / "fig3_sabr_nu_vs_tau.png"); fig.savefig(FIG / "fig3_sabr_nu_vs_tau.pdf"); plt.close(fig)
print(f"Figure 3: power-law fit nu ~ tau^{b:.3f} (n = {len(sB)} smiles)")

# ------------------------------------------------------------------ fig4: error map
groups = ["put_deep", "put_otm", "near_atm", "call_otm", "call_deep"]
glab = ["deep\nOTM put", "OTM\nput", "near\nATM", "OTM\ncall", "deep\nOTM call"]
ins4 = reg[(reg.stage == "insample") & (reg.region.isin(groups)) & (reg.pillar_set.isin(["P1", "odd/even"]))]
models4 = ["BS", "VV:P1", "SABR:P1", "SABR(B)"]
mats = [[m for m in models4 if m in ins4.model_key.unique()]][0]
vmax = 0
grids = {}
for mk in mats:
    g = ins4[ins4.model_key == mk].pivot_table(index="bucket", columns="region", values="iv_rmse", aggfunc="mean", observed=False)
    g = g.reindex(index=BUCKETS, columns=groups)
    grids[mk] = g
    vmax = max(vmax, np.nanmax(g.values))
fig, axes = plt.subplots(1, len(mats), figsize=(3.2 * len(mats) + 1, 4.2), sharey=True)
for ax, mk in zip(np.atleast_1d(axes), mats):
    g = grids[mk]
    im = ax.imshow(g.values, cmap="YlOrRd", vmin=0, vmax=min(vmax, 12), aspect="auto")
    ax.set_xticks(range(5)); ax.set_xticklabels(glab, fontsize=7)
    ax.set_yticks(range(5)); ax.set_yticklabels(["0", "1", "2", "3-7", "~30"])
    ax.set_title(LABELS[mk])
    for i in range(5):
        for j in range(5):
            v = g.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7, color="black" if v < 6 else "white")
np.atleast_1d(axes)[0].set_ylabel("days to expiry")
fig.colorbar(im, ax=axes, shrink=0.8, label="IV-RMSE (vol points), capped at 12")
fig.savefig(FIG / "fig4_error_map.png", bbox_inches="tight"); fig.savefig(FIG / "fig4_error_map.pdf", bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------------------ fig5: availability of the 0DTE smiles, from smile_summary.csv
sm = pd.read_csv(RESULTS / "smile_summary.csv", parse_dates=["day"])
vix = pd.read_csv(HOME / "data" / "fred_VIXCLS.csv"); vix.columns = ["date", "v"]
vix["date"] = pd.to_datetime(vix["date"]); vix["v"] = pd.to_numeric(vix["v"], errors="coerce"); vix = vix.dropna().set_index("date")["v"]
prev = vix.shift(1).reindex(sm.day.unique(), method="ffill")       # last close strictly before the day
sm["vix_tm1"] = sm.day.map(prev)
sm["regime"] = pd.cut(sm.vix_tm1, [0, 15, 20, 25, 100], labels=["calm (<15)", "normal (15-20)", "20-25", "stressed (>25)"])
z = sm[sm.dte == 0]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
t = z.pivot_table(index="regime", columns="snap", values="calib_ok", aggfunc="mean", observed=False)
t.plot(kind="bar", ax=axes[0], rot=0, color=["#1f77b4", "#ff7f0e", "#d62728"])
axes[0].set_ylabel("share of days with ≥ 8 strikes"); axes[0].set_title("0DTE: calibratable smiles by regime and snapshot")
axes[0].set_xlabel(""); axes[0].legend(title="snapshot", labels=["10:30", "11:30", "14:30"]); axes[0].set_ylim(0, 1.05); axes[0].grid(axis="y", alpha=0.3)
t2 = z.pivot_table(index="regime", columns="snap", values="n_options", aggfunc="mean", observed=False)
t2.plot(kind="bar", ax=axes[1], rot=0, color=["#1f77b4", "#ff7f0e", "#d62728"])
axes[1].set_ylabel("mean number of OTM options"); axes[1].set_title("0DTE: smile size by regime and snapshot")
axes[1].set_xlabel(""); axes[1].legend(title="snapshot", labels=["10:30", "11:30", "14:30"]); axes[1].grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(FIG / "fig5_0dte_availability.png"); fig.savefig(FIG / "fig5_0dte_availability.pdf"); plt.close(fig)

print("Figures saved in", FIG)
for f in sorted(FIG.glob("*.png")):
    print("  ", f.name)
