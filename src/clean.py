"""
Cleaning of the raw option quotes (Chapter 4 of the thesis).

For each trading day and snapshot (10:30 and 14:30; 11:30 when requested) the script:
  1. reads the quotes file, drops duplicates, and keeps the expiries with 0 to 7 days to expiry
     plus the control expiry at about 30 days;
  2. finds half days and stale snapshots from the age of the quotes (timestamp = last update);
  3. adds open interest and traded volume (missing = 0);
  4. applies filters 1-4 (bid and ask, minimum price, maximum spread, liquidity);
  5. takes the discount factor D from the 4-week T-bill and the forward F from put-call parity
     (median of K + (C - P) / D over the 8 strike pairs closest to the money; a regression of
     C - P on K is kept only as a check);
  6. keeps the out-of-the-money options and computes the time to expiry, implied volatility
     (Black-76), vega, delta, ATM volatility and standardized moneyness m;
  7. applies filters 5-9 (out of the money, moneyness window, no arbitrage, implied volatility
     exists, at least 4 strikes to keep the smile as a test smile and at least 8 strikes to use
     it for calibration: column calib_ok);
  8. saves data/clean/clean_<day>_<snap>.parquet (this file still contains prices, so it was
     deleted at the end of the data licence), the filter counts and a summary of each smile
     (aggregates, which can be kept).

How to run (from the src folder):
  python3 clean.py 20240508                 one day, detailed output and plots
  python3 clean.py 20240508 --snap 1430     one snapshot only
  python3 clean.py --all                    all days (plots only for the sample days)
  python3 clean.py --all --spread 0.7       robustness: another maximum spread
  python3 clean.py --all --filters aft --snap 1030 1130 1430 --days sample
      robustness: the minimal filters of Andersen, Fusari and Todorov (bid > 0, ask/bid <= 10,
      -15 < m < 5; no price, spread or liquidity threshold), saved in data/clean_aft/ and
      results/*_aft.csv, sample days only
"""

import sys, argparse, pathlib, datetime as dt
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import black

HOME = pathlib.Path.home() / "thesis_spy"
RAW, DATA = HOME / "data" / "raw", HOME / "data"
CLEAN, RESULTS = HOME / "data" / "clean", HOME / "results"
FILTERS = "main"                   # "main" = filters 1-9 of Chapter 4; "aft" = minimal filters (robustness)


def set_output(filters):
    """Output folder: data/clean for the main filters, data/clean_aft for the minimal filters."""
    global CLEAN, FILTERS
    FILTERS = filters
    CLEAN = HOME / "data" / ("clean" if filters == "main" else f"clean_{filters}")
    for d in (CLEAN, CLEAN / "plots", CLEAN / "counts", RESULTS):
        d.mkdir(parents=True, exist_ok=True)


set_output("main")

# ----------------------------------------------------------------- parameters (Chapter 4)
MID_MIN = 0.05                     # filter 2: minimum mid price, in dollars
OI_MIN = 100                       # filter 4: minimum open interest
M_LO, M_HI = -6.0, 3.0             # filter 6: window of standardized moneyness
MIN_STRIKES_KEEP = 4               # below this number the smile is dropped
MIN_STRIKES_CALIB = 8              # from this number on the smile can be used for calibration (calib_ok)
PARITY_STRIKES = 8                 # strike pairs closest to the money used for the forward
SNAP_HM = {"1030": (10, 30), "1130": (11, 30), "1430": (14, 30)}
VOL_COL = {"1030": "vol_1030", "1130": "vol_1030", "1430": "vol_1430"}   # 11:30 uses the volume at 10:30
MINUTES_YEAR = 525_600             # calendar minutes in a year
BUS_MINUTES_YEAR = 252 * 390       # trading minutes in a year (252 days of 390 minutes)

# NYSE closures in the data period (checked against the missing files) + end of 2026 (NYSE calendar)
HOLIDAYS = pd.to_datetime("""
2022-12-26 2023-01-02 2023-01-16 2023-02-20 2023-04-07 2023-05-29 2023-06-19 2023-07-04 2023-09-04
2023-11-23 2023-12-25 2024-01-01 2024-01-15 2024-02-19 2024-03-29 2024-05-27 2024-06-19 2024-07-04
2024-09-02 2024-11-28 2024-12-25 2025-01-01 2025-01-09 2025-01-20 2025-02-17 2025-04-18 2025-05-26
2025-06-19 2025-07-04 2025-09-01 2025-11-27 2025-12-25 2026-01-01 2026-01-19 2026-02-16 2026-04-03
2026-05-25 2026-06-19 2026-07-03 2026-09-07 2026-11-26 2026-12-25
""".split())
TRADING_DAYS = pd.bdate_range("2022-12-01", "2026-12-31").difference(HOLIDAYS)


def fred(series):
    """One FRED series saved in data/fred_<series>.csv, as a series indexed by date."""
    f = DATA / f"fred_{series}.csv"
    df = pd.read_csv(f)
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna().set_index("date")["value"]


TBILL = fred("DTB4WK") if (DATA / "fred_DTB4WK.csv").exists() else None


def tbill_rate(day):
    """4-week T-bill rate (as a decimal) of the last day before or on `day`."""
    if TBILL is None:
        return np.nan
    s = TBILL[TBILL.index <= day]
    return float(s.iloc[-1]) / 100 if len(s) else np.nan


def tau_calendar(day, hm, exp):
    """Time to expiry in years: calendar minutes from the snapshot to 16:00 New York on the expiry day."""
    minutes = (exp - day).days * 1440 + (16 * 60 - (hm[0] * 60 + hm[1]))
    return minutes / MINUTES_YEAR


def tau_business(day, hm, exp):
    """Time to expiry in years counting trading minutes only (robustness check)."""
    today = 16 * 60 - (hm[0] * 60 + hm[1])
    later = ((TRADING_DAYS > day) & (TRADING_DAYS <= exp)).sum()
    return (today + 390 * later) / BUS_MINUTES_YEAR


# ----------------------------------------------------------------- building blocks
def parity_forward(df, tau, r):
    """D from the T-bill, F from put-call parity (median of K + (C-P)/D over the 8 pairs around the money).
    df: options of one expiry after filters 1-4. The regression of C-P on K is kept as a diagnostic."""
    c = df[df.right == "CALL"][["strike", "mid"]].rename(columns={"mid": "C"})
    p = df[df.right == "PUT"][["strike", "mid"]].rename(columns={"mid": "P"})
    pairs = c.merge(p, on="strike")
    if len(pairs) < 4:
        return None
    pairs["diff"] = pairs.C - pairs.P
    k0 = pairs.loc[pairs["diff"].abs().idxmin(), "strike"]
    sel = pairs.assign(dist=(pairs.strike - k0).abs()).sort_values(["dist", "strike"]).head(PARITY_STRIKES)
    D = float(np.exp(-r * tau)) if np.isfinite(r) else 1.0
    est = sel.strike + sel["diff"] / D
    F = float(np.median(est))
    b, a = np.polyfit(sel.strike, sel["diff"], 1)
    F_reg, D_reg = (float(a / -b) if b != 0 else np.nan), float(-b)
    return dict(F=F, D=D, n_pairs=int(len(sel)), F_disp=float(est.max() - est.min()),
                F_reg=F_reg, D_reg=D_reg)


def sigma_atm(df, F):
    """ATM volatility: implied volatility at K = F, interpolated linearly between the put just
    below F and the call just above F (Chapter 3)."""
    ok = df.dropna(subset=["iv"])
    puts = ok[(ok.right == "PUT") & (ok.strike < F)]
    calls = ok[(ok.right == "CALL") & (ok.strike >= F)]
    if len(calls) and calls.strike.min() == F:
        return float(calls.loc[calls.strike.idxmin(), "iv"])
    if not len(puts) or not len(calls):
        return np.nan
    pl, cl = puts.loc[puts.strike.idxmax()], calls.loc[calls.strike.idxmin()]
    w = (F - pl.strike) / (cl.strike - pl.strike)
    return float((1 - w) * pl.iv + w * cl.iv)


def _violations(K, C, hs):
    """Indices involved in violations of monotonicity (C non-increasing in K) or convexity,
    with a tolerance equal to half the spread of the options involved. Returns (number, set of indices)."""
    n_viol, bad = 0, set()
    for i in range(len(K) - 1):
        if C[i + 1] - C[i] > hs[i] + hs[i + 1]:
            n_viol += 1
            bad |= {i, i + 1}
    for i in range(1, len(K) - 1):
        dk1, dk2 = K[i] - K[i - 1], K[i + 1] - K[i]
        s1, s2 = (C[i] - C[i - 1]) / dk1, (C[i + 1] - C[i]) / dk2
        tol = (hs[i - 1] + hs[i]) / dk1 + (hs[i] + hs[i + 1]) / dk2
        if s2 - s1 < -tol:
            n_viol += 1
            bad |= {i - 1, i, i + 1}
    return n_viol, bad


def no_arbitrage(df, F, D):
    """Filter 7, no static arbitrage. Puts are turned into call prices with put-call parity. The call
    prices C(K) must fall (or stay equal) when K rises and must be convex in K, with a tolerance of
    half the bid-ask spread. While there are violations, the option whose removal leaves the fewest
    violations is removed (in a tie, the one with the larger relative spread)."""
    df = df.sort_values("strike").copy()
    removed = 0
    for _ in range(100):
        K = df.strike.to_numpy(dtype=float)
        C = np.where(df.right == "CALL", df.mid, df.mid + D * (F - df.strike)).astype(float)
        hs = ((df.ask - df.bid) / 2).to_numpy(dtype=float)
        rel = df.spread_rel.to_numpy(dtype=float)
        n_viol, bad = _violations(K, C, hs)
        if n_viol == 0:
            break
        best, best_key = None, None
        for i in sorted(bad):
            keep = np.ones(len(K), bool); keep[i] = False
            n_after, _ = _violations(K[keep], C[keep], hs[keep])
            key = (n_after, -rel[i])
            if best_key is None or key < best_key:
                best, best_key = i, key
        df = df.drop(df.index[best])
        removed += 1
    return df, removed


def process_snapshot(day, snap, spread_max, verbose=False, plot=False, period_id=""):
    """Clean one quotes file (one day, one snapshot).

    Returns the cleaned options, the filter counts, the summary of each smile and a status text."""
    hm = SNAP_HM[snap]
    f = RAW / f"quotes_{day:%Y%m%d}_{snap}.parquet"
    if not f.exists():
        return None, None, None, "file missing"
    q = pd.read_parquet(f).drop_duplicates()
    q["expiration"] = pd.to_datetime(q["expiration"].astype(str))
    q["dte"] = (q.expiration - day).dt.days
    snap_time = day + pd.Timedelta(hours=hm[0], minutes=hm[1])
    q["age_sec"] = (snap_time - pd.to_datetime(q["timestamp"])).dt.total_seconds().clip(lower=0)

    # half day or stale snapshot: no quote was updated in the 30 minutes before the snapshot
    last_update = pd.to_datetime(q["timestamp"]).max()
    if (snap_time - last_update) > pd.Timedelta(minutes=30):
        status = "half day" if (snap == "1430" and last_update.hour <= 13) else "stale snapshot"
        return None, None, None, f"{status} (last update {last_update.time()})"

    # expiries: 0 to 7 days, plus the control expiry closest to 30 days (between 25 and 40)
    ctrl = q.loc[q.dte.between(25, 40), "dte"]
    ctrl_dte = int(ctrl.iloc[(ctrl - 30).abs().argmin()]) if len(ctrl) else None
    keep = q.dte.between(0, 7) | (q.dte == ctrl_dte)
    q = q[keep].copy()
    q["is_control"] = q.dte == ctrl_dte

    # open interest and volume (missing = 0)
    oi_f = RAW / f"oi_{day:%Y%m%d}.parquet"
    if oi_f.exists():
        oi = pd.read_parquet(oi_f)
        oi["expiration"] = pd.to_datetime(oi["expiration"].astype(str))
        q = q.merge(oi[["expiration", "strike", "right", "open_interest"]].drop_duplicates(
            ["expiration", "strike", "right"]), on=["expiration", "strike", "right"], how="left")
    else:
        q["open_interest"] = np.nan
    q["oi"] = q.open_interest.fillna(0).astype(int)
    q["volume"] = 0
    for exp in q.expiration.unique():
        vf = RAW / f"vol_{exp:%Y%m%d}.parquet"
        if not vf.exists():
            continue
        v = pd.read_parquet(vf)
        v = v[v["day"].astype(str) == str(day.date())]
        if not len(v):
            continue
        v = v[["strike", "right", VOL_COL[snap]]].rename(columns={VOL_COL[snap]: "vol_snap"})
        m = q.loc[q.expiration == exp, ["strike", "right"]].merge(v, on=["strike", "right"], how="left")
        q.loc[q.expiration == exp, "volume"] = m.vol_snap.fillna(0).to_numpy().astype(int)

    q["mid"] = (q.bid + q.ask) / 2
    q["spread_rel"] = np.where(q.mid > 0, (q.ask - q.bid) / q.mid, np.inf)
    r = tbill_rate(day)

    clean_parts, counts, summary = [], [], []
    for exp, g in q.groupby("expiration"):
        dte = int(g.dte.iloc[0])
        is_ctrl = bool(g.is_control.iloc[0])
        tau = tau_calendar(day, hm, exp)
        if tau <= 0:
            continue
        cnt = dict(day=day.date(), snap=snap, expiration=exp.date(), dte=dte, control=is_ctrl, n0_quote=len(g))
        g0 = g
        # filter 1: a real bid, and an ask above the bid
        g = g[(g.bid > 0) & (g.ask > g.bid)];                    cnt["f1_bid_ask"] = len(g)
        if FILTERS == "aft":                                     # minimal filters: ask/bid <= 10 instead of filters 2-4
            g = g[g.ask / g.bid <= 10];                          cnt["f2_mid"] = cnt["f3_spread"] = cnt["f4_liquidity"] = len(g)
        else:
            # filter 2: mid price of at least 5 cents; filter 3: relative spread not above spread_max;
            # filter 4: traded before the snapshot or open interest of at least 100 (only OI for the control expiry)
            g = g[g.mid >= MID_MIN];                                 cnt["f2_mid"] = len(g)
            g = g[g.spread_rel <= spread_max];                       cnt["f3_spread"] = len(g)
            liq = (g.oi >= OI_MIN) if is_ctrl else ((g.volume > 0) | (g.oi >= OI_MIN))
            g = g[liq];                                              cnt["f4_liquidity"] = len(g)
        # forward and discount factor from put-call parity (needed to know which options are OTM)
        fd = parity_forward(g, tau, r)
        if fd is None:
            cnt.update(f5_otm=0, f6_moneyness=0, f7_noarb=0, f8_iv=0, f9_final=0, note="parity not available")
            counts.append(cnt)
            continue
        F, D = fd["F"], fd["D"]
        # filter 5: out-of-the-money options only (puts below F, calls at or above F)
        g = g[((g.right == "PUT") & (g.strike < F)) | ((g.right == "CALL") & (g.strike >= F))].copy()
        cnt["f5_otm"] = len(g)
        # the same funnel of filters 1-4, counted on the OTM options only (filter table of Chapter 4)
        o = g0[((g0.right == "PUT") & (g0.strike < F)) | ((g0.right == "CALL") & (g0.strike >= F))]
        cnt["otm_n0"] = len(o)
        o = o[(o.bid > 0) & (o.ask > o.bid)];                    cnt["otm_f1_bid_ask"] = len(o)
        if FILTERS == "aft":
            o = o[o.ask / o.bid <= 10];                          cnt["otm_f2_mid"] = cnt["otm_f3_spread"] = cnt["otm_f4_liquidity"] = len(o)
        else:
            o = o[o.mid >= MID_MIN];                                 cnt["otm_f2_mid"] = len(o)
            o = o[o.spread_rel <= spread_max];                       cnt["otm_f3_spread"] = len(o)
            o = o[(o.oi >= OI_MIN) if is_ctrl else ((o.volume > 0) | (o.oi >= OI_MIN))]
            cnt["otm_f4_liquidity"] = len(o)
        # implied volatility of each option, then the ATM volatility of the smile
        is_call = (g.right == "CALL").to_numpy()
        g["iv"] = [black.implied_vol(p, F, k, tau, D, c) for p, k, c in zip(g.mid, g.strike, is_call)]
        s_atm = sigma_atm(g, F)
        if not np.isfinite(s_atm):
            cnt.update(f6_moneyness=0, f7_noarb=0, f8_iv=0, f9_final=0, note="sigma_ATM not available")
            counts.append(cnt)
            continue
        g["m"] = np.log(g.strike / F) / (s_atm * np.sqrt(tau))         # standardized moneyness
        # filter 6: moneyness window; filter 7: no arbitrage; filter 8: implied volatility exists
        lo, hi = (-15.0, 5.0) if FILTERS == "aft" else (M_LO, M_HI)
        g = g[g.m.between(lo, hi)];                              cnt["f6_moneyness"] = len(g)
        g, n_arb = no_arbitrage(g, F, D);                        cnt["f7_noarb"] = len(g)
        g = g.dropna(subset=["iv"]);                             cnt["f8_iv"] = len(g)
        # filter 9: at least 4 strikes to keep the smile, at least 8 to use it for calibration
        n_strikes = g.strike.nunique()
        if n_strikes < MIN_STRIKES_KEEP:
            cnt.update(f9_final=0, f9b_calib=0, note=f"only {n_strikes} strikes")
            counts.append(cnt)
            continue
        calib_ok = n_strikes >= MIN_STRIKES_CALIB
        cnt["f9_final"] = len(g); cnt["f9b_calib"] = len(g) if calib_ok else 0
        cnt["note"] = "" if calib_ok else f"only {n_strikes} strikes: test only"
        counts.append(cnt)

        # Greeks and smile-level values stored with each option
        is_call = (g.right == "CALL").to_numpy()
        g["vega"] = black.vega(F, g.strike.to_numpy(), tau, g.iv.to_numpy(), D)
        g["delta"] = black.delta_fwd(F, g.strike.to_numpy(), tau, g.iv.to_numpy(), is_call)
        g["F"], g["D"], g["calib_ok"] = F, D, calib_ok
        g["tau"], g["tau_bus"] = tau, tau_business(day, hm, exp)
        g["sigma_atm"], g["period_id"], g["snap"], g["day"] = s_atm, period_id, snap, day.date()
        clean_parts.append(g)
        summary.append(dict(day=day.date(), snap=snap, expiration=exp.date(), dte=dte, control=is_ctrl,
                            n_options=len(g), n_strikes=int(n_strikes), calib_ok=calib_ok,
                            n_puts=int((~is_call).sum()), n_calls=int(is_call.sum()),
                            F=round(F, 4), D=round(D, 6), F_disp=round(fd["F_disp"], 4),
                            F_reg=round(fd["F_reg"], 4), D_reg=round(fd["D_reg"], 5),
                            tau=tau, sigma_atm=round(s_atm, 5),
                            m_min=round(g.m.min(), 2), m_max=round(g.m.max(), 2),
                            age_median_sec=round(g.age_sec.median(), 1),
                            share_age_gt60s=round((g.age_sec > 60).mean(), 3),
                            n_noarb_removed=n_arb, period_id=period_id))

    counts, summary = pd.DataFrame(counts), pd.DataFrame(summary)
    clean = pd.concat(clean_parts, ignore_index=True) if clean_parts else pd.DataFrame()
    if len(clean):
        cols = ["day", "snap", "period_id", "expiration", "dte", "is_control", "calib_ok", "tau", "tau_bus",
                "F", "D", "sigma_atm", "right", "strike", "bid", "ask", "mid", "spread_rel", "volume", "oi",
                "age_sec", "iv", "vega", "delta", "m"]
        clean = clean[cols].sort_values(["expiration", "strike"]).reset_index(drop=True)
        clean.to_parquet(CLEAN / f"clean_{day:%Y%m%d}_{snap}.parquet", index=False)
    counts.to_csv(CLEAN / "counts" / f"counts_{day:%Y%m%d}_{snap}.csv", index=False)
    summary.to_csv(CLEAN / "counts" / f"summary_{day:%Y%m%d}_{snap}.csv", index=False)

    if verbose:
        print(f"\n--- {day.date()} {snap[:2]}:{snap[2:]} | last quote update: {last_update.time()} "
              f"| T-bill 4w: {r*100:.2f}% | spread max {spread_max:.0%}")
        show = ["dte", "control", "n0_quote", "f1_bid_ask", "f2_mid", "f3_spread", "f4_liquidity",
                "f5_otm", "f6_moneyness", "f7_noarb", "f8_iv", "f9_final", "f9b_calib", "note"]
        print(counts[show].to_string(index=False))
        if len(summary):
            show2 = ["dte", "n_options", "n_puts", "n_calls", "calib_ok", "F", "F_reg", "F_disp", "D", "D_reg",
                     "sigma_atm", "m_min", "m_max", "age_median_sec", "share_age_gt60s", "n_noarb_removed"]
            print(summary[show2].to_string(index=False))
    if plot and len(clean):
        plot_smiles(clean, day, snap)
    return clean, counts, summary, "ok"


def plot_smiles(clean, day, snap):
    """Plot of the cleaned smiles of one snapshot, for the visual check (local use only)."""
    exps = sorted(clean.expiration.unique())
    n = len(exps)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.6 * rows), squeeze=False)
    for ax, exp in zip(axes.ravel(), exps):
        g = clean[clean.expiration == exp]
        p, c = g[g.right == "PUT"], g[g.right == "CALL"]
        ax.plot(p.m, p.iv * 100, "o", ms=3, label="put")
        ax.plot(c.m, c.iv * 100, "^", ms=3, label="call")
        ax.axvline(0, color="grey", lw=0.5)
        ax.set_title(f"{pd.Timestamp(exp).date()} | dte {int(g.dte.iloc[0])} | n={len(g)} | F={g.F.iloc[0]:.2f} | "
                     f"sigma_ATM={g.sigma_atm.iloc[0]*100:.1f}%", fontsize=8)
        ax.set_xlabel("m = ln(K/F)/(sigma_ATM sqrt(tau))", fontsize=8); ax.set_ylabel("IV (%)", fontsize=8)
        ax.tick_params(labelsize=7); ax.legend(fontsize=7)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(f"SPY smiles {day.date()} {snap[:2]}:{snap[2:]}", fontsize=10)
    fig.tight_layout()
    fig.savefig(CLEAN / "plots" / f"smiles_{day:%Y%m%d}_{snap}.png", dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("day", nargs="?", help="YYYYMMDD (omitted with --all)")
    ap.add_argument("--all", action="store_true", help="all trading days")
    ap.add_argument("--snap", nargs="+", default=["1030", "1430"], help="snapshot: 1030 1430 1130")
    ap.add_argument("--spread", type=float, default=0.50, help="maximum relative spread (default 0.50)")
    ap.add_argument("--plot", action="store_true", help="smile plots also outside the sample")
    ap.add_argument("--filters", choices=["main", "aft"], default="main", help="main = filters 1-9; aft = minimal AFT filters")
    ap.add_argument("--days", choices=["all", "sample"], default="all", help="with --all: all days or the sample days only")
    a = ap.parse_args()
    set_output(a.filters)

    sample = pd.read_csv(DATA / "sample_dates.csv") if (DATA / "sample_dates.csv").exists() else pd.DataFrame()
    pid = {}
    for _, row in sample.iterrows():
        pid[pd.Timestamp(row.t)] = str(row.period_id)
        pid[pd.Timestamp(row.t1)] = str(row.period_id)

    if a.all:
        days = sorted(pd.Timestamp(f.name[7:15]) for f in RAW.glob("quotes_*_1030.parquet"))
        if a.days == "sample":
            days = [d for d in days if d in pid]
    elif a.day:
        days = [pd.Timestamp(a.day)]
    else:
        ap.error("give a day YYYYMMDD or --all")

    log, all_counts, all_summary = [], [], []
    t0 = dt.datetime.now()
    for i, day in enumerate(days, 1):
        for snap in a.snap:
            in_sample = day in pid
            clean, counts, summary, status = process_snapshot(
                day, snap, a.spread, verbose=not a.all, plot=(a.plot or in_sample), period_id=pid.get(day, ""))
            log.append(dict(day=day.date(), snap=snap, status=status,
                            n_smiles=0 if summary is None else len(summary),
                            n_options=0 if clean is None else len(clean)))
            if counts is not None:
                all_counts.append(counts)
            if summary is not None:
                all_summary.append(summary)
            if status != "ok" and a.all:
                print(f"{day.date()} {snap}: {status}")
        if a.all and (i % 50 == 0 or i == len(days)):
            el = (dt.datetime.now() - t0).total_seconds() / 60
            print(f"... {i}/{len(days)} days | {el:.1f} min elapsed | about {el / i * (len(days) - i):.1f} min left")

    log = pd.DataFrame(log)
    tag = "" if abs(a.spread - 0.5) < 1e-9 else f"_spread{int(a.spread*100)}"
    if a.filters != "main":
        tag += f"_{a.filters}"
    if a.all:
        pd.concat(all_counts).to_csv(RESULTS / f"filter_counts{tag}.csv", index=False)
        pd.concat(all_summary).to_csv(RESULTS / f"smile_summary{tag}.csv", index=False)
        log.to_csv(RESULTS / f"clean_log{tag}.csv", index=False)
        print("\nSnapshot status:")
        print(log.groupby(["snap", "status"]).size().to_string())
        print(f"\nCleaned smiles: {log.n_smiles.sum()} | options: {log.n_options.sum()}")
        print(f"Saved results/filter_counts{tag}.csv, results/smile_summary{tag}.csv, results/clean_log{tag}.csv")
    else:
        print("\nStatus:", log.to_dict("records"))
        print(f"Files saved in {CLEAN} (parquet, counts/, plots/)")


if __name__ == "__main__":
    main()
