"""
Run the three models on all sample periods (the main computation of Chapter 5 of the thesis).

For each period (t, t+1) in data/sample_dates.csv and each expiry that can be calibrated at the
calibration snapshot (t 10:30 by default):
  - in sample: Design A (Black-Scholes, SABR on 3 pillars, Vanna-Volga on 3 pillars, for the
    pillar sets P1, P2 and P3) and Design B (SABR on every second strike);
  - out of sample, intraday: the models fitted at t 10:30 are applied to the same expiry at t 14:30;
  - out of sample, next day: the same models are applied to the same expiry at t+1 10:30.
Out of sample, no model is fitted again. Black-Scholes keeps its ATM volatility; SABR keeps
alpha, rho and nu and uses the new forward and time to expiry; Vanna-Volga keeps the implied
volatility and the delta of each pillar ("sticky delta"). A simple benchmark, "sticky strike",
keeps the implied volatility that each strike had at calibration.

Output: results/model_results<tag>.csv, one row per smile, stage, design, model, pillar set and
region, with error measures and parameters only (no row for a single option), and
results/model_log<tag>.csv with the smiles that were skipped.

How to run (from the src folder):
  python3 run_models.py                     main specification
  python3 run_models.py --beta 0.5          robustness: SABR with beta = 0.5
  python3 run_models.py --vega              robustness: vega-weighted SABR calibration
  python3 run_models.py --time business     robustness: time to expiry in trading minutes
  python3 run_models.py --calib 1130        robustness: calibration at 11:30
  python3 run_models.py --spread 0.3        robustness: only options with relative spread <= 30%
  python3 run_models.py --filters aft       robustness: minimal filters (reads data/clean_aft/, tag _aft)
"""

import sys, argparse, pathlib, datetime as dt
import numpy as np
import pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import black, sabr, vv, evaluate

HOME = pathlib.Path.home() / "thesis_spy"         # project folder
CLEAN, DATA, RESULTS = HOME / "data" / "clean", HOME / "data", HOME / "results"
MAX_OPTIONS_SMILE = 400           # safety cap: skip absurdly large smiles (never hit with the main filters)
PILLAR_SETS = ("P1", "P2", "P3")
MIN_TEST = 4                      # minimum number of options for an out-of-sample test smile

def load(day, snap):
    """Cleaned options of one day and snapshot (None if the file does not exist)."""
    f = CLEAN / f"clean_{day:%Y%m%d}_{snap}.parquet"
    return pd.read_parquet(f) if f.exists() else None


def business_time(sm):
    """Robustness check: recompute IV, vega, delta, ATM volatility and moneyness of one smile with
    business time (tau_bus, trading minutes only) instead of calendar time."""
    sm = sm.copy()
    F, D, tau = float(sm.F.iloc[0]), float(sm.D.iloc[0]), float(sm.tau_bus.iloc[0])
    is_call = (sm.right == "CALL").to_numpy()
    sm["iv"] = [black.implied_vol(p, F, k, tau, D, c) for p, k, c in zip(sm.mid, sm.strike, is_call)]
    sm = sm.dropna(subset=["iv"])
    if len(sm) < 2:
        return None
    is_call = (sm.right == "CALL").to_numpy()
    sm["vega"] = black.vega(F, sm.strike.to_numpy(), tau, sm.iv.to_numpy(), D)
    sm["delta"] = black.delta_fwd(F, sm.strike.to_numpy(), tau, sm.iv.to_numpy(), is_call)
    puts, calls = sm[(sm.right == "PUT") & (sm.strike < F)], sm[(sm.right == "CALL") & (sm.strike >= F)]
    if not len(puts) or not len(calls):
        return None
    pl, cl = puts.loc[puts.strike.idxmax()], calls.loc[calls.strike.idxmin()]
    w = (F - pl.strike) / (cl.strike - pl.strike)
    s_atm = (1 - w) * pl.iv + w * cl.iv
    sm["sigma_atm"], sm["tau"] = s_atm, tau
    sm["m"] = np.log(sm.strike / F) / (s_atm * np.sqrt(tau))
    return sm


def add_regions(rows_base, test, iv_model, price_model, extra=None, more=()):
    """One result row per region: all + the five moneyness groups (by the market delta of the test options),
    plus any extra (name, mask) pairs in `more` (e.g. the strikes common to both snapshots)."""
    out = []
    for name, mask in list(evaluate.regions_of(test)) + list(more):
        if mask.sum() == 0:
            continue
        r = dict(rows_base, region=name, **evaluate.metrics(test[mask], iv_model[mask], price_model[mask]))
        if extra:
            r.update(extra)
        out.append(r)
    return out


def apply_oos(fitted, sm_cal, sm_test, stage, base, beta):
    """Out-of-sample test: apply the models fitted on sm_cal (day t, calibration snapshot) to sm_test
    (same expiry at a later snapshot) without fitting them again."""
    rows = []
    F, D, tau = float(sm_test.F.iloc[0]), float(sm_test.D.iloc[0]), float(sm_test.tau.iloc[0])
    K = sm_test.strike.to_numpy()
    is_call = (sm_test.right == "CALL").to_numpy()
    b = dict(base, stage=stage, snap_test=sm_test.snap.iloc[0], day_test=sm_test.day.iloc[0],
             dte_test=int(sm_test.dte.iloc[0]), tau_test=tau, F_test=F, n_test_smile=len(sm_test))

    # sticky strike: each strike keeps the IV it had at calibration (only strikes present at both times)
    m = sm_test[["strike", "right"]].merge(sm_cal[["strike", "right", "iv"]], on=["strike", "right"], how="left")
    iv_ss = m.iv.to_numpy()
    common = [("common", np.isfinite(iv_ss))]          # strikes quoted at both snapshots: fair comparison
    b["n_common"] = int(np.isfinite(iv_ss).sum())
    rows += add_regions(dict(b, design="A", model="StickyStrike", pillar_set="-"), sm_test, iv_ss,
                        evaluate._prices_from_iv(sm_test, iv_ss, F, tau, D), more=common)
    # BS keeps sigma_ATM of the calibration snapshot
    iv_bs = np.full(len(sm_test), fitted["sigma_atm"])
    rows += add_regions(dict(b, design="A", model="BS", pillar_set="-"), sm_test, iv_bs,
                        evaluate._prices_from_iv(sm_test, iv_bs, F, tau, D), more=common)
    # SABR, Design A (each pillar set) and Design B: same parameters, new forward and time to expiry
    for pset, cal in fitted["sabr_A"].items():
        if not np.isfinite(cal["alpha"]):
            continue
        ivs = sabr.hagan_iv(K, F, tau, cal["alpha"], cal["rho"], cal["nu"], beta)
        rows += add_regions(dict(b, design="A", model="SABR", pillar_set=pset), sm_test, ivs,
                            evaluate._prices_from_iv(sm_test, ivs, F, tau, D),
                            extra=dict(alpha=cal["alpha"], rho=cal["rho"], nu=cal["nu"]), more=common)
    if "sabr_B" in fitted and np.isfinite(fitted["sabr_B"]["alpha"]):
        cal = fitted["sabr_B"]
        ivs = sabr.hagan_iv(K, F, tau, cal["alpha"], cal["rho"], cal["nu"], beta)
        rows += add_regions(dict(b, design="B", model="SABR", pillar_set="odd/even"), sm_test, ivs,
                            evaluate._prices_from_iv(sm_test, ivs, F, tau, D),
                            extra=dict(alpha=cal["alpha"], rho=cal["rho"], nu=cal["nu"]), more=common)
    # Vanna-Volga, sticky delta (each pillar set): the pillars move with the forward, see vv.shifted
    for pset, (model, delta_call) in fitted["vv"].items():
        sh = model.shifted(F, tau, D, delta_call)
        p = sh.price(K, is_call)
        ivv = sh.iv(K, is_call)
        rows += add_regions(dict(b, design="A", model="VV", pillar_set=pset), sm_test, ivv, p,
                            extra=dict(K1=sh.K[0], K2=sh.K[1], K3=sh.K[2], sigma_ref=sh.sref), more=common)
    return rows


def main():
    # ------------------------------------------------ options of the run and name of the output file
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--vega", action="store_true", help="vega-weighted SABR calibration")
    ap.add_argument("--time", choices=["calendar", "business"], default="calendar")
    ap.add_argument("--calib", default="1030", help="calibration snapshot: 1030 or 1130")
    ap.add_argument("--spread", type=float, default=None, help="keep only options with spread_rel <= this")
    ap.add_argument("--tag", default="", help="suffix for the output file (default: built from the options)")
    ap.add_argument("--filters", choices=["main", "aft"], default="main", help="which cleaned data set to use")
    a = ap.parse_args()
    global CLEAN
    if a.filters != "main":
        CLEAN = HOME / "data" / f"clean_{a.filters}"
    tag = a.tag or "".join([f"_beta{a.beta:g}" if a.beta != 1.0 else "", "_vega" if a.vega else "",
                            "_bustime" if a.time == "business" else "", f"_calib{a.calib}" if a.calib != "1030" else "",
                            f"_spread{int(a.spread*100)}" if a.spread else "",
                            f"_{a.filters}" if a.filters != "main" else ""])
    # ------------------------------------------------ loop over the sample periods and the expiries
    sample = pd.read_csv(DATA / "sample_dates.csv")
    rows, log = [], []
    t0 = dt.datetime.now()
    for _, s in sample.iterrows():
        t, t1 = pd.Timestamp(s.t), pd.Timestamp(s.t1)
        cal_df, intra_df, next_df = load(t, a.calib), load(t, "1430"), load(t1, a.calib)
        if cal_df is None:
            log.append(dict(period=s.period_id, note="calibration file missing")); continue
        if a.spread:
            cal_df, intra_df, next_df = [None if d is None else d[d.spread_rel <= a.spread] for d in (cal_df, intra_df, next_df)]
        for exp, sm in cal_df.groupby("expiration"):
            sm = sm.sort_values("strike").reset_index(drop=True)
            if a.time == "business":
                sm = business_time(sm)
                if sm is None:
                    continue
            if not (sm.calib_ok.iloc[0] and sm.strike.nunique() >= 8):      # at least 8 strikes to calibrate
                log.append(dict(period=s.period_id, day=t.date(), expiration=pd.Timestamp(exp).date(),
                                note=f"not calibratable ({sm.strike.nunique()} strikes)"))
                continue
            dte = int(sm.dte.iloc[0])
            base = dict(period_id=str(s.period_id), group=s.group, regime=s.regime, day=t.date(), snap_calib=a.calib,
                        expiration=pd.Timestamp(exp).date(), dte=dte, control=bool(sm.is_control.iloc[0]),
                        tau=float(sm.tau.iloc[0]), F=float(sm.F.iloc[0]), sigma_atm=float(sm.sigma_atm.iloc[0]),
                        n_options=len(sm), vix_tm1=s.vix_tm1)
            # in-sample
            r_in, fitted = evaluate.eval_smile(sm, beta=a.beta, vega_weights=a.vega, pillar_sets=PILLAR_SETS)
            for r in r_in:
                r.update(base); r["stage"] = "insample"
            rows += r_in
            # out-of-sample
            for stage, df in (("intraday", intra_df), ("nextday", next_df)):
                if df is None:
                    continue
                st = df[df.expiration == exp].sort_values("strike").reset_index(drop=True)
                if a.time == "business" and len(st):
                    st = business_time(st)
                if st is None or len(st) < MIN_TEST:
                    continue
                rows += apply_oos(fitted, sm, st, stage, base, a.beta)
        print(f"period {s.period_id} ({t.date()} -> {t1.date()}, {s.regime}) done | {len(rows)} rows so far")

    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / f"model_results{tag}.csv", index=False)
    pd.DataFrame(log).to_csv(RESULTS / f"model_log{tag}.csv", index=False)
    el = (dt.datetime.now() - t0).total_seconds() / 60
    print(f"\nDone in {el:.1f} min: {len(res)} result rows -> results/model_results{tag}.csv")
    print("Smiles by stage (BS rows, region = all):")
    print(res[(res.region == "all") & (res.model == "BS") & (res.pillar_set.isin(["P1", "-"]))]
          .groupby(["stage"]).size().to_string())


if __name__ == "__main__":
    main()
