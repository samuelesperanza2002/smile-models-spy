"""
In-sample test of the three models on one smile (Chapters 3 and 4 of the thesis).

A smile is the set of cleaned out-of-the-money options of one expiry at one snapshot.
The models are compared in two designs:
  Design A  the three models get the same information. Black-Scholes uses the at-the-money
            volatility; SABR and Vanna-Volga are fitted to the same three pillars. The test uses
            all the other options of the smile.
  Design B  SABR gets more information: it is fitted to every second strike and tested on the
            strikes in between.
One "result row" describes one smile, design, model, pillar set and region, with its error
measures. A region is a group of test options: all of them, those inside or outside the pillar
range, or one of the five moneyness groups.
"""

import numpy as np
import pandas as pd
import black, sabr, vv

MODELS_A = ("BS", "SABR", "VV")

# the five moneyness groups, defined by the forward delta of the market option (Chapter 4);
# d = delta, c = True for a call
MNY = [("put_deep", lambda d, c: (~c) & (d > -0.10)),
       ("put_otm", lambda d, c: (~c) & (d <= -0.10) & (d > -0.25)),
       ("near_atm", lambda d, c: ((~c) & (d <= -0.25)) | (c & (d >= 0.25))),
       ("call_otm", lambda d, c: c & (d < 0.25) & (d >= 0.10)),
       ("call_deep", lambda d, c: c & (d < 0.10))]


def regions_of(test, inside=None):
    """Groups of test options as (name, mask) pairs: all, inside and outside the pillar range
    (when `inside` is given), and the five moneyness groups."""
    d, c = test.delta.to_numpy(), (test.right == "CALL").to_numpy()
    out = [("all", np.ones(len(test), bool))]
    if inside is not None:
        out += [("inside", inside), ("outside", ~inside)]
    out += [(name, f(d, c)) for name, f in MNY]
    return out


def metrics(test, iv_model, price_model):
    """Error measures of one model on a set of test options.

    iv_model, price_model: model implied volatilities and prices (nan = the model failed).
    IV errors are in volatility points (model IV - market IV) x 100; price errors are in dollars
    per share; inside_spread is the share of model prices between the market bid and ask."""
    ok = np.isfinite(iv_model) & np.isfinite(price_model)
    n = len(test)
    out = dict(n_test=n, n_fail=int((~ok).sum()), fail_rate=float((~ok).mean()) if n else np.nan)
    if ok.sum() == 0:
        out.update(iv_rmse=np.nan, iv_mae=np.nan, iv_bias=np.nan, iv_rmse_rel=np.nan, price_rmse=np.nan, price_mae=np.nan, inside_spread=np.nan)
        return out
    e = (iv_model[ok] - test.iv.to_numpy()[ok]) * 100
    pe = price_model[ok] - test.mid.to_numpy()[ok]
    inside = (price_model[ok] >= test.bid.to_numpy()[ok] - 1e-9) & (price_model[ok] <= test.ask.to_numpy()[ok] + 1e-9)
    er = (iv_model[ok] / test.iv.to_numpy()[ok] - 1) * 100          # relative IV error in %
    out.update(iv_rmse=float(np.sqrt(np.mean(e ** 2))), iv_mae=float(np.mean(np.abs(e))), iv_bias=float(np.mean(e)),
               iv_rmse_rel=float(np.sqrt(np.mean(er ** 2))),
               price_rmse=float(np.sqrt(np.mean(pe ** 2))), price_mae=float(np.mean(np.abs(pe))),
               inside_spread=float(inside.mean()))
    return out


def _prices_from_iv(test, iv_model, F, tau, D):
    """Black prices of the test options from model implied volatilities (nan stays nan)."""
    is_call = (test.right == "CALL").to_numpy()
    p = np.full(len(test), np.nan)
    ok = np.isfinite(iv_model)
    if ok.any():
        p[ok] = black.price(F, test.strike.to_numpy()[ok], tau, iv_model[ok], D, is_call[ok])
    return p


def eval_smile(sm, beta=1.0, vega_weights=False, pillar_sets=("P1", "P2", "P3")):
    """Fit and test all models on one smile.

    sm: cleaned table of one smile (same day, snapshot and expiry) that can be used for calibration.
    Returns the list of result rows and a dictionary with the fitted models, which run_models.py
    reuses for the out-of-sample tests."""
    sm = sm.sort_values("strike").reset_index(drop=True)
    F, D, tau, s_atm = float(sm.F.iloc[0]), float(sm.D.iloc[0]), float(sm.tau.iloc[0]), float(sm.sigma_atm.iloc[0])
    base = dict(day=sm.day.iloc[0], snap=sm.snap.iloc[0], expiration=sm.expiration.iloc[0], dte=int(sm.dte.iloc[0]),
                control=bool(sm.is_control.iloc[0]), period_id=sm.period_id.iloc[0], tau=tau, F=F, sigma_atm=s_atm,
                n_options=len(sm))
    rows, fitted = [], dict(F=F, D=D, tau=tau, sigma_atm=s_atm, sabr_A={}, vv={}, pillars={})
    w = sm.vega.to_numpy() if vega_weights else None

    # ---------------- Design A: for each pillar set, BS, SABR on 3 pillars and VV on 3 pillars,
    #                  tested on the options that are not pillars
    for pset in pillar_sets:
        pil = vv.select_pillars(sm, pset)
        if pil is None:
            rows.append(dict(base, design="A", model="(pillar)", pillar_set=pset, region="all", n_test=0,
                             note="pillars not distinct"))
            continue
        pidx = set(pil.idx)
        test = sm[~sm.index.isin(pidx)]
        is_call = (test.right == "CALL").to_numpy()
        K = test.strike.to_numpy()
        K1, K3 = pil.strike.min(), pil.strike.max()
        inside = (K >= K1) & (K <= K3)
        fitted["pillars"][pset] = pil

        # Black-Scholes: one flat volatility, the ATM volatility of the smile
        iv_bs = np.full(len(test), s_atm)
        # SABR calibrated on the 3 pillars
        wp = sm.loc[list(pidx), "vega"].to_numpy() if vega_weights else None
        cal = sabr.calibrate(pil.strike.to_numpy(), pil.iv.to_numpy(), F, tau, beta=beta, sigma_atm=s_atm, weights=wp)
        fitted["sabr_A"][pset] = cal
        iv_sabr = sabr.hagan_iv(K, F, tau, cal["alpha"], cal["rho"], cal["nu"], beta) if np.isfinite(cal["alpha"]) \
            else np.full(len(test), np.nan)
        arb = sabr.check_arbitrage(F, tau, D, cal, sm.strike.min(), sm.strike.max()) if np.isfinite(cal["alpha"]) \
            else dict(n_viol=np.nan, worst=np.nan)
        # exact Vanna-Volga on the same 3 pillars
        model = vv.VannaVolga(F, tau, D, pil.strike.to_numpy(), pil.iv.to_numpy(), vv.ref_index(pset))
        fitted["vv"][pset] = (model, pil.delta_call.to_numpy())
        price_vv = model.price(K, is_call)
        iv_vv = model.iv(K, is_call)

        for mname, ivm, pm in [("BS", iv_bs, None), ("SABR", iv_sabr, None), ("VV", iv_vv, price_vv)]:
            pm = _prices_from_iv(test, ivm, F, tau, D) if pm is None else pm
            for region, mask in regions_of(test, inside):
                if mask.sum() == 0:
                    continue
                r = dict(base, design="A", model=mname, pillar_set=pset, region=region,
                         **metrics(test[mask], ivm[mask], pm[mask]))
                if mname == "SABR":
                    r.update(alpha=cal["alpha"], rho=cal["rho"], nu=cal["nu"], beta=beta, converged=cal["converged"],
                             at_bound=cal["at_bound"], calib_rmse=cal["rmse"], arb_viol=arb["n_viol"])
                if mname == "VV":
                    r.update(K1=K1, K2=pil.strike.iloc[1], K3=K3, sigma_ref=model.sref)
                rows.append(r)

    # ---------------- Design B: SABR fitted to the 2nd, 4th, 6th ... strike, tested on the 1st, 3rd, 5th ...
    odd = sm.iloc[1::2]
    even = sm.iloc[0::2]
    if len(odd) >= 3 and len(even) >= 1:
        wo = odd.vega.to_numpy() if vega_weights else None
        calB = sabr.calibrate(odd.strike.to_numpy(), odd.iv.to_numpy(), F, tau, beta=beta, sigma_atm=s_atm, weights=wo)
        fitted["sabr_B"] = calB
        K = even.strike.to_numpy()
        iv_sabrB = sabr.hagan_iv(K, F, tau, calB["alpha"], calB["rho"], calB["nu"], beta) if np.isfinite(calB["alpha"]) \
            else np.full(len(even), np.nan)
        arbB = sabr.check_arbitrage(F, tau, D, calB, sm.strike.min(), sm.strike.max()) if np.isfinite(calB["alpha"]) \
            else dict(n_viol=np.nan)
        for mname, ivm in [("BS", np.full(len(even), s_atm)), ("SABR", iv_sabrB)]:
            pm = _prices_from_iv(even, ivm, F, tau, D)
            for region, mask in regions_of(even):
                if mask.sum() == 0:
                    continue
                r = dict(base, design="B", model=mname, pillar_set="odd/even", region=region,
                         **metrics(even[mask], ivm[mask], pm[mask]))
                if mname == "SABR":
                    r.update(alpha=calB["alpha"], rho=calB["rho"], nu=calB["nu"], beta=beta, converged=calB["converged"],
                             at_bound=calB["at_bound"], calib_rmse=calB["rmse"], arb_viol=arbB["n_viol"], n_calib=len(odd))
                rows.append(r)
    return rows, fitted
