"""
Trading exercise with transaction costs (Chapter 6 of the thesis).

At t 10:30, for each smile of the regular sample periods that can be calibrated, each model gives
a value V to every option that is not a pillar and whose forward delta is between DELTA_MIN and
DELTA_MAX in absolute value. Then:
  buy at the ask   if V > ask + c
  sell at the bid  if V < bid - c
where c = COMMISSION per share (the half spread is already paid by trading at the ask or bid).
Each position is delta hedged with SPY at the forward F. There are no stock data, so a cost of
HEDGE_COST per share is assumed. Positions are closed at t 14:30 at the market bid or ask
(a buy is sold at the bid, a sale is bought back at the ask) and the hedge is closed at the new
forward. Variant: positions with 1 to 7 days to expiry are closed at t+1 10:30 instead.

Output: results/trading_results.csv (one row per closing time, model and maturity), results by
side and by regime, a summary per smile (results/trading_by_smile.csv) and the LaTeX table.
Results are aggregated: no row for a single trade is saved (data licence).

How to run (from the src folder):  python3 trading.py
"""

import sys, pathlib
import numpy as np
import pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import black, sabr, vv, evaluate

HOME = pathlib.Path.home() / "thesis_spy"
CLEAN, DATA, RESULTS = HOME / "data" / "clean", HOME / "data", HOME / "results"
COMMISSION = 0.65 / 100          # $ per share: 0.65 $ per contract (typical US retail broker), stated as an assumption
HEDGE_COST = 0.01                # $ per SPY share, each way (assumption, Chapter 6)
DELTA_MIN, DELTA_MAX = 0.05, 0.50   # options traded: 0.05 <= |delta| <= 0.50
CONTRACT = 100                   # one contract = 100 shares
MODELS = ["VV:P1", "SABR:P1", "SABR(B)"]
BUCKETS = ["0", "1", "2", "3-7", "30"]


def load(day, snap):
    """Cleaned options of one day and snapshot (None if the file does not exist)."""
    f = CLEAN / f"clean_{day:%Y%m%d}_{snap}.parquet"
    return pd.read_parquet(f) if f.exists() else None


def model_values(sm):
    """Model values at t 10:30 for the options of one smile.

    Returns the sorted smile and a dictionary model -> (mask, values); the mask marks the options
    the model may trade (not pillars for Design A, not calibration strikes for Design B)."""
    sm = sm.sort_values("strike").reset_index(drop=True)
    F, D, tau, s_atm = float(sm.F.iloc[0]), float(sm.D.iloc[0]), float(sm.tau.iloc[0]), float(sm.sigma_atm.iloc[0])
    K, is_call = sm.strike.to_numpy(), (sm.right == "CALL").to_numpy()
    out = {}
    pil = vv.select_pillars(sm, "P1")
    if pil is not None:
        non_pillar = ~sm.index.isin(set(pil.idx))
        m = vv.VannaVolga(F, tau, D, pil.strike.to_numpy(), pil.iv.to_numpy(), vv.ref_index("P1"))
        out["VV:P1"] = (non_pillar, m.price(K, is_call))
        cal = sabr.calibrate(pil.strike.to_numpy(), pil.iv.to_numpy(), F, tau, beta=1.0, sigma_atm=s_atm)
        if np.isfinite(cal["alpha"]):
            iv = sabr.hagan_iv(K, F, tau, cal["alpha"], cal["rho"], cal["nu"], 1.0)
            out["SABR:P1"] = (non_pillar, black.price(F, K, tau, iv, D, is_call))
    odd = sm.iloc[1::2]
    if len(odd) >= 3:
        calB = sabr.calibrate(odd.strike.to_numpy(), odd.iv.to_numpy(), F, tau, beta=1.0, sigma_atm=s_atm)
        if np.isfinite(calB["alpha"]):
            iv = sabr.hagan_iv(K, F, tau, calB["alpha"], calB["rho"], calB["nu"], 1.0)
            even_mask = np.zeros(len(sm), bool); even_mask[0::2] = True       # test only on strikes not used
            out["SABR(B)"] = (even_mask, black.price(F, K, tau, iv, D, is_call))
    return sm, out


def trades_for_smile(sm, out, close_df, close_label):
    """Simulate the trades of one smile, closed at the snapshot in close_df.

    Returns one dictionary per trade; they are only kept in memory and aggregated in main()."""
    rows = []
    if close_df is None:
        return rows
    cl = close_df[close_df.expiration == sm.expiration.iloc[0]]
    if not len(cl):
        return rows
    cl = cl.set_index(["strike", "right"])
    F0, F1 = float(sm.F.iloc[0]), float(cl.F.iloc[0])
    for model, (mask, V) in out.items():
        for i in np.where(mask)[0]:
            o = sm.iloc[i]
            if not (DELTA_MIN <= abs(o.delta) <= DELTA_MAX):
                continue
            key = (o.strike, o.right)
            if key not in cl.index:
                continue
            c = cl.loc[key]
            if V[i] > o.ask + COMMISSION:
                side, open_px, close_px = +1, o.ask, c.bid          # buy at ask, sell at bid later
            elif V[i] < o.bid - COMMISSION:
                side, open_px, close_px = -1, o.bid, c.ask          # sell at bid, buy back at ask later
            else:
                continue
            opt_pnl = side * (close_px - open_px)                   # profit on the option, $ per share, before costs
            hedge_pnl = -side * o.delta * (F1 - F0)                 # hedge: opposite position of delta shares
            costs = 2 * COMMISSION + 2 * HEDGE_COST * abs(o.delta)  # paid when opening and when closing
            mispricing = (V[i] - o.mid)
            rows.append(dict(model=model, dte=int(o.dte), control=bool(o.is_control), side=side, delta=float(o.delta),
                             mispricing=float(mispricing), spread=float(o.ask - o.bid),
                             pnl_gross=float(opt_pnl), pnl_hedged=float(opt_pnl + hedge_pnl),
                             pnl_net=float(opt_pnl + hedge_pnl - costs), close=close_label))
    return rows


def main():
    sample = pd.read_csv(DATA / "sample_dates.csv")
    sample = sample[sample.group == "regular"]
    all_trades, by_smile = [], []
    n_candidates = {m: 0 for m in MODELS}
    for _, s in sample.iterrows():
        t, t1 = pd.Timestamp(s.t), pd.Timestamp(s.t1)
        cal_df, intra_df, next_df = load(t, "1030"), load(t, "1430"), load(t1, "1030")
        if cal_df is None:
            continue
        for exp, sm in cal_df.groupby("expiration"):
            if not sm.calib_ok.iloc[0]:
                continue
            sm, out = model_values(sm)
            for model, (mask, V) in out.items():
                n_candidates[model] += int((mask & (sm.delta.abs().between(DELTA_MIN, DELTA_MAX))).sum())
            tr = trades_for_smile(sm, out, intra_df, "t 14:30")
            if int(sm.dte.iloc[0]) >= 1:
                tr += trades_for_smile(sm, out, next_df, "t+1 10:30")
            for r in tr:
                r.update(period_id=str(s.period_id), regime=s.regime, expiration=pd.Timestamp(exp).date())
            all_trades += tr
            df = pd.DataFrame(tr)
            if len(df):
                g = df.groupby(["model", "close"]).agg(n_trades=("pnl_net", "size"), n_buy=("side", lambda x: (x > 0).sum()),
                                                      pnl_net_mean=("pnl_net", "mean"), win_rate=("pnl_net", lambda x: (x > 0).mean())).reset_index()
                g["period_id"], g["expiration"], g["dte"], g["regime"] = str(s.period_id), pd.Timestamp(exp).date(), int(sm.dte.iloc[0]), s.regime
                by_smile.append(g)
    T = pd.DataFrame(all_trades)
    T["bucket"] = pd.Categorical(np.where(T.control, "30", np.where(T.dte <= 2, T.dte.astype(str), "3-7")), BUCKETS, ordered=True)
    pd.concat(by_smile).to_csv(RESULTS / "trading_by_smile.csv", index=False)

    def agg(g):
        """Summary of a group of trades, in dollars per contract."""
        side = g["side"] if "side" in g.columns else pd.Series(g.name[1] if isinstance(g.name, tuple) else 0, index=g.index)
        return pd.Series(dict(n_trades=len(g), share_buy=(side > 0).mean(), mispricing_mean=g.mispricing.abs().mean(),
                              gross_mean=g.pnl_gross.mean() * CONTRACT, hedged_mean=g.pnl_hedged.mean() * CONTRACT,
                              net_mean=g.pnl_net.mean() * CONTRACT, net_median=g.pnl_net.median() * CONTRACT,
                              win_rate=(g.pnl_net > 0).mean(), net_total=g.pnl_net.sum() * CONTRACT,
                              worst=g.pnl_net.min() * CONTRACT, best=g.pnl_net.max() * CONTRACT,
                              net_std=g.pnl_net.std() * CONTRACT))

    pd.set_option("display.width", 220)
    print("Candidate options (non-pillar, 0.05 <= |delta| <= 0.50) per model:", n_candidates)
    res = T.groupby(["close", "model", "bucket"], observed=True).apply(agg, include_groups=False).round(3)
    print("\nTABLE 10 - trading results per contract ($), by close, model and maturity (regular periods):")
    print(res.to_string())
    res.to_csv(RESULTS / "trading_results.csv")
    out = RESULTS / "tables"; out.mkdir(exist_ok=True)
    with open(out / "T10_trading.tex", "w") as f:
        f.write(res.to_latex(float_format="%.2f", caption="Trading exercise with transaction costs", label="tab:T10_trading", escape=True))

    print("\nBy side (buy vs sell), close t 14:30, maturities 0-7 days:")
    sub = T[(T.close == "t 14:30") & (~T.control)]
    bside = sub.assign(side_lab=np.where(sub.side > 0, "buy", "sell")).groupby(["model", "side_lab"]).apply(agg, include_groups=False)
    print(bside.round(2)[["n_trades", "mispricing_mean", "net_mean", "win_rate", "worst", "best"]].to_string())
    bside.round(3).to_csv(RESULTS / "trading_by_side.csv")
    print("\nBy regime, close t 14:30, maturities 0-7 days (net $ per contract):")
    breg = sub.groupby(["model", "regime"]).apply(agg, include_groups=False)
    print(breg.round(2)[["n_trades", "share_buy", "net_mean", "win_rate", "worst"]].to_string())
    breg.round(3).to_csv(RESULTS / "trading_by_regime.csv")
    print("\nShare of smiles with at least one trade (close t 14:30):")
    bs = pd.concat(by_smile); bs = bs[bs.close == "t 14:30"]
    print(bs.groupby("model").apply(lambda g: pd.Series(dict(smiles_with_trades=g.expiration.size, trades_per_smile=g.n_trades.mean())), include_groups=False).round(2).to_string())
    print("\nSaved results/trading_results.csv, results/trading_by_smile.csv, results/tables/T10_trading.tex")


if __name__ == "__main__":
    main()
