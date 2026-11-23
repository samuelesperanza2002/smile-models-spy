"""
Tables of Chapter 5 of the thesis, built from results/model_results<tag>.csv.

  T0  number of smiles by stage and maturity
  T3  in-sample IV-RMSE by model and maturity (Designs A and B), plus price errors
  T4  in-sample IV-RMSE by moneyness group
  T5  out-of-sample errors, intraday and next day
  T6  Wilcoxon signed-rank tests on the errors of each smile
  T7  Vanna-Volga with the three pillar sets, failure rates
  T8  SABR parameters by maturity
  T9  errors by VIX regime; TE errors on the two FOMC days
Each table is printed and saved in results/tables<tag>/ as CSV and as LaTeX.

How to run (from the src folder):
  python3 make_tables.py                 main results
  python3 make_tables.py --tag _beta0.5  results of a robustness run
"""

import argparse, pathlib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HOME = pathlib.Path.home() / "thesis_spy"
RESULTS = HOME / "results"
pd.set_option("display.width", 220)
BUCKETS = ["0", "1", "2", "3-7", "30"]


def bucket(df):
    """Maturity group of each row: 0, 1, 2, 3-7 days, or 30 for the control expiry."""
    b = np.where(df.control, "30", np.where(df.dte <= 2, df.dte.astype(str), "3-7"))
    return pd.Categorical(b, BUCKETS, ordered=True)


def save(df, name, out, caption):
    """Save one table as CSV and as LaTeX."""
    df.to_csv(out / f"{name}.csv")
    with open(out / f"{name}.tex", "w") as f:
        f.write(df.to_latex(float_format="%.2f", caption=caption, label=f"tab:{name}", escape=True))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default=""); a = ap.parse_args()
    res = pd.read_csv(RESULTS / f"model_results{a.tag}.csv", dtype={"period_id": str}, low_memory=False)
    res["bucket"] = bucket(res)
    res["model_key"] = res.model + np.where(res.design == "B", "(B)", "") + \
        np.where((res.model.isin(["SABR", "VV"])) & (res.design == "A"), ":" + res.pillar_set, "")
    out = RESULTS / f"tables{a.tag}"; out.mkdir(exist_ok=True)
    regular = res[res.group == "regular"]
    print(f"=== model_results{a.tag}.csv: {len(res)} rows | periods: {res.period_id.nunique()} "
          f"(regular {regular.period_id.nunique()}) ===\n")

    # ------------------------------------------------ smile counts
    cnt = regular[(regular.region == "all") & (regular.model == "BS") & (regular.design == "A") & (regular.pillar_set == "P1") |
                  (regular.region == "all") & (regular.model == "BS") & (regular.stage != "insample")]
    cnt = cnt.drop_duplicates(["period_id", "expiration", "stage"])
    T0 = cnt.pivot_table(index="stage", columns="bucket", values="expiration", aggfunc="size", observed=False).fillna(0).astype(int)
    print("SMILES per stage and maturity (regular periods):"); print(T0.to_string()); save(T0, "T0_smile_counts", out, "Number of smiles by stage and maturity")

    # ------------------------------------------------ T3: in-sample IV-RMSE by model x maturity
    ins = regular[(regular.stage == "insample") & (regular.region == "all")]
    A = ins[(ins.design == "A") & (ins.pillar_set == "P1")]
    B = ins[ins.design == "B"]
    t3 = pd.concat([A, B])
    T3 = t3.pivot_table(index="model_key", columns="bucket", values="iv_rmse", aggfunc=["mean", "median"], observed=False).round(2)
    print("\nTABLE 3 - in-sample IV-RMSE (vol points), Design A with P1 pillars and Design B (odd/even):")
    print(T3.to_string()); save(T3, "T3_ivrmse_maturity", out, "In-sample IV-RMSE by model and maturity")
    T3b = t3.pivot_table(index="model_key", columns="bucket", values=["price_rmse", "inside_spread"], aggfunc="mean", observed=False).round(3)
    print("\nTABLE 3b - price RMSE ($) and share inside the bid-ask (means):"); print(T3b.to_string())
    save(T3b, "T3b_price_inside", out, "In-sample price RMSE and share of model prices inside the bid-ask spread")

    # ------------------------------------------------ T4: by moneyness group (0-7 days, Design A P1 and B)
    short = regular[(regular.stage == "insample") & (~regular.control) & (regular.region.isin(["put_deep", "put_otm", "near_atm", "call_otm", "call_deep"]))]
    t4 = pd.concat([short[(short.design == "A") & (short.pillar_set == "P1")], short[short.design == "B"]])
    T4 = t4.pivot_table(index="model_key", columns="region", values="iv_rmse", aggfunc="mean")[["put_deep", "put_otm", "near_atm", "call_otm", "call_deep"]].round(2)
    print("\nTABLE 4 - in-sample IV-RMSE by moneyness group (maturities 0-7 days):"); print(T4.to_string())
    save(T4, "T4_ivrmse_moneyness", out, "In-sample IV-RMSE by model and moneyness group")

    # ------------------------------------------------ T5: out-of-sample
    for stage in ("intraday", "nextday"):
        for region in ("all", "common"):
            o = regular[(regular.stage == stage) & (regular.region == region)]
            o = o[(o.pillar_set.isin(["P1", "-", "odd/even"]))]
            if not len(o):
                continue
            T5 = o.pivot_table(index="model_key", columns="bucket", values="iv_rmse", aggfunc=["mean", "median"], observed=False).round(2)
            lab = "all test options" if region == "all" else "strikes quoted at both snapshots (fair comparison with sticky strike)"
            print(f"\nTABLE 5 ({stage}, {lab}) - out-of-sample IV-RMSE (vol points; maturity = at calibration):"); print(T5.to_string())
            save(T5, f"T5_oos_{stage}_{region}", out, f"Out-of-sample IV-RMSE ({stage}, {region})")
            T5b = o.pivot_table(index="model_key", columns="bucket", values="inside_spread", aggfunc="mean", observed=False).round(3)
            save(T5b, f"T5b_oos_{stage}_{region}_inside", out, f"Out-of-sample share inside the bid-ask ({stage}, {region})")

    # ------------------------------------------------ T6: Wilcoxon signed-rank on per-smile IV-RMSE
    def pairs_table(df, label):
        """Wilcoxon tests for pairs of models on the same smiles (negative difference = first model better)."""
        rows = []
        key = ["period_id", "expiration"]
        piv = df.pivot_table(index=key + ["bucket"], columns="model_key", values="iv_rmse", observed=False).reset_index()
        for m1, m2 in [("SABR:P1", "BS"), ("VV:P1", "BS"), ("SABR:P1", "VV:P1"), ("SABR(B)", "BS")]:
            if m1 not in piv.columns or m2 not in piv.columns:
                continue
            for b in BUCKETS + ["all"]:
                sub = piv if b == "all" else piv[piv.bucket == b]
                d = (sub[m1] - sub[m2]).dropna()
                if len(d) < 5:
                    continue
                try:
                    p = wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue
                except ValueError:
                    p = np.nan
                rows.append(dict(stage=label, pair=f"{m1} vs {m2}", maturity=b, n=len(d), median_diff=d.median(),
                                 share_wins=(d < 0).mean(), p_value=p))
        return pd.DataFrame(rows)

    T6 = pd.concat([pairs_table(ins[(ins.design == "A") & (ins.pillar_set == "P1") | (ins.design == "B")], "insample"),
                    pairs_table(regular[(regular.stage == "intraday") & (regular.region == "all")], "intraday"),
                    pairs_table(regular[(regular.stage == "nextday") & (regular.region == "all")], "nextday")])
    T6 = T6.set_index(["stage", "pair", "maturity"])
    print("\nTABLE 6 - Wilcoxon signed-rank tests on per-smile IV-RMSE (negative median_diff = first model better):")
    print(T6.round(4).to_string()); save(T6.round(4), "T6_wilcoxon", out, "Wilcoxon signed-rank tests on per-smile IV-RMSE")

    # ------------------------------------------------ T7: VV pillar sets, inside/outside, failures
    v = regular[(regular.stage == "insample") & (regular.model == "VV") & (regular.region.isin(["all", "inside", "outside"]))]
    T7 = v.pivot_table(index=["pillar_set", "region"], columns="bucket", values="iv_rmse", aggfunc="mean", observed=False).round(2)
    T7f = v[v.region == "all"].pivot_table(index="pillar_set", columns="bucket", values="fail_rate", aggfunc="mean", observed=False).round(3)
    s7 = regular[(regular.stage == "insample") & (regular.model == "SABR") & (regular.design == "A") & (regular.region == "all")]
    T7s = s7.pivot_table(index="pillar_set", columns="bucket", values="iv_rmse", aggfunc="mean", observed=False).round(2)
    print("\nTABLE 7 - VV in-sample IV-RMSE by pillar set, inside/outside [K1,K3]:"); print(T7.to_string())
    print("VV failure rate (share of test options with no implied vol):"); print(T7f.to_string())
    print("SABR calibrated on the same pillars (Design A):"); print(T7s.to_string())
    save(T7, "T7_vv_pillars", out, "Vanna-Volga IV-RMSE by pillar set, inside and outside the pillar range")
    save(T7f, "T7b_vv_failures", out, "Vanna-Volga failure rates"); save(T7s, "T7c_sabr_pillars", out, "SABR IV-RMSE by pillar set")

    # ------------------------------------------------ T8: SABR parameters by maturity
    sB = regular[(regular.stage == "insample") & (regular.model == "SABR") & (regular.design == "B") & (regular.region == "all")].copy()
    sB["nu_sqrt_tau"] = sB.nu * np.sqrt(sB.tau)
    T8 = sB.groupby("bucket", observed=False).agg(n=("nu", "size"), alpha=("alpha", "median"), rho=("rho", "median"),
                                                  nu=("nu", "median"), nu_sqrt_tau=("nu_sqrt_tau", "median"),
                                                  nu_p10=("nu", lambda x: x.quantile(.1)), nu_p90=("nu", lambda x: x.quantile(.9)),
                                                  converged=("converged", "mean"), arb_viol=("arb_viol", lambda x: (x > 0).mean())).round(3)
    print("\nTABLE 8 - SABR parameters by maturity (Design B, medians):"); print(T8.to_string())
    save(T8, "T8_sabr_params", out, "SABR parameters by maturity")
    sB[["period_id", "regime", "day", "expiration", "dte", "bucket", "tau", "alpha", "rho", "nu", "nu_sqrt_tau", "converged"]] \
        .to_csv(out / "fig3_nu_vs_tau.csv", index=False)

    # ------------------------------------------------ T9: regime split
    r9 = regular[(regular.region == "all") & (regular.pillar_set.isin(["P1", "-", "odd/even"]))]
    T9 = r9.pivot_table(index=["stage", "model_key"], columns="regime", values=["iv_rmse", "iv_rmse_rel"], aggfunc="mean").round(2)
    print("\nREGIME SPLIT - mean IV-RMSE (vol points) and relative IV-RMSE (%) by stage, model and VIX regime:"); print(T9.to_string())
    save(T9, "T9_regime", out, "IV-RMSE by VIX regime")

    # ------------------------------------------------ event days
    ev = res[(res.group == "event") & (res.region == "all") & (res.pillar_set.isin(["P1", "-", "odd/even"]))]
    if len(ev):
        TE = ev.pivot_table(index=["stage", "model_key"], columns="bucket", values="iv_rmse", aggfunc="mean", observed=False).round(2)
        print("\nEVENT DAYS (FOMC) - mean IV-RMSE:"); print(TE.to_string()); save(TE, "TE_event_days", out, "FOMC event days")
    print(f"\nTables saved in {out}")


if __name__ == "__main__":
    main()
