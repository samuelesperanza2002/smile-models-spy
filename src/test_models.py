"""
Quick test of the models on one cleaned file. For every smile that can be calibrated it prints
the in-sample errors of Black-Scholes, SABR and Vanna-Volga (Design A with the pillar sets
P1, P2 and P3; Design B for SABR). It was used while writing the model code.

How to run (from the src folder):
  python3 test_models.py 20240508 1030
  python3 test_models.py 20240508 1030 --beta 0.5
"""

import sys, argparse, pathlib
import numpy as np
import pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import evaluate

HOME = pathlib.Path.home() / "thesis_spy"
CLEAN = HOME / "data" / "clean"
pd.set_option("display.width", 220)

ap = argparse.ArgumentParser()
ap.add_argument("day"); ap.add_argument("snap")
ap.add_argument("--beta", type=float, default=1.0)
a = ap.parse_args()

df = pd.read_parquet(CLEAN / f"clean_{a.day}_{a.snap}.parquet")
rows = []
for exp, sm in df.groupby("expiration"):
    if not sm.calib_ok.iloc[0]:
        print(f"expiry {pd.Timestamp(exp).date()} (dte {int(sm.dte.iloc[0])}): only {sm.strike.nunique()} strikes, not calibratable")
        continue
    r, fitted = evaluate.eval_smile(sm, beta=a.beta)
    rows += r
res = pd.DataFrame(rows)

print(f"\n=== {a.day} {a.snap[:2]}:{a.snap[2:]} | beta = {a.beta} | IV-RMSE in volatility points, price errors in $ ===")
print("\nDESIGN A, pillar set P1 (25d put / ATM / 25d call), test on the non-pillar options:")
A1 = res[(res.design == "A") & (res.pillar_set == "P1") & (res.region == "all")]
print(A1.pivot_table(index="dte", columns="model", values=["iv_rmse", "price_rmse", "inside_spread", "n_test"],
                     aggfunc="first").round(3).to_string())

print("\nSABR (Design A, P1): parameters and convergence")
S = A1[A1.model == "SABR"][["dte", "n_options", "alpha", "rho", "nu", "converged", "at_bound", "calib_rmse", "arb_viol"]]
print(S.round(4).to_string(index=False))

print("\nVV: IV-RMSE and failure rate by pillar set, inside and outside [K1, K3]")
V = res[(res.design == "A") & (res.model == "VV")]
print(V.pivot_table(index="dte", columns=["pillar_set", "region"], values="iv_rmse", aggfunc="first").round(2).to_string())
print(V.pivot_table(index="dte", columns=["pillar_set"], values=["fail_rate", "n_test"], aggfunc="first").round(3).to_string())

print("\nSABR with the same pillars as P2 and P3 (Design A) - IV-RMSE on the non-pillar options:")
S2 = res[(res.design == "A") & (res.model == "SABR") & (res.region == "all")]
print(S2.pivot_table(index="dte", columns="pillar_set", values="iv_rmse", aggfunc="first").round(2).to_string())

print("\nDESIGN B: SABR calibrated on the odd strikes, test on the even ones (BS on the same test strikes)")
B = res[res.design == "B"]
print(B.pivot_table(index="dte", columns="model", values=["iv_rmse", "price_rmse", "inside_spread", "n_test"],
                    aggfunc="first").round(3).to_string())
print(B[B.model == "SABR"][["dte", "n_calib", "alpha", "rho", "nu", "converged", "arb_viol"]].round(4).to_string(index=False))
print("\nEND")
