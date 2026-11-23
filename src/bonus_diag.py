"""
Diagnostic for Chapter 7: filter funnel of the USO and GLD smiles for the control expiry
(about 30 days) and for the expiries with 0 and 1 days, at 10:30 on day t. It explains why some
of these smiles could not be calibrated. It reads aggregated files only.

How to run (from the src folder):  python3 bonus_diag.py
"""

import pathlib
import pandas as pd

HOME = pathlib.Path.home() / "thesis_spy"
sample = pd.read_csv(HOME / "data" / "sample_dates.csv")
tdays = set(pd.to_datetime(sample[sample.group == "regular"].t).dt.date.astype(str))
pd.set_option("display.width", 250)
cols = ["day", "expiration", "dte", "n0_quote", "f1_bid_ask", "f2_mid", "f3_spread", "f4_liquidity",
        "f5_otm", "f6_moneyness", "f7_noarb", "f8_iv", "f9_final", "note"]
for sym in ["uso", "gld"]:
    fc = pd.read_csv(HOME / f"data_{sym}" / "results" / "filter_counts.csv")
    fc = fc[(fc.snap.astype(str) == "1030") & (fc.day.astype(str).isin(tdays))]
    print(f"\n=== {sym.upper()}: control expiry (~30 days) at 10:30 on day t ===")
    print(fc[fc.control][cols].to_string(index=False))
    print(f"\n=== {sym.upper()}: 0 and 1 DTE at 10:30 on day t ===")
    print(fc[(~fc.control) & (fc.dte <= 1)][cols].to_string(index=False))
    missing = sorted(tdays - set(fc[fc.control].day.astype(str)))
    print(f"\n{sym.upper()}: days t without any expiry between 25 and 40 days: {missing}")
