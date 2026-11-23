"""
One-day probe of a new underlying, used to choose the ETFs of Chapter 7.

It prints the expiries available on a given day, the step between strikes and how many options
would pass filters 1-2 of clean.py (bid > 0, mid >= 0.05). Nothing is saved.

How to run (from the src folder, with the Theta Terminal running):
  python3 probe_symbol.py USO 20261021
"""

import io, sys, time
import requests
import pandas as pd

BASE = "http://127.0.0.1:25503/v3"
symbol, ds = sys.argv[1], sys.argv[2]


def get(endpoint, **params):
    """Ask the Theta Terminal for data and return it as a table, or None if there is no data."""
    params["format"] = "csv"
    r = requests.get(f"{BASE}/{endpoint}", params=params, timeout=300)
    if r.status_code == 200 and r.text.strip():
        return pd.read_csv(io.StringIO(r.text))
    print(f"   HTTP {r.status_code}: {r.text[:120]!r}")
    return None


exps = get("option/list/expirations", symbol=symbol)
if exps is None:
    sys.exit(f"no expirations for {symbol}")
e = pd.to_datetime(exps["expiration"].astype(str))
e26 = e[e.dt.year == 2026]
print(f"{symbol}: {len(e)} expirations in total; in 2026 by weekday:",
      e26.dt.day_name().value_counts().to_dict())
first_wed = e[(e.dt.weekday == 2) & (e.dt.year >= 2023)].min()
first_mon = e[(e.dt.weekday == 0) & (e.dt.year >= 2023)].min()
print(f"first Wednesday expiry: {first_wed.date() if pd.notna(first_wed) else 'none'} | "
      f"first Monday expiry: {first_mon.date() if pd.notna(first_mon) else 'none'}")

for hhmm in ("10:30", "14:30"):
    q = get("option/at_time/quote", symbol=symbol, expiration="*", start_date=ds, end_date=ds,
            time_of_day=f"{hhmm}:00.000", max_dte=7)
    if q is None:
        print(ds, hhmm, "no data")
        continue
    q["expiration"] = pd.to_datetime(q["expiration"].astype(str))
    q["dte"] = (q.expiration - pd.Timestamp(ds)).dt.days
    q["mid"] = (q.bid + q.ask) / 2
    print(f"\n{ds} {hhmm}: {len(q)} quotes, columns: {list(q.columns)[:8]} ...")
    for exp, g in q.groupby("expiration"):
        ks = sorted(g.strike.unique())
        step = pd.Series(ks).diff().median() if len(ks) > 1 else float("nan")
        ok = g[(g.bid > 0) & (g.ask > g.bid)]
        ok5 = ok[ok.mid >= 0.05]
        print(f"  expiry {exp.date()} (dte {int(g.dte.iloc[0])}): {len(g)} quotes, strikes {ks[0]:g}-{ks[-1]:g} "
              f"step {step:g} | bid>0: {len(ok)} | mid>=0.05: {len(ok5)} | distinct strikes with mid>=0.05: {ok5.strike.nunique()}")
