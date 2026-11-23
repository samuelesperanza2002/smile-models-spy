"""
Download of the raw data for another underlying (Chapter 7 of the thesis: USO and GLD),
for the sample days only.

Same files and same format as download_spy.py, saved in ~/thesis_spy/data_<symbol>/raw:
  quotes_<day>_1030.parquet, quotes_<day>_1430.parquet   last NBBO quote of every listed option
  oi_<day>.parquet                                       open interest, up to 35 days to expiry
  vol_<expiry>.parquet                                   volume traded before 10:30 and before 14:30
Days: t and t+1 of every period in data/sample_dates.csv. Files that already exist are skipped.

How to run (from the src folder, with the Theta Terminal running):
  python3 download_symbol.py USO
"""

import io, sys, time, pathlib
from datetime import timedelta
import requests
import pandas as pd

if len(sys.argv) < 2:
    sys.exit("usage: python3 download_symbol.py SYMBOL")
SYMBOL = sys.argv[1].upper()
BASE = "http://127.0.0.1:25503/v3"
HOME = pathlib.Path.home() / "thesis_spy"
RAW = HOME / f"data_{SYMBOL.lower()}" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
SNAPSHOTS = ["10:30", "14:30"]


def get(endpoint, **params):
    """Ask the Theta Terminal for data and return it as a table, or None if there is no data.
    Network errors, server errors and rate limits are retried up to 3 times."""
    params["format"] = "csv"
    for _ in range(3):
        try:
            r = requests.get(f"{BASE}/{endpoint}", params=params, timeout=600)
        except requests.RequestException as e:
            print("   network error:", e)
            time.sleep(10)
            continue
        if r.status_code == 200 and r.text.strip():
            return pd.read_csv(io.StringIO(r.text))
        print(f"   HTTP {r.status_code}: {r.text[:100]!r}")
        if r.status_code >= 500 or r.status_code == 429:
            time.sleep(10)
            continue
        return None
    return None


sample = pd.read_csv(HOME / "data" / "sample_dates.csv")
days = sorted(set(pd.to_datetime(sample.t).dt.date) | set(pd.to_datetime(sample.t1).dt.date))
print(f"{SYMBOL}: {len(days)} sample days, files in {RAW}")

expiries_needed = set()        # expiries with 0-7 or 25-40 days to expiry on some sample day
for day in days:
    ds = day.strftime("%Y%m%d")
    for hhmm in SNAPSHOTS:
        f = RAW / f"quotes_{ds}_{hhmm.replace(':', '')}.parquet"
        if f.exists():
            q = pd.read_parquet(f)
        else:
            q = get("option/at_time/quote", symbol=SYMBOL, expiration="*",
                    start_date=ds, end_date=ds, time_of_day=f"{hhmm}:00.000")
            if q is None:
                print(day, hhmm, "no data")
                continue
            q.to_parquet(f, index=False)
            print(day, hhmm, len(q), "rows")
        exp = pd.to_datetime(q["expiration"].astype(str)).dt.date
        dte = (pd.to_datetime(q["expiration"].astype(str)) - pd.Timestamp(day)).dt.days
        expiries_needed |= set(exp[(dte.between(0, 7)) | (dte.between(25, 40))])
    f_oi = RAW / f"oi_{ds}.parquet"
    if not f_oi.exists():
        oi = get("option/history/open_interest", symbol=SYMBOL, expiration="*", date=ds, max_dte=35)
        if oi is not None:
            oi.to_parquet(f_oi, index=False)
            print(day, "open interest", len(oi), "rows")

print(f"\nVolume files for {len(expiries_needed)} expiries")
for exp in sorted(expiries_needed):
    f = RAW / f"vol_{exp:%Y%m%d}.parquet"
    if f.exists():
        continue
    v = get("option/history/ohlc", symbol=SYMBOL, expiration=f"{exp:%Y%m%d}",
            start_date=f"{exp - timedelta(days=8):%Y%m%d}", end_date=f"{exp:%Y%m%d}",
            interval="30m", start_time="09:30:00", end_time="14:30:00")
    if v is None:
        print("no volume for", exp)
        continue
    v["ts"] = pd.to_datetime(v["timestamp"])
    v["day"] = v.ts.dt.date
    v["hhmm"] = v.ts.dt.strftime("%H:%M")
    keys = ["day", "expiration", "strike", "right"]
    out = pd.concat([
        v[v.hhmm < "10:30"].groupby(keys).volume.sum().rename("vol_1030"),
        v[v.hhmm < "14:30"].groupby(keys).volume.sum().rename("vol_1430"),
    ], axis=1).fillna(0).reset_index()
    out.to_parquet(f, index=False)
    print("volume", exp, len(out), "rows")

print("DONE")
