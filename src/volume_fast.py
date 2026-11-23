"""
Faster download of the traded volume (step 3 of download_spy.py), with two requests at a time.

For every SPY expiry it asks for 30-minute bars and sums the volume traded before 10:30 and
before 14:30, for each day, strike and option type. Each file is first written with the ending
.tmp and renamed at the end, so an interrupted download never leaves a broken file.
Files that already exist are skipped.

How to run (from the src folder, with the Theta Terminal running):
  caffeinate -i python3 volume_fast.py
"""

import io, time, pathlib
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor
import requests
import pandas as pd

BASE = "http://127.0.0.1:25503/v3"
RAW = pathlib.Path.home() / "thesis_spy" / "data" / "raw"
START, END = "2022-12-01", "2026-10-23"


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


def download_expiry(exp):
    """Download and save the volume file of one expiry."""
    f = RAW / f"vol_{exp:%Y%m%d}.parquet"
    if f.exists():
        return
    v = get("option/history/ohlc", symbol="SPY", expiration=f"{exp:%Y%m%d}",
            start_date=f"{exp - timedelta(days=8):%Y%m%d}", end_date=f"{exp:%Y%m%d}",
            interval="30m", start_time="09:30:00", end_time="14:30:00")
    if v is None:
        print("no data for", exp)
        return
    v["ts"] = pd.to_datetime(v["timestamp"])
    v["day"] = v.ts.dt.date
    v["hhmm"] = v.ts.dt.strftime("%H:%M")
    keys = ["day", "expiration", "strike", "right"]
    out = pd.concat([
        v[v.hhmm < "10:30"].groupby(keys).volume.sum().rename("vol_1030"),
        v[v.hhmm < "14:30"].groupby(keys).volume.sum().rename("vol_1430"),
    ], axis=1).fillna(0).reset_index()
    tmp = RAW / f"vol_{exp:%Y%m%d}.tmp"
    out.to_parquet(tmp, index=False)
    tmp.rename(f)
    print("volume", exp, len(out), "rows | first bar:", v.hhmm.min())


exps = get("option/list/expirations", symbol="SPY")
exps = sorted(pd.to_datetime(exps["expiration"].astype(str)).dt.date.unique())
start_d, end_d = pd.Timestamp(START).date(), pd.Timestamp(END).date()
todo = [e for e in exps if start_d <= e <= end_d
        and not (RAW / f"vol_{e:%Y%m%d}.parquet").exists()]
print("expiries to download:", len(todo))

with ThreadPoolExecutor(max_workers=2) as ex:
    list(ex.map(download_expiry, todo))

print("DONE")
