"""
Download of the raw SPY option data from the Theta Terminal (ThetaData API v3).

The Theta Terminal is a program from ThetaData that runs on the computer and answers requests at
http://127.0.0.1:25503. It needs a paid ThetaData subscription (here: Options Value).

The script downloads, for every business day from START to END:
  1. the last NBBO quote (best bid and ask of all US exchanges) of all SPY options at 10:30 and
     14:30 New York time, for all expiries, plus the expiry closest to 30 days;
  2. the open interest of the same day (up to 35 days to expiry);
and then, for every expiry,
  3. the traded volume from 30-minute bars, summed before 10:30 and before 14:30.
Files that already exist are skipped, so the script can be stopped and started again.
The files are saved in data/raw/. They are licensed data: they are not part of this repository.

How to run (from the src folder, with the Theta Terminal running):
  caffeinate -i python3 download_spy.py [END_DATE]
(caffeinate keeps the Mac awake during the download.)
"""

import io, sys, time, pathlib
from datetime import timedelta
import requests
import pandas as pd

BASE = "http://127.0.0.1:25503/v3"
RAW = pathlib.Path.home() / "thesis_spy" / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

START = "2022-12-01"
END = sys.argv[1] if len(sys.argv) > 1 else "2026-10-23"
SNAPSHOTS = ["10:30", "14:30"]


def get(endpoint, **params):
    """Ask the Theta Terminal for data and return it as a table, or None if there is no data.
    Network errors, server errors and rate limits are retried up to 3 times."""
    params["format"] = "csv"
    for _ in range(3):
        try:
            r = requests.get(f"{BASE}/{endpoint}", params=params, timeout=300)
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


def expiry_30(day):
    """Friday of the week that contains day + 30 days (for a weekend, the Friday before):
    the control expiry at about 30 days."""
    target = day + timedelta(days=30)
    shift = 4 - target.weekday()
    if shift > 3:
        shift -= 7
    return target + timedelta(days=shift)


print(f"Download from {START} to {END}")

# ---------- 1) QUOTES 10:30 and 14:30 + OPEN INTEREST ----------
for d in pd.bdate_range(START, END):
    day = d.date()
    ds = day.strftime("%Y%m%d")

    for hhmm in SNAPSHOTS:
        f = RAW / f"quotes_{ds}_{hhmm.replace(':', '')}.parquet"
        if f.exists():
            continue
        q = get("option/at_time/quote", symbol="SPY", expiration="*",
                start_date=ds, end_date=ds,
                time_of_day=f"{hhmm}:00.000", max_dte=7)
        if q is None:                       # no data: the market was closed that day
            print(day, hhmm, "no data (holiday?)")
            break
        parts = [q]
        e30 = expiry_30(day)
        for e in (e30, e30 - timedelta(days=1)):          # if that Friday is a holiday, try the Thursday
            q30 = get("option/at_time/quote", symbol="SPY",
                      expiration=e.strftime("%Y%m%d"),
                      start_date=ds, end_date=ds,
                      time_of_day=f"{hhmm}:00.000")
            if q30 is not None:
                parts.append(q30)
                break
        out = pd.concat(parts, ignore_index=True)
        out.to_parquet(f, index=False)
        print(day, hhmm, len(out), "rows")

    f_oi = RAW / f"oi_{ds}.parquet"
    if not f_oi.exists():
        oi = get("option/history/open_interest", symbol="SPY",
                 expiration="*", date=ds, max_dte=35)
        if oi is not None:
            oi.to_parquet(f_oi, index=False)

# ---------- 2) VOLUME (one request per expiry) ----------
exps = get("option/list/expirations", symbol="SPY")
exps = sorted(pd.to_datetime(exps["expiration"].astype(str)).dt.date.unique())
start_d, end_d = pd.Timestamp(START).date(), pd.Timestamp(END).date()
exps = [e for e in exps if start_d <= e <= end_d]

for exp in exps:
    f = RAW / f"vol_{exp:%Y%m%d}.parquet"
    if f.exists():
        continue
    v = get("option/history/ohlc", symbol="SPY", expiration=f"{exp:%Y%m%d}",
            start_date=f"{exp - timedelta(days=8):%Y%m%d}", end_date=f"{exp:%Y%m%d}",
            interval="30m", start_time="09:30:00", end_time="14:30:00")
    if v is None:
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
    print("volume", exp, len(out), "rows | first bar:", v.hhmm.min())

print("DONE")
