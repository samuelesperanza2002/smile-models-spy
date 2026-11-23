"""
Extra download: the last NBBO quote at 11:30 New York time, for the robustness check with
calibration at 11:30 (Chapter 5 of the thesis).

It uses every trading day already in data/raw (every day with a file quotes_YYYYMMDD_1030.parquet)
and follows the same logic as download_spy.py, with one snapshot only. Days already downloaded
are skipped, so the script can be stopped (Ctrl+C) and started again.

How to run (from the src folder, with the Theta Terminal running):
  caffeinate -i python3 download_1130.py
"""

import io, time, pathlib
from datetime import timedelta
import requests
import pandas as pd

BASE = "http://127.0.0.1:25503/v3"
RAW = pathlib.Path.home() / "thesis_spy" / "data" / "raw"
HHMM = "11:30"


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


# trading days = the days that already have a 10:30 file
days = sorted(f.name[7:15] for f in RAW.glob("quotes_*_1030.parquet"))
todo = [d for d in days if not (RAW / f"quotes_{d}_1130.parquet").exists()]
print(f"Trading days: {len(days)} | already downloaded at 11:30: {len(days) - len(todo)} "
      f"| to download: {len(todo)}")

t0 = time.time()
done = 0
for ds in todo:
    day = pd.Timestamp(ds).date()
    f = RAW / f"quotes_{ds}_1130.parquet"
    q = get("option/at_time/quote", symbol="SPY", expiration="*",
            start_date=ds, end_date=ds, time_of_day=f"{HHMM}:00.000")
    if q is None:
        print(day, HHMM, "NO DATA: to be checked")
        continue
    parts = [q]
    e30 = expiry_30(day)
    for e in (e30, e30 - timedelta(days=1)):
        q30 = get("option/at_time/quote", symbol="SPY",
                  expiration=e.strftime("%Y%m%d"),
                  start_date=ds, end_date=ds, time_of_day=f"{HHMM}:00.000")
        if q30 is not None:
            parts.append(q30)
            break
    out = pd.concat(parts, ignore_index=True).drop_duplicates()
    tmp = f.with_suffix(".tmp")
    out.to_parquet(tmp, index=False)
    tmp.rename(f)
    done += 1
    if done % 20 == 0 or done == len(todo):
        mins = (time.time() - t0) / 60
        rate = mins / done
        print(f"{day} {HHMM} {len(out)} rows | done {done}/{len(todo)} | "
              f"{mins:.0f} min elapsed, about {rate * (len(todo) - done):.0f} min left")

n = len(list(RAW.glob("quotes_*_1130.parquet")))
print(f"DONE: quotes_*_1130.parquet files present = {n} (expected {len(days)})")
