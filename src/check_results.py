"""
Licence check of the results folder. Under the data licence, results/ may contain only
aggregated numbers (statistics of each smile, calibrated parameters, tables, figures), never one
row per option quote.

The script reads every CSV file in results/ and looks for columns that belong to a single option
(strike, bid, ask, mid, iv, volume, open interest, delta, vega, quote age, put/call flag):
  ERROR = per-option columns found: the file must not be kept
  WARN  = columns that point to single options indirectly (the Vanna-Volga pillar strikes and
          the reference volatility of each smile). Before the data were deleted, these columns
          were removed from the results of the thesis.
It also lists the folders with raw or cleaned quotes, which had to be deleted at the end of the
licence. The script only reads: nothing is changed.

How to run (from the src folder):  python3 check_results.py
"""

import os
import pathlib
import pandas as pd

HOME = pathlib.Path.home() / "thesis_spy"
RESULTS, DATA = HOME / "results", HOME / "data"
# columns that belong to a single option
FORBIDDEN = {"strike", "bid", "ask", "mid", "iv", "volume", "oi", "delta", "vega", "age_sec", "right", "spread_rel", "price"}
INDIRECT = {"K1", "K2", "K3", "sigma_ref"}           # columns that point to single options indirectly
TO_DELETE = ["raw", "clean", "clean_aft"]           # sub-folders of data/ with per-option data
KEEP_IN_DATA = ["sample_dates.csv", "sample_candidates.csv", "event_candidates.csv", "fred_VIXCLS.csv", "fred_DTB4WK.csv"]


def folder_size(p):
    """Number of files and size in MB of a folder."""
    n, b = 0, 0
    for root, _, files in os.walk(p):
        for f in files:
            n += 1; b += os.path.getsize(os.path.join(root, f))
    return n, b / 1e6


rows, n_err, n_warn = [], 0, 0
for f in sorted(RESULTS.rglob("*")):
    if f.is_dir():
        continue
    rel = f.relative_to(RESULTS).as_posix()
    if f.suffix == ".parquet":
        rows.append((rel, "parquet", "", "", "ERROR: parquet file (raw-style data)")); n_err += 1
    elif f.suffix == ".csv":
        try:
            df = pd.read_csv(f, low_memory=False)
        except Exception as e:
            rows.append((rel, "csv", 0, 0, f"empty or unreadable ({type(e).__name__})")); continue
        cols = {str(c) for c in df.columns}
        bad, ind = sorted(cols & FORBIDDEN), sorted(cols & INDIRECT)
        if bad:
            v = "ERROR: per-option columns " + ", ".join(bad); n_err += 1
        elif ind:
            v = "WARN: pillar columns " + ", ".join(ind); n_warn += 1
        else:
            v = "ok"
        rows.append((rel, "csv", len(df), len(df.columns), v))
    else:
        rows.append((rel, f.suffix.lstrip("."), "", "", "ok (not a data file)"))

pd.set_option("display.width", 220); pd.set_option("display.max_rows", 500)
rep = pd.DataFrame(rows, columns=["file", "type", "rows", "columns", "verdict"])
print(rep.to_string(index=False))
print(f"\nFiles in results/: {len(rep)} | ERROR: {n_err} | WARN: {n_warn}")
if n_err:
    print("ACTION NEEDED: the files marked ERROR contain per-option data and must be removed or aggregated.")
else:
    print("RESULTS FOLDER OK: no file with per-option rows.")
if n_warn:
    print("WARN: model_results*.csv keep the three pillar strikes (K1-K3) and the reference volatility of each smile,\n"
          "      i.e. the Vanna-Volga inputs. They are calibration inputs, not quotes, but they can be dropped at the freeze.")

print("\nFolders with per-option data (to delete at the end of the licence):")
for d in TO_DELETE:
    p = DATA / d
    if p.exists():
        n, mb = folder_size(p)
        print(f"  data/{d:<10} {n:>6} files  {mb:>9.1f} MB")
    else:
        print(f"  data/{d:<10} (not present)")
loose = sorted(x.name for x in DATA.glob("*.parquet"))
print(f"Loose parquet files directly in data/ (to delete as well): {len(loose)}" + (f", e.g. {loose[0]} ... {loose[-1]}" if loose else ""))
print("Files in data/ that can be kept (no quotes):", ", ".join(k for k in KEEP_IN_DATA if (DATA / k).exists()))
