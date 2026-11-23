"""
Chapter 7 of the thesis (other underlyings): clean the raw files of a second underlying and run
the three models on the sample days.

It reuses the functions of clean.py and run_models.py without changing them. Only the folders
are changed: everything is read from and written to ~/thesis_spy/data_<symbol>/ (raw, clean,
results), so the SPY files are not touched. Same filters as for SPY; main specification only.

How to run (from the src folder, after download_symbol.py USO):
  python3 bonus.py USO
"""

import sys, pathlib, datetime as dt
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import clean, run_models

if len(sys.argv) < 2:
    sys.exit("usage: python3 bonus.py SYMBOL")
SYMBOL = sys.argv[1].upper()
HOME = pathlib.Path.home() / "thesis_spy"
ROOT = HOME / f"data_{SYMBOL.lower()}"
if not (ROOT / "raw").exists():
    sys.exit(f"{ROOT / 'raw'} not found: run download_symbol.py {SYMBOL} first")

# ---------------------------------------------------------------- 1) cleaning, same filters as SPY
# point the folders of clean.py to data_<symbol>/
clean.RAW = ROOT / "raw"
clean.CLEAN = ROOT / "clean"
clean.RESULTS = ROOT / "results"
for d in (clean.CLEAN, clean.CLEAN / "plots", clean.CLEAN / "counts", clean.RESULTS, clean.RESULTS / "tables"):
    d.mkdir(parents=True, exist_ok=True)

sample = pd.read_csv(HOME / "data" / "sample_dates.csv")
pid = {}
for _, row in sample.iterrows():
    pid[pd.Timestamp(row.t)] = str(row.period_id)
    pid[pd.Timestamp(row.t1)] = str(row.period_id)
days = sorted(pid)

print(f"=== {SYMBOL}: cleaning {len(days)} days x 2 snapshots (filters 1-9 of Chapter 4) ===")
t0 = dt.datetime.now()
log, all_counts, all_summary = [], [], []
for day in days:
    for snap in ("1030", "1430"):
        cl, counts, summary, status = clean.process_snapshot(day, snap, 0.50, verbose=False, plot=False,
                                                              period_id=pid[day])
        log.append(dict(day=day.date(), snap=snap, status=status,
                        n_smiles=0 if summary is None else len(summary),
                        n_options=0 if cl is None else len(cl)))
        if counts is not None:
            all_counts.append(counts)
        if summary is not None:
            all_summary.append(summary)
        if status != "ok":
            print(f"{day.date()} {snap}: {status}")
log = pd.DataFrame(log)
pd.concat(all_counts).to_csv(clean.RESULTS / "filter_counts.csv", index=False)
pd.concat(all_summary).to_csv(clean.RESULTS / "smile_summary.csv", index=False)
log.to_csv(clean.RESULTS / "clean_log.csv", index=False)
print(log.groupby(["snap", "status"]).size().to_string())
print(f"Cleaned smiles: {log.n_smiles.sum()} | options: {log.n_options.sum()} | "
      f"{(dt.datetime.now() - t0).total_seconds():.0f} s")

# ---------------------------------------------------------------- 2) models, main specification only
# point the folders of run_models.py to data_<symbol>/ and run it with a tag in the file name
print(f"\n=== {SYMBOL}: models on the sample days (Design A with P1, P2, P3 and Design B) ===")
run_models.CLEAN = ROOT / "clean"
run_models.RESULTS = ROOT / "results"
sys.argv = ["run_models.py", "--tag", f"_{SYMBOL.lower()}"]
run_models.main()
print(f"\nDONE: results in {ROOT / 'results'}")
