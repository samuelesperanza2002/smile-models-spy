#!/bin/bash
# Full rerun of the pipeline: from the raw ThetaData files already on disk to all tables and figures.
# Nothing is downloaded. Before the run, the previous results/ folder is renamed results_prev_<date>;
# at the end, every new results file is compared with the previous one. Everything printed on the
# screen is also saved in ~/thesis_spy/run_all_<date>.log.
# This file is not Python: it is a list of commands for the Terminal, run one after the other.
#
# How to run (from ~/thesis_spy/src):
#   caffeinate -i bash run_all.sh          full run: cleaning of all 977 days, then models, tables, figures
#   caffeinate -i bash run_all.sh --fast   keep data/clean and data/clean_aft, rerun only the models and tables
# (caffeinate keeps the Mac awake; set -e below stops the run at the first error.)
set -e
export PYTHONUNBUFFERED=1      # show the Python progress lines at once, also through tee
cd "$(dirname "$0")"
T=$HOME/thesis_spy
STAMP=$(date +%Y%m%d_%H%M)
LOG=$T/run_all_$STAMP.log
exec > >(tee -a "$LOG") 2>&1
FAST=0; [ "$1" = "--fast" ] && FAST=1
T0=$(date +%s)
step() { echo; echo "=== $(date +%H:%M:%S) [$(( ($(date +%s) - T0) / 60 )) min] $*"; }

step "Start (fast=$FAST). Log: $LOG"
python3 -c "import numpy, scipy, pandas, pyarrow, matplotlib; print('python ok')"
test -f "$T/data/sample_dates.csv" || { echo "data/sample_dates.csv missing"; exit 1; }

# 0. keep the previous results, for the comparison at the end
if [ -d "$T/results" ]; then
    PREV=$T/results_prev_$STAMP
    mv "$T/results" "$PREV"
    echo "previous results moved to $PREV"
    mkdir -p "$T/results"
    if [ $FAST -eq 1 ]; then
        cp "$PREV"/filter_counts*.csv "$PREV"/smile_summary*.csv "$PREV"/clean_log*.csv "$T/results/"
        [ -f "$PREV/diag_filters_sample.csv" ] && cp "$PREV/diag_filters_sample.csv" "$T/results/"
        echo "fast mode: cleaning outputs copied from the previous results"
    fi
else
    PREV=""
    mkdir -p "$T/results"
fi

# 1. cleaning (filters 1-9) of all trading days, three snapshots; minimal (AFT) filters on the sample days only
if [ $FAST -eq 0 ]; then
    step "1a. clean.py --all --snap 1030 1130 1430 (main filters, all days)"
    python3 clean.py --all --snap 1030 1130 1430
    step "1b. clean.py --all --days sample --snap 1030 1430 --filters aft"
    python3 clean.py --all --days sample --snap 1030 1430 --filters aft
    step "1c. diag_filters.py (diagnostic, optional)"
    python3 diag_filters.py || echo "warning: diag_filters.py failed (not needed for the thesis tables)"
else
    step "1. cleaning skipped (--fast)"
fi

# 2. models: main specification, five robustness variants, AFT filters
step "2a. run_models.py (main)"
python3 run_models.py
step "2b. run_models.py --beta 0.5"
python3 run_models.py --beta 0.5
step "2c. run_models.py --vega"
python3 run_models.py --vega
step "2d. run_models.py --time business"
python3 run_models.py --time business
step "2e. run_models.py --calib 1130"
python3 run_models.py --calib 1130
step "2f. run_models.py --spread 0.3"
python3 run_models.py --spread 0.3
step "2g. run_models.py --filters aft"
python3 run_models.py --filters aft

# 3. tables, figures, trading exercise
step "3a. make_tables.py"
python3 make_tables.py
step "3b. robustness_summary.py"
python3 robustness_summary.py
step "3c. compare_filters.py"
python3 compare_filters.py
step "3d. make_figures.py"
python3 make_figures.py
step "3e. trading.py"
python3 trading.py
step "3f. tables_chapter4.py"
python3 tables_chapter4.py

# 4. licence check, then comparison with the previous results
step "4a. check_results.py"
python3 check_results.py
if [ -n "$PREV" ]; then
    step "4b. comparison with the previous results ($PREV)"
    PREV="$PREV" python3 - <<'PYEOF'
import os, pathlib
import numpy as np, pandas as pd
new, old = pathlib.Path.home() / "thesis_spy" / "results", pathlib.Path(os.environ["PREV"])
same, text_only, numbers, missing = 0, [], [], []
for f in sorted(new.rglob("*.csv")):
    g = old / f.relative_to(new)
    rel = f.relative_to(new).as_posix()
    if not g.exists():
        missing.append(rel); continue
    try:
        a, b = pd.read_csv(f, low_memory=False), pd.read_csv(g, low_memory=False)
    except Exception:
        continue
    na, nb = a.select_dtypes("number"), b.select_dtypes("number")
    if a.shape != b.shape or na.shape != nb.shape:
        numbers.append(f"{rel} (shape {a.shape} vs {b.shape})"); continue
    num_same = all(np.allclose(na.iloc[:, i].astype(float), nb.iloc[:, i].astype(float), equal_nan=True, atol=1e-9) for i in range(na.shape[1]))
    if not num_same:
        d = max((np.nanmax(np.abs(na.iloc[:, i].astype(float) - nb.iloc[:, i].astype(float))) for i in range(na.shape[1])), default=np.nan)
        numbers.append(f"{rel} (max abs difference {d:.3g})"); continue
    if list(a.columns) != list(b.columns):
        text_only.append(f"{rel} (column names changed, numbers identical)"); continue
    txt = [c for c in a.columns if c not in na.columns]
    if all(a[c].astype(str).equals(b[c].astype(str)) for c in txt):
        same += 1
    else:
        text_only.append(f"{rel} (text labels changed, numbers identical)")
print(f"CSV files identical to the previous run: {same}")
print("CSV files with the same numbers and different text labels:", len(text_only)); [print("  -", x) for x in text_only]
print("CSV files with DIFFERENT NUMBERS:", len(numbers)); [print("  -", x) for x in numbers]
print("CSV files new in this run:", len(missing)); [print("  -", x) for x in missing]
PYEOF
fi
step "Done. Total time $(( ($(date +%s) - T0) / 60 )) minutes. Log saved in $LOG"
