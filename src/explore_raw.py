"""
First look at the raw files: reads one quotes file, one open-interest file and one volume file
and prints the columns, the data types, a few example values and some counts. Nothing is changed.
It was used once, before writing clean.py, to check the format of the data.

How to run (from the src folder):
  python3 explore_raw.py              uses 2024-05-08 (a Wednesday)
  python3 explore_raw.py 20240214     uses another date (format YYYYMMDD)
"""

import sys, pathlib
import pandas as pd

RAW = pathlib.Path.home() / "thesis_spy" / "data" / "raw"
DAY = sys.argv[1] if len(sys.argv) > 1 else "20240508"

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)


def sep(title):
    """Print a title between two lines."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def find(prefix):
    """File of the chosen day for a given prefix, or the first file available."""
    f = RAW / f"{prefix}_{DAY}.parquet"
    if f.exists():
        return f
    alt = sorted(RAW.glob(f"{prefix}_*.parquet"))
    print(f"  [{prefix}] file for {DAY} not found: using {alt[0].name}")
    return alt[0]


# ---------------------------------------------------------------- QUOTES
fq = RAW / f"quotes_{DAY}_1030.parquet"
if not fq.exists():
    fq = sorted(RAW.glob("quotes_*_1030.parquet"))[0]
    print(f"  [quotes] file for {DAY} not found: using {fq.name}")
q = pd.read_parquet(fq)

sep(f"QUOTES: {fq.name}  ->  {len(q)} rows, {q.shape[1]} columns")
print("\nColumns and types:")
print(q.dtypes.to_string())
print("\nFirst 3 rows:")
print(q.head(3).to_string())

sep("QUOTES: format of 'right', 'strike', 'expiration', 'timestamp'")
print("Values of right:", q["right"].unique().tolist())
print("Count by right:")
print(q["right"].value_counts().to_string())
s = q["strike"]
print(f"\nstrike: type={s.dtype}, min={s.min()}, max={s.max()}")
print("Examples of strikes (sorted, first 10):", [float(x) for x in sorted(s.unique())[:10]])
print("Examples of strikes (last 5):", [float(x) for x in sorted(s.unique())[-5:]])
steps = pd.Series(sorted(s.unique())).diff().dropna()
print("Step between consecutive strikes (most common values):")
print(steps.value_counts().head(5).to_string())
print("\nexpiration: type =", q["expiration"].dtype,
      "| examples:", q["expiration"].unique()[:5].tolist())
print("timestamp: type =", q["timestamp"].dtype,
      "| examples:", q["timestamp"].unique()[:3].tolist())

sep("QUOTES: expiries in the file")
exp = pd.to_datetime(q["expiration"].astype(str))
day = pd.Timestamp(fq.name.split("_")[1])
dte = (exp - day).dt.days
tab = pd.DataFrame({"expiration": exp.dt.date, "dte": dte})
cnt = tab.groupby(["expiration", "dte"]).size().rename("n_quote").reset_index()
print(f"Snapshot date: {day.date()}  (day of the week: {day.day_name()})")
print(f"Number of expiries in the file: {len(cnt)}")
print("Expiries with dte <= 10:")
print(cnt[cnt.dte <= 10].to_string(index=False))
print("Expiries with 25 <= dte <= 40 (the control expiry at ~30 days):")
print(cnt[(cnt.dte >= 25) & (cnt.dte <= 40)].to_string(index=False))
print(f"Expiries beyond 40 days: {int((cnt.dte > 40).sum())}  "
      f"(rows: {int(cnt[cnt.dte > 40].n_quote.sum())})")
print(f"Maximum dte in the file: {int(cnt.dte.max())}")

sep("QUOTES: duplicated rows (same expiry + strike + right)")
dup = q.duplicated(subset=["expiration", "strike", "right"], keep=False)
print(f"Rows involved in duplicates: {int(dup.sum())}")
if dup.any():
    d = q[dup]
    print("Expiries with duplicates:", sorted(pd.to_datetime(d["expiration"].astype(str)).dt.date.unique()))
    print("Are the duplicates identical copies?",
          bool(q[dup].duplicated(keep=False).all()))

sep("QUOTES: quality of bid and ask")
print(f"bid <= 0:            {int((q.bid <= 0).sum())}")
print(f"ask <= 0:            {int((q.ask <= 0).sum())}")
print(f"ask <= bid:          {int((q.ask <= q.bid).sum())}")
print(f"bid or ask missing:  {int(q[['bid', 'ask']].isna().any(axis=1).sum())}")
print("bid_condition, most common values:", q["bid_condition"].value_counts().head(5).to_dict())
print("ask_condition, most common values:", q["ask_condition"].value_counts().head(5).to_dict())
print("bid_size: min/median/max =", q.bid_size.min(), q.bid_size.median(), q.bid_size.max())

# comparison with the 14:30 snapshot
fq2 = RAW / fq.name.replace("_1030", "_1430")
if fq2.exists():
    q2 = pd.read_parquet(fq2)
    sep(f"QUOTES: comparison with {fq2.name}")
    print(f"rows 10:30 = {len(q)} | rows 14:30 = {len(q2)}")
    print("timestamp at 14:30, examples:", q2["timestamp"].unique()[:2].tolist())
    k = ["expiration", "strike", "right"]
    m = q.drop_duplicates(k).merge(q2.drop_duplicates(k), on=k, suffixes=("_1030", "_1430"))
    print(f"options present in both: {len(m)}")
    print(f"share of bids that changed: {(m.bid_1030 != m.bid_1430).mean():.1%}")

# ---------------------------------------------------------------- OPEN INTEREST
fo = find("oi")
oi = pd.read_parquet(fo)
sep(f"OPEN INTEREST: {fo.name}  ->  {len(oi)} rows")
print("\nColumns and types:")
print(oi.dtypes.to_string())
print("\nFirst 3 rows:")
print(oi.head(3).to_string())
print("\nValues of right:", oi["right"].unique().tolist())
print("timestamp, examples:", oi["timestamp"].unique()[:3].tolist())
print("expiration, examples:", oi["expiration"].unique()[:3].tolist())
eo = pd.to_datetime(oi["expiration"].astype(str))
print(f"Number of expiries: {eo.nunique()} | dte min = {(eo - day).dt.days.min()}, "
      f"dte max = {(eo - day).dt.days.max()}")
print(f"open_interest: min = {oi.open_interest.min()}, median = {oi.open_interest.median()}, "
      f"max = {oi.open_interest.max()}, rows with OI = 0: {int((oi.open_interest == 0).sum())}")
dup_oi = oi.duplicated(subset=["expiration", "strike", "right"]).sum()
print(f"Duplicates (expiry + strike + right): {int(dup_oi)}")

# do they match the quotes?
qk = q[["expiration", "strike", "right"]].drop_duplicates()
ok = oi[["expiration", "strike", "right"]].drop_duplicates()
qk["expiration"] = pd.to_datetime(qk["expiration"].astype(str))
ok["expiration"] = pd.to_datetime(ok["expiration"].astype(str))
qk7 = qk[(qk.expiration - day).dt.days <= 7]
mm = qk7.merge(ok, on=["expiration", "strike", "right"], how="left", indicator=True)
print(f"Quotes with dte <= 7: {len(qk7)} | of which with an OI row: "
      f"{int((mm._merge == 'both').sum())}")

# ---------------------------------------------------------------- VOLUME
# the volume file is named after the EXPIRY date: take the closest expiry
fv = None
for cand in sorted(RAW.glob("vol_*.parquet")):
    if cand.name[4:12] >= DAY:
        fv = cand
        break
if fv is None:
    fv = sorted(RAW.glob("vol_*.parquet"))[0]
v = pd.read_parquet(fv)
sep(f"VOLUME: {fv.name}  ->  {len(v)} rows")
print("\nColumns and types:")
print(v.dtypes.to_string())
print("\nFirst 3 rows:")
print(v.head(3).to_string())
print("\nValues of right:", v["right"].unique().tolist())
print("day: type =", v["day"].dtype, "| values:", sorted(pd.Series(v["day"].unique()).astype(str)))
print("expiration, examples:", v["expiration"].unique()[:3].tolist())
print(f"vol_1030: min = {v.vol_1030.min()}, median = {v.vol_1030.median()}, max = {v.vol_1030.max()}")
print(f"vol_1430: min = {v.vol_1430.min()}, median = {v.vol_1430.median()}, max = {v.vol_1430.max()}")
print(f"rows with vol_1430 < vol_1030 (should not exist): "
      f"{int((v.vol_1430 < v.vol_1030).sum())}")
dup_v = v.duplicated(subset=["day", "expiration", "strike", "right"]).sum()
print(f"Duplicates (day + expiry + strike + right): {int(dup_v)}")

# do today's quotes of that expiry find today's volume?
vexp = pd.to_datetime(str(v["expiration"].iloc[0])).date()
vd = v[v["day"].astype(str) == str(day.date())]
qx = q[pd.to_datetime(q["expiration"].astype(str)).dt.date == vexp]
if len(vd) and len(qx):
    mv = qx.merge(vd, on=["strike", "right"], how="left", indicator=True)
    print(f"\nExpiry {vexp}: quotes at 10:30 = {len(qx)}, "
          f"with a volume row for today = {int((mv._merge == 'both').sum())}")
else:
    print(f"\nExpiry {vexp}: no match possible (volume rows today = {len(vd)}, "
          f"quotes = {len(qx)})")

sep("END OF EXPLORATION")
