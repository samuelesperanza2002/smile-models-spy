"""
Choice of the sample: pairs of consecutive trading days (t, t+1), classified with the VIX.

It reads only the names of the quotes files (to know which days are trading days) and the VIX
from FRED. It does not read any option data and does not need the Theta Terminal.

Rules (Chapter 4 of the thesis):
  - t is a Tuesday or a Wednesday; t+1 is the next trading day and the next calendar day;
  - no Mondays: the day before t must be a trading day (after a Monday holiday the Tuesday
    behaves like a Monday and is excluded);
  - no day before a holiday: the trading day after t+1 must be the next calendar day;
  - no half days (24 December, 3 July, Friday after Thanksgiving) at t or t+1;
  - no SPY ex-dividend weeks (third Friday of March, June, September and December);
  - no FOMC days (Federal Reserve announcement at 14:00) at t or t+1: they form a separate
    "event" group.
The volatility regime comes from the VIX close of the day BEFORE t (known at 10:30 of t):
  calm < 15, normal 15-20, stressed > 25 (20-25 is a buffer zone and is excluded).

Output: data/fred_VIXCLS.csv and data/fred_DTB4WK.csv (copies of the FRED series),
data/sample_candidates.csv (all pairs that pass the rules), data/sample_proposal.csv
(a proposal with the same number of periods per regime, spread over time) and
data/event_candidates.csv. The final data/sample_dates.csv was written by hand from the
proposal and is never overwritten by this script.

How to run (from the src folder):
  python3 select_sample.py        6 periods per regime
  python3 select_sample.py 8      8 periods per regime
"""

import sys, pathlib
from datetime import date, timedelta
import numpy as np
import pandas as pd

HOME = pathlib.Path.home() / "thesis_spy"
RAW, DATA = HOME / "data" / "raw", HOME / "data"
PER_REGIME = int(sys.argv[1]) if len(sys.argv) > 1 else 6

# FOMC announcement days (second day of each meeting), checked on the Federal Reserve website.
FOMC = pd.to_datetime("""
2023-02-01 2023-03-22 2023-05-03 2023-06-14 2023-07-26 2023-09-20 2023-11-01 2023-12-13
2024-01-31 2024-03-20 2024-05-01 2024-06-12 2024-07-31 2024-09-18 2024-11-07 2024-12-18
2025-01-29 2025-03-19 2025-05-07 2025-06-18 2025-07-30 2025-09-17 2025-10-29 2025-12-10
2026-01-28 2026-03-18 2026-04-29 2026-06-17 2026-07-29 2026-09-16 2026-10-28
""".split())


def fred(series):
    """Download a FRED series (once) and save it in data/fred_<series>.csv."""
    f = DATA / f"fred_{series}.csv"
    if not f.exists():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
        pd.read_csv(url).to_csv(f, index=False)
        print(f"downloaded {series} from FRED -> {f.name}")
    df = pd.read_csv(f)
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")   # "." = missing value
    return df.dropna().set_index("date")["value"]


# ---------------------------------------------------------------- trading days
tdays = sorted(pd.Timestamp(f.name[7:15]) for f in RAW.glob("quotes_*_1030.parquet"))
tset = set(tdays)
print(f"Trading days in the data: {len(tdays)} ({tdays[0].date()} -> {tdays[-1].date()})")

vix = fred("VIXCLS")
print(f"VIX: {len(vix)} observations, last {vix.index[-1].date()}")
fred("DTB4WK")   # 4-week T-bill: used for the discount factor


def third_friday(y, m):
    """Third Friday of month m of year y."""
    d = date(y, m, 15)
    return d + timedelta(days=(4 - d.weekday()) % 7)


def thanksgiving(y):
    """Thanksgiving Day (fourth Thursday of November) of year y."""
    d = date(y, 11, 1)
    first_thu = d + timedelta(days=(3 - d.weekday()) % 7)
    return first_thu + timedelta(weeks=3)


years = range(tdays[0].year, tdays[-1].year + 1)
half_days = set()
div_weeks = set()        # (year, ISO week number) of the ex-dividend weeks
for y in years:
    half_days |= {pd.Timestamp(date(y, 12, 24)), pd.Timestamp(date(y, 7, 3)),
                  pd.Timestamp(thanksgiving(y) + timedelta(days=1))}
    for m in (3, 6, 9, 12):
        tf = third_friday(y, m)
        div_weeks.add((tf.isocalendar()[0], tf.isocalendar()[1]))

fomc_set = set(FOMC)
bad = [d.date() for d in FOMC if d not in tset and tdays[0] <= d <= tdays[-1]]
if bad:
    print("WARNING: FOMC dates that are not trading days in the data:", bad)
notwed = [d.date() for d in FOMC if d.weekday() != 2]
if notwed:
    print("Note: FOMC dates that do not fall on a Wednesday:", notwed)

# ---------------------------------------------------------------- admissible pairs
rows = []
one = pd.Timedelta(days=1)
for t in tdays:
    if t.weekday() not in (1, 2):                      # 1 = Tuesday, 2 = Wednesday
        continue
    t1 = t + one
    reasons = []
    if t1 not in tset:
        reasons.append("t+1 is not a trading day")
    if (t - one) not in tset:
        reasons.append("holiday the day before (t behaves like a Monday)")
    if (t1 + one) not in tset:
        reasons.append("t+1 is a pre-holiday day")
    if t in half_days or t1 in half_days:
        reasons.append("half day")
    if (t.isocalendar()[0], t.isocalendar()[1]) in div_weeks:
        reasons.append("SPY ex-dividend week")
    is_fomc = t in fomc_set or t1 in fomc_set
    prev = vix[vix.index < t]
    if len(prev) == 0 or (t - prev.index[-1]).days > 4:
        reasons.append("VIX missing")
        v_tm1 = np.nan
    else:
        v_tm1 = prev.iloc[-1]
    v_t = vix.get(t, np.nan)
    v_t1 = vix.get(t1, np.nan)
    if np.isnan(v_tm1):
        regime = "n.a."
    elif v_tm1 < 15:
        regime = "calm"
    elif v_tm1 <= 20:
        regime = "normal"
    elif v_tm1 > 25:
        regime = "stressed"
    else:
        regime = "buffer (20-25)"
    rows.append(dict(t=t.date(), t1=t1.date(), weekday_t=t.day_name()[:3],
                     vix_tm1=v_tm1, vix_t=v_t, vix_t1=v_t1, regime=regime,
                     fomc=is_fomc, excluded="; ".join(reasons)))

allp = pd.DataFrame(rows)
ok = allp[(allp.excluded == "") & ~allp.fomc & allp.regime.isin(["calm", "normal", "stressed"])]
ok.to_csv(DATA / "sample_candidates.csv", index=False)
events = allp[(allp.excluded == "") & allp.fomc & (allp.t.map(lambda d: pd.Timestamp(d)).isin(fomc_set))]
events.to_csv(DATA / "event_candidates.csv", index=False)

print(f"\nTue/Wed -> Wed/Thu pairs in the data: {len(allp)} | excluded by the rules: "
      f"{int((allp.excluded != '').sum())} | FOMC: {int(allp.fomc.sum())} | admissible: {len(ok)}")
print("\nAdmissible pairs by regime and year (regime = VIX close of the day before t):")
print(pd.crosstab(ok.regime, ok.t.map(lambda d: d.year), margins=True).to_string())
print("\nPairs in the buffer zone (VIX 20-25), excluded by construction:",
      int(((allp.excluded == "") & (allp.regime == "buffer (20-25)")).sum()))

# ---------------------------------------------------------------- proposal: k per regime, spread over time
def spread_pick(df, k):
    """Pick k pairs spread evenly over time, using a different month for each pick when possible."""
    df = df.sort_values("t").reset_index(drop=True)
    if len(df) <= k:
        return df
    days = np.array([pd.Timestamp(d).value for d in df.t], dtype=float)
    targets = np.linspace(days.min(), days.max(), k)
    chosen, used_months = [], set()
    for tg in targets:
        order = np.argsort(np.abs(days - tg))
        pick = None
        for i in order:                        # first try a month not used yet
            if i in chosen:
                continue
            ym = (df.t[i].year, df.t[i].month)
            if ym not in used_months:
                pick = i
                break
        if pick is None:                       # otherwise the closest one not chosen yet
            pick = next(i for i in order if i not in chosen)
        chosen.append(pick)
        used_months.add((df.t[pick].year, df.t[pick].month))
    return df.loc[sorted(chosen)]


parts = []
for reg in ["calm", "normal", "stressed"]:
    sub = ok[ok.regime == reg]
    pick = spread_pick(sub, PER_REGIME)
    if len(pick) < PER_REGIME:
        print(f"WARNING: regime {reg}: only {len(pick)} admissible pairs (requested {PER_REGIME})")
    parts.append(pick)
prop = pd.concat(parts).sort_values("t").reset_index(drop=True)
prop.insert(0, "period_id", range(1, len(prop) + 1))
prop = prop.drop(columns=["fomc", "excluded"])
prop.to_csv(DATA / "sample_proposal.csv", index=False)

print(f"\nPROPOSAL ({len(prop)} periods) -> data/sample_proposal.csv")
print(prop.to_string(index=False, float_format=lambda x: f"{x:.1f}"))

print(f"\nAll admissible STRESSED pairs ({int((ok.regime == 'stressed').sum())}), to choose by hand:")
print(ok[ok.regime == "stressed"].drop(columns=["fomc", "excluded"])
      .to_string(index=False, float_format=lambda x: f"{x:.1f}"))

print(f"\nEVENT days (FOMC as t, with a regular t+1): {len(events)} -> data/event_candidates.csv")
print(events.drop(columns=["fomc", "excluded"]).to_string(index=False, float_format=lambda x: f"{x:.1f}"))
print("\nEND. Check the proposal and edit data/sample_dates.csv by hand.")
