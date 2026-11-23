# Smile models at the shortest horizon

Python code of my master's thesis *Smile models at the shortest horizon: Black-Scholes, SABR and
Vanna-Volga on ultra-short-term SPY options* (M.Sc. in International Finance, double degree
University of Pisa and HfWU Nürtingen-Geislingen, 2026).

Samuele Speranza

## What the thesis studies

Options on SPY (the ETF on the S&P 500) with zero to seven days to expiry, including 0DTE
options that expire on the same day, are now the most traded part of the S&P 500 options
market. The thesis tests whether two simple smile models used by traders price them well:

- **Black-Scholes** (benchmark): one flat volatility, the at-the-money one;
- **SABR**: Hagan's implied volatility formula with beta = 1, three parameters per smile;
- **Vanna-Volga**: the exact method of Castagna and Mercurio (2007), built from three liquid
  options (the "pillars"), adapted to equity options with three sets of pillars.

The sample has 18 periods of two consecutive trading days between December 2022 and October 2026
(6 calm, 6 normal and 6 stressed, by the VIX of the day before), plus 2 days with a Federal
Reserve (FOMC) announcement. The models are fitted at 10:30 New York time and tested in sample,
at 14:30 on the same day and at 10:30 on the next day. The thesis also runs robustness checks, a
trading exercise with transaction costs, and the same tests on two commodity ETFs (USO and GLD).

## Main results

The numbers below come from Chapters 5 to 7 of the thesis. Errors are in volatility points
(1 point = one percentage point of implied volatility, for example 21% against 20%).

- In sample, on 0DTE options, the average error falls from 8.2 points for Black-Scholes to 4.2
  for Vanna-Volga, 1.0 for SABR fitted to the same three pillars and 0.45 for SABR fitted to half
  of the strikes. SABR beats Black-Scholes on every smile of the sample.
- Vanna-Volga is almost exact between its pillars but misses by several points outside them,
  above all on deep out-of-the-money puts. Pillars moved towards the puts work better than the
  standard choice taken from FX markets.
- The SABR volatility of volatility (nu) grows strongly as the expiry gets closer, roughly as
  the time to expiry to the power -0.4.
- Out of sample, a simple rule beats every model: keep for each strike the implied volatility
  it had at 10:30 ("sticky strike"). On 0DTE options all methods have errors of 5 to 7 points
  between 10:30 and 14:30.
- In the trading exercise, the differences between model and market prices are mostly too
  small to pay the transaction costs. The trades suggested by Vanna-Volga lose money.
- On USO and GLD the in-sample ranking of the models is the same as on SPY.

## Data and licence

The option quotes come from ThetaData (OPRA NBBO quotes, Options Value subscription), under a
licence for personal use. **The data are not in this repository and cannot be shared.** As the
licence requires, all raw and cleaned quotes were deleted at the end of the subscription; only
aggregated results were kept, and they are not published here either.

Options market data provided by Theta Data (https://www.thetadata.net).
This is independent research by the author. It is not reviewed, approved or sponsored by
Theta Data.

The VIX (series VIXCLS) and the 4-week T-bill rate (series DTB4WK) come from FRED and are free.
The file `data/sample_dates.csv` lists the 20 sample periods: dates, VIX values and regime.
It contains no option data.

To run the code you need your own ThetaData subscription (Options Value) and the Theta Terminal,
the ThetaData program that runs on your computer and answers the download requests.

## Scripts

All scripts are in `src/`. Each one starts with a short description: what it does, what it reads,
what it writes and how to run it. The table follows the order in which the scripts run.

| Step | Script | What it does |
|---|---|---|
| 1. Download | `download_spy.py`, `volume_fast.py`, `download_1130.py` | quotes at 10:30, 11:30 and 14:30, open interest and traded volume, from the Theta Terminal |
| 2. Sample | `select_sample.py` | candidate periods from the VIX, the FOMC calendar and the exclusion rules |
| 3. Cleaning | `clean.py` | nine filters, forward and discount factor from put-call parity, implied volatility, standardized moneyness |
| 4. Models | `black.py`, `sabr.py`, `vv.py`, `evaluate.py` | Black's formula and implied volatility; SABR; Vanna-Volga; error measures and the two test designs |
| 5. Tests | `run_models.py` | all models on all sample periods, in sample and out of sample, with options for the robustness checks |
| 6. Tables and figures | `tables_chapter4.py`, `make_tables.py`, `robustness_summary.py`, `compare_filters.py`, `make_figures.py`, `make_fig1_smiles.py`, `make_fig6_oos.py` | the tables and figures of Chapters 4 and 5 |
| 7. Trading | `trading.py` | trading exercise with transaction costs (Chapter 6) |
| 8. Other underlyings | `download_symbol.py`, `bonus.py`, `bonus_tables.py` | the same pipeline on USO and GLD (Chapter 7) |
| 9. Checks | `run_all.sh`, `check_results.py` | full rerun from the raw files (cleaning, models, the main tables and figures, trading) with a comparison against the previous results; licence check of the results folder |

Other scripts, used for checks and diagnostics while the work was in progress:
`explore_raw.py` (first look at the raw files), `summarize_clean.py` and `diag_filters.py`
(checks on the cleaning), `test_models.py` (models on one file), `contact_sheet.py` and
`check_outliers.py` (visual check of the smiles), `probe_symbol.py` and `bonus_diag.py`
(choice and checks of the other underlyings).

## How to run

Requirements: Python 3 (tested with Python 3.14 on macOS with an Apple M1 chip) and the packages in
`requirements.txt`, with the versions used for the thesis. The scripts expect the project in the folder `~/thesis_spy`.

```
git clone <this repository> ~/thesis_spy
cd ~/thesis_spy
pip3 install -r requirements.txt
cd src
caffeinate -i python3 download_spy.py        # needs the Theta Terminal; takes several hours
caffeinate -i python3 download_1130.py
caffeinate -i bash run_all.sh                # cleaning, models, tables, figures, trading
```

`caffeinate` is a macOS command that keeps the computer awake; on other systems, leave it out.
The other underlyings run with `download_symbol.py USO`, `bonus.py USO` and
`bonus_tables.py USO GLD`.

## Notes

- The scripts call the tables by working names (T0 to T11). The thesis numbers them by chapter.
- The thesis (Chapter 8) shows short excerpts of this code with their line numbers. For this
  repository the comments were rewritten and extended, so the line numbers may differ by a few
  lines. The code itself is the same.
- I wrote this code with the help of an AI assistant (Claude, by Anthropic), which also helped
  with the comments and this README. I am responsible for the study design, the choices
  described in the thesis and the results.
