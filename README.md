# Donchian + option-buying overlay — a falsification study

Tests whether a day-scoped Donchian channel breakout on 5-minute NIFTY 50 bars,
overlaid with a bought weekly option leg, generates alpha.

**It does not.** The breakout carries no directional information, the channel's
volatility information is already in the option price, and every long-premium
arm tested is negative-expectancy in both the in-sample and holdout periods.
The one component worth keeping is the channel *width* as a range forecast for
position sizing.

Full result: `results/findings.json`, and the write-up in
`docs/findings.md`.

## Data you need to supply

Not bundled. Both come from the weekly Zerodha archives `YYYYMMDD.zip`
(one per weekly expiry, LZMA-compressed):

- `nifty_spot.csv` inside each archive — 1-minute NIFTY 50 bars. Script 01
  stitches and de-duplicates all of them into one continuous series.
- `{strike}{CE|PE}_{expiry}.csv` — 1-minute option bars, columns
  `Timestamp, Open, High, Low, Close, Volume, OI`, `Timestamp` as
  `DD-MM-YYYY HH:MM:SS`.

Point `DL_DIR` at the folder holding the archives (defaults to
`~/mnt/Downloads`) and `STORE_DIR` at where the derived store should live
(defaults to `~/scratch/store`).

## Running it

```
python3 scripts/core/01_build_store.py          # spot parquet + option index
python3 scripts/core/02_spot_grid.py            # 2024 design grid vs random-entry control
python3 scripts/core/03_event_study.py          # conditional vs unconditional forward move
python3 scripts/core/04_conditioning.py         # Q1 direction conditioners, Q2 width -> move size
python3 scripts/core/05_build_option_store.py   # extract near-the-money legs (~257 MB)
python3 scripts/core/06_straddle_by_width.py    # price 53,627 ATM straddles on real bars
python3 scripts/core/07_analyse_straddle.py     # sort by width quintile, IS/OOS
python3 scripts/core/08_daily_arms.py           # three arms, one decision per session
python3 scripts/core/09_verify.py               # independent re-pricing + look-ahead audit
python3 scripts/core/10_placebo_and_export.py   # forward-leak placebo, findings.json
```

Scripts 02-04 need only 01. Scripts 06-08 need 05. Each caches to `STORE_DIR`,
so re-runs are fast.

`python3 tests/smoke_test.py` builds fabricated weekly-expiry archives with
the same schema and runs 01-08 end to end -- no real data required, no
assertion on trading results, just a check that the pipeline still runs
after a dependency bump or a refactor.

## Beyond the core pipeline (`scripts/iron_condor/`, `scripts/extensions/`)

`scripts/core/` (01-10) tests and rejects the long-premium Donchian overlay.
`scripts/iron_condor/` (11-15) and `scripts/extensions/` (16-50) are a
separate, larger body of work: they drop the long-premium thesis and test
short-premium structures (iron condors, naked strangles) plus a battery of
robustness checks. Grouped by what they test:

| Scripts | Theme |
|---|---|
| `iron_condor/` 11-15 | Delta-selected iron condor on the Donchian-triggered entry, its metrics, independent verification, a rotation placebo, and export |
| `extensions/` 16-19 | Spread/slippage cost modelling for the option legs |
| `extensions/` 20-22 | Weekly (vs. daily) holding period, intraday wing behaviour |
| `extensions/` 23-25 | Regime features, regime decomposition, whether IS/OOS split is itself a regime effect |
| `extensions/` 27-29, 33, 37 | Forecast panels/regression on vol and term structure |
| `extensions/` 31, 35, 47 | Cost audit, slippage ladder, time-of-day cost |
| `extensions/` 32, 34 | Term structure, VRP (variance risk premium) state |
| `extensions/` 36, 38 | Skew panel and skew conditioning |
| `extensions/` 39 | Volatility-regime gating |
| `extensions/` 40-41 | Adversarial stress tests and confound checks |
| `extensions/` 43-44 | A harder backtest and a walk-forward that picks structure from past data only |
| `extensions/` 45-46, 48 | Overnight gap behaviour, liquidity/persistence |
| `extensions/` 49-50 | Naked premium selling and its adversarial battery (block bootstrap, deflated Sharpe, PBO, stop-loss overlay, sub-period stability) |

Headline numbers from both bodies of work are synthesised in
`docs/findings.md`.

## Layout

- `lib/core.py` — data loading, 5-minute resampling, day-scoped Donchian
  signals, the 1-minute-path trade simulator, day-clustered bootstrap.
- `lib/features.py` — causal features: channel width relative to the same
  bar-of-day over the prior 20 sessions, breakout extension, opening-range
  expansion, overnight gap.
- `lib/costs.py` / `lib/ic_costs.py` — Zerodha round-trip cost models
  (brokerage, STT, exchange charge, SEBI, stamp, GST, slippage). `costs.py`
  holds the rates flat (used only by `scripts/core/`, whose sample sits
  almost entirely inside one statutory regime -- see
  `scripts/extensions/31_cost_audit.py` for the quantified effect of that
  simplification); `ic_costs.py` is date-aware across the statutory-rate
  changes and is what everything from `scripts/iron_condor/` onward uses.
- `scripts/core/` — the ten-script falsification pipeline (see above).
- `scripts/iron_condor/` — the delta-selected condor + hedge, and its export.
- `scripts/extensions/` — the wider robustness/regime/other-market battery,
  numbered 16-50. Five numbers in that range (26, 30, 42, 51, 52) were
  BankNifty replication scripts, removed once the project's scope settled
  on NIFTY only -- the gap is deliberate, not a missing file.
- `audit/` — independent, from-scratch reimplementations used to cross-check
  the main pipeline.
- `results/` — `findings.json` / `ic_findings.json` plus the per-test CSVs.
- `docs/` — the synthesised write-up.
- `tests/` — a smoke test covering `scripts/core/`.

## Method notes

- The channel at bar *i* uses the *N* bars strictly before *i*, within the same
  session (`shift(1)` before `rolling`). Script 09 proves this with a placebo
  that deliberately builds the channel from future bars.
- Entry is the open of the first 1-minute bar after the signalling 5-minute bar
  closes. Option marks require non-zero volume at entry; staleness is capped at
  three minutes.
- Expiry mapping is the nearest weekly expiry strictly after the session date
  (no 0-DTE).
- 2024 is the design sample, 2025 in-sample, Jan-Apr 2026 a locked holdout with
  every cut point fitted on 2025 and applied verbatim.
- All inference is day-clustered — intraday observations on one session are not
  independent draws.
