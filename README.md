# Donchian + option-buying overlay — a falsification study

**The question:** does a Donchian channel breakout — a popular retail signal —
generate real alpha once it's overlaid with an actual weekly NIFTY option and
priced against real bars and real Zerodha costs, rather than assumed on paper?

**The method:** build the breakout signal causally (no lookahead, proven by a
placebo that deliberately breaks that rule), price every leg against real
1-minute option data, hold out 2026 and never touch it until every cut point
was fixed on 2025, and run inference clustered by day rather than by
observation.

**The answer: no.** The breakout carries no directional information, the
channel's volatility information is already priced into the option, and every
long-premium arm tested loses money in both the in-sample and holdout periods.
The one thing that survives is the channel's *width* as a forecast of the next
move's size — a sizing input, not a trading signal on its own.

**Then what happened:** dropping the long-premium idea and testing the mirror
trade — selling premium instead of buying it, via an iron condor — produced a
strong-looking equity curve in-sample that reversed sign the moment it met its
own holdout. That reversal is itself the more interesting result: a case study
in what an unvalidated backtest looks like right up until it isn't, and why
the discipline in the method above (locked holdout, day-clustered inference,
placebo checks) exists in the first place.

Full result: `results/findings.json`, and the write-up in
`docs/findings.md`.

## Two acts, in the order the code runs them

| | |
|---|---|
| Spot sessions | 570 (2024-01-01 → 2026-05-05) |
| Option sessions matched | 353, across 73 weekly expiries |
| Option legs priced | 54,275 |
| Design / in-sample / holdout | 2024 / 2025 / Jan–May 2026 (locked) |
| Inference | day-clustered bootstrap throughout |

**Act 1 — `scripts/core/` (01-10): does the long-premium overlay work?** Build
the signal on spot alone, then price it against a real option, then check
whether anything survives the checking.

**1. Is there a directional edge in the breakout itself, before any option
touches it?** (`02_spot_grid.py`, `03_event_study.py` — 2024 design sample,
spot points only, no options collected for 2024 so nothing here could have
seen 2025 or 2026):

![Breakout continuation: no signal at any horizon, any period](docs/img/event_study.png)

No. Not at any forward horizon, in any of the three periods. `04_conditioning.py`
runs the same question as a 93-cell grid over other conditioning variables in
the design sample — 5 came back significant at the 5% level, against 4.7
expected from chance alone.

**2. Price it against a real option anyway** (`05_build_option_store.py`
through `08_daily_arms.py` — 54,275 legs, real 1-minute bars, full Zerodha
cost stack):

![Every long-premium arm loses money, both periods](docs/img/daily_arms.png)

| | |
|---|---:|
| Straddle round-trip cost | 3.79 pts (1.54% of premium) |
| Clean forward-move placebo, 2,068 events | −0.53 pts (wrong sign, ≈0) |
| Deliberately leaked placebo, 179 events | −49.95 pts (proves the pipeline isn't leaking) |

All three arms lose money in both periods. Costs matter but aren't the whole
story — even at zero cost the mean move doesn't flip sign, and the leak
placebo confirms this isn't an artifact of an accidentally forward-looking
signal.

**3. Does anything survive?** (`04_conditioning.py`'s Q2, `06`/`07` sorted by
width quintile) — yes, one thing:

![Channel width forecasts range, holds out of sample](docs/img/width_signal.png)

The channel's width relative to the same time of day over the prior 20
sessions forecasts the size of the next move, in-sample and on the 2026
holdout. It's a sizing input, not a directional edge — the option market
already prices that same volatility information into the premium, which is
exactly why buying premium conditioned on width is still a wash.

**Act 2 — `scripts/iron_condor/` (11-15) + `scripts/extensions/` (16-50): drop
the long-premium thesis, sell the option instead.** A delta-selected iron
condor with a Donchian buy-stop hedge, tested against the same holdout
discipline, then stress-tested by the 35-script robustness battery.

**4. The mirror trade, run over the same two periods:**

![Iron condor equity curve, strong in-sample then negative in 2026](docs/img/ic_equity_curves.png)

| Arm | 2025 (in-sample) | 2026 YTD (holdout) |
|---|---:|---:|
| IC blind | +₹172,572 | −₹128,239 |
| + hedge 1x | +₹440,506 | −₹136,574 |
| + hedge 2x | +₹726,092 | −₹138,490 |
| + hedge 3x | +₹1,011,681 | −₹140,405 |

Strongly positive in 2025, negative at every hedge multiplier the moment it
meets 2026 — and it gets worse in absolute terms as the hedge multiplier goes
up, which is what added gross exposure looks like, not what a hedge looks
like.

**5. Before taking that reversal at face value, the robustness battery checks
the two obvious objections.** Is it just a pessimistic cost assumption
(`extensions/16-19, 31, 35, 47`)?

![Sharpe vs slippage, IC blind: 2026 negative even before slippage](docs/img/slippage_sweep.png)

No — the 2026 curve is already Sharpe-negative at zero assumed slippage,
so no slippage number rescues it. And is the hedge itself picking up a
forward-looking leak (`iron_condor/14`'s rotation placebo)?

| Hedge-gate placebo | pts | trades | win rate |
|---|---:|---:|---:|
| Clean (real gate) | +0.97 | 518 | 39.6% |
| Random gate | −0.32 | — | — |
| Deliberately leaked gate | −9.21 | — | — |

No — the clean gate's attribution sits near zero and looks nothing like the
leaked one. Neither objection explains the reversal, which leaves the
uncomfortable one: the `0.35/0.15` config and `optg` hedge style behind the
equity curve above were picked from a ~30-way grid with no stated selection
rule, and `extensions/50`'s deflated-Sharpe/PBO check exists for exactly that
failure mode but was never run against this particular choice. Full context
is in `docs/findings.md`.

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
python3 scripts/core/06_straddle_by_width.py    # price ~54,000 ATM straddles on real bars
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

## What each Act 2 script tests

Act 2's 35 scripts, grouped by what they check:

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
