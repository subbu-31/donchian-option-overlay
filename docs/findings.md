# Donchian + option overlay: findings

One-page synthesis of `results/findings.json` and `results/ic_findings.json`.
Full numbers live in those files and the per-test CSVs in `results/`.

## Data

566 NIFTY 50 spot sessions (2024-01-01 to 2026-04-28), 349 sessions with
matched option coverage across 73 weekly expiries, 53,627 near-the-money
option legs extracted from real 1-minute bars. 2024 is the design sample,
2025 is in-sample, Jan-Apr 2026 is a locked holdout with every cut point
fitted on 2025 and applied verbatim.

## Headline result: the breakout carries no tradeable signal

- **Directional edge**: the clean forward-move placebo across 2,068 breakout
  events shows a mean 30-minute move of -0.53 points -- indistinguishable
  from zero, and the *wrong* sign. A deliberately leaked (forward-looking)
  version of the same channel shows -49.95 points over 179 events -- proof
  the pipeline isn't accidentally leaking; a leaking pipeline looks nothing
  like the clean one.
- **All three long-premium trading arms are net-negative, in-sample and out
  of sample.** Arm A (long ATM straddle) and Arm C (long leg on the first
  breakout) both lose money in 2025 and the 2026 holdout after the full
  Zerodha cost stack; none of the daily-arm t-stats clear significance in
  the direction that would matter.
- **Costs are material but not the whole story**: a straddle round trip
  costs 3.79 points, ~1.55% of premium. Even zeroing out costs would not
  flip the sign of the mean move.

## The one thing that survives: width as a range forecast

Channel width relative to the same bar-of-day over the prior 20 sessions
correlates with the realised forward range at 0.542 (2025) / 0.464 (2026
holdout) -- a real, out-of-sample-stable relationship. It's a sizing input,
not a directional edge: the option market already prices the volatility
information in the channel, so buying premium conditioned on width is a
wash (script 06/07), but using width to size positions (wider channel ->
larger expected range -> smaller position for constant risk) is a
legitimate, holdout-validated use of the same feature.

## The extension that looked promising but does not survive its own holdout

Scripts 11-52 move away from the Donchian/long-premium thesis entirely and
test short-premium structures (iron condors, naked strangles) with delta-
selected strikes, GTT stop-loss overlays, and vol-regime gating. The blind
iron condor and its hedged variants show a strong cumulative equity curve
over 2025-2026 in `results/ic_findings.json`, and the rotation placebo does
support that the edge isn't from a leaked gate (leaked-label attribution
-9.21 pts vs. clean +0.97 pts).

**But split by year, the picture reverses.** The condor is strongly positive
in 2025 and uniformly negative in the locked 2026 holdout, at every hedge
multiplier, and gets worse in absolute terms as the hedge multiplier
increases (more "hedge" -> bigger 2025 gain AND bigger 2026 loss, which is
what increased gross exposure looks like, not what a hedge looks like):

| Arm | 2025 | 2026 YTD (holdout) |
|---|---|---|
| IC blind | +172,596 | -101,969 |
| + hedge 1x | +440,546 | -128,420 |
| + hedge 2x | +726,150 | -148,737 |
| + hedge 3x | +1,011,752 | -169,053 |

The pipeline's own live-computed `slip_surface` confirms this isn't a
slippage artifact: the blind condor is already Sharpe -0.91 in the 2026
holdout at **zero** assumed slippage. The `CFG = "0.35/0.15"` and `optg`
hedge style promoted to this headline export were picked from a ~30-way
grid (3 delta configs x 10 arm variants tested in `12_ic_metrics.py`) with
no stated selection criterion -- worth checking in review whether that cell
was chosen because it was the best-looking one in 2025, since that is
exactly the failure mode script 50's PBO/deflated-Sharpe machinery exists
to catch, and that machinery was not applied to this particular choice.

Separately: `hedge_attrib` and `breakeven_slip` in `results/ic_findings.json`
are literal constants hardcoded in `scripts/15_export_ic.py`, not computed
by that script -- they trace back to console output from
`scripts/14_hedge_placebo.py` and `scripts/12_ic_metrics.py` that was never
wired into the JSON export, so they will silently go stale on a future
re-run even though the rest of the export recomputes live.

**Read the short-premium result as: promising in-sample, currently failing
its own out-of-sample test, not yet validated.**

## Caveat: multiple comparisons

52 scripts' worth of structures, gates, and cuts were tried against ~350
option sessions. Script 50 (`50_naked_battery.py`) runs a deflated-Sharpe
adjustment and a PBO (probability of backtest overfitting) check across the
structure grid specifically to address this -- see that script's output for
the adjusted significance of the short-premium result. Treat any single
positive cut in scripts 11-52 as a candidate for further out-of-sample
validation, not a finished result, until you've checked what fraction of
the grid it came from.
