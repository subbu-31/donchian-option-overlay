# Donchian + option overlay: findings

One-page synthesis of `results/findings.json` and `results/ic_findings.json`.
Full numbers live in those files and the per-test CSVs in `results/`.

## Data

570 NIFTY 50 spot sessions (2024-01-01 to 2026-05-05), 353 sessions with
matched option coverage across 73 weekly expiries, 54,275 near-the-money
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
  costs 3.79 points, ~1.54% of premium. Even zeroing out costs would not
  flip the sign of the mean move.

## The one thing that survives: width as a range forecast

Channel width relative to the same bar-of-day over the prior 20 sessions
correlates with the realised forward range at 0.542 (2025) / 0.465 (2026
holdout) -- a real, out-of-sample-stable relationship. It's a sizing input,
not a directional edge: the option market already prices the volatility
information in the channel, so buying premium conditioned on width is a
wash (`scripts/core/06`, `07`), but sizing positions off it (wider channel
-> larger expected range -> smaller position for constant risk) holds up.

## The extension that looks promising but doesn't survive its own holdout

`scripts/iron_condor/` and `scripts/extensions/` drop the long-premium
thesis and test short-premium structures instead: a delta-selected iron
condor with a Donchian buy-stop hedge, plus a wider battery (weekly holds,
regime decomposition, term structure/skew, vol-regime gating,
stress tests, walk-forward, naked premium). The blind condor and its hedged
variants show a strong cumulative curve over 2025-2026, and the rotation
placebo confirms the edge isn't from a leaked gate (leaked-label
attribution -9.21 pts vs. clean +0.97 pts).

Split by year, the picture reverses. Strongly positive in 2025, uniformly
negative in the 2026 holdout at every hedge multiplier -- and it gets worse
in absolute terms as the hedge multiplier goes up, which is what added
gross exposure looks like, not what a hedge looks like:

| Arm | 2025 | 2026 YTD (holdout) |
|---|---|---|
| IC blind | +172,572 | -128,239 |
| + hedge 1x | +440,506 | -136,574 |
| + hedge 2x | +726,092 | -138,490 |
| + hedge 3x | +1,011,681 | -140,405 |

Not a slippage artifact: the blind condor is already Sharpe -1.22 in the
2026 holdout at zero assumed slippage. The `0.35/0.15` config and `optg`
hedge style behind this export were picked from a ~30-way grid (3 delta
configs x 10 arm variants in `scripts/iron_condor/12_ic_metrics.py`) with no
stated selection rule -- worth checking whether that cell just happened to
be the best-looking one in 2025, since `50_naked_battery.py`'s
PBO/deflated-Sharpe check exists for exactly that failure mode and wasn't
run against this choice.

Read the short-premium result as promising in-sample, not yet validated
out of sample.

## Caveat: multiple comparisons

47 scripts' worth of structures, gates, and cuts were tried against ~350
option sessions. `scripts/extensions/50_naked_battery.py` runs a deflated-Sharpe adjustment
and a PBO check across the structure grid to address exactly this -- see
its output for the adjusted significance of the short-premium result.
Treat any single positive cut here as a candidate for further validation,
not a finished result, until you've checked what fraction of the grid it
came from.
