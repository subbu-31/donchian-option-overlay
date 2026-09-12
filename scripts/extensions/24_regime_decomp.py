#!/usr/bin/env python3
"""
24_regime_decomp.py -- The paradox and its resolution.

The variance risk premium did not collapse in 2026: mean ATM VRP rose from
+5.98 to +7.90 annualised points and stayed positive on 96% of sessions.  Yet
every condor variant lost money there.  A short-gamma book is not paid the mean
of the distribution, it is short its tail, so the mean can improve while the
book loses.

Three tests:

  1  STANDARDISED MOVE.  Scale each session's realised 09:45-15:00 move by the
     move the 09:45 ATM implied vol priced for that same window.  z = 1 means
     the index moved exactly what the option market charged for.  If 2026's
     mean |z| is fine but its tail is fatter, that is the whole story.

  2  P&L CONCENTRATION.  How much of each year's condor P&L comes from its
     worst five sessions.  A drift problem is spread out; a tail problem is not.

  3  REGIME DECOMPOSITION.  Sessions are labelled Quiet / Normal / Stressed by
     TRAILING 20-session realised vol -- causal, so the label is knowable at
     the open.  Cut points are fitted on 2024-2025 and applied to 2026.  Then
     2026's shortfall is split into a MIX effect (more stressed sessions) and a
     PERFORMANCE effect (the same regime paying worse), the standard
     counterfactual decomposition.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.core import STORE

HOURS = 5.25 / 24.0 / 365.0       # 09:45 -> 15:00 as a fraction of a year


def main():
    f = pd.read_csv(STORE / "regime_features.csv")
    w = pd.read_csv(STORE / "intraday_wings.csv")
    w = w[w.cfg == "0.35/0.15"][["date", "net", "gross", "credit", "ic_maxdd"]]
    f = f.merge(w, on="date", how="left")
    f["year"] = pd.to_datetime(f["date"]).dt.year

    # ---------- 1. standardised move ----------
    f["move_pct"] = (f["close"] / f["spot_945"] - 1).abs() * 100
    f["implied_move_pct"] = f["atm_iv"] * np.sqrt(HOURS)   # atm_iv already in %
    f["z"] = f["move_pct"] / f["implied_move_pct"]
    print("=" * 100)
    print("1  STANDARDISED MOVE   z = realised 09:45-15:00 move / move the 09:45 ATM IV priced")
    print("=" * 100)
    print(f"  {'year':<6}{'n':>5}{'mean z':>9}{'median z':>10}{'sd z':>8}{'kurtosis':>10}"
          f"{'P(z>1)':>9}{'P(z>1.5)':>10}{'P(z>2)':>9}{'max z':>8}")
    for y in (2025, 2026):
        g = f[(f.year == y) & f.z.notna()]
        z = g["z"]
        print(f"  {y:<6}{len(z):>5}{z.mean():>9.3f}{z.median():>10.3f}{z.std():>8.3f}"
              f"{z.kurt():>10.2f}{100*(z>1).mean():>8.1f}%{100*(z>1.5).mean():>9.1f}%"
              f"{100*(z>2).mean():>8.1f}%{z.max():>8.2f}")
    print("\n  A lower mean z with a fatter right tail is exactly the shape that pays the")
    print("  average seller and ruins the one who is short the wings.")

    # ---------- 2. P&L concentration ----------
    print("\n" + "=" * 100)
    print("2  P&L CONCENTRATION   condor 0.35/0.15, net of modelled cost")
    print("=" * 100)
    print(f"  {'year':<6}{'n':>5}{'total':>12}{'worst 5 sum':>14}{'worst5 / total':>16}"
          f"{'median day':>13}{'mean day':>11}{'win %':>8}")
    for y in (2025, 2026):
        g = f[(f.year == y) & f.net.notna()].sort_values("net")
        tot, w5 = g.net.sum(), g.net.head(5).sum()
        print(f"  {y:<6}{len(g):>5}{tot:>12,.0f}{w5:>14,.0f}"
              f"{(w5/tot if tot else np.nan):>15.2f}x{g.net.median():>13,.0f}"
              f"{g.net.mean():>11,.0f}{100*(g.net>0).mean():>7.1f}%")
    g26 = f[(f.year == 2026) & f.net.notna()]
    print(f"\n  2026 without its worst 5 sessions: "
          f"Rs {g26.sort_values('net').iloc[5:].net.sum():,.0f} "
          f"({g26.sort_values('net').iloc[5:].net.mean():,.0f}/session)")

    # ---------- 3. regime decomposition ----------
    f = f.sort_values("date").reset_index(drop=True)
    f["rv20"] = f["rv"].shift(1).rolling(20).mean()          # causal
    fit = f[(f.year <= 2025) & f.rv20.notna()]
    q1, q2 = fit["rv20"].quantile([1 / 3, 2 / 3])
    f["regime"] = pd.cut(f["rv20"], [-np.inf, q1, q2, np.inf],
                         labels=["Quiet", "Normal", "Stressed"])
    print("\n" + "=" * 100)
    print(f"3  REGIME DECOMPOSITION   trailing-20-session realised vol, cut points fitted on "
          f"2024-25 ({q1:.2f} / {q2:.2f})")
    print("=" * 100)
    tab = {}
    print(f"  {'regime':<10}{'2025 n':>9}{'2025 share':>12}{'2025 Rs/day':>14}"
          f"{'2026 n':>9}{'2026 share':>12}{'2026 Rs/day':>14}")
    for r in ["Quiet", "Normal", "Stressed"]:
        row = []
        for y in (2025, 2026):
            g = f[(f.year == y) & (f.regime == r) & f.net.notna()]
            tot = f[(f.year == y) & f.net.notna()]
            row += [len(g), len(g) / len(tot) if len(tot) else np.nan,
                    g.net.mean() if len(g) else np.nan]
        tab[r] = row
        print(f"  {r:<10}{row[0]:>9}{100*row[1]:>11.1f}%{row[2]:>14,.0f}"
              f"{row[3]:>9}{100*row[4]:>11.1f}%{row[5]:>14,.0f}")

    print("\n" + "=" * 100)
    print("   the sessions that did the damage")
    print("=" * 100)
    print(f"  {'date':<12}{'regime':<10}{'net Rs':>11}{'move %':>9}{'z':>7}"
          f"{'rv':>7}{'atm_iv':>8}{'gap %':>8}{'credit':>8}")
    for y in (2025, 2026):
        g = f[(f.year == y) & f.net.notna()].sort_values("net").head(5)
        for _, r in g.iterrows():
            print(f"  {r['date']:<12}{str(r['regime']):<10}{r['net']:>11,.0f}"
                  f"{r['move_pct']:>9.2f}{r['z']:>7.2f}{r['rv']:>7.1f}{r['atm_iv']:>8.1f}"
                  f"{r['gap_pct']:>8.2f}{r['credit']:>8.1f}")
        print()

    m25 = f[(f.year == 2025) & f.net.notna()].net.mean()
    m26 = f[(f.year == 2026) & f.net.notna()].net.mean()
    mix = sum((tab[r][4] - tab[r][1]) * tab[r][2] for r in tab
              if np.isfinite(tab[r][2]) and np.isfinite(tab[r][4]))
    perf = sum(tab[r][4] * (tab[r][5] - tab[r][2]) for r in tab
               if np.isfinite(tab[r][2]) and np.isfinite(tab[r][5]))
    print(f"\n  2025 mean Rs/session {m25:>10,.0f}")
    print(f"  2026 mean Rs/session {m26:>10,.0f}")
    print(f"  total change         {m26-m25:>10,.0f}")
    print(f"    of which MIX       {mix:>10,.0f}   (2026 had a different balance of regimes)")
    print(f"    of which PERFORM   {perf:>10,.0f}   (the same regime paid differently)")
    print(f"    unexplained        {m26-m25-mix-perf:>10,.0f}")
    f.to_csv(STORE / "regime_decomp.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
