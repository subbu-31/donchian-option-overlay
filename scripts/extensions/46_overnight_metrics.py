#!/usr/bin/env python3
"""46_overnight_metrics.py -- Score the overnight window against the day session."""
import sys
from math import erf, sqrt
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.core import STORE
QTY = 750
ncdf = lambda x: 0.5 * (1 + erf(x / sqrt(2)))
RNG = np.random.default_rng(20260905)


def sc(x):
    x = np.asarray(pd.Series(x).dropna(), float); n = len(x)
    if n < 5 or x.std(ddof=1) == 0:
        return n, np.nan, np.nan, np.nan, np.nan
    sr = x.mean() / x.std(ddof=1); t = sr / np.sqrt((1 + 0.5 * sr ** 2) / n)
    return n, x.mean(), sr * np.sqrt(252), t, 2 * (1 - ncdf(abs(t)))


def line(tag, x, unit="Rs"):
    n, mu, sr, t, p = sc(x)
    print(f"    {tag:<26} n={n:>4}  {unit} {mu:>9,.1f}  SR {sr:>6.2f}  t {t:>5.2f}  p {p:>5.3f}")


d = pd.read_csv(STORE / "overnight_panel.csv")
d["date"] = pd.to_datetime(d["date"]); d["nx"] = pd.to_datetime(d["next"])
d["weekend"] = (d.nx - d.date).dt.days > 1
d["year"] = d.date.dt.year

print(f"{len(d)} overnight windows  {d.date.min().date()} .. {d.date.max().date()}")
print(f"  weekday holds {int((~d.weekend).sum())} (mean {d.loc[~d.weekend,'h_on'].mean():.1f} h)"
      f"   weekend/holiday holds {int(d.weekend.sum())} (mean {d.loc[d.weekend,'h_on'].mean():.1f} h)\n")

print("1) RAW PREMIUM CAPTURED, points per window")
for tag, a, b in [("short ATM straddle", "strad_on", "strad_day"),
                  ("short 0.25/0.10 condor", "condor_on", "condor_day")]:
    print(f"  {tag}")
    line("overnight", d[a], "pts"); line("day session", d[b], "pts")

print("\n2) PER UNIT OF TIME -- the convention-free comparison")
for a, b, nm in [("strad_on", "strad_day", "straddle"), ("condor_on", "condor_day", "condor")]:
    on_h = (d[a] / d.h_on).mean(); dy_h = (d[b] / d.h_day).mean()
    on_t = (d[a] / (d.h_on / 24)).mean(); dy_t = (d[b] / (d.h_day / 24)).mean()
    print(f"  {nm:<9} per CLOCK hour  overnight {on_h:>7.3f} pts   day {dy_h:>7.3f} pts"
          f"   ratio {on_h/dy_h if dy_h else float('nan'):>5.2f}x")
    print(f"  {nm:<9} per CALENDAR day overnight {on_t:>7.2f} pts   day {dy_t:>7.2f} pts")

print("\n3) IS THE OVERNIGHT WINDOW OVERPRICED? (secondary read)")
r = (d.real_move_on / d.imp_move_on).dropna()
print(f"  realised / implied overnight move: mean {r.mean():.3f}  median {r.median():.3f}")
print(f"  unbiased benchmark sqrt(2/pi) = 0.798")
print(f"  intraday benchmark from the width study: 0.286 (narrow) .. 0.650 (wide), mean ~0.44")
print(f"  -> overnight sits {'ABOVE' if r.mean()>0.44 else 'BELOW'} the intraday band, i.e. "
      f"{'LESS' if r.mean()>0.44 else 'MORE'} overpriced than the day session")
print(f"  realised move: overnight {d.real_move_on.mean():.1f} pts vs day {d.real_move_day.mean():.1f} pts"
      f"  ({100*d.real_move_on.mean()/d.real_move_day.mean():.0f}% of the day's move)")

print("\n4) NET OF FULL COSTS, rupees at 10 lots x 75")
for tag, c in [("overnight condor", "condor_net_on"), ("day condor", "condor_net_day")]:
    print(f"  {tag}")
    line("all windows", d[c])
    for lab, m in [("2025", d.year == 2025), ("2026", d.year == 2026),
                   ("weekday hold", ~d.weekend), ("weekend hold", d.weekend)]:
        if m.sum() >= 15:
            line(f"  {lab}", d.loc[m, c])

print("\n5) COST DECOMPOSITION, points per condor round trip")
for tag in ("on", "day"):
    sl, st = d[f"condor_slip_{tag}"], d[f"stat_{tag}"] / QTY
    print(f"  {tag:<4} gross {d[f'condor_{tag}'].mean():>6.2f}   slippage {sl.mean():>5.2f}"
          f"   statutory {st.mean():>5.2f}   all-in cost {sl.mean()+st.mean():>5.2f} pts"
          f"   -> net {d[f'condor_{tag}'].mean()-sl.mean()-st.mean():>6.2f} pts")

print("\n6) THE TAIL -- what a short overnight condor is actually short")
s = d.condor_net_on.dropna().sort_values()
print(f"  worst 5 overnight: {', '.join(f'{v:,.0f}' for v in s.head(5))}")
print(f"  5th pct {np.percentile(s,5):,.0f}   win% {100*(s>0).mean():.1f}"
      f"   median {s.median():,.0f}   mean {s.mean():,.0f}")
sd_ = d.condor_net_day.dropna().sort_values()
print(f"  worst 5 day:       {', '.join(f'{v:,.0f}' for v in sd_.head(5))}")
print(f"  gap |%|: mean {d.gap_pct.abs().mean():.3f}  p95 {d.gap_pct.abs().quantile(.95):.3f}"
      f"  max {d.gap_pct.abs().max():.3f}   by year "
      + "  ".join(f"{y}:{g.gap_pct.abs().mean():.3f}" for y, g in d.groupby('year')))

print("\n7) BOOTSTRAP on the overnight condor")
x = d.condor_net_on.dropna().to_numpy(float); n = len(x)
bs = np.array([np.mean(RNG.choice(x, n, replace=True)) for _ in range(10000)])
print(f"  95% CI [{np.percentile(bs,2.5):,.0f}, {np.percentile(bs,97.5):,.0f}]"
      f"   P(mean>0) = {(bs>0).mean():.3f}")
