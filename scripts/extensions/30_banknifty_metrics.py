#!/usr/bin/env python3
"""
30_banknifty_metrics.py -- Score the BANKNIFTY replication.

Same rules as the NIFTY intraday test, applied to a different instrument over
eight years and several regimes.  Points per unit is the primary figure because
the BANKNIFTY lot size changed repeatedly across this span; rupee columns are
quoted at a stated 10 lots of 30.

The question is not whether BANKNIFTY is profitable -- one instrument on 239
sessions cannot answer that either -- but whether the SIGN and rough magnitude
of the NIFTY result reappear in data that had no part in producing it.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.core import STORE

# BANKNIFTY position size, matching scripts/26 (10 lots x a stated lot of 30).
# NOT the NIFTY 750 -- mixing the two silently rescales every per-unit figure.
QTY, RF = 10 * 30, 0.065
NIFTY_QTY = 10 * 75


def boot(x, n=20000, seed=3):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    if len(x) < 8:
        return (np.nan, np.nan)
    m = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def score(pnl, cap, label, per_year=252):
    pnl = np.asarray(pnl, float)
    if len(pnl) < 8:
        return None
    base = float(np.percentile(cap, 95))
    eq = np.cumsum(pnl)
    dd = float(np.max(np.maximum.accumulate(np.concatenate([[0.], eq])) - np.concatenate([[0.], eq])))
    r = pnl / base
    sd = r.std(ddof=1)
    ann = r.mean() * per_year
    lo, hi = boot(pnl)
    sr = (ann - RF) / (sd * math.sqrt(per_year)) if sd > 0 else np.nan
    # Lo (2002) standard error of the annualised Sharpe
    yrs = len(pnl) / per_year
    se = math.sqrt((1 + 0.5 * sr ** 2) / (yrs * per_year)) * math.sqrt(per_year) if np.isfinite(sr) else np.nan
    return {"period": label, "n": len(pnl), "net": round(float(pnl.sum())),
            "per_sess": round(float(pnl.mean())), "ci_lo": round(lo), "ci_hi": round(hi),
            "sharpe": round(sr, 2), "sharpe_se": round(se, 2),
            "t": round(sr / se, 2) if se and np.isfinite(se) and se > 0 else np.nan,
            "maxDD": round(dd), "win_%": round(100 * float((pnl > 0).mean()), 1)}


def main():
    t = pd.read_csv(STORE / "banknifty_condor.csv")
    t["year"] = pd.to_datetime(t["date"]).dt.year
    print(f"{len(t)} trades, {t.date.nunique()} sessions, {t.expiry.nunique()} expiries, "
          f"{t.date.min()} .. {t.date.max()}\n")

    for cfg in ("0.35/0.15", "0.25/0.1"):
        g = t[t.cfg == cfg]
        if g.empty:
            continue
        print("=" * 112)
        print(f"  BANKNIFTY  {cfg}   entry 09:45  exit 15:00  DTE 1-8   "
              f"net of modelled per-leg slippage + statutory")
        print("=" * 112)
        rows = [score(g.net.to_numpy(), g.maxloss.to_numpy(), "ALL 2018-2026")]
        for lo, hi, lab in ((2018, 2020, "2018-2020"), (2021, 2022, "2021-2022"),
                            (2023, 2024, "2023-2024"), (2025, 2026, "2025-2026")):
            s = g[(g.year >= lo) & (g.year <= hi)]
            r = score(s.net.to_numpy(), s.maxloss.to_numpy(), lab)
            if r:
                rows.append(r)
        df = pd.DataFrame([r for r in rows if r])
        print(df.to_string(index=False))
        print(f"\n  points per unit: gross {(g.pnl_pts).mean():+.2f}  "
              f"cost {((g.slip + g.stat) / QTY).mean():.2f}  net {(g.net / QTY).mean():+.2f}")
        print(f"  credit {g.credit.mean():.1f} pts, max loss {(g.maxloss/QTY).mean():.1f} pts, "
              f"mean intra-trade drawdown {g.max_dd_pts.mean():.1f} pts")
        mu, sd = g.net.mean(), g.net.std(ddof=1)
        if mu > 0:
            print(f"  sessions needed to detect this mean at 80% power: "
                  f"{(2.8 * sd / mu) ** 2:,.0f}")
        else:
            print("  mean is negative; no power figure applies")
        print()

    print("=" * 112)
    print("  CROSS-INSTRUMENT COMPARISON, 0.35/0.15, net points per session")
    print("=" * 112)
    ni = pd.read_csv(STORE / "intraday_wings.csv")
    ni["year"] = pd.to_datetime(ni["date"]).dt.year
    ni = ni[ni.cfg == "0.35/0.15"]
    bn = t[t.cfg == "0.35/0.15"]
    print(f"  {'':<34}{'sessions':>10}{'net pts/session':>18}{'win %':>9}{'Sharpe':>9}")
    for lab, s, col in (("NIFTY 2025 (in-sample)", ni[ni.year == 2025], "net"),
                        ("NIFTY 2026 (held out)", ni[ni.year == 2026], "net"),
                        ("BANKNIFTY 2018-2026 (all)", bn, "net"),
                        ("BANKNIFTY 2025-2026 only", bn[bn.year >= 2025], "net")):
        if s.empty:
            continue
        r = score(s[col].to_numpy(), s.maxloss.to_numpy(), lab)
        q = NIFTY_QTY if lab.startswith("NIFTY") else QTY
        print(f"  {lab:<34}{r['n']:>10}{s[col].mean()/q:>18.2f}{r['win_%']:>9.1f}{r['sharpe']:>9.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
