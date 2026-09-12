#!/usr/bin/env python3
"""
21_weekly_metrics.py -- Score the 6-DTE-to-0-DTE condor against the intraday
version, with small-sample discipline.

Annualisation is by trade frequency: ~52 weekly trades a year against ~252
intraday ones, so a weekly Sharpe uses sqrt(52).  Because the sample is ~50
in-sample and ~16 held out, every mean carries a bootstrap interval and the
Sharpe should be read as an estimate with wide error, not a number.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.core import STORE

RF = 0.065
QTY = 750


def boot(x, n=5000, seed=9):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    if len(x) < 6:
        return (np.nan, np.nan)
    m = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def score(pnl, cap, per_year, label, extra=None):
    pnl = np.asarray(pnl, float)
    if len(pnl) < 6:
        return None
    base = float(np.percentile(cap, 95))
    eq = np.cumsum(pnl)
    dd = float(np.max(np.maximum.accumulate(np.concatenate([[0.], eq])) - np.concatenate([[0.], eq])))
    r = pnl / base
    sd = r.std(ddof=1)
    dn = r[r < 0]
    dsd = dn.std(ddof=1) if len(dn) > 1 else np.nan
    ann = r.mean() * per_year
    lo, hi = boot(pnl)
    out = {"arm": label, "n": len(pnl), "net": round(float(pnl.sum())),
           "per_trade": round(float(pnl.mean())),
           "ci_lo": round(lo), "ci_hi": round(hi),
           "capital": round(base), "ret_%": round(float(pnl.sum()) / base * 100, 1),
           "ann_%": round(ann * 100, 1),
           "sharpe": round((ann - RF) / (sd * np.sqrt(per_year)), 2) if sd > 0 else np.nan,
           "sortino": round((ann - RF) / (dsd * np.sqrt(per_year)), 2) if dsd and dsd > 0 else np.nan,
           "maxDD": round(dd), "DD_%": round(dd / base * 100, 1),
           "win_%": round(float((pnl > 0).mean()) * 100, 1),
           "worst": round(float(pnl.min()))}
    if extra:
        out.update(extra)
    return out


def main():
    w = pd.read_csv(STORE / "weekly_hold.csv")
    w = w[w.year.isin([2025, 2026])]
    rows = []
    for cfg, gc in w.groupby("cfg"):
        for pn, yr in (("2025 in-sample", 2025), ("2026 holdout", 2026)):
            g = gc[gc.year == yr]
            if len(g) < 6:
                continue
            base = g.maxloss.to_numpy()
            r = score(g.net.to_numpy(), base, 52, "condor only",
                      {"cfg": cfg, "period": pn})
            if r:
                rows.append(r)
            for m in (1, 2, 3):
                tot = g.net + g.h_pnl * QTY * m - g.h_cost * m
                # hedge triggers are sequential, not simultaneous: at most one
                # bought leg per short leg is open at a time, so the capital the
                # overlay actually ties up is the average premium per trigger
                # across the two legs -- not the sum of every trigger in the week
                per_trig = (g.h_prem / g.h_n.clip(lower=1)).to_numpy()
                cap = base + per_trig * 2 * QTY * m
                r = score(tot.to_numpy(), cap, 52, f"+ hedge {m}x", {"cfg": cfg, "period": pn})
                if r:
                    rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(STORE / "weekly_metrics.csv", index=False)

    cols = ["arm", "n", "net", "per_trade", "ci_lo", "ci_hi", "capital", "ret_%",
            "ann_%", "sharpe", "sortino", "maxDD", "DD_%", "win_%", "worst"]
    for cfg in sorted(res.cfg.unique()):
        for pn in ("2025 in-sample", "2026 holdout"):
            s = res[(res.cfg == cfg) & (res.period == pn)]
            if s.empty:
                continue
            print("\n" + "=" * 118)
            print(f"  WEEKLY HOLD  6 DTE 09:45 -> 0 DTE 14:30   |   {cfg}   |   {pn}   "
                  f"|   10 lots x 75   |   net of modelled slippage + statutory")
            print("=" * 118)
            print(s[cols].to_string(index=False))

    print("\n" + "=" * 118)
    print("  RISK OF THE WEEK-LONG HOLD  (intra-trade, in index points -- this is the "
          "number the no-0-DTE rule exists for)")
    print("=" * 118)
    d = w[w.cfg == "0.25/0.1"]
    print(f"  {'':<16}{'n':>5}{'credit':>9}{'exit pnl':>10}{'max profit':>12}"
          f"{'max DD':>10}{'DD > credit':>13}{'worst DD':>11}")
    for pn, yr in (("2025 in-sample", 2025), ("2026 holdout", 2026)):
        g = d[d.year == yr]
        if g.empty:
            continue
        print(f"  {pn:<16}{len(g):>5}{g.credit.mean():>9.1f}{g.exit_pnl_pts.mean():>10.1f}"
              f"{g.max_profit_pts.mean():>12.1f}{g.max_dd_pts.mean():>10.1f}"
              f"{100*(g.max_dd_pts.abs() > g.credit).mean():>12.0f}%{g.max_dd_pts.min():>11.1f}")
    print("\n  'DD > credit' = share of trades whose worst intraweek mark-to-market loss")
    print("  exceeded the entire credit collected at entry.")

    print("\n" + "=" * 118)
    print("  COMPARISON  --  same capital basis, per SESSION of capital tied up")
    print("=" * 118)
    daily = pd.read_csv(STORE / "ic_modelled_daily.csv")
    daily["year"] = pd.to_datetime(daily["date"]).dt.year
    print(f"  {'variant':<42}{'trades':>8}{'net/trade':>12}{'sessions':>10}{'net/session':>13}{'sharpe':>9}")
    for cfg_w, cfg_d in (("0.35/0.15", "0.35/0.15"),):
        for pn, yr in (("2025", 2025), ("2026", 2026)):
            gw = w[(w.cfg == cfg_w) & (w.year == yr)]
            gd = daily[(daily.cfg == cfg_d) & (daily.year == yr)]
            if gw.empty or gd.empty:
                continue
            sw = score(gw.net.to_numpy(), gw.maxloss.to_numpy(), 52, "w")
            sd_ = score(gd.ic_net.to_numpy(), gd.maxloss.to_numpy(), 252, "d")
            print(f"  {pn+' weekly 6DTE->0DTE ('+cfg_w+')':<42}{sw['n']:>8}"
                  f"{sw['per_trade']:>12,.0f}{gw.sessions_held.mean():>10.1f}"
                  f"{sw['per_trade']/gw.sessions_held.mean():>13,.0f}{sw['sharpe']:>9.2f}")
            print(f"  {pn+' intraday 09:45->15:00 ('+cfg_d+')':<42}{sd_['n']:>8}"
                  f"{sd_['per_trade']:>12,.0f}{1.0:>10.1f}{sd_['per_trade']:>13,.0f}{sd_['sharpe']:>9.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
