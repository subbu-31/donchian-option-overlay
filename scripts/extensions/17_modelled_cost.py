#!/usr/bin/env python3
"""
17_modelled_cost.py -- The backtest with slippage estimated rather than assumed,
and the trade redesigned around what that cost structure implies.

Part 1  every leg is charged its OWN estimated half-spread, twice (in and out),
        instead of one flat number applied to all four legs.  The wings turn
        out to be cheap to cross and the short legs expensive, which a flat
        assumption gets exactly backwards in proportional terms.

Part 2  cost attribution -- where the gross edge actually goes.

Part 3  the redesign.  Crossing cost is close to FIXED per session while the
        edge is VARIABLE per session, so the response is to spend the fixed
        cost only on sessions where the edge is expected to be there.  The
        Donchian channel width at 10:15 is a validated forecast of the day's
        realised range (r = 0.54 / 0.46, replicated), and a short-premium book
        wants quiet days.  Thresholds are fitted on 2025 and applied verbatim
        to the 2026 holdout.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.core import STORE
import lib.ic_costs as IC

LOTS, LOT_SIZE = 10, 75
QTY = LOTS * LOT_SIZE
RF = 0.065
ROLES = {"short_call": "sc", "short_put": "sp", "long_call": "lc", "long_put": "lp"}


def statutory(entry, exit_, qty, orders=8, on=None):
    sell = (entry["sc"] + entry["sp"] + exit_["lc"] + exit_["lp"]) * qty
    buy = (entry["lc"] + entry["lp"] + exit_["sc"] + exit_["sp"]) * qty
    return IC._statutory(buy, sell, orders, on=on)


def score(pnl, cap, label, extra=None):
    pnl = np.asarray(pnl, float)
    if len(pnl) < 10:
        return None
    base = float(np.percentile(cap, 95))
    eq = np.cumsum(pnl)
    dd = float(np.max(np.maximum.accumulate(np.concatenate([[0.], eq])) - np.concatenate([[0.], eq])))
    r = pnl / base
    sd = r.std(ddof=1)
    dn = r[r < 0]
    dsd = dn.std(ddof=1) if len(dn) > 1 else np.nan
    ann = r.mean() * 252
    out = {"arm": label, "n": len(pnl), "net": round(float(pnl.sum())),
           "capital": round(base), "ret_%": round(float(pnl.sum()) / base * 100, 1),
           "sharpe": round((ann - RF) / (sd * np.sqrt(252)), 2) if sd > 0 else np.nan,
           "sortino": round((ann - RF) / (dsd * np.sqrt(252)), 2) if dsd and dsd > 0 else np.nan,
           "maxDD": round(dd), "DD_%": round(dd / base * 100, 1),
           "ret/DD": round(float(pnl.sum()) / dd, 2) if dd > 0 else np.nan,
           "win_%": round(float((pnl > 0).mean()) * 100, 1),
           "day": round(float(pnl.mean()))}
    if extra:
        out.update(extra)
    return out


def main():
    df = pd.read_csv(STORE / "ic_donchian_daily.csv")
    df = df[df.cfg == "0.35/0.15"].copy()
    df["year"] = pd.to_datetime(df["date"]).dt.year
    sp = pd.read_csv(STORE / "spread_estimates.csv")

    # per-leg half spread, wide; fall back to that role's median where missing
    piv = sp.pivot_table(index="date", columns="role", values="half_spread", aggfunc="first")
    med = sp.groupby("role")["half_spread"].median()
    for role in ROLES:
        if role not in piv:
            piv[role] = np.nan
        piv[role] = piv[role].fillna(med[role])
    df = df.merge(piv.rename(columns=ROLES).add_prefix("hs_"), on="date", how="left")
    for r in ROLES.values():
        df[f"hs_{r}"] = df[f"hs_{r}"].fillna(df[f"hs_{r}"].median())

    # width state at 10:15 from the daily-arms run
    wa = pd.read_csv(STORE / "daily_arms.csv")[["date", "width_rel"]]
    df = df.merge(wa, on="date", how="left")

    # ---------------- Part 1: modelled cost ----------------
    df["slip_ic"] = 2 * QTY * (df.hs_sc + df.hs_sp + df.hs_lc + df.hs_lp)
    df["stat_ic"] = [statutory({"sc": r.entry_sc, "sp": r.entry_sp, "lc": r.entry_lc, "lp": r.entry_lp},
                               {"sc": r.exit_sc, "sp": r.exit_sp, "lc": r.exit_lc, "lp": r.exit_lp},
                               QTY, on=r.date) for r in df.itertuples()]
    df["ic_gross"] = df.ic_pnl * QTY
    df["ic_net"] = df.ic_gross - df.slip_ic - df.stat_ic
    df["maxloss"] = (np.maximum(df.call_width, df.put_width) - df.credit).clip(lower=0) * QTY

    def hedge_cols(mult):
        hp = df["hoptg_pnl"].fillna(0) * QTY * mult
        prem = df["hoptg_prem"].fillna(0) * QTY * mult
        slip = np.zeros(len(df))
        stat = np.zeros(len(df))
        for i, r in enumerate(df.itertuples()):
            for leg, hs in (("sc", r.hs_sc), ("sp", r.hs_sp)):
                e = getattr(r, f"hoptg_{leg}_ent", np.nan)
                x = getattr(r, f"hoptg_{leg}_ext", np.nan)
                if np.isfinite(e) and np.isfinite(x):
                    q = QTY * mult
                    slip[i] += 2 * q * hs
                    stat[i] += IC._statutory(float(e) * q, float(x) * q, 2, on=r.date)
        return hp, prem, slip, stat

    print("=" * 104)
    print("PART 1  --  slippage estimated per leg from its own 1-minute path, not assumed")
    print("=" * 104)
    print(f"  modelled crossing cost per session : Rs {df.slip_ic.mean():,.0f} mean, "
          f"Rs {df.slip_ic.median():,.0f} median   ({df.slip_ic.mean()/QTY/8:.4f} pts/leg/side)")
    print(f"  statutory cost per session         : Rs {df.stat_ic.mean():,.0f}")
    print(f"  short legs cost {sp[sp.role.str.startswith('short')].half_spread.median():.3f} "
          f"per crossing, wings {sp[sp.role.str.startswith('long')].half_spread.median():.3f} "
          f"-- a flat assumption gets the split backwards")

    rows = []
    for pname, yr in (("2025 in-sample", 2025), ("2026 holdout", 2026)):
        g = df[df.year == yr]
        rows.append(score(g.ic_net.to_numpy(), g.maxloss.to_numpy(), "IC blind",
                          {"period": pname}))
        for m in (1, 2, 3):
            hp, prem, slip, stat = hedge_cols(m)
            tot = df.ic_net + hp - slip - stat
            cap = df.maxloss + prem
            mask = (df.year == yr).to_numpy()
            rows.append(score(tot[mask].to_numpy(), cap[mask].to_numpy(),
                              f"IC + hedge {m}x", {"period": pname}))
    res = pd.DataFrame([r for r in rows if r])
    cols = ["period", "arm", "n", "net", "capital", "ret_%", "sharpe", "sortino",
            "maxDD", "DD_%", "ret/DD", "win_%", "day"]
    print("\n  --- all arms, modelled slippage ---")
    print(res[cols].to_string(index=False))

    # ---------------- Part 2: attribution ----------------
    print("\n" + "=" * 104)
    print("PART 2  --  where the gross edge goes, per session (Rs, 10 lots)")
    print("=" * 104)
    for pname, yr in (("2025", 2025), ("2026", 2026)):
        g = df[df.year == yr]
        hp, prem, slip, stat = hedge_cols(2)
        m = (df.year == yr).to_numpy()
        print(f"\n  {pname}   ({len(g)} sessions)")
        print(f"    condor gross                {g.ic_gross.mean():>10,.0f}")
        print(f"    - crossing cost (modelled)  {-g.slip_ic.mean():>10,.0f}"
              f"   {100*g.slip_ic.mean()/abs(g.ic_gross.mean()):>6.1f}% of gross")
        print(f"    - statutory                 {-g.stat_ic.mean():>10,.0f}"
              f"   {100*g.stat_ic.mean()/abs(g.ic_gross.mean()):>6.1f}%")
        print(f"    = condor net                {g.ic_net.mean():>10,.0f}")
        print(f"    + hedge gross (2x)          {hp[m].mean():>10,.0f}")
        print(f"    - hedge crossing+statutory  {-(slip[m]+stat[m]).mean():>10,.0f}")
        print(f"    = total net                 {(df.ic_net+hp-slip-stat)[m].mean():>10,.0f}")

    # ---------------- Part 3: spend the fixed cost only where the edge is ----------------
    print("\n" + "=" * 104)
    print("PART 3  --  cost is fixed per session, edge is not: gate on the width state")
    print("=" * 104)
    is_ = df[(df.year == 2025) & df.width_rel.notna()]
    qs = {f"q{int(q*100)}": float(is_.width_rel.quantile(q)) for q in (0.2, 0.4, 0.6, 0.8)}
    print(f"  width_rel thresholds fitted on 2025 only: "
          f"{ {k: round(v,3) for k, v in qs.items()} }")
    hp, prem, slip, stat = hedge_cols(2)
    df["tot_net"] = df.ic_net + hp - slip - stat
    df["tot_cap"] = df.maxloss + prem
    out = []
    for label, thr in [("trade all sessions", np.inf)] + [(f"only width_rel <= {k} ({v:.3f})", v)
                                                          for k, v in qs.items()]:
        for pname, yr in (("2025 in-sample", 2025), ("2026 holdout", 2026)):
            g = df[(df.year == yr) & (df.width_rel <= thr)]
            if len(g) < 15:
                continue
            for col, cap, nm in (("ic_net", "maxloss", "IC blind"),
                                 ("tot_net", "tot_cap", "IC + hedge 2x")):
                s = score(g[col].to_numpy(), g[cap].to_numpy(), nm,
                          {"period": pname, "filter": label,
                           "share_%": round(100 * len(g) / (df.year == yr).sum(), 0)})
                if s:
                    out.append(s)
    f = pd.DataFrame(out)
    f.to_csv(STORE / "cost_modelled_results.csv", index=False)
    c2 = ["filter", "share_%", "arm", "n", "net", "ret_%", "sharpe", "sortino", "DD_%", "ret/DD", "win_%", "day"]
    for pname in ("2025 in-sample", "2026 holdout"):
        print(f"\n  --- {pname} ---")
        print(f[f.period == pname][c2].to_string(index=False))
    df.to_csv(STORE / "ic_modelled_daily.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
