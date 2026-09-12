#!/usr/bin/env python3
"""
49_naked_premium.py -- Naked short strangles (0.25/0.30/0.35 delta) and the ATM
straddle, 09:45 -> 15:00, same expiry mapping and DTE band as every prior test.

Why this is worth running rather than being test number fifteen: it is a
STRUCTURE change, not another signal. The condor sells exactly this strangle and
then buys wings. Removing the wings retains more credit and halves the crossings
(4 instead of 8), and cost has been the binding constraint throughout. The price
is an unbounded tail, so intraday MAX ADVERSE EXCURSION is reported as a
first-class result, not a footnote -- his standing rule makes a GTT stop
compulsory on short positions, and a no-stop backtest is not how it would trade.

QTY = 650 as instructed. The stored studies use 750 (NIFTY lot 75 in this
window), so rupee figures here are NOT directly comparable with earlier work;
points per unit are, and are the primary column.
"""
import sys
from math import erf, sqrt
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.core import STORE, load_spot
from lib.bs import implied_vol, delta, years_to_expiry
from lib.spread import half_spread_rupees, TICK
import lib.ic_costs as IC

OPT, CK = STORE / "opt2", STORE / "nk_parts"
QTY, STEP = 650, 50
ENTRY, EXIT = "09:45", "15:00"
SNAP, EST = 10, 30
DTE_LO, DTE_HI, MIN_PREM = 1.0, 8.5, 1.0
DELTAS = [0.25, 0.30, 0.35]
ncdf = np.vectorize(lambda v: 0.5 * (1 + erf(v / sqrt(2))))


def snap(day, t, back=SNAP):
    w = day[(day.ts >= t - pd.Timedelta(minutes=back)) & (day.ts <= t) & (day.v > 0)]
    if w.empty: return {}
    return w.sort_values("ts").groupby(["strike", "cp"], observed=True)["c"].last().to_dict()


def hs(day, k, cp, t0, t1, price):
    g = day[(day.strike == k) & (day.cp == cp) & (day.ts >= t0) & (day.ts <= t1)]
    if len(g) < 3: return TICK / 2.0
    v = half_spread_rupees(g.h.to_numpy(float), g.l.to_numpy(float), g.c.to_numpy(float), price)
    return float(v) if np.isfinite(v) else TICK / 2.0


def pick(px, S, T, cp, tgt):
    best, bd = None, 9e9
    for (k, c), p in px.items():
        if c != cp or p <= MIN_PREM: continue
        v = implied_vol(float(p), S, int(k), T, cp)
        if v is None: continue
        d = abs(delta(S, int(k), T, v, cp))
        if abs(d - tgt) < bd: bd, best = abs(d - tgt), (int(k), float(p), d)
    return best


def path_mae(day, legs, t0, t1, credit):
    """Worst mark-to-market of the short position between entry and exit."""
    grid = pd.date_range(t0, t1, freq="1min")
    tot = np.zeros(len(grid))
    ok = False
    for k, cp in legs:
        g = day[(day.strike == k) & (day.cp == cp)].drop_duplicates("ts").set_index("ts")
        if g.empty: continue
        s = g["c"].reindex(grid).ffill(limit=5)
        if s.notna().sum() < len(grid) * 0.5: continue
        tot = tot + s.bfill().to_numpy(float)
        ok = True
    if not ok: return np.nan, np.nan
    pnl = credit - tot                      # short: profit when total premium falls
    return float(np.nanmin(pnl)), float(np.nanmax(pnl))


def main():
    spot = load_spot(); spot["ts"] = pd.to_datetime(spot["ts"])
    g = spot.set_index("ts")
    emap = pd.read_csv(STORE / "expiry_map.csv", index_col=0)["expiry"].astype(str)
    emap.index = emap.index.astype(str)
    by_exp = {}
    for d, e in emap.items(): by_exp.setdefault(e, []).append(d)
    CK.mkdir(exist_ok=True)

    for e in sorted(by_exp):
        f, part = OPT / f"{e}.parquet", CK / f"{e}.parquet"
        if not f.exists() or part.exists(): continue
        opt = pd.read_parquet(f); opt["ts"] = pd.to_datetime(opt["ts"])
        opt["date"] = opt["ts"].dt.date.astype(str)
        rows = []
        for d in sorted(by_exp[e]):
            t0, t1 = pd.Timestamp(f"{d} {ENTRY}:00"), pd.Timestamp(f"{d} {EXIT}:00")
            if t0 not in g.index or t1 not in g.index: continue
            S, T = float(g.loc[t0, "c"]), years_to_expiry(t0, e)
            if not (DTE_LO <= T * 365 <= DTE_HI): continue
            day = opt[opt.date == d]
            pa, pb = snap(day, t0), snap(day, t1)
            if not pa or not pb: continue
            specs = [(f"strangle_{int(x*100)}", x) for x in DELTAS] + [("straddle_atm", None)]
            for name, tgt in specs:
                if tgt is None:
                    k = int(round(S / STEP) * STEP)
                    if not all((k, cp) in pa and (k, cp) in pb for cp in ("CE", "PE")): continue
                    legs = [(k, "CE"), (k, "PE")]
                    dd = {"c_d": np.nan, "p_d": np.nan, "c_k": k, "p_k": k}
                else:
                    c_, p_ = pick(pa, S, T, "CE", tgt), pick(pa, S, T, "PE", tgt)
                    if not (c_ and p_): continue
                    legs = [(c_[0], "CE"), (p_[0], "PE")]
                    if not all(L in pa and L in pb for L in legs): continue
                    dd = {"c_d": c_[2], "p_d": p_[2], "c_k": c_[0], "p_k": p_[0]}
                a = {L: float(pa[L]) for L in legs}
                b = {L: float(pb[L]) for L in legs}
                credit = sum(a.values())
                gross = credit - sum(b.values())
                slip = sum(hs(day, L[0], L[1], t0 - pd.Timedelta(minutes=EST), t0, a[L])
                           + hs(day, L[0], L[1], t1, t1 + pd.Timedelta(minutes=EST), b[L])
                           for L in legs)
                stat = IC._statutory(buy_turn=sum(b.values()) * QTY,
                                     sell_turn=credit * QTY, orders=4, on=d)
                mae, mfe = path_mae(day, legs, t0, t1, credit)
                rows.append(dict(date=d, expiry=e, spec=name, S=S, dte=T * 365,
                                 credit=credit, gross=gross, slip=slip,
                                 stat_pts=stat / QTY, mae=mae, mfe=mfe, **dd,
                                 net=(gross - slip) * QTY - stat))
        pd.DataFrame(rows).to_parquet(part, index=False)
        print(f"  {e} {len(rows)}", flush=True)

    parts = [pd.read_parquet(x) for x in sorted(CK.glob("*.parquet"))]
    parts = [x for x in parts if len(x)]
    t = pd.concat(parts, ignore_index=True)
    t["year"] = pd.to_datetime(t["date"]).dt.year
    t.to_csv(STORE / "naked_premium.csv", index=False)
    print(f"\n{t.date.nunique()} sessions, {len(t)} trades, QTY={QTY}\n")

    def sc(x):
        x = np.asarray(pd.Series(x).dropna(), float); n = len(x)
        if n < 5 or x.std(ddof=1) == 0: return n, np.nan, np.nan, np.nan
        sr = x.mean() / x.std(ddof=1)
        return n, x.mean(), sr * np.sqrt(252), sr / np.sqrt((1 + 0.5 * sr ** 2) / n)

    print("POINTS PER UNIT  (invariant to lot size -- the comparable column)")
    print(f"  {'structure':<16}{'n':>5}{'credit':>9}{'gross':>8}{'slip':>7}{'stat':>7}"
          f"{'net':>8}{'MAE':>9}{'win%':>7}")
    for s in ["strangle_25", "strangle_30", "strangle_35", "straddle_atm"]:
        x = t[t.spec == s]
        if not len(x): continue
        net_p = x.gross - x.slip - x.stat_pts
        print(f"  {s:<16}{len(x):>5}{x.credit.mean():>9.1f}{x.gross.mean():>8.2f}"
              f"{x.slip.mean():>7.2f}{x.stat_pts.mean():>7.2f}{net_p.mean():>8.2f}"
              f"{x.mae.mean():>9.1f}{100*(net_p>0).mean():>7.1f}")

    print(f"\nRUPEES at QTY={QTY}")
    print(f"  {'structure':<16}{'period':<7}{'n':>5}{'Rs/sess':>10}{'total':>12}"
          f"{'Sharpe':>8}{'t':>6}{'worst':>10}{'MAE p5':>10}")
    for s in ["strangle_25", "strangle_30", "strangle_35", "straddle_atm"]:
        for lab, m in [("all", t.spec == s), ("2025", (t.spec == s) & (t.year == 2025)),
                       ("2026", (t.spec == s) & (t.year == 2026))]:
            x = t[m]
            if len(x) < 20: continue
            n, mu, sr, tt = sc(x.net)
            print(f"  {s:<16}{lab:<7}{n:>5}{mu:>10,.0f}{x.net.sum():>12,.0f}"
                  f"{sr:>8.2f}{tt:>6.2f}{x.net.min():>10,.0f}"
                  f"{np.nanpercentile(x.mae,5)*QTY:>10,.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
