#!/usr/bin/env python3
"""
36_skew_panel.py -- Build the per-session implied-skew panel.

At 09:45 on every session, for the nearest weekly expiry:
  * back out IV for every strike with a live quote (bisection, entry tick only)
  * compute BS delta at that IV
  * interpolate the IV surface onto fixed deltas: 0.25/0.10 call, -0.25/-0.10 put
  * RR25 = IV(-0.25 put) - IV(0.25 call)        risk reversal, the skew level
    BF25 = mean(IV25c, IV25p) - IV_atm          butterfly, the smile curvature
  * ATM IV from the ATM straddle (CE and PE averaged)

Outcomes measured 09:45 -> 15:00 on the same session:
  ret, absr, down = min(ret,0), up = max(ret,0)
  rv   realised vol of 1-min log returns, annualised
  rskw realised skewness of 1-min log returns

Everything on the left of the regression is known at 09:45.  Nothing on the
right uses a later tick.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.core import STORE, load_spot
from lib.bs import implied_vol, delta, years_to_expiry

OPT = STORE / "opt2"
CK = STORE / "skew_parts"
ENTRY, EXIT = "09:45", "15:00"
LOOKBACK_MIN = 10          # accept the last quote within 10 min before entry
DTE_LO, DTE_HI = 0.20, 8.5
STEP = 50
TARGETS = [0.10, 0.25, 0.40]


def surface_iv(rows, S, T, cp):
    """rows: list of (K, price).  Returns (deltas, ivs) sorted by |delta| asc."""
    out = []
    for K, px in rows:
        if not np.isfinite(px) or px <= 1.0:
            continue
        iv = implied_vol(float(px), S, K, T, cp)
        if iv is None or not (0.02 < iv < 2.5):
            continue
        dl = delta(S, K, T, iv, cp)
        if not np.isfinite(dl):
            continue
        a = abs(dl)
        if not (0.02 < a < 0.80):
            continue
        out.append((a, iv))
    if len(out) < 4:
        return None, None
    out.sort()
    a = np.array([x[0] for x in out])
    v = np.array([x[1] for x in out])
    keep = np.concatenate([[True], np.diff(a) > 1e-6])
    return a[keep], v[keep]


def at_delta(a, v, tgt):
    if a is None or tgt < a[0] or tgt > a[-1]:
        return np.nan
    return float(np.interp(tgt, a, v))


def main():
    spot = load_spot()
    emap = pd.read_csv(STORE / "expiry_map.csv", index_col=0)["expiry"].astype(str)
    emap.index = emap.index.astype(str)
    by_exp = {}
    for d, e in emap.items():
        by_exp.setdefault(e, []).append(d)

    CK.mkdir(exist_ok=True)
    for e in sorted(by_exp):
        f, part = OPT / f"{e}.parquet", CK / f"{e}.parquet"
        if not f.exists() or part.exists():
            continue
        opt = pd.read_parquet(f)
        opt["ts"] = pd.to_datetime(opt["ts"])
        opt["date"] = opt["ts"].dt.date.astype(str)
        rows = []
        for d in by_exp[e]:
            ds_ = spot[spot["date"] == d]
            if ds_.empty:
                continue
            g = ds_.set_index(pd.DatetimeIndex(ds_["ts"]))
            t0 = pd.Timestamp(f"{d} {ENTRY}:00")
            t1 = pd.Timestamp(f"{d} {EXIT}:00")
            if t0 not in g.index or t1 not in g.index:
                continue
            S = float(g.loc[t0, "c"])
            T = years_to_expiry(t0, e)
            dte = T * 365
            if not (DTE_LO <= dte <= DTE_HI):
                continue

            day = opt[(opt["date"] == d) & (opt["ts"] <= t0)
                      & (opt["ts"] >= t0 - pd.Timedelta(minutes=LOOKBACK_MIN))]
            if day.empty:
                continue
            day = day[day["v"] > 0]
            last = day.sort_values("ts").groupby(["strike", "cp"], observed=True)["c"].last()
            if len(last) < 10:
                continue
            ce = [(int(k), float(p)) for (k, cp), p in last.items() if cp == "CE"]
            pe = [(int(k), float(p)) for (k, cp), p in last.items() if cp == "PE"]

            ac, vc = surface_iv(ce, S, T, "CE")
            ap, vp = surface_iv(pe, S, T, "PE")
            if ac is None or ap is None:
                continue

            atm = int(round(S / STEP) * STEP)
            pc = dict(ce).get(atm)
            pp = dict(pe).get(atm)
            iv_atm = np.nan
            if pc and pp:
                a = implied_vol(pc, S, atm, T, "CE")
                b = implied_vol(pp, S, atm, T, "PE")
                if a and b:
                    iv_atm = (a + b) / 2.0

            rec = {"date": d, "expiry": e, "S": S, "dte": dte,
                   "n_ce": len(ac), "n_pe": len(ap), "iv_atm": iv_atm}
            for t in TARGETS:
                rec[f"ivc{int(t*100)}"] = at_delta(ac, vc, t)
                rec[f"ivp{int(t*100)}"] = at_delta(ap, vp, t)

            # outcomes
            seg = g.loc[t0:t1, "c"].astype(float)
            r = float(g.loc[t1, "c"]) / S - 1.0
            lr = np.diff(np.log(seg.to_numpy()))
            lr = lr[np.isfinite(lr)]
            rec.update({
                "ret": r, "absr": abs(r), "down": min(r, 0.0), "up": max(r, 0.0),
                "rv": float(np.std(lr, ddof=1) * np.sqrt(252 * 375)) if len(lr) > 30 else np.nan,
                "rskw": float(pd.Series(lr).skew()) if len(lr) > 30 else np.nan,
                "nmin": len(lr)})
            rows.append(rec)
        pd.DataFrame(rows).to_parquet(part, index=False)
        print(f"  {e}  {len(rows)}", flush=True)

    parts = [pd.read_parquet(x) for x in sorted(CK.glob("*.parquet"))]
    parts = [x for x in parts if len(x)]
    if not parts:
        print("nothing built")
        return 1
    t = pd.concat(parts, ignore_index=True).drop_duplicates("date").sort_values("date")
    for k in (10, 25, 40):
        t[f"rr{k}"] = t[f"ivp{k}"] - t[f"ivc{k}"]
        t[f"bf{k}"] = (t[f"ivp{k}"] + t[f"ivc{k}"]) / 2.0 - t["iv_atm"]
    t["year"] = pd.to_datetime(t["date"]).dt.year
    t.to_csv(STORE / "skew_panel.csv", index=False)
    print(f"\n{len(t)} sessions  {t.date.min()} .. {t.date.max()}")
    cols = ["iv_atm", "ivc25", "ivp25", "rr25", "bf25", "rr10", "bf10",
            "ret", "absr", "rv", "rskw", "dte"]
    print(t[cols].describe().T[["count", "mean", "std", "25%", "50%", "75%"]]
          .round(4).to_string())
    print("\nby year:")
    print(t.groupby("year")[["rr25", "bf25", "iv_atm", "rv", "absr"]].mean().round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
