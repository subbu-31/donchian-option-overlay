#!/usr/bin/env python3
"""
04_conditioning.py -- Two questions, both answered on the 2024 design sample
only, with 2025 / 2026 reported alongside purely as replication.

Q1 DIRECTION: does any causal conditioner turn the (useless) raw breakout
   into a directional signal?  Quintile-sorted excess forward move.

Q2 VOLATILITY: does the Donchian channel WIDTH predict the size of the next
   move, irrespective of sign?  That is the quantity an option BUYER is
   actually long -- a narrow channel that reliably precedes a big move is a
   long-gamma edge even when direction is a coin flip.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.core import STORE, load_spot, to_5m, donchian_signals
from lib.features import add_features

N = 12                      # fixed a priori: 1 hour of 5-minute bars
HORIZONS = [15, 30, 60]
PERIODS = {"2024 design": 2024, "2025 in-sample": 2025, "2026 holdout": 2026}
CONDS = ["width_rel", "ext", "bar_no", "or_rel", "gap_rel", "day_move_sgn", "seq"]


def dcci(df, col, n_boot=1500, seed=13):
    rng = np.random.default_rng(seed)
    gs = [g[col].to_numpy() for _, g in df.groupby("date")]
    k = len(gs)
    if k < 8:
        return (np.nan, np.nan)
    m = np.empty(n_boot)
    for i in range(n_boot):
        m[i] = np.concatenate([gs[j] for j in rng.integers(0, k, k)]).mean()
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def build(spot, year):
    sub = spot[spot["year"] == year].reset_index(drop=True)
    px = sub.set_index("ts")["c"]
    b = donchian_signals(to_5m(sub), N)
    b = add_features(b, N)
    b["tod"] = b["bar_close_ts"].dt.floor("30min").dt.time.astype(str)
    for h in HORIZONS:
        fwd = (b["bar_close_ts"] + pd.Timedelta(minutes=h)).map(px).to_numpy()
        b[f"mv{h}"] = fwd - b["c"].to_numpy()
        b[f"abs{h}"] = np.abs(b[f"mv{h}"])
    bench = b.groupby("tod")[[f"mv{h}" for h in HORIZONS]].mean()
    absbench = b.groupby("tod")[[f"abs{h}" for h in HORIZONS]].mean()
    for h in HORIZONS:
        b[f"bm{h}"] = b["tod"].map(bench[f"mv{h}"])
        b[f"ba{h}"] = b["tod"].map(absbench[f"abs{h}"])
    b["day_move_sgn"] = np.sign(b["day_move"]) * np.sign(b["sig"].replace(0, np.nan))
    ev = b[b["sig"] != 0].copy()
    ev["seq"] = ev.groupby("date").cumcount() + 1
    return b, ev


def main():
    spot = load_spot()
    spot["year"] = pd.to_datetime(spot["date"]).dt.year
    built = {p: build(spot, y) for p, y in PERIODS.items()}

    # ---------------- Q1 : conditioned direction, design sample -------------
    print("=" * 78)
    print(f"Q1  conditioned DIRECTION of the n={N} breakout -- excess pts vs "
          f"same-time-of-day baseline")
    print("=" * 78)
    q1 = []
    for pname, (b, ev) in built.items():
        for cond in CONDS:
            if cond not in ev.columns:
                continue
            d = ev.dropna(subset=[cond]).copy()
            if len(d) < 100:
                continue
            try:
                d["q"] = pd.qcut(d[cond], 5, labels=False, duplicates="drop")
            except ValueError:
                continue
            for h in HORIZONS:
                for q, g in d.groupby("q"):
                    x = g[["date"]].copy()
                    x["ex"] = g["sig"] * (g[f"mv{h}"] - g[f"bm{h}"])
                    x = x.dropna()
                    if len(x) < 50:
                        continue
                    lo, hi = dcci(x, "ex")
                    q1.append({"period": pname, "cond": cond, "q": int(q), "h": h,
                               "nobs": len(x), "excess": round(float(x["ex"].mean()), 2),
                               "lo": round(lo, 2), "hi": round(hi, 2),
                               "sig": (lo > 0) or (hi < 0)})
    q1 = pd.DataFrame(q1)
    q1.to_csv(STORE / "q1_conditioning.csv", index=False)
    des = q1[(q1["period"] == "2024 design") & q1["sig"]]
    print(f"\n  design-sample cells tested: {int((q1['period']=='2024 design').sum())}"
          f"   significant at 95%: {len(des)}   (expected by chance ~"
          f"{0.05*int((q1['period']=='2024 design').sum()):.0f})")
    if len(des):
        for _, r in des.iterrows():
            rep = q1[(q1["cond"] == r["cond"]) & (q1["q"] == r["q"]) & (q1["h"] == r["h"])]
            s25 = rep[rep["period"] == "2025 in-sample"]["excess"].values
            s26 = rep[rep["period"] == "2026 holdout"]["excess"].values
            print(f"    {r['cond']:>12} q{r['q']} h={r['h']:>3}m  2024 excess={r['excess']:>7.2f} "
                  f"[{r['lo']:>6.2f},{r['hi']:>6.2f}] n={r['nobs']:<5} | "
                  f"2025={s25[0] if len(s25) else float('nan'):>7.2f} | "
                  f"2026={s26[0] if len(s26) else float('nan'):>7.2f}")

    # ---------------- Q2 : channel width -> size of next move ---------------
    print("\n" + "=" * 78)
    print(f"Q2  does a NARROW n={N} channel precede a BIGGER move?  "
          "(long-gamma / option-buying edge)")
    print("=" * 78)
    q2 = []
    for pname, (b, ev) in built.items():
        d = b.dropna(subset=["width_rel"]).copy()
        d = d[d["bar_no"] >= N]
        d["q"] = pd.qcut(d["width_rel"], 5, labels=False, duplicates="drop")
        for h in HORIZONS:
            for q, g in d.groupby("q"):
                x = g[["date"]].copy()
                x["ratio"] = g[f"abs{h}"] / g[f"ba{h}"]
                x["exabs"] = g[f"abs{h}"] - g[f"ba{h}"]
                x = x.dropna()
                if len(x) < 50:
                    continue
                lo, hi = dcci(x, "exabs")
                q2.append({"period": pname, "q": int(q), "h": h, "nobs": len(x),
                           "abs_ratio": round(float(x["ratio"].mean()), 3),
                           "exabs": round(float(x["exabs"].mean()), 2),
                           "lo": round(lo, 2), "hi": round(hi, 2),
                           "sig": (lo > 0) or (hi < 0)})
    q2 = pd.DataFrame(q2)
    q2.to_csv(STORE / "q2_width_vol.csv", index=False)
    for pname in PERIODS:
        p = q2[q2["period"] == pname]
        if p.empty:
            continue
        print(f"\n  {pname}   width quintile 0 = narrowest channel")
        print("     h  q0     q1     q2     q3     q4    (mean |move| / same-time-of-day |move|)")
        for h in HORIZONS:
            r = p[p["h"] == h].sort_values("q")
            print(f"   {h:>3}m " + "  ".join(f"{v:.3f}" for v in r["abs_ratio"]))
    print("\nwrote q1_conditioning.csv, q2_width_vol.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
