#!/usr/bin/env python3
"""
38_skew_conditioning.py -- Two follow-ups to the skew forecast nulls.

(1) The quintile read showed the variance risk premium widest at BOTH ends of
    the skew distribution, which is a level effect, not a slope effect.  Test
    |rrz| against vrp with the ATM vol level controlled -- if |rrz| only proxies
    for vol level it dies here.

(2) The money question: merge the skew state onto the stored condor P&L and ask
    whether any 09:45 skew variable forecasts the P&L that day.  Then trade the
    best one out of sample -- fit the rule on 2025, apply it unchanged to 2026.
"""
import sys
from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.core import STORE

QTY = 750


def ncdf(x):
    return 0.5 * (1 + np.vectorize(lambda v: erf(v / sqrt(2)))(x))


def ols_hac(y, X, lags=5):
    y = np.asarray(y, float); X = np.asarray(X, float)
    n, k = X.shape
    XtXi = np.linalg.pinv(X.T @ X)
    b = XtXi @ (X.T @ y)
    u = y - X @ b
    S = (X * u[:, None]).T @ (X * u[:, None])
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        A = (X[L:] * u[L:, None]).T @ (X[:-L] * u[:-L, None])
        S += w * (A + A.T)
    V = XtXi @ S @ XtXi * (n / (n - k))
    se = np.sqrt(np.diag(V))
    t = b / se
    return b, se, t, 2 * (1 - ncdf(np.abs(t))), 1 - float(u @ u) / float(((y - y.mean()) ** 2).sum())


def show(m, yname, xs, label):
    mm = m[[yname] + xs].dropna()
    if len(mm) < 40:
        print(f"  {label:<40} n={len(mm)} -- skipped"); return
    y = mm[yname].to_numpy(float)
    X = np.column_stack([np.ones(len(mm))] + [mm[c].to_numpy(float) for c in xs])
    b, se, t, p, r2 = ols_hac(y, X)
    print(f"  {label:<40} n={len(mm):>4} R2={r2:>6.3f}")
    for i, nm in enumerate(["const"] + xs):
        if nm == "const":
            continue
        print(f"      {nm:<10} b={b[i]:>12.4f} se={se[i]:>10.4f} t={t[i]:>6.2f} "
              f"p={p[i]:>6.3f}{'  <--' if p[i] < 0.05 else ''}")


def lo_t(x):
    x = np.asarray(x, float); n = len(x)
    if n < 5 or x.std(ddof=1) == 0:
        return n, np.nan, np.nan, np.nan
    sr = x.mean() / x.std(ddof=1)
    t = sr / np.sqrt((1 + 0.5 * sr ** 2) / n)
    return n, sr * np.sqrt(252), t, float(2 * (1 - ncdf(np.array([abs(t)]))[0]))


def main():
    d = pd.read_csv(STORE / "skew_panel.csv")
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date").reset_index(drop=True)
    for k in ("rr25", "bf25"):
        r = d[k].rolling(20, min_periods=15)
        d[k + "z"] = (d[k] - r.mean().shift(1)) / r.std().shift(1)
    d["arrz"] = d["rr25z"].abs()
    d["vrp"] = np.log(d["rv"] / d["iv_atm"])
    d["liv"] = np.log(d["iv_atm"])
    d["year"] = d["date"].dt.year

    print("(1) is |rrz| anything once the vol level is controlled?\n")
    show(d, "vrp", ["arrz"], "vrp ~ |rrz|")
    show(d, "vrp", ["liv", "arrz"], "vrp ~ log(iv) + |rrz|")
    show(d, "vrp", ["liv", "arrz", "bf25z"], "vrp ~ log(iv) + |rrz| + bfz")
    show(d, "vrp", ["liv"], "vrp ~ log(iv)                [benchmark]")

    print("\n(2) does any 09:45 skew variable forecast the condor P&L?\n")
    w = pd.read_csv(STORE / "intraday_wings_costfix.csv")
    if "stat_new" in w.columns:
        w["stat"] = w["stat_new"]
    w["date"] = pd.to_datetime(w["date"])
    w["net"] = w["gross"] - w["slip"] - w["stat"]

    for cfg in ["0.35/0.15", "0.25/0.05"]:
        g = w[w.cfg == cfg][["date", "net", "maxloss"]]
        m = d.merge(g, on="date", how="inner")
        m["netu"] = m["net"] / QTY
        print(f"--- cfg {cfg}   merged n={len(m)} ---")
        show(m, "netu", ["rr25z"],           "net(pts) ~ rrz")
        show(m, "netu", ["arrz"],            "net(pts) ~ |rrz|")
        show(m, "netu", ["liv"],             "net(pts) ~ log(iv)")
        show(m, "netu", ["liv", "rr25z", "bf25z"], "net(pts) ~ log(iv) + rrz + bfz")

        print("\n    tercile of each signal -> mean net Rs/session (full sample)")
        for sig in ["rr25z", "arrz", "liv", "bf25z"]:
            mm = m.dropna(subset=[sig]).copy()
            mm["b"] = pd.qcut(mm[sig], 3, labels=["lo", "mid", "hi"])
            row = mm.groupby("b", observed=True)["net"].agg(["mean", "size"])
            print(f"      {sig:<7} " + "  ".join(
                f"{i}:{r['mean']:>8,.0f}(n={int(r['size'])})" for i, r in row.iterrows()))

        print("\n    OOS rule: pick the better half on 2025, apply unchanged to 2026")
        for sig in ["rr25z", "arrz", "liv", "bf25z"]:
            tr = m[(m.year == 2025) & m[sig].notna()]
            te = m[(m.year == 2026) & m[sig].notna()]
            if len(tr) < 100 or len(te) < 40:
                continue
            cut = tr[sig].median()
            hi_better = tr.loc[tr[sig] > cut, "net"].mean() > tr.loc[tr[sig] <= cut, "net"].mean()
            sel_tr = tr[sig] > cut if hi_better else tr[sig] <= cut
            sel_te = te[sig] > cut if hi_better else te[sig] <= cut
            a = tr.loc[sel_tr, "net"].to_numpy()
            b_ = te.loc[sel_te, "net"].to_numpy()
            n1, s1, t1, p1 = lo_t(a)
            n2, s2, t2, p2 = lo_t(b_)
            side = "high" if hi_better else "low"
            print(f"      {sig:<7} trade {side:<4} half | IS n={n1:>3} Rs/s {a.mean():>7,.0f} "
                  f"SR {s1:>5.2f} t {t1:>5.2f} | OOS n={n2:>3} Rs/s {b_.mean():>7,.0f} "
                  f"SR {s2:>5.2f} t {t2:>5.2f}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
