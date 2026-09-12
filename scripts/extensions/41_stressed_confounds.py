#!/usr/bin/env python3
"""41_stressed_confounds.py -- what else is 'Stressed' quietly selecting?"""
import sys
from math import erf, sqrt
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.core import STORE

def ncdf(x): return 0.5*(1+erf(x/sqrt(2)))
def sc(x):
    x=np.asarray(x,float); n=len(x)
    if n<5 or x.std(ddof=1)==0: return n,np.nan,np.nan,np.nan
    sr=x.mean()/x.std(ddof=1); t=sr/np.sqrt((1+0.5*sr**2)/n)
    return n,x.mean(),sr*np.sqrt(252),t

d=pd.read_csv(STORE/"regime_decomp.csv")[["date","rv20"]]
d["date"]=pd.to_datetime(d["date"]); d=d.sort_values("date").reset_index(drop=True)
hi=d["rv20"].expanding(120).quantile(0.6667).shift(1)
d["hi"]=(d.rv20>=hi)&hi.notna()

w=pd.read_csv(STORE/"intraday_wings_costfix.csv")
if "stat_new" in w.columns: w["stat"]=w["stat_new"]
w["date"]=pd.to_datetime(w["date"]); w["net"]=w.gross-w.slip-w.stat
w["dow"]=w.date.dt.dayofweek

for cfg in ["0.35/0.15","0.25/0.05"]:
    m=w[w.cfg==cfg].merge(d[["date","hi","rv20"]],on="date").dropna(subset=["rv20"])
    m["year"]=m.date.dt.year
    print(f"===== {cfg} =====")
    print("  composition  Stressed vs rest:")
    for k in ["dte","credit","ic_maxdd","spot","dow"]:
        a,b=m.loc[m.hi,k],m.loc[~m.hi,k]
        print(f"    {k:<10} stressed {a.mean():>10.2f}   rest {b.mean():>10.2f}")
    print("  DTE-matched: Stressed vs rest within each DTE bucket")
    m["db"]=pd.cut(m.dte,[0,2,4,6,9],labels=["0-2","2-4","4-6","6-9"])
    for db,g in m.groupby("db",observed=True):
        a=sc(g.loc[g.hi,"net"]); b=sc(g.loc[~g.hi,"net"])
        print(f"    dte {db:<4} stressed n={a[0]:>3} Rs/s {a[1]:>8,.0f} t {a[3]:>5.2f} | "
              f"rest n={b[0]:>3} Rs/s {b[1]:>8,.0f} t {b[3]:>5.2f}")
    print("  two-way OLS: net ~ stressed + dte  (HC1)")
    mm=m.dropna(subset=["net","dte"])
    X=np.column_stack([np.ones(len(mm)),mm.hi.astype(float),mm.dte])
    y=mm.net.to_numpy(float)
    XtXi=np.linalg.pinv(X.T@X); bta=XtXi@(X.T@y); u=y-X@bta
    V=XtXi@((X*u[:,None]).T@(X*u[:,None]))@XtXi*(len(mm)/(len(mm)-3))
    se=np.sqrt(np.diag(V)); t=bta/se
    for i,nm in enumerate(["const","stressed","dte"]):
        print(f"    {nm:<9} b={bta[i]:>10,.0f} se={se[i]:>9,.0f} t={t[i]:>6.2f} "
              f"p={2*(1-ncdf(abs(t[i]))):>6.4f}")
    print("  tail inside Stressed:")
    s=m.loc[m.hi,"net"].sort_values()
    print(f"    worst 5 {', '.join(f'{v:,.0f}' for v in s.head(5))}")
    print(f"    5th pct {np.percentile(s,5):,.0f}  win% {100*(s>0).mean():.1f}  "
          f"mean {s.mean():,.0f}  median {s.median():,.0f}")
    eq=m.loc[m.hi].sort_values("date").net.cumsum()
    print(f"    max drawdown inside the gated equity curve Rs {(eq-eq.cummax()).min():,.0f}")
    print()
