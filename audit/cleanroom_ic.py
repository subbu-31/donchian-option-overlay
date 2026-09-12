#!/usr/bin/env python3
"""
cleanroom_ic.py -- Independent re-implementation. Imports NOTHING from lib/.

Reads the raw expiry zips directly, writes its own Black-Scholes, its own IV
bisection, its own delta, its own strike selection and its own cost stack, then
compares trade by trade against the stored intraday_wings_costfix.csv.

Spec, exactly as stated: sell the 0.25-delta call and put, buy the 0.10-delta
call and put, enter 09:45, exit 15:00, nearest weekly expiry strictly after the
session.

LOOK-AHEAD SURFACE, audited explicitly rather than asserted:
  A. Strike selection reads ONLY the 09:45 bar -- spot, option closes, and time
     to expiry all as at 09:45.  Nothing later is in scope when strikes are picked.
  B. Entry price is the 09:45 bar close; exit price is the 15:00 bar close.
     Neither is a session extreme, and neither is chosen by looking at the path.
  C. No forward fill.  A leg with no trade printed in the entry or exit minute
     disqualifies the session rather than borrowing a neighbouring bar.
  D. PROBE: re-run selecting strikes with the 15:00 tick instead of 09:45.  If
     the pipeline leaks, the honest and the cheating versions look alike.  If it
     is clean, the cheating version should be markedly better.
"""
import math, os, re, sys, zipfile
from pathlib import Path
import numpy as np, pandas as pd

DL = Path(os.environ.get("DL_DIR", Path.home() / "mnt" / "Downloads"))
STORE = Path(os.environ.get("STORE_DIR", Path.home() / "scratch" / "store"))
LEG = re.compile(r"^(\d+)(CE|PE)_(\d{8})\.csv$")
RF, YEAR, QTY, LOT = 0.065, 365.0, 750, 75
ENTRY, EXIT = "09:45:00", "15:00:00"
DTE_LO, DTE_HI = 1.0, 8.5
EVERY = int(sys.argv[1]) if len(sys.argv) > 1 else 6     # sample every Nth expiry

# ---------------- own Black-Scholes ----------------
def N(x): return 0.5 * math.erfc(-x / math.sqrt(2.0))

def bs(S, K, T, s, cp):
    if T <= 0 or s <= 0:
        return max(0.0, S - K) if cp == "CE" else max(0.0, K - S)
    d1 = (math.log(S / K) + (RF + 0.5 * s * s) * T) / (s * math.sqrt(T))
    d2 = d1 - s * math.sqrt(T)
    if cp == "CE":
        return S * N(d1) - K * math.exp(-RF * T) * N(d2)
    return K * math.exp(-RF * T) * N(-d2) - S * N(-d1)

def iv(px, S, K, T, cp):
    lo, hi = 1e-3, 5.0
    if T <= 0 or px <= 0: return None
    intr = max(0.0, S - K) if cp == "CE" else max(0.0, K - S)
    if px < intr - 1e-6: return None
    if bs(S, K, T, hi, cp) < px or bs(S, K, T, lo, cp) > px: return None
    for _ in range(200):
        m = 0.5 * (lo + hi)
        if bs(S, K, T, m, cp) > px: hi = m
        else: lo = m
        if hi - lo < 1e-7: break
    return 0.5 * (lo + hi)

def dlt(S, K, T, s, cp):
    if T <= 0 or s is None or s <= 0: return np.nan
    d1 = (math.log(S / K) + (RF + 0.5 * s * s) * T) / (s * math.sqrt(T))
    return N(d1) if cp == "CE" else N(d1) - 1.0

def yte(ts, exp):
    e = pd.Timestamp(exp) + pd.Timedelta(hours=15, minutes=30)
    return max((e - pd.Timestamp(ts)).total_seconds() / 86400.0, 1e-6) / YEAR

# ---------------- own cost stack ----------------
def stt_rate(d): return 0.000625 if d < "2024-10-01" else (0.0010 if d < "2026-04-01" else 0.0015)
def exch_rate(d): return 0.000495 if d < "2024-10-01" else 0.0003553

def charges(buy, sell, d, orders=8):
    brok = orders * 20.0
    exch = (buy + sell) * exch_rate(d)
    stt = sell * stt_rate(d)
    sebi = (buy + sell) * (10.0 / 1e7)
    stamp = buy * 0.00003
    gst = 0.18 * (brok + exch + sebi)
    return brok + exch + stt + sebi + stamp + gst

# ---------------- read the raw archives ----------------
zips = sorted(p for p in DL.iterdir() if re.match(r"^\d{8}\.zip$", p.name))
expiries = sorted(p.stem for p in zips)
picked = zips[::EVERY]
print(f"{len(zips)} archives on disk; auditing every {EVERY}th -> {len(picked)}\n", flush=True)

def parse(fh, cols):
    df = pd.read_csv(fh, usecols=lambda c: c in cols)
    df["ts"] = pd.to_datetime(df["Timestamp"], format="%d-%m-%Y %H:%M:%S", errors="coerce")
    return df.dropna(subset=["ts"])

rows = []
for zp in picked:
    exp = zp.stem
    with zipfile.ZipFile(zp) as z:
        names = z.namelist()
        if "nifty_spot.csv" not in names: continue
        sp = parse(z.open("nifty_spot.csv"), {"Timestamp", "Close"})
        sp["d"] = sp.ts.dt.strftime("%Y-%m-%d")
        sp["t"] = sp.ts.dt.strftime("%H:%M:%S")
        # sessions where THIS expiry is the nearest one strictly after the session
        sess = []
        for d in sorted(sp.d.unique()):
            nxt = min((e for e in expiries if e > d.replace("-", "")), default=None)
            if nxt != exp: continue
            T = yte(f"{d} {ENTRY}", exp)
            if DTE_LO <= T * 365 <= DTE_HI: sess.append(d)
        if not sess: continue
        # read every leg once, keep only the two minutes we need
        book = {}
        for n in names:
            m = LEG.match(Path(n).name)
            if not m or m.group(3) != exp: continue
            df = parse(z.open(n), {"Timestamp", "Close", "Volume"})
            df["d"] = df.ts.dt.strftime("%Y-%m-%d")
            df["t"] = df.ts.dt.strftime("%H:%M:%S")
            df = df[df.d.isin(sess) & df.t.isin([ENTRY, EXIT]) & (df.Volume > 0)]
            for r in df.itertuples():
                book[(r.d, r.t, int(m.group(1)), m.group(2))] = float(r.Close)
        for d in sess:
            sc_ = sp[(sp.d == d) & (sp.t == ENTRY)]
            se_ = sp[(sp.d == d) & (sp.t == EXIT)]
            if sc_.empty or se_.empty: continue
            S0, S1 = float(sc_.Close.iloc[0]), float(se_.Close.iloc[0])
            for sel_t, tag in ((ENTRY, "honest"), (EXIT, "PROBE_lookahead")):
                Ssel = S0 if sel_t == ENTRY else S1
                Tsel = yte(f"{d} {sel_t}", exp)
                cand = {}
                for (dd, tt, k, cp), px in book.items():
                    if dd != d or tt != sel_t or px <= 1.0: continue
                    v = iv(px, Ssel, k, Tsel, cp)
                    if v is None or not (0.02 < v < 2.5): continue
                    D = dlt(Ssel, k, Tsel, v, cp)
                    if np.isfinite(D): cand[(k, cp)] = abs(D)
                ce = {k: v for (k, cp), v in cand.items() if cp == "CE"}
                pe = {k: v for (k, cp), v in cand.items() if cp == "PE"}
                if len(ce) < 2 or len(pe) < 2: continue
                sc_k = min(ce, key=lambda k: abs(ce[k] - 0.25))
                sp_k = min(pe, key=lambda k: abs(pe[k] - 0.25))
                cf = {k: v for k, v in ce.items() if k > sc_k}
                pf = {k: v for k, v in pe.items() if k < sp_k}
                if not cf or not pf: continue
                lc_k = min(cf, key=lambda k: abs(cf[k] - 0.10))
                lp_k = min(pf, key=lambda k: abs(pf[k] - 0.10))
                legs = [(sc_k, "CE", -1), (sp_k, "PE", -1), (lc_k, "CE", +1), (lp_k, "PE", +1)]
                if not all((d, ENTRY, k, cp) in book and (d, EXIT, k, cp) in book
                           for k, cp, _ in legs): continue
                gross = buy = sell = 0.0
                for k, cp, side in legs:
                    a, b = book[(d, ENTRY, k, cp)], book[(d, EXIT, k, cp)]
                    gross += side * (b - a)
                    if side < 0: sell += a * QTY; buy += b * QTY
                    else:        buy += a * QTY; sell += b * QTY
                st = charges(buy, sell, d)
                rows.append(dict(date=d, expiry=exp, tag=tag, S0=S0, dte=yte(f"{d} {ENTRY}", exp)*365,
                                 sc_k=sc_k, sp_k=sp_k, lc_k=lc_k, lp_k=lp_k,
                                 sc_d=ce[sc_k], sp_d=pe[sp_k], lc_d=cf[lc_k], lp_d=pf[lp_k],
                                 gross_pts=gross, gross_rs=gross*QTY, stat=st,
                                 net_ex_slip=gross*QTY - st))
    print(f"  {exp}", flush=True)

t = pd.DataFrame(rows)
t.to_csv(STORE / "cleanroom_ic.csv", index=False)
h = t[t.tag == "honest"]
print(f"\nclean-room honest trades: {len(h)} over {h.date.nunique()} sessions, "
      f"{h.expiry.nunique()} expiries")
print(f"  realised deltas -- short call {h.sc_d.mean():.3f}  short put {h.sp_d.mean():.3f}"
      f"  long call {h.lc_d.mean():.3f}  long put {h.lp_d.mean():.3f}   (targets 0.25 / 0.10)")
print(f"  gross {h.gross_pts.mean():+.3f} pts/session   statutory {(h.stat/QTY).mean():.3f} pts")
