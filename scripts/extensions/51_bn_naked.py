#!/usr/bin/env python3
"""
51_bn_naked.py -- The naked structures on BANKNIFTY. Replicates script 49.

Naked short strangles at 0.25 / 0.30 / 0.35 delta and the ATM straddle, entry
09:45, exit 15:00, priced on each leg's own traded 1-minute path.

TWO DELIBERATE DEPARTURES FROM SCRIPT 26, both reported rather than chosen:
  * NO DTE BAND. He instructed that the 1-8 restriction be dropped -- it only
    ever existed to mirror the NIFTY study design. Every session in each expiry
    folder is traded, and DTE is carried as a column so the 1-8.5 slice stays
    separately reportable and comparable with all prior work.
  * CONTRACT KIND is recorded. Only 16 of 56 folders are true monthly expiries
    (last Wed/Thu of their month); the rest are weeklies. Results are split by
    kind rather than pooled, because a weekly folder's early sessions are a
    back-week contract and a monthly folder's are the liquid front month.

QTY = 200 (10 lots x 20) as instructed. The real BANKNIFTY lot changed
repeatedly over 2018-2026 (~20, 25, 15 from Oct-2023, 30 from Nov-2024, 35),
and flat-per-order brokerage does not scale with quantity, so rupee figures are
indicative only. POINTS PER UNIT is the primary column and the only one
comparable across eras and against the NIFTY run.
"""
import calendar, re, sys, zipfile
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.bs import implied_vol, delta, years_to_expiry
from lib.spread import half_spread_rupees, TICK
import lib.ic_costs as IC

STORE = Path.home() / "scratch" / "store"
ZIP = Path.home() / "mnt" / "Downloads" / "banknifty_data-[cloudtraderpro.in]-20260825T105024Z-1-001.zip"
BASE = "banknifty_data-[cloudtraderpro.in]"
CK = STORE / "bnk_parts"
QTY, BAND, MIN_PREM = 200, 0.06, 1.0
ENTRY_T, EXIT_T = pd.Timestamp("09:45").time(), pd.Timestamp("15:00").time()
DELTAS = [0.25, 0.30, 0.35]
LEG_RE = re.compile(r"^(\d+)(CE|PE)_(\d{8})\.csv$")


def kind_of(e):
    d = pd.Timestamp(e); last = calendar.monthrange(d.year, d.month)[1]
    lt = max(x for x in range(1, last + 1) if pd.Timestamp(d.year, d.month, x).dayofweek == 3)
    lw = max(x for x in range(1, last + 1) if pd.Timestamp(d.year, d.month, x).dayofweek == 2)
    return "MONTHLY" if (d.day in (lt, lw) and d.dayofweek in (2, 3)) else "weekly"


def hs(h, l, c, price):
    ok = np.isfinite(h) & np.isfinite(l) & np.isfinite(c)
    if ok.sum() < 3: return TICK / 2.0
    v = half_spread_rupees(h[ok], l[ok], c[ok], price)
    return float(v) if np.isfinite(v) else TICK / 2.0


def main():
    z = zipfile.ZipFile(ZIP); names = z.namelist()
    folders = sorted({n.split("/")[1] for n in names
                      if n.startswith(BASE + "/") and len(n.split("/")) > 2 and n.split("/")[1]})
    CK.mkdir(parents=True, exist_ok=True)
    for e in folders:
        part = CK / f"{e}.parquet"
        sp_name = f"{BASE}/{e}/banknifty_spot.csv"
        if sp_name not in names or part.exists(): continue
        try: sp = pd.read_csv(z.open(sp_name))
        except Exception: continue
        sp["ts"] = pd.to_datetime(sp["timestamp"], errors="coerce")
        sp = sp.dropna(subset=["ts"]).sort_values("ts")
        sp["date"] = sp["ts"].dt.date.astype(str); sp["tm"] = sp["ts"].dt.time
        ed = pd.Timestamp(e).date()
        ent = sp[sp.tm == ENTRY_T]
        keep = [(r["date"], years_to_expiry(r["ts"], e) * 365, float(r["close"]))
                for _, r in ent.iterrows()
                if pd.Timestamp(r["date"]).date() < ed]        # never expiry day
        if not keep:
            pd.DataFrame().to_parquet(part, index=False); continue
        lo = min(k[2] for k in keep) * (1 - BAND); hi = max(k[2] for k in keep) * (1 + BAND)
        legs = {}
        for n in names:
            if not n.startswith(f"{BASE}/{e}/"): continue
            m = LEG_RE.match(n.split("/")[-1])
            if not m: continue
            k = int(m.group(1))
            if not (lo <= k <= hi): continue
            try:
                df = pd.read_csv(z.open(n), usecols=["timestamp", "high", "low", "close", "volume"])
            except Exception: continue
            df["ts"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["ts"]); df["date"] = df["ts"].dt.date.astype(str)
            legs[(k, m.group(2))] = df
        if len(legs) < 6:
            pd.DataFrame().to_parquet(part, index=False)
            print(f"  {e}: {len(legs)} legs, skipped", flush=True); continue

        kd = kind_of(e); rows = []
        for d0, dte, S0 in keep:
            win = sp[(sp.date == d0) & (sp.tm >= ENTRY_T) & (sp.tm <= EXIT_T)]
            if len(win) < 120: continue
            grid = pd.DatetimeIndex(win["ts"])
            book = {}
            for (k, cp), df in legs.items():
                g = df[df.date == d0]
                if g.empty: continue
                g = g.drop_duplicates("ts").set_index("ts").reindex(grid)
                book[(k, cp)] = (g["high"].ffill(limit=3).to_numpy(float),
                                 g["low"].ffill(limit=3).to_numpy(float),
                                 g["close"].ffill(limit=3).to_numpy(float),
                                 g["volume"].to_numpy(float))
            if len(book) < 6: continue
            T0 = years_to_expiry(grid[0], e)
            ch = {}
            for (k, cp), (h, l, c, v) in book.items():
                p = c[0]
                if not np.isfinite(p) or p < MIN_PREM or not (v[0] > 0): continue
                if not np.isfinite(c[-1]): continue
                iv = implied_vol(float(p), S0, float(k), T0, cp)
                if iv is None: continue
                D = delta(S0, float(k), T0, iv, cp)
                if np.isfinite(D): ch[(k, cp)] = abs(D)
            ce = {k: v for (k, cp), v in ch.items() if cp == "CE"}
            pe = {k: v for (k, cp), v in ch.items() if cp == "PE"}
            if len(ce) < 2 or len(pe) < 2: continue
            specs = [(f"strangle_{int(x*100)}", x) for x in DELTAS] + [("straddle_atm", None)]
            for name, tgt in specs:
                if tgt is None:
                    avail = sorted({k for (k, cp) in ch})
                    k0 = min(avail, key=lambda k: abs(k - S0))
                    if (k0, "CE") not in ch or (k0, "PE") not in ch: continue
                    L = [(k0, "CE"), (k0, "PE")]; dd = {"c_d": np.nan, "p_d": np.nan}
                else:
                    ck = min(ce, key=lambda k: abs(ce[k] - tgt))
                    pk = min(pe, key=lambda k: abs(pe[k] - tgt))
                    L = [(ck, "CE"), (pk, "PE")]; dd = {"c_d": ce[ck], "p_d": pe[pk]}
                a = {x: float(book[x][2][0]) for x in L}
                b = {x: float(book[x][2][-1]) for x in L}
                credit = sum(a.values()); gross = credit - sum(b.values())
                slip = sum(hs(book[x][0][:30], book[x][1][:30], book[x][2][:30], a[x])
                           + hs(book[x][0][-30:], book[x][1][-30:], book[x][2][-30:], b[x])
                           for x in L)
                stat = IC._statutory(buy_turn=sum(b.values()) * QTY,
                                     sell_turn=credit * QTY, orders=4, on=d0)
                tot = np.nansum(np.vstack([book[x][2] for x in L]), axis=0)
                pnl = credit - tot
                rows.append(dict(date=d0, expiry=e, kind=kd, spec=name, S=S0, dte=dte,
                                 c_k=L[0][0], p_k=L[1][0], credit=credit, gross=gross,
                                 slip=slip, stat_pts=stat / QTY,
                                 mae=float(np.nanmin(pnl)), **dd,
                                 net_pts=gross - slip - stat / QTY))
        pd.DataFrame(rows).to_parquet(part, index=False)
        print(f"  {e} [{kd}] {len(rows)}", flush=True)

    parts = [pd.read_parquet(x) for x in sorted(CK.glob("*.parquet"))]
    parts = [x for x in parts if len(x)]
    if not parts: print("nothing"); return 1
    t = pd.concat(parts, ignore_index=True)
    t["year"] = pd.to_datetime(t["date"]).dt.year
    t.to_csv(STORE / "bn_naked.csv", index=False)
    print(f"\n{t.date.nunique()} sessions, {len(t)} trades, {t.expiry.nunique()} contracts, QTY={QTY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
