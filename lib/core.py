"""Shared machinery: data loading, 5-minute resampling, Donchian signals,
and a 1-minute-path trade simulator used by both the spot and option stages."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

STORE = Path(os.environ.get("STORE_DIR", Path.home() / "scratch" / "store"))

SESSION_START = pd.Timestamp("09:15").time()
SESSION_END = pd.Timestamp("15:29").time()

# ---------------------------------------------------------------- data


def load_spot():
    s = pd.read_parquet(STORE / "spot_1min.parquet")
    s["ts"] = pd.to_datetime(s["ts"])
    return s.sort_values("ts").reset_index(drop=True)


def load_option_index():
    return json.loads((STORE / "option_index.json").read_text())


def to_5m(spot):
    """Resample 1-min bars to non-overlapping 5-min bars, per session.

    Bar labelled by its OPENING minute; a bar is only usable once its
    closing minute has printed, which the signal builder enforces.
    """
    df = spot.set_index("ts")
    out = (df.groupby("date")[["o", "h", "l", "c"]]
             .resample("5min", origin="start_day", label="left", closed="left")
             .agg(o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"))
             .dropna()
             .reset_index())
    out["bar_close_ts"] = out["ts"] + pd.Timedelta(minutes=4)
    return out


# ---------------------------------------------------------------- signal


def donchian_signals(bars5, n, entry_from="09:45", entry_to="15:00"):
    """Day-scoped Donchian breakout on completed 5-min bars.

    Channel at bar i uses the n bars STRICTLY BEFORE i, within the same
    session.  A long fires when bar i closes above that channel's high; a
    short when it closes below the channel's low.  The signal is only
    actionable from the minute after bar i closes.
    """
    b = bars5.copy()
    g = b.groupby("date")
    # shift(1) => channel excludes the signalling bar itself (no look-ahead)
    b["up"] = g["h"].transform(lambda x: x.shift(1).rolling(n).max())
    b["dn"] = g["l"].transform(lambda x: x.shift(1).rolling(n).min())
    b["bar_no"] = g.cumcount()

    t = b["ts"].dt.time
    window = (t >= pd.Timestamp(entry_from).time()) & (t <= pd.Timestamp(entry_to).time())

    long_ = (b["c"] > b["up"]) & window & b["up"].notna()
    short = (b["c"] < b["dn"]) & window & b["dn"].notna()

    b["sig"] = 0
    b.loc[long_, "sig"] = 1
    b.loc[short, "sig"] = -1
    # a bar cannot break both ways; if it somehow did, drop it
    b.loc[long_ & short, "sig"] = 0
    return b


def dedupe_signals(sig_bars, max_per_day=None, cooldown_bars=0, one_at_a_time=True):
    """Turn raw breakout bars into an ordered, non-overlapping entry list.

    `one_at_a_time` drops any signal that fires while a prior trade of the
    same day is still open -- applied later, in the simulator, since the
    holding period is exit-rule dependent.  Here we only apply a per-day cap
    and a bar cooldown, both of which are known before the trade opens.
    """
    s = sig_bars[sig_bars["sig"] != 0].copy()
    keep = []
    for _, day in s.groupby("date", sort=True):
        last_bar = -10 ** 9
        taken = 0
        for idx, row in day.iterrows():
            if max_per_day is not None and taken >= max_per_day:
                break
            if row["bar_no"] - last_bar < cooldown_bars:
                continue
            keep.append(idx)
            last_bar = row["bar_no"]
            taken += 1
    return s.loc[keep].sort_values("ts").reset_index(drop=True)


# ---------------------------------------------------------------- simulate


def prep_days(minute_bars):
    """{date: (numpy arrays of ts, o, h, l, c)} for fast per-day path walks."""
    out = {}
    for d, g in minute_bars.groupby("date", sort=False):
        out[d] = (g["ts"].to_numpy(), g["o"].to_numpy(dtype=float),
                  g["h"].to_numpy(dtype=float), g["l"].to_numpy(dtype=float),
                  g["c"].to_numpy(dtype=float))
    return out


def simulate(entries, minute_bars, stop=np.inf, target=np.inf, max_minutes=10 ** 6,
             flat_time="15:15", one_at_a_time=True, direction_col="sig",
             price_is_option=False, days=None):
    """Walk each entry forward on 1-minute bars.

    Entry is the OPEN of the first minute bar strictly after the signalling
    5-minute bar closes.  Exit is the first of: stop touched, target touched,
    max_minutes elapsed, or the flat_time cut -- checked bar by bar, and when
    a bar touches both stop and target the STOP is assumed first (pessimistic).

    `stop`/`target` are in price units of whatever series `minute_bars`
    holds, expressed as a distance from entry in the trade's own direction.
    """
    if days is None:
        days = prep_days(minute_bars)
    flat = pd.Timestamp(flat_time).time()
    rows = []
    open_until = {}

    for _, e in entries.iterrows():
        d = e["date"]
        if d not in days:
            continue
        ts, o, h, l, c = days[d]
        after = e["bar_close_ts"]
        if one_at_a_time and open_until.get(d) is not None and after < open_until[d]:
            continue
        j = int(np.searchsorted(ts, np.datetime64(after), side="right"))
        if j >= len(ts):
            continue
        if pd.Timestamp(ts[j]).time() >= flat:
            continue

        side = 1 if e[direction_col] > 0 else -1
        # a long option leg is always BOUGHT: the option series itself carries
        # the direction, so the trade on that series is always long.
        pos = 1 if price_is_option else side
        entry_px = o[j]
        if not np.isfinite(entry_px) or entry_px <= 0:
            continue

        stop_px = entry_px - pos * stop
        targ_px = entry_px + pos * target

        exit_px, exit_ts, reason = None, None, None
        mfe, mae = 0.0, 0.0
        k_end = min(len(ts) - 1, j + max_minutes)
        for k in range(j, len(ts)):
            fav = pos * (h[k] - entry_px) if pos > 0 else pos * (l[k] - entry_px)
            adv = pos * (l[k] - entry_px) if pos > 0 else pos * (h[k] - entry_px)
            mfe = max(mfe, fav)
            mae = min(mae, adv)
            hit_stop = (l[k] <= stop_px) if pos > 0 else (h[k] >= stop_px)
            hit_targ = (h[k] >= targ_px) if pos > 0 else (l[k] <= targ_px)
            if np.isfinite(stop) and hit_stop:
                exit_px, exit_ts, reason = stop_px, ts[k], "stop"
                break
            if np.isfinite(target) and hit_targ:
                exit_px, exit_ts, reason = targ_px, ts[k], "target"
                break
            if pd.Timestamp(ts[k]).time() >= flat:
                exit_px, exit_ts, reason = c[k], ts[k], "flat"
                break
            if k >= k_end:
                exit_px, exit_ts, reason = c[k], ts[k], "time"
                break
        if exit_px is None:
            exit_px, exit_ts, reason = c[-1], ts[-1], "eod"

        if one_at_a_time:
            open_until[d] = pd.Timestamp(exit_ts)

        rows.append({
            "date": d, "entry_ts": pd.Timestamp(ts[j]), "exit_ts": pd.Timestamp(exit_ts),
            "dir": side, "entry_px": float(entry_px), "exit_px": float(exit_px),
            "pnl": float(pos * (exit_px - entry_px)), "mfe": float(mfe), "mae": float(mae),
            "minutes": int((pd.Timestamp(exit_ts) - pd.Timestamp(ts[j])).total_seconds() // 60),
            "reason": reason,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- stats


def block_bootstrap_ci(trades, col="pnl", n_boot=2000, seed=7):
    """Day-clustered bootstrap CI on the mean -- intraday trades on the same
    session are not independent, so resample whole days, not trades."""
    if trades.empty:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    by_day = [g[col].to_numpy() for _, g in trades.groupby("date")]
    days = len(by_day)
    means = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, days, days)
        means[i] = np.concatenate([by_day[p] for p in pick]).mean()
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def summarise(trades, label="", col="pnl"):
    if trades.empty:
        return {"label": label, "n": 0}
    x = trades[col]
    daily = trades.groupby("date")[col].sum()
    lo, hi = block_bootstrap_ci(trades, col)
    return {
        "label": label,
        "n": int(len(x)),
        "days": int(trades["date"].nunique()),
        "trades_per_day": round(len(x) / trades["date"].nunique(), 2),
        "mean": round(float(x.mean()), 3),
        "ci_lo": round(lo, 3),
        "ci_hi": round(hi, 3),
        "median": round(float(x.median()), 3),
        "hit": round(float((x > 0).mean()), 3),
        "win": round(float(x[x > 0].mean()) if (x > 0).any() else 0.0, 2),
        "loss": round(float(x[x <= 0].mean()) if (x <= 0).any() else 0.0, 2),
        "total": round(float(x.sum()), 1),
        "daily_sharpe": round(float(daily.mean() / daily.std() * np.sqrt(252)), 2) if daily.std() > 0 else np.nan,
        "max_dd": round(float((x.cumsum().cummax() - x.cumsum()).max()), 1),
        "avg_minutes": round(float(trades["minutes"].mean()), 1),
    }
