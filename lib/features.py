"""Causal, look-ahead-free features attached to every 5-minute bar."""
import numpy as np


def add_features(bars5, n):
    """Assumes `bars5` already carries up/dn/bar_no from donchian_signals(n)."""
    b = bars5.copy()
    b["width"] = b["up"] - b["dn"]

    # channel width relative to the SAME bar-of-day over the prior 20 sessions
    piv = b.pivot_table(index="date", columns="bar_no", values="width")
    ref = piv.shift(1).rolling(20, min_periods=10).mean()
    ref_long = ref.stack(future_stack=True).rename("width_ref").reset_index()
    b = b.merge(ref_long, on=["date", "bar_no"], how="left")
    b["width_rel"] = b["width"] / b["width_ref"]

    # how decisively the bar cleared the channel, in channel widths
    ext = np.where(b["c"] > b["up"], (b["c"] - b["up"]) / b["width"],
          np.where(b["c"] < b["dn"], (b["dn"] - b["c"]) / b["width"], np.nan))
    b["ext"] = ext

    # opening 30-minute range, known by 09:45, vs its own 20-session average
    o30 = (b[b["bar_no"] < 6].groupby("date")
             .agg(or_hi=("h", "max"), or_lo=("l", "min"), or_open=("o", "first")))
    o30["or_range"] = o30["or_hi"] - o30["or_lo"]
    o30["or_ref"] = o30["or_range"].shift(1).rolling(20, min_periods=10).mean()
    o30["or_rel"] = o30["or_range"] / o30["or_ref"]
    b = b.merge(o30[["or_range", "or_rel", "or_hi", "or_lo", "or_open"]], on="date", how="left")

    # overnight gap vs prior close, in units of the 20-session average OR range
    day_close = b.groupby("date")["c"].last()
    day_open = b.groupby("date")["o"].first()
    gap = (day_open - day_close.shift(1)).rename("gap").reset_index()
    b = b.merge(gap, on="date", how="left")
    b["gap_rel"] = b["gap"] / b["or_ref"] if "or_ref" in b else np.nan
    b = b.merge(o30[["or_ref"]], on="date", how="left", suffixes=("", "_y"))
    b["gap_rel"] = b["gap"] / b["or_ref"]

    # position within the day so far
    b["day_move"] = b["c"] - b.groupby("date")["o"].transform("first")

    # sequence number of the breakout within the day
    return b
