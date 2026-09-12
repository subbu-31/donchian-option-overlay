"""Zerodha / NSE cost model for equity options, DATE-AWARE.

Statutory rates changed twice inside the periods studied, so a single set of
constants is wrong for any sample spanning them.  Verified against Zerodha's
published schedule and its 1 October 2024 revision notice:

  Exchange transaction charge, NSE equity options, on premium turnover
    to 2024-09-30   0.0495%
    from 2024-10-01 0.03503%   (0.03553% on the current published schedule;
                                the intermediate revision date is not stated,
                                and the 1.4% difference is immaterial here)

  STT, options, SELL side, on premium
    to 2024-09-30   0.0625%
    2024-10-01      0.10%
    from 2026-04-01 0.15%      (Budget 2026-27)

  Brokerage      flat Rs 20 per executed order
  SEBI           Rs 10 per crore of turnover
  Stamp duty     0.003% on the BUY side
  GST            18% on (brokerage + exchange charge + SEBI), not on STT or stamp

A four-leg iron condor round trip is eight executed orders.
"""
import pandas as pd

BROKERAGE = 20.0
SEBI = 10.0 / 1e7
STAMP_BUY = 0.00003
GST = 0.18
SLIP_PTS = 0.5

_STT = [("2024-10-01", 0.000625), ("2026-04-01", 0.0010), (None, 0.0015)]
_EXCH = [("2024-10-01", 0.000495), (None, 0.0003553)]


def _rate(schedule, on):
    """schedule is [(cutoff_date, rate_before_cutoff), ..., (None, rate_after_last)]"""
    if on is None:
        return schedule[-1][1]
    d = pd.Timestamp(on)
    for cutoff, rate in schedule:
        if cutoff is None or d < pd.Timestamp(cutoff):
            return rate
    return schedule[-1][1]


def stt_rate(on=None):
    return _rate(_STT, on)


def exch_rate(on=None):
    return _rate(_EXCH, on)


def _statutory(buy_turn, sell_turn, orders, on=None):
    brok = orders * BROKERAGE
    exch = (buy_turn + sell_turn) * exch_rate(on)
    stt = sell_turn * stt_rate(on)
    sebi = (buy_turn + sell_turn) * SEBI
    stamp = buy_turn * STAMP_BUY
    gst = GST * (brok + exch + sebi)
    return brok + exch + stt + sebi + stamp + gst


def ic_cost(entry, exit_, qty, slip_pts=SLIP_PTS, on=None):
    """entry/exit are dicts {sc, sp, lc, lp} of per-unit premiums."""
    sell_turn = (entry["sc"] + entry["sp"] + exit_["lc"] + exit_["lp"]) * qty
    buy_turn = (entry["lc"] + entry["lp"] + exit_["sc"] + exit_["sp"]) * qty
    return _statutory(buy_turn, sell_turn, 8, on) + 8 * slip_pts * qty


def hedge_cost(entry_prem, exit_prem, qty, slip_pts=SLIP_PTS, on=None):
    return _statutory(entry_prem * qty, exit_prem * qty, 2, on) + 2 * slip_pts * qty
