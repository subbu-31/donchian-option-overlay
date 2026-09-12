"""Zerodha NIFTY-options round-trip cost model, in rupees per lot.

Rates are held FLAT, not date-aware -- unlike lib/ic_costs.py, which the
iron-condor pipeline (scripts/iron_condor/ onward) uses instead because its
sample spans multiple statutory-rate regimes. This module is only used by
scripts/core/, whose 2024-2026 sample sits almost entirely inside one STT/
exchange-charge regime; scripts/extensions/31_cost_audit.py re-prices the
same trades under the date-aware schedule and quantifies the (small) effect
of that simplification rather than assuming it away.
"""
LOT = 75
BROKERAGE = 20.0          # flat per executed order
STT_SELL = 0.0010         # 0.10% of premium, sell side only
EXCH_TXN = 0.00035503     # NSE F&O options, both sides, on premium turnover
SEBI = 10.0 / 1e7
STAMP_BUY = 0.00003
GST = 0.18


def leg_cost(entry_prem, exit_prem, lot=LOT, slippage_pts=0.5):
    """Round-trip statutory + brokerage cost of ONE bought option leg, in rupees.

    `slippage_pts` is charged twice (once each side) and is the crossing cost
    the OHLC bars cannot see.
    """
    buy_turn = entry_prem * lot
    sell_turn = exit_prem * lot
    brok = 2 * BROKERAGE
    txn = (buy_turn + sell_turn) * EXCH_TXN
    stt = sell_turn * STT_SELL
    sebi = (buy_turn + sell_turn) * SEBI
    stamp = buy_turn * STAMP_BUY
    gst = GST * (brok + txn + sebi)
    slip = 2 * slippage_pts * lot
    return brok + txn + stt + sebi + stamp + gst + slip
