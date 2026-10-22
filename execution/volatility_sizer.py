# execution/volatility_sizer.py
"""
Volatility-Adjusted Fixed-Capital & MIS Leverage Position Sizer.
Engineered specifically for Indian Equity Intraday Cash (MIS):
- Deploys the exact user-defined CAPITAL_PER_TRADE across all trades
- Utilizes 5x SEBI MIS leverage when enabled
- Normalizes monetary stop loss risk across all price levels
- Enforces strict Indian Tax & Brokerage Feasibility (minimum 4x Reward-to-Tax ratio)
"""

import math
from typing import Optional, Dict, Any, Tuple
from execution.execution_config import (
    CAPITAL_PER_TRADE,
    USE_MIS_LEVERAGE,
    MIS_LEVERAGE_MULTIPLIER,
    STOP_LOSS_PCT,
    TARGET_PCT,
    BROKERAGE_PER_ORDER,
    STT_INTRADAY_PCT,
    EXCHANGE_TXN_PCT,
    SEBI_CHARGES_PCT,
    STAMP_DUTY_PCT,
    GST_PCT,
    MIN_REWARD_TO_TAX_RATIO
)


def calculate_indian_statutory_taxes(buy_turnover: float, sell_turnover: float) -> Dict[str, float]:
    """
    Computes exact statutory taxes and brokerage for Indian Equity Intraday Cash (MIS).
    """
    brokerage_buy = min(BROKERAGE_PER_ORDER, buy_turnover * 0.0005)
    brokerage_sell = min(BROKERAGE_PER_ORDER, sell_turnover * 0.0005)
    total_brokerage = brokerage_buy + brokerage_sell

    stt = sell_turnover * STT_INTRADAY_PCT
    txn_charges = (buy_turnover + sell_turnover) * EXCHANGE_TXN_PCT
    sebi_charges = (buy_turnover + sell_turnover) * SEBI_CHARGES_PCT
    stamp_duty = buy_turnover * STAMP_DUTY_PCT

    gst = (total_brokerage + txn_charges + sebi_charges) * GST_PCT
    total_drag = total_brokerage + stt + txn_charges + sebi_charges + stamp_duty + gst

    return {
        "brokerage": round(total_brokerage, 2),
        "stt": round(stt, 2),
        "txn_charges": round(txn_charges, 2),
        "sebi": round(sebi_charges, 2),
        "stamp_duty": round(stamp_duty, 2),
        "gst": round(gst, 2),
        "total_charges": round(total_drag, 2)
    }


def calculate_fixed_capital_quantity(
    price: float,
    user_capital: float = CAPITAL_PER_TRADE,
    use_leverage: bool = USE_MIS_LEVERAGE,
    available_margin: Optional[float] = None
) -> int:
    """
    Computes exact share quantity using the user's configured CAPITAL_PER_TRADE.
    Applies 5x MIS leverage if enabled.
    """
    if price <= 0:
        return 0

    effective_margin = user_capital
    if available_margin and 0 < available_margin < effective_margin:
        effective_margin = available_margin * 0.95

    effective_buying_power = effective_margin * (MIS_LEVERAGE_MULTIPLIER if use_leverage else 1.0)
    qty = math.floor(effective_buying_power / price)
    return max(0, qty)


def calculate_risk_parity_quantity(
    price: float,
    atr: Optional[float] = None,
    risk_rupees: float = 750.0,
    max_capital: float = CAPITAL_PER_TRADE,
    available_margin: Optional[float] = None
) -> int:
    """
    Computes share quantity respecting the user's fixed capital limit and stop loss distance.
    """
    if price <= 0:
        return 0

    base_qty = calculate_fixed_capital_quantity(
        price=price,
        user_capital=max_capital,
        use_leverage=USE_MIS_LEVERAGE,
        available_margin=available_margin
    )

    if atr and atr > 0:
        sl_dist = max(atr * 1.2, price * STOP_LOSS_PCT)
    else:
        sl_dist = price * STOP_LOSS_PCT

    if sl_dist > 0:
        shares_by_risk = math.floor(risk_rupees / sl_dist)
        return max(1, min(base_qty, shares_by_risk))

    return max(1, base_qty)


def calculate_kelly_conviction_quantity(
    price: float,
    conviction_score: float = 0.75,
    user_capital: float = CAPITAL_PER_TRADE,
    use_leverage: bool = USE_MIS_LEVERAGE,
    available_margin: Optional[float] = None
) -> Tuple[int, str, float]:
    """
    Dynamic Kelly-inspired conviction sizing:
    - Grade A+ (conviction >= 0.85): 1.4x base capital multiplier (Max sizing on high conviction)
    - Grade A  (conviction >= 0.75): 1.0x base capital multiplier (Standard sizing)
    - Grade B  (conviction 0.70 - 0.74): 0.65x base capital multiplier (Defensive sizing)
    Returns (quantity, grade_label, effective_capital_used).
    """
    if price <= 0:
        return 0, "INVALID", 0.0

    if conviction_score >= 0.85:
        multiplier = 1.40
        grade = "GRADE_A+ (Conviction >= 85%)"
    elif conviction_score >= 0.75:
        multiplier = 1.00
        grade = "GRADE_A (Conviction >= 75%)"
    else:
        multiplier = 0.65
        grade = "GRADE_B (Conviction ~ 70%)"

    effective_capital = user_capital * multiplier
    qty = calculate_fixed_capital_quantity(
        price=price,
        user_capital=effective_capital,
        use_leverage=use_leverage,
        available_margin=available_margin
    )
    return max(1, qty), grade, effective_capital


def validate_trade_tax_viability(
    entry_price: float,
    qty: int,
    target_price: float
) -> Tuple[bool, float, float]:
    """
    Guarantees that expected profit easily overpowers Indian brokerage & taxes.
    Returns (is_viable, expected_net_profit, total_taxes).
    """
    if qty <= 0 or entry_price <= 0:
        return False, 0.0, 0.0

    buy_turnover = entry_price * qty
    sell_turnover = target_price * qty
    gross_profit = abs(sell_turnover - buy_turnover)

    tax_info = calculate_indian_statutory_taxes(buy_turnover, sell_turnover)
    total_taxes = tax_info["total_charges"]
    net_profit = gross_profit - total_taxes

    # Must exceed minimum reward-to-tax ratio (at least 4x taxes)
    is_viable = (gross_profit >= total_taxes * MIN_REWARD_TO_TAX_RATIO) and (net_profit > 0)
    return is_viable, round(net_profit, 2), total_taxes


class VolatilitySizer:
    """
    Object-oriented wrapper for institutional position sizing and Indian tax validation.
    """
    def __init__(self, account_capital: float = CAPITAL_PER_TRADE, risk_per_trade_pct: float = 0.015):
        self.account_capital = account_capital
        self.risk_rupees = account_capital * risk_per_trade_pct

    def calculate_position(self, entry_price: float, atr: Optional[float] = None, direction: str = "BUY"):
        """
        Returns (quantity, stop_loss_price, target_price)
        """
        qty = calculate_risk_parity_quantity(
            price=entry_price,
            atr=atr,
            risk_rupees=self.risk_rupees,
            max_capital=self.account_capital
        )
        if atr and atr > 0:
            sl_dist = max(atr * 1.2, entry_price * STOP_LOSS_PCT)
        else:
            sl_dist = entry_price * STOP_LOSS_PCT

        tgt_dist = max(sl_dist * 2.2, entry_price * TARGET_PCT)

        if direction.upper() == "BUY":
            sl = round(entry_price - sl_dist, 2)
            tgt = round(entry_price + tgt_dist, 2)
        else:
            sl = round(entry_price + sl_dist, 2)
            tgt = round(entry_price - tgt_dist, 2)

        return qty, sl, tgt


