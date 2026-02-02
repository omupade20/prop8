# strategy/advanced_indicators.py

from typing import List, Optional


# ---- EMA Helper ----

def _ema(series: List[float], period: int) -> Optional[float]:
    """
    Compute Exponential Moving Average.
    Returns None if not enough data.
    """
    if not series or len(series) < period:
        return None
    k = 2 / (period + 1)
    ema_val = series[0]
    for price in series[1:]:
        ema_val = (price - ema_val) * k + ema_val
    return ema_val


# ---- MACD ----

def compute_macd(
    prices: List[float],
    short_period: int = 12,
    long_period: int = 26,
    signal_period: int = 9
) -> Optional[dict]:
    """
    Compute MACD line, Signal line, and Histogram.

    MACD line = EMA(short_period) - EMA(long_period)
    Signal line = EMA of MACD line over signal_period
    Histogram = MACD line - Signal line

    Returns None if not enough data.
    """

    if not prices or len(prices) < long_period + signal_period:
        return None

    short_ema = _ema(prices, short_period)
    long_ema = _ema(prices, long_period)
    if short_ema is None or long_ema is None:
        return None

    macd_line = short_ema - long_ema

    # Build approximate MACD history for signal line
    macd_hist_series = []
    for i in range(len(prices) - long_period + 1):
        # For each position, compute MACD line
        sub = prices[: long_period + i]
        s = _ema(sub, short_period)
        l = _ema(sub, long_period)
        if s is not None and l is not None:
            macd_hist_series.append(s - l)

    if len(macd_hist_series) < signal_period:
        return None

    signal_line = _ema(macd_hist_series, signal_period)
    if signal_line is None:
        return None

    histogram = macd_line - signal_line

    return {
        "macd": round(macd_line, 6),
        "signal": round(signal_line, 6),
        "hist": round(histogram, 6),
    }


# ---- True Range / ATR (safe) ----

def compute_true_range(highs: List[float], lows: List[float], closes: List[float]) -> Optional[List[float]]:
    """
    True Range list for each bar (except the first).
    """
    if not highs or not lows or not closes or len(highs) < 2:
        return None
    tr_values = []
    for i in range(1, len(highs)):
        tr_values.append(
            max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
        )
    return tr_values


def compute_atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Optional[float]:
    """
    Average True Range (ATR).
    """
    tr_values = compute_true_range(highs, lows, closes)
    if not tr_values or len(tr_values) < period:
        return None
    return sum(tr_values[-period:]) / period


# ---- ADX Approximation (safe) ----

def compute_adx(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Optional[float]:
    """
    Simple ADX approximation (same as market_regime but usable independently).
    """
    if not highs or not lows or not closes or len(highs) < period + 1:
        return None

    plus_dm, minus_dm = [], []
    for i in range(1, len(highs)):
        up_move = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]
        plus_dm.append(max(up_move, 0) if up_move > down_move else 0)
        minus_dm.append(max(down_move, 0) if down_move > up_move else 0)

    atr_val = compute_atr(highs, lows, closes, period)
    if atr_val is None or atr_val == 0:
        return None

    plus_di = (sum(plus_dm[-period:]) / atr_val) * 100
    minus_di = (sum(minus_dm[-period:]) / atr_val) * 100

    denom = plus_di + minus_di
    if denom == 0:
        return None

    dx = abs(plus_di - minus_di) / denom * 100

    # Smooth ADX over last `period`
    adx_vals = [dx] * period
    return sum(adx_vals) / period
