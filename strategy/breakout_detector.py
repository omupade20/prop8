# strategy/breakout_detector.py

from typing import Optional, Dict
from strategy.volume_filter import volume_spike_confirmed
from strategy.volatility_filter import compute_atr


# =========================
# Compression Detection
# =========================

def detect_compression(
    prices: list[float],
    lookback: int = 20,
    compression_ratio: float = 0.65
) -> bool:
    """
    Detect volatility contraction prior to expansion.
    """
    if len(prices) < lookback * 2:
        return False

    recent = prices[-lookback:]
    previous = prices[-lookback * 2:-lookback]

    recent_range = max(recent) - min(recent)
    previous_range = max(previous) - min(previous)

    if previous_range <= 0:
        return False

    return recent_range < previous_range * compression_ratio


# =========================
# Breakout / Intent Detector
# =========================

def breakout_signal(
    inst_key: str,
    prices: list[float],
    volume_history: Optional[list[float]] = None,
    high_prices: Optional[list[float]] = None,
    low_prices: Optional[list[float]] = None,
    close_prices: Optional[list[float]] = None,
    breakout_pct: float = 0.0015,
    vol_threshold: float = 1.2,
    atr_multiplier: float = 0.8
) -> Optional[Dict]:
    """
    AUTHORITATIVE Breakout / Intent Detector.

    Responsibilities:
    - Detect REAL expansion intent
    - Classify POTENTIAL vs CONFIRMED
    - DO NOT decide direction by opinion
    """

    if len(prices) < 30:
        return None

    last_price = prices[-1]
    prev_price = prices[-2]

    # ---------------------
    # Define Reference Range
    # ---------------------

    base = prices[-21:-1]   # last 20 completed bars
    range_high = max(base)
    range_low = min(base)
    range_span = max(range_high - range_low, 1e-9)

    # ---------------------
    # Direction ONLY from Range Break
    # ---------------------

    direction = None
    if last_price > range_high:
        direction = "LONG"
    elif last_price < range_low:
        direction = "SHORT"
    else:
        # no breakout, no intent
        return None

    # ---------------------
    # Intent Components
    # ---------------------

    components = {
        "compression": 0.0,
        "atr_expansion": 0.0,
        "volume": 0.0
    }

    # Compression (contextual, not mandatory)
    if detect_compression(prices):
        components["compression"] = 1.0

    # ATR Expansion (MANDATORY for CONFIRMED)
    atr_ok = False
    atr_val = None
    if high_prices and low_prices and close_prices:
        atr_val = compute_atr(high_prices, low_prices, close_prices, period=14)
        if atr_val and abs(last_price - prev_price) >= atr_val * atr_multiplier:
            atr_ok = True
            components["atr_expansion"] = 1.5

    # Volume Participation (MANDATORY for CONFIRMED)
    volume_ok = False
    if volume_history and len(volume_history) >= 20:
        if volume_spike_confirmed(volume_history, threshold_multiplier=vol_threshold):
            volume_ok = True
            components["volume"] = 1.2

    # ---------------------
    # Intent Score (informational only)
    # ---------------------

    intent_score = sum(components.values())

    # ---------------------
    # Classification Logic
    # ---------------------

    signal = "POTENTIAL"
    confirmed = False

    # CONFIRMED requires:
    # - price outside range
    # - ATR or Volume confirmation
    if atr_ok or volume_ok:
        signal = "CONFIRMED"
        confirmed = True

    raw_flag = f"{signal}_{direction}"

    return {
        "signal": signal,
        "direction": direction,
        "intent_score": round(intent_score, 2),
        "components": components,
        "range_high": range_high,
        "range_low": range_low,
        "last_price": last_price,
        "atr": round(atr_val, 6) if atr_val else None,
        "atr_ok": atr_ok,
        "volume_ok": volume_ok,
        "raw_flag": raw_flag,
        "reason": raw_flag
    }
