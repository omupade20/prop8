from dataclasses import dataclass
from typing import Optional, List, Dict
from strategy.indicators import exponential_moving_average


# ------------------------
# HTF Bias Output
# ------------------------
@dataclass
class HTFBias:
    direction: str
    strength: float
    label: str
    comment: str


# ------------------------
# HTF Bias Logic (5m candles)
# ------------------------
def get_htf_bias(
    candles_5m: List[Dict],
    vwap_value: Optional[float] = None,
    short_period: int = 21,
    long_period: int = 55,
    vwap_tolerance: float = 0.006
) -> HTFBias:
    """
    HTF bias computed using 5-minute candles.

    candles_5m example element:
    {
        "time_start": "...",
        "open": ...,
        "high": ...,
        "low": ...,
        "close": ...
    }
    """

    if not candles_5m or len(candles_5m) < long_period + 5:
        return HTFBias("NEUTRAL", 0.5, "NEUTRAL", "Insufficient 5m data")

    # Extract close prices
    prices = [c["close"] for c in candles_5m]

    ema_short = exponential_moving_average(prices, short_period)
    ema_long = exponential_moving_average(prices, long_period)

    if ema_short is None or ema_long is None:
        return HTFBias("NEUTRAL", 0.5, "NEUTRAL", "EMA unavailable")

    price = prices[-1]

    # ------------------------
    # Direction
    # ------------------------
    ema_diff = ema_short - ema_long

    if ema_diff > 0:
        direction = "BULLISH"
    elif ema_diff < 0:
        direction = "BEARISH"
    else:
        return HTFBias("NEUTRAL", 1.0, "NEUTRAL", "Flat EMA")

    # ------------------------
    # Strength (structure)
    # ------------------------
    lookback = min(20, len(prices))
    recent = prices[-lookback:]

    recent_range = max(recent) - min(recent)

    if recent_range <= 0:
        base_strength = 1.5
    else:
        base_strength = min(abs(ema_diff) / recent_range * 10.0, 6.0)

    strength = base_strength
    comment = ["EMA alignment"]

    # ------------------------
    # Trend maturity
    # ------------------------
    if len(prices) >= long_period + 10:

        past_prices = prices[:-5]

        past_short = exponential_moving_average(past_prices, short_period)
        past_long = exponential_moving_average(past_prices, long_period)

        if past_short and past_long:

            past_diff = past_short - past_long

            if direction == "BULLISH" and past_diff > 0:
                strength += 1.0
                comment.append("Trend persistence")

            if direction == "BEARISH" and past_diff < 0:
                strength += 1.0
                comment.append("Trend persistence")

    # ------------------------
    # VWAP influence
    # ------------------------
    if vwap_value:

        dist = (price - vwap_value) / vwap_value

        if direction == "BULLISH":

            if dist > vwap_tolerance:
                strength += 1.0
                comment.append("Above VWAP")

            elif dist < -vwap_tolerance:
                strength -= 1.0
                comment.append("Below VWAP pressure")

        else:

            if dist < -vwap_tolerance:
                strength += 1.0
                comment.append("Below VWAP")

            elif dist > vwap_tolerance:
                strength -= 1.0
                comment.append("Above VWAP pressure")

    # ------------------------
    # Clamp strength
    # ------------------------
    strength = max(0.5, min(round(strength, 2), 10.0))

    # ------------------------
    # Label
    # ------------------------
    if direction == "BULLISH":
        label = "BULLISH_STRONG" if strength >= 7 else "BULLISH_WEAK"
    else:
        label = "BEARISH_STRONG" if strength >= 7 else "BEARISH_WEAK"

    return HTFBias(
        direction=direction,
        strength=strength,
        label=label,
        comment=" | ".join(comment)
    )
