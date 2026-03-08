# strategy/htf_bias.py

from dataclasses import dataclass
from typing import Optional, List
from strategy.indicators import exponential_moving_average


# ------------------------
# HTF Bias Output
# ------------------------
@dataclass
class HTFBias:
    direction: str     # BULLISH | BEARISH | NEUTRAL
    strength: float    # 0 – 10
    label: str         # BULLISH_STRONG, BULLISH_WEAK, etc.
    comment: str


# ------------------------
# HTF Bias Logic
# ------------------------
def get_htf_bias(
    prices: List[float],
    vwap_value: Optional[float] = None,
    short_period: int = 34,
    long_period: int = 89,
    vwap_tolerance: float = 0.006
) -> HTFBias:
    """
    Compute an HTF directional bias (direction + confidence).
    - Uses EMA short_period vs long_period (defaults 20 / 50).
    - Strength is structural (normalized by recent range), scaled 0.5..10.
    - Returns HTFBias(direction, strength, label, comment).

    Usage notes:
    - This is a soft directional trust signal — do NOT treat as an absolute veto.
    - Pass recent prices (1-minute or aggregated HTF prices). If using 1-minute data,
      ensure the list length is >= long_period + 5 for reliable output.
    """

    # Safety: require some minimal data
    min_required = long_period + 5
    if not prices or len(prices) < min_required:
        return HTFBias("NEUTRAL", 0.5, "NEUTRAL", "Insufficient HTF data")

    # Compute EMAs (these functions may return None if not enough data)
    ema_short = exponential_moving_average(prices, short_period)
    ema_long = exponential_moving_average(prices, long_period)

    if ema_short is None or ema_long is None:
        return HTFBias("NEUTRAL", 0.5, "NEUTRAL", "EMA unavailable")

    price = prices[-1]

    # 1) EMA Direction
    ema_diff = ema_short - ema_long
    if ema_diff > 0:
        direction = "BULLISH"
    elif ema_diff < 0:
        direction = "BEARISH"
    else:
        return HTFBias("NEUTRAL", 1.0, "NEUTRAL", "Flat EMA")

    # 2) Structural strength: normalize EMA separation by recent price movement (range)
    lookback_for_range = min(20, len(prices))
    recent_slice = prices[-lookback_for_range:]
    recent_range = max(recent_slice) - min(recent_slice)
    if recent_range <= 0:
        base_strength = 1.5
    else:
        # scale so that reasonable separations give values in 0..6
        base_strength = min(abs(ema_diff) / recent_range * 10.0, 6.0)

    strength = base_strength
    comment_parts = ["EMA alignment"]

    # 3) Trend maturity (slope stability): check EMA separation some bars ago
    # Use a safe previous window only if we have enough data
    if len(prices) >= (long_period + 10):
        try:
            past_prices = prices[:-5]  # look a few bars back to test persistence
            past_ema_short = exponential_moving_average(past_prices, short_period)
            past_ema_long = exponential_moving_average(past_prices, long_period)
            if past_ema_short is not None and past_ema_long is not None:
                past_diff = past_ema_short - past_ema_long
                # If past direction matches current direction, reward maturity
                if (direction == "BULLISH" and past_diff > 0) or (direction == "BEARISH" and past_diff < 0):
                    strength += 1.0
                    comment_parts.append("EMA trend holding")
        except Exception:
            # don't fail HTF on computation errors
            pass

    # 4) VWAP context (soft)
    if vwap_value is not None and vwap_value > 0:
        dist = (price - vwap_value) / vwap_value
        # small nudges only — VWAP should not flip the HTF fully
        if direction == "BULLISH":
            if dist > vwap_tolerance:
                strength += 1.0
                comment_parts.append("Above VWAP")
            elif dist < -vwap_tolerance:
                strength -= 1.0
                comment_parts.append("Below VWAP (counter pressure)")
            else:
                comment_parts.append("Near VWAP")
        else:  # BEARISH
            if dist < -vwap_tolerance:
                strength += 1.0
                comment_parts.append("Below VWAP")
            elif dist > vwap_tolerance:
                strength -= 1.0
                comment_parts.append("Above VWAP (counter pressure)")
            else:
                comment_parts.append("Near VWAP")

    # 5) final clamp & label
    # keep within [0.5, 10.0] — avoid zero so downstream logic has room to interpret
    strength = max(0.5, min(round(strength, 2), 10.0))

    if direction == "BULLISH":
        label = "BULLISH_STRONG" if strength >= 7.0 else "BULLISH_WEAK"
    else:
        label = "BEARISH_STRONG" if strength >= 7.0 else "BEARISH_WEAK"

    return HTFBias(direction=direction, strength=strength, label=label, comment=" | ".join(comment_parts))
