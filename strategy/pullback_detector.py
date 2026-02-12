"""
Next-Generation Pullback Detector – Focused on BIG MOVES only.

Philosophy Shift:
- Not "any pullback" → only HIGH ENERGY pullbacks
- ATR-normalized rules
- Minimum 0.6–0.7% move potential
- Designed to eliminate 0.1–0.3% junk signals
"""

from typing import Optional, Dict, List

from strategy.sr_levels import compute_sr_levels, get_nearest_sr
from strategy.volume_filter import analyze_volume
from strategy.volatility_filter import compute_atr, analyze_volatility
from strategy.price_action import price_action_context


def detect_pullback_signal(
    opens: List[float],
    highs: List[float],
    lows: List[float],
    closes: List[float],
    volumes: List[float],
    htf_direction: str,
    min_bars: int = 40
) -> Optional[Dict]:
    """
    PROFESSIONAL PULLBACK DETECTOR – VERSION 2.0

    Only accepts setups capable of REAL moves.
    """

    if len(closes) < min_bars:
        return None

    last_price = closes[-1]

    # --------------------------------------------------
    # 1) STRUCTURAL LOCATION (STRONGER)
    # --------------------------------------------------

    sr = compute_sr_levels(highs, lows)

    # Stricter proximity: only very clean SR
    nearest = get_nearest_sr(last_price, sr, max_search_pct=0.015)

    if not nearest:
        return None

    trade_direction = None

    if nearest["type"] == "support" and htf_direction == "BULLISH":
        trade_direction = "LONG"

    elif nearest["type"] == "resistance" and htf_direction == "BEARISH":
        trade_direction = "SHORT"

    else:
        return None

    # --------------------------------------------------
    # 2) ATR CONTEXT (CORE FILTER)
    # --------------------------------------------------

    atr = compute_atr(highs, lows, closes)

    if not atr or atr <= 0:
        return None

    # Skip instruments with tiny daily movement ability
    if atr / max(last_price, 1e-9) < 0.0045:   # ~0.45%
        return None

    # --------------------------------------------------
    # 3) EXTENSION FILTER (STRICTER)
    # --------------------------------------------------

    recent_move = abs(closes[-1] - closes[-8])

    if recent_move > atr * 1.3:
        return None   # already too extended

    # --------------------------------------------------
    # 4) VOLATILITY QUALITY (ONLY EXPANDING)
    # --------------------------------------------------

    volat_ctx = analyze_volatility(
        current_move=closes[-1] - closes[-2],
        atr_value=atr
    )

    if volat_ctx.state != "EXPANDING":
        return None

    # --------------------------------------------------
    # 5) PRICE ACTION QUALITY (NEW MODULE)
    # --------------------------------------------------

    pa = price_action_context(
        opens=opens,
        highs=highs,
        lows=lows,
        closes=closes
    )

    if pa["quality"] != "HIGH":
        return None

    # --------------------------------------------------
    # 6) VOLUME CONFIRMATION (STRONGER)
    # --------------------------------------------------

    vol_ctx = analyze_volume(volumes, close_prices=closes)

    if vol_ctx.score < 1.0:
        return None

    # --------------------------------------------------
    # 7) MOMENTUM REQUIREMENT (ATR NORMALIZED)
    # --------------------------------------------------

    move_strength = abs(closes[-1] - closes[-6]) / atr

    if move_strength < 0.45:
        return None

    # --------------------------------------------------
    # 8) MINIMUM TARGET POTENTIAL (CRITICAL)
    # --------------------------------------------------

    # Distance to next SR must allow at least ~0.65% move
    potential = nearest["dist_pct"]

    if potential < 0.0065:
        return None

    # --------------------------------------------------
    # 9) CONFIDENCE SCORING (REBUILT)
    # --------------------------------------------------

    components = {
        "location": 0.0,
        "price_action": 0.0,
        "volume": 0.0,
        "volatility": 0.0,
        "momentum": 0.0,
        "potential": 0.0
    }

    # Location quality
    proximity_score = max(0, (0.015 - nearest["dist_pct"]) * 80)
    components["location"] = min(proximity_score, 2.0)

    # Price action score from new module
    components["price_action"] = abs(pa["score"])

    # Volume strength
    components["volume"] = vol_ctx.score

    # Volatility expansion bonus
    components["volatility"] = 1.5

    # Momentum
    components["momentum"] = min(move_strength, 2.0)

    # Target potential
    components["potential"] = min(potential * 100, 2.0)

    total_score = sum(components.values())

    # --------------------------------------------------
    # 10) FINAL CLASSIFICATION
    # --------------------------------------------------

    if total_score >= 7.0:
        signal = "CONFIRMED"
    elif total_score >= 5.0:
        signal = "POTENTIAL"
    else:
        return None

    return {
        "signal": signal,
        "direction": trade_direction,
        "score": round(total_score, 2),
        "nearest_level": nearest,
        "components": components,
        "context": {
            "volatility": volat_ctx.state,
            "volume": vol_ctx.strength,
            "price_action_quality": pa["quality"]
        },
        "reason": f"{signal}_{trade_direction}"
    }
