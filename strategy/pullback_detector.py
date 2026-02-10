# strategy/pullback_detector.py

from typing import Optional, Dict, List
from strategy.sr_levels import compute_sr_levels, get_nearest_sr
from strategy.volume_filter import analyze_volume
from strategy.volatility_filter import compute_atr, analyze_volatility
from strategy.price_action import rejection_info


def detect_pullback_signal(
    prices: List[float],
    highs: List[float],
    lows: List[float],
    closes: List[float],
    volumes: List[float],
    htf_direction: str,
    max_proximity: float = 0.018,
    min_bars: int = 35
) -> Optional[Dict]:
    """
    PROFESSIONAL PULLBACK DETECTOR

    Purpose:
    - Detect HIGH QUALITY mean reversion entries
    - Avoid chasing extended moves
    - Enter at smart locations

    CORE LOGIC:
    LONG  -> price NEAR SUPPORT with confirmation
    SHORT -> price NEAR RESISTANCE with confirmation
    """

    if len(prices) < min_bars:
        return None

    last_price = closes[-1]

    # --------------------------------------------------
    # 1) STRUCTURAL LOCATION (WHERE ARE WE?)
    # --------------------------------------------------

    sr = compute_sr_levels(highs, lows)
    nearest = get_nearest_sr(last_price, sr, max_search_pct=max_proximity)

    if not nearest:
        return None

    # Direction intent from SR
    trade_direction = None

    if nearest["type"] == "support" and htf_direction == "BULLISH":
        trade_direction = "LONG"

    elif nearest["type"] == "resistance" and htf_direction == "BEARISH":
        trade_direction = "SHORT"

    else:
        return None

    # --------------------------------------------------
    # 2) EXTENSION FILTER (AVOID CHASING)
    # --------------------------------------------------

    recent_move = abs(closes[-1] - closes[-6])

    atr = compute_atr(highs, lows, closes)

    if atr and recent_move > atr * 1.6:
        return None  # too extended

    # --------------------------------------------------
    # 3) VOLATILITY QUALITY CHECK
    # --------------------------------------------------

    volat_ctx = analyze_volatility(
        current_move=closes[-1] - closes[-2],
        atr_value=atr
    )

    if volat_ctx.state in ["CONTRACTING", "EXHAUSTION"]:
        return None

    # --------------------------------------------------
    # 4) PRICE ACTION CONFIRMATION
    # --------------------------------------------------

    last_bar_rejection = rejection_info(
        closes[-2], highs[-1], lows[-1], closes[-1]
    )

    price_reaction = False

    if trade_direction == "LONG" and last_bar_rejection["rejection_type"] == "BULLISH":
        price_reaction = True

    if trade_direction == "SHORT" and last_bar_rejection["rejection_type"] == "BEARISH":
        price_reaction = True

    # Basic directional reaction
    if trade_direction == "LONG" and closes[-1] > closes[-3]:
        price_reaction = True

    if trade_direction == "SHORT" and closes[-1] < closes[-3]:
        price_reaction = True

    # --------------------------------------------------
    # 5) VOLUME CONFIRMATION
    # --------------------------------------------------

    vol_ctx = analyze_volume(volumes, close_prices=closes)

    volume_ok = vol_ctx.score >= 0.6

    # --------------------------------------------------
    # 6) MOMENTUM FILTER (DON’T BUY WEAK)
    # --------------------------------------------------

    short_term_trend = closes[-1] - closes[-5]

    momentum_ok = False

    if trade_direction == "LONG" and short_term_trend > 0:
        momentum_ok = True

    if trade_direction == "SHORT" and short_term_trend < 0:
        momentum_ok = True

    # --------------------------------------------------
    # 7) CONFIDENCE SCORING SYSTEM
    # --------------------------------------------------

    components = {
        "location": 0.0,
        "price_action": 0.0,
        "volume": 0.0,
        "volatility": 0.0,
        "momentum": 0.0
    }

    # Location quality
    proximity_score = max(0, (max_proximity - nearest["dist_pct"]) * 60)
    components["location"] = min(proximity_score, 2.0)

    if price_reaction:
        components["price_action"] = 2.0

    if volume_ok:
        components["volume"] = 1.5

    if volat_ctx.state == "EXPANDING":
        components["volatility"] = 1.2

    if momentum_ok:
        components["momentum"] = 1.3

    total_score = sum(components.values())

    # --------------------------------------------------
    # 8) CLASSIFICATION
    # --------------------------------------------------

    if total_score >= 5.0:
        signal = "CONFIRMED"
    elif total_score >= 3.0:
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
            "rejection": last_bar_rejection["rejection_type"]
        },
        "reason": f"{signal}_{trade_direction}"
    }
