"""
Institutional-grade price action helpers focused on BIG MOVES, not micro reactions.

Upgraded Philosophy:
- ATR normalized logic (ignore tiny moves)
- Real candle construction
- Momentum + expansion aware
- Designed to filter 0.1–0.3% noise and focus on 0.6–1% type moves
"""

from typing import List, Optional, Dict


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Optional[float]:
    if len(closes) < period + 1:
        return None

    trs = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1])
        )
        trs.append(tr)

    if len(trs) < period:
        return None

    return sum(trs[-period:]) / period


# ------------------------------------------------------------
# Stronger Pullback Detection
# ------------------------------------------------------------

def detect_quality_pullback(
    closes: List[float],
    highs: List[float],
    lows: List[float],
    ema_short: Optional[float],
    ema_long: Optional[float],
    atr_value: Optional[float],
    lookback: int = 8
) -> Optional[Dict]:
    """
    Detects MEANINGFUL pullbacks only – ignores tiny noise.

    Rules:
    - Pullback must be at least 0.25 ATR deep
    - But not more than 1.1 ATR (avoid overextended reversals)
    - Must occur in aligned EMA trend
    """

    if len(closes) < lookback + 2 or atr_value is None or atr_value <= 0:
        return None

    last = closes[-1]
    swing_high = max(closes[-(lookback + 1):-1])
    swing_low = min(closes[-(lookback + 1):-1])

    up_depth = swing_high - last
    down_depth = last - swing_low

    min_depth = atr_value * 0.25
    max_depth = atr_value * 1.1

    trend = None
    if ema_short is not None and ema_long is not None:
        if ema_short > ema_long:
            trend = "UP"
        elif ema_short < ema_long:
            trend = "DOWN"

    # Only meaningful pullbacks
    if trend == "UP" and min_depth <= up_depth <= max_depth:
        return {"type": "PULLBACK_UP", "depth": round(up_depth / max(last, 1e-9), 4)}

    if trend == "DOWN" and min_depth <= down_depth <= max_depth:
        return {"type": "PULLBACK_DOWN", "depth": round(down_depth / max(last, 1e-9), 4)}

    return None


# ------------------------------------------------------------
# ATR-Normalized Rejection Logic
# ------------------------------------------------------------

def rejection_info(
    open_p: float,
    high: float,
    low: float,
    close: float,
    atr_value: Optional[float]
) -> Dict:
    """
    Rejection detection normalized by ATR instead of raw %.

    Only consider rejections that are at least 0.2 ATR in size.
    """

    total_range = max(high - low, 1e-9)
    body = abs(close - open_p)

    upper_wick = max(0.0, high - max(close, open_p))
    lower_wick = max(0.0, min(close, open_p) - low)

    result = {
        "rejection_type": None,
        "rejection_score": 0.0
    }

    if atr_value is None or atr_value <= 0:
        return result

    min_wick = atr_value * 0.2

    if lower_wick >= min_wick and lower_wick > body * 1.4:
        result["rejection_type"] = "BULLISH"
        result["rejection_score"] = min(1.0, lower_wick / (atr_value * 1.5))

    elif upper_wick >= min_wick and upper_wick > body * 1.4:
        result["rejection_type"] = "BEARISH"
        result["rejection_score"] = min(1.0, upper_wick / (atr_value * 1.5))

    return result


# ------------------------------------------------------------
# Momentum Filter
# ------------------------------------------------------------

def momentum_quality(
    closes: List[float],
    atr_value: Optional[float],
    bars: int = 5
) -> float:
    """
    Returns momentum strength normalized to ATR.
    Only values above 0.4 ATR are considered meaningful.
    """

    if len(closes) < bars + 1 or atr_value is None:
        return 0.0

    move = closes[-1] - closes[-(bars + 1)]
    strength = abs(move) / max(atr_value, 1e-9)

    return round(strength, 3)


# ------------------------------------------------------------
# Unified Price Action Context (REDESIGNED)
# ------------------------------------------------------------

def price_action_context(
    opens: List[float],
    highs: List[float],
    lows: List[float],
    closes: List[float],
    ema_short: Optional[float] = None,
    ema_long: Optional[float] = None
) -> Dict:
    """
    New generation price action context focused on BIG MOVES.

    Output:
    {
        "pullback": str | None,
        "rejection_type": str | None,
        "momentum": float,
        "score": float (-2..+2),
        "quality": "HIGH/MEDIUM/LOW",
        "comment": str
    }
    """

    result = {
        "pullback": None,
        "rejection_type": None,
        "momentum": 0.0,
        "score": 0.0,
        "quality": "LOW",
        "comment": ""
    }

    if len(closes) < 15:
        result["comment"] = "insufficient data"
        return result

    atr_value = _atr(highs, lows, closes)

    # Pullback detection
    pb = detect_quality_pullback(
        closes, highs, lows, ema_short, ema_long, atr_value
    )

    if pb:
        result["pullback"] = pb["type"]

    # Rejection on last bar (REAL OHLC used)
    rej = rejection_info(opens[-1], highs[-1], lows[-1], closes[-1], atr_value)

    result["rejection_type"] = rej["rejection_type"]

    # Momentum check
    mom = momentum_quality(closes, atr_value)
    result["momentum"] = mom

    # Scoring focused on BIG MOVE POTENTIAL
    score = 0.0
    comments = []

    if pb:
        score += 0.8
        comments.append("quality_pullback")

    if rej["rejection_type"] == "BULLISH":
        score += 0.9
        comments.append("bullish_rejection")
    elif rej["rejection_type"] == "BEARISH":
        score -= 0.9
        comments.append("bearish_rejection")

    # Momentum must be at least 0.4 ATR to be meaningful
    if mom >= 0.4:
        score += 0.7
        comments.append("good_momentum")
    else:
        score -= 0.4
        comments.append("weak_momentum")

    # EMA alignment bonus
    if ema_short and ema_long:
        if ema_short > ema_long and pb and pb["type"] == "PULLBACK_UP":
            score += 0.4
            comments.append("ema_aligned_bull")
        elif ema_short < ema_long and pb and pb["type"] == "PULLBACK_DOWN":
            score -= 0.4
            comments.append("ema_aligned_bear")

    result["score"] = round(score, 3)

    # Quality classification
    if abs(score) >= 1.6:
        result["quality"] = "HIGH"
    elif abs(score) >= 1.0:
        result["quality"] = "MEDIUM"
    else:
        result["quality"] = "LOW"

    result["comment"] = " | ".join(comments)
    return result
