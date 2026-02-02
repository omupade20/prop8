# strategy/price_action.py
"""
Price action helpers for intraday use.

Functions:
- detect_pullback_in_trend(...)  -> identifies small pullbacks inside a trend (PULLBACK_UP / PULLBACK_DOWN / None)
- rejection_info(...)            -> detects rejection wicks and returns a score 0..1
- price_action_context(...)      -> combined context used by decision_engine:
                                   { pullback: str|None,
                                     pullback_depth: float,
                                     rejection_type: str|None,
                                     rejection_score: float,
                                     score: float,   # -1.0 .. +1.0 (positive supports LONG)
                                     comment: str }
Design: conservative, additive (soft), and safe for intraday.
"""

from typing import List, Optional, Dict


def _safe_last(seq: List[float], idx: int = -1) -> Optional[float]:
    try:
        return seq[idx]
    except Exception:
        return None


def detect_pullback_in_trend(
    prices: List[float],
    ema_short: Optional[float] = None,
    ema_long: Optional[float] = None,
    lookback: int = 6,
    max_depth_pct: float = 0.006
) -> Optional[Dict]:
    """
    Detects a shallow pullback inside a trend.
    - prices: list of closes (oldest->newest)
    - ema_short / ema_long: optional numeric EMAs (most recent values) to determine trend direction
    - lookback: number of bars used to define local swing (default 6)
    - max_depth_pct: maximum pullback depth (fraction) to still call it a 'safe' pullback

    Returns dict:
      { "type": "PULLBACK_UP"|"PULLBACK_DOWN"|None,
        "depth": float (0..1) }
    or None if not enough data.
    """
    if not prices or len(prices) < lookback + 1:
        return None

    last = prices[-1]
    window = prices[-(lookback + 1):-1]  # exclude last bar when computing recent swing
    if not window:
        return None

    recent_high = max(window)
    recent_low = min(window)

    # compute depth relative to immediate recent swing
    if recent_high <= 0 or recent_low <= 0:
        return None

    pullback_up = (recent_high - last) / recent_high  # positive if price pulled back from high
    pullback_down = (last - recent_low) / recent_low  # positive if price rebounded from low

    # trend inference (prefer EMA if provided)
    trend = None  # "UP" / "DOWN" / None
    if ema_short is not None and ema_long is not None:
        if ema_short > ema_long:
            trend = "UP"
        elif ema_short < ema_long:
            trend = "DOWN"

    # Accept only shallow pullbacks inside the matching trend
    if trend == "UP" and 0 < pullback_up <= max_depth_pct:
        return {"type": "PULLBACK_UP", "depth": round(pullback_up, 6)}
    if trend == "DOWN" and 0 < pullback_down <= max_depth_pct:
        return {"type": "PULLBACK_DOWN", "depth": round(pullback_down, 6)}

    # If trend unknown, allow a looser check (but be conservative)
    if trend is None:
        if 0 < pullback_up <= max_depth_pct * 0.8:
            return {"type": "PULLBACK_UP", "depth": round(pullback_up, 6)}
        if 0 < pullback_down <= max_depth_pct * 0.8:
            return {"type": "PULLBACK_DOWN", "depth": round(pullback_down, 6)}

    return None


def rejection_info(open_p: float, high: float, low: float, close: float) -> Dict:
    """
    Quantifies rejection wicks for the last bar.
    Returns:
      { "rejection_type": "BULLISH"|"BEARISH"|None,
        "rejection_score": 0.0..1.0,
        "upper_wick": float,
        "lower_wick": float }
    Logic:
      - Compute upper and lower wick sizes relative to bar range.
      - If one wick >> body, it's a rejection in opposite direction of the wick.
      - Score is normalized to 0..1.
    """
    body = abs(close - open_p)
    total_range = max(high - low, 1e-9)

    upper_wick = max(0.0, high - max(close, open_p))
    lower_wick = max(0.0, min(close, open_p) - low)

    # normalized measures (0..1)
    upper_rel = upper_wick / total_range
    lower_rel = lower_wick / total_range
    body_rel = body / total_range

    rejection_type = None
    rejection_score = 0.0

    # bullish rejection: long lower wick (buyers rejected lower prices)
    if lower_rel > body_rel * 1.5 and lower_rel > 0.12:
        rejection_type = "BULLISH"
        # score grows with lower_rel up to ~0.8
        rejection_score = min(1.0, (lower_rel - 0.12) / 0.6)
    # bearish rejection: long upper wick (sellers rejected higher prices)
    elif upper_rel > body_rel * 1.5 and upper_rel > 0.12:
        rejection_type = "BEARISH"
        rejection_score = min(1.0, (upper_rel - 0.12) / 0.6)

    # small noise guard
    if rejection_score < 0.05:
        rejection_score = 0.0
        rejection_type = None

    return {
        "rejection_type": rejection_type,
        "rejection_score": round(rejection_score, 3),
        "upper_wick": round(upper_wick, 6),
        "lower_wick": round(lower_wick, 6),
        "body": round(body, 6),
        "range": round(total_range, 6)
    }


def price_action_context(
    prices: List[float],
    highs: List[float],
    lows: List[float],
    opens: List[float],
    closes: List[float],
    ema_short: Optional[float] = None,
    ema_long: Optional[float] = None
) -> Dict:
    """
    Returns a combined price-action context:
      {
        "pullback": "PULLBACK_UP"/"PULLBACK_DOWN"/None,
        "pullback_depth": float,
        "rejection_type": str|None,
        "rejection_score": float,
        "score": float,   # -1.0 .. +1.0 (positive supports LONG)
        "comment": str
      }

    How score is computed (conservative):
      - Pullback in trend gives +0.25 (LONG) or -0.25 (SHORT)
      - Strong rejection supporting direction gives +0.4 etc.
      - Conflicting signals reduce score.
    """
    result = {
        "pullback": None,
        "pullback_depth": 0.0,
        "rejection_type": None,
        "rejection_score": 0.0,
        "score": 0.0,
        "comment": ""
    }

    # Basic safety
    if not prices or not highs or not lows or not closes or len(prices) < 6:
        result["comment"] = "insufficient data"
        return result

    # pullback detection (prefer closes list)
    pb = detect_pullback_in_trend(prices, ema_short=ema_short, ema_long=ema_long)
    if pb:
        result["pullback"] = pb["type"]
        result["pullback_depth"] = pb["depth"]

    # rejection on last bar
    last_open = opens[-1]
    last_high = highs[-1]
    last_low = lows[-1]
    last_close = closes[-1]
    rej = rejection_info(last_open, last_high, last_low, last_close)
    result["rejection_type"] = rej["rejection_type"]
    result["rejection_score"] = rej["rejection_score"]

    # score composition
    score = 0.0
    comments = []

    # Pullback helps entries in trend
    if result["pullback"] == "PULLBACK_UP":
        score += 0.25
        comments.append("pullback_up")
    elif result["pullback"] == "PULLBACK_DOWN":
        score -= 0.25
        comments.append("pullback_down")

    # Rejection: bullish rejection supports LONG, bearish supports SHORT
    if rej["rejection_type"] == "BULLISH":
        score += 0.4 * rej["rejection_score"]
        comments.append(f"bullish_rejection+{rej['rejection_score']}")
    elif rej["rejection_type"] == "BEARISH":
        score -= 0.4 * rej["rejection_score"]
        comments.append(f"bearish_rejection-{rej['rejection_score']}")

    # EMA trend consistency: if EMA provided and matches pullback / rejection, boost slightly
    if ema_short is not None and ema_long is not None:
        if ema_short > ema_long:
            # bullish environment
            if result["pullback"] == "PULLBACK_UP" or rej["rejection_type"] == "BULLISH":
                score += 0.15
                comments.append("ema_align_bull")
            # if bearish rejection in bullish EMA, penalize
            if rej["rejection_type"] == "BEARISH":
                score -= 0.2
                comments.append("ema_conflict_bear")
        elif ema_short < ema_long:
            if result["pullback"] == "PULLBACK_DOWN" or rej["rejection_type"] == "BEARISH":
                score -= 0.15
                comments.append("ema_align_bear")
            if rej["rejection_type"] == "BULLISH":
                score += 0.0
                comments.append("ema_conflict_bull")

    # clamp the score to -1..+1 (conservative)
    if score > 1.0:
        score = 1.0
    if score < -1.0:
        score = -1.0

    result["score"] = round(score, 3)
    result["comment"] = " | ".join(comments) if comments else "no_pa"
    return result
