# strategy/mtf_context.py

from dataclasses import dataclass
from typing import List, Optional


# =========================
# MTF Context Output
# =========================

@dataclass
class MTFContext:
    direction: str        # BULLISH / BEARISH / NEUTRAL
    strength: float       # 0 – 2 (raw strength)
    confidence: str       # HIGH / MEDIUM / LOW
    conflict: bool        # True if 5m & 15m disagree
    comment: str


# =========================
# Candle Helpers
# =========================

def _is_bullish(candle: dict) -> bool:
    return candle and candle.get("close", 0) > candle.get("open", 0)


def _is_bearish(candle: dict) -> bool:
    return candle and candle.get("close", 0) < candle.get("open", 0)


# =========================
# Persistence Logic
# =========================

def _persistence_score(history: List[dict]) -> float:
    """
    Persistence bonus based on last 3 candles.
    Returns 0.0 / 0.3 / 0.6
    """
    if not history or len(history) < 2:
        return 0.0

    last = history[-3:]
    bull = sum(1 for c in last if _is_bullish(c))
    bear = sum(1 for c in last if _is_bearish(c))

    if bull == 3 or bear == 3:
        return 0.6
    if bull >= 2 or bear >= 2:
        return 0.3
    return 0.0


# =========================
# Main Analyzer
# =========================

def analyze_mtf(
    candle_5m: Optional[dict],
    candle_15m: Optional[dict],
    history_5m: Optional[List[dict]] = None,
    history_15m: Optional[List[dict]] = None
) -> MTFContext:
    """
    Multi-Timeframe Context Analyzer (AUTHORITATIVE).

    Purpose:
    - Decide higher-timeframe DIRECTION
    - Measure CONFIDENCE
    - Detect CONFLICT explicitly

    This output IS MEANT to be used as a DIRECTIONAL GATE upstream.
    """

    score = 0.0
    comments = []
    conflict = False

    # ---------------------
    # Base Direction Votes
    # ---------------------

    if candle_5m:
        if _is_bullish(candle_5m):
            score += 0.7
            comments.append("5m bullish")
        elif _is_bearish(candle_5m):
            score -= 0.7
            comments.append("5m bearish")

    if candle_15m:
        if _is_bullish(candle_15m):
            score += 1.3   # 15m has more authority
            comments.append("15m bullish")
        elif _is_bearish(candle_15m):
            score -= 1.3
            comments.append("15m bearish")

    # ---------------------
    # Conflict Detection
    # ---------------------

    if candle_5m and candle_15m:
        if (_is_bullish(candle_5m) and _is_bearish(candle_15m)) or \
           (_is_bearish(candle_5m) and _is_bullish(candle_15m)):
            conflict = True
            score *= 0.5   # dampen conviction heavily
            comments.append("5m/15m conflict")

    # ---------------------
    # Persistence Bonus
    # ---------------------

    if history_5m:
        p5 = _persistence_score(history_5m)
        if p5:
            score += p5
            comments.append(f"5m persistence +{p5}")

    if history_15m:
        p15 = _persistence_score(history_15m)
        if p15:
            score += p15
            comments.append(f"15m persistence +{p15}")

    # ---------------------
    # Final Direction
    # ---------------------

    if abs(score) < 0.4:
        direction = "NEUTRAL"
    elif score > 0:
        direction = "BULLISH"
    else:
        direction = "BEARISH"

    strength = round(min(abs(score), 2.0), 2)

    # ---------------------
    # Confidence Buckets
    # ---------------------

    if strength >= 1.6 and not conflict:
        confidence = "HIGH"
    elif strength >= 1.0:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    comment = " | ".join(comments) if comments else "No HTF structure"

    return MTFContext(
        direction=direction,
        strength=strength,
        confidence=confidence,
        conflict=conflict,
        comment=comment
    )
