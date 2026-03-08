# strategy/market_regime.py

from typing import List, Optional
from dataclasses import dataclass


# =========================
# Core Calculations
# =========================

def compute_true_range(highs: List[float], lows: List[float], closes: List[float]) -> List[float]:
    if len(highs) < 2:
        return []
    tr = []
    for i in range(1, len(highs)):
        tr.append(
            max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1])
            )
        )
    return tr


def compute_atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Optional[float]:
    tr = compute_true_range(highs, lows, closes)
    if len(tr) < period:
        return None
    return sum(tr[-period:]) / period


def compute_adx(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Optional[float]:
    if len(highs) < period + 1:
        return None

    plus_dm, minus_dm = [], []

    for i in range(1, len(highs)):
        up = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]

        plus_dm.append(up if up > down and up > 0 else 0)
        minus_dm.append(down if down > up and down > 0 else 0)

    atr = compute_atr(highs, lows, closes, period)
    if atr is None or atr == 0:
        return None

    plus_di = (sum(plus_dm[-period:]) / atr) * 100
    minus_di = (sum(minus_dm[-period:]) / atr) * 100

    if plus_di + minus_di == 0:
        return 0.0

    dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    return dx


# =========================
# Regime Output
# =========================

@dataclass
class MarketRegime:
    state: str            # WEAK | COMPRESSION | EARLY_TREND | TRENDING | EXHAUSTION
    mode: str             # TREND_DAY | RANGE_DAY
    strength: float       # 0 – 10
    volatility: float     # normalized ATR
    comment: str


# =========================
# Regime Detection
# =========================

def detect_market_regime(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    index_regime: Optional["MarketRegime"] = None,
    min_bars: int = 30
) -> MarketRegime:
    """
    AUTHORITATIVE Market Regime Detector.

    Responsibilities:
    - Detect STRUCTURAL state
    - Decide TRADING MODE (TREND vs RANGE)
    - Provide STRENGTH as confidence
    """

    # ---------------------
    # Safety
    # ---------------------

    if len(highs) < min_bars or len(lows) < min_bars or len(closes) < min_bars:
        return MarketRegime(
            state="WEAK",
            mode="RANGE_DAY",
            strength=0.5,
            volatility=0.0,
            comment="Insufficient data"
        )

    adx = compute_adx(highs, lows, closes)
    atr = compute_atr(highs, lows, closes)

    if adx is None or atr is None:
        return MarketRegime(
            state="WEAK",
            mode="RANGE_DAY",
            strength=0.5,
            volatility=0.0,
            comment="Indicators unavailable"
        )

    # ---------------------
    # Volatility Normalization
    # ---------------------

    recent_n = min(10, len(closes))
    avg_price = sum(closes[-recent_n:]) / recent_n if recent_n > 0 else 1.0
    vol_norm = atr / avg_price if avg_price > 0 else 0.0

    # ---------------------
    # Range Comparison
    # ---------------------

    recent_range = max(highs[-10:]) - min(lows[-10:])
    prev_highs = highs[-20:-10] if len(highs) >= 20 else highs[:len(highs)//2]
    prev_lows = lows[-20:-10] if len(lows) >= 20 else lows[:len(lows)//2]
    prev_range = (max(prev_highs) - min(prev_lows)) if prev_highs and prev_lows else 0.0
    if prev_range <= 0:
        prev_range = max(recent_range * 0.8, 1e-9)

    def cap(x: float) -> float:
        return max(0.0, min(10.0, x))

    # =====================
    # REGIME LOGIC
    # =====================

    # EARLY TREND
    if adx >= 18 and recent_range > prev_range * 1.3:
        strength = cap(4.5 + (adx - 18) * 0.2)
        return MarketRegime(
            state="EARLY_TREND",
            mode="TREND_DAY",
            strength=strength,
            volatility=vol_norm,
            comment="Fresh expansion with momentum"
        )

    # TRENDING
    if adx >= 28:
        strength = cap(6.5 + (adx - 28) * 0.15)
        return MarketRegime(
            state="TRENDING",
            mode="TREND_DAY",
            strength=strength,
            volatility=vol_norm,
            comment="Established directional trend"
        )

    # COMPRESSION
    if recent_range < prev_range * 0.7:
        strength = cap(2.5 + (prev_range - recent_range) / (prev_range + 1e-9))
        return MarketRegime(
            state="COMPRESSION",
            mode="RANGE_DAY",
            strength=strength,
            volatility=vol_norm,
            comment="Volatility contraction"
        )

    # EXHAUSTION
    if adx > 28 and recent_range < prev_range * 0.85 and vol_norm < 0.008:
        strength = cap(3.5 + (adx - 28) * 0.1)
        return MarketRegime(
            state="EXHAUSTION",
            mode="RANGE_DAY",
            strength=strength,
            volatility=vol_norm,
            comment="Trend losing energy"
        )

    # DEFAULT: WEAK / CHOPPY
    strength = cap(1.5 + (adx / 30.0) * 1.2)
    regime = MarketRegime(
        state="WEAK",
        mode="RANGE_DAY",
        strength=strength,
        volatility=vol_norm,
        comment="Low momentum / mixed structure"
    )

    # ---------------------
    # Optional Index Bias
    # ---------------------

    if index_regime:
        try:
            if index_regime.mode == "TREND_DAY":
                regime.strength = cap(regime.strength + min(1.2, index_regime.strength * 0.15))
                regime.comment += " | aligned with index trend"
            else:
                regime.strength = cap(regime.strength - 0.7)
                regime.comment += " | index not trending"
        except Exception:
            pass

    return regime
