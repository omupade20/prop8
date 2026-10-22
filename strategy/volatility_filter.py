# strategy/volatility_context.py
"""
Volatility context (STRICT, continuation-focused).

Purpose:
- CONFIRM real expansion
- PENALIZE noisy spikes
- FILTER choppy conditions

Score range: -1.5 .. +1.5 (conservative)
"""

from dataclasses import dataclass
from typing import List, Optional


# =========================
# Core ATR Calculations
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


def compute_atr(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    period: int = 14
) -> Optional[float]:
    tr = compute_true_range(highs, lows, closes)
    if len(tr) < period:
        return None
    return sum(tr[-period:]) / period


# =========================
# Volatility Context Output
# =========================

@dataclass
class VolatilityContext:
    state: str          # CONTRACTING | BUILDING | EXPANDING | EXHAUSTION
    score: float        # -1.5 .. +1.5
    atr: float
    move_pct_atr: float
    comment: str


# =========================
# Volatility Intelligence
# =========================

def analyze_volatility(
    current_move: float,
    atr_value: Optional[float],
    atr_history: Optional[List[float]] = None
) -> VolatilityContext:
    """
    Institutional volatility analysis.

    Rules:
    - <0.6 ATR = noise
    - 0.6–1.1 ATR = building
    - 1.1–1.6 ATR = healthy expansion
    - >1.6 ATR = spike / exhaustion risk
    """

    if atr_value is None or atr_value <= 0:
        return VolatilityContext(
            "UNKNOWN", 0.0, 0.0, 0.0, "ATR unavailable"
        )

    move_pct_atr = abs(current_move) / atr_value

    # ----------------------
    # 1️⃣ Contracting / Noise
    # ----------------------
    if move_pct_atr < 0.6:
        return VolatilityContext(
            state="CONTRACTING",
            score=-0.6,
            atr=round(atr_value, 6),
            move_pct_atr=round(move_pct_atr, 2),
            comment="volatility_too_low"
        )

    # ----------------------
    # 2️⃣ Building Phase
    # ----------------------
    if move_pct_atr < 1.2:
        return VolatilityContext(
            state="BUILDING",
            score=0.2,
            atr=round(atr_value, 6),
            move_pct_atr=round(move_pct_atr, 2),
            comment="volatility_building"
        )

    # ----------------------
    # 3️⃣ Healthy Expansion
    # ----------------------
    if move_pct_atr < 1.8:
        score = 1.1
        comment = "healthy_expansion"

        # ATR rising confirmation (optional)
        if atr_history and len(atr_history) >= 5:
            if atr_history[-1] > atr_history[-3]:
                score += 0.2
                comment += "_atr_rising"

        return VolatilityContext(
            state="EXPANDING",
            score=min(score, 1.5),
            atr=round(atr_value, 6),
            move_pct_atr=round(move_pct_atr, 2),
            comment=comment
        )

    # ----------------------
    # 4️⃣ Spike / Exhaustion
    # ----------------------
    return VolatilityContext(
        state="EXHAUSTION",
        score=-1.0,
        atr=round(atr_value, 6),
        move_pct_atr=round(move_pct_atr, 2),
        comment="volatility_spike_risk"
    )


# =========================
# Backward Compatibility
# =========================

def volatility_breakout_confirmed(
    current_move: float,
    atr_value: Optional[float],
    atr_multiplier: float = 1.1
) -> bool:
    """
    STRICT legacy confirmation.
    """
    if atr_value is None:
        return False
    return abs(current_move) >= atr_value * atr_multiplier
