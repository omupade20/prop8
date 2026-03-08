# strategy/liquidity_context.py

from dataclasses import dataclass
from typing import List, Optional


# =========================
# Liquidity Context Output
# =========================

@dataclass
class LiquidityContext:
    score: float            # -2 to +2
    level: str              # HIGH | MEDIUM | LOW | ILLIQUID
    avg_volume: float
    consistency: str        # STABLE | UNSTABLE
    comment: str


# =========================
# Liquidity Intelligence
# =========================

def analyze_liquidity(
    volume_history: List[float],
    min_avg_volume: int = 400_000,
    lookback: int = 30
) -> LiquidityContext:
    """
    Intraday liquidity analysis for MIS / cash segment.

    Interpret liquidity as:
      - HIGH / MEDIUM / LOW / ILLIQUID
    with an associated score in [-2 .. +2].
    """

    # Safety: not enough data
    if not volume_history or len(volume_history) < lookback:
        return LiquidityContext(
            score=-2.0,
            level="ILLIQUID",
            avg_volume=0.0,
            consistency="UNSTABLE",
            comment="Insufficient volume history"
        )

    recent = volume_history[-lookback:]
    avg_vol = sum(recent) / lookback

    # -----------------------------
    # 1️⃣ Liquidity Level
    # -----------------------------
    # Simple thresholds (tunable)
    if avg_vol >= min_avg_volume * 4:
        level = "HIGH"
        base_score = 2.0
        comment_base = "Very high average volume"
    elif avg_vol >= min_avg_volume * 2:
        level = "MEDIUM"
        base_score = 1.2
        comment_base = "Moderate average volume"
    elif avg_vol >= min_avg_volume:
        level = "LOW"
        base_score = 0.5
        comment_base = "Low average volume"
    else:
        level = "ILLIQUID"
        base_score = -1.5
        comment_base = "Below minimum volume threshold"

    # -----------------------------
    # 2️⃣ Consistency Check
    # -----------------------------
    non_zero_bars = sum(1 for v in recent if v > 0)
    consistency_ratio = non_zero_bars / lookback

    if consistency_ratio < 0.80:
        consistency = "UNSTABLE"
        score = base_score - 0.8
        comment = f"{comment_base} | Inconsistent intraday volume"
    else:
        consistency = "STABLE"
        score = base_score
        comment = f"{comment_base} | Stable intraday volume"

    # clamp final score
    score = max(min(score, 2.0), -2.0)

    return LiquidityContext(
        score=round(score, 2),
        level=level,
        avg_volume=round(avg_vol),
        consistency=consistency,
        comment=comment
    )


# =========================
# Backward Compatibility
# =========================

def is_liquid(volume_history: List[float], min_avg_volume: int = 250000, lookback: int = 30) -> bool:
    """
    Legacy boolean liquidity filter.

    Returns True if liquidity context score >= 0.
    """
    ctx = analyze_liquidity(volume_history, min_avg_volume=min_avg_volume, lookback=lookback)
    return ctx.score >= 0
