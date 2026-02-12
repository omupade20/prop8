# strategy/volume_context.py

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class VolumeContext:
    score: float               # -2 to +2
    strength: str              # STRONG | MODERATE | WEAK | NONE
    trend: str                 # RISING | FALLING | FLAT
    comment: str


def analyze_volume(
    volume_history: List[float],
    close_prices: Optional[List[float]] = None,
    lookback: int = 20,
    rising_bars: int = 4
) -> VolumeContext:
    """
    Institutional volume analysis producing:
      - score (-2..+2): additive input to decision_engine
      - strength: descriptive label
      - trend: volume trend structure
      - comment: context for logs

    Score interpretation:
      +2 strong confirmation
      +0.5–1.5 moderate support
      around 0 neutral
      negative indicates weak / absorbing volume
    """

    if not volume_history or len(volume_history) < lookback + rising_bars:
        return VolumeContext(0.0, "NONE", "FLAT", "Insufficient volume data")

    recent = volume_history[-lookback:]
    avg_volume = sum(recent) / lookback if lookback > 0 else 0
    current_volume = volume_history[-1]

    # ----------------------
    # 1) Volume Strength Relative to Average
    # ----------------------
    if avg_volume > 0:
        rel = current_volume / avg_volume
    else:
        rel = 1.0

    if rel >= 1.8:
        strength = "STRONG"
        score = 2.0
    elif rel >= 1.4:
        strength = "MODERATE"
        score = 1.2
    elif rel >= 0.95:
        strength = "WEAK"
        score = 0.4
    else:
        strength = "NONE"
        score = -0.5

    # ----------------------
    # 2) Recent Volume Trend
    # ----------------------
    last_n = volume_history[-rising_bars:]
    if all(last_n[i] > last_n[i - 1] for i in range(1, len(last_n))):
        trend = "RISING"
        score += 0.5
    elif all(last_n[i] < last_n[i - 1] for i in range(1, len(last_n))):
        trend = "FALLING"
        score -= 0.5
    else:
        trend = "FLAT"

    # ----------------------
    # 3) Price–Volume Relationship
    # ----------------------
    comment = ""
    if close_prices and len(close_prices) >= rising_bars:
        price_move = close_prices[-1] - close_prices[-rising_bars]
        # small threshold guard for price motion
        if abs(price_move) < 0.002 * close_prices[-1]:
            # Large volume but little price move → possible absorption
            if strength in ("STRONG", "MODERATE"):
                score -= 0.7
                comment = "absorption suspicion"
            else:
                comment = "volume, no price move"
        else:
            comment = "volume supports price move"
    else:
        comment = "volume only"

    # clamp final score
    final_score = max(min(score, 2.0), -2.0)

    return VolumeContext(
        score=round(final_score, 2),
        strength=strength,
        trend=trend,
        comment=comment
    )


# backward-compatible legacy boolean
def volume_spike_confirmed(
    volume_history,
    threshold_multiplier: float = 1.25,
    lookback: int = 20,
    rising_bars: int = 4
) -> bool:
    """
    Legacy boolean volume confirmation.
    """
    ctx = analyze_volume(volume_history, lookback=lookback, rising_bars=rising_bars)
    return ctx.score > 0.5
