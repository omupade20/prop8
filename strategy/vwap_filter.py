# strategy/vwap_filter.py

from collections import deque
from dataclasses import dataclass
from typing import Optional

# =========================
# VWAP Context Output
# =========================

@dataclass
class VWAPContext:
    vwap: Optional[float]
    distance_pct: float             # price minus VWAP as percent
    slope: float                    # VWAP slope over recent history
    acceptance: str                 # ABOVE | BELOW | NEAR
    pressure: str                   # BUYING | SELLING | NEUTRAL
    score: float                    # -2 to +2 (for decision_engine scoring)
    comment: str


# =========================
# VWAP Calculator
# =========================

class VWAPCalculator:
    """
    Intraday VWAP calculator + context.

    Usage:
      - reset() at session start
      - update(price, volume) per tick or per bar
      - get_vwap(): current VWAP
      - get_context(price): returns VWAPContext with score
    """

    def __init__(self, window: Optional[int] = None, slope_window: int = 5):
        self.window = window
        self.slope_window = slope_window

        self.price_volume_sum = 0.0
        self.volume_sum = 0.0

        self.vwap_history = deque(maxlen=slope_window)

        if window:
            self.price_volume_deque = deque(maxlen=window)
            self.volume_deque = deque(maxlen=window)

        self.reset()

    def reset(self):
        """
        Reset VWAP state at session start.
        """
        self.price_volume_sum = 0.0
        self.volume_sum = 0.0
        self.vwap_history.clear()

        if hasattr(self, "price_volume_deque"):
            self.price_volume_deque.clear()
            self.volume_deque.clear()

    def update(self, price: float, volume: float) -> Optional[float]:
        """
        Update VWAP running sums with new price & volume.
        If window is set, it rolls using deques.
        """
        if price is None or volume is None or volume <= 0:
            return None

        if self.window:
            self.price_volume_deque.append(price * volume)
            self.volume_deque.append(volume)
            # recompute sums from deques
            self.price_volume_sum = sum(self.price_volume_deque)
            self.volume_sum = sum(self.volume_deque)
        else:
            # accumulate full session sums
            self.price_volume_sum += price * volume
            self.volume_sum += volume

        if self.volume_sum <= 0:
            return None

        vwap = self.price_volume_sum / self.volume_sum
        self.vwap_history.append(vwap)
        return vwap

    def get_vwap(self) -> Optional[float]:
        """
        Return the latest VWAP (None if no volume yet).
        """
        if self.volume_sum <= 0:
            return None
        return self.price_volume_sum / self.volume_sum

    # =========================
    # VWAP Intelligence
    # =========================

    def get_context(self, price: float) -> VWAPContext:
        """
        Return a VWAPContext object for the given price.
        VWAPContext includes:
          - vwap: current VWAP
          - distance_pct: (price - vwap) / vwap * 100
          - slope: recent VWAP slope
          - acceptance: ABOVE/BELOW/NEAR
          - pressure: BUYING/SELLING/NEUTRAL
          - score: numeric score (-2..+2) for decision_engine
          - comment: text of interpretation
        """
        vwap = self.get_vwap()

        # no VWAP available
        if vwap is None or price is None:
            return VWAPContext(
                vwap=None,
                distance_pct=0.0,
                slope=0.0,
                acceptance="NEAR",
                pressure="NEUTRAL",
                score=0.0,
                comment="VWAP not available"
            )

        # distance from VWAP in percent
        distance_pct = (price - vwap) / vwap * 100.0

        # slope over recent history
        if len(self.vwap_history) >= 2:
            slope = self.vwap_history[-1] - self.vwap_history[0]
        else:
            slope = 0.0

        # classify acceptance zone
        if distance_pct > 0.2:  # price > 0.2% above VWAP
            acceptance = "ABOVE"
        elif distance_pct < -0.2:
            acceptance = "BELOW"
        else:
            acceptance = "NEAR"

        # pressure interpretation
        if acceptance == "ABOVE" and slope > 0:
            pressure = "BUYING"
            score = 1.5
            comment = "Accepted above VWAP with rising slope"
        elif acceptance == "BELOW" and slope < 0:
            pressure = "SELLING"
            score = -1.5
            comment = "Accepted below VWAP with falling slope"
        elif acceptance == "NEAR":
            pressure = "NEUTRAL"
            score = 0.0
            comment = "Near VWAP (magnet zone)"
        else:
            pressure = "NEUTRAL"
            # slight penalty when price is above but slope falling, or below but slope rising
            score = -0.5
            comment = "VWAP uncertainty or weak pressure"

        # clamp score
        score = max(min(score, 2.0), -2.0)

        return VWAPContext(
            vwap=round(vwap, 6),
            distance_pct=round(distance_pct, 3),
            slope=round(slope, 6),
            acceptance=acceptance,
            pressure=pressure,
            score=score,
            comment=comment
        )
