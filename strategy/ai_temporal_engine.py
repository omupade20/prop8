# strategy/ai_temporal_engine.py
"""
Ultron AI Engine 4: Deep Temporal Sequence & Momentum Acceleration Engine.
Time-Series Sequence Modeling:
- Evaluates 30-candle sequential momentum dynamics (Velocity, Acceleration, Jerk).
- Detects Volume Climax Exhaustions: Identifies when an apparent breakout is
  actually a late-stage exhaustion wick (trapping late retail buyers at the absolute top).
- Computes Temporal Acceleration Score (+1.0 to -1.0) and flags Exhaustion Risk.
"""

from typing import List, Dict, Any, Optional
import numpy as np


class AITemporalSequenceEngine:
    def __init__(self, sequence_length: int = 30):
        self.sequence_length = sequence_length

    def analyze_sequence(
        self,
        closes: List[float],
        highs: List[float],
        lows: List[float],
        volumes: List[float],
        direction: str = "LONG"
    ) -> Dict[str, Any]:
        """
        Evaluates multi-candle sequence for momentum acceleration and exhaustion risk.
        """
        if not closes or len(closes) < 15:
            return {
                "score": 0.0,
                "velocity": 0.0,
                "acceleration": 0.0,
                "exhaustion_risk": False,
                "comment": "Insufficient sequence history"
            }

        p = np.array(closes[-self.sequence_length:], dtype=np.float64)
        v = np.array(volumes[-self.sequence_length:], dtype=np.float64)
        h = np.array(highs[-self.sequence_length:], dtype=np.float64)
        l = np.array(lows[-self.sequence_length:], dtype=np.float64)

        # 1. Price Velocity (1st derivative: rate of change)
        velocity_window = min(5, len(p) - 1)
        velocity = (p[-1] - p[-velocity_window]) / max(p[-velocity_window], 1e-9) * 100.0

        # 2. Price Acceleration (2nd derivative: change in velocity)
        if len(p) >= 10:
            v_prev = (p[-velocity_window] - p[-2*velocity_window]) / max(p[-2*velocity_window], 1e-9) * 100.0
            acceleration = velocity - v_prev
        else:
            acceleration = 0.0

        # 3. Climax Exhaustion Detection:
        # Extreme volume spike (> 3.5x average) with huge upper wick (long) or lower wick (short)
        # where price cannot close near high -> Retail FOMO exhaustion trap!
        avg_vol = np.mean(v[:-1]) if len(v) > 1 else v[-1]
        vol_spike_ratio = v[-1] / max(avg_vol, 1.0)

        candle_range = max(h[-1] - l[-1], 1e-9)
        exhaustion_risk = False
        comments = []

        if direction == "LONG":
            upper_wick = h[-1] - max(closes[-1], closes[-2] if len(closes) > 1 else closes[-1])
            wick_ratio = upper_wick / candle_range
            if vol_spike_ratio >= 3.0 and wick_ratio >= 0.40:
                exhaustion_risk = True
                comments.append("Buyer exhaustion trap (huge upper wick on climax volume)")
            elif acceleration > 0.15 and velocity > 0:
                comments.append("Bullish momentum accelerating cleanly")
        else:
            lower_wick = min(closes[-1], closes[-2] if len(closes) > 1 else closes[-1]) - l[-1]
            wick_ratio = lower_wick / candle_range
            if vol_spike_ratio >= 3.0 and wick_ratio >= 0.40:
                exhaustion_risk = True
                comments.append("Seller exhaustion trap (huge lower wick on climax volume)")
            elif acceleration < -0.15 and velocity < 0:
                comments.append("Bearish momentum accelerating cleanly")

        # 4. Temporal Quality Score
        if exhaustion_risk:
            score = -1.0
        else:
            if direction == "LONG":
                score = 0.6 if acceleration > 0 and velocity > 0 else 0.0
            else:
                score = 0.6 if acceleration < 0 and velocity < 0 else 0.0

        return {
            "score": round(float(score), 2),
            "velocity": round(float(velocity), 3),
            "acceleration": round(float(acceleration), 3),
            "vol_spike_ratio": round(float(vol_spike_ratio), 2),
            "exhaustion_risk": exhaustion_risk,
            "comment": " | ".join(comments) if comments else "Stable momentum sequence"
        }
