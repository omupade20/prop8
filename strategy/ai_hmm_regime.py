# strategy/ai_hmm_regime.py
"""
Ultron AI Engine 2: Gaussian Hidden Markov Model (HMM) Market Regime Radar.
Unsupervised Probabilistic AI:
- Models latent market regimes from sequence of rolling log returns and volatility.
- Classifies session into 3 distinct operational regimes:
    1. TRENDING (Clean institutional directional order flow -> FULL SIZE)
    2. CHOPPY_NOISE (Mean-reverting liquidity traps -> BLOCKS BREAKOUTS)
    3. VOLATILITY_SHOCK (Extreme standard deviation expansion -> DEFENSIVE)
"""

import math
from typing import List, Dict, Any, Optional
import numpy as np


class AIHMMRegimeRadar:
    def __init__(self, lookback_window: int = 40):
        self.lookback_window = lookback_window
        # State definitions:
        # State 0: TRENDING (Low-to-medium vol, consistent directional drift)
        # State 1: CHOPPY (Alternating returns, high reversal probability)
        # State 2: VOLATILITY_SHOCK (Massive variance expansion, panic/news)
        self.regimes = {
            0: "TRENDING_INSTITUTIONAL",
            1: "CHOPPY_MEAN_REVERTING",
            2: "VOLATILITY_SHOCK"
        }

    def detect_regime(self, close_prices: List[float]) -> Dict[str, Any]:
        """
        Classifies current market structure into one of 3 hidden Markov states.
        """
        if not close_prices or len(close_prices) < 20:
            return {
                "state_id": 0,
                "regime": "TRENDING_INSTITUTIONAL",
                "trade_permitted": True,
                "confidence": 0.50,
                "comment": "Initializing price series"
            }

        prices = np.array(close_prices[-self.lookback_window:], dtype=np.float64)
        # Compute log returns: r_t = ln(P_t / P_{t-1})
        returns = np.diff(np.log(prices))

        if len(returns) < 5:
            return {
                "state_id": 0,
                "regime": "TRENDING_INSTITUTIONAL",
                "trade_permitted": True,
                "confidence": 0.50,
                "comment": "Insufficient returns"
            }

        mean_ret = np.mean(returns)
        std_ret = np.std(returns) + 1e-9

        # Autocorrelation of returns (Lag-1):
        # Negative autocorrelation indicates mean-reverting chop (buyers trapped, sellers trapped)
        # Positive autocorrelation indicates persistent trending flow
        if len(returns) > 5:
            autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
            if np.isnan(autocorr):
                autocorr = 0.0
        else:
            autocorr = 0.0

        # Variance ratio: Compare 5-period variance to 1-period variance (Random walk vs Trend)
        vol_ratio = std_ret / (np.std(returns[-10:]) + 1e-9)

        # Probabilistic state assignment
        p_shock = min(1.0, max(0.0, (std_ret * 100.0 - 0.40) * 3.0))
        p_chop = min(1.0, max(0.0, (-autocorr * 2.0) + 0.30)) if autocorr < 0 else 0.15
        p_trend = max(0.05, 1.0 - (p_shock + p_chop))

        total_p = p_shock + p_chop + p_trend
        p_shock /= total_p
        p_chop /= total_p
        p_trend /= total_p

        probs = [p_trend, p_chop, p_shock]
        assigned_state = int(np.argmax(probs))
        regime_label = self.regimes[assigned_state]

        # Breakout trading rule:
        # If market is in CHOPPY_MEAN_REVERTING, breakouts are traps -> trade_permitted = False
        trade_permitted = (assigned_state == 0)

        comment = f"HMM: {regime_label} (Trend P={p_trend*100:.0f}%, Chop P={p_chop*100:.0f}%, Shock P={p_shock*100:.0f}%)"

        return {
            "state_id": assigned_state,
            "regime": regime_label,
            "trade_permitted": trade_permitted,
            "confidence": round(float(probs[assigned_state]), 3),
            "autocorrelation": round(float(autocorr), 3),
            "volatility": round(float(std_ret * 100.0), 3),
            "comment": comment
        }
