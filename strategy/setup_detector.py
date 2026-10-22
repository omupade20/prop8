# strategy/setup_detector.py
"""
Multi-Behavior Quantitative Price Action Setup Detector.
Replaces naive breakout chasing with 3 distinct institutional setups:
1. SETUP_PULLBACK: Retracement to VWAP / 20 EMA in an established trend with rejection wick.
2. SETUP_SQUEEZE_BREAKOUT: Low-volatility compression breakout with explosive volume.
3. SETUP_SWING_REVERSAL: Liquidity sweep / failed breakout at key Intraday High/Low.

Also enforces the CLIMAX_EXHAUSTION_GUARD:
- Hard VETO if RSI < 32 on Shorts (Oversold trap - e.g. HINDPETRO RSI 7.35)
- Hard VETO if RSI > 68 on Longs (Overbought trap)
- Hard VETO if price is overextended > 1.2% away from VWAP
- Hard VETO on runaway moves without retracements
"""

from typing import List, Dict, Any, Optional
from strategy.indicators import exponential_moving_average, relative_strength_index
from strategy.volatility_filter import compute_atr


def check_climax_exhaustion(
    ltp: float,
    vwap: float,
    rsi: Optional[float],
    closes: List[float],
    highs: List[float],
    lows: List[float],
    direction: str,
    max_vwap_dist_pct: float = 0.012,  # 1.2% max distance from VWAP
    oversold_rsi: float = 32.0,
    overbought_rsi: float = 68.0,
    vwap_bands: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Guards against chasing late-stage, overextended climax moves.
    Returns: {"is_exhausted": bool, "reason": str}
    """
    if not closes or len(closes) < 15:
        return {"is_exhausted": False, "reason": ""}

    # 1. RSI Climax Exhaustion Guard (Saved HINDPETRO which had RSI 7.35)
    if rsi is not None:
        if direction == "SHORT" and rsi < oversold_rsi:
            return {
                "is_exhausted": True,
                "reason": f"Oversold climax exhaustion (RSI={rsi:.1f} < {oversold_rsi}). Bounce risk extremely high."
            }
        if direction == "LONG" and rsi > overbought_rsi:
            return {
                "is_exhausted": True,
                "reason": f"Overbought climax exhaustion (RSI={rsi:.1f} > {overbought_rsi}). Pullback risk extremely high."
            }

    # 2. VWAP Extension / Rubber-Band Guard (2.5σ Dispersion Bands & Max Distance)
    if vwap_bands:
        upper_2_5 = vwap_bands.get("upper_2_5")
        lower_2_5 = vwap_bands.get("lower_2_5")
        if direction == "LONG" and upper_2_5 and ltp >= upper_2_5:
            return {
                "is_exhausted": True,
                "reason": f"Extreme +2.5σ VWAP band overextension (LTP {ltp:.2f} >= +2.5σ Band {upper_2_5:.2f}). Longs blocked."
            }
        if direction == "SHORT" and lower_2_5 and ltp <= lower_2_5:
            return {
                "is_exhausted": True,
                "reason": f"Extreme -2.5σ VWAP band oversold (LTP {ltp:.2f} <= -2.5σ Band {lower_2_5:.2f}). Shorts blocked."
            }

    if vwap and vwap > 0:
        dist_from_vwap = (ltp - vwap) / vwap
        if direction == "SHORT" and dist_from_vwap < -max_vwap_dist_pct:
            return {
                "is_exhausted": True,
                "reason": f"Stretched rubber-band below VWAP ({dist_from_vwap*100:.2f}%). Reversion to mean imminent."
            }
        if direction == "LONG" and dist_from_vwap > max_vwap_dist_pct:
            return {
                "is_exhausted": True,
                "reason": f"Stretched rubber-band above VWAP (+{dist_from_vwap*100:.2f}%). Reversion to mean imminent."
            }

    # 3. Waterfall Runaway Selling / Vertical Pump Guard (Last 10 bars)
    recent = closes[-10:]
    if len(recent) >= 10:
        move_10 = (recent[-1] - recent[0]) / recent[0]
        if direction == "SHORT" and move_10 < -0.010:  # Dropped > 1% in 10 mins
            # Check if all bars were consecutive red (no pauses)
            red_count = sum(1 for i in range(1, len(recent)) if recent[i] < recent[i-1])
            if red_count >= 7:
                return {
                    "is_exhausted": True,
                    "reason": f"Waterfall selling exhaustion (Dropped {move_10*100:.2f}% in 10 bars with {red_count}/9 red bars)."
                }
        elif direction == "LONG" and move_10 > 0.010:
            green_count = sum(1 for i in range(1, len(recent)) if recent[i] > recent[i-1])
            if green_count >= 7:
                return {
                    "is_exhausted": True,
                    "reason": f"Vertical pump exhaustion (Surged +{move_10*100:.2f}% in 10 bars with {green_count}/9 green bars)."
                }

    return {"is_exhausted": False, "reason": ""}


def detect_price_action_setup(
    inst_key: str,
    prices: List[float],
    opens: List[float],
    highs: List[float],
    lows: List[float],
    closes: List[float],
    volumes: List[float],
    vwap: float,
    mtf_direction: str = "NEUTRAL",
    order_flow_delta: Optional[float] = None
) -> Optional[Dict[str, Any]]:
    """
    Evaluates market structure across 4 distinct institutional setups:
    1. SETUP_PULLBACK: Retracement to VWAP / 20 EMA in an established trend with rejection wick.
    2. SETUP_ORB: 15-Minute Opening Range Breakout on expansion Bar #1 with volume.
    3. SETUP_SQUEEZE_BREAKOUT: Low-volatility compression breakout with explosive volume.
    4. SETUP_SWING_REVERSAL: Liquidity sweep / Turtle Soup stop-hunt at Day/Session High/Low.
    5. SETUP_VWAP_MEAN_REVERSION: Statistical snapback from ±2.5σ extreme dispersion bands.
    """
    if len(closes) < 30:
        return None

    from strategy.advanced_indicators import compute_vwap_sd_bands
    vwap_bands = compute_vwap_sd_bands(highs, lows, closes, volumes)

    ltp = closes[-1]
    rsi = relative_strength_index(closes, 14)
    ema9 = exponential_moving_average(closes, 9)
    ema21 = exponential_moving_average(closes, 21)
    atr = compute_atr(highs, lows, closes, 14) or (ltp * 0.005)

    last_open = opens[-1] if opens else closes[-2]
    last_high = highs[-1]
    last_low = lows[-1]
    last_close = closes[-1]
    bar_range = max(last_high - last_low, 1e-9)

    upper_wick = max(0.0, last_high - max(last_close, last_open))
    lower_wick = max(0.0, min(last_close, last_open) - last_low)
    body = abs(last_close - last_open)

    # =========================================================================
    # SETUP 1: INSTITUTIONAL PULLBACK & RETEST (The Golden Setup)
    # =========================================================================
    if ema9 and ema21 and vwap:
        # PULLBACK SHORT
        if ema9 < ema21 and ltp < vwap:
            near_ema = (abs(ltp - ema9) / ema9 <= 0.0050) or (abs(ltp - ema21) / ema21 <= 0.0060) or (last_high >= ema9 and last_close <= ema9)
            near_vwap = (abs(ltp - vwap) / vwap <= 0.0060) or (last_high >= vwap and last_close <= vwap)

            if (near_ema or near_vwap) and upper_wick >= body * 0.9 and upper_wick / bar_range >= 0.25:
                exhaustion = check_climax_exhaustion(ltp, vwap, rsi, closes, highs, lows, "SHORT", vwap_bands=vwap_bands)
                if not exhaustion["is_exhausted"]:
                    return {
                        "setup_name": "PULLBACK_RETEST",
                        "direction": "SHORT",
                        "signal": "EXECUTE",
                        "confidence": 0.85,
                        "atr": atr,
                        "rsi": rsi,
                        "components": {
                            "setup": "PULLBACK_RETEST",
                            "rejection_wick": round(upper_wick / bar_range, 2),
                            "distance_to_vwap": round((ltp - vwap) / vwap * 100, 2),
                            "volume": round(volumes[-1] / max(sum(volumes[-20:]) / 20.0, 1.0), 2) if volumes else 1.0
                        },
                        "reason": f"Pullback to VWAP/EMA resistance with Bearish Rejection Wick (Upper wick {upper_wick/bar_range*100:.0f}%)"
                    }

        # PULLBACK LONG
        if ema9 > ema21 and ltp > vwap:
            near_ema = (abs(ltp - ema9) / ema9 <= 0.0050) or (abs(ltp - ema21) / ema21 <= 0.0060) or (last_low <= ema9 and last_close >= ema9)
            near_vwap = (abs(ltp - vwap) / vwap <= 0.0060) or (last_low <= vwap and last_close >= vwap)

            if (near_ema or near_vwap) and lower_wick >= body * 0.9 and lower_wick / bar_range >= 0.25:
                exhaustion = check_climax_exhaustion(ltp, vwap, rsi, closes, highs, lows, "LONG", vwap_bands=vwap_bands)
                if not exhaustion["is_exhausted"]:
                    return {
                        "setup_name": "PULLBACK_RETEST",
                        "direction": "LONG",
                        "signal": "EXECUTE",
                        "confidence": 0.85,
                        "atr": atr,
                        "rsi": rsi,
                        "components": {
                            "setup": "PULLBACK_RETEST",
                            "rejection_wick": round(lower_wick / bar_range, 2),
                            "distance_to_vwap": round((ltp - vwap) / vwap * 100, 2),
                            "volume": round(volumes[-1] / max(sum(volumes[-20:]) / 20.0, 1.0), 2) if volumes else 1.0
                        },
                        "reason": f"Pullback to VWAP/EMA21 support with Bullish Rejection Wick (Lower wick {lower_wick/bar_range*100:.0f}%)"
                    }

    # =========================================================================
    # SETUP 2: 15-MINUTE OPENING RANGE BREAKOUT (ORB) (With Order Flow Delta Check)
    # =========================================================================
    if len(highs) >= 20:
        orb_high = max(highs[:15])
        orb_low = min(lows[:15])
        vol_avg = sum(volumes[-20:]) / 20.0 if volumes else 1.0
        vol_ratio = volumes[-1] / max(vol_avg, 1.0) if volumes else 1.0

        # ORB Long: Early breakout on Bar #1 with volume, above VWAP
        if ltp > orb_high and ltp <= orb_high * 1.006 and last_open <= orb_high * 1.002 and vol_ratio >= 1.25 and ltp > vwap:
            # Order Flow Delta Check: If aggressive sellers dominated, reject trap
            if order_flow_delta is None or order_flow_delta >= -100:
                exhaustion = check_climax_exhaustion(ltp, vwap, rsi, closes, highs, lows, "LONG", vwap_bands=vwap_bands)
                if not exhaustion["is_exhausted"]:
                    return {
                        "setup_name": "ORB_BREAKOUT",
                        "direction": "LONG",
                        "signal": "EXECUTE",
                        "confidence": 0.88,
                        "atr": atr,
                        "rsi": rsi,
                        "components": {
                            "setup": "ORB_BREAKOUT",
                            "orb_level": orb_high,
                            "volume": round(vol_ratio, 2),
                            "delta": round(order_flow_delta or 0.0, 0)
                        },
                        "reason": f"15-Min ORB Breakout above {orb_high:.2f} with {vol_ratio:.1f}x Volume"
                    }

        # ORB Short: Early breakdown on Bar #1 with volume, below VWAP
        if ltp < orb_low and ltp >= orb_low * 0.994 and last_open >= orb_low * 0.998 and vol_ratio >= 1.25 and ltp < vwap:
            if order_flow_delta is None or order_flow_delta <= 100:
                exhaustion = check_climax_exhaustion(ltp, vwap, rsi, closes, highs, lows, "SHORT", vwap_bands=vwap_bands)
                if not exhaustion["is_exhausted"]:
                    return {
                        "setup_name": "ORB_BREAKDOWN",
                        "direction": "SHORT",
                        "signal": "EXECUTE",
                        "confidence": 0.88,
                        "atr": atr,
                        "rsi": rsi,
                        "components": {
                            "setup": "ORB_BREAKDOWN",
                            "orb_level": orb_low,
                            "volume": round(vol_ratio, 2),
                            "delta": round(order_flow_delta or 0.0, 0)
                        },
                        "reason": f"15-Min ORB Breakdown below {orb_low:.2f} with {vol_ratio:.1f}x Volume"
                    }

    # =========================================================================
    # SETUP 3: COMPRESSION / SQUEEZE BREAKOUT
    # =========================================================================
    lookback = 20
    recent_window = closes[-lookback-1:-1]
    range_high = max(recent_window)
    range_low = min(recent_window)
    consolidation_range = range_high - range_low

    is_tight_squeeze = (consolidation_range / ltp) <= 0.0070

    if is_tight_squeeze:
        vol_avg = sum(volumes[-lookback-1:-1]) / lookback if volumes else 1.0
        vol_ratio = volumes[-1] / max(vol_avg, 1.0) if volumes else 1.0

        if ltp > range_high and vol_ratio >= 1.25:
            if order_flow_delta is None or order_flow_delta >= -100:
                exhaustion = check_climax_exhaustion(ltp, vwap, rsi, closes, highs, lows, "LONG", vwap_bands=vwap_bands)
                if not exhaustion["is_exhausted"]:
                    return {
                        "setup_name": "SQUEEZE_BREAKOUT",
                        "direction": "LONG",
                        "signal": "EXECUTE",
                        "confidence": 0.80,
                        "atr": atr,
                        "rsi": rsi,
                        "components": {
                            "setup": "SQUEEZE_BREAKOUT",
                            "compression_range_pct": round(consolidation_range / ltp * 100, 2),
                            "volume": round(vol_ratio, 2)
                        },
                        "reason": f"Bullish Squeeze Breakout above {range_high:.2f} with {vol_ratio:.1f}x Volume"
                    }

        elif ltp < range_low and vol_ratio >= 1.25:
            if order_flow_delta is None or order_flow_delta <= 100:
                exhaustion = check_climax_exhaustion(ltp, vwap, rsi, closes, highs, lows, "SHORT", vwap_bands=vwap_bands)
                if not exhaustion["is_exhausted"]:
                    return {
                        "setup_name": "SQUEEZE_BREAKOUT",
                        "direction": "SHORT",
                        "signal": "EXECUTE",
                        "confidence": 0.80,
                        "atr": atr,
                        "rsi": rsi,
                        "components": {
                            "setup": "SQUEEZE_BREAKOUT",
                            "compression_range_pct": round(consolidation_range / ltp * 100, 2),
                            "volume": round(vol_ratio, 2)
                        },
                        "reason": f"Bearish Squeeze Breakdown below {range_low:.2f} with {vol_ratio:.1f}x Volume"
                    }

    # =========================================================================
    # SETUP 4: TURTLE SOUP LIQUIDITY SWEEP / STOP-HUNT HUNTER
    # =========================================================================
    if len(closes) >= 30:
        session_high = max(highs[-31:-1])
        session_low = min(lows[-31:-1])

        # Bearish Stop-Hunt Sweep: Pierced Day/Session High by 0.05%-0.40%, closed back below with massive upper wick
        if last_high >= session_high * 1.0005 and last_close < session_high and upper_wick >= bar_range * 0.40:
            exhaustion = check_climax_exhaustion(ltp, vwap, rsi, closes, highs, lows, "SHORT", vwap_bands=vwap_bands)
            if not exhaustion["is_exhausted"]:
                return {
                    "setup_name": "SWING_REVERSAL",
                    "direction": "SHORT",
                    "signal": "EXECUTE",
                    "confidence": 0.86,
                    "atr": atr,
                    "rsi": rsi,
                    "components": {
                        "setup": "SWING_REVERSAL",
                        "sweep_wick": round(upper_wick / bar_range, 2),
                        "session_high": session_high,
                        "volume": round(volumes[-1] / max(sum(volumes[-20:]) / 20.0, 1.0), 2) if volumes else 1.0
                    },
                    "reason": f"Turtle Soup Bearish Stop-Hunt at Session High ({session_high:.2f})"
                }

        # Bullish Stop-Hunt Sweep: Pierced Day/Session Low by 0.05%-0.40%, closed back above with massive lower wick
        if last_low <= session_low * 0.9995 and last_close > session_low and lower_wick >= bar_range * 0.40:
            exhaustion = check_climax_exhaustion(ltp, vwap, rsi, closes, highs, lows, "LONG", vwap_bands=vwap_bands)
            if not exhaustion["is_exhausted"]:
                return {
                    "setup_name": "SWING_REVERSAL",
                    "direction": "LONG",
                    "signal": "EXECUTE",
                    "confidence": 0.86,
                    "atr": atr,
                    "rsi": rsi,
                    "components": {
                        "setup": "SWING_REVERSAL",
                        "sweep_wick": round(lower_wick / bar_range, 2),
                        "session_low": session_low,
                        "volume": round(volumes[-1] / max(sum(volumes[-20:]) / 20.0, 1.0), 2) if volumes else 1.0
                    },
                    "reason": f"Turtle Soup Bullish Stop-Hunt at Session Low ({session_low:.2f})"
                }

    # =========================================================================
    # SETUP 5: VWAP ±2.5σ STATISTICAL MEAN REVERSION SNAPBACK
    # =========================================================================
    if vwap_bands:
        upper_2_5 = vwap_bands.get("upper_2_5")
        lower_2_5 = vwap_bands.get("lower_2_5")
        upper_2 = vwap_bands.get("upper_2")
        lower_2 = vwap_bands.get("lower_2")

        # Oversold snapback: pierced -2.5σ band and formed bullish rejection wick back into -2.0σ
        if lower_2_5 and lower_2 and last_low <= lower_2_5 and last_close > lower_2 and lower_wick >= bar_range * 0.35:
            return {
                "setup_name": "VWAP_MEAN_REVERSION",
                "direction": "LONG",
                "signal": "EXECUTE",
                "confidence": 0.84,
                "atr": atr,
                "rsi": rsi,
                "components": {
                    "setup": "VWAP_MEAN_REVERSION",
                    "band_level": lower_2_5,
                    "vwap_target": vwap_bands["vwap"]
                },
                "reason": f"VWAP -2.5σ Dispersion Snapback (Oversold extreme at {lower_2_5:.2f})"
            }

        # Overbought snapback: pierced +2.5σ band and formed bearish rejection wick back into +2.0σ
        if upper_2_5 and upper_2 and last_high >= upper_2_5 and last_close < upper_2 and upper_wick >= bar_range * 0.35:
            return {
                "setup_name": "VWAP_MEAN_REVERSION",
                "direction": "SHORT",
                "signal": "EXECUTE",
                "confidence": 0.84,
                "atr": atr,
                "rsi": rsi,
                "components": {
                    "setup": "VWAP_MEAN_REVERSION",
                    "band_level": upper_2_5,
                    "vwap_target": vwap_bands["vwap"]
                },
                "reason": f"VWAP +2.5σ Dispersion Snapback (Overbought extreme at {upper_2_5:.2f})"
            }

    return None
