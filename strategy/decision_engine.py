from dataclasses import dataclass
from typing import Optional, Dict

from strategy.volume_filter import analyze_volume
from strategy.volatility_filter import analyze_volatility, compute_atr
from strategy.liquidity_filter import analyze_liquidity
from strategy.price_action import price_action_context
from strategy.sr_levels import sr_location_score
from strategy.vwap_filter import VWAPContext


# =====================================================
# OUTPUT STRUCTURE
# =====================================================

@dataclass
class DecisionResult:
    state: str                 # IGNORE | PREPARE_LONG | PREPARE_SHORT | EXECUTE_LONG | EXECUTE_SHORT
    score: float               # 0 – 10
    direction: Optional[str]
    components: Dict[str, float]
    reason: str


# =====================================================
# BIG-MOVE FOCUSED DECISION ENGINE (VERSION 2.0)
# =====================================================

def final_trade_decision(
    inst_key: str,
    prices: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    volumes: list[float],
    market_regime: str,
    htf_bias_direction: str,
    vwap_ctx: VWAPContext,
    pullback_signal: Optional[Dict],
) -> DecisionResult:

    components: Dict[str, float] = {}
    score = 0.0

    # --------------------------------------------------
    # 1) HARD STRUCTURE GATE
    # --------------------------------------------------

    if not pullback_signal:
        return DecisionResult("IGNORE", 0.0, None, {}, "no pullback setup")

    direction = pullback_signal["direction"]
    signal_type = pullback_signal["signal"]

    # POTENTIAL signals are only watchlist candidates
    if signal_type == "POTENTIAL":
        return DecisionResult(
            state=f"PREPARE_{direction}",
            score=2.0,
            direction=direction,
            components={"structure": 2.0},
            reason="potential setup – waiting for strength"
        )

    # CONFIRMED only
    components["structure"] = 4.0
    score += 4.0

    # --------------------------------------------------
    # 2) HTF ALIGNMENT (NON NEGOTIABLE)
    # --------------------------------------------------

    if direction == "LONG" and htf_bias_direction != "BULLISH":
        return DecisionResult("IGNORE", 0.0, None, {}, "htf misaligned")

    if direction == "SHORT" and htf_bias_direction != "BEARISH":
        return DecisionResult("IGNORE", 0.0, None, {}, "htf misaligned")

    components["htf"] = 2.0
    score += 2.0

    # --------------------------------------------------
    # 3) MARKET REGIME (TREND DAYS ONLY)
    # --------------------------------------------------

    if market_regime != "TRENDING":
        return DecisionResult("IGNORE", 0.0, None, {}, "not a trend regime")

    components["regime"] = 1.5
    score += 1.5

    # --------------------------------------------------
    # 4) VWAP ENVIRONMENT (STRICT)
    # --------------------------------------------------

    if direction == "LONG" and vwap_ctx.acceptance != "ABOVE":
        return DecisionResult("IGNORE", 0.0, None, {}, "not accepted above VWAP")

    if direction == "SHORT" and vwap_ctx.acceptance != "BELOW":
        return DecisionResult("IGNORE", 0.0, None, {}, "not accepted below VWAP")

    components["vwap"] = 1.5
    score += 1.5

    # --------------------------------------------------
    # 5) VOLUME QUALITY (STRONG ONLY)
    # --------------------------------------------------

    vol_ctx = analyze_volume(volumes, close_prices=closes)

    if vol_ctx.score < 1.0:
        return DecisionResult("IGNORE", 0.0, None, {}, "volume not strong enough")

    components["volume"] = vol_ctx.score
    score += vol_ctx.score

    # --------------------------------------------------
    # 6) VOLATILITY QUALITY (EXPANSION ONLY)
    # --------------------------------------------------

    atr = compute_atr(highs, lows, closes)
    move = closes[-1] - closes[-2] if len(closes) > 1 else 0.0

    volat_ctx = analyze_volatility(move, atr)

    if volat_ctx.state != "EXPANDING":
        return DecisionResult("IGNORE", 0.0, None, {}, "volatility not expanding")

    components["volatility"] = 1.5
    score += 1.5

    # --------------------------------------------------
    # 7) LIQUIDITY SAFETY
    # --------------------------------------------------

    liq_ctx = analyze_liquidity(volumes)

    if liq_ctx.score < 0:
        return DecisionResult("IGNORE", 0.0, None, {}, "illiquid instrument")

    components["liquidity"] = 1.0
    score += 1.0

    # --------------------------------------------------
    # 8) PRICE ACTION QUALITY (NEW STRICT)
    # --------------------------------------------------

    pa_ctx = price_action_context(
        opens=closes,
        highs=highs,
        lows=lows,
        closes=closes
    )

    if pa_ctx["quality"] != "HIGH":
        return DecisionResult("IGNORE", 0.0, None, {}, "price action not high quality")

    components["price_action"] = abs(pa_ctx["score"])
    score += abs(pa_ctx["score"])

    # --------------------------------------------------
    # 9) SR LOCATION EDGE
    # --------------------------------------------------

    nearest = pullback_signal.get("nearest_level")
    sr_score = sr_location_score(closes[-1], nearest, direction)

    if sr_score <= 0:
        return DecisionResult("IGNORE", 0.0, None, {}, "poor SR location")

    components["sr"] = sr_score
    score += sr_score * 1.5

    # --------------------------------------------------
    # 10) FINAL DECISION – MUCH STRICTER
    # --------------------------------------------------

    score = round(min(score, 10.0), 2)

    # Only very high quality become EXECUTE
    if score >= 8.0:
        state = f"EXECUTE_{direction}"
        reason = "institutional-grade setup"

    elif score >= 6.5:
        state = f"PREPARE_{direction}"
        reason = "good setup but needs trigger"

    else:
        state = "IGNORE"
        reason = "edge not strong enough"

    return DecisionResult(
        state=state,
        score=score,
        direction=direction if state != "IGNORE" else None,
        components=components,
        reason=reason
    )
