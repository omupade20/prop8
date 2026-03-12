# strategy/decision_engine.py

from dataclasses import dataclass
from typing import Optional, Dict

from strategy.volume_filter import analyze_volume
from strategy.volatility_filter import analyze_volatility, compute_atr
from strategy.liquidity_filter import analyze_liquidity
from strategy.price_action import price_action_context
from strategy.sr_levels import sr_location_score
from strategy.vwap_filter import VWAPContext


# =========================
# Output Structure
# =========================

@dataclass
class DecisionResult:
    state: str                 # IGNORE | PREPARE_LONG | PREPARE_SHORT | EXECUTE_LONG | EXECUTE_SHORT
    score: float               # 0 – 10
    direction: Optional[str]
    components: Dict[str, float]
    reason: str


# =========================
# NEW PULLBACK BASED ENGINE
# =========================

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

    # ==================================================
    # 1️⃣ STRUCTURE GATE (MOST IMPORTANT)
    # ==================================================

    if not pullback_signal:
        return DecisionResult("IGNORE", 0.0, None, {}, "no pullback setup")

    direction = pullback_signal["direction"]
    signal_type = pullback_signal["signal"]

    # Potential setups only PREPARE
    if signal_type == "POTENTIAL":
        components["structure"] = 1.5
        return DecisionResult(
            state=f"PREPARE_{direction}",
            score=1.5,
            direction=direction,
            components=components,
            reason="potential pullback"
        )

    # CONFIRMED pullback gets structural priority
    components["structure"] = 3.0
    score += 3.0

    # ==================================================
    # 2️⃣ HIGHER TIMEFRAME AUTHORITY
    # ==================================================

    if direction == "LONG" and htf_bias_direction != "BULLISH":
        return DecisionResult("IGNORE", 0.0, None, {}, "htf not bullish")

    if direction == "SHORT" and htf_bias_direction != "BEARISH":
        return DecisionResult("IGNORE", 0.0, None, {}, "htf not bearish")

    components["htf"] = 1.5
    score += 1.5

    # ==================================================
    # 3️⃣ MARKET REGIME GATE
    # ==================================================

    if market_regime in ("WEAK", "COMPRESSION"):
        return DecisionResult("IGNORE", 0.0, None, {}, "bad market regime")

    if market_regime == "EARLY_TREND":
        components["regime"] = 1.0
        score += 1.0
    elif market_regime == "TRENDING":
        components["regime"] = 1.4
        score += 1.4

    # ==================================================
    # 4️⃣ VWAP CONTEXT (ENVIRONMENT FILTER)
    # ==================================================

    if direction == "LONG" and vwap_ctx.acceptance == "BELOW":
        return DecisionResult("IGNORE", 0.0, None, {}, "below VWAP")

    if direction == "SHORT" and vwap_ctx.acceptance == "ABOVE":
        return DecisionResult("IGNORE", 0.0, None, {}, "above VWAP")

    components["vwap"] = vwap_ctx.score
    score += vwap_ctx.score

    # ==================================================
    # 5️⃣ VOLUME QUALITY
    # ==================================================

    vol_ctx = analyze_volume(volumes, close_prices=closes)

    if vol_ctx.score < 0:
        return DecisionResult("IGNORE", 0.0, None, {}, "bad volume")

    components["volume"] = vol_ctx.score
    score += vol_ctx.score

    # ==================================================
    # 6️⃣ VOLATILITY QUALITY
    # ==================================================

    atr = compute_atr(highs, lows, closes)
    move = closes[-1] - closes[-2] if len(closes) > 1 else 0.0

    volat_ctx = analyze_volatility(move, atr)

    if volat_ctx.state in ["CONTRACTING", "EXHAUSTION"]:
        return DecisionResult("IGNORE", 0.0, None, {}, "bad volatility")

    components["volatility"] = volat_ctx.score
    score += volat_ctx.score

    # ==================================================
    # 7️⃣ LIQUIDITY SAFETY
    # ==================================================

    liq_ctx = analyze_liquidity(volumes)

    if liq_ctx.score < 0:
        return DecisionResult("IGNORE", 0.0, None, {}, "illiquid instrument")

    components["liquidity"] = liq_ctx.score
    score += liq_ctx.score

    # ==================================================
    # 8️⃣ PRICE ACTION TIMING
    # ==================================================

    pa_ctx = price_action_context(
        prices=closes,
        highs=highs,
        lows=lows,
        opens=closes,
        closes=closes
    )

    components["price_action"] = pa_ctx["score"]
    score += pa_ctx["score"]

    # ==================================================
    # 9️⃣ SR LOCATION CONFIRMATION
    # ==================================================

    nearest = pullback_signal.get("nearest_level")

    sr_score = sr_location_score(closes[-1], nearest, direction)

    components["sr"] = sr_score
    score += sr_score * 1.2

    # ==================================================
    # 🔟 FINAL DECISION LOGIC
    # ==================================================

    score = round(max(min(score, 10.0), 0.0), 2)

    if score >= 7.5:
        state = f"EXECUTE_{direction}"
        reason = "high quality pullback trade"

    elif score >= 6.0:
        state = f"PREPARE_{direction}"
        reason = "developing pullback setup"

    else:
        state = "IGNORE"
        reason = "insufficient edge"

    return DecisionResult(
        state=state,
        score=score,
        direction=direction if state != "IGNORE" else None,
        components=components,
        reason=reason
    )
