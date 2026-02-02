# strategy/decision_engine.py

from dataclasses import dataclass
from typing import Optional, Dict

from strategy.indicators import exponential_moving_average, relative_strength_index
from strategy.volume_filter import analyze_volume
from strategy.volatility_filter import analyze_volatility, compute_atr
from strategy.liquidity_filter import analyze_liquidity
from strategy.price_action import price_action_context
from strategy.sr_levels import compute_sr_levels, get_nearest_sr, sr_location_score
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
# Decision Engine
# =========================

def final_trade_decision(
    inst_key: str,
    prices: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    volumes: list[float],
    market_regime: str,
    htf_bias_label: str,
    vwap_ctx: VWAPContext,
    breakout_signal: Optional[Dict],
) -> DecisionResult:

    components: Dict[str, float] = {}
    score = 0.0

    # ==================================================
    # 1️⃣ STRUCTURE GATE — NON-NEGOTIABLE
    # ==================================================

    if not breakout_signal:
        return DecisionResult("IGNORE", 0.0, None, {}, "no breakout")

    direction = breakout_signal["direction"]
    signal_type = breakout_signal["signal"]

    # POTENTIAL → NEVER EXECUTE
    if signal_type == "POTENTIAL":
        components["breakout"] = 1.0
        return DecisionResult(
            state=f"PREPARE_{direction}",
            score=1.0,
            direction=direction,
            components=components,
            reason="potential breakout"
        )

    # CONFIRMED breakout
    components["breakout"] = 3.0
    score += 3.0

    # ==================================================
    # 2️⃣ HTF & REGIME AUTHORITY
    # ==================================================

    # HTF must NOT oppose
    if direction == "LONG" and htf_bias_label.startswith("BEARISH"):
        return DecisionResult("IGNORE", 0.0, None, {}, "htf opposes long")

    if direction == "SHORT" and htf_bias_label.startswith("BULLISH"):
        return DecisionResult("IGNORE", 0.0, None, {}, "htf opposes short")

    components["htf"] = 1.2
    score += 1.2

    # Regime filter
    if market_regime in ("WEAK", "COMPRESSION"):
        return DecisionResult("IGNORE", 0.0, None, {}, "bad market regime")

    if market_regime == "EARLY_TREND":
        components["regime"] = 0.8
        score += 0.8
    elif market_regime == "TRENDING":
        components["regime"] = 1.2
        score += 1.2

    # ==================================================
    # 3️⃣ VWAP ACCEPTANCE (ENVIRONMENT)
    # ==================================================

    if direction == "LONG" and vwap_ctx.acceptance == "BELOW":
        return DecisionResult("IGNORE", 0.0, None, {}, "below VWAP")

    if direction == "SHORT" and vwap_ctx.acceptance == "ABOVE":
        return DecisionResult("IGNORE", 0.0, None, {}, "above VWAP")

    components["vwap"] = vwap_ctx.score
    score += vwap_ctx.score

    # ==================================================
    # 4️⃣ PARTICIPATION (VOLUME + VOLATILITY + LIQUIDITY)
    # ==================================================

    vol_ctx = analyze_volume(volumes, close_prices=closes)
    components["volume"] = vol_ctx.score
    score += vol_ctx.score

    atr = compute_atr(highs, lows, closes)
    move = closes[-1] - closes[-2] if len(closes) > 1 else 0.0
    volat_ctx = analyze_volatility(move, atr)
    components["volatility"] = volat_ctx.score
    score += volat_ctx.score

    liq_ctx = analyze_liquidity(volumes)
    if liq_ctx.score < 0:
        return DecisionResult("IGNORE", 0.0, None, {}, "illiquid")

    components["liquidity"] = liq_ctx.score
    score += liq_ctx.score

    # ==================================================
    # 5️⃣ TIMING (PA + SR + MOMENTUM)
    # ==================================================

    pa_ctx = price_action_context(
        prices=closes,
        highs=highs,
        lows=lows,
        opens=closes,
        closes=closes,
        ema_short=exponential_moving_average(prices, 9),
        ema_long=exponential_moving_average(prices, 21),
    )
    components["price_action"] = pa_ctx["score"]
    score += pa_ctx["score"]

    sr_levels = compute_sr_levels(highs, lows)
    nearest = get_nearest_sr(closes[-1], sr_levels)
    sr_score = sr_location_score(closes[-1], nearest, direction)
    components["sr"] = sr_score
    score += sr_score * 0.7

    # Momentum sanity
    rsi = relative_strength_index(prices, 14)
    if rsi:
        if direction == "LONG" and rsi < 45:
            score -= 0.5
        elif direction == "SHORT" and rsi > 55:
            score -= 0.5

    # ==================================================
    # 6️⃣ FINAL DECISION
    # ==================================================

    score = round(max(min(score, 10.0), 0.0), 2)

    if score >= 6.0:
        state = f"EXECUTE_{direction}"
        reason = "high quality breakout"
    elif score >= 3.5:
        state = f"PREPARE_{direction}"
        reason = "setup forming"
    else:
        state = "IGNORE"
        reason = "weak follow-through"

    return DecisionResult(
        state=state,
        score=score,
        direction=direction if state != "IGNORE" else None,
        components=components,
        reason=reason
    )
