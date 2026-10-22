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
    breakout_signal: Optional[Dict],
    opens: Optional[list[float]] = None,
) -> DecisionResult:

    components: Dict[str, float] = {}
    score = 0.0

    # ==================================================
    # 1️⃣ STRUCTURE GATE (MOST IMPORTANT)
    # ==================================================

    if not breakout_signal:
        return DecisionResult("IGNORE", 0.0, None, {}, "no valid setup")

    direction = breakout_signal["direction"]
    signal_type = breakout_signal.get("signal", "EXECUTE")
    setup_name = breakout_signal.get("setup_name", "BREAKOUT")
    components["setup_type"] = setup_name

    # Potential setups only PREPARE
    if signal_type == "POTENTIAL":
        components["structure"] = 1.5
        return DecisionResult(
            state=f"PREPARE_{direction}",
            score=1.5,
            direction=direction,
            components=components,
            reason=f"potential {setup_name}"
        )

    # High-conviction setups (Pullbacks & Squeezes) get high structural score
    base_score = 3.2 if setup_name in ["PULLBACK_RETEST", "SQUEEZE_BREAKOUT"] else 3.0
    components["breakout"] = base_score
    score += base_score

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

    if market_regime == "WEAK":
        return DecisionResult("IGNORE", 0.0, None, {}, "weak choppy market regime")
    elif market_regime == "COMPRESSION":
        # Pre-breakout compression is valid setup foundation
        components["regime"] = 0.8
        score += 0.8
    elif market_regime == "EARLY_TREND":
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

    # Invert sign for SHORT so bearish VWAP adds to the execution score
    if direction == "SHORT":
        vwap_contribution = abs(vwap_ctx.score) if vwap_ctx.acceptance == "BELOW" else -abs(vwap_ctx.score)
    else:
        vwap_contribution = vwap_ctx.score if vwap_ctx.acceptance == "ABOVE" else -abs(vwap_ctx.score)

    components["vwap"] = round(vwap_contribution, 2)
    score += vwap_contribution

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

    # -------------------- PRICE ACTION --------------------

    actual_opens = opens if opens and len(opens) == len(closes) else closes
    pa_ctx = price_action_context(
        prices=closes,
        highs=highs,
        lows=lows,
        opens=actual_opens,
        closes=closes,
        ema_short=exponential_moving_average(prices, 9),
        ema_long=exponential_moving_average(prices, 21),
    )

    # small alignment gate
    if direction == "LONG" and pa_ctx["score"] < 0:
        return DecisionResult("IGNORE", 0.0, None, {}, "price action not supportive")

    if direction == "SHORT" and pa_ctx["score"] > 0:
        return DecisionResult("IGNORE", 0.0, None, {}, "price action not supportive")

    # For SHORT, bearish price action (negative pa_ctx score) adds positive conviction
    if direction == "SHORT":
        pa_contribution = abs(pa_ctx["score"]) if pa_ctx["score"] <= 0 else -pa_ctx["score"]
    else:
        pa_contribution = pa_ctx["score"]

    components["price_action"] = round(pa_contribution, 2)
    score += pa_contribution

    # ==================================================
    # 9️⃣ SR LOCATION CONFIRMATION
    # ==================================================

    nearest = pullback_signal.get("nearest_level")

    sr_score = sr_location_score(closes[-1], nearest, direction)

    components["sr"] = sr_score
    score += sr_score * 1.2

    # -------------------- RSI (STRICT MOMENTUM & CLIMAX EXHAUSTION GUARDS) --------------------

    rsi = relative_strength_index(prices, 14)

    if rsi is not None:
        components["rsi"] = round(rsi, 1)
        if direction == "LONG":
            if rsi > 68:
                return DecisionResult("IGNORE", 0.0, None, {}, f"overbought climax exhaustion long (RSI={rsi:.1f} > 68)")
            if rsi < 50:
                score -= 1.0
            if rsi < 40:
                return DecisionResult("IGNORE", 0.0, None, {}, "weak momentum long")

        if direction == "SHORT":
            if rsi < 32:
                return DecisionResult("IGNORE", 0.0, None, {}, f"oversold climax exhaustion short (RSI={rsi:.1f} < 32)")
            if rsi > 50:
                score -= 1.0
            if rsi > 60:
                return DecisionResult("IGNORE", 0.0, None, {}, "weak momentum short")

    # -------------------- SECTOR SYMPATHY GATE --------------------
    try:
        from strategy.sector_manager import get_sector_manager
        sec_mgr = get_sector_manager()
        is_aligned, sec_reason = sec_mgr.check_sector_sympathy(inst_key, direction)
        if not is_aligned:
            return DecisionResult("IGNORE", 0.0, None, {}, sec_reason)
    except Exception:
        pass

    # -------------------- FINAL DECISION --------------------

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
