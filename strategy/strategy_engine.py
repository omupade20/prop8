# strategy/strategy_engine.py

from strategy.market_regime import detect_market_regime
from strategy.htf_bias import get_htf_bias
from strategy.pullback_detector import detect_pullback_signal
from strategy.decision_engine import final_trade_decision

from strategy.vwap_filter import VWAPCalculator
from strategy.mtf_builder import MTFBuilder
from strategy.mtf_context import analyze_mtf


class StrategyEngine:
    """
    PROFESSIONAL PULLBACK-BASED STRATEGY ENGINE

    Strict Hierarchy:

    MTF → Regime → HTF → VWAP → PULLBACK DETECTION → Decision Engine
    """

    def __init__(self, scanner, vwap_calculators):
        self.scanner = scanner
        self.vwap_calculators = vwap_calculators
        self.mtf_builder = MTFBuilder()

    def evaluate(self, inst_key: str, ltp: float):

        # ==================================================
        # 1️⃣ DATA SUFFICIENCY
        # ==================================================

        if not self.scanner.has_enough_data(inst_key, min_bars=40):
            return None

        prices = self.scanner.get_prices(inst_key)
        highs = self.scanner.get_highs(inst_key)
        lows = self.scanner.get_lows(inst_key)
        closes = self.scanner.get_closes(inst_key)
        volumes = self.scanner.get_volumes(inst_key)

        if not (prices and highs and lows and closes and volumes):
            return None

        # ==================================================
        # 2️⃣ MULTI TIMEFRAME CONTEXT
        # ==================================================

        last_bar = self.scanner.get_last_n_bars(inst_key, 1)
        if not last_bar:
            return None

        bar = last_bar[0]

        # Build MTF candles from 1m bars
        self.mtf_builder.update(
            inst_key,
            bar["time"],
            bar["open"],
            bar["high"],
            bar["low"],
            bar["close"],
            bar["volume"]
        )

        candle_5m = self.mtf_builder.get_latest_5m(inst_key)
        hist_5m = self.mtf_builder.get_tf_history(inst_key, minutes=5, lookback=3)

        candle_15m = self.mtf_builder.get_latest_15m(inst_key)
        hist_15m = self.mtf_builder.get_tf_history(inst_key, minutes=15, lookback=3)

        mtf_ctx = analyze_mtf(
            candle_5m,
            candle_15m,
            history_5m=hist_5m,
            history_15m=hist_15m
        )

        # HARD GATES ON MTF
        if mtf_ctx.direction == "NEUTRAL":
            return None

        if mtf_ctx.conflict:
            return None

        # Require at least MEDIUM confidence on higher timeframe
        if mtf_ctx.confidence == "LOW":
            return None

        # ==================================================
        # 3️⃣ MARKET REGIME FILTER (STRICT)
        # ==================================================

        regime = detect_market_regime(
            highs=highs,
            lows=lows,
            closes=closes
        )

        # TRADE ONLY IN PROPER TRENDING MARKETS
        if regime.state != "TRENDING":
            return None

        # ==================================================
        # 4️⃣ VWAP CONTEXT
        # ==================================================

        if inst_key not in self.vwap_calculators:
            self.vwap_calculators[inst_key] = VWAPCalculator()

        vwap_calc = self.vwap_calculators[inst_key]

        # Update VWAP using completed bar data (cleaner than tick LTP)
        vwap_calc.update(
            bar["close"],
            bar["volume"]
        )

        vwap_ctx = vwap_calc.get_context(ltp)

        # ==================================================
        # 5️⃣ HTF BIAS (EMA BASED)
        # ==================================================

        htf_bias = get_htf_bias(
            prices=prices,
            vwap_value=vwap_ctx.vwap
        )

        # HTF must align with MTF direction
        if mtf_ctx.direction == "BULLISH" and htf_bias.direction != "BULLISH":
            return None

        if mtf_ctx.direction == "BEARISH" and htf_bias.direction != "BEARISH":
            return None

        # ==================================================
        # 6️⃣ PULLBACK DETECTION
        # ==================================================

        # Use HTF bias as master directional input
        pullback = detect_pullback_signal(
            prices=prices,
            highs=highs,
            lows=lows,
            closes=closes,
            volumes=volumes,
            htf_direction=htf_bias.direction
        )

        if not pullback:
            return None

        # ==================================================
        # 7️⃣ FINAL DECISION ENGINE
        # ==================================================

        decision = final_trade_decision(
            inst_key=inst_key,
            prices=prices,
            highs=highs,
            lows=lows,
            closes=closes,
            volumes=volumes,
            market_regime=regime.state,
            htf_bias_direction=htf_bias.direction,
            vwap_ctx=vwap_ctx,
            pullback_signal=pullback
        )

        # Attach debugging context for logging / dashboard
        decision.components["mtf_direction"] = mtf_ctx.direction
        decision.components["mtf_strength"] = mtf_ctx.strength
        decision.components["regime"] = regime.state
        decision.components["htf_bias"] = htf_bias.label

        return decision
