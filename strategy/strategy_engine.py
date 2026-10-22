# strategy/strategy_engine.py
"""
Vision-Level Multi-Asset Quantitative Strategy Engine.
Enforces complete institutional hierarchy:
1. Data Sufficiency
2. MTF Candle Aggregation & Timeframe Alignment
3. Market Regime Filter (Decoupled Compression Breakouts)
4. VWAP Acceptance & Dynamic Slope
5. Higher-Timeframe Structural Trend (HTF Bias)
6. Breakout & Volume Intent
7. Level-2 Order Book Imbalance & Spread Verification (Depth)
8. Macro Benchmark Index Beta Alignment (Nifty 50)
9. Composite Final Trade Decision
"""

from typing import Optional, Dict, Any

from strategy.market_regime import detect_market_regime
from strategy.htf_bias import get_htf_bias
from strategy.breakout_detector import breakout_signal
from strategy.decision_engine import final_trade_decision
from strategy.market_depth_analyzer import analyze_market_depth
from strategy.vwap_filter import VWAPCalculator
from strategy.mtf_builder import MTFBuilder
from strategy.mtf_context import analyze_mtf

# Ultron AI Engines
from strategy.ai_probability_engine import AIProbabilityEngine
from strategy.ai_hmm_regime import AIHMMRegimeRadar
from strategy.ai_sentiment_engine import AISentimentEngine
from strategy.ai_temporal_engine import AITemporalSequenceEngine
from strategy.ai_rl_policy import AIRLExitPolicy
from execution.execution_config import MIN_AI_PROBABILITY


class StrategyEngine:
    def __init__(self, scanner, vwap_calculators, macro_filter=None):
        self.scanner = scanner
        self.vwap_calculators = vwap_calculators
        self.mtf_builder = MTFBuilder()
        self.macro_filter = macro_filter

        # Ultron AI Suite
        self.ai_prob_engine = AIProbabilityEngine(min_probability=MIN_AI_PROBABILITY)
        self.hmm_radar = AIHMMRegimeRadar(lookback_window=40)
        self.sentiment_engine = AISentimentEngine()
        self.temporal_engine = AITemporalSequenceEngine(sequence_length=30)
        self.rl_exit_policy = AIRLExitPolicy()

        # Pre-market initial sentiment
        self.sentiment_data = self.sentiment_engine.fetch_market_sentiment()

        # Real-time AI Hotlist Radar
        self.candidate_hotlist: Dict[str, Dict[str, Any]] = {}

    def get_top_hotlist(self, limit: int = 3):
        """Returns top candidates sorted by AI probability and setup readiness."""
        if not self.candidate_hotlist:
            return []
        sorted_candidates = sorted(
            self.candidate_hotlist.values(),
            key=lambda x: x.get("ai_prob", 0.0),
            reverse=True
        )
        return sorted_candidates[:limit]

    def evaluate(self, inst_key: str, ltp: float, market_ff: Optional[Dict[str, Any]] = None):
        # ==================================================
        # 1️⃣ DATA SUFFICIENCY
        # ==================================================

        if not self.scanner.has_enough_data(inst_key, min_bars=30):
            return None

        prices = self.scanner.get_prices(inst_key)
        opens = self.scanner.get_opens(inst_key)
        highs = self.scanner.get_highs(inst_key)
        lows = self.scanner.get_lows(inst_key)
        closes = self.scanner.get_closes(inst_key)
        volumes = self.scanner.get_volumes(inst_key)

        if not (prices and highs and lows and closes and volumes):
            return None

        # ==================================================
        # 2️⃣ BUILD MTF CANDLES (BAR-CLOSE DEDUPLICATED)
        # ==================================================

        # Ensure MTF builder buffer is synchronized with scanner preloaded history
        current_mtf_len = len(self.mtf_builder.buffers.get(inst_key, []))
        if current_mtf_len < 15:
            all_scanner_bars = self.scanner.get_last_n_bars(inst_key, 60)
            for b in all_scanner_bars:
                self.mtf_builder.update(
                    inst_key,
                    b["time"],
                    b["open"],
                    b["high"],
                    b["low"],
                    b["close"],
                    b["volume"]
                )

        last_bar = self.scanner.get_last_n_bars(inst_key, 1)
        if not last_bar:
            return None

        bar = last_bar[0]
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

        # 🔒 HARD MTF GATE
        if mtf_ctx.direction == "NEUTRAL":
            return None

        if mtf_ctx.conflict:
            return None

        # ==================================================
        # 3️⃣ MARKET REGIME FILTER
        # ==================================================

        regime = detect_market_regime(highs=highs, lows=lows, closes=closes)

        # Reject purely weak/choppy noise, but allow compression breakouts
        if regime.state == "WEAK":
            return None

        # ==================================================
        # 4️⃣ VWAP CONTEXT
        # ==================================================

        if inst_key not in self.vwap_calculators:
            self.vwap_calculators[inst_key] = VWAPCalculator()

        vwap_calc = self.vwap_calculators[inst_key]
        vwap_calc.update(ltp, volumes[-1] if volumes else 0)
        vwap_ctx = vwap_calc.get_context(ltp)

        # ==================================================
        # 5️⃣ HTF BIAS (EMA STRUCTURE)
        # ==================================================

        htf_bias = get_htf_bias(prices=prices, vwap_value=vwap_ctx.vwap)

        # HTF must not oppose MTF
        if mtf_ctx.direction == "BULLISH" and htf_bias.direction == "BEARISH":
            return None

        if mtf_ctx.direction == "BEARISH" and htf_bias.direction == "BULLISH":
            return None

        # ==================================================
        # 6️⃣ DYNAMIC RSI & MULTI-BEHAVIOR SETUP DETECTION
        # ==================================================
        from strategy.indicators import relative_strength_index
        from strategy.setup_detector import detect_price_action_setup, check_climax_exhaustion

        rsi_val = relative_strength_index(closes, 14) or 50.0
        delta = self.scanner.get_order_flow_delta(inst_key) if hasattr(self.scanner, "get_order_flow_delta") else None

        setup_signal = detect_price_action_setup(
            inst_key=inst_key,
            prices=prices,
            opens=opens,
            highs=highs,
            lows=lows,
            closes=closes,
            volumes=volumes,
            vwap=vwap_ctx.vwap or ltp,
            mtf_direction=mtf_ctx.direction,
            order_flow_delta=delta
        )

        # Fallback to standard breakout ONLY IF NOT exhausted
        if not setup_signal:
            raw_breakout = breakout_signal(
                inst_key=inst_key,
                prices=prices,
                volume_history=volumes,
                high_prices=highs,
                low_prices=lows,
                close_prices=closes
            )
            if raw_breakout:
                exhaustion = check_climax_exhaustion(
                    ltp=ltp,
                    vwap=vwap_ctx.vwap or ltp,
                    rsi=rsi_val,
                    closes=closes,
                    highs=highs,
                    lows=lows,
                    direction=raw_breakout["direction"]
                )
                if not exhaustion["is_exhausted"]:
                    setup_signal = {
                        "setup_name": "BREAKOUT",
                        "direction": raw_breakout["direction"],
                        "signal": raw_breakout["signal"],
                        "confidence": 0.75,
                        "atr": raw_breakout.get("atr"),
                        "rsi": rsi_val,
                        "components": raw_breakout.get("components", {}),
                        "reason": f"Standard breakout ({raw_breakout['direction']})"
                    }

        if not setup_signal:
            return None

        direction = setup_signal["direction"]
        setup_name = setup_signal.get("setup_name", "BREAKOUT")

        # Direction must align with MTF
        if direction == "LONG" and mtf_ctx.direction != "BULLISH":
            return None

        if direction == "SHORT" and mtf_ctx.direction != "BEARISH":
            return None

        # ==================================================
        # 7️⃣ LEVEL-2 ORDER BOOK DEPTH & SPREAD GATE
        # ==================================================

        depth_ctx = None
        if market_ff:
            depth_ctx = analyze_market_depth(market_ff, direction=direction)
            # If spread is too wide (> 0.15%), abort to avoid slippage trap
            if not depth_ctx.is_liquid:
                return None

        # ==================================================
        # 8️⃣ MACRO BENCHMARK (NIFTY 50) & RELATIVE STRENGTH (RS/RW) ALIGNMENT
        # ==================================================

        macro_ctx = None
        rs_data = None
        stock_open = opens[0] if opens else closes[0]
        if self.macro_filter:
            macro_ctx = self.macro_filter.evaluate_macro_alignment(direction)
            rs_data = self.macro_filter.calculate_relative_strength(ltp, stock_open)

            # Institutional Alpha Filter: Reject trades that conflict with Relative Strength
            # If a stock is heavily underperforming (RS <= -0.35%), NEVER BUY IT!
            # If a stock is heavily outperforming (RS >= +0.35%), NEVER SHORT IT!
            if direction not in rs_data.get("allowed_directions", [direction]):
                return None

        # ==================================================
        # 9️⃣ FINAL DECISION ENGINE
        # ==================================================

        decision = final_trade_decision(
            inst_key=inst_key,
            prices=prices,
            highs=highs,
            lows=lows,
            closes=closes,
            volumes=volumes,
            market_regime=regime.state,
            htf_bias_label=htf_bias.label,
            vwap_ctx=vwap_ctx,
            breakout_signal=setup_signal,
            opens=opens
        )

        # Apply Level-2 Depth Modifier
        if depth_ctx:
            decision.score = round(max(0.0, min(10.0, decision.score + depth_ctx.score)), 2)
            decision.components["depth"] = depth_ctx.score

        # Apply Macro Benchmark Modifier
        if macro_ctx:
            decision.score = round(max(0.0, min(10.0, decision.score + macro_ctx.score_modifier)), 2)
            decision.components["macro"] = macro_ctx.score_modifier

        # ==================================================
        # 🔟 ULTRON AI SUITE (HMM, TEMPORAL, SENTIMENT, XGBOOST)
        # ==================================================

        # AI Gate A: Gaussian HMM Market Regime Radar (Blocks choppy traps)
        hmm_result = self.hmm_radar.detect_regime(closes)
        if not hmm_result["trade_permitted"]:
            return None

        # AI Gate B: Temporal Sequence Acceleration & Climax Exhaustion
        temporal_result = self.temporal_engine.analyze_sequence(
            closes=closes,
            highs=highs,
            lows=lows,
            volumes=volumes,
            direction=direction
        )
        if temporal_result["exhaustion_risk"]:
            return None

        # AI Gate C: Tabular Gradient-Boosted Win-Probability Classifier
        open_val = opens[-1] if opens else closes[-1]
        atr_val = setup_signal.get("atr") or (highs[-1] - lows[-1])
        vol_surge_val = setup_signal["components"].get("volume", 1.0)
        depth_imbalance_val = depth_ctx.imbalance_ratio if depth_ctx else 1.0
        spread_val = depth_ctx.spread_pct if depth_ctx else 0.05
        macro_mod = macro_ctx.score_modifier if macro_ctx else 0.0

        features = self.ai_prob_engine.extract_features(
            ltp=ltp,
            vwap=vwap_ctx.vwap or ltp,
            rsi=rsi_val,
            volume_surge=vol_surge_val,
            depth_imbalance=depth_imbalance_val,
            spread_pct=spread_val,
            mtf_strength=mtf_ctx.strength,
            macro_modifier=macro_mod,
            atr=atr_val,
            high=highs[-1],
            low=lows[-1],
            open_p=open_val,
            direction=direction
        )

        ai_prob = self.ai_prob_engine.predict_win_probability(features)

        # 📡 Record Live Candidate in AI Hotlist Radar
        p_val = ai_prob["win_probability"]
        clean_sym = inst_key.split("|")[-1].replace("NSE_EQ", "").strip(":")
        status_text = f"🟢 READY ({setup_name})" if ai_prob["is_approved"] else f"🟡 {setup_name} ({p_val*100:.0f}%)"
        rs_str = f"{rs_data.get('rs_score', 0.0):+.1f}%" if rs_data else "0.0%"
        self.candidate_hotlist[clean_sym] = {
            "symbol": clean_sym,
            "ltp": ltp,
            "ai_prob": p_val,
            "ai_prob_pct": f"{p_val*100:.1f}%",
            "vol_surge": f"{vol_surge_val:.1f}x",
            "direction": direction,
            "setup": setup_name,
            "rs": rs_str,
            "status": status_text,
            "regime": hmm_result["regime"]
        }

        # 🔒 HARD AI GATE: Discard any setup below 70% win-probability
        if not ai_prob["is_approved"]:
            return None

        decision.atr = atr_val
        if rs_data:
            decision.components["rs_score"] = rs_str
        decision.components["ai_win_prob"] = f"{ai_prob['win_probability']*100:.1f}%"
        decision.components["ai_confidence"] = ai_prob["confidence"]
        decision.components["hmm_regime"] = hmm_result["regime"]
        decision.components["temporal_accel"] = temporal_result["acceleration"]
        decision.score = round(min(10.0, decision.score + (ai_prob["win_probability"] - 0.5) * 2.0), 2)

        # Re-evaluate final state after microstructural modifiers
        if decision.score >= 6.0 and decision.state.startswith("PREPARE"):
            decision.state = f"EXECUTE_{direction}"
        elif decision.score < 6.0 and decision.state.startswith("EXECUTE"):
            decision.state = f"PREPARE_{direction}"

        # Attach debug context
        decision.components["mtf_direction"] = mtf_ctx.direction
        decision.components["mtf_strength"] = mtf_ctx.strength
        decision.components["regime"] = regime.state
        decision.components["htf_bias"] = htf_bias.label

        return decision

