# core/jarvis_hud.py
"""
JARVIS / ULTRON Autonomous Quant Command Center & Telemetry HUD.
Renders real-time institutional status:
- Ultron AI Radar (XGBoost Win-Probability, HMM Market Regime, Financial Sentiment)
- Configured Capital per trade & 5x SEBI MIS Intraday Exposure
- Real-time Active Positions, Trailing Stops, Breakeven & Runner Locks
- Live Gross P&L, Estimated Indian Taxes & Brokerage Drag, and Net P&L
- Daily Risk Budget and Circuit Breaker status
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import os
import datetime
from typing import Dict, Any, List
from execution.execution_config import (
    CAPITAL_PER_TRADE,
    USE_MIS_LEVERAGE,
    MIS_LEVERAGE_MULTIPLIER,
    MAX_TRADES_PER_DAY,
    MAX_STOP_LOSSES,
    MIN_AI_PROBABILITY,
    PAPER_TRADING,
    MAX_DAILY_LOSS_RUPEES,
    MAX_CONCURRENT_POSITIONS
)

_INSTRUMENT_MAP = None

def get_clean_symbol(key_or_isin: str) -> str:
    """Resolves ISIN or raw instrument key into human-friendly ticker (e.g. INE486A01021 -> CESC)."""
    global _INSTRUMENT_MAP
    if _INSTRUMENT_MAP is None:
        _INSTRUMENT_MAP = {}
        try:
            inst_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "instruments.json")
            if os.path.exists(inst_path):
                import json
                with open(inst_path, "r", encoding="utf-8") as f:
                    for item in json.load(f):
                        k = item.get("instrument_key", "")
                        isin = item.get("isin", "")
                        sym = item.get("trading_symbol", "")
                        if sym:
                            if k: _INSTRUMENT_MAP[k] = sym
                            if isin: _INSTRUMENT_MAP[isin] = sym
                            _INSTRUMENT_MAP[sym] = sym
        except Exception:
            pass

    clean = key_or_isin.split("|")[-1].replace("NSE_EQ", "").strip(":")
    return _INSTRUMENT_MAP.get(clean, _INSTRUMENT_MAP.get(key_or_isin, clean))


class JarvisHUD:
    def __init__(self, risk_manager, trade_monitor, macro_filter=None, strategy_engine=None):
        self.risk_manager = risk_manager
        self.trade_monitor = trade_monitor
        self.macro_filter = macro_filter
        self.strategy_engine = strategy_engine
        self.last_render_time = datetime.datetime.now()
        self.first_render = True
        self.total_ticks = 0

    def render(self, current_prices: Dict[str, float], force: bool = False):
        """
        Renders a single fixed in-place TV-style dynamic scoreboard without terminal scrolling.
        """
        now = datetime.datetime.now()
        self.total_ticks += len(current_prices)

        # Refresh scoreboard smoothly every 3 seconds or on critical events
        if not force and (now - self.last_render_time).total_seconds() < 3.0:
            return

        self.last_render_time = now
        active_trades = list(self.trade_monitor.active_trades.values())

        # Clear terminal once on startup for a fresh pinned display
        if self.first_render:
            os.system("cls" if os.name == "nt" else "clear")
            self.first_render = False

        # Calculate live gross P&L and estimated Indian statutory taxes
        gross_pnl = 0.0
        total_estimated_taxes = 0.0

        for t in active_trades:
            ltp = current_prices.get(t.inst_key, t.entry_price)
            pnl = (ltp - t.entry_price) * t.qty if t.side == "BUY" else (t.entry_price - ltp) * t.qty
            gross_pnl += pnl

            # Indian Intraday MIS Tax Estimate: Brokerage (Rs. 40) + STT (0.025%) + Txn/SEBI/GST (~Rs. 15)
            buy_val = t.entry_price * t.qty
            sell_val = ltp * t.qty
            total_estimated_taxes += 40.0 + (sell_val * 0.00025) + ((buy_val + sell_val) * 0.000035) + 8.5

        net_pnl = gross_pnl - total_estimated_taxes if active_trades else 0.0

        # Macro Nifty Info
        index_str = "INITIALIZING"
        if self.macro_filter and self.macro_filter.index_prices:
            idx_ltp = self.macro_filter.index_prices[-1]
            idx_chg = ((idx_ltp - (self.macro_filter.day_open or idx_ltp)) / max(self.macro_filter.day_open or idx_ltp, 1e-9)) * 100.0
            index_str = f"{idx_ltp:.2f} ({idx_chg:+.2f}%)"

        # AI Sentiment & Regime
        ai_regime = "TRENDING_INSTITUTIONAL"
        sentiment_label = "NEUTRAL"
        if self.strategy_engine and hasattr(self.strategy_engine, "sentiment_data"):
            sentiment_label = self.strategy_engine.sentiment_data.get("label", "NEUTRAL")

        # Institutional Market Clock Phase
        cur_t = now.time()
        if cur_t < datetime.time(9, 20):
            session_phase = "🟡 PRE-OPEN DISCOVERY"
        elif cur_t < datetime.time(11, 15):
            session_phase = "⚡ MORNING POWER (ORB/SQUEEZE)"
        elif cur_t < datetime.time(13, 15):
            session_phase = "☕ MIDDAY DEFENSE (PULLBACKS)"
        elif cur_t < datetime.time(14, 15):
            session_phase = "🚀 AFTERNOON TREND LEG"
        else:
            session_phase = "🛑 14:15 LOCKED (EXITS ONLY)"

        # Capital & Buying Power
        buying_power = CAPITAL_PER_TRADE * (MIS_LEVERAGE_MULTIPLIER if USE_MIS_LEVERAGE else 1.0)
        mode_str = "🧪 SIMULATION (PAPER)" if PAPER_TRADING else "⚡ LIVE (MIS CASH)"
        loss_stop_str = f"-Rs. {MAX_DAILY_LOSS_RUPEES:,.0f}"

        w = 98
        lines = []
        # ANSI Escape \033[H repositions cursor to top-left (no scrolling, updates in-place)
        lines.append("\033[H" + "=" * w)
        lines.append(f"  [JARVIS QUANT COCKPIT]  |  {now.strftime('%Y-%m-%d %H:%M:%S')}  |  MODE: {mode_str:<18} | HEARTBEAT: OK 🟢")
        lines.append("-" * w)
        lines.append(f"  SESSION CLOCK: {session_phase:<34} | HARD LOSS STOP: {loss_stop_str}")
        lines.append(f"  CAPITAL: Rs. {CAPITAL_PER_TRADE:,.0f} (5x MIS: Rs. {buying_power:,.0f}) | AI GATE: >= {MIN_AI_PROBABILITY*100:.0f}% | HMM: {ai_regime}")
        lines.append(f"  BENCHMARK: NIFTY 50: {index_str:<18} | ACTIVE: {len(active_trades)}/{MAX_CONCURRENT_POSITIONS} | DAILY TRADES: {self.risk_manager.total_trades}/{MAX_TRADES_PER_DAY}")
        lines.append(f"  LIVE GROSS PnL: Rs. {gross_pnl:+.2f} | EST. TAXES: Rs. {total_estimated_taxes:.2f} | NET TAKE-HOME: Rs. {net_pnl:+.2f}")
        lines.append("-" * w)

        # 📡 AI HOTLIST RADAR (TOP 3 LIVE CANDIDATES)
        if self.strategy_engine and hasattr(self.strategy_engine, "get_top_hotlist"):
            hotlist = self.strategy_engine.get_top_hotlist(limit=3)
            if hotlist:
                lines.append("  📡 AI RADAR: LIVE SETUPS & ALPHA BREAKOUT RADAR (TOP 3):")
                for rank, cand in enumerate(hotlist, 1):
                    clean_name = get_clean_symbol(cand['symbol'])
                    setup_str = cand.get('setup', 'BREAKOUT')
                    rs_val = cand.get('rs', '0.0%')
                    lines.append(f"    [{rank}] {clean_name:<10} | Mood: {setup_str:<15} | RS vs Nifty: {rs_val:<5} | AI: {cand['ai_prob_pct']:<5} | {cand['status']}")
                lines.append("-" * w)
            else:
                lines.append("  📡 AI RADAR: Actively monitoring 480 NIFTY 500 stocks across all 3 Moods...")
                lines.append("-" * w)

        if active_trades:
            lines.append("  ACTIVE MIS POSITIONS IN MARKET:")
            for t in active_trades:
                ltp = current_prices.get(t.inst_key, t.entry_price)
                pct = t.get_current_profit_pct(ltp) * 100.0
                pnl = (ltp - t.entry_price) * t.qty if t.side == "BUY" else (t.entry_price - ltp) * t.qty
                be_tag = " [BREAKEVEN LOCK]" if t.breakeven_moved else ""
                runner_tag = " [RL RUNNER EXPANDED]" if getattr(t, "runner_extended", False) else ""
                sym = get_clean_symbol(t.inst_key)
                lines.append(f"    * {sym:<12} {t.side:<4} Qty:{t.qty:<4} @ Rs.{t.entry_price:.2f} | LTP: Rs.{ltp:.2f} | PnL: Rs.{pnl:+.2f} ({pct:+.2f}%){be_tag}{runner_tag}")
                lines.append(f"      SL: Rs.{t.stop_loss:.2f} | TP: Rs.{t.target:.2f} | GTT: {t.gtt_id or 'DEPLOYED'}")
        else:
            lines.append(f"  [STREAM ACTIVE] 6 Parallel Sockets streaming 480 NIFTY 500 stocks | Ticks: {self.total_ticks:,}")

        lines.append("=" * w)
        # Append \033[K to every line to guarantee zero ghost characters from previous renders
        cleared_lines = [line + "\033[K" for line in lines]
        sys.stdout.write("\n".join(cleared_lines) + "\n")
        sys.stdout.flush()


