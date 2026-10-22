# core/market_streamer.py
"""
Upstox Market Data Streamer & Master JARVIS Controller.
Coordinates:
- Market Status & Timing Check (MarketHolidaysAndTimingsApi)
- Historical Pre-loading Warmup (HistoryV3Api)
- Startup Position Reconciliation (PortfolioApi)
- Real-time Order & Portfolio WebSocket (PortfolioDataStreamer)
- Real-time Market Data Protobuf Feed (MarketDataStreamerV3)
- Level-2 Order Book Depth & Microstructure Analysis
- Macro Benchmark Beta Alignment (Nifty 50)
- Automated 15:15 IST EOD MIS Square-Off
- Real-Time JARVIS Quant Command Center Telemetry HUD
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import json
import os
import datetime
import time
from typing import Dict, Set

import upstox_client
from config.settings import ACCESS_TOKEN

from core.upstox_service import UpstoxService
from core.portfolio_streamer import PortfolioStreamer
from core.eod_square_off import EODSquareOffManager
from core.jarvis_hud import JarvisHUD

from strategy.scanner import MarketScanner
from strategy.vwap_filter import VWAPCalculator
from strategy.strategy_engine import StrategyEngine
from strategy.macro_index_filter import MacroIndexFilter

from execution.execution_engine import ExecutionEngine
from execution.order_executor import OrderExecutor
from execution.trade_monitor import TradeMonitor
from execution.risk_manager import RiskManager
from execution.trade_logger import TradeLogger
from execution.execution_config import MAX_WATCHLIST_SIZE, USE_PORTFOLIO_STREAMER


# ---------------- INITIALIZE UPSTOX SERVICE ----------------
upstox_service = UpstoxService()

# ---------------- LOAD & FILTER UNIVERSE ----------------
UNIVERSE_PATH = "data/nifty500_keys.json"
if os.path.exists(UNIVERSE_PATH):
    with open(UNIVERSE_PATH, "r") as f:
        ALL_KEYS = json.load(f)
else:
    ALL_KEYS = ["NSE_EQ|INE002A01018", "NSE_EQ|INE009A01021", "NSE_EQ|INE040A01034"]

FEED_MODE = "full"
# Include Nifty 50 Index for macro beta tracking + top stocks up to connection limit
INDEX_KEY = "NSE_INDEX|Nifty 50"
INSTRUMENT_LIST = [INDEX_KEY] + [k for k in ALL_KEYS if k != INDEX_KEY][:MAX_WATCHLIST_SIZE - 1]
print(f"[UNIVERSE] Loaded {len(INSTRUMENT_LIST)} symbols for Full Mode streaming (Broker limit: {MAX_WATCHLIST_SIZE}).")

# ---------------- CORE QUANT & JARVIS OBJECTS ----------------
macro_filter = MacroIndexFilter(INDEX_KEY)
scanner = MarketScanner(max_len=600)
vwap_calculators = {inst: VWAPCalculator() for inst in INSTRUMENT_LIST}
strategy_engine = StrategyEngine(scanner, vwap_calculators, macro_filter=macro_filter)

order_executor = OrderExecutor(upstox_service)
trade_monitor = TradeMonitor()
risk_manager = RiskManager()
trade_logger = TradeLogger()

execution_engine = ExecutionEngine(
    order_executor,
    trade_monitor,
    risk_manager,
    trade_logger
)

# Automated EOD Square-Off & JARVIS Telemetry
eod_manager = EODSquareOffManager(execution_engine, trade_monitor)
jarvis_hud = JarvisHUD(risk_manager, trade_monitor, macro_filter=macro_filter, strategy_engine=strategy_engine)


signals_today: Dict[str, Set[str]] = {}
ALLOW_NEW_TRADES = True


# ---------------- PRELOAD / WARMUP ----------------
def warmup_system():
    """
    Preloads 60 minutes of historical 1-minute candles from Upstox History API.
    Eliminates the 30-55 minute startup wait so the bot can trade immediately.
    """
    print("[WARMUP] Pre-loading historical candles from Upstox History API...")
    loaded_count = 0
    warmup_subset = [k for k in INSTRUMENT_LIST if not k.startswith("NSE_INDEX")][:30]
    for inst in warmup_subset:
        try:
            candles = upstox_service.get_intraday_candles(inst, unit="minute", interval=1)
            if candles:
                scanner.warmup_candles(inst, candles)
                loaded_count += 1
        except Exception:
            continue
    print(f"[WARMUP] Completed: {loaded_count}/{len(warmup_subset)} symbols preloaded with live historical bars.")


# ---------------- RECONCILE POSITIONS ----------------
def reconcile_positions():
    """
    Queries Upstox PortfolioApi to discover and monitor any open broker positions.
    """
    print("[RECON] Checking existing open positions from Upstox...")
    live_positions = upstox_service.get_live_positions()
    if live_positions:
        trade_monitor.reconcile_broker_positions(live_positions)
        print(f"[RECON] Reconciled {len(live_positions)} active broker positions.")
    else:
        print("[RECON] No existing open intraday positions found.")


# ---------------- MASTER STREAMER ----------------
def start_market_streamer():
    global ALLOW_NEW_TRADES

    # 1. Market Session Check
    market_info = upstox_service.get_market_status("NSE")
    m_status = market_info.get("status", "UNKNOWN")
    print(f"[MARKET STATUS] NSE Session: {m_status}")

    # 2. Reconcile existing broker positions from Upstox
    reconcile_positions()

    # 3. Warm up scanner with historical data
    warmup_system()

    # 4. Start Portfolio WebSocket Streamer (Real-time order fill updates)
    if USE_PORTFOLIO_STREAMER:
        try:
            portfolio_streamer = PortfolioStreamer(
                api_client=upstox_service.api_client,
                on_order_callback=trade_monitor.on_order_update
            )
            portfolio_streamer.start()
        except Exception as e:
            print(f"[PORTFOLIO STREAMER] Warning: {e}")

    # 5. Stream Message Handler
    def on_message(message):
        global ALLOW_NEW_TRADES

        feeds = message.get("feeds", {})
        now = datetime.datetime.now()
        today = now.date().isoformat()

        if today not in signals_today:
            signals_today[today] = set()

        current_prices = {}

        for inst_key, feed_info in feeds.items():
            # Check for Index Feed vs Stock MarketFF Feed
            full_feed = feed_info.get("fullFeed", {})
            data = full_feed.get("marketFF", {})
            index_data = full_feed.get("indexFF", {})

            # Handle Index Beta Update
            if index_data or inst_key.startswith("NSE_INDEX"):
                idx_ltp_raw = index_data.get("ltpc", {}).get("ltp") or feed_info.get("ltpc", {}).get("ltp")
                if idx_ltp_raw:
                    try:
                        macro_filter.update(float(idx_ltp_raw))
                    except Exception:
                        pass
                continue

            try:
                ltp = float(data["ltpc"]["ltp"])
            except Exception:
                continue

            current_prices[inst_key] = ltp

            # Extract OHLC and timestamp
            ohlc_list = data.get("marketOHLC", {}).get("ohlc", [])
            open_p = ltp
            high = ltp
            low = ltp
            close = ltp
            volume = 0.0
            time_iso = None

            if ohlc_list:
                bar = ohlc_list[-1]
                try:
                    open_p = float(bar.get("open", ltp))
                    high = float(bar.get("high", ltp))
                    low = float(bar.get("low", ltp))
                    close = float(bar.get("close", ltp))
                    volume = float(bar.get("vol", 0.0))
                    ts_raw = bar.get("ts")
                    if ts_raw and isinstance(ts_raw, (int, float)) and ts_raw > 1e11:
                        dt = datetime.datetime.fromtimestamp(ts_raw / 1000.0)
                        time_iso = dt.strftime("%Y-%m-%dT%H:%M:%S")
                except Exception:
                    pass

            # Ingest clean 1-minute OHLCV data & Order Flow Tick
            scanner.update(inst_key, ltp, high, low, close, volume, time_iso=time_iso, open_p=open_p)
            scanner.update_order_flow_tick(inst_key, ltp, volume)

            # Update Sector Performance Tracker for Sector Sympathy Gate
            try:
                from strategy.sector_manager import get_sector_manager
                ret = (ltp - open_p) / open_p if open_p > 0 else 0.0
                get_sector_manager().update_symbol_return(inst_key, ret)
            except Exception:
                pass


            # On-Demand Fast History Warmup: If stock has < 30 bars, fetch once in 80ms
            if not scanner.has_enough_data(inst_key, min_bars=30):
                if not hasattr(scanner, "_warmed_on_demand"):
                    scanner._warmed_on_demand = set()
                if inst_key not in scanner._warmed_on_demand:
                    scanner._warmed_on_demand.add(inst_key)
                    try:
                        candles = upstox_service.get_intraday_candles(inst_key, unit="minute", interval=1)
                        if candles:
                            scanner.warmup_candles(inst_key, candles)
                    except Exception:
                        pass

            # ⏰ Institutional Market Clock Gate:
            # - Strict start after 09:20 IST (Allow 5m price discovery)
            # - ABSOLUTE HARD STOP at 14:15 IST (Zero entries in final 60 mins before 15:15 square-off)
            from execution.execution_config import MARKET_START_TIME, MARKET_ENTRY_CUTOFF_TIME
            start_t = datetime.time.fromisoformat(MARKET_START_TIME)
            cutoff_t = datetime.time.fromisoformat(MARKET_ENTRY_CUTOFF_TIME)

            if now.time() < start_t or now.time() >= cutoff_t or not ALLOW_NEW_TRADES:
                continue

            # Multi-Factor Strategy Evaluation (with Level-2 Depth + Macro Alignment)
            decision = strategy_engine.evaluate(inst_key, ltp, market_ff=data)

            if not decision:
                continue

            if decision.state.startswith("EXECUTE"):
                if inst_key in signals_today[today]:
                    continue

                success = execution_engine.handle_entry(inst_key, decision, ltp)
                if success:
                    signals_today[today].add(inst_key)

        # Handle active position exits & EOD auto-square-off with complete price map
        if current_prices:
            eod_manager.check_and_execute(current_prices)
            execution_engine.handle_exits(current_prices, now)
            order_executor.check_stale_limit_orders()
            # Render JARVIS Command Center Telemetry HUD
            jarvis_hud.render(current_prices)


    # 6. Connect Parallel Streamers for All Nifty 500 Stocks (Upstox limit: max 100/socket)
    chunk_size = 95
    chunks = [INSTRUMENT_LIST[i:i + chunk_size] for i in range(0, len(INSTRUMENT_LIST), chunk_size)]
    active_streamers = []

    for idx, batch in enumerate(chunks, 1):
        try:
            s = upstox_client.MarketDataStreamerV3(
                upstox_service.api_client,
                batch,
                FEED_MODE
            )
            s.on("message", on_message)
            s.connect()
            active_streamers.append(s)
            time.sleep(0.12)
        except Exception as e:
            print(f"[STREAM POOL WARNING] Socket {idx} error: {e}")

    print(f"[JARVIS ONLINE] {len(active_streamers)} Parallel Sockets Active streaming {len(INSTRUMENT_LIST)} NIFTY 500 stocks.")
