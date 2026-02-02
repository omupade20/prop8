# core/market_streamer.py

import json
import datetime
import upstox_client
from config.settings import ACCESS_TOKEN

from strategy.scanner import MarketScanner
from strategy.vwap_filter import VWAPCalculator
from strategy.strategy_engine import StrategyEngine

from execution.execution_engine import ExecutionEngine
from execution.order_executor import OrderExecutor
from execution.trade_monitor import TradeMonitor
from execution.risk_manager import RiskManager
from execution.trade_logger import TradeLogger


FEED_MODE = "full"

# ---------------- LOAD UNIVERSE ----------------
with open("data/nifty500_keys.json", "r") as f:
    INSTRUMENT_LIST = json.load(f)

# ---------------- CORE OBJECTS ----------------
scanner = MarketScanner(max_len=600)
vwap_calculators = {inst: VWAPCalculator() for inst in INSTRUMENT_LIST}

strategy_engine = StrategyEngine(scanner, vwap_calculators)

order_executor = OrderExecutor()
trade_monitor = TradeMonitor()
risk_manager = RiskManager()
trade_logger = TradeLogger()

execution_engine = ExecutionEngine(
    order_executor,
    trade_monitor,
    risk_manager,
    trade_logger
)

signals_today = {}
ALLOW_NEW_TRADES = True


# ---------------- STREAMER ----------------
def start_market_streamer():
    global ALLOW_NEW_TRADES

    config = upstox_client.Configuration()
    config.access_token = ACCESS_TOKEN
    api_client = upstox_client.ApiClient(config)

    streamer = upstox_client.MarketDataStreamerV3(
        api_client,
        INSTRUMENT_LIST,
        FEED_MODE
    )

    def on_message(message):
        global ALLOW_NEW_TRADES

        feeds = message.get("feeds", {})
        now = datetime.datetime.now()
        today = now.date().isoformat()

        if today not in signals_today:
            signals_today[today] = set()

        current_prices = {}

        for inst_key, feed_info in feeds.items():
            data = feed_info.get("fullFeed", {}).get("marketFF", {})

            try:
                ltp = float(data["ltpc"]["ltp"])
            except Exception:
                continue

            current_prices[inst_key] = ltp

            ohlc = data.get("marketOHLC", {}).get("ohlc", [])
            if not ohlc:
                continue

            bar = ohlc[-1]
            try:
                high = float(bar["high"])
                low = float(bar["low"])
                close = float(bar["close"])
                volume = float(bar["vol"])
            except Exception:
                continue

            # ---- Update Market State ----
            scanner.update(inst_key, ltp, high, low, close, volume)

            # ---- Strategy Evaluation ----
            decision = strategy_engine.evaluate(inst_key, ltp)

            if not decision:
                continue

            if decision.state.startswith("EXECUTE"):
                if inst_key in signals_today[today]:
                    continue

                signals_today[today].add(inst_key)
                execution_engine.handle_entry(inst_key, decision, ltp)

        # ---- Exit Handling ----
        execution_engine.handle_exits(current_prices, now)

    streamer.on("message", on_message)
    streamer.connect()

    print("🚀 Elite intraday trading system started (refactored & stable)")
