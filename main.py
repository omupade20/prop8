# main.py
import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from core.upstox_auth import ensure_valid_session
from execution.execution_config import CAPITAL_PER_TRADE, MIS_LEVERAGE_MULTIPLIER, USE_MIS_LEVERAGE, PAPER_TRADING

def start_system():
    print("[INIT] Starting JARVIS Trading System ...")
    
    # 1. Pre-flight session check & automated token verification
    ensure_valid_session()

    # 2. Dispatch Mobile Telegram Broadcast
    buying_power = CAPITAL_PER_TRADE * (MIS_LEVERAGE_MULTIPLIER if USE_MIS_LEVERAGE else 1.0)
    try:
        from core.telegram_notifier import get_telegram_notifier
        telegram = get_telegram_notifier()
        telegram.notify_startup(
            capital=CAPITAL_PER_TRADE,
            buying_power=buying_power,
            paper_mode=PAPER_TRADING
        )
    except Exception:
        pass

    # 3. Launch institutional multi-socket Nifty 500 streamer & execution engine
    from core.market_streamer import start_market_streamer
    start_market_streamer()

    # Keep the script running so WebSocket stays alive
    import time
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[STOP] Trading system stopped by user.")

if __name__ == "__main__":
    start_system()

