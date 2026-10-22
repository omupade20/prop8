# main.py
import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from core.upstox_auth import ensure_valid_session
from execution.execution_config import CAPITAL_PER_TRADE, MIS_LEVERAGE_MULTIPLIER, USE_MIS_LEVERAGE, PAPER_TRADING

def start_health_server():
    """Starts a minimal background HTTP server so cloud platforms (Koyeb, Render) see an active web port."""
    import os
    import threading
    from http.server import HTTPServer, BaseHTTPRequestHandler

    port = int(os.environ.get("PORT", 8000))

    class HealthHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "JARVIS_QUANT_ONLINE", "code": 200}')

        def log_message(self, format, *args):
            return

    try:
        server = HTTPServer(("0.0.0.0", port), HealthHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        print(f"[CLOUD HEALTH] HTTP Health Server listening on port {port} for cloud deployment.")
    except Exception as e:
        pass

def start_system():
    print("[INIT] Starting JARVIS Trading System ...")
    
    # 0. Start Cloud Health Server (for Koyeb/Render deployment)
    start_health_server()

    # 1. Pre-flight session check & automated token verification
    valid_token = ensure_valid_session()

    if not valid_token or len(valid_token) < 20:
        print("[JARVIS STANDBY] Awaiting fresh Upstox Access Token. Cloud health check remains 200 OK.")
    else:
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

