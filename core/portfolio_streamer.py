# core/portfolio_streamer.py
"""
Portfolio and Order Data Streamer.
Listens to real-time WebSocket pushes for:
- Order status changes (OPEN, COMPLETE/FILLED, CANCELLED, REJECTED)
- Position updates
- GTT order triggers
"""

import json
import threading
from typing import Callable, Optional
import upstox_client
from config.settings import ACCESS_TOKEN


class PortfolioStreamer:
    def __init__(
        self,
        api_client: Optional[upstox_client.ApiClient] = None,
        on_order_callback: Optional[Callable[[dict], None]] = None,
        on_position_callback: Optional[Callable[[dict], None]] = None
    ):
        if api_client is None:
            config = upstox_client.Configuration()
            config.access_token = ACCESS_TOKEN
            self.api_client = upstox_client.ApiClient(config)
        else:
            self.api_client = api_client

        self.on_order_callback = on_order_callback
        self.on_position_callback = on_position_callback
        self.streamer: Optional[upstox_client.PortfolioDataStreamer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """
        Connect and start listening to portfolio and order feeds in a daemon thread.
        """
        try:
            self.streamer = upstox_client.PortfolioDataStreamer(
                api_client=self.api_client,
                order_update=True,
                position_update=True,
                holding_update=False,
                gtt_update=True
            )

            def handle_open():
                print("[PortfolioStreamer] Connected to Upstox Order/Portfolio Stream.")

            def handle_message(message):
                try:
                    data = message
                    if isinstance(message, str):
                        data = json.loads(message)

                    update_type = data.get("update_type") or data.get("type")

                    if update_type == "order":
                        order_id = data.get("order_id")
                        status = data.get("status", "").upper()
                        print(f"[PortfolioStreamer] Order Update: {order_id} -> {status}")
                        if self.on_order_callback:
                            self.on_order_callback(data)

                    elif update_type == "position":
                        if self.on_position_callback:
                            self.on_position_callback(data)

                except Exception as e:
                    print(f"[PortfolioStreamer] Error processing message: {e}")

            def handle_error(err):
                print(f"[PortfolioStreamer] Stream error: {err}")

            def handle_close(code, reason):
                print(f"[PortfolioStreamer] Stream closed ({code}): {reason}")

            self.streamer.on("open", handle_open)
            self.streamer.on("message", handle_message)
            self.streamer.on("error", handle_error)
            self.streamer.on("close", handle_close)

            # Start connection
            self.streamer.connect()
        except Exception as e:
            print(f"[PortfolioStreamer] Failed to start portfolio streamer: {e}")
