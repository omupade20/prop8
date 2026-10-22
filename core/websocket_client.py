# core/websocket_client.py
import json
import websocket
import threading
import requests
from config.settings import ACCESS_TOKEN
try:
    from upstox_client.feeder.proto import MarketDataFeedV3_pb2 as pb
except ImportError:
    import MarketDataFeedV3_pb2 as pb

"""
WebSocket V3 client to connect to Upstox Market Data Feed,
decode messages using protobuf, and print structured fields.
"""

def get_v3_authorized_url():
    """
    Calls Upstox v3 feed authorization endpoint to get the real WebSocket URL.
    """
    url = "https://api.upstox.com/v3/feed/market-data-feed/authorize"
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Accept": "application/json"
    }
    resp = requests.get(url, headers=headers)
    data = resp.json()

    if "data" not in data or "authorized_redirect_uri" not in data["data"]:
        print("Failed to authorize WebSocket:", data)
        return None

    return data["data"]["authorized_redirect_uri"]


def on_open(ws):
    """
    When socket opens, send subscription.
    """
    print("WebSocket connection established.")

    # Example subscription payload
    sub_req = {
        "guid": "sub1",
        "method": "sub",
        "data": {
            "mode": "full",
            "instrumentKeys": [
                # Example instrument keys
                "NSE_EQ|RELIANCE",
                "NSE_INDEX|NIFTY 50",
                "NSE_FO|<replace-with-futures-or-options>"
            ]
        }
    }

    # Send subscribe request
    ws.send(json.dumps(sub_req))
    print("Sent subscription:", sub_req)


def on_message(ws, message):
    """
    Called when a binary message arrives — decode it using Protobuf.
    """
    try:
        feed_response = pb.FeedResponse()
        feed_response.ParseFromString(message)

        # Example: Iterate through feeds
        for key, feed in feed_response.feeds.items():
            print(f"Instrument: {key}")

            # Last traded price
            if feed.HasField("ltpc"):
                print(" LTP:", feed.ltpc.ltp)

            # Print some depth / full mode info
            for lvl in feed.marketLevel.bidAskQuote:
                print(f"  BidP: {lvl.bidP}, BidQ: {lvl.bidQ}, AskP: {lvl.askP}, AskQ: {lvl.askQ}")

            # Example: Option Greeks (if present)
            if feed.HasField("optionGreeks"):
                print("  Delta:", feed.optionGreeks.delta)
                print("  Gamma:", feed.optionGreeks.gamma)

    except Exception as e:
        print("Error decoding message:", e)


def on_error(ws, error):
    print("WebSocket error:", error)


def on_close(ws, close_status_code, close_msg):
    print("WebSocket closed:", close_status_code, close_msg)


def start_market_feed():
    """
    Main entry to run the V3 WebSocket feed.
    """
    ws_url = get_v3_authorized_url()
    if not ws_url:
        print("Cannot start market feed.")
        return

    print("Connecting to:", ws_url)

    ws = websocket.WebSocketApp(
        ws_url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )

    # Run it in background
    thread = threading.Thread(target=ws.run_forever)
    thread.daemon = True
    thread.start()
