# core/upstox_service.py
"""
Central Upstox API Service.
Integrates the full suite of Upstox V2/V3 APIs:
- OrderApiV3: Order placement, cancellation, modification, GTT OCO orders
- HistoryV3Api: Intraday historical candle fetching for startup warm-up
- PortfolioApi: Live position reconciliation and holdings
- UserApi: Real-time buying power, margin, and funds
- ChargeApi: Pre-trade margin calculations and brokerage estimates
- MarketHolidaysAndTimingsApi: Market session and exchange timings
"""

import math
import datetime
from typing import Optional, Dict, List, Any
import upstox_client
from upstox_client.rest import ApiException
from config.settings import ACCESS_TOKEN


class UpstoxService:
    def __init__(self, access_token: Optional[str] = None):
        import importlib
        import config.settings
        importlib.reload(config.settings)
        self.access_token = access_token or config.settings.ACCESS_TOKEN
        self.config = upstox_client.Configuration()
        self.config.access_token = self.access_token
        self.api_client = upstox_client.ApiClient(self.config)

        # Initialize all Upstox API clients
        self.order_v3 = upstox_client.OrderApiV3(self.api_client)
        self.order_v2 = upstox_client.OrderApi(self.api_client)
        self.history_v3 = upstox_client.HistoryV3Api(self.api_client)
        self.history_v2 = upstox_client.HistoryApi(self.api_client)
        self.portfolio_api = upstox_client.PortfolioApi(self.api_client)
        self.user_api = upstox_client.UserApi(self.api_client)
        self.charge_api = upstox_client.ChargeApi(self.api_client)
        self.timing_api = upstox_client.MarketHolidaysAndTimingsApi(self.api_client)

    # -------------------------------------------------------------
    # 1. Market Status & Timings (MarketHolidaysAndTimingsApi)
    # -------------------------------------------------------------
    def get_market_status(self, exchange: str = "NSE") -> Dict[str, Any]:
        """
        Check if market is OPEN, PRE_OPEN, CLOSED, or in AUCTION.
        """
        try:
            resp = self.timing_api.get_market_status(exchange)
            status_data = resp.data if hasattr(resp, "data") else None
            status_str = status_data.status if status_data and hasattr(status_data, "status") else "UNKNOWN"
            return {"status": status_str, "exchange": exchange, "raw": status_data}
        except ApiException as e:
            print(f"[UpstoxService] get_market_status error: {e.status} {e.body}")
            return {"status": "UNKNOWN", "error": str(e)}
        except Exception as e:
            print(f"[UpstoxService] Unexpected get_market_status error: {e}")
            return {"status": "UNKNOWN", "error": str(e)}

    def is_market_open(self, exchange: str = "NSE") -> bool:
        """
        Returns True if market status indicates trading is allowed.
        """
        res = self.get_market_status(exchange)
        st = str(res.get("status", "")).upper()
        # Normal trading states in NSE: "NORMAL", "OPEN", "REGULAR"
        return st in ("NORMAL", "OPEN", "REGULAR", "TRADING")

    # -------------------------------------------------------------
    # 2. Account Funds & Margin (UserApi)
    # -------------------------------------------------------------
    def get_available_margin(self) -> float:
        """
        Query Upstox for live available equity margin.
        """
        try:
            resp = self.user_api.get_user_fund_margin(api_version="2.0")
            if hasattr(resp, "data") and resp.data:
                eq = getattr(resp.data, "equity", None)
                if eq:
                    return float(getattr(eq, "available_margin", 0.0) or 0.0)
            return 0.0
        except ApiException as e:
            print(f"[UpstoxService] get_user_fund_margin API error: {e.status}")
            return 0.0
        except Exception as e:
            print(f"[UpstoxService] Error fetching funds: {e}")
            return 0.0

    # -------------------------------------------------------------
    # 3. Historical Candle Pre-loading / Warm-up (HistoryV3Api)
    # -------------------------------------------------------------
    def get_intraday_candles(
        self,
        instrument_key: str,
        unit: str = "minute",
        interval: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Fetch completed intraday bars for today to pre-load MarketScanner.
        Returns sorted list (oldest to newest):
        [{'time': ISO, 'open': float, 'high': float, 'low': float, 'close': float, 'volume': float}]
        """
        clean_unit = "minutes" if unit in ("minute", "minutes") else unit
        raw_candles = []

        try:
            # 1. Try HistoryV3Api Intraday Candles
            resp = self.history_v3.get_intra_day_candle_data(
                instrument_key=instrument_key,
                unit=clean_unit,
                interval=interval
            )
            if hasattr(resp, "data") and resp.data:
                raw_candles = getattr(resp.data, "candles", []) or []
        except Exception:
            pass

        # 2. If intraday has 0 candles (e.g. before 09:15 AM or market closed), fetch recent historical bars
        if not raw_candles:
            try:
                today = datetime.date.today()
                from_d = (today - datetime.timedelta(days=5)).strftime("%Y-%m-%d")
                to_d = today.strftime("%Y-%m-%d")
                resp_hist = self.history_v3.get_historical_candle_data1(
                    instrument_key=instrument_key,
                    unit=clean_unit,
                    interval=interval,
                    to_date=to_d,
                    from_date=from_d
                )
                if hasattr(resp_hist, "data") and resp_hist.data:
                    raw_candles = getattr(resp_hist.data, "candles", []) or []
                    # Keep latest 60 bars
                    raw_candles = raw_candles[:60]
            except Exception:
                pass

        out = []
        for c in reversed(raw_candles):
            try:
                ts = c[0]
                time_str = ts.split("+")[0] if "+" in ts else ts
                out.append({
                    "time": time_str,
                    "open": float(c[1]),
                    "high": float(c[2]),
                    "low": float(c[3]),
                    "close": float(c[4]),
                    "volume": float(c[5])
                })
            except Exception:
                continue
        return out


    # -------------------------------------------------------------
    # 4. Position Reconciliation (PortfolioApi)
    # -------------------------------------------------------------
    def get_live_positions(self) -> List[Dict[str, Any]]:
        """
        Fetch all open intraday positions from Upstox to restore state on startup.
        """
        try:
            resp = self.portfolio_api.get_positions(api_version="2.0")
            if not hasattr(resp, "data") or not resp.data:
                return []

            positions = []
            for p in resp.data:
                net_qty = int(getattr(p, "quantity", 0) or 0)
                if net_qty != 0:
                    positions.append({
                        "instrument_token": getattr(p, "instrument_token", ""),
                        "trading_symbol": getattr(p, "trading_symbol", "") or getattr(p, "tradingsymbol", ""),
                        "quantity": abs(net_qty),
                        "side": "BUY" if net_qty > 0 else "SELL",
                        "average_price": float(getattr(p, "average_price", 0.0) or 0.0),
                        "product": getattr(p, "product", "I"),
                        "pnl": float(getattr(p, "pnl", 0.0) or 0.0)
                    })
            return positions
        except ApiException as e:
            print(f"[UpstoxService] get_positions API error: {e.status}")
            return []
        except Exception as e:
            print(f"[UpstoxService] Error fetching positions: {e}")
            return []

    # -------------------------------------------------------------
    # 5. Order Execution & Cancellation (OrderApiV3)
    # -------------------------------------------------------------
    def place_order(
        self,
        inst_key: str,
        side: str,
        qty: int,
        order_type: str = "LIMIT",
        price: float = 0.0,
        product: str = "I"
    ) -> Optional[Dict[str, Any]]:
        """
        Places an order on Upstox API V3.
        """
        if qty <= 0:
            return None

        body = upstox_client.PlaceOrderV3Request(
            quantity=qty,
            product=product,
            validity="DAY",
            price=round(price, 2) if order_type == "LIMIT" else 0.0,
            instrument_token=inst_key,
            order_type=order_type,
            transaction_type=side,
            disclosed_quantity=0,
            trigger_price=0.0,
            is_amo=False
        )

        try:
            resp = self.order_v3.place_order(body)
            resp_dict = resp.to_dict() if hasattr(resp, "to_dict") else {}
            data = resp_dict.get("data") or {}
            order_id = None
            if isinstance(data, dict):
                order_id = data.get("order_id")
                if not order_id and data.get("order_ids") and len(data["order_ids"]) > 0:
                    order_id = data["order_ids"][0]
            elif hasattr(resp, "data") and hasattr(resp.data, "order_ids") and resp.data.order_ids:
                order_id = resp.data.order_ids[0]

            print(f"[UpstoxService] Placed {side} {order_type} qty={qty} @ {price:.2f} for {inst_key}. OrderID: {order_id}")
            return {"status": "success", "order_id": order_id, "data": data}
        except ApiException as e:
            print(f"[UpstoxService] API error placing {side} order for {inst_key}: {e.status} {e.body}")
            return None
        except Exception as e:
            print(f"[UpstoxService] Unexpected error placing order: {e}")
            return None

    def get_order_book(self) -> List[Dict[str, Any]]:
        """
        Retrieves today's order book from Upstox.
        """
        try:
            resp = self.order_v2.get_order_book("2.0")
            if resp and resp.status == "success" and resp.data:
                return [o.to_dict() if hasattr(o, "to_dict") else o for o in resp.data]
            return []
        except Exception as e:
            print(f"[UpstoxService] Error fetching order book: {e}")
            return []

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancels an open order by ID.
        """
        if not order_id:
            return False
        try:
            self.order_v3.cancel_order(order_id)
            print(f"[UpstoxService] Cancelled order {order_id}")
            return True
        except ApiException as e:
            print(f"[UpstoxService] Error cancelling order {order_id}: {e.status}")
            return False
        except Exception as e:
            print(f"[UpstoxService] Error cancelling order {order_id}: {e}")
            return False

    # -------------------------------------------------------------
    # 6. Server-Side GTT OCO Orders (Stop Loss & Target)
    # -------------------------------------------------------------
    def place_gtt_oco(
        self,
        inst_key: str,
        side: str,
        qty: int,
        sl_trigger_price: float,
        target_trigger_price: float,
        product: str = "I"
    ) -> Optional[Dict[str, Any]]:
        """
        Places a broker server-side GTT OCO (One-Cancels-Other) order.
        For a BUY entry position, the exit side is SELL.
        For a SELL entry position, the exit side is BUY.
        """
        exit_side = "SELL" if side == "BUY" else "BUY"

        rules = [
            upstox_client.GttRule(
                strategy="STOPLOSS",
                trigger_price=round(sl_trigger_price, 2),
                trigger_type="IMMEDIATE"
            ),
            upstox_client.GttRule(
                strategy="TARGET",
                trigger_price=round(target_trigger_price, 2),
                trigger_type="IMMEDIATE"
            )
        ]

        body = upstox_client.GttPlaceOrderRequest(
            type="MULTIPLE",
            quantity=qty,
            product=product,
            rules=rules,
            transaction_type=exit_side,
            instrument_token=inst_key
        )

        try:
            resp = self.order_v3.place_gtt_order(body)
            resp_dict = resp.to_dict()
            gtt_id = resp_dict.get("data", {}).get("gtt_order_id") if isinstance(resp_dict.get("data"), dict) else None
            print(f"[UpstoxService] Placed GTT OCO for {inst_key}. SL={sl_trigger_price:.2f}, TP={target_trigger_price:.2f}. GTT ID: {gtt_id}")
            return {"status": "success", "gtt_id": gtt_id, "data": resp_dict.get("data")}
        except ApiException as e:
            print(f"[UpstoxService] API error placing GTT for {inst_key}: {e.status} {e.body}")
            return None
        except Exception as e:
            print(f"[UpstoxService] Unexpected error placing GTT: {e}")
            return None

    def cancel_gtt(self, gtt_order_id: str) -> bool:
        """
        Cancels an active server-side GTT order on Upstox.
        """
        if not gtt_order_id:
            return False
        try:
            body = upstox_client.GttCancelOrderRequest(gtt_order_id=str(gtt_order_id))
            self.order_v3.cancel_gtt_order(body)
            print(f"[UpstoxService] Successfully cancelled GTT order {gtt_order_id}")
            return True
        except ApiException as e:
            print(f"[UpstoxService] API error cancelling GTT {gtt_order_id}: {e.status}")
            return False
        except Exception as e:
            print(f"[UpstoxService] Error cancelling GTT {gtt_order_id}: {e}")
            return False

    # -------------------------------------------------------------
    # 7. Brokerage & Charges Estimation (ChargeApi)
    # -------------------------------------------------------------

    def estimate_brokerage(
        self,
        instrument_token: str,
        quantity: int,
        price: float,
        side: str = "BUY",
        product: str = "I"
    ) -> Dict[str, float]:
        """
        Calculate exact brokerage, STT, exchange charges, and taxes for an order.
        """
        try:
            resp = self.charge_api.get_brokerage(
                instrument_token=instrument_token,
                quantity=quantity,
                product=product,
                transaction_type=side,
                price=price,
                api_version="2.0"
            )
            charges = getattr(resp, "data", None)
            if hasattr(charges, "charges"):
                c = charges.charges
                return {
                    "brokerage": float(getattr(c, "brokerage", 0.0) or 0.0),
                    "stt": float(getattr(c, "stt", 0.0) or 0.0),
                    "taxes": float(getattr(c, "gst", 0.0) or 0.0) + float(getattr(c, "stamp_duty", 0.0) or 0.0),
                    "total_charges": float(getattr(charges, "total_charges", 0.0) or 0.0)
                }
            return {"brokerage": 20.0, "total_charges": 25.0}
        except Exception:
            # Standard Upstox intraday estimate: ₹20 or 0.05% per order
            return {"brokerage": 20.0, "total_charges": 25.0}
