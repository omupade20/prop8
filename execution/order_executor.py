# execution/order_executor.py
"""
Order Executor using Upstox API V3.
Supports:
- Dynamic margin checking via UserApi
- Limit entry orders with slippage buffer
- Market / Limit exit orders (Opposing transaction type)
- Server-side GTT OCO Stop Loss & Target orders
- Order cancellation
"""

import math
import time
from typing import Optional, Dict, Any
from core.upstox_service import UpstoxService
from execution.execution_config import (
    CAPITAL_PER_TRADE,
    DYNAMIC_MARGIN_SIZING,
    LIMIT_BUFFER_PCT,
    USE_PASSIVE_PEGGING,
    PASSIVE_PEG_BUFFER_PCT,
    PASSIVE_PEG_TIMEOUT_SECONDS,
    EXIT_ORDER_TYPE,
    USE_GTT_OCO,
    PAPER_TRADING,
    LIMIT_ORDER_TIMEOUT_SECONDS
)


class OrderExecutor:
    def __init__(self, upstox_service: Optional[UpstoxService] = None):
        self.service = upstox_service or UpstoxService()
        self.pending_orders: Dict[str, Dict[str, Any]] = {}

    def calculate_quantity(self, price: float, atr: Optional[float] = None) -> int:
        """
        Calculate number of shares based on institutional ATR risk parity and available margin.
        """
        if price <= 0:
            return 0

        avail_margin = self.service.get_available_margin() if DYNAMIC_MARGIN_SIZING else None

        try:
            from execution.volatility_sizer import calculate_risk_parity_quantity
            return calculate_risk_parity_quantity(
                price=price,
                atr=atr,
                max_capital=CAPITAL_PER_TRADE,
                available_margin=avail_margin
            )
        except Exception:
            target_capital = CAPITAL_PER_TRADE
            if avail_margin and 0 < avail_margin < target_capital:
                target_capital = avail_margin * 0.95
            return max(0, math.floor(target_capital / price))

    def place_limit_order(
        self,
        inst_key: str,
        side: str,
        price: float,
        qty: Optional[int] = None,
        atr: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Places a LIMIT order on Upstox V3 using current LTP + buffer.
        """
        if qty is None:
            qty = self.calculate_quantity(price, atr=atr)

        if qty < 1:
            qty = 1

        # Adjust limit price: Smart Passive Pegging (Bid/Ask peg) vs Aggressive Limit
        if USE_PASSIVE_PEGGING:
            if side == "BUY":
                limit_price = price * (1 - PASSIVE_PEG_BUFFER_PCT)
            elif side == "SELL":
                limit_price = price * (1 + PASSIVE_PEG_BUFFER_PCT)
            peg_mode_str = "PASSIVE_PEG (Spread Saved)"
        else:
            if side == "BUY":
                limit_price = price * (1 + LIMIT_BUFFER_PCT)
            elif side == "SELL":
                limit_price = price * (1 - LIMIT_BUFFER_PCT)
            peg_mode_str = "AGGRESSIVE_LIMIT"

        if side not in ("BUY", "SELL"):
            print(f"[OrderExecutor] Invalid order side: {side}")
            return None

        # 🧪 Layered Paper Trading Simulation Mode
        if PAPER_TRADING:
            sim_id = f"SIM_{int(time.time() * 1000)}"
            print(f"[PAPER TRADING] [{peg_mode_str}] Simulated LIMIT {side} order: {qty} shares of {inst_key} @ Rs. {limit_price:.2f}")
            return {
                "order_id": sim_id,
                "quantity": qty,
                "limit_price": round(limit_price, 2),
                "side": side,
                "raw": {"status": "success", "order_id": sim_id}
            }

        result = self.service.place_order(
            inst_key=inst_key,
            side=side,
            qty=qty,
            order_type="LIMIT",
            price=limit_price,
            product="I"
        )

        if not result:
            return None

        order_id = result.get("order_id")
        if order_id:
            self.pending_orders[order_id] = {
                "time": time.time(),
                "inst_key": inst_key,
                "qty": qty
            }

        return {
            "order_id": order_id,
            "quantity": qty,
            "limit_price": round(limit_price, 2),
            "side": side,
            "raw": result
        }

    def place_exit_order(
        self,
        inst_key: str,
        entry_side: str,
        qty: int,
        current_price: float = 0.0
    ) -> Optional[Dict[str, Any]]:
        """
        Places an opposing exit order to close out an intraday position.
        BUY position -> SELL exit.
        SELL position -> BUY exit.
        """
        if qty <= 0:
            return None

        exit_side = "SELL" if entry_side.upper() == "BUY" else "BUY"

        if PAPER_TRADING:
            sim_id = f"SIM_EXIT_{int(time.time() * 1000)}"
            print(f"[PAPER TRADING] Simulated EXIT {exit_side} order: {qty} shares of {inst_key} @ Rs. {current_price:.2f}")
            return {"order_id": sim_id, "quantity": qty, "side": exit_side, "raw": {"status": "success"}}

        order_type = EXIT_ORDER_TYPE  # "MARKET" or "LIMIT"

        # If LIMIT exit, apply safety buffer
        if order_type == "LIMIT" and current_price > 0:
            limit_price = current_price * (1 - LIMIT_BUFFER_PCT) if exit_side == "SELL" else current_price * (1 + LIMIT_BUFFER_PCT)
        else:
            limit_price = 0.0
            order_type = "MARKET"

        print(f"[OrderExecutor] Placing EXIT {exit_side} {order_type} qty={qty} for {inst_key}")
        result = self.service.place_order(
            inst_key=inst_key,
            side=exit_side,
            qty=qty,
            order_type=order_type,
            price=limit_price,
            product="I"
        )

        if not result:
            return None

        return {
            "order_id": result.get("order_id"),
            "quantity": qty,
            "side": exit_side,
            "raw": result
        }

    def place_gtt_protection(
        self,
        inst_key: str,
        side: str,
        qty: int,
        sl_price: float,
        tp_price: float
    ) -> Optional[Dict[str, Any]]:
        """
        Places a broker server-side GTT OCO order for automated exchange-level Stop Loss and Target.
        """
        if not USE_GTT_OCO:
            return None

        if PAPER_TRADING:
            return {"gtt_id": f"SIM_GTT_{int(time.time() * 1000)}", "status": "success"}

        return self.service.place_gtt_oco(
            inst_key=inst_key,
            side=side,
            qty=qty,
            sl_trigger_price=sl_price,
            target_trigger_price=tp_price,
            product="I"
        )

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an open unfilled limit order.
        """
        return self.service.cancel_order(order_id)

    def cancel_gtt(self, gtt_id: str) -> bool:
        """
        Cancel an open GTT OCO order to prevent duplicate fills.
        """
        return self.service.cancel_gtt(gtt_id)

    def check_stale_limit_orders(self):
        """
        Cancels limit orders that have stayed unfilled past LIMIT_ORDER_TIMEOUT_SECONDS.
        """
        if PAPER_TRADING or not self.pending_orders:
            return

        now = time.time()
        for order_id, info in list(self.pending_orders.items()):
            if now - info["time"] >= LIMIT_ORDER_TIMEOUT_SECONDS:
                print(f"[OrderExecutor] Stale limit order {order_id} for {info['inst_key']} timed out ({LIMIT_ORDER_TIMEOUT_SECONDS}s). Canceling...")
                try:
                    self.cancel_order(order_id)
                except Exception:
                    pass
                del self.pending_orders[order_id]

