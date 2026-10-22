# execution/execution_engine.py
"""
Execution Engine.
Connects signals from StrategyEngine to:
- OrderExecutor (places real Upstox orders + GTT OCO protection)
- TradeMonitor (tracks open positions and trailing stops)
- RiskManager (enforces daily risk circuit breakers)
- TradeLogger (records executions and P&L)
"""

from typing import Dict, Any
from strategy.sector_manager import get_sector_manager
from core.telegram_notifier import get_telegram_notifier


class ExecutionEngine:
    def __init__(self, order_executor, trade_monitor, risk_manager, trade_logger):
        self.order_executor = order_executor
        self.trade_monitor = trade_monitor
        self.risk_manager = risk_manager
        self.trade_logger = trade_logger
        self.sector_manager = get_sector_manager()
        self.telegram = get_telegram_notifier()

    def handle_entry(self, inst_key: str, decision: Any, ltp: float) -> bool:
        """
        Executes a new trade entry when risk rules allow.
        Returns True if order was placed, False otherwise.
        """
        if not self.risk_manager.can_trade_now():
            print(f"[ExecutionEngine] Trade entry blocked by RiskManager for {inst_key}")
            return False

        # Sector & Concurrent Exposure Gate
        active_symbols = [t.inst_key for t in self.trade_monitor.active_trades.values()]
        can_enter, reason = self.sector_manager.can_enter_trade(inst_key, active_symbols)
        if not can_enter:
            print(f"[ExecutionEngine] Diversification Gate: {reason} for {inst_key}. Skipped.")
            return False

        side = "BUY" if "LONG" in decision.state else "SELL"

        # 🧠 Dynamic Kelly Conviction Sizing
        from execution.volatility_sizer import calculate_kelly_conviction_quantity
        conviction = (getattr(decision, "score", 7.5) / 10.0) if hasattr(decision, "score") else 0.75
        kelly_qty, grade_str, cap_used = calculate_kelly_conviction_quantity(price=ltp, conviction_score=conviction)
        print(f"[ExecutionEngine] [KELLY SIZING] {grade_str} -> {kelly_qty} shares (Deploys Rs. {cap_used:,.0f})")

        order = self.order_executor.place_limit_order(
            inst_key=inst_key,
            side=side,
            price=ltp,
            qty=kelly_qty
        )

        if not order:
            return False

        trade_id = order.get("order_id")
        qty = order.get("quantity", 0)
        atr_val = getattr(decision, "atr", None) or (decision.components.get("atr") if hasattr(decision, "components") else None)

        trade = self.trade_monitor.add_trade(
            trade_id=trade_id,
            inst_key=inst_key,
            side=side,
            entry_price=ltp,
            qty=qty,
            atr=atr_val
        )

        # Place broker server-side GTT Stop Loss / Target order
        gtt_res = self.order_executor.place_gtt_protection(
            inst_key=inst_key,
            side=side,
            qty=qty,
            sl_price=trade.stop_loss,
            tp_price=trade.target
        )
        if gtt_res and gtt_res.get("gtt_id"):
            trade.gtt_id = gtt_res.get("gtt_id")

        # Telegram Mobile Notification
        ai_prob = getattr(decision, "win_probability", 0.75) * 100.0 if hasattr(decision, "win_probability") else 75.0
        self.telegram.notify_trade_entry(
            symbol=inst_key,
            side=side,
            qty=qty,
            price=ltp,
            win_prob=ai_prob
        )

        return True

    def handle_exits(self, current_prices: Dict[str, float], now):
        """
        Evaluates active positions and executes opposing exit orders at the broker.
        """
        exits = self.trade_monitor.check_trades(current_prices)

        for trade_id, reason, exit_price in exits:
            trade = self.trade_monitor.active_trades.get(trade_id)
            if not trade:
                continue

            # Cancel server-side GTT OCO order to prevent phantom fills
            if getattr(trade, "gtt_id", None):
                self.order_executor.cancel_gtt(trade.gtt_id)

            # Execute closing order at Upstox
            exit_order = self.order_executor.place_exit_order(
                inst_key=trade.inst_key,
                entry_side=trade.side,
                qty=trade.qty,
                current_price=exit_price
            )

            # Log completed trade and compute exact Indian statutory taxes
            log_result = self.trade_logger.log_trade(
                instrument=trade.inst_key,
                side=trade.side,
                quantity=trade.qty,
                entry_price=trade.entry_price,
                exit_price=exit_price,
                entry_time=trade.open_time,
                exit_time=now,
                exit_reason=reason,
                strategy="elite_intraday_v2"
            )

            net_pnl = log_result.get("net_pnl", 0.0) if log_result else 0.0
            gross_pnl = log_result.get("gross_pnl", 0.0) if log_result else 0.0
            charges = log_result.get("charges", 0.0) if log_result else 0.0

            # Update daily risk metrics with actual post-tax net PnL
            self.risk_manager.record_trade_outcome(reason, net_pnl=net_pnl)
            
            # Adaptive Sector Cooldown: lock sector if stopped out
            if reason == "STOP_LOSS":
                from strategy.sector_manager import get_sector_manager
                sec_mgr = get_sector_manager()
                sec = sec_mgr.get_sector(trade.inst_key)
                sec_mgr.trigger_cooldown(sec)

            self.trade_monitor.remove_trade(trade_id)

            # Send Mobile Alert
            self.telegram.notify_trade_exit(
                symbol=trade.inst_key,
                side=trade.side,
                qty=trade.qty,
                entry_price=trade.entry_price,
                exit_price=exit_price,
                gross_pnl=gross_pnl,
                charges=charges,
                net_pnl=net_pnl,
                reason=reason
            )

            # Check if Hard Daily Loss Circuit Breaker was triggered
            if self.risk_manager.circuit_breaker_triggered:
                self.telegram.notify_circuit_breaker(self.risk_manager.circuit_breaker_reason)
