# execution/execution_engine.py

class ExecutionEngine:
    def __init__(self, order_executor, trade_monitor, risk_manager, trade_logger):
        self.order_executor = order_executor
        self.trade_monitor = trade_monitor
        self.risk_manager = risk_manager
        self.trade_logger = trade_logger

    def handle_entry(self, inst_key, decision, ltp):
        if not self.risk_manager.can_trade_now():
            return

        side = "BUY" if "LONG" in decision.state else "SELL"

        order = self.order_executor.place_limit_order(
            inst_key=inst_key,
            side=side,
            price=ltp
        )

        if not order:
            return

        trade_id = order.get("order_id") or order.get("orderId")
        qty = order.get("quantity", 0)

        self.trade_monitor.add_trade(
            trade_id=trade_id,
            inst_key=inst_key,
            side=side,
            entry_price=ltp,
            qty=qty
        )

    def handle_exits(self, current_prices, now):
        exits = self.trade_monitor.check_trades(current_prices)

        for trade_id, reason, exit_price in exits:
            trade = self.trade_monitor.trades.get(trade_id)
            if not trade:
                continue

            self.trade_logger.log_trade(
                instrument=trade.inst_key,
                side=trade.side,
                quantity=trade.qty,
                entry_price=trade.entry_price,
                exit_price=exit_price,
                entry_time=trade.entry_time,
                exit_time=now,
                exit_reason=reason,
                strategy="elite_intraday_v2"
            )

            self.risk_manager.record_trade_outcome(reason)
            self.trade_monitor.remove_trade(trade_id)
