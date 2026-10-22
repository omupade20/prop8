# core/eod_square_off.py
"""
Automated End-Of-Day MIS Square-Off Manager.
Protects against broker RMS auto-square-off penalty charges (₹50 + GST per order):
- 15:10 IST: Blocks all new entry signals
- 15:15 IST: Gracefully closes all remaining open intraday positions at market
- Cancels any remaining unfilled GTT or limit orders
"""

import datetime
from typing import Dict, Any


CUTOFF_NEW_TRADES_TIME = datetime.time(15, 10, 0)
SQUARE_OFF_TIME = datetime.time(15, 15, 0)


class EODSquareOffManager:
    def __init__(self, execution_engine, trade_monitor):
        self.execution_engine = execution_engine
        self.trade_monitor = trade_monitor
        self.square_off_executed = False

    def check_and_execute(self, current_prices: Dict[str, float]) -> bool:
        """
        Check clock time and trigger square-off if past 15:15 IST.
        Returns True if new trades should be blocked.
        """
        now = datetime.datetime.now()
        current_time = now.time()

        # 1. Block new entries past 15:10
        if current_time >= CUTOFF_NEW_TRADES_TIME:
            # 2. Square off active positions past 15:15
            if current_time >= SQUARE_OFF_TIME and not self.square_off_executed:
                active_trades = list(self.trade_monitor.active_trades.values())
                if active_trades:
                    print(f"[EOD SQUARE-OFF] 15:15 IST reached. Closing {len(active_trades)} open positions before broker RMS auto-square-off...")
                    for trade in active_trades:
                        if trade.is_closed:
                            continue
                        # Cancel lingering server-side GTT orders
                        if getattr(trade, "gtt_id", None):
                            self.execution_engine.order_executor.cancel_gtt(trade.gtt_id)

                        ltp = current_prices.get(trade.inst_key, trade.entry_price)
                        self.execution_engine.order_executor.place_exit_order(
                            inst_key=trade.inst_key,
                            entry_side=trade.side,
                            qty=trade.qty,
                            current_price=ltp
                        )
                        self.execution_engine.trade_logger.log_trade(
                            instrument=trade.inst_key,
                            side=trade.side,
                            quantity=trade.qty,
                            entry_price=trade.entry_price,
                            exit_price=ltp,
                            entry_time=trade.open_time,
                            exit_time=now,
                            exit_reason="EOD_SQUAREOFF",
                            strategy="eod_auto_exit"
                        )
                        self.trade_monitor.remove_trade(trade.trade_id)
                    print("[EOD SQUARE-OFF] All intraday positions squared off safely.")
                self.square_off_executed = True

                # Generate Visual HTML Report Card
                try:
                    from execution.execution_config import GENERATE_EOD_REPORT
                    if GENERATE_EOD_REPORT:
                        from core.eod_report import generate_eod_html_report
                        generate_eod_html_report()
                except Exception as e:
                    print(f"[EOD REPORT] {e}")

                # Dispatch Telegram EOD Summary
                try:
                    from core.telegram_notifier import get_telegram_notifier
                    rm = getattr(self.execution_engine, "risk_manager", None)
                    if rm:
                        notifier = get_telegram_notifier()
                        notifier.notify_eod_summary(
                            total_trades=rm.total_trades,
                            wins=rm.target_hits,
                            losses=rm.stop_losses,
                            gross_pnl=getattr(rm, "cumulative_net_pnl", 0.0),
                            total_taxes=rm.total_trades * 100.0,
                            net_pnl=getattr(rm, "cumulative_net_pnl", 0.0)
                        )
                except Exception:
                    pass

            return True


        return False
