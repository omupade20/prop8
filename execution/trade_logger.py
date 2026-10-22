# execution/trade_logger.py
"""
Trade Logger.
Records completed trades to CSV with gross P&L, estimated brokerage & taxes, and net P&L.
"""

import csv
import os
import datetime
from typing import Union


class TradeLogger:
    """
    Logs all completed trades into ONE master CSV file.
    """

    def __init__(self, file_path="logs/trades/all_trades.csv"):
        self.file_path = file_path
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        self._ensure_header()

    def _ensure_header(self):
        if not os.path.exists(self.file_path):
            with open(self.file_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "date",
                    "entry_time",
                    "exit_time",
                    "instrument",
                    "side",
                    "quantity",
                    "entry_price",
                    "exit_price",
                    "gross_pnl",
                    "charges",
                    "net_pnl",
                    "pnl_pct",
                    "exit_reason",
                    "strategy",
                    "remarks"
                ])

    def log_trade(
        self,
        instrument: str,
        side: str,
        quantity: int,
        entry_price: float,
        exit_price: float,
        entry_time: Union[datetime.datetime, str],
        exit_time: Union[datetime.datetime, str],
        exit_reason: str,
        strategy: str = "elite_intraday_v2",
        remarks: str = ""
    ):
        """
        Append one completed trade to the master CSV.
        """
        now = datetime.datetime.now()
        if isinstance(entry_time, str):
            try:
                entry_dt = datetime.datetime.fromisoformat(entry_time)
            except Exception:
                entry_dt = now
        else:
            entry_dt = entry_time or now

        if isinstance(exit_time, str):
            try:
                exit_dt = datetime.datetime.fromisoformat(exit_time)
            except Exception:
                exit_dt = now
        else:
            exit_dt = exit_time or now

        # P&L Calculations
        if side.upper() == "BUY":
            gross_pnl = (exit_price - entry_price) * quantity
            pnl_pct = ((exit_price - entry_price) / max(entry_price, 1e-9)) * 100
        else:
            gross_pnl = (entry_price - exit_price) * quantity
            pnl_pct = ((entry_price - exit_price) / max(entry_price, 1e-9)) * 100

        # Exact Indian statutory taxes & brokerage calculation
        buy_turnover = entry_price * quantity if side.upper() == "BUY" else exit_price * quantity
        sell_turnover = exit_price * quantity if side.upper() == "BUY" else entry_price * quantity
        try:
            from execution.volatility_sizer import calculate_indian_statutory_taxes
            tax_dict = calculate_indian_statutory_taxes(buy_turnover, sell_turnover)
            estimated_charges = tax_dict["total_charges"]
        except Exception:
            estimated_charges = 50.0

        net_pnl = gross_pnl - estimated_charges


        try:
            with open(self.file_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    entry_dt.date().isoformat(),
                    entry_dt.strftime("%H:%M:%S"),
                    exit_dt.strftime("%H:%M:%S"),
                    instrument,
                    side,
                    quantity,
                    round(entry_price, 2),
                    round(exit_price, 2),
                    round(gross_pnl, 2),
                    round(estimated_charges, 2),
                    round(net_pnl, 2),
                    round(pnl_pct, 2),
                    exit_reason,
                    strategy,
                    remarks
                ])
            print(f"[TradeLogger] Logged {side} trade for {instrument}: Net PnL = Rs. {net_pnl:.2f} ({pnl_pct:.2f}%)")
            return {
                "instrument": instrument,
                "side": side,
                "quantity": quantity,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_pnl": gross_pnl,
                "charges": estimated_charges,
                "net_pnl": net_pnl,
                "pnl_pct": pnl_pct,
                "exit_reason": exit_reason
            }
        except Exception as e:
            print(f"[TradeLogger] Failed to write trade to CSV: {e}")
            return {
                "instrument": instrument,
                "side": side,
                "quantity": quantity,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_pnl": gross_pnl,
                "charges": estimated_charges,
                "net_pnl": net_pnl,
                "pnl_pct": pnl_pct,
                "exit_reason": exit_reason
            }
