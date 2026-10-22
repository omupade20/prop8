# core/telegram_notifier.py
"""
JARVIS Telegram Notifier.
Delivers real-time push notifications to your mobile phone:
- System Startup & Morning Capital Status
- Trade Entries with Shares, Buying Power, & AI Win-Probability
- Dynamic Breakeven Locks (+0.50% profit achieved)
- Trade Exits with Gross PnL, Indian Taxes, & Net Take-Home PnL
- Hard Daily Loss Circuit Breakers
- 15:15 IST End-of-Day Performance Summary
"""

import threading
import requests
from typing import Optional
from execution.execution_config import (
    TELEGRAM_ENABLED,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID
)


class TelegramNotifier:
    def __init__(
        self,
        enabled: bool = TELEGRAM_ENABLED,
        bot_token: str = TELEGRAM_BOT_TOKEN,
        chat_id: str = TELEGRAM_CHAT_ID
    ):
        self.enabled = enabled and bool(bot_token) and bool(chat_id)
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.api_url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"

    def _send_async(self, text: str):
        """Dispatches HTTP request in a background daemon thread to avoid blocking execution."""
        if not self.enabled:
            return

        def _worker():
            try:
                payload = {
                    "chat_id": self.chat_id,
                    "text": text,
                    "parse_mode": "Markdown"
                }
                requests.post(self.api_url, json=payload, timeout=4)
            except Exception:
                # Do not interrupt trading if telegram network times out
                pass

        threading.Thread(target=_worker, daemon=True).start()

    def notify_startup(self, capital: float, buying_power: float, paper_mode: bool = False):
        mode = "🧪 *PAPER SIMULATION*" if paper_mode else "⚡ *LIVE UPSTOX MIS CASH*"
        msg = (
            f"🤖 *JARVIS QUANT ENGINE ONLINE*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• Mode: {mode}\n"
            f"• Cash / Trade: `Rs. {capital:,.2f}`\n"
            f"• 5x Buying Power: `Rs. {buying_power:,.2f}`\n"
            f"• AI Gate: `70% Win-Probability`\n"
            f"• Status: *Active & Monitoring 100 Symbols*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"_Have a productive day at work! JARVIS is on watch._"
        )
        self._send_async(msg)

    def notify_trade_entry(self, symbol: str, side: str, qty: int, price: float, win_prob: float):
        clean_sym = symbol.split("|")[-1].replace("NSE_EQ", "").strip(":")
        pos_val = qty * price
        direction = "🟢 *BUY (LONG)*" if side == "BUY" else "🔴 *SELL (SHORT)*"
        msg = (
            f"🚀 *NEW TRADE EXECUTED*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• Asset: *{clean_sym}*\n"
            f"• Action: {direction}\n"
            f"• Shares: `{qty}` @ `Rs. {price:.2f}`\n"
            f"• Total Exposure: `Rs. {pos_val:,.2f}`\n"
            f"• AI Win-Prob: *{win_prob:.1f}%* 🎯\n"
            f"• Target: `+1.40%` | Stop Loss: `-0.45%`\n"
            f"━━━━━━━━━━━━━━━━━━━━━━"
        )
        self._send_async(msg)

    def notify_breakeven_locked(self, symbol: str, entry_price: float):
        clean_sym = symbol.split("|")[-1].replace("NSE_EQ", "").strip(":")
        msg = (
            f"🛡️ *RISK-FREE LOCK: {clean_sym}*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• Profit hit `+0.50%` threshold.\n"
            f"• Stop Loss moved to Entry: `Rs. {entry_price:.2f}`\n"
            f"• *Status: Zero Downside Risk!* 🔒"
        )
        self._send_async(msg)

    def notify_trade_exit(
        self,
        symbol: str,
        side: str,
        qty: int,
        entry_price: float,
        exit_price: float,
        gross_pnl: float,
        charges: float,
        net_pnl: float,
        reason: str
    ):
        clean_sym = symbol.split("|")[-1].replace("NSE_EQ", "").strip(":")
        icon = "🟢" if net_pnl >= 0 else "🔴"
        pnl_label = "PROFIT" if net_pnl >= 0 else "LOSS"

        msg = (
            f"{icon} *TRADE CLOSED: {pnl_label}*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• Asset: *{clean_sym}* ({side})\n"
            f"• Entry: `Rs. {entry_price:.2f}` -> Exit: `Rs. {exit_price:.2f}`\n"
            f"• Exit Reason: *{reason}*\n"
            f"• Gross PnL: `Rs. {gross_pnl:+,.2f}`\n"
            f"• Taxes & Brokerage: `-Rs. {charges:,.2f}`\n"
            f"• *NET P&L (IN POCKET):* `{icon} Rs. {net_pnl:+,.2f}`\n"
            f"━━━━━━━━━━━━━━━━━━━━━━"
        )
        self._send_async(msg)

    def notify_circuit_breaker(self, reason: str):
        msg = (
            f"🛑 *CAPITAL PROTECTION CIRCUIT BREAKER*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{reason}\n"
            f"• *Trading halted for today to protect capital.*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━"
        )
        self._send_async(msg)

    def notify_eod_summary(
        self,
        total_trades: int,
        wins: int,
        losses: int,
        gross_pnl: float,
        total_taxes: float,
        net_pnl: float
    ):
        icon = "🟢" if net_pnl >= 0 else "🔴"
        win_rate = (wins / max(1, total_trades)) * 100.0
        msg = (
            f"🏁 *END OF DAY SESSION REPORT*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• Total Trades: `{total_trades}`\n"
            f"• Wins: `{wins}` | Losses: `{losses}`\n"
            f"• Win Rate: `{win_rate:.1f}%`\n"
            f"• Gross PnL: `Rs. {gross_pnl:+,.2f}`\n"
            f"• Total Taxes/Fees Paid: `-Rs. {total_taxes:,.2f}`\n"
            f"• *FINAL NET P&L:* `{icon} Rs. {net_pnl:+,.2f}`\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"_All MIS positions squared off before 15:20 IST._"
        )
        self._send_async(msg)


# Global Singleton
_notifier_instance = None

def get_telegram_notifier() -> TelegramNotifier:
    global _notifier_instance
    if _notifier_instance is None:
        _notifier_instance = TelegramNotifier()
    return _notifier_instance
