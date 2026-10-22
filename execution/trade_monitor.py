# execution/trade_monitor.py
"""
Trade Monitor for active intraday positions.
Tracks Stop Loss, Profit Targets, Breakeven adjustments, and Trailing partial exits.
Supports:
- Real-time updates from PortfolioStreamer
- Startup reconciliation from PortfolioApi
- Dual attribute compatibility (.trades and .active_trades, .entry_time and .open_time)
"""

from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from execution.execution_config import (
    STOP_LOSS_PCT,
    TARGET_PCT,
    BREAKEVEN_MOVE_PCT,
    PARTIAL_EXIT_MOVE_PCT,
    PARTIAL_EXIT_LIMIT_PCT,
    ENABLE_ATR_BASED_TARGET_SL,
    ATR_TARGET_MULTIPLIER,
    ATR_SL_MULTIPLIER
)


class TrackedTrade:
    """
    Holds state for a live trade with dynamic ATR volatility boundaries.
    """

    def __init__(self, inst_key: str, side: str, entry_price: float, qty: int, trade_id: Optional[str] = None, atr: Optional[float] = None):
        self.trade_id = trade_id
        self.inst_key = inst_key
        self.side = side.upper()  # "BUY" or "SELL"
        self.entry_price = float(entry_price)
        self.qty = int(qty)
        self.atr = float(atr) if atr and atr > 0 else None

        self.stop_loss = self.calc_stop_loss(entry_price, self.side)
        self.initial_sl = self.stop_loss
        self.initial_risk = max(abs(self.entry_price - self.stop_loss), self.entry_price * 0.003)
        self.target = self.calc_target(entry_price, self.side)
        self.breakeven_moved = False
        self.is_never_red_locked = False
        self.highest_r_multiple = 0.0
        self.partial_exit_done = False
        self.is_closed = False
        self.status = "OPEN"  # OPEN, EXIT_PENDING, CLOSED
        self.gtt_id: Optional[str] = None

        self.open_time = datetime.now()
        self.entry_time = self.open_time  # Alias for backward compatibility

    def calc_stop_loss(self, entry_price: float, side: str) -> float:
        if ENABLE_ATR_BASED_TARGET_SL and self.atr:
            sl_dist = max(self.atr * ATR_SL_MULTIPLIER, entry_price * 0.0030)
            return round(entry_price - sl_dist if side == "BUY" else entry_price + sl_dist, 2)

        if side == "BUY":
            return round(entry_price * (1 - STOP_LOSS_PCT), 2)
        else:
            return round(entry_price * (1 + STOP_LOSS_PCT), 2)

    def calc_target(self, entry_price: float, side: str) -> float:
        if ENABLE_ATR_BASED_TARGET_SL and self.atr:
            target_dist = max(self.atr * ATR_TARGET_MULTIPLIER, entry_price * 0.0090)
            return round(entry_price + target_dist if side == "BUY" else entry_price - target_dist, 2)

        if side == "BUY":
            return round(entry_price * (1 + TARGET_PCT), 2)
        else:
            return round(entry_price * (1 - TARGET_PCT), 2)

    def get_current_profit_pct(self, current_price: float) -> float:
        if self.entry_price <= 0:
            return 0.0
        if self.side == "BUY":
            return (current_price - self.entry_price) / self.entry_price
        return (self.entry_price - current_price) / self.entry_price

    def get_r_multiple(self, current_price: float) -> float:
        if not hasattr(self, "initial_risk") or self.initial_risk <= 0:
            return 0.0
        if self.side == "BUY":
            return (current_price - self.entry_price) / self.initial_risk
        return (self.entry_price - current_price) / self.initial_risk


class TradeMonitor:
    """
    Monitors live trades and triggers exit logic.
    Persists active state to SQLite database for instant crash recovery.
    """

    def __init__(self, state_store=None):
        self.active_trades: Dict[str, TrackedTrade] = {}
        try:
            from core.state_store import StateStore
            self.state_store = state_store or StateStore()
            self._restore_from_db()
        except Exception as e:
            print(f"[TradeMonitor] Notice: StateStore offline: {e}")
            self.state_store = None

    def _restore_from_db(self):
        if not self.state_store:
            return
        saved = self.state_store.load_active_trades()
        for s in saved:
            t = TrackedTrade(
                inst_key=s["inst_key"],
                side=s["side"],
                entry_price=s["entry_price"],
                qty=s["qty"],
                trade_id=s["trade_id"]
            )
            t.stop_loss = s["stop_loss"]
            t.target = s["target"]
            t.breakeven_moved = bool(s.get("breakeven_moved", 0))
            t.gtt_id = s.get("gtt_id")
            t.status = s.get("status", "OPEN")
            self.active_trades[s["trade_id"]] = t
        if saved:
            print(f"[RESTORE] [TradeMonitor] Restored {len(saved)} active positions from database.")

    @property
    def trades(self) -> Dict[str, TrackedTrade]:
        """Backward-compatible property mapping to active_trades."""
        return self.active_trades

    def add_trade(self, trade_id: str, inst_key: str, side: str, entry_price: float, qty: int, atr: Optional[float] = None) -> TrackedTrade:
        trade_key = str(trade_id) if trade_id else f"{inst_key}_{datetime.now().strftime('%H%M%S')}"
        trade = TrackedTrade(
            inst_key=inst_key,
            side=side,
            entry_price=entry_price,
            qty=qty,
            trade_id=trade_key,
            atr=atr
        )
        self.active_trades[trade_key] = trade
        if self.state_store:
            self.state_store.save_active_trade(trade)
        print(f"[TradeMonitor] Added trade {trade_key}: {side} {qty} {inst_key} @ {entry_price:.2f} (SL: {trade.stop_loss:.2f}, TP: {trade.target:.2f})")
        return trade

    def remove_trade(self, trade_id: str):
        if trade_id in self.active_trades:
            del self.active_trades[trade_id]
        if self.state_store:
            self.state_store.remove_active_trade(trade_id)

    def reconcile_broker_positions(self, live_positions: List[Dict[str, Any]]):
        """
        Reconcile with open positions from Upstox PortfolioApi on system startup.
        """
        for pos in live_positions:
            token = pos.get("instrument_token")
            qty = pos.get("quantity", 0)
            side = pos.get("side", "BUY")
            avg_price = pos.get("average_price", 0.0)

            # Check if this position is already tracked
            already_tracked = any(t.inst_key == token and not t.is_closed for t in self.active_trades.values())
            if not already_tracked and qty > 0 and avg_price > 0:
                trade_id = f"RECON_{token}_{side}"
                self.add_trade(
                    trade_id=trade_id,
                    inst_key=token,
                    side=side,
                    entry_price=avg_price,
                    qty=qty
                )
                print(f"[TradeMonitor] Reconciled existing broker position: {side} {qty} of {token} @ {avg_price:.2f}")

    def on_order_update(self, order_event: Dict[str, Any]):
        """
        Callback from PortfolioStreamer when an order updates on Upstox.
        """
        order_id = str(order_event.get("order_id") or "")
        status = str(order_event.get("status", "")).upper()

        if not order_id:
            return

        if order_id in self.active_trades:
            trade = self.active_trades[order_id]
            if status in ("CANCELLED", "REJECTED"):
                print(f"[TradeMonitor] Order {order_id} was {status} at broker. Removing trade.")
                self.remove_trade(order_id)
            elif status == "COMPLETE":
                trade.status = "OPEN"

    def check_trades(self, current_prices: Dict[str, float]) -> List[Tuple[str, str, float]]:
        """
        Check each active trade against exit conditions.
        Returns list of tuples: [(trade_id, exit_reason, exit_price)]
        """
        exits = []

        for trade_id, trade in list(self.active_trades.items()):
            if trade.is_closed:
                continue

            ltp = current_prices.get(trade.inst_key)
            if ltp is None or ltp <= 0:
                continue

            profit_pct = trade.get_current_profit_pct(ltp)

            # 0) AI REINFORCEMENT LEARNING EXIT POLICY
            try:
                from strategy.ai_rl_policy import AIRLExitPolicy
                if not hasattr(self, "_rl_policy"):
                    self._rl_policy = AIRLExitPolicy()

                rl_action = self._rl_policy.evaluate_exit_policy(
                    current_profit_pct=profit_pct,
                    entry_price=trade.entry_price,
                    current_price=ltp,
                    side=trade.side
                )

                if rl_action["action"] == "HARVEST_PROFIT_NOW":
                    exits.append((trade_id, "AI_RL_HARVEST", ltp))
                    trade.is_closed = True
                    continue
                elif rl_action["action"] == "EXTEND_TARGET_RUNNER" and not getattr(trade, "runner_extended", False):
                    trade.target = rl_action["new_target"]
                    trade.stop_loss = rl_action["new_sl"]
                    trade.runner_extended = True
                    if self.state_store:
                        self.state_store.save_active_trade(trade)
                    print(f"[TradeMonitor] [RL RUNNER] Extended Target to {trade.target:.2f} (Locked SL at {trade.stop_loss:.2f}) for {trade_id}")
                elif rl_action["action"] == "EARLY_BREAKEVEN_LOCK" and not trade.breakeven_moved:
                    trade.stop_loss = trade.entry_price
                    trade.breakeven_moved = True
                    if self.state_store:
                        self.state_store.save_active_trade(trade)
                    print(f"[TradeMonitor] [RL BREAKEVEN] Early Risk-Free Lock ({trade.entry_price:.2f}) for {trade_id}")
            except Exception:
                pass

            # 1) STOP LOSS

            if trade.side == "BUY" and ltp <= trade.stop_loss:
                exits.append((trade_id, "STOP_LOSS", ltp))
                trade.is_closed = True
                continue
            if trade.side == "SELL" and ltp >= trade.stop_loss:
                exits.append((trade_id, "STOP_LOSS", ltp))
                trade.is_closed = True
                continue

            # 2) TARGET
            if trade.side == "BUY" and ltp >= trade.target:
                exits.append((trade_id, "TARGET", ltp))
                trade.is_closed = True
                continue
            if trade.side == "SELL" and ltp <= trade.target:
                exits.append((trade_id, "TARGET", ltp))
                trade.is_closed = True
                continue

            # 3) 🛡️ NEVER-RED R-MULTIPLE PROFIT LOCK & BREAKEVEN STEP
            r_mult = trade.get_r_multiple(ltp)
            trade.highest_r_multiple = max(trade.highest_r_multiple, r_mult)

            # A. Touching +1.0R (or BREAKEVEN_MOVE_PCT): Instantly lock SL to Entry + Brokerage Buffer
            if not trade.is_never_red_locked and (r_mult >= 1.0 or profit_pct >= BREAKEVEN_MOVE_PCT):
                # Full statutory coverage: Brokerage (Rs. 40) + STT + GST + Stamp Duty (approx Rs. 75 round-trip)
                buffer = round(75.0 / max(trade.qty, 1), 2)
                buffer = min(buffer, trade.entry_price * 0.0015)  # Cap at 0.15%
                if trade.side == "BUY":
                    trade.stop_loss = round(trade.entry_price + buffer, 2)
                else:
                    trade.stop_loss = round(trade.entry_price - buffer, 2)
                trade.breakeven_moved = True
                trade.is_never_red_locked = True
                if self.state_store:
                    self.state_store.save_active_trade(trade)
                print(f"[NEVER-RED BREAKEVEN] Trade {trade_id} touched +1.0R (+{profit_pct*100:.2f}%)! Locked SL at Rs. {trade.stop_loss:.2f} (Entry + buffer). Risk is permanently ZERO.")

            # B. Touching +1.5R: Dynamically lock in at least +0.5R profit (Trailing Profit Lock)
            if r_mult >= 1.5:
                if trade.side == "BUY":
                    lock_sl = round(trade.entry_price + (0.5 * trade.initial_risk), 2)
                    if lock_sl > trade.stop_loss:
                        trade.stop_loss = lock_sl
                        if self.state_store:
                            self.state_store.save_active_trade(trade)
                        print(f"[PROFIT LOCK +0.5R] Trade {trade_id} reached +1.5R! Trailing SL locked at Rs. {trade.stop_loss:.2f} (+0.5R guaranteed profit).")
                else:
                    lock_sl = round(trade.entry_price - (0.5 * trade.initial_risk), 2)
                    if lock_sl < trade.stop_loss:
                        trade.stop_loss = lock_sl
                        if self.state_store:
                            self.state_store.save_active_trade(trade)
                        print(f"[PROFIT LOCK +0.5R] Trade {trade_id} reached +1.5R! Trailing SL locked at Rs. {trade.stop_loss:.2f} (+0.5R guaranteed profit).")

            # 4) PARTIAL EXIT (Reached +0.70%, fails back below threshold)
            if not trade.partial_exit_done:
                if profit_pct >= PARTIAL_EXIT_MOVE_PCT:
                    if trade.side == "BUY" and ltp <= trade.entry_price * (1 + PARTIAL_EXIT_LIMIT_PCT):
                        exits.append((trade_id, "PARTIAL_EXIT", ltp))
                        trade.is_closed = True
                        continue
                    if trade.side == "SELL" and ltp >= trade.entry_price * (1 - PARTIAL_EXIT_LIMIT_PCT):
                        exits.append((trade_id, "PARTIAL_EXIT", ltp))
                        trade.is_closed = True
                        continue

        return exits
