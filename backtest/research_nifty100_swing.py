# backtest/research_nifty100_swing.py
"""
Multi-Day Swing Momentum on the Nifty 100 Universe.
Capital: Rs. 10,000 cash (Delivery / CNC).
No 15:15 intraday square-off force.
Holding 3 to 7 days on 20-day high breakouts.
"""

import os
import json
import math
from typing import Dict, List, Any

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "backtest_cache")

def load_universe():
    files = [f for f in os.listdir(CACHE_DIR) if f.endswith("_1m_6m.json") and not f.startswith("NIFTY_50")]
    universe = {}
    for f in files:
        sym = f.replace("_1m_6m.json", "")
        with open(os.path.join(CACHE_DIR, f), "r") as fp:
            universe[sym] = json.load(fp)
    return universe

def calc_cnc_taxes(buy_val: float, sell_val: float) -> float:
    brokerage = min(20.0, buy_val * 0.0005) + min(20.0, sell_val * 0.0005)
    stt = (buy_val + sell_val) * 0.0010
    txn = (buy_val + sell_val) * 0.0000297
    sebi = (buy_val + sell_val) * 0.0000010
    stamp = buy_val * 0.0001500
    gst = (brokerage + txn + sebi) * 0.18
    return round(brokerage + stt + txn + sebi + stamp + gst, 2)

def evaluate_trades(trades: List[Dict[str, Any]], base_capital: float = 10000.0) -> Dict[str, Any]:
    if not trades:
        return {"trades": 0, "win_rate": 0, "gross_pnl": 0, "taxes": 0, "net_pnl": 0, "pf": 0}
    total = len(trades)
    wins = [t for t in trades if t["net_pnl"] > 0]
    losses = [t for t in trades if t["net_pnl"] < 0]
    wr = len(wins) / total * 100.0
    gross_win = sum(t["gross_pnl"] for t in wins)
    gross_loss = abs(sum(t["gross_pnl"] for t in losses))
    taxes = sum(t["taxes"] for t in trades)
    net_pnl = sum(t["net_pnl"] for t in trades)
    pf = (gross_win / gross_loss) if gross_loss > 0 else 999.0
    return {
        "trades": total,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(wr, 1),
        "gross_pnl": round(gross_win - gross_loss, 2),
        "taxes": round(taxes, 2),
        "net_pnl": round(net_pnl, 2),
        "pf": round(pf, 2),
        "return_pct": round(net_pnl / base_capital * 100.0, 1),
        "avg_win": round(gross_win / len(wins), 2) if wins else 0,
        "avg_loss": round(gross_loss / len(losses), 2) if losses else 0
    }

def main():
    stocks = load_universe()
    symbols = list(stocks.keys())
    
    # Compile daily candles
    daily: Dict[str, List[Dict[str, Any]]] = {}
    for s in symbols:
        by_day = {}
        for c in stocks[s]:
            d = c["timestamp"][:10]
            by_day.setdefault(d, []).append(c)
        d_bars = []
        for d in sorted(by_day.keys()):
            cs = by_day[d]
            d_bars.append({
                "date": d,
                "open": cs[0]["open"],
                "high": max(x["high"] for x in cs),
                "low": min(x["low"] for x in cs),
                "close": cs[-1]["close"],
                "volume": sum(x["volume"] for x in cs)
            })
        daily[s] = d_bars
        
    all_dates = sorted(list(set(b["date"] for s in symbols for b in daily[s])))
    print(f"Loaded {len(symbols)} Nifty 100 instruments over {len(all_dates)} days.")
    
    print("\n" + "="*85)
    print("NIFTY 100 SWING BREAKOUT (Stage 2 Momentum + Trailing Low)")
    print("Capital: Rs. 10,000 cash | Taxes: Delivery CNC (STT 0.1%)")
    print("="*85)
    
    for lookback in [15, 20]:
        for sl_pct in [0.035, 0.045]:
            for trail_days in [3, 5]:
                active_pos = None
                trades = []
                
                for i, cur_date in enumerate(all_dates):
                    if active_pos:
                        s = active_pos["sym"]
                        d_bar = next((b for b in daily[s] if b["date"] == cur_date), None)
                        if d_bar:
                            s_history = [b for b in daily[s] if b["date"] <= cur_date]
                            if len(s_history) > trail_days:
                                recent_lows = [b["low"] for b in s_history[-trail_days-1:-1]]
                                active_pos["sl"] = max(active_pos["sl"], min(recent_lows))
                                
                            if d_bar["low"] <= active_pos["sl"]:
                                exit_px = active_pos["sl"]
                                buy_val = active_pos["entry_px"] * active_pos["qty"]
                                sell_val = exit_px * active_pos["qty"]
                                gross = round(sell_val - buy_val, 2)
                                tax = calc_cnc_taxes(buy_val, sell_val)
                                trades.append({"sym": s, "date": cur_date, "gross_pnl": gross, "taxes": tax, "net_pnl": round(gross - tax, 2)})
                                active_pos = None
                                
                    if not active_pos and i >= lookback:
                        best_candidate = None
                        best_vol_ratio = 0.0
                        
                        for s in symbols:
                            s_history = [b for b in daily[s] if b["date"] <= cur_date]
                            if len(s_history) < lookback + 1:
                                continue
                            today = s_history[-1]
                            prev_window = s_history[-lookback-1:-1]
                            prior_high = max(b["high"] for b in prev_window)
                            avg_vol = sum(b["volume"] for b in prev_window) / len(prev_window)
                            
                            if today["close"] > prior_high and avg_vol > 0:
                                vol_r = today["volume"] / avg_vol
                                if vol_r >= 1.3 and vol_r > best_vol_ratio:
                                    best_vol_ratio = vol_r
                                    best_candidate = (s, today["close"])
                                    
                        if best_candidate:
                            s, ep = best_candidate
                            qty = max(1, math.floor(10000.0 / ep))
                            active_pos = {
                                "sym": s, "entry_px": ep, "qty": qty, "sl": round(ep * (1 - sl_pct), 2)
                            }
                            
                res = evaluate_trades(trades)
                if res["trades"] >= 5:
                    status = "[PROFITABLE]" if res["net_pnl"] > 0 else "[LOSS]"
                    print(f"{status} Lookback:{lookback}d | SL:{sl_pct*100:.1f}% | Trail:{trail_days}d --> Trades:{res['trades']:<2} | WR:{res['win_rate']:>4.1f}% | Gross:Rs.{res['gross_pnl']:>8.2f} | Taxes:Rs.{res['taxes']:>6.2f} | Net Take-Home:Rs.{res['net_pnl']:>+8.2f} | Ret:{res['return_pct']:>+5.1f}% | PF:{res['pf']}")

if __name__ == "__main__":
    main()
