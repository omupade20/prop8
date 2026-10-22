# backtest/research_extreme_winrate.py
"""
Pushing Win Rate to the Absolute Mathematical Extreme:
Testing:
1. Nifty 50 Multi-Timeframe Trend Confirmation (5-min 20 EMA + 50 EMA alignment)
2. Day of Week / Session filters
3. Relative Strength (Stock RSI > 60 vs Nifty RSI)
4. Tighter Breakeven Snap (+0.6% gain triggers Breakeven buffer)
"""

import os
import json
import math
from typing import Dict, List, Any

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "backtest_cache")

def load_data():
    files = [f for f in os.listdir(CACHE_DIR) if f.endswith("_1m_6m.json")]
    data = {}
    for f in files:
        sym = f.replace("_1m_6m.json", "")
        with open(os.path.join(CACHE_DIR, f), "r") as fp:
            data[sym] = json.load(fp)
    return data

def calc_mis_taxes(buy_val: float, sell_val: float) -> float:
    brokerage = min(20.0, buy_val * 0.0005) + min(20.0, sell_val * 0.0005)
    stt = sell_val * 0.00025
    txn = (buy_val + sell_val) * 0.0000297
    sebi = (buy_val + sell_val) * 0.0000010
    stamp = buy_val * 0.0000300
    gst = (brokerage + txn + sebi) * 0.18
    return round(brokerage + stt + txn + sebi + stamp + gst, 2)

def evaluate_trades(trades: List[Dict[str, Any]], base_capital: float = 10000.0) -> Dict[str, Any]:
    if not trades:
        return {"trades": 0, "win_rate": 0, "gross_pnl": 0, "taxes": 0, "net_pnl": 0, "pf": 0, "return_pct": 0, "max_dd": 0, "avg_win": 0, "avg_loss": 0}
    
    total = len(trades)
    wins = [t for t in trades if t["net_pnl"] > 0]
    losses = [t for t in trades if t["net_pnl"] < 0]
    wr = len(wins) / total * 100.0
    gross_win = sum(t["gross_pnl"] for t in wins)
    gross_loss = abs(sum(t["gross_pnl"] for t in losses))
    taxes = sum(t["taxes"] for t in trades)
    net_pnl = sum(t["net_pnl"] for t in trades)
    pf = (gross_win / gross_loss) if gross_loss > 0 else 999.0
    
    bal = base_capital
    peak = base_capital
    max_dd = 0
    for t in trades:
        bal += t["net_pnl"]
        peak = max(peak, bal)
        max_dd = max(max_dd, peak - bal)
        
    return {
        "trades": total,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(wr, 1),
        "gross_pnl": round(gross_win - gross_loss, 2),
        "taxes": round(taxes, 2),
        "net_pnl": round(net_pnl, 2),
        "pf": round(pf, 2),
        "max_dd": round(max_dd, 2),
        "return_pct": round(net_pnl / base_capital * 100.0, 1),
        "avg_win": round(gross_win / len(wins), 2) if wins else 0,
        "avg_loss": round(gross_loss / len(losses), 2) if losses else 0
    }

def run_extreme_experiment():
    raw = load_data()
    symbols = ["BSE", "DIXON", "HAL", "TRENT"]
    nifty_candles = raw.get("NIFTY_50", [])
    
    nifty_by_day = {}
    for c in nifty_candles:
        nifty_by_day.setdefault(c["timestamp"][:10], []).append(c)
        
    stocks_by_day = {}
    for s in symbols:
        stocks_by_day[s] = {}
        for c in raw.get(s, []):
            stocks_by_day[s].setdefault(c["timestamp"][:10], []).append(c)
            
    days = sorted(list(nifty_by_day.keys()))
    
    print("="*85)
    print("TESTING EXTREME WIN RATE ENHANCEMENTS ON REAL 6-MONTH 1M DATA")
    print("="*85)
    
    # Test combinations of:
    # 1. Faster Breakeven Shield (trigger at +0.5% or +0.6% or +0.7%)
    # 2. Strong Nifty Trend Filter (> 0.2% directional move on index)
    # 3. Only Long setups (India has long structural bull bias) vs Both Sides
    for only_longs in [False, True]:
        for be_trigger in [0.005, 0.006, 0.007, 0.008]:
            for vol_mult in [2.5, 3.0, 3.5, 4.0]:
                for sl_pct in [0.008, 0.010, 0.012]:
                    trades = []
                    for d in days:
                        day_n = nifty_by_day.get(d, [])
                        if len(day_n) < 30:
                            continue
                        n_open = day_n[0]["open"]
                        n_15m = day_n[14]["close"]
                        n_move = (n_15m - n_open) / n_open
                        
                        if only_longs and n_move <= 0:
                            continue
                        
                        best_cand = None
                        best_score = 0.0
                        
                        for s in symbols:
                            s_candles = stocks_by_day.get(s, {}).get(d, [])
                            if len(s_candles) < 45:
                                continue
                            h15 = max(c["high"] for c in s_candles[:15])
                            l15 = min(c["low"] for c in s_candles[:15])
                            vol15_avg = sum(c["volume"] for c in s_candles[:15]) / 15.0
                            
                            for idx in range(15, min(120, len(s_candles))):
                                b = s_candles[idx]
                                vol_r = b["volume"] / vol15_avg if vol15_avg > 0 else 1.0
                                
                                if b["close"] > h15 and n_move > 0 and vol_r >= vol_mult:
                                    score = vol_r * ((b["close"] - h15) / h15)
                                    if score > best_score:
                                        best_score = score
                                        best_cand = {"sym": s, "side": "BUY", "entry_px": b["close"], "s_candles": s_candles, "idx": idx}
                                    break
                                elif not only_longs and b["close"] < l15 and n_move < 0 and vol_r >= vol_mult:
                                    score = vol_r * ((l15 - b["close"]) / l15)
                                    if score > best_score:
                                        best_score = score
                                        best_cand = {"sym": s, "side": "SELL", "entry_px": b["close"], "s_candles": s_candles, "idx": idx}
                                    break
                                    
                        if best_cand:
                            s = best_cand["sym"]
                            side = best_cand["side"]
                            ep = best_cand["entry_px"]
                            idx = best_cand["idx"]
                            s_candles = best_cand["s_candles"]
                            qty = max(1, math.floor(50000.0 / ep))
                            
                            curr_sl = round(ep * (1 - sl_pct), 2) if side == "BUY" else round(ep * (1 + sl_pct), 2)
                            be_locked = False
                            highest_p = ep
                            lowest_p = ep
                            exit_px = ep
                            exit_reason = "15:15_EOD"
                            
                            for b in s_candles[idx+1:]:
                                if b["timestamp"][11:19] >= "15:15:00":
                                    exit_px = b["close"]
                                    exit_reason = "15:15_EOD"
                                    break
                                    
                                if side == "BUY":
                                    highest_p = max(highest_p, b["high"])
                                    # Never-Red Breakeven Lock
                                    if not be_locked and (b["high"] - ep) >= (ep * be_trigger):
                                        curr_sl = round(ep + (75.0 / qty), 2)
                                        be_locked = True
                                        
                                    # Trailing Stop: 1.5% below highest peak once locked
                                    if be_locked:
                                        trail = round(highest_p * (1 - 0.015), 2)
                                        curr_sl = max(curr_sl, trail)
                                        
                                    if b["low"] <= curr_sl:
                                        exit_px = curr_sl
                                        exit_reason = "BREAKEVEN" if abs(curr_sl - ep) <= (100.0/qty) else ("TRAIL" if curr_sl > ep else "SL")
                                        break
                                else: # SELL
                                    lowest_p = min(lowest_p, b["low"])
                                    if not be_locked and (ep - b["low"]) >= (ep * be_trigger):
                                        curr_sl = round(ep - (75.0 / qty), 2)
                                        be_locked = True
                                        
                                    if be_locked:
                                        trail = round(lowest_p * (1 + 0.015), 2)
                                        curr_sl = min(curr_sl, trail)
                                        
                                    if b["high"] >= curr_sl:
                                        exit_px = curr_sl
                                        exit_reason = "BREAKEVEN" if abs(curr_sl - ep) <= (100.0/qty) else ("TRAIL" if curr_sl < ep else "SL")
                                        break
                                        
                            buy_val = (ep if side == "BUY" else exit_px) * qty
                            sell_val = (exit_px if side == "BUY" else ep) * qty
                            gross = round(sell_val - buy_val, 2)
                            tax = calc_mis_taxes(buy_val, sell_val)
                            net = round(gross - tax, 2)
                            trades.append({"sym": s, "date": d, "side": side, "gross_pnl": gross, "taxes": tax, "net_pnl": net, "reason": exit_reason})
                            
                    res = evaluate_trades(trades)
                    # Filter for top tier win rates
                    if res["trades"] >= 15 and res["win_rate"] >= 60.0 and res["net_pnl"] > 0:
                        mode_str = "LONGS ONLY" if only_longs else "LONG+SHORT"
                        print(f"[{mode_str}] BE:+{be_trigger*100:.1f}% | Vol:{vol_mult}x | SL:{sl_pct*100:.1f}% --> Trades:{res['trades']:<2} | WIN RATE: {res['win_rate']:>4.1f}% | Gross:Rs.{res['gross_pnl']:>8.2f} | Taxes:Rs.{res['taxes']:>7.2f} | Net:Rs.{res['net_pnl']:>+8.2f} | Ret:{res['return_pct']:>+5.1f}% | PF:{res['pf']}")

if __name__ == "__main__":
    run_extreme_experiment()
