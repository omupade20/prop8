# backtest/run_nifty100_audit.py
"""
Comprehensive 6-Month Backtest on Nifty 100 Universe.
Tests Diamond High-Beta Ultra-Sniper with 100% Real 1-Minute Tick Data and Exact Indian Taxes.
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

def load_index():
    path = os.path.join(CACHE_DIR, "NIFTY_50_1m_6m.json")
    if os.path.exists(path):
        with open(path, "r") as fp:
            return json.load(fp)
    return []

def calc_mis_taxes(buy_val: float, sell_val: float) -> float:
    brokerage = min(20.0, buy_val * 0.0005) + min(20.0, sell_val * 0.0005)
    stt = sell_val * 0.00025
    txn = (buy_val + sell_val) * 0.0000297
    sebi = (buy_val + sell_val) * 0.0000010
    stamp = buy_val * 0.0000300
    gst = (brokerage + txn + sebi) * 0.18
    return round(brokerage + stt + txn + sebi + stamp + gst, 2)

def run_nifty100_simulation(base_capital: float = 10000.0, max_trades_per_day: int = 1):
    stocks = load_universe()
    index_candles = load_index()
    symbols = list(stocks.keys())
    
    print(f"[INGESTION] Loaded {len(symbols)} Nifty 100 instruments from cache.")
    
    # 1. Volatility Analysis
    stock_volatility = {}
    for sym in symbols:
        c_list = stocks[sym]
        by_d = {}
        for c in c_list:
            by_d.setdefault(c["timestamp"][:10], []).append(c)
        r_list = []
        for d, cs in by_d.items():
            dh = max(x["high"] for x in cs)
            dl = min(x["low"] for x in cs)
            dop = cs[0]["open"]
            if dop > 0:
                r_list.append((dh - dl) / dop * 100.0)
        avg_r = sum(r_list) / len(r_list) if r_list else 0.0
        stock_volatility[sym] = round(avg_r, 2)

    # Sort and rank by volatility
    sorted_by_vol = sorted(stock_volatility.items(), key=lambda x: x[1], reverse=True)
    print("\n--- TOP 10 HIGHEST VOLATILITY STOCKS IN NIFTY 100 UNIVERSE ---")
    for s, v in sorted_by_vol[:10]:
        print(f"  {s:<14}: Avg Daily Range = {v:.2f}%")
        
    print("\n--- BOTTOM 5 LOWEST VOLATILITY STOCKS ---")
    for s, v in sorted_by_vol[-5:]:
        print(f"  {s:<14}: Avg Daily Range = {v:.2f}%")

    # 2. Pre-index by day
    nifty_by_day = {}
    for c in index_candles:
        nifty_by_day.setdefault(c["timestamp"][:10], []).append(c)
        
    stocks_by_day = {}
    for s in symbols:
        stocks_by_day[s] = {}
        for c in stocks[s]:
            stocks_by_day[s].setdefault(c["timestamp"][:10], []).append(c)

    days = sorted(list(nifty_by_day.keys()))
    print(f"\n[UNIVERSE READY] Total Trading Sessions: {len(days)} days (April - October 2026).")
    
    executed_trades = []
    
    # 3. Simulate Day by Day
    for d in days:
        day_nifty = nifty_by_day.get(d, [])
        if len(day_nifty) < 30:
            continue
        n_open = day_nifty[0]["open"]
        n_15m = day_nifty[14]["close"] if len(day_nifty) >= 15 else n_open
        nifty_dir = "BULL" if n_15m > n_open else "BEAR"
        
        # Find the single best setup across the whole Nifty 100 universe today
        best_cand = None
        best_score = 0.0
        
        for s in symbols:
            s_candles = stocks_by_day[s].get(d, [])
            if len(s_candles) < 45:
                continue
                
            h15 = max(c["high"] for c in s_candles[:15])
            l15 = min(c["low"] for c in s_candles[:15])
            vol15_avg = sum(c["volume"] for c in s_candles[:15]) / 15.0
            
            # Avoid doji bars
            c15_open = s_candles[0]["open"]
            c15_close = s_candles[14]["close"]
            body_size = abs(c15_close - c15_open)
            full_range = h15 - l15
            if full_range <= 0 or (body_size / full_range) < 0.35:
                continue
                
            for idx in range(15, min(120, len(s_candles))):
                b = s_candles[idx]
                vol_ratio = b["volume"] / vol15_avg if vol15_avg > 0 else 1.0
                
                # Check 4.0x volume surge with Nifty alignment
                if b["close"] > h15 and nifty_dir == "BULL" and vol_ratio >= 4.0:
                    score = vol_ratio * ((b["close"] - h15) / h15) * stock_volatility.get(s, 1.0)
                    if score > best_score:
                        best_score = score
                        best_cand = {"sym": s, "side": "BUY", "entry_px": b["close"], "bar_idx": idx, "s_candles": s_candles}
                    break
                elif b["close"] < l15 and nifty_dir == "BEAR" and vol_ratio >= 4.0:
                    score = vol_ratio * ((l15 - b["close"]) / l15) * stock_volatility.get(s, 1.0)
                    if score > best_score:
                        best_score = score
                        best_cand = {"sym": s, "side": "SELL", "entry_px": b["close"], "bar_idx": idx, "s_candles": s_candles}
                    break

        if best_cand:
            s = best_cand["sym"]
            side = best_cand["side"]
            ep = best_cand["entry_px"]
            idx = best_cand["bar_idx"]
            s_candles = best_cand["s_candles"]
            
            qty = max(1, math.floor(50000.0 / ep)) # Rs 10,000 * 5x = Rs 50k
            
            current_sl = round(ep * (1 - 0.012), 2) if side == "BUY" else round(ep * (1 + 0.012), 2)
            be_threshold = ep * (1 + 0.008) if side == "BUY" else ep * (1 - 0.008)
            is_trailing = False
            best_fav_px = ep
            exit_px = ep
            exit_reason = "15:15_EOD"
            
            for b in s_candles[idx+1:]:
                if b["timestamp"][11:19] >= "15:15:00":
                    exit_px = b["close"]
                    exit_reason = "15:15_EOD"
                    break
                    
                if side == "BUY":
                    best_fav_px = max(best_fav_px, b["high"])
                    # Never-Red Breakeven Lock (+0.8%)
                    if not is_trailing and b["high"] >= be_threshold:
                        is_trailing = True
                        current_sl = round(ep + (75.0 / qty), 2)
                        
                    if is_trailing:
                        trail_sl = round(best_fav_px * (1 - 0.015), 2)
                        current_sl = max(current_sl, trail_sl)
                        
                    if b["low"] <= current_sl:
                        exit_px = current_sl
                        exit_reason = "BREAKEVEN" if abs(current_sl - ep) <= (100.0/qty) else ("TRAIL" if current_sl > ep else "SL")
                        break
                else: # SELL
                    best_fav_px = min(best_fav_px, b["low"])
                    if not is_trailing and b["low"] <= be_threshold:
                        is_trailing = True
                        current_sl = round(ep - (75.0 / qty), 2)
                        
                    if is_trailing:
                        trail_sl = round(best_fav_px * (1 + 0.015), 2)
                        current_sl = min(current_sl, trail_sl)
                        
                    if b["high"] >= current_sl:
                        exit_px = current_sl
                        exit_reason = "BREAKEVEN" if abs(current_sl - ep) <= (100.0/qty) else ("TRAIL" if current_sl < ep else "SL")
                        break
                        
            buy_val = (ep if side == "BUY" else exit_px) * qty
            sell_val = (exit_px if side == "BUY" else ep) * qty
            gross = round(sell_val - buy_val, 2)
            tax = calc_mis_taxes(buy_val, sell_val)
            net = round(gross - tax, 2)
            
            executed_trades.append({
                "date": d, "sym": s, "side": side, "entry_px": ep, "exit_px": exit_px, "qty": qty,
                "gross_pnl": gross, "taxes": tax, "net_pnl": net, "reason": exit_reason
            })

    # 4. Performance Audit
    total_trades = len(executed_trades)
    wins = [t for t in executed_trades if t["net_pnl"] > 0]
    losses = [t for t in executed_trades if t["net_pnl"] < 0]
    wr = len(wins) / total_trades * 100.0 if total_trades > 0 else 0
    gross_win = sum(t["gross_pnl"] for t in wins)
    gross_loss = abs(sum(t["gross_pnl"] for t in losses))
    taxes = sum(t["taxes"] for t in executed_trades)
    net_pnl = sum(t["net_pnl"] for t in executed_trades)
    pf = (gross_win / gross_loss) if gross_loss > 0 else 999.0
    
    print("\n" + "="*85)
    print("           NIFTY 100 AUDITED 6-MONTH MASTER PERFORMANCE SUMMARY             ")
    print("="*85)
    print(f"• Initial Cash Capital:       Rs. {base_capital:,.2f}")
    print(f"• Final Account Balance:      Rs. {base_capital + net_pnl:,.2f}")
    print(f"• Total Net Take-Home PnL:    Rs. {net_pnl:+,.2f}  (Return: {(net_pnl/base_capital)*100:+.1f}%)")
    print(f"• Total Trades Executed:      {total_trades} ({round(total_trades/6.0, 1)} trades/month)")
    print(f"• Win Rate:                   {wr:.1f}%  ({len(wins)} Wins / {len(losses)} Losses)")
    print(f"• Profit Factor:              {pf:.2f}")
    print(f"• Total Statutory Taxes Paid: Rs. {taxes:,.2f} (Full Upstox Brokerage + STT + GST + Fees)")
    print(f"• Average Win:                Rs. {gross_win/len(wins):,.2f}" if wins else "")
    print(f"• Average Loss:               Rs. {gross_loss/len(losses):,.2f}" if losses else "")
    print("="*85)
    
    # Monthly breakdown
    by_m = {}
    for t in executed_trades:
        m = t["date"][:7]
        by_m.setdefault(m, {"trades": 0, "wins": 0, "gross": 0.0, "tax": 0.0, "net": 0.0})
        by_m[m]["trades"] += 1
        if t["net_pnl"] > 0:
            by_m[m]["wins"] += 1
        by_m[m]["gross"] += t["gross_pnl"]
        by_m[m]["tax"] += t["taxes"]
        by_m[m]["net"] += t["net_pnl"]
        
    print("\n" + "="*85)
    print("                        MONTH-BY-MONTH AUDIT TABLE                          ")
    print("="*85)
    print(f"{'Month':<10} | {'Trades':<7} | {'Win Rate':<9} | {'Gross PnL':<12} | {'Taxes':<10} | {'Net PnL':<12} | {'Return':<8}")
    print("-"*85)
    for m in sorted(by_m.keys()):
        st = by_m[m]
        w_r = st["wins"] / st["trades"] * 100.0 if st["trades"] > 0 else 0
        ret_m = st["net"] / base_capital * 100.0
        print(f"{m:<10} | {st['trades']:<7} | {w_r:>7.1f}% | Rs. {st['gross']:>9.2f} | Rs. {st['tax']:>7.2f} | Rs. {st['net']:>9.2f} | {ret_m:>+6.1f}%")

    # Stock breakdown
    by_sym = {}
    for t in executed_trades:
        s = t["sym"]
        by_sym.setdefault(s, {"trades": 0, "wins": 0, "net": 0.0})
        by_sym[s]["trades"] += 1
        if t["net_pnl"] > 0:
            by_sym[s]["wins"] += 1
        by_sym[s]["net"] += t["net_pnl"]
        
    print("\n" + "="*85)
    print("                     MOST TRADED NIFTY 100 CONSTITUENTS                     ")
    print("="*85)
    for s, st in sorted(by_sym.items(), key=lambda x: x[1]["net"], reverse=True):
        print(f"  {s:<14}: {st['trades']} trades | {st['wins']} wins | Net PnL: Rs. {st['net']:>+8.2f}")

if __name__ == "__main__":
    run_nifty100_simulation()
