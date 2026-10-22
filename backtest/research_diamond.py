# backtest/research_diamond.py
"""
Deep Quantitative Laboratory: Finding the Profitable Diamond Strategy
Testing:
1. Wider trailing buffers (0.8% to 1.5%) to let runners reach 2.5% - 4.5% gains
2. Stricter volume surge (3.0x to 5.0x) to reduce statutory tax count to 25-40 trades
3. Strong 15m candle body filter (conviction breakout)
4. Comparison against High-Beta Multi-Day Swing
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

def run_diamond_simulation(
    data: Dict[str, List[Dict[str, Any]]],
    min_vol_mult: float,
    sl_pct: float,
    be_activation_pct: float,
    trail_dist_pct: float,
    symbols: List[str] = ["BSE", "DIXON", "HAL", "TRENT"]
):
    nifty_candles = data.get("NIFTY_50", [])
    
    # Pre-group by day
    nifty_by_day = {}
    for c in nifty_candles:
        nifty_by_day.setdefault(c["timestamp"][:10], []).append(c)
        
    stocks_by_day = {}
    for s in symbols:
        stocks_by_day[s] = {}
        for c in data.get(s, []):
            stocks_by_day[s].setdefault(c["timestamp"][:10], []).append(c)
            
    days = sorted(list(nifty_by_day.keys()))
    trades = []
    
    for d in days:
        day_nifty = nifty_by_day.get(d, [])
        if len(day_nifty) < 30:
            continue
        n_open = day_nifty[0]["open"]
        n_15m = day_nifty[14]["close"] if len(day_nifty) >= 15 else n_open
        nifty_dir = "BULL" if n_15m > n_open else "BEAR"
        
        best_cand = None
        best_score = 0.0
        
        for s in symbols:
            s_candles = stocks_by_day.get(s, {}).get(d, [])
            if len(s_candles) < 45:
                continue
                
            # First 15 min stats
            h15 = max(c["high"] for c in s_candles[:15])
            l15 = min(c["low"] for c in s_candles[:15])
            vol15_avg = sum(c["volume"] for c in s_candles[:15]) / 15.0
            
            # Check 15m candle structure (conviction body)
            c15_open = s_candles[0]["open"]
            c15_close = s_candles[14]["close"]
            body_size = abs(c15_close - c15_open)
            full_range = h15 - l15
            if full_range <= 0 or (body_size / full_range) < 0.35: # avoid pure doji churn
                continue
            
            # Scan from 09:30 to 11:15
            for idx in range(15, min(120, len(s_candles))):
                b = s_candles[idx]
                vol_ratio = b["volume"] / vol15_avg if vol15_avg > 0 else 1.0
                
                # Long Condition
                if b["close"] > h15 and nifty_dir == "BULL" and vol_ratio >= min_vol_mult:
                    # Breakout strength score
                    score = vol_ratio * ((b["close"] - h15) / h15)
                    if score > best_score:
                        best_score = score
                        best_cand = {
                            "sym": s, "side": "BUY", "entry_px": b["close"], "bar_idx": idx, "s_candles": s_candles
                        }
                    break
                # Short Condition
                elif b["close"] < l15 and nifty_dir == "BEAR" and vol_ratio >= min_vol_mult:
                    score = vol_ratio * ((l15 - b["close"]) / l15)
                    if score > best_score:
                        best_score = score
                        best_cand = {
                            "sym": s, "side": "SELL", "entry_px": b["close"], "bar_idx": idx, "s_candles": s_candles
                        }
                    break
                    
        if best_cand:
            s = best_cand["sym"]
            side = best_cand["side"]
            ep = best_cand["entry_px"]
            idx = best_cand["bar_idx"]
            s_candles = best_cand["s_candles"]
            
            qty = max(1, math.floor(50000.0 / ep))
            
            current_sl = round(ep * (1 - sl_pct), 2) if side == "BUY" else round(ep * (1 + sl_pct), 2)
            be_threshold = ep * (1 + be_activation_pct) if side == "BUY" else ep * (1 - be_activation_pct)
            is_trailing = False
            best_favorable_px = ep
            
            exit_px = ep
            exit_reason = "15:15_EOD"
            
            for b in s_candles[idx+1:]:
                t_str = b["timestamp"][11:19]
                if t_str >= "15:15:00":
                    exit_px = b["close"]
                    exit_reason = "15:15_EOD"
                    break
                    
                if side == "BUY":
                    best_favorable_px = max(best_favorable_px, b["high"])
                    
                    # 1. Activate Never-Red Breakeven
                    if not is_trailing and b["high"] >= be_threshold:
                        is_trailing = True
                        current_sl = round(ep + (75.0 / qty), 2)
                        
                    # 2. Dynamic Wide Trailing Stop
                    if is_trailing:
                        trail_sl = round(best_favorable_px * (1 - trail_dist_pct), 2)
                        current_sl = max(current_sl, trail_sl)
                        
                    # 3. Check SL hit
                    if b["low"] <= current_sl:
                        exit_px = current_sl
                        exit_reason = "TRAIL_STOP" if is_trailing else "STOP_LOSS"
                        break
                        
                else: # SELL
                    best_favorable_px = min(best_favorable_px, b["low"])
                    
                    if not is_trailing and b["low"] <= be_threshold:
                        is_trailing = True
                        current_sl = round(ep - (75.0 / qty), 2)
                        
                    if is_trailing:
                        trail_sl = round(best_favorable_px * (1 + trail_dist_pct), 2)
                        current_sl = min(current_sl, trail_sl)
                        
                    if b["high"] >= current_sl:
                        exit_px = current_sl
                        exit_reason = "TRAIL_STOP" if is_trailing else "STOP_LOSS"
                        break
                        
            buy_val = (ep if side == "BUY" else exit_px) * qty
            sell_val = (exit_px if side == "BUY" else ep) * qty
            gross = round(sell_val - buy_val, 2)
            tax = calc_mis_taxes(buy_val, sell_val)
            net = round(gross - tax, 2)
            
            trades.append({
                "sym": s, "date": d, "side": side, "entry_px": ep, "exit_px": exit_px, "qty": qty,
                "gross_pnl": gross, "taxes": tax, "net_pnl": net, "reason": exit_reason
            })
            
    return evaluate_trades(trades), trades


if __name__ == "__main__":
    print("[*] Loading historical data...")
    raw = load_data()
    print("Loaded.\n")
    
    print("="*95)
    print("SEARCHING FOR POSITIVE NET PROFIT STRATEGIES (NET PNL > 0 AFTER ALL TAXES & UPSTOX FEES)")
    print("="*95)
    
    profitable_runs = []
    
    for vol in [2.5, 3.0, 3.5, 4.0]:
        for sl in [0.007, 0.009, 0.012]:
            for be_act in [0.008, 0.010, 0.012, 0.015]:
                for trail_dist in [0.008, 0.010, 0.012, 0.015]:
                    res, trs = run_diamond_simulation(
                        raw,
                        min_vol_mult=vol,
                        sl_pct=sl,
                        be_activation_pct=be_act,
                        trail_dist_pct=trail_dist,
                        symbols=["BSE", "DIXON", "HAL", "TRENT"]
                    )
                    if res["trades"] >= 15 and res["net_pnl"] > 0:
                        profitable_runs.append((res, trs, (vol, sl, be_act, trail_dist)))
                        print(f"[FOUND WINNER] Vol:{vol}x SL:{sl*100:.1f}% BE:+{be_act*100:.1f}% Trail:{trail_dist*100:.1f}% --> Trades:{res['trades']:<2} WR:{res['win_rate']:>4.1f}% | Gross:Rs.{res['gross_pnl']:>8.2f} | Taxes:Rs.{res['taxes']:>7.2f} | Net:Rs.{res['net_pnl']:>+8.2f} | Ret:{res['return_pct']:>+5.1f}% | PF:{res['pf']}")

    print("\n" + "="*95)
    print(f"TOTAL PROFITABLE CONFIGURATIONS DISCOVERED: {len(profitable_runs)}")
    print("="*95)
    
    if profitable_runs:
        # Sort by Net PnL
        profitable_runs.sort(key=lambda x: x[0]["net_pnl"], reverse=True)
        best_res, best_trs, best_cfg = profitable_runs[0]
        v, s, b, t = best_cfg
        print(f"\n[TOP DIAMOND STRATEGY]")
        print(f"Parameters: Vol Surge >= {v}x | SL: {s*100:.1f}% | Never-Red Lock at: +{b*100:.1f}% | Trailing Buffer: {t*100:.1f}%")
        print(f"Total Trades:         {best_res['trades']} ({round(best_res['trades']/6.0, 1)} trades/month)")
        print(f"Win Rate:             {best_res['win_rate']}%")
        print(f"Gross Market Gain:    Rs. {best_res['gross_pnl']:+,.2f}")
        print(f"Statutory Taxes Paid: Rs. {best_res['taxes']:,.2f}")
        print(f"NET TAKE-HOME PROFIT: Rs. {best_res['net_pnl']:+,.2f}  (Return on Capital: {best_res['return_pct']:+.1f}%)")
        print(f"Profit Factor:        {best_res['pf']}")
        print(f"Max Drawdown:         Rs. {best_res['max_dd']:,.2f}")
        print(f"Average Win:          Rs. {best_res['avg_win']:,.2f}")
        print(f"Average Loss:         Rs. {best_res['avg_loss']:,.2f}")
        print(f"Payoff Ratio (W / L): {best_res['avg_win'] / best_res['avg_loss']:.2f}x")
        
        print("\n" + "="*85)
        print("                         MONTH-BY-MONTH BREAKDOWN                           ")
        print("="*85)
        by_month = {}
        for tr in best_trs:
            m = tr["date"][:7]
            by_month.setdefault(m, {"trades": 0, "wins": 0, "gross": 0.0, "tax": 0.0, "net": 0.0})
            by_month[m]["trades"] += 1
            if tr["net_pnl"] > 0:
                by_month[m]["wins"] += 1
            by_month[m]["gross"] += tr["gross_pnl"]
            by_month[m]["tax"] += tr["taxes"]
            by_month[m]["net"] += tr["net_pnl"]
            
        print(f"{'Month':<10} | {'Trades':<7} | {'Win Rate':<9} | {'Gross PnL':<12} | {'Taxes':<10} | {'Net PnL':<12}")
        print("-"*85)
        for m in sorted(by_month.keys()):
            st = by_month[m]
            wr = st["wins"] / st["trades"] * 100.0 if st["trades"] > 0 else 0
            print(f"{m:<10} | {st['trades']:<7} | {wr:>7.1f}% | Rs. {st['gross']:>9.2f} | Rs. {st['tax']:>7.2f} | Rs. {st['net']:>9.2f}")
            
        print("\n" + "="*85)
        print("                         INDIVIDUAL TRADE AUDIT LOG                         ")
        print("="*85)
        for idx, tr in enumerate(best_trs):
            print(f"#{idx+1:<2} | {tr['date']} | {tr['sym']:<8} | {tr['side']:<4} | In: {tr['entry_px']:>8.2f} | Out: {tr['exit_px']:>8.2f} | Qty: {tr['qty']:<3} | Gross: {tr['gross_pnl']:>+8.2f} | Tax: {tr['taxes']:>5.2f} | Net: {tr['net_pnl']:>+8.2f} | {tr['reason']}")

