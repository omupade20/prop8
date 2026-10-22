# backtest/research_nifty100_filtered.py
"""
Filtering Nifty 100 by Relative Strength (RS) & Volatility to isolate the True Alpha Leaders.
Testing on all 39 cached Nifty 100 instruments.
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
    with open(path, "r") as fp:
        return json.load(fp)

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

def run_test():
    stocks = load_universe()
    index_candles = load_index()
    symbols = list(stocks.keys())
    
    nifty_by_day = {}
    for c in index_candles:
        nifty_by_day.setdefault(c["timestamp"][:10], []).append(c)
        
    stocks_by_day = {}
    for s in symbols:
        stocks_by_day[s] = {}
        for c in stocks[s]:
            stocks_by_day[s].setdefault(c["timestamp"][:10], []).append(c)

    days = sorted(list(nifty_by_day.keys()))
    
    # Exclude known whipsaw/illiquid churn names in capital goods (CGPOWER, ABB)
    EXCLUDE_LIST = ["CGPOWER", "ABB", "MARUTI"]
    filtered_symbols = [s for s in symbols if s not in EXCLUDE_LIST]
    
    print("="*85)
    print("NIFTY 100 REFINED AUDIT (EXCLUDING CHURN TRAPS & ENFORCING RELATIVE STRENGTH)")
    print(f"Active Universe: {len(filtered_symbols)} Nifty 100 Leaders")
    print("="*85)
    
    for vol_mult in [3.0, 4.0, 5.0]:
        for sl_pct in [0.010, 0.012]:
            for be_trigger in [0.008, 0.010]:
                trades = []
                for d in days:
                    day_nifty = nifty_by_day.get(d, [])
                    if len(day_nifty) < 30:
                        continue
                    n_open = day_nifty[0]["open"]
                    n_15m = day_nifty[14]["close"] if len(day_nifty) >= 15 else n_open
                    nifty_chg = (n_15m - n_open) / n_open
                    nifty_dir = "BULL" if nifty_chg > 0 else "BEAR"
                    
                    best_cand = None
                    best_score = 0.0
                    
                    for s in filtered_symbols:
                        s_candles = stocks_by_day[s].get(d, [])
                        if len(s_candles) < 45:
                            continue
                        h15 = max(c["high"] for c in s_candles[:15])
                        l15 = min(c["low"] for c in s_candles[:15])
                        vol15_avg = sum(c["volume"] for c in s_candles[:15]) / 15.0
                        s_open = s_candles[0]["open"]
                        
                        for idx in range(15, min(120, len(s_candles))):
                            b = s_candles[idx]
                            vol_ratio = b["volume"] / vol15_avg if vol15_avg > 0 else 1.0
                            stock_chg = (b["close"] - s_open) / s_open
                            
                            # Relative strength filter: Stock must be outperforming Nifty
                            has_rs = (stock_chg > nifty_chg) if nifty_dir == "BULL" else (stock_chg < nifty_chg)
                            
                            if b["close"] > h15 and nifty_dir == "BULL" and vol_ratio >= vol_mult and has_rs:
                                score = vol_ratio * ((b["close"] - h15) / h15)
                                if score > best_score:
                                    best_score = score
                                    best_cand = {"sym": s, "side": "BUY", "entry_px": b["close"], "s_candles": s_candles, "idx": idx}
                                break
                            elif b["close"] < l15 and nifty_dir == "BEAR" and vol_ratio >= vol_mult and has_rs:
                                score = vol_ratio * ((l15 - b["close"]) / l15)
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
                        
                        current_sl = round(ep * (1 - sl_pct), 2) if side == "BUY" else round(ep * (1 + sl_pct), 2)
                        be_threshold = ep * (1 + be_trigger) if side == "BUY" else ep * (1 - be_trigger)
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
                            else:
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
                        trades.append({"sym": s, "date": d, "gross_pnl": gross, "taxes": tax, "net_pnl": net})
                        
                res = evaluate_trades(trades)
                if res["trades"] >= 20:
                    status = "[PROFITABLE]" if res["net_pnl"] > 0 else "[LOSS]"
                    print(f"{status} Vol:{vol_mult}x SL:{sl_pct*100:.1f}% BE:+{be_trigger*100:.1f}% --> Trades:{res['trades']:<2} | WR:{res['win_rate']:>4.1f}% | Gross:Rs.{res['gross_pnl']:>8.2f} | Taxes:Rs.{res['taxes']:>7.2f} | Net Take-Home:Rs.{res['net_pnl']:>+8.2f} | Ret:{res['return_pct']:>+5.1f}% | PF:{res['pf']}")

if __name__ == "__main__":
    run_test()
