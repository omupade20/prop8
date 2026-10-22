# backtest/download_nifty100.py
"""
Batch Downloader for Top Nifty 100 6-Month 1-Minute Historical Candles from Upstox API.
"""

import os
import json
import time
import datetime
import requests
import sys

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "backtest_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

INSTRUMENTS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "instruments.json")

# Top 35 High-Volume / High-Momentum Nifty 100 Constituents across all sectors
TARGET_NIFTY100_SYMBOLS = [
    # High Beta & Growth Leaders
    "DIXON", "TRENT", "BSE", "HAL", "BEL", "POLYCAB", "PERSISTENT", "CGPOWER", "CHOLAFIN", "SHRIRAMFIN",
    # Auto & Industrials
    "M&M", "MARUTI", "BAJAJ-AUTO", "TVSMOTOR", "SIEMENS", "ABB", "EICHERMOT",
    # Banking & Finance
    "HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK", "BAJFINANCE", "MUTHOOTFIN",
    # IT Leaders
    "TCS", "INFY", "HCLTECH", "TECHM",
    # Energy, Metals & Infra
    "RELIANCE", "LT", "TATASTEEL", "JSWSTEEL", "HINDALCO", "VEDL", "COALINDIA",
    # Consumer & Pharma
    "TITAN", "VBL", "SUNPHARMA", "CIPLA", "BHARTIARTL"
]

def get_symbol_map():
    with open(INSTRUMENTS_FILE, "r", encoding="utf-8") as f:
        instruments = json.load(f)
    symbol_map = {}
    for item in instruments:
        ts = item.get("trading_symbol")
        ik = item.get("instrument_key")
        if ts and ik:
            symbol_map[ts] = ik
    return symbol_map

def download_stock_6m(sym: str, key: str) -> bool:
    cache_file = os.path.join(CACHE_DIR, f"{sym}_1m_6m.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                data = json.load(f)
                if len(data) > 10000:
                    print(f"  [CACHED] {sym:<12}: already loaded ({len(data)} candles).", flush=True)
                    return True
        except Exception:
            pass

    end_date = datetime.date(2026, 10, 22)
    all_candles = []
    seen_timestamps = set()

    for i in range(6):
        to_d = (end_date - datetime.timedelta(days=i * 30)).strftime("%Y-%m-%d")
        from_d = (end_date - datetime.timedelta(days=(i + 1) * 30)).strftime("%Y-%m-%d")
        url = f"https://api.upstox.com/v2/historical-candle/{requests.utils.quote(key)}/1minute/{to_d}/{from_d}"
        
        for attempt in range(3):
            try:
                resp = requests.get(url, timeout=12)
                if resp.status_code == 200:
                    raw = resp.json().get("data", {}).get("candles", [])
                    for c in raw:
                        ts = c[0]
                        if ts not in seen_timestamps:
                            seen_timestamps.add(ts)
                            all_candles.append({
                                "timestamp": ts,
                                "open": float(c[1]),
                                "high": float(c[2]),
                                "low": float(c[3]),
                                "close": float(c[4]),
                                "volume": float(c[5])
                            })
                    break
                elif resp.status_code == 429:
                    time.sleep(1.0)
                else:
                    time.sleep(0.4)
            except Exception:
                time.sleep(0.4)
        time.sleep(0.08)

    all_candles.sort(key=lambda x: x["timestamp"])
    if len(all_candles) > 1000:
        with open(cache_file, "w") as f:
            json.dump(all_candles, f)
        print(f"  [DOWNLOADED] {sym:<12}: {len(all_candles)} candles saved.", flush=True)
        return True
    else:
        print(f"  [FAILED] {sym:<12}: only {len(all_candles)} candles.", flush=True)
        return False

def main():
    print("="*75, flush=True)
    print("   DOWNLOADING NIFTY 100 HISTORICAL 1-MINUTE CANDLES (6 MONTHS)   ", flush=True)
    print("="*75, flush=True)
    
    sym_map = get_symbol_map()
    downloaded = 0
    total = len(TARGET_NIFTY100_SYMBOLS)
    
    for idx, s in enumerate(TARGET_NIFTY100_SYMBOLS):
        clean = s.replace("&", "").replace("-", "")
        k = sym_map.get(s) or sym_map.get(clean)
        if not k:
            # prefix match
            k = next((sym_map[cand] for cand in sym_map if cand.startswith(clean)), None)
            
        print(f"[{idx+1}/{total}] Processing {s}...", flush=True)
        if k:
            ok = download_stock_6m(s, k)
            if ok:
                downloaded += 1
        else:
            print(f"  [KEY NOT FOUND] {s}", flush=True)
            
    print("\n" + "="*75, flush=True)
    print(f"DOWNLOAD COMPLETE: {downloaded} / {total} Nifty 100 symbols ready in cache.", flush=True)
    print("="*75, flush=True)

if __name__ == "__main__":
    main()
