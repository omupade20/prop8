# backtest/probe_nifty100.py
import json
import os
import requests

INSTRUMENTS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "instruments.json")

# Top Nifty 100 symbols list (standard official NSE Nifty 100 constituents)
NIFTY100_SYMBOLS = [
    # Nifty 50
    "RELIANCE", "TCS", "HDFCBANK", "BHARTIARTL", "ICICIBANK", "INFY", "SBIN", "HINDUNILVR", "ITC", "LT",
    "BAJFINANCE", "HCLTECH", "M&M", "SUNPHARMA", "MARUTI", "KOTAKBANK", "AXISBANK", "ONGC", "TITAN", "NTPC",
    "TATAMOTORS", "POWERGRID", "ADANIENT", "ULTRACEMCO", "COALINDIA", "BAJAJFINSV", "TATASTEEL", "ASIANPAINT",
    "SIEMENS", "JSWSTEEL", "ADANIPORTS", "GRASIM", "TECHM", "NESTLEIND", "ZOMATO", "HINDALCO", "BEL",
    "BPCL", "TRENT", "CIPLA", "WIPRO", "APOLLOHOSP", "DRREDDY", "BAJAJ-AUTO", "HEROMOTOCO", "EICHERMOT",
    "SHRIRAMFIN", "DIVISLAB", "TATACONSUM", "SBILIFE",
    # Nifty Next 50
    "HAL", "VBL", "DLF", "CHOLAFIN", "PFC", "VEDL", "RECLTD", "IOC", "PIDILITIND", "GAIL",
    "AMBUJACEM", "INDIGO", "TORNTPHARM", "CANBK", "BANKBARODA", "HAVELLS", "ICICIPRULI", "JINDALSTEL",
    "ABB", "SRF", "DABUR", "PNB", "MOTHERSON", "ZYDUSLIFE", "BERGEPAINT", "ICICIGI", "BOSCHLTD",
    "TVSMOTOR", "LODHA", "CGPOWER", "POLYCAB", "NAUKRI", "IRFC", "MAXHEALTH", "SOLARINDS", "BSE",
    "DIXON", "PERSISTENT", "COLPAL", "OFSS", "AUBANK", "GODREJCP", "MUTHOOTFIN", "CUMMINSIND", "LICI"
]

def main():
    with open(INSTRUMENTS_FILE, "r", encoding="utf-8") as f:
        instruments = json.load(f)
        
    symbol_map = {}
    for item in instruments:
        ts = item.get("trading_symbol")
        ik = item.get("instrument_key")
        if ts and ik:
            symbol_map[ts] = ik
            
    found = {}
    missing = []
    for s in NIFTY100_SYMBOLS:
        clean = s.replace("&", "").replace("-", "")
        # Check direct or variations
        if s in symbol_map:
            found[s] = symbol_map[s]
        elif clean in symbol_map:
            found[s] = symbol_map[clean]
        else:
            # search
            m = next((symbol_map[k] for k in symbol_map if k.startswith(s)), None)
            if m:
                found[s] = m
            else:
                missing.append(s)
                
    print(f"Total Nifty 100 symbols matched in instruments.json: {len(found)} / {len(NIFTY100_SYMBOLS)}")
    print(f"Sample matches: {list(found.items())[:5]}")
    if missing:
        print(f"Missing ({len(missing)}): {missing[:10]}")

if __name__ == "__main__":
    main()
