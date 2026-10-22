# strategy/sector_manager.py
"""
Institutional Sector Diversification Manager.
Prevents dangerous correlated sector dump (e.g. buying 3 banking stocks at once).
Enforces MAX_CONCURRENT_POSITIONS and MAX_POSITIONS_PER_SECTOR.
"""

from typing import Dict, List, Tuple
from execution.execution_config import MAX_CONCURRENT_POSITIONS, MAX_POSITIONS_PER_SECTOR

# Canonical Sector Mapping for Nifty 100/200 Equity Universe
SECTOR_MAP: Dict[str, str] = {
    # Banking
    "HDFCBANK": "BANKING", "ICICIBANK": "BANKING", "SBIN": "BANKING", "KOTAKBANK": "BANKING",
    "AXISBANK": "BANKING", "INDUSINDBK": "BANKING", "BANKBARODA": "BANKING", "PNB": "BANKING",
    "CANBK": "BANKING", "FEDERALBNK": "BANKING", "IDFCFIRSTB": "BANKING", "AUBANK": "BANKING",
    
    # Financial Services
    "BAJFINANCE": "FINANCE", "BAJAJFINSV": "FINANCE", "HDFCLIFE": "FINANCE", "SBILIFE": "FINANCE",
    "CHOLAFIN": "FINANCE", "SHRIRAMFIN": "FINANCE", "MUTHOOTFIN": "FINANCE", "RECLTD": "FINANCE",
    "PFC": "FINANCE", "ICICIPRULI": "FINANCE", "ICICIGI": "FINANCE",
    
    # Information Technology
    "TCS": "IT", "INFY": "IT", "HCLTECH": "IT", "WIPRO": "IT", "TECHM": "IT", "LTIM": "IT",
    "PERSISTENT": "IT", "COFORGE": "IT", "MPHASIS": "IT", "LTTS": "IT", "TATAELXSI": "IT",
    
    # Oil, Gas & Energy
    "RELIANCE": "ENERGY", "ONGC": "ENERGY", "NTPC": "ENERGY", "POWERGRID": "ENERGY",
    "BPCL": "ENERGY", "IOC": "ENERGY", "COALINDIA": "ENERGY", "GAIL": "ENERGY",
    "ADANIGREEN": "ENERGY", "ADANIPOWER": "ENERGY", "TATAPOWER": "ENERGY",
    
    # Automobile & Auto Ancillary
    "TATAMOTORS": "AUTO", "M&M": "AUTO", "MARUTI": "AUTO", "BAJAJ-AUTO": "AUTO",
    "EICHERMOT": "AUTO", "HEROMOTOCO": "AUTO", "TVSMOTOR": "AUTO", "BHARATFORG": "AUTO",
    "BOSCHLTD": "AUTO", "MRF": "AUTO", "APOLLOTYRE": "AUTO",
    
    # Pharmaceuticals & Healthcare
    "SUNPHARMA": "PHARMA", "CIPLA": "PHARMA", "DRREDDY": "PHARMA", "DIVISLAB": "PHARMA",
    "APOLLOHOSP": "PHARMA", "MAXHEALTH": "PHARMA", "LUPIN": "PHARMA", "AUROPHARMA": "PHARMA",
    "TORNTPHARM": "PHARMA", "ZYDUSLIFE": "PHARMA", "MANKIND": "PHARMA",
    
    # Metals & Mining
    "TATASTEEL": "METALS", "JSWSTEEL": "METALS", "HINDALCO": "METALS", "JINDALSTEL": "METALS",
    "VEDL": "METALS", "NMDC": "METALS", "SAIL": "METALS", "NATIONALUM": "METALS",
    
    # FMCG & Consumer Goods
    "ITC": "FMCG", "HINDUNILVR": "FMCG", "NESTLEIND": "FMCG", "BRITANNIA": "FMCG",
    "TATACONSUM": "FMCG", "GODREJCP": "FMCG", "DABUR": "FMCG", "MARICO": "FMCG",
    "COLPAL": "FMCG", "VBL": "FMCG", "RADICO": "FMCG",
    
    # Infrastructure, Cement & Industrials
    "LT": "INFRA", "ULTRACEMCO": "CEMENT", "GRASIM": "CEMENT", "AMBUJACEM": "CEMENT",
    "SHREECEM": "CEMENT", "ADANIENT": "INFRA", "ADANIPORTS": "INFRA", "SIEMENS": "INDUSTRIALS",
    "ABB": "INDUSTRIALS", "BEL": "DEFENSE", "HAL": "DEFENSE", "BHEL": "INDUSTRIALS"
}


import time
from execution.execution_config import MAX_CONCURRENT_POSITIONS, MAX_POSITIONS_PER_SECTOR, SECTOR_COOLDOWN_MINUTES

class SectorManager:
    """
    Guarantees structural risk diversification across Indian equity sectors.
    Includes Adaptive Sector Cooldown Memory: pauses stopped-out sectors for 45 mins.
    """
    def __init__(
        self,
        max_concurrent: int = MAX_CONCURRENT_POSITIONS,
        max_per_sector: int = MAX_POSITIONS_PER_SECTOR
    ):
        self.max_concurrent = max_concurrent
        self.max_per_sector = max_per_sector
        self.cooldowns: Dict[str, float] = {}
        self.symbol_returns: Dict[str, float] = {}

    def update_symbol_return(self, inst_key: str, return_pct: float):
        """Records intraday % change for a stock to compute real-time sector momentum."""
        sym = self.get_symbol_clean(inst_key)
        self.symbol_returns[sym] = return_pct

    def check_sector_sympathy(self, candidate_inst: str, direction: str) -> Tuple[bool, str]:
        """
        Validates directional alignment with parent sector performance.
        Vetoes BUY if sector is in a downtrend (return < -0.35%).
        Vetoes SHORT if sector is in a strong rally (return > +0.35%).
        """
        sector = self.get_sector(candidate_inst)
        if sector == "GENERAL_EQUITY":
            return True, "GENERAL_EQUITY_NEUTRAL"

        # Calculate average return for this sector from recorded constituents
        sector_stocks = [sym for sym, sec in SECTOR_MAP.items() if sec == sector]
        rets = [self.symbol_returns[sym] for sym in sector_stocks if sym in self.symbol_returns]

        if not rets or len(rets) < 2:
            return True, "INSUFFICIENT_SECTOR_DATA"

        avg_sec_ret = sum(rets) / len(rets)

        if direction.upper() in ("LONG", "BUY") and avg_sec_ret < -0.0035:
            return False, f"SECTOR_SYMPATHY_DIVERGENCE ({sector} is falling {avg_sec_ret*100:+.2f}%. Longs blocked.)"

        if direction.upper() in ("SHORT", "SELL") and avg_sec_ret > 0.0035:
            return False, f"SECTOR_SYMPATHY_DIVERGENCE ({sector} is rallying {avg_sec_ret*100:+.2f}%. Shorts blocked.)"

        return True, f"SECTOR_SYMPATHY_ALIGNED ({sector} {avg_sec_ret*100:+.2f}%)"

    def get_symbol_clean(self, inst_key: str) -> str:
        """Extracts plain ticker symbol, e.g. NSE_EQ|INE002A01018 -> RELIANCE or clean symbol."""
        try:
            from core.jarvis_hud import get_clean_symbol
            sym = get_clean_symbol(inst_key)
            if sym and not sym.startswith("INE"):
                return sym.upper()
        except Exception:
            pass
        parts = inst_key.split("|")
        sym = parts[-1].replace("NSE_EQ", "").strip(":")
        return sym.upper()

    def get_sector(self, inst_key: str) -> str:
        sym = self.get_symbol_clean(inst_key)
        return SECTOR_MAP.get(sym, "GENERAL_EQUITY")

    def trigger_cooldown(self, sector: str, minutes: int = SECTOR_COOLDOWN_MINUTES):
        """Freezes an entire sector for specified minutes after a Stop-Loss hit."""
        if sector and sector != "GENERAL_EQUITY":
            self.cooldowns[sector] = time.time() + (minutes * 60.0)
            print(f"[SECTOR DEFENSE] {sector} sector locked for {minutes}m to prevent cascade losses.")

    def is_in_cooldown(self, sector: str) -> Tuple[bool, float]:
        expiry = self.cooldowns.get(sector, 0.0)
        remaining = expiry - time.time()
        if remaining > 0:
            return True, remaining
        return False, 0.0

    def can_enter_trade(self, candidate_inst: str, active_inst_keys: List[str]) -> Tuple[bool, str]:
        """
        Validates whether taking this trade violates concurrent, sectoral, or cooldown limits.
        """
        # 1. Check max concurrent positions limit
        if len(active_inst_keys) >= self.max_concurrent:
            return False, f"MAX_CONCURRENT_LIMIT_REACHED ({len(active_inst_keys)}/{self.max_concurrent} active)"

        candidate_sector = self.get_sector(candidate_inst)

        # 2. Check Adaptive Sector Cooldown
        in_cd, remaining = self.is_in_cooldown(candidate_sector)
        if in_cd:
            return False, f"SECTOR_IN_COOLDOWN ({candidate_sector} locked for {remaining/60.0:.1f}m to prevent revenge trade)"

        # 3. Check sector concentration
        if candidate_sector != "GENERAL_EQUITY":
            for active_inst in active_inst_keys:
                active_sector = self.get_sector(active_inst)
                if active_sector == candidate_sector:
                    return False, f"SECTOR_CONCENTRATION_LIMIT ({candidate_sector} already active in {active_inst})"

        return True, "APPROVED"


# Global Singleton
_sector_manager_instance = None

def get_sector_manager() -> SectorManager:
    global _sector_manager_instance
    if _sector_manager_instance is None:
        _sector_manager_instance = SectorManager()
    return _sector_manager_instance
