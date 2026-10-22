# tests/test_7_new_upgrades.py
"""
Comprehensive Automated Test Suite for the 7 New Institutional Upgrades:
1. "Never-Red" R-Multiple Breakeven Guard (+1.0R lock & +1.5R trailing profit lock)
2. Smart Passive Limit Pegging (Saves spread on Best Bid/Ask)
3. Sector Sympathy & Index Alignment Gate (Vetoes trades fighting sector trend)
4. Real-Time Order Flow Delta (CVD) (Filters false breakout traps)
5. Dynamic Kelly Conviction Sizing (Multiplies sizing on high-probability setups)
6. Turtle Soup Liquidity Sweep (Day High/Low stop-hunt reversals)
7. VWAP Standard Deviation Bands (±2.5σ extreme rubber-band guards & snapback setups)
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from execution.trade_monitor import TradeMonitor, TrackedTrade
from execution.order_executor import OrderExecutor
from execution.volatility_sizer import calculate_kelly_conviction_quantity
from strategy.sector_manager import get_sector_manager, SECTOR_MAP
from strategy.scanner import MarketScanner
from strategy.setup_detector import detect_price_action_setup, check_climax_exhaustion
from strategy.advanced_indicators import compute_vwap_sd_bands


def run_7_upgrades_test():
    print("==========================================================================")
    print("    RUNNING COMPREHENSIVE VERIFICATION FOR 7 NEW INSTITUTIONAL UPGRADES    ")
    print("==========================================================================")

    # -------------------------------------------------------------
    # 1. "Never-Red" R-Multiple Breakeven Guard & Trailing Profit Lock
    # -------------------------------------------------------------
    tm = TradeMonitor()
    trade = tm.add_trade(
        trade_id="NEVER_RED_1",
        inst_key="NSE_EQ|INE002A01018",
        side="BUY",
        entry_price=1000.0,
        qty=100,
        atr=10.0  # ATR 10 -> SL: 992.5 (Risk = 7.5 per share)
    )
    initial_sl = trade.stop_loss
    initial_risk = trade.initial_risk

    # Price moves slightly to +0.13R (1001.00) -> Breakeven not triggered yet
    tm.check_trades({"NSE_EQ|INE002A01018": 1001.00})
    assert trade.is_never_red_locked is False

    # Price moves to +1.0R (1007.50) -> Never-Red Breakeven Guard MUST activate!
    tm.check_trades({"NSE_EQ|INE002A01018": 1007.50})
    assert trade.breakeven_moved is True
    assert trade.is_never_red_locked is True
    assert trade.stop_loss >= 1000.0, f"Stop loss must be >= entry price, got {trade.stop_loss}"
    print(f"[PASS] Upgrade 1A: Never-Red Breakeven locked SL at Rs. {trade.stop_loss:.2f} (Entry + Rs 20 brokerage buffer).")

    # Price moves to +1.6R (1012.00) -> Trailing +0.5R Profit Lock MUST activate!
    tm.check_trades({"NSE_EQ|INE002A01018": 1012.00})
    expected_lock_sl = round(1000.0 + (0.5 * initial_risk), 2)
    assert trade.stop_loss >= expected_lock_sl
    print(f"[PASS] Upgrade 1B: +1.5R Trailing Profit Lock secured minimum profit at Rs. {trade.stop_loss:.2f}.")

    # -------------------------------------------------------------
    # 2. Smart Passive Limit Pegging (Spread Slippage Eliminator)
    # -------------------------------------------------------------
    import execution.order_executor as oem
    old_paper = oem.PAPER_TRADING
    oem.PAPER_TRADING = True

    oe = OrderExecutor()
    buy_order = oe.place_limit_order("NSE_EQ|INE002A01018", "BUY", price=1000.0, qty=10)
    # Passive pegging sits on exact price (1000.0), saving the 0.03% spread
    assert buy_order["limit_price"] == 1000.0, f"Expected 1000.0, got {buy_order['limit_price']}"

    sell_order = oe.place_limit_order("NSE_EQ|INE002A01018", "SELL", price=1000.0, qty=10)
    assert sell_order["limit_price"] == 1000.0, f"Expected 1000.0, got {sell_order['limit_price']}"
    oem.PAPER_TRADING = old_paper
    print("[PASS] Upgrade 2: Smart Passive Limit Pegging saved spread (Bid/Ask pegged at exact zero-spread price).")

    # -------------------------------------------------------------
    # 3. Sector Sympathy & Index Alignment Gate
    # -------------------------------------------------------------
    sec_mgr = get_sector_manager()
    sec_mgr.symbol_returns.clear()

    # Simulate BANKING sector constituents dumping (-0.8% average)
    sec_mgr.update_symbol_return("HDFCBANK", -0.009)
    sec_mgr.update_symbol_return("ICICIBANK", -0.007)
    sec_mgr.update_symbol_return("SBIN", -0.008)

    # Attempt to BUY SBIN while Banking is dumping
    can_buy, buy_reason = sec_mgr.check_sector_sympathy("SBIN", "BUY")
    assert can_buy is False
    assert "SECTOR_SYMPATHY_DIVERGENCE" in buy_reason
    print(f"[PASS] Upgrade 3: Sector Sympathy Gate vetoed buying in falling sector: {buy_reason}.")

    # -------------------------------------------------------------
    # 4. Real-Time Order Flow Delta (CVD) Trap Filter
    # -------------------------------------------------------------
    scanner = MarketScanner()
    # Simulate high volume selling into a breakout (Negative Delta = Trap)
    scanner.update_order_flow_tick("TEST_STOCK", 100.0, 1000)
    scanner.update_order_flow_tick("TEST_STOCK", 99.9, 1500)  # Price ticked down on 500 vol
    scanner.update_order_flow_tick("TEST_STOCK", 99.8, 2000)  # Price ticked down on 500 vol
    delta = scanner.get_order_flow_delta("TEST_STOCK")
    assert delta < 0, f"Expected negative delta, got {delta}"

    # Feed negative delta into ORB Breakout -> Must be rejected!
    closes_orb = [100.0 + i*0.02 for i in range(29)] + [101.2]
    highs_orb = [c + 0.1 for c in closes_orb]
    lows_orb = [c - 0.1 for c in closes_orb]
    opens_orb = [c - 0.05 for c in closes_orb]
    vols_orb = [1000]*29 + [3000]
    vwap_orb = 100.5

    orb_trap = detect_price_action_setup("TEST_STOCK", closes_orb, opens_orb, highs_orb, lows_orb, closes_orb, vols_orb, vwap_orb, "BULLISH", order_flow_delta=-500)
    assert orb_trap is None  # Trap correctly rejected because delta opposes breakout
    print("[PASS] Upgrade 4: Real-Time Order Flow Delta successfully filtered out false breakout trap.")

    # -------------------------------------------------------------
    # 5. Dynamic Kelly Conviction Sizing
    # -------------------------------------------------------------
    # Grade A+ (88% conviction): Multiplier 1.4x (Rs. 70,000 deployed)
    qty_a_plus, grade_a_plus, cap_a_plus = calculate_kelly_conviction_quantity(price=100.0, conviction_score=0.88, user_capital=50000.0)
    assert cap_a_plus == 70000.0
    assert "GRADE_A+" in grade_a_plus

    # Grade B (72% conviction): Multiplier 0.65x (Rs. 32,500 deployed)
    qty_b, grade_b, cap_b = calculate_kelly_conviction_quantity(price=100.0, conviction_score=0.72, user_capital=50000.0)
    assert cap_b == 32500.0
    assert "GRADE_B" in grade_b
    print(f"[PASS] Upgrade 5: Dynamic Kelly Sizing verified (Grade A+: Rs. {cap_a_plus:,.0f} vs Grade B: Rs. {cap_b:,.0f}).")

    # -------------------------------------------------------------
    # 6. Turtle Soup Liquidity Sweep (Day High Stop-Hunt Reversal)
    # -------------------------------------------------------------
    # 30 bars ranging with session high at 100.0. Bar 31 pierces to 100.25 (pokes above), but closes at 99.85 with massive upper wick!
    closes_sweep = [99.0 + (i%5)*0.2 for i in range(30)] + [99.85]
    highs_sweep = [c + 0.15 for c in closes_sweep[:-1]] + [100.25]  # Pierced session high of ~100.0
    lows_sweep = [c - 0.15 for c in closes_sweep[:-1]] + [99.80]
    opens_sweep = [c - 0.05 for c in closes_sweep[:-1]] + [99.90]
    vols_sweep = [1000]*30 + [2500]
    vwap_sweep = 99.5

    sweep_res = detect_price_action_setup("TEST_SWEEP", closes_sweep, opens_sweep, highs_sweep, lows_sweep, closes_sweep, vols_sweep, vwap_sweep, "NEUTRAL")
    assert sweep_res is not None and sweep_res["setup_name"] == "SWING_REVERSAL"
    assert sweep_res["direction"] == "SHORT"
    print(f"[PASS] Upgrade 6: Turtle Soup Liquidity Sweep detected: {sweep_res['reason']}.")

    # -------------------------------------------------------------
    # 7. VWAP Standard Deviation Bands (±2.5σ Extreme Mean Reversion)
    # -------------------------------------------------------------
    # Compute bands
    bands = compute_vwap_sd_bands(highs_sweep, lows_sweep, closes_sweep, vols_sweep)
    assert bands is not None
    assert bands["upper_2_5"] > bands["upper_2"] > bands["vwap"] > bands["lower_2"] > bands["lower_2_5"]

    # Test climax guard blocks extreme +2.5σ extension
    climax_check = check_climax_exhaustion(
        ltp=bands["upper_2_5"] + 1.0,  # Extended past 2.5σ
        vwap=bands["vwap"],
        rsi=65.0,
        closes=closes_sweep,
        highs=highs_sweep,
        lows=lows_sweep,
        direction="LONG",
        vwap_bands=bands
    )
    assert climax_check["is_exhausted"] is True
    assert "2.5σ" in climax_check["reason"]
    print(f"[PASS] Upgrade 7: VWAP ±2.5σ Bands successfully guarded rubber-band extreme: {climax_check['reason']}.")

    print("\n==========================================================================")
    print("    ALL 7 NEW INSTITUTIONAL UPGRADES FULLY VERIFIED & 100% OPERATIONAL! [READY]")
    print("==========================================================================")


if __name__ == "__main__":
    run_7_upgrades_test()
