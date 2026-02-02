# execution/execution_config.py

"""
Execution configuration.
These values are editable directly by you
without changing logic elsewhere.
"""

# -------- CAPITAL PER TRADE --------
# Amount you *intend* to put per trade (in ₹)
CAPITAL_PER_TRADE = 50000  # e.g., ₹75,000

# -------- PRICE BUFFER FOR LIMIT --------
# When placing limit orders, adjust price slightly
# to help execution without slippage.
# +0.03% for BUY, -0.03% for SELL by default
LIMIT_BUFFER_PCT = 0.0003

# -------- RISK PARAMETERS --------
STOP_LOSS_PCT = 0.0045        # 0.45% initial stop loss
TARGET_PCT = 0.0120           # 1.20% primary target

# If reached +0.50% → move SL to entry breakeven
BREAKEVEN_MOVE_PCT = 0.0050   # 0.50%

# If reached +0.70%, but fails continuation → exit at +0.65%
PARTIAL_EXIT_MOVE_PCT = 0.0070   # +0.70%
PARTIAL_EXIT_LIMIT_PCT = 0.0065  # exit level

# -------- DAILY RISK LIMITS --------
MAX_STOP_LOSSES = 5
MAX_TARGET_HITS = 5
MAX_PARTIAL_EXITS = 5

# -------- ORDER SETTINGS --------
ORDER_TYPE = "LIMIT"      # Always "LIMIT" (not market)

# -------- SAFETY LIMITS --------
MAX_TRADES_PER_DAY = 15
"""
Number of maximum executed trades per day regardless of outcome.
This prevents overtrading.
"""
