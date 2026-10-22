# execution/execution_config.py
"""
Execution & Risk Configuration.
Edit CAPITAL_PER_TRADE directly to control the exact amount of money used per trade.
All calculations automatically factor in Indian MIS leverage and statutory taxation.
"""

# ==============================================================================
# 💰 MASTER CAPITAL SIZING PER TRADE (EDIT THIS NUMBER)
# ==============================================================================
# The exact cash amount (in Rs.) you want to allocate for each trade.
# Example: 10000, 20000, 50000, 100000
CAPITAL_PER_TRADE = 10000.0

# In India, SEBI allows 5x intraday leverage on Equity MIS Cash.
# When True: Buying power per trade = CAPITAL_PER_TRADE * 5.0
# When False: Buying power per trade = strictly CAPITAL_PER_TRADE (no leverage)
USE_MIS_LEVERAGE = True
MIS_LEVERAGE_MULTIPLIER = 5.0

# Always size based on CAPITAL_PER_TRADE and dispatch straight to Upstox.
# Upstox order book will log the footprint even if account balance is low.
DYNAMIC_MARGIN_SIZING = False

# ==============================================================================
# 🤖 ULTRON AI THRESHOLDS & GATES
# ==============================================================================
# Minimum AI Win-Probability required from the XGBoost Ensemble (0.0 to 1.0)
# Trades below 70% probability are automatically rejected.
MIN_AI_PROBABILITY = 0.70

# Minimum Reward-to-Tax Ratio:
# Gross target profit MUST be at least 4x total estimated Indian taxes + brokerage.
# Protects against low-margin scalps where broker and government taxes eat all profits.
MIN_REWARD_TO_TAX_RATIO = 4.0

# ==============================================================================
# 🇮🇳 INDIAN STATUTORY TAXES & BROKERAGE SPECIFICATION (EQUITY MIS CASH)
# ==============================================================================
BROKERAGE_PER_ORDER = 20.0       # Rs. 20 per executed order (or 0.05%, whichever is lower)
STT_INTRADAY_PCT = 0.00025       # Securities Transaction Tax: 0.025% on sell side
EXCHANGE_TXN_PCT = 0.0000297     # NSE Transaction Charges: 0.00297% round trip
SEBI_CHARGES_PCT = 0.0000010     # SEBI Turnover Fee: 0.0001%
STAMP_DUTY_PCT = 0.0000300       # Stamp Duty: 0.003% on buy turnover
GST_PCT = 0.18                   # Goods & Services Tax: 18% on (Brokerage + Txn + SEBI)

# ==============================================================================
# ⏰ INSTITUTIONAL MARKET CLOCK (HARD TIME GATES)
# ==============================================================================
# No entries during opening 5-minute volatility chaos (wait for price discovery)
MARKET_START_TIME = "09:20:00"

# ABSOLUTE HARD CUTOFF: No new entries after 14:15 IST (2:15 PM).
# Upstox auto-squares off at 15:15 IST. Trading in final hour is strictly forbidden.
MARKET_ENTRY_CUTOFF_TIME = "14:15:00"

# Midday Lull Window (11:15 AM to 13:15 PM): Higher selectivity, pullbacks only
MIDDAY_START_TIME = "11:15:00"
MIDDAY_END_TIME = "13:15:00"

# ==============================================================================
# 🛡️ RISK PARAMETERS & DYNAMIC ATR VOLATILITY EXITS
# ==============================================================================
ENABLE_ATR_BASED_TARGET_SL = True
ATR_TARGET_MULTIPLIER = 2.0       # 2.0x ATR target
ATR_SL_MULTIPLIER = 0.75          # 0.75x ATR initial stop loss (Asymmetric 1:2.7 R:R)

STOP_LOSS_PCT = 0.0045           # 0.45% fallback stop loss
TARGET_PCT = 0.0140              # 1.40% fallback target

# Breakeven Step: Once trade hits +0.50% profit, move SL to entry price
BREAKEVEN_MOVE_PCT = 0.0050

# Partial Runner Step: Reaches +0.70%, fails continuation -> lock in profit
PARTIAL_EXIT_MOVE_PCT = 0.0070
PARTIAL_EXIT_LIMIT_PCT = 0.0065

# Broker Server-Side GTT OCO Orders (Automated exchange-level SL & Target)
USE_GTT_OCO = True

# ==============================================================================
# 🛑 DAILY CIRCUIT BREAKERS & ADAPTIVE SECTOR MEMORY
# ==============================================================================
# Reduced to 8 trades: Prioritizes elite quality over trade churn
MAX_TRADES_PER_DAY = 8
MAX_STOP_LOSSES = 4              # Shuts down engine after 4 SL hits
MAX_TARGET_HITS = 5
MAX_PARTIAL_EXITS = 4

# Adaptive Sector Cooldown: If a stock in a sector gets stopped out, freeze that sector
SECTOR_COOLDOWN_MINUTES = 45

# 🛡️ HARD DAILY LOSS CIRCUIT BREAKER IN RUPEES
# If the session's cumulative net loss touches this amount, all trading halts immediately.
MAX_DAILY_LOSS_RUPEES = 2500.0

# 🔀 DIVERSIFICATION & EXPOSURE CAPS
MAX_CONCURRENT_POSITIONS = 3     # Maximum open trades allowed at the same time
MAX_POSITIONS_PER_SECTOR = 1     # Max 1 open trade per sector (prevents correlated sector dump)

# ⏳ STALE ORDER TIMEOUT
# Limit orders not filled within this many seconds are automatically canceled from Upstox
LIMIT_ORDER_TIMEOUT_SECONDS = 30

# 🧪 PAPER TRADING / SIMULATION LAYER
# False = Real Indian MIS cash orders sent to Upstox
# True = Virtual simulated fills with realistic slippage & live P&L tracking (Zero real risk)
PAPER_TRADING = False

# 📱 TELEGRAM INSTANT MOBILE NOTIFICATIONS
# Get live trade pings, breakeven alerts, and EOD summaries on your phone at your day job.
TELEGRAM_ENABLED = True
TELEGRAM_BOT_TOKEN = "8923836777:AAGgMlM2kPeOrH1bwVbauepFxk7oKjsR_QM"
TELEGRAM_CHAT_ID = "6215201988"

# 📊 VISUAL REPORTING
GENERATE_EOD_REPORT = True       # Automatically generates dark-mode HTML trade card at 15:15 IST

# 🎯 SMART PASSIVE LIMIT PEGGING (Spread Slippage Eliminator)
# Sits passively on the best bid/ask to eliminate crossing the bid-ask spread
# Saves ₹400-₹800/day in pure friction costs across 8 trades
USE_PASSIVE_PEGGING = True
PASSIVE_PEG_BUFFER_PCT = 0.0000       # 0.0000 = exact bid/ask (zero spread crossing)
PASSIVE_PEG_TIMEOUT_SECONDS = 10      # Escalate/cancel if liquidity doesn't come within 10s

# Price buffer for aggressive limit fallback (+0.03% buy / -0.03% sell to force fills if needed)
LIMIT_BUFFER_PCT = 0.0003

# Order Types
ORDER_TYPE = "LIMIT"             # Entry order type
EXIT_ORDER_TYPE = "MARKET"        # Guaranteed exit type

# Upstox V3 Full Feed Watchlist Limit (Multi-Socket Pool enables scanning all Nifty 500 stocks)
MAX_WATCHLIST_SIZE = 500
USE_PORTFOLIO_STREAMER = True

