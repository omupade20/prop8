# strategy/macro_index_filter.py
"""
Macro Benchmark Index Filter (Nifty 50 Market Beta).
Ensures individual stock trades align with the broader market tide:
- Tracks Nifty 50 intraday trend, EMA alignment, and percentage change
- Prevents buying into an index freefall (counter-trend risk)
- Prevents shorting into an aggressive index rally
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from strategy.indicators import exponential_moving_average


@dataclass
class MacroContext:
    index_bias: str         # BULLISH | BEARISH | NEUTRAL
    score_modifier: float   # -1.5 to +1.0 (multiplier/additive to stock score)
    index_ltp: float
    index_change_pct: float
    comment: str


class MacroIndexFilter:
    def __init__(self, index_key: str = "NSE_INDEX|Nifty 50"):
        self.index_key = index_key
        self.index_prices: List[float] = []
        self.day_open: Optional[float] = None

    def update(self, ltp: float, day_open: Optional[float] = None):
        """
        Record latest index tick.
        """
        if ltp > 0:
            self.index_prices.append(ltp)
            if len(self.index_prices) > 300:
                self.index_prices.pop(0)

        if day_open and day_open > 0 and self.day_open is None:
            self.day_open = day_open
        elif self.day_open is None and self.index_prices:
            self.day_open = self.index_prices[0]

    def evaluate_macro_alignment(self, trade_direction: str) -> MacroContext:
        """
        Evaluates whether trade_direction aligns with broader Nifty 50 market trend.
        """
        if not self.index_prices or len(self.index_prices) < 10:
            return MacroContext("NEUTRAL", 0.0, 0.0, 0.0, "Index data initializing")

        current_ltp = self.index_prices[-1]
        open_price = self.day_open or self.index_prices[0]
        change_pct = ((current_ltp - open_price) / max(open_price, 1e-9)) * 100.0

        ema_short = exponential_moving_average(self.index_prices, 9)
        ema_long = exponential_moving_average(self.index_prices, 21)

        # Classify index bias
        if change_pct > 0.3 and ema_short and ema_long and ema_short > ema_long:
            bias = "BULLISH"
        elif change_pct < -0.3 and ema_short and ema_long and ema_short < ema_long:
            bias = "BEARISH"
        else:
            bias = "NEUTRAL"

        score_modifier = 0.0
        comment = f"Nifty {change_pct:+.2f}% | {bias}"

        if trade_direction == "LONG":
            if bias == "BULLISH":
                score_modifier = 0.8
                comment += " | Index tailwind for Long"
            elif bias == "BEARISH":
                # Market is falling strongly -> penalize longs
                score_modifier = -1.2
                comment += " | Index headwind against Long"
        elif trade_direction == "SHORT":
            if bias == "BEARISH":
                score_modifier = 0.8
                comment += " | Index tailwind for Short"
            elif bias == "BULLISH":
                # Market is surging -> penalize shorts
                score_modifier = -1.2
                comment += " | Index headwind against Short"

        return MacroContext(
            index_bias=bias,
            score_modifier=score_modifier,
            index_ltp=round(current_ltp, 2),
            index_change_pct=round(change_pct, 2),
            comment=comment
        )

    def get_market_regime(self) -> str:
        ctx = self.evaluate_macro_alignment("LONG")
        return ctx.index_bias

    def filter_signal(self, inst_key: str, direction: str) -> tuple[bool, str]:
        ctx = self.evaluate_macro_alignment(direction)
        if ctx.score_modifier < -1.0:
            return False, f"Macro headwind: {ctx.comment}"
        return True, ctx.comment

    def calculate_relative_strength(self, stock_ltp: float, stock_open: float) -> Dict[str, Any]:
        """
        Computes Relative Strength (RS) of a stock compared to Nifty 50:
        RS = Stock % Change - Nifty 50 % Change
        - Outperforming (RS >= +0.35%): Strong Alpha -> Long Preferred
        - Underperforming (RS <= -0.35%): Weak Alpha -> Short Preferred
        - In-Line: Neutral
        """
        if not self.index_prices or not stock_open or stock_open <= 0:
            return {
                "rs_score": 0.0,
                "stock_pct": 0.0,
                "nifty_pct": 0.0,
                "classification": "NEUTRAL",
                "allowed_directions": ["LONG", "SHORT"]
            }

        current_idx = self.index_prices[-1]
        open_idx = self.day_open or self.index_prices[0]
        nifty_pct = ((current_idx - open_idx) / max(open_idx, 1e-9)) * 100.0

        stock_pct = ((stock_ltp - stock_open) / max(stock_open, 1e-9)) * 100.0
        rs_score = stock_pct - nifty_pct

        if rs_score >= 0.35:
            classification = "OUTPERFORMING"
            allowed = ["LONG"]
        elif rs_score <= -0.35:
            classification = "UNDERPERFORMING"
            allowed = ["SHORT"]
        else:
            classification = "NEUTRAL"
            allowed = ["LONG", "SHORT"]

        return {
            "rs_score": round(rs_score, 2),
            "stock_pct": round(stock_pct, 2),
            "nifty_pct": round(nifty_pct, 2),
            "classification": classification,
            "allowed_directions": allowed
        }

