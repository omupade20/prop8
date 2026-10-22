# strategy/market_depth_analyzer.py
"""
Level-2 Order Book & Market Depth Intelligence.
Analyzes Level-2 bid/ask microstructure from Upstox MarketFF feed:
- Top-5 Bid/Ask queue depth and volume concentration
- Total Buy Quantity (TBQ) vs Total Sell Quantity (TSQ) order book imbalance
- Bid-Ask spread penalty (detecting illiquid slippage traps)
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List


@dataclass
class DepthContext:
    score: float           # -1.0 to +1.0 (conviction contribution)
    imbalance_ratio: float # TBQ / TSQ
    spread_pct: float      # (Ask - Bid) / Bid * 100
    top5_ratio: float      # Sum(BidQ[5]) / Sum(AskQ[5])
    is_liquid: bool        # True if spread is tight and depth is adequate
    comment: str


def analyze_market_depth(
    market_ff: Optional[Dict[str, Any]],
    direction: str = "LONG",
    max_spread_pct: float = 0.15
) -> DepthContext:
    """
    Analyzes Upstox Level-2 market depth.
    market_ff contains: 'marketLevel', 'tbq', 'tsq', 'ltpc', etc.
    """
    if not market_ff:
        return DepthContext(
            score=0.0,
            imbalance_ratio=1.0,
            spread_pct=0.0,
            top5_ratio=1.0,
            is_liquid=True,
            comment="No depth data"
        )

    tbq = float(market_ff.get("tbq", 0.0) or 0.0)
    tsq = float(market_ff.get("tsq", 0.0) or 0.0)

    # Imbalance: Total Buy Quantity vs Total Sell Quantity
    if tsq > 0:
        imbalance = tbq / tsq
    elif tbq > 0:
        imbalance = 2.0
    else:
        imbalance = 1.0

    quotes: List[Dict[str, Any]] = market_ff.get("marketLevel", {}).get("bidAskQuote", []) or []

    top_bid_qty = 0.0
    top_ask_qty = 0.0
    best_bid = 0.0
    best_ask = 0.0

    if quotes:
        try:
            best_bid = float(quotes[0].get("bidP", 0.0) or 0.0)
            best_ask = float(quotes[0].get("askP", 0.0) or 0.0)
            top_bid_qty = sum(float(q.get("bidQ", 0.0) or 0.0) for q in quotes)
            top_ask_qty = sum(float(q.get("askQ", 0.0) or 0.0) for q in quotes)
        except Exception:
            pass

    # Top-5 ratio
    top5_ratio = (top_bid_qty / top_ask_qty) if top_ask_qty > 0 else 1.0

    # Bid-Ask spread
    if best_bid > 0 and best_ask >= best_bid:
        spread_pct = ((best_ask - best_bid) / best_bid) * 100.0
    else:
        spread_pct = 0.0

    # 1. Spread Check: Illiquid wide spread penalty
    if spread_pct > max_spread_pct:
        return DepthContext(
            score=-1.0,
            imbalance_ratio=round(imbalance, 2),
            spread_pct=round(spread_pct, 3),
            top5_ratio=round(top5_ratio, 2),
            is_liquid=False,
            comment=f"Wide spread ({spread_pct:.2f}%) - slippage risk"
        )

    score = 0.0
    comments = []

    # 2. Imbalance Scoring based on direction
    if direction == "LONG":
        if imbalance >= 1.5 and top5_ratio >= 1.2:
            score += 0.6
            comments.append("Heavy buyer depth wall")
        elif imbalance >= 1.15:
            score += 0.3
            comments.append("Buyer book advantage")
        elif imbalance < 0.7:
            score -= 0.5
            comments.append("Seller order book resistance")
    else:  # SHORT
        if imbalance <= 0.65 and top5_ratio <= 0.8:
            score += 0.6
            comments.append("Heavy seller ask wall")
        elif imbalance <= 0.85:
            score += 0.3
            comments.append("Seller book advantage")
        elif imbalance > 1.3:
            score -= 0.5
            comments.append("Strong buyer bid absorption")

    # Clamping
    score = max(-1.0, min(1.0, score))

    return DepthContext(
        score=round(score, 2),
        imbalance_ratio=round(imbalance, 2),
        spread_pct=round(spread_pct, 3),
        top5_ratio=round(top5_ratio, 2),
        is_liquid=True,
        comment=" | ".join(comments) if comments else "Neutral order book"
    )


class MarketDepthAnalyzer:
    """
    Object-oriented Level-2 order book depth analyzer.
    """
    def __init__(self, max_spread_pct: float = 0.15):
        self.max_spread_pct = max_spread_pct

    def analyze_feed(self, market_ff: Optional[Dict[str, Any]], direction: str = "LONG") -> DepthContext:
        return analyze_market_depth(market_ff, direction=direction, max_spread_pct=self.max_spread_pct)

    def analyze(self, bids: List[Dict[str, Any]], asks: List[Dict[str, Any]], ltp: float = 0.0, direction: str = "LONG") -> Dict[str, Any]:
        """
        Accepts raw bids and asks list: [{'price': p, 'quantity': q}, ...]
        Returns dictionary of microstructure metrics.
        """
        total_bid_qty = sum(b.get('quantity', 0) for b in bids)
        total_ask_qty = sum(a.get('quantity', 0) for a in asks)
        best_bid = bids[0].get('price', 0.0) if bids else 0.0
        best_ask = asks[0].get('price', 0.0) if asks else 0.0

        spread_pct = ((best_ask - best_bid) / best_bid * 100.0) if best_bid > 0 and best_ask >= best_bid else 0.0
        imbalance_ratio = (total_bid_qty / total_ask_qty) if total_ask_qty > 0 else 2.0

        return {
            'best_bid': best_bid,
            'best_ask': best_ask,
            'total_bid_qty': total_bid_qty,
            'total_ask_qty': total_ask_qty,
            'imbalance_ratio': round(imbalance_ratio, 2),
            'spread_pct': round(spread_pct, 4),
            'is_liquid': spread_pct <= self.max_spread_pct
        }

