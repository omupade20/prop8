# strategy/mtf_builder.py
"""
MTFBuilder — build higher-timeframe candles from 1-minute bars.

Design goals:
- No extra API calls (aggregate 1m bars you already have).
- Low latency: aggregates the last N 1-minute bars immediately.
- Memory-safe: uses deque with configurable maxlen.
- Simple, deterministic API: update(...) + get_latest_tf(...) / get_tf_history(...).
"""

from collections import defaultdict, deque
from datetime import datetime
from typing import Deque, Dict, List, Optional, Union

ISOFMT = "%Y-%m-%dT%H:%M:%S"


def _to_minute_iso(ts: Union[str, datetime]) -> str:
    if isinstance(ts, str):
        try:
            dt = datetime.strptime(ts, ISOFMT)
        except Exception:
            # try parsing ISO flexibly
            dt = datetime.fromisoformat(ts)
    else:
        dt = ts
    dt = dt.replace(second=0, microsecond=0)
    return dt.strftime(ISOFMT)


class MTFBuilder:
    """
    Builds higher timeframe candles (N-minute) from 1-minute bars.

    Usage:
      - Call update(inst_key, timestamp, o,h,l,c,v) for each 1-minute bar (or register a callback on bar close).
      - Use get_latest_tf(inst_key, minutes=5) to get aggregated candle for last `minutes` 1-minute bars.
      - Use get_tf_history(inst_key, minutes=5, lookback=3) to get last 3 aggregated 5-min candles (oldest->newest).
    """

    def __init__(self, max_1m_bars: int = 2000):
        # store recent 1-minute bars per instrument
        # each element is dict: {"time": "YYYY-MM-DDTHH:MM:SS", "open":, "high":, "low":, "close":, "volume":}
        self.max_1m_bars = max_1m_bars
        self.buffers: Dict[str, Deque[dict]] = defaultdict(lambda: deque(maxlen=self.max_1m_bars))

    def update(self, inst_key: str, timestamp: Union[str, datetime], o: float, h: float, l: float, c: float, v: float):
        """
        Add a 1-minute bar. timestamp may be ISO string or datetime.
        We normalize to minute boundary automatically.
        """
        t_iso = _to_minute_iso(timestamp)
        bar = {"time": t_iso, "open": o, "high": h, "low": l, "close": c, "volume": v}
        self.buffers[inst_key].append(bar)

    def _aggregate(self, bars: List[dict]) -> dict:
        """
        Aggregate a list of 1-minute bar dicts (oldest->newest) into one N-minute candle.
        """
        return {
            "time_start": bars[0]["time"],
            "time_end": bars[-1]["time"],
            "open": bars[0]["open"],
            "high": max(b["high"] for b in bars),
            "low": min(b["low"] for b in bars),
            "close": bars[-1]["close"],
            "volume": sum(b.get("volume", 0) for b in bars)
        }

    def get_latest_tf(self, inst_key: str, minutes: int = 5) -> Optional[dict]:
        """
        Return aggregated candle of the last `minutes` 1-minute bars (oldest->newest inside).
        If not enough bars, returns None.
        """
        bars = self.buffers.get(inst_key)
        if not bars or len(bars) < minutes:
            return None
        chunk = list(bars)[-minutes:]
        return self._aggregate(chunk)

    def get_tf_history(self, inst_key: str, minutes: int = 5, lookback: int = 3) -> List[dict]:
        """
        Return a list of aggregated candles for the given TF.
        lookback = how many aggregated candles to return (oldest -> newest).
        Each aggregated candle uses contiguous blocks of `minutes` 1-minute bars.
        If there isn't enough data to fill all lookback candles, returns as many as possible.
        """
        out: List[dict] = []
        bars = self.buffers.get(inst_key)
        if not bars:
            return out

        bar_list = list(bars)
        total = len(bar_list)
        # compute aggregated windows from the tail
        for i in range(lookback, 0, -1):
            end = total - (i - 1) * minutes
            start = end - minutes
            if start < 0:
                continue
            chunk = bar_list[start:end]
            out.append(self._aggregate(chunk))

        return out

    # convenience helpers
    def get_latest_5m(self, inst_key: str) -> Optional[dict]:
        return self.get_latest_tf(inst_key, minutes=5)

    def get_latest_15m(self, inst_key: str) -> Optional[dict]:
        return self.get_latest_tf(inst_key, minutes=15)
