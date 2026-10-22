# strategy/scanner.py
"""
Hardened MarketScanner (production-ready).
- keeps rolling 1-minute OHLCV bars per instrument (dict-of-deques)
- supports tick aggregation, direct OHLC bar ingestion (append_ohlc_bar)
- snapshot persistence and resume
- on_bar_close callbacks so MTF/strategy can run immediately when a bar closes
- alert throttling helpers (last_alert_time, dedupe)
- basic health checks and replay utilities
- thread-safe for use from websocket threads
"""

import json
import os
import threading
import time
from collections import deque, defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Callable, Optional

DEFAULT_MAX_LEN = 600  # keep 600 1-minute bars (~10 hours)

ISOFMT = "%Y-%m-%dT%H:%M:%S"  # simple ISO without tz


def _now_iso():
    return datetime.now().strftime(ISOFMT)


class MarketScanner:
    def __init__(self, max_len: int = DEFAULT_MAX_LEN, snapshot_path: Optional[str] = None):
        self.max_len = max_len
        self.snapshot_path = snapshot_path

        # core storage: per-symbol deque of bar dicts
        # bar dict: {"time": "YYYY-MM-DDTHH:MM:SS", "open":, "high":, "low":, "close":, "volume":}
        self._bars: Dict[str, deque] = {}
        self._locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)
        self._global_lock = threading.Lock()

        # quick-access caches (kept in sync) to preserve compatibility with your old getters
        # These will be derived from self._bars on request to avoid duplication.
        # last_alert_time and dedupe state
        self.last_alert_time: Dict[str, float] = {}
        self._dedupe_map: Dict[str, Dict[str, float]] = defaultdict(dict)  # inst -> {direction: ts}
        self._paused_until: Dict[str, float] = {}  # inst -> timestamp (epoch) until which instrument is paused

        # callbacks that are called when a 1-minute bar is appended / closed
        self._on_bar_close_callbacks: List[Callable[[str, dict], None]] = []

        # metrics
        self.bars_received = 0
        self.bars_closed = 0
        # Real-time Order Flow Delta (CVD - Cumulative Volume Delta)
        self._order_flow_delta: Dict[str, float] = defaultdict(float)
        self._last_ticks: Dict[str, Dict[str, float]] = {}

        # ensure snapshot directory exists when saving
        if self.snapshot_path:
            os.makedirs(os.path.dirname(self.snapshot_path), exist_ok=True)

    def update_order_flow_tick(self, inst: str, price: float, volume: float):
        """Calculates tick-level aggressive market orders (Buyer vs Seller delta)."""
        if inst in self._last_ticks:
            last_p = self._last_ticks[inst]["price"]
            last_v = self._last_ticks[inst]["volume"]
            delta_vol = max(0.0, volume - last_v)
            if price > last_p:
                self._order_flow_delta[inst] += delta_vol
            elif price < last_p:
                self._order_flow_delta[inst] -= delta_vol
        self._last_ticks[inst] = {"price": price, "volume": volume}

    def get_order_flow_delta(self, inst: str) -> float:
        """Returns Cumulative Volume Delta for instrument."""
        return self._order_flow_delta.get(inst, 0.0)

    # ---------------------
    # Internal helpers
    # ---------------------
    def _ensure_inst(self, inst: str):
        with self._global_lock:
            if inst not in self._bars:
                self._bars[inst] = deque(maxlen=self.max_len)

    def _lock_for(self, inst: str):
        # simple per-instrument lock object
        return self._locks[inst]

    def warmup_candles(self, inst: str, candles: List[dict]):
        """
        Preload historical candles (from Upstox History API) into scanner buffer.
        """
        if not candles:
            return
        self._ensure_inst(inst)
        with self._lock_for(inst):
            for c in candles:
                if all(k in c for k in ("time", "open", "high", "low", "close", "volume")):
                    self._bars[inst].append(c)

    # ---------------------
    # Append / ingestion
    # ---------------------
    def append_ohlc_bar(
        self,
        inst: str,
        time_iso: str,
        open_p: float,
        high_p: float,
        low_p: float,
        close_p: float,
        volume: float
    ):
        """
        Ingest a completed 1-minute bar. Triggers on_bar_close callbacks.
        Safe to call from websocket thread.
        """
        self._ensure_inst(inst)
        with self._lock_for(inst):
            bar = {
                "time": time_iso,
                "open": open_p,
                "high": high_p,
                "low": low_p,
                "close": close_p,
                "volume": volume
            }
            self._bars[inst].append(bar)
            self.bars_closed += 1

        # call callbacks outside lock to avoid deadlocks
        for cb in list(self._on_bar_close_callbacks):
            try:
                cb(inst, bar)
            except Exception:
                pass

        return bar

    def append_tick(self, inst: str, timestamp: datetime, price: float, volume: float):
        """
        Aggregate a tick into the current minute bar.
        Detects minute boundary roll-over and triggers on_bar_close.
        """
        self._ensure_inst(inst)
        ts_min = timestamp.replace(second=0, microsecond=0)
        time_iso = ts_min.strftime(ISOFMT)

        closed_bar = None
        with self._lock_for(inst):
            bars = self._bars[inst]
            if not bars or bars[-1]["time"] != time_iso:
                # Previous bar has closed
                if bars:
                    closed_bar = dict(bars[-1])
                    self.bars_closed += 1
                # start a new bar
                bar = {"time": time_iso, "open": price, "high": price, "low": price, "close": price, "volume": volume}
                bars.append(bar)
                self.bars_received += 1
            else:
                # update existing in-progress bar
                bar = bars[-1]
                bar["high"] = max(bar["high"], price)
                bar["low"] = min(bar["low"], price)
                bar["close"] = price
                # volume: track maximum observed or updated value, avoiding tick-multiplied inflation
                bar["volume"] = max(bar.get("volume", 0), volume)
                self.bars_received += 1

        # notify callbacks when previous minute bar closed
        if closed_bar:
            for cb in list(self._on_bar_close_callbacks):
                try:
                    cb(inst, closed_bar)
                except Exception:
                    pass

    def update(
        self,
        instrument: str,
        price: float,
        high: float,
        low: float,
        close: float,
        volume: float,
        time_iso: Optional[str] = None,
        open_p: Optional[float] = None
    ):
        """
        Ingest incoming feed update. If time_iso is passed, updates or appends bar.
        """
        if time_iso:
            self._ensure_inst(instrument)
            closed_bar = None
            with self._lock_for(instrument):
                bars = self._bars[instrument]
                if bars and bars[-1]["time"] == time_iso:
                    # Update in-progress bar with latest high, low, close, volume
                    bars[-1]["high"] = max(bars[-1]["high"], high, price)
                    bars[-1]["low"] = min(bars[-1]["low"], low, price)
                    bars[-1]["close"] = close or price
                    bars[-1]["volume"] = max(bars[-1].get("volume", 0), volume)
                else:
                    if bars:
                        closed_bar = dict(bars[-1])
                        self.bars_closed += 1
                    bar = {
                        "time": time_iso,
                        "open": open_p if open_p is not None else price,
                        "high": max(high, price),
                        "low": min(low, price),
                        "close": close or price,
                        "volume": volume
                    }
                    bars.append(bar)
                    self.bars_received += 1


            if closed_bar:
                for cb in list(self._on_bar_close_callbacks):
                    try:
                        cb(instrument, closed_bar)
                    except Exception:
                        pass
        else:
            self.append_tick(instrument, datetime.now(), price, volume)

    # ---------------------
    # Accessors & getters
    # ---------------------
    def get_last_n_bars(self, inst: str, n: int) -> List[dict]:
        """
        Return last n bars as list (oldest -> newest).
        """
        if inst not in self._bars:
            return []
        with self._lock_for(inst):
            bars = list(self._bars[inst])[-n:]
            return bars

    def get_last_bar(self, inst: str) -> Optional[dict]:
        if inst not in self._bars or not self._bars[inst]:
            return None
        with self._lock_for(inst):
            return dict(self._bars[inst][-1])

    def get_prices(self, inst: str) -> List[float]:
        return [b["close"] for b in self.get_last_n_bars(inst, self.max_len)]

    def get_opens(self, inst: str) -> List[float]:
        return [b["open"] for b in self.get_last_n_bars(inst, self.max_len)]

    def get_highs(self, inst: str) -> List[float]:
        return [b["high"] for b in self.get_last_n_bars(inst, self.max_len)]

    def get_lows(self, inst: str) -> List[float]:
        return [b["low"] for b in self.get_last_n_bars(inst, self.max_len)]

    def get_closes(self, inst: str) -> List[float]:
        return [b["close"] for b in self.get_last_n_bars(inst, self.max_len)]

    def get_volumes(self, inst: str) -> List[float]:
        return [b["volume"] for b in self.get_last_n_bars(inst, self.max_len)]

    def get_last_n_closes(self, inst: str, n: int) -> List[float]:
        return [b["close"] for b in self.get_last_n_bars(inst, n)]

    def get_bar_count(self, inst: str) -> int:
        if inst not in self._bars:
            return 0
        with self._lock_for(inst):
            return len(self._bars[inst])


    def has_enough_data(self, inst: str, min_bars: int = 30) -> bool:
        return (inst in self._bars and len(self._bars[inst]) >= min_bars)

    def active_instruments(self) -> List[str]:
        return list(self._bars.keys())

    # ---------------------
    # Callbacks / events
    # ---------------------
    def register_on_bar_close(self, cb: Callable[[str, dict], None]):
        """
        Register a callback to be called when a 1-minute bar is appended via append_ohlc_bar.
        Callback signature: func(inst_key, bar_dict)
        """
        if cb not in self._on_bar_close_callbacks:
            self._on_bar_close_callbacks.append(cb)

    def unregister_on_bar_close(self, cb: Callable[[str, dict], None]):
        if cb in self._on_bar_close_callbacks:
            self._on_bar_close_callbacks.remove(cb)

    # ---------------------
    # Alert throttling / dedupe helpers
    # ---------------------
    def can_emit_alert(self, inst: str, cooldown_seconds: int = 600) -> bool:
        """
        Return True if we are allowed to emit a new alert for `inst` (respect cooldown).
        """
        now_ts = time.time()
        paused_until = self._paused_until.get(inst)
        if paused_until and now_ts < paused_until:
            return False

        last = self.last_alert_time.get(inst)
        if last is None:
            return True
        return (now_ts - last) >= cooldown_seconds

    def mark_alert_emitted(self, inst: str):
        self.last_alert_time[inst] = time.time()

    def dedupe_alert(self, inst: str, direction: str, window_seconds: int = 600) -> bool:
        """
        Returns True if alert should be suppressed because same direction was emitted within window_seconds.
        If it returns False, we record this emission (caller should then send alert and optionally call mark_alert_emitted).
        """
        now_ts = time.time()
        last_for_dir = self._dedupe_map[inst].get(direction)
        if last_for_dir and (now_ts - last_for_dir) < window_seconds:
            return True
        # record and allow
        self._dedupe_map[inst][direction] = now_ts
        return False

    def mark_instrument_paused(self, inst: str, until_ts: float):
        """
        Pause instrument until epoch timestamp `until_ts`.
        """
        self._paused_until[inst] = until_ts

    # ---------------------
    # Persistence / snapshot
    # ---------------------
    def save_snapshot(self, path: Optional[str] = None):
        """
        Save minimal scanner state: bars (last N), last_alert_time, dedupe timestamps, paused_until.
        """
        path = path or self.snapshot_path
        if not path:
            raise ValueError("No snapshot path configured")

        data = {
            "bars": {},
            "last_alert_time": self.last_alert_time,
            "dedupe_map": self._dedupe_map,
            "paused_until": self._paused_until,
            "timestamp": _now_iso()
        }

        with self._global_lock:
            for inst, dq in self._bars.items():
                data["bars"][inst] = list(dq)

        tmp = f"{path}.tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, path)

    def load_snapshot(self, path: Optional[str] = None):
        path = path or self.snapshot_path
        if not path or not os.path.exists(path):
            return False
        with open(path, "r") as f:
            data = json.load(f)

        with self._global_lock:
            for inst, bars in data.get("bars", {}).items():
                dq = deque(bars, maxlen=self.max_len)
                self._bars[inst] = dq
            self.last_alert_time = data.get("last_alert_time", {})
            self._dedupe_map = defaultdict(dict, data.get("dedupe_map", {}))
            self._paused_until = data.get("paused_until", {})

        return True

    # ---------------------
    # Replay / testing utilities
    # ---------------------
    def replay_bars(self, inst: str, bars: List[dict], call_callbacks: bool = False):
        """
        Replay a list of bar dicts (oldest->newest). Useful for unit tests/backtests.
        """
        self.replay_mode = True
        self._ensure_inst(inst)
        with self._lock_for(inst):
            for bar in bars:
                # minimal validation
                if not all(k in bar for k in ("time", "open", "high", "low", "close", "volume")):
                    continue
                self._bars[inst].append(bar)
                self.bars_closed += 1
                if call_callbacks:
                    for cb in list(self._on_bar_close_callbacks):
                        try:
                            cb(inst, bar)
                        except Exception:
                            pass
        self.replay_mode = False

    def validate_bar_sequence(self, inst: str, max_gap_seconds: int = 90) -> List[dict]:
        """
        Return a list of gaps (as dicts) where time difference between consecutive bars > max_gap_seconds.
        """
        gaps = []
        bars = self.get_last_n_bars(inst, self.max_len)
        prev_ts = None
        for b in bars:
            try:
                ts = datetime.strptime(b["time"], ISOFMT)
            except Exception:
                continue
            if prev_ts and (ts - prev_ts).total_seconds() > max_gap_seconds:
                gaps.append({"from": prev_ts.strftime(ISOFMT), "to": ts.strftime(ISOFMT)})
            prev_ts = ts
        return gaps

    # ---------------------
    # Health & introspection
    # ---------------------
    def health_check(self) -> dict:
        """
        Basic scanner health summary.
        """
        now_ts = datetime.now()
        busy = sum(1 for k in self._bars if self._bars[k])
        last_bar_diff = {}
        for k, dq in self._bars.items():
            if dq:
                try:
                    ts = datetime.strptime(dq[-1]["time"], ISOFMT)
                    last_bar_diff[k] = (now_ts - ts).total_seconds()
                except Exception:
                    last_bar_diff[k] = None

        return {
            "instruments_tracked": len(self._bars),
            "bars_received": self.bars_received,
            "bars_closed": self.bars_closed,
            "recent_busy": busy,
            "last_bar_age_sample": dict(list(last_bar_diff.items())[:10])
        }

    # ---------------------
    # Utilities
    # ---------------------
    def get_bars_since(self, inst: str, since_iso: str) -> List[dict]:
        """
        Return bars newer than since_iso (inclusive).
        """
        out = []
        try:
            since_dt = datetime.strptime(since_iso, ISOFMT)
        except Exception:
            return out
        for b in self.get_last_n_bars(inst, self.max_len):
            try:
                ts = datetime.strptime(b["time"], ISOFMT)
                if ts >= since_dt:
                    out.append(b)
            except Exception:
                continue
        return out
