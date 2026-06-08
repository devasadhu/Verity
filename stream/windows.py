"""
Sliding window aggregator.

Maintains per-entity (user, device, merchant) statistics over rolling
time windows: 1 minute, 5 minutes, 1 hour, 24 hours.

Each window tracks:
    - transaction count
    - amount sum, mean, std, max
    - unique cities seen
    - unique devices seen (for user windows)
    - timestamps (for velocity computation)

Design: circular buffer of (timestamp, amount) pairs per entity.
On each query, expired entries are evicted before computing stats.
This is O(k) per query where k is entries in the window — acceptable
because fraud windows are short and most entities have low velocity.

For truly high-scale systems (millions of entities, thousands of TPS),
you'd use approximate structures (Count-Min Sketch for counts,
t-digest for quantiles). We use exact computation here because the
dataset size is manageable and exact numbers matter for features.
"""

import time
import math
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


WINDOW_SIZES = {
    "1m":  60,
    "5m":  300,
    "1h":  3600,
    "24h": 86400,
}


@dataclass
class WindowEntry:
    timestamp: float
    amount: float
    city: str = ""
    device_id: str = ""


@dataclass
class WindowStats:
    window: str
    count: int
    amount_sum: float
    amount_mean: float
    amount_std: float
    amount_max: float
    unique_cities: int
    unique_devices: int
    velocity_per_minute: float   # count / window_minutes


class EntityWindow:
    """
    Rolling time window for a single entity (one user, device, or merchant).
    Maintains a buffer of recent entries and evicts expired ones on access.
    """

    def __init__(self, max_window_seconds: int = 86400):
        self._entries: list[WindowEntry] = []
        self._max_window = max_window_seconds
        self._lock = threading.Lock()

    def add(self, amount: float, city: str = "", device_id: str = "", ts: Optional[float] = None):
        if ts is None:
            ts = time.time()
        entry = WindowEntry(timestamp=ts, amount=amount, city=city, device_id=device_id)
        with self._lock:
            self._entries.append(entry)
            self._evict(ts)

    def _evict(self, now: float):
        """Remove entries older than max_window_seconds."""
        cutoff = now - self._max_window
        # Binary search would be faster but entries are nearly always in order
        while self._entries and self._entries[0].timestamp < cutoff:
            self._entries.pop(0)

    def stats(self, window_name: str, now: Optional[float] = None) -> WindowStats:
        """Compute stats for a named window (1m, 5m, 1h, 24h)."""
        if now is None:
            now = time.time()
        window_seconds = WINDOW_SIZES[window_name]
        cutoff = now - window_seconds

        with self._lock:
            self._evict(now)
            relevant = [e for e in self._entries if e.timestamp >= cutoff]

        if not relevant:
            return WindowStats(
                window=window_name, count=0,
                amount_sum=0, amount_mean=0, amount_std=0, amount_max=0,
                unique_cities=0, unique_devices=0, velocity_per_minute=0,
            )

        amounts = [e.amount for e in relevant]
        count = len(amounts)
        amt_sum = sum(amounts)
        amt_mean = amt_sum / count
        amt_std = math.sqrt(sum((a - amt_mean) ** 2 for a in amounts) / count)
        amt_max = max(amounts)

        unique_cities   = len({e.city for e in relevant if e.city})
        unique_devices  = len({e.device_id for e in relevant if e.device_id})
        velocity        = count / (window_seconds / 60)

        return WindowStats(
            window=window_name,
            count=count,
            amount_sum=round(amt_sum, 2),
            amount_mean=round(amt_mean, 2),
            amount_std=round(amt_std, 2),
            amount_max=round(amt_max, 2),
            unique_cities=unique_cities,
            unique_devices=unique_devices,
            velocity_per_minute=round(velocity, 4),
        )

    def all_stats(self, now: Optional[float] = None) -> dict[str, WindowStats]:
        """Return stats for all four windows."""
        if now is None:
            now = time.time()
        return {name: self.stats(name, now) for name in WINDOW_SIZES}

    def total_count(self) -> int:
        with self._lock:
            return len(self._entries)


class SlidingWindowAggregator:
    """
    Maintains EntityWindows for all users, devices, and merchants seen.
    Central entry point for the stream processing layer.

    Usage:
        agg = SlidingWindowAggregator()
        agg.record(txn)
        features = agg.get_features(txn)
    """

    def __init__(self):
        self._user_windows:     dict[str, EntityWindow] = defaultdict(EntityWindow)
        self._device_windows:   dict[str, EntityWindow] = defaultdict(EntityWindow)
        self._merchant_windows: dict[str, EntityWindow] = defaultdict(EntityWindow)
        # Count-Min Sketches for approximate global counts
        from stream.sketch import CountMinSketch
        self._user_sketch     = CountMinSketch(epsilon=0.005, delta=0.001)
        self._device_sketch   = CountMinSketch(epsilon=0.005, delta=0.001)
        self._merchant_sketch = CountMinSketch(epsilon=0.005, delta=0.001)
        self._known_users:    set[str] = set()
        self._known_devices:  set[str] = set()
        self._known_merchants: set[str] = set()

    def record(self, txn: dict, ts: Optional[float] = None):
        """Record a transaction into all relevant windows."""
        if ts is None:
            ts = time.time()

        amount   = float(txn.get("amount", 0))
        city     = txn.get("payer_city", "")
        device   = txn.get("device_id", "")
        user     = txn.get("payer_vpa", "")
        merchant = txn.get("payee_vpa", "")

        if user:
            self._user_windows[user].add(amount, city, device, ts)
            self._user_sketch.update(user)
            self._known_users.add(user)

        if device:
            self._device_windows[device].add(amount, city, device, ts)
            self._device_sketch.update(device)
            self._known_devices.add(device)

        if merchant:
            self._merchant_windows[merchant].add(amount, city, device, ts)
            self._merchant_sketch.update(merchant)
            self._known_merchants.add(merchant)

    def get_user_stats(self, user_id: str, now: Optional[float] = None) -> dict:
        return {
            k: vars(v)
            for k, v in self._user_windows[user_id].all_stats(now).items()
        }

    def get_device_stats(self, device_id: str, now: Optional[float] = None) -> dict:
        return {
            k: vars(v)
            for k, v in self._device_windows[device_id].all_stats(now).items()
        }

    def get_features(self, txn: dict, now: Optional[float] = None) -> dict:
        """
        Extract ML-ready features for a transaction.
        This is what gets passed to the ML scoring layer.
        """
        if now is None:
            now = time.time()

        user     = txn.get("payer_vpa", "")
        device   = txn.get("device_id", "")
        merchant = txn.get("payee_vpa", "")
        amount   = float(txn.get("amount", 0))

        u_stats  = self._user_windows[user].all_stats(now)     if user     else {}
        d_stats  = self._device_windows[device].all_stats(now) if device   else {}
        m_stats  = self._merchant_windows[merchant].all_stats(now) if merchant else {}

        def s(stats, window, field, default=0):
            w = stats.get(window)
            return getattr(w, field, default) if w else default

        # Amount deviation from user's recent mean
        user_mean_1h = s(u_stats, "1h", "amount_mean", 0)
        user_std_1h  = s(u_stats, "1h", "amount_std", 1)
        amount_zscore = (amount - user_mean_1h) / max(user_std_1h, 1)

        return {
            # User velocity features
            "user_txn_count_1m":   s(u_stats, "1m",  "count"),
            "user_txn_count_5m":   s(u_stats, "5m",  "count"),
            "user_txn_count_1h":   s(u_stats, "1h",  "count"),
            "user_txn_count_24h":  s(u_stats, "24h", "count"),
            # User amount features
            "user_amount_mean_1h":  user_mean_1h,
            "user_amount_std_1h":   user_std_1h,
            "user_amount_max_24h":  s(u_stats, "24h", "amount_max"),
            "amount_zscore_1h":     round(amount_zscore, 4),
            # User geo/device features
            "user_unique_cities_1h":   s(u_stats, "1h",  "unique_cities"),
            "user_unique_devices_24h": s(u_stats, "24h", "unique_devices"),
            # Device features
            "device_txn_count_1h":  s(d_stats, "1h",  "count"),
            "device_txn_count_24h": s(d_stats, "24h", "count"),
            # Merchant features
            "merchant_txn_count_1h":  s(m_stats, "1h",  "count"),
            "merchant_amount_mean_1h": s(m_stats, "1h",  "amount_mean"),
            # Approximate global counts via sketch
            "user_global_freq":     self._user_sketch.query(user)     if user     else 0,
            "device_global_freq":   self._device_sketch.query(device) if device   else 0,
            "merchant_global_freq": self._merchant_sketch.query(merchant) if merchant else 0,
        }

    def stats(self) -> dict:
        return {
            "tracked_users":     len(self._user_windows),
            "tracked_devices":   len(self._device_windows),
            "tracked_merchants": len(self._merchant_windows),
            "sketch_user":       repr(self._user_sketch),
            "sketch_device":     repr(self._device_sketch),
        }