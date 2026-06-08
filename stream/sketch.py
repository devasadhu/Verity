"""
Count-Min Sketch — probabilistic frequency estimation.

Problem: counting exact transaction frequencies per user/device/merchant
requires O(n) space. With millions of entities this is expensive.

Count-Min Sketch gives approximate counts using O(w * d) space where
w * d is much smaller than n, with a controllable error bound.

Structure: d hash functions, each mapping keys to one of w counters.
    - Update: increment counter[i][hash_i(key)] for each row i
    - Query:  return min(counter[i][hash_i(key)]) across all rows

The minimum is an upper bound on the true count — it overestimates
due to hash collisions, never underestimates.

Error guarantee: with probability 1 - delta, the estimate is within
(true_count + epsilon * total_count). Setting w = ceil(e/epsilon) and
d = ceil(ln(1/delta)) gives the desired accuracy.

Used in production at: Twitter (trending topics), Google (network monitoring),
Akamai (heavy hitter detection), and most fraud detection systems for
velocity counting at scale.
"""

import math
import hashlib
import struct
from typing import Sequence


def _hash_family(key: str, seed: int, width: int) -> int:
    """
    One hash function from our family. Uses SHA-256 with a seed prefix
    to get independent hash functions cheaply.
    """
    data = struct.pack(">I", seed) + key.encode("utf-8")
    digest = hashlib.sha256(data).digest()
    # Take first 8 bytes as uint64, mod width
    return struct.unpack(">Q", digest[:8])[0] % width


class CountMinSketch:
    """
    Count-Min Sketch for approximate frequency counting.

    Parameters:
        epsilon:  max relative error (e.g. 0.01 = 1% error)
        delta:    failure probability (e.g. 0.001 = 0.1% chance of exceeding error)

    These determine width and depth automatically:
        width = ceil(e / epsilon)   ← controls error magnitude
        depth = ceil(ln(1/delta))   ← controls confidence
    """

    def __init__(self, epsilon: float = 0.01, delta: float = 0.001):
        self.epsilon = epsilon
        self.delta   = delta
        self.width   = math.ceil(math.e / epsilon)
        self.depth   = math.ceil(math.log(1.0 / delta))
        self._table  = [[0] * self.width for _ in range(self.depth)]
        self._total  = 0

    def update(self, key: str, count: int = 1):
        """Increment frequency of key by count."""
        for row in range(self.depth):
            col = _hash_family(key, row, self.width)
            self._table[row][col] += count
        self._total += count

    def query(self, key: str) -> int:
        """Return estimated frequency of key. Always >= true frequency."""
        return min(
            self._table[row][_hash_family(key, row, self.width)]
            for row in range(self.depth)
        )

    def total(self) -> int:
        """Total number of events recorded."""
        return self._total

    def heavy_hitters(self, threshold_fraction: float) -> list[str]:
        """
        Cannot directly enumerate heavy hitters from a sketch alone
        (we don't store keys). In practice, combine with a separate
        key set and filter by query() > threshold.
        This method documents the limitation explicitly.
        """
        raise NotImplementedError(
            "Count-Min Sketch cannot enumerate keys — "
            "maintain a separate key set and filter with query(). "
            "See stream/windows.py for the pattern."
        )

    def error_bound(self) -> float:
        """Maximum additive error as a fraction of total count."""
        return self.epsilon * self._total

    def memory_bytes(self) -> int:
        """Approximate memory usage of the table."""
        return self.width * self.depth * 8  # 8 bytes per int64

    def reset(self):
        self._table = [[0] * self.width for _ in range(self.depth)]
        self._total = 0

    def __repr__(self) -> str:
        return (
            f"CountMinSketch(width={self.width}, depth={self.depth}, "
            f"total={self._total}, memory={self.memory_bytes()/1024:.1f}KB)"
        )