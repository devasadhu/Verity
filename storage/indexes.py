"""
Secondary indexes for fast lookups on non-primary fields.

BTreeIndex:
    On transaction timestamp — supports range queries like
    "give me all transactions between T1 and T2".
    Implemented as a sorted list with bisect (models B-tree range behavior).
    A real B-tree would keep data in fixed-size disk pages; this captures
    the interface and algorithmic behavior without the page management.

InvertedIndex:
    Maps user_id → list of transaction_ids.
    Used for "give me all transactions for user X" queries.
    Same structure used in search engines (term → document list).
    Stored in the LSM tree for durability; in-memory cache for hot users.
"""

import bisect
import json
import threading
from typing import Optional
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# B-Tree Index (timestamp → transaction_id range queries)
# ---------------------------------------------------------------------------

@dataclass
class BTreeEntry:
    key: float          # timestamp as Unix float
    value: str          # transaction_id

    def __lt__(self, other: "BTreeEntry") -> bool:
        return self.key < other.key

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BTreeEntry):
            return False
        return self.key == other.key and self.value == other.value


class BTreeIndex:
    """
    Sorted index on a numeric key (timestamp).
    Supports O(log n) point lookup and O(log n + k) range queries
    where k is the number of results.

    Backed by a sorted list + bisect — same asymptotic behavior as a
    B-tree for in-memory data. The key interview point: a real B-tree
    organizes data into fixed-size disk pages (typically 4KB or 16KB)
    to minimize page faults during range scans. This implementation
    captures the interface; the page management is what makes on-disk
    B-trees complex.
    """

    def __init__(self):
        self._entries: list[BTreeEntry] = []
        self._lock = threading.RLock()

    def insert(self, timestamp: float, transaction_id: str):
        """Insert a (timestamp, txn_id) pair. O(log n) find + O(n) insert."""
        entry = BTreeEntry(key=timestamp, value=transaction_id)
        with self._lock:
            bisect.insort(self._entries, entry)

    def delete(self, timestamp: float, transaction_id: str):
        """Remove a specific entry. O(log n) find + O(n) delete."""
        entry = BTreeEntry(key=timestamp, value=transaction_id)
        with self._lock:
            idx = bisect.bisect_left(self._entries, entry)
            while idx < len(self._entries) and self._entries[idx].key == timestamp:
                if self._entries[idx].value == transaction_id:
                    self._entries.pop(idx)
                    return
                idx += 1

    def range_query(
        self,
        start_ts: float,
        end_ts: float,
    ) -> list[str]:
        """
        Return all transaction_ids with timestamp in [start_ts, end_ts].
        O(log n) to find start + O(k) to collect results.
        """
        with self._lock:
            lo = bisect.bisect_left(self._entries, BTreeEntry(start_ts, ""))
            hi = bisect.bisect_right(self._entries, BTreeEntry(end_ts, "\xff" * 100))
            return [e.value for e in self._entries[lo:hi]]

    def point_query(self, timestamp: float) -> list[str]:
        """Return all transaction_ids at exactly this timestamp."""
        return self.range_query(timestamp, timestamp)

    def size(self) -> int:
        return len(self._entries)

    def min_key(self) -> Optional[float]:
        if not self._entries:
            return None
        return self._entries[0].key

    def max_key(self) -> Optional[float]:
        if not self._entries:
            return None
        return self._entries[-1].key


# ---------------------------------------------------------------------------
# Inverted Index (user_id → [transaction_ids])
# ---------------------------------------------------------------------------

class InvertedIndex:
    """
    Maps an entity (user_id, device_id, merchant_id) to a list of
    transaction_ids associated with that entity.

    Used for:
        - "Get all transactions for user X in last 24h"
        - "How many transactions came from device Y?"
        - "What's merchant Z's transaction history?"

    Same data structure used in full-text search engines (word → document list).
    In Elasticsearch, these posting lists are stored in Lucene segment files.
    Here we keep them in memory with optional persistence via the LSM tree.

    Space tradeoff: posting lists can get large for active users.
    Real systems cap list length and offload older entries to cold storage.
    """

    def __init__(self, max_list_length: int = 10_000):
        self._index: dict[str, list[str]] = {}
        self._lock = threading.RLock()
        self._max_list_length = max_list_length

    def add(self, entity_id: str, transaction_id: str):
        """
        Add transaction_id to entity's posting list.
        If list exceeds max_list_length, oldest entries are dropped (sliding window).
        """
        with self._lock:
            if entity_id not in self._index:
                self._index[entity_id] = []
            lst = self._index[entity_id]
            lst.append(transaction_id)
            if len(lst) > self._max_list_length:
                # Drop oldest half — amortized O(1) instead of O(n) every insert
                self._index[entity_id] = lst[self._max_list_length // 2:]

    def get(self, entity_id: str) -> list[str]:
        """Return all transaction_ids for this entity. Returns [] if not found."""
        with self._lock:
            return list(self._index.get(entity_id, []))

    def get_recent(self, entity_id: str, n: int) -> list[str]:
        """Return last n transaction_ids for this entity."""
        with self._lock:
            lst = self._index.get(entity_id, [])
            return list(lst[-n:])

    def count(self, entity_id: str) -> int:
        """Number of transactions indexed for this entity."""
        with self._lock:
            return len(self._index.get(entity_id, []))

    def remove_entity(self, entity_id: str):
        """Remove all entries for an entity (e.g. account closure)."""
        with self._lock:
            self._index.pop(entity_id, None)

    def entity_count(self) -> int:
        """Total number of distinct entities indexed."""
        return len(self._index)

    def serialize(self) -> str:
        """Serialize to JSON for persistence in LSM tree."""
        with self._lock:
            return json.dumps(self._index)

    @classmethod
    def deserialize(cls, data: str, max_list_length: int = 10_000) -> "InvertedIndex":
        idx = cls(max_list_length)
        idx._index = json.loads(data)
        return idx


# ---------------------------------------------------------------------------
# Composite index manager — used by the rest of the system
# ---------------------------------------------------------------------------

class IndexManager:
    """
    Single entry point for all index operations.
    Call index_transaction() on every new transaction.
    Then use the individual indexes for queries.
    """

    def __init__(self):
        self.timestamp_index = BTreeIndex()
        self.user_index      = InvertedIndex()
        self.device_index    = InvertedIndex()
        self.merchant_index  = InvertedIndex()

    def index_transaction(self, txn: dict):
        """
        Index a transaction dict. Expected fields:
            transaction_id, timestamp (ISO or unix float),
            payer_vpa, device_id, payee_vpa
        """
        txn_id = txn.get("transaction_id", "")

        # Timestamp index
        ts = txn.get("timestamp_unix")
        if ts is None:
            # Parse ISO timestamp if unix not provided
            from datetime import datetime
            ts_str = txn.get("timestamp", "")
            try:
                ts = datetime.fromisoformat(ts_str).timestamp()
            except (ValueError, TypeError):
                ts = 0.0
        self.timestamp_index.insert(ts, txn_id)

        # Entity indexes
        if payer := txn.get("payer_vpa"):
            self.user_index.add(payer, txn_id)
        if device := txn.get("device_id"):
            self.device_index.add(device, txn_id)
        if merchant := txn.get("payee_vpa"):
            self.merchant_index.add(merchant, txn_id)

    def stats(self) -> dict:
        return {
            "timestamp_index_size": self.timestamp_index.size(),
            "indexed_users":        self.user_index.entity_count(),
            "indexed_devices":      self.device_index.entity_count(),
            "indexed_merchants":    self.merchant_index.entity_count(),
        }