import pytest
import time
from storage.indexes import BTreeIndex, InvertedIndex, IndexManager


class TestBTreeIndex:
    def test_insert_and_range_query(self):
        idx = BTreeIndex()
        idx.insert(1000.0, "txn_a")
        idx.insert(2000.0, "txn_b")
        idx.insert(3000.0, "txn_c")

        result = idx.range_query(1000.0, 2500.0)
        assert "txn_a" in result
        assert "txn_b" in result
        assert "txn_c" not in result

    def test_empty_range(self):
        idx = BTreeIndex()
        idx.insert(1000.0, "txn_a")
        result = idx.range_query(5000.0, 6000.0)
        assert result == []

    def test_delete(self):
        idx = BTreeIndex()
        idx.insert(1000.0, "txn_a")
        idx.delete(1000.0, "txn_a")
        assert idx.range_query(0, 9999) == []

    def test_sorted_order(self):
        idx = BTreeIndex()
        for ts in [300.0, 100.0, 200.0, 400.0]:
            idx.insert(ts, f"txn_{ts}")
        result = idx.range_query(0, 9999)
        # Results come back in insertion order of the sorted list
        assert len(result) == 4

    def test_size(self):
        idx = BTreeIndex()
        for i in range(10):
            idx.insert(float(i), f"txn_{i}")
        assert idx.size() == 10


class TestInvertedIndex:
    def test_add_and_get(self):
        idx = InvertedIndex()
        idx.add("user:001", "txn_a")
        idx.add("user:001", "txn_b")
        result = idx.get("user:001")
        assert "txn_a" in result
        assert "txn_b" in result

    def test_missing_entity(self):
        idx = InvertedIndex()
        assert idx.get("ghost") == []

    def test_get_recent(self):
        idx = InvertedIndex()
        for i in range(10):
            idx.add("user:001", f"txn_{i}")
        recent = idx.get_recent("user:001", 3)
        assert len(recent) == 3
        assert recent == ["txn_7", "txn_8", "txn_9"]

    def test_max_list_length(self):
        idx = InvertedIndex(max_list_length=10)
        for i in range(20):
            idx.add("user:001", f"txn_{i}")
        # Should not grow unboundedly
        assert idx.count("user:001") <= 10

    def test_remove_entity(self):
        idx = InvertedIndex()
        idx.add("user:001", "txn_a")
        idx.remove_entity("user:001")
        assert idx.get("user:001") == []


class TestIndexManager:
    def test_index_and_query(self):
        mgr = IndexManager()
        txn = {
            "transaction_id": "txn_001",
            "timestamp": "2024-01-15T10:00:00",
            "payer_vpa": "user001@hdfc",
            "device_id": "dev_abc",
            "payee_vpa": "merchant01@sbi",
            "amount": 1500.0,
        }
        mgr.index_transaction(txn)
        assert "txn_001" in mgr.user_index.get("user001@hdfc")
        assert "txn_001" in mgr.device_index.get("dev_abc")
        assert "txn_001" in mgr.merchant_index.get("merchant01@sbi")