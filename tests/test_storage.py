import pytest
import tempfile
from storage.hashmap import HashMap
from storage.wal import WriteAheadLog


class TestHashMap:
    def test_set_and_get(self):
        hm = HashMap()
        hm.set("key1", "value1")
        assert hm.get("key1") == "value1"

    def test_missing_key_returns_none(self):
        hm = HashMap()
        assert hm.get("nonexistent") is None

    def test_update_existing_key(self):
        hm = HashMap()
        hm.set("k", "v1")
        hm.set("k", "v2")
        assert hm.get("k") == "v2"
        assert len(hm) == 1

    def test_delete(self):
        hm = HashMap()
        hm.set("k", "v")
        assert hm.delete("k") == True
        assert hm.get("k") is None
        assert len(hm) == 0

    def test_delete_nonexistent(self):
        hm = HashMap()
        assert hm.delete("ghost") == False

    def test_count(self):
        hm = HashMap()
        for i in range(10):
            hm.set(f"key{i}", f"val{i}")
        assert len(hm) == 10

    def test_resize_under_load(self):
        """Insert enough entries to trigger multiple resizes."""
        hm = HashMap(initial_capacity=8)
        for i in range(200):
            hm.set(f"key_{i:05d}", f"value_{i}")
        for i in range(200):
            assert hm.get(f"key_{i:05d}") == f"value_{i}"

    def test_dict_style_access(self):
        hm = HashMap()
        hm["foo"] = "bar"
        assert hm["foo"] == "bar"
        del hm["foo"]
        assert "foo" not in hm


class TestWAL:
    def test_write_and_replay(self):
        with tempfile.TemporaryDirectory() as d:
            wal = WriteAheadLog(f"{d}/test.wal")
            lsn1 = wal.write({"type": "txn", "id": "abc"})
            lsn2 = wal.write({"type": "txn", "id": "def"})

            records = list(wal.replay())
            assert len(records) == 2
            assert records[0].lsn == lsn1
            assert records[0].payload["id"] == "abc"
            assert records[1].lsn == lsn2

    def test_replay_from_lsn(self):
        with tempfile.TemporaryDirectory() as d:
            wal = WriteAheadLog(f"{d}/test.wal")
            for i in range(5):
                wal.write({"seq": i})
            records = list(wal.replay(start_lsn=2))
            assert all(r.lsn > 2 for r in records)
            assert len(records) == 3

    def test_lsn_monotonic(self):
        with tempfile.TemporaryDirectory() as d:
            wal = WriteAheadLog(f"{d}/test.wal")
            lsns = [wal.write({"i": i}) for i in range(10)]
            assert lsns == list(range(1, 11))

    def test_recovery_after_crash(self):
        """Simulate crash by truncating WAL mid-record."""
        import os
        with tempfile.TemporaryDirectory() as d:
            path = f"{d}/test.wal"
            wal = WriteAheadLog(path)
            wal.write({"id": "good_record"})
            good_size = os.path.getsize(path)

            wal.write({"id": "partial_record"})
            # Truncate to simulate crash mid-write
            with open(path, "r+b") as f:
                f.truncate(good_size + 6)  # partial write

            # Should recover cleanly — yields only the first record
            wal2 = WriteAheadLog(path)
            records = list(wal2.replay())
            assert len(records) == 1
            assert records[0].payload["id"] == "good_record"