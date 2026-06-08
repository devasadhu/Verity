import pytest
import tempfile
from storage.lsm import LSMTree, Memtable, SSTableWriter, SSTableReader, TOMBSTONE


class TestMemtable:
    def test_set_and_get(self):
        m = Memtable()
        m.set("k1", "v1")
        assert m.get("k1") == "v1"

    def test_delete_returns_none(self):
        m = Memtable()
        m.set("k1", "v1")
        m.delete("k1")
        assert m.get("k1") is None

    def test_tombstone_in_data(self):
        m = Memtable()
        m.delete("ghost")
        assert "ghost" in m._data
        assert m._data["ghost"] == TOMBSTONE

    def test_sorted_items(self):
        m = Memtable()
        m.set("c", "3")
        m.set("a", "1")
        m.set("b", "2")
        keys = [k for k, v in m.sorted_items()]
        assert keys == ["a", "b", "c"]

    def test_size_limit(self):
        m = Memtable(size_limit=5)
        for i in range(4):
            m.set(f"k{i}", "v")
        assert not m.is_full()
        m.set("k4", "v")
        assert m.is_full()


class TestSSTable:
    def test_write_and_read(self):
        with tempfile.TemporaryDirectory() as d:
            path = f"{d}/test.sst"
            writer = SSTableWriter(path)
            writer.add("user:001", '{"amount": 1500}')
            writer.add("user:002", '{"amount": 2000}')
            writer.add("user:003", '{"amount": 500}')
            meta = writer.write()

            assert meta.min_key == "user:001"
            assert meta.max_key == "user:003"
            assert meta.entry_count == 3

            reader = SSTableReader(path)
            assert reader.get("user:001") == '{"amount": 1500}'
            assert reader.get("user:002") == '{"amount": 2000}'
            assert reader.get("user:999") is None

    def test_scan_all_sorted(self):
        with tempfile.TemporaryDirectory() as d:
            path = f"{d}/test.sst"
            writer = SSTableWriter(path)
            for key in ["zoo", "apple", "mango", "banana"]:
                writer.add(key, key + "_val")
            writer.write()

            reader = SSTableReader(path)
            items = list(reader.scan_all())
            keys = [k for k, v in items]
            assert keys == sorted(keys)


class TestLSMTree:
    def test_basic_set_get(self):
        with tempfile.TemporaryDirectory() as d:
            lsm = LSMTree(d, memtable_limit=100)
            lsm.set("txn:001", '{"amount": 1500}')
            assert lsm.get("txn:001") == '{"amount": 1500}'

    def test_get_missing_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            lsm = LSMTree(d)
            assert lsm.get("nonexistent") is None

    def test_flush_to_sstable(self):
        with tempfile.TemporaryDirectory() as d:
            lsm = LSMTree(d, memtable_limit=5)
            for i in range(6):
                lsm.set(f"key:{i:03d}", f"val:{i}")
            # After 6 inserts with limit=5, memtable should have flushed
            assert lsm.stats()["sstable_count"] >= 1

    def test_read_after_flush(self):
        with tempfile.TemporaryDirectory() as d:
            lsm = LSMTree(d, memtable_limit=3)
            for i in range(10):
                lsm.set(f"key:{i:03d}", f"val:{i}")
            # All keys should still be readable
            for i in range(10):
                assert lsm.get(f"key:{i:03d}") == f"val:{i}"

    def test_delete(self):
        with tempfile.TemporaryDirectory() as d:
            lsm = LSMTree(d)
            lsm.set("k", "v")
            lsm.delete("k")
            assert lsm.get("k") is None

    def test_compaction_reduces_sstable_count(self):
        with tempfile.TemporaryDirectory() as d:
            lsm = LSMTree(d, memtable_limit=3)
            for i in range(20):
                lsm.set(f"key:{i:03d}", f"val:{i}")
            # After compaction, should be fewer SSTables than raw flushes
            assert lsm.stats()["sstable_count"] <= 4