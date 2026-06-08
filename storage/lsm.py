"""
LSM Tree (Log-Structured Merge Tree) — simplified two-level implementation.

Write path:
    1. Write to WAL (durability)
    2. Write to Memtable (in-memory sorted dict)
    3. When Memtable exceeds size threshold → flush to SSTable on disk
    4. When too many SSTables accumulate → compact (merge into one)

Read path:
    1. Check Memtable first (most recent writes)
    2. Check SSTables newest-to-oldest (older = more likely stale)

This is the core architecture of RocksDB, LevelDB, and Cassandra.
The key insight: turn random writes into sequential writes by buffering
in memory and flushing sorted runs to disk. Sequential I/O is 10-100x
faster than random I/O on both HDDs and SSDs.

SSTable format (binary):
    Header:   [4 bytes magic][4 bytes version][8 bytes entry_count]
    Entries:  repeated [4 bytes key_len][key bytes][4 bytes val_len][val bytes][4 bytes crc32]
    Footer:   [8 bytes index_offset] — byte offset where the sparse index starts
    Index:    [4 bytes index_entry_count] then repeated [4 bytes key_len][key][8 bytes offset]
"""

import os
import json
import struct
import zlib
import time
import threading
from pathlib import Path
from typing import Optional, Iterator
from dataclasses import dataclass


MEMTABLE_SIZE_LIMIT = 1000          # flush after this many entries
COMPACTION_SSTABLE_THRESHOLD = 4    # compact when this many SSTables exist

SSTABLE_MAGIC   = b"SSTT"
SSTABLE_VERSION = 1
SSTABLE_HEADER  = 16                # magic(4) + version(4) + entry_count(8)

TOMBSTONE = "__TOMBSTONE__"         # sentinel value for deleted keys


@dataclass
class SSTableMeta:
    path: str
    min_key: str
    max_key: str
    entry_count: int
    created_at: float


class SSTableWriter:
    """
    Writes a sorted sequence of key-value pairs to an SSTable file.
    Also builds a sparse index (every Nth key) for fast lookups.
    """

    SPARSE_INDEX_INTERVAL = 16      # index every 16th key

    def __init__(self, path: str):
        self.path = path
        self._entries: list[tuple[str, str]] = []

    def add(self, key: str, value: str):
        self._entries.append((key, value))

    def write(self) -> SSTableMeta:
        """Serialize entries to disk. Returns metadata for this SSTable."""
        if not self._entries:
            raise ValueError("Cannot write empty SSTable")

        self._entries.sort(key=lambda x: x[0])

        sparse_index: list[tuple[str, int]] = []
        data_buf = bytearray()

        for i, (key, value) in enumerate(self._entries):
            offset = SSTABLE_HEADER + len(data_buf)
            if i % self.SPARSE_INDEX_INTERVAL == 0:
                sparse_index.append((key, offset))

            key_b   = key.encode("utf-8")
            val_b   = value.encode("utf-8")
            payload = key_b + val_b
            crc     = zlib.crc32(payload) & 0xFFFFFFFF

            data_buf += struct.pack(">I", len(key_b))
            data_buf += key_b
            data_buf += struct.pack(">I", len(val_b))
            data_buf += val_b
            data_buf += struct.pack(">I", crc)

        index_offset = SSTABLE_HEADER + len(data_buf)

        # Serialize sparse index
        index_buf = bytearray()
        index_buf += struct.pack(">I", len(sparse_index))
        for idx_key, idx_off in sparse_index:
            idx_key_b = idx_key.encode("utf-8")
            index_buf += struct.pack(">I", len(idx_key_b))
            index_buf += idx_key_b
            index_buf += struct.pack(">Q", idx_off)

        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "wb") as f:
            f.write(SSTABLE_MAGIC)
            f.write(struct.pack(">I", SSTABLE_VERSION))
            f.write(struct.pack(">Q", len(self._entries)))
            f.write(data_buf)
            f.write(struct.pack(">Q", index_offset))
            f.write(index_buf)

        return SSTableMeta(
            path=self.path,
            min_key=self._entries[0][0],
            max_key=self._entries[-1][0],
            entry_count=len(self._entries),
            created_at=time.time(),
        )


class SSTableReader:
    """
    Reads from a single SSTable file.
    Uses the sparse index to skip to the right region, then scans linearly.
    """

    def __init__(self, path: str):
        self.path = path
        self._sparse_index: list[tuple[str, int]] = []
        self._entry_count = 0
        self._load_index()

    def _load_index(self):
        with open(self.path, "rb") as f:
            magic = f.read(4)
            if magic != SSTABLE_MAGIC:
                raise ValueError(f"Invalid SSTable at {self.path}")
            f.read(4)  # version
            self._entry_count = struct.unpack(">Q", f.read(8))[0]

            # Jump to footer to get index offset
            f.seek(-8 - 4, 2)   # 8 bytes index_offset + 4 bytes index entry count, from end
            # Actually: footer is just the 8-byte index_offset right before index data
            # Recalculate: index_offset is stored right after data section
            # Re-read properly
            f.seek(0, 2)
            file_size = f.tell()

            # Scan from header to find index_offset (stored as last 8 bytes before index)
            # Simple approach: read entire file, parse sequentially
            f.seek(SSTABLE_HEADER)
            data_start = SSTABLE_HEADER

            # Read all entries to find where data ends
            offset = SSTABLE_HEADER
            for _ in range(self._entry_count):
                f.seek(offset)
                key_len = struct.unpack(">I", f.read(4))[0]
                f.seek(offset + 4 + key_len)
                val_len = struct.unpack(">I", f.read(4))[0]
                offset += 4 + key_len + 4 + val_len + 4  # +4 for crc

            # offset now points to where index_offset footer is
            f.seek(offset)
            index_offset = struct.unpack(">Q", f.read(8))[0]

            # Read sparse index
            f.seek(index_offset + 8)  # skip the index_offset value itself
            n_index = struct.unpack(">I", f.read(4))[0]
            for _ in range(n_index):
                key_len = struct.unpack(">I", f.read(4))[0]
                key = f.read(key_len).decode("utf-8")
                off = struct.unpack(">Q", f.read(8))[0]
                self._sparse_index.append((key, off))

    def get(self, key: str) -> Optional[str]:
        """Look up a key. Returns None if not found."""
        start_offset = SSTABLE_HEADER

        # Use sparse index to find best starting offset
        for i, (idx_key, idx_off) in enumerate(self._sparse_index):
            if idx_key > key:
                break
            start_offset = idx_off

        with open(self.path, "rb") as f:
            f.seek(start_offset)
            # Scan forward up to SPARSE_INDEX_INTERVAL entries
            for _ in range(SSTableWriter.SPARSE_INDEX_INTERVAL + 1):
                pos = f.tell()
                header = f.read(4)
                if len(header) < 4:
                    break
                key_len = struct.unpack(">I", header)[0]
                cur_key = f.read(key_len).decode("utf-8")
                val_len = struct.unpack(">I", f.read(4))[0]
                cur_val = f.read(val_len).decode("utf-8")
                f.read(4)  # crc (skip for reads)

                if cur_key == key:
                    return None if cur_val == TOMBSTONE else cur_val
                if cur_key > key:
                    break  # sorted — won't find it further

        return None

    def scan_all(self) -> Iterator[tuple[str, str]]:
        """Iterate all key-value pairs in sorted order."""
        with open(self.path, "rb") as f:
            f.seek(SSTABLE_HEADER)
            for _ in range(self._entry_count):
                header = f.read(4)
                if len(header) < 4:
                    break
                key_len = struct.unpack(">I", header)[0]
                key = f.read(key_len).decode("utf-8")
                val_len = struct.unpack(">I", f.read(4))[0]
                val = f.read(val_len).decode("utf-8")
                f.read(4)  # crc
                yield key, val


class Memtable:
    """
    In-memory write buffer. Sorted dict (Python dict preserves insertion order
    but we sort on flush). Accepts writes and deletes (tombstones).
    """

    def __init__(self, size_limit: int = MEMTABLE_SIZE_LIMIT):
        self._data: dict[str, str] = {}
        self._size_limit = size_limit

    def set(self, key: str, value: str):
        self._data[key] = value

    def delete(self, key: str):
        self._data[key] = TOMBSTONE

    def get(self, key: str) -> Optional[str]:
        val = self._data.get(key)
        if val is None:
            return None
        return None if val == TOMBSTONE else val

    def is_full(self) -> bool:
        return len(self._data) >= self._size_limit

    def sorted_items(self) -> list[tuple[str, str]]:
        return sorted(self._data.items())

    def size(self) -> int:
        return len(self._data)

    def clear(self):
        self._data.clear()


class LSMTree:
    """
    Two-level LSM tree: Memtable → L0 SSTables → compacted SSTable.

    Coordinates writes through WAL → Memtable → SSTable pipeline.
    Handles reads by checking Memtable first, then SSTables newest-to-oldest.
    Triggers compaction when SSTable count exceeds threshold.
    """

    def __init__(self, data_dir: str, memtable_limit: int = MEMTABLE_SIZE_LIMIT):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._memtable = Memtable(memtable_limit)
        self._sstables: list[SSTableMeta] = []
        self._lock = threading.Lock()
        self._sstable_counter = 0
        self._load_existing_sstables()

    def _load_existing_sstables(self):
        """On startup, discover any SSTables left from previous runs."""
        for f in sorted(self.data_dir.glob("sst_*.sst")):
            try:
                reader = SSTableReader(str(f))
                # Recover min/max key by scanning
                items = list(reader.scan_all())
                if items:
                    self._sstables.append(SSTableMeta(
                        path=str(f),
                        min_key=items[0][0],
                        max_key=items[-1][0],
                        entry_count=reader._entry_count,
                        created_at=f.stat().st_mtime,
                    ))
                    # Keep counter ahead of existing files
                    num = int(f.stem.split("_")[1])
                    self._sstable_counter = max(self._sstable_counter, num + 1)
            except Exception:
                pass  # corrupt SSTable — skip

        # Sort by creation time, oldest first
        self._sstables.sort(key=lambda m: m.created_at)

    def _flush_memtable(self):
        """Write Memtable contents to a new SSTable. Called when Memtable is full."""
        if self._memtable.size() == 0:
            return

        path = str(self.data_dir / f"sst_{self._sstable_counter:06d}.sst")
        self._sstable_counter += 1

        writer = SSTableWriter(path)
        for key, value in self._memtable.sorted_items():
            writer.add(key, value)

        meta = writer.write()
        self._sstables.append(meta)
        self._memtable.clear()

        if len(self._sstables) >= COMPACTION_SSTABLE_THRESHOLD:
            self._compact()

    def _compact(self):
        """
        Merge all existing SSTables into one.
        Newer entries win over older ones for the same key.
        Tombstones are dropped during compaction (GC).
        """
        if len(self._sstables) < 2:
            return

        # Merge: iterate all SSTables oldest-to-newest, newer overwrites older
        merged: dict[str, str] = {}
        for meta in self._sstables:
            reader = SSTableReader(meta.path)
            for key, value in reader.scan_all():
                merged[key] = value  # newer SSTable overwrites older

        # Remove tombstoned keys
        live = {k: v for k, v in merged.items() if v != TOMBSTONE}

        if not live:
            # Nothing left after removing tombstones — delete old SSTables
            for meta in self._sstables:
                try:
                    os.remove(meta.path)
                except OSError:
                    pass
            self._sstables.clear()
            return

        # Write compacted SSTable
        path = str(self.data_dir / f"sst_{self._sstable_counter:06d}.sst")
        self._sstable_counter += 1

        writer = SSTableWriter(path)
        for key, value in sorted(live.items()):
            writer.add(key, value)
        meta = writer.write()

        # Delete old SSTables
        for old_meta in self._sstables:
            try:
                os.remove(old_meta.path)
            except OSError:
                pass

        self._sstables = [meta]

    def set(self, key: str, value: str):
        with self._lock:
            self._memtable.set(key, value)
            if self._memtable.is_full():
                self._flush_memtable()

    def delete(self, key: str):
        with self._lock:
            self._memtable.delete(key)
            if self._memtable.is_full():
                self._flush_memtable()

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            # Check Memtable first
            val = self._memtable.get(key)
            if val is not None:
                return val
            # Check if key is tombstoned in memtable
            if key in self._memtable._data:
                return None  # deleted

            # Check SSTables newest-to-oldest
            for meta in reversed(self._sstables):
                reader = SSTableReader(meta.path)
                val = reader.get(key)
                if val is not None:
                    return val

        return None

    def flush(self):
        """Force flush Memtable to disk. Call before shutdown."""
        with self._lock:
            self._flush_memtable()

    def stats(self) -> dict:
        return {
            "memtable_size": self._memtable.size(),
            "memtable_limit": self._memtable._size_limit,
            "sstable_count": len(self._sstables),
            "sstables": [
                {"path": m.path, "entries": m.entry_count,
                 "min_key": m.min_key, "max_key": m.max_key}
                for m in self._sstables
            ],
        }