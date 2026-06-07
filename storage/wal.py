"""
Write-Ahead Log (WAL).

Every transaction is written here before being acknowledged.
On restart, replay the WAL to recover in-memory state.

This is exactly how PostgreSQL's WAL and Redis AOF (Append-Only File) work.

Record format (identical to message_queue.py — reusable pattern):
  [4 bytes: payload length big-endian]
  [N bytes: JSON payload]
  [4 bytes: CRC32 of payload]
  [8 bytes: sequence number big-endian]

The sequence number is the key difference from the message queue —
WAL records are identified by LSN (Log Sequence Number), not byte offset.
"""

import os
import struct
import zlib
import json
import threading
from pathlib import Path
from typing import Iterator, Optional
from dataclasses import dataclass


WAL_MAGIC   = b"WALY"
WAL_VERSION = 1
WAL_HEADER  = 8   # magic(4) + version(4)
WAL_RECORD_OVERHEAD = 16  # length(4) + crc(4) + lsn(8)


@dataclass
class WALRecord:
    lsn: int          # Log Sequence Number — monotonically increasing
    payload: dict     # Deserialized record data
    crc: int


class WALCorruptionError(Exception):
    pass


class WriteAheadLog:
    """
    Append-only write-ahead log with CRC32 integrity checking.

    Recovery pattern:
        wal = WriteAheadLog("data/wal/transactions.wal")
        for record in wal.replay():
            apply_to_memtable(record)  # rebuild in-memory state
    """

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._lsn = 0

        if self.path.exists():
            self._lsn = self._scan_max_lsn()
        else:
            self._write_header()

    def _write_header(self):
        with open(self.path, "wb") as f:
            f.write(WAL_MAGIC)
            f.write(struct.pack(">I", WAL_VERSION))

    def _scan_max_lsn(self) -> int:
        """Scan entire WAL to find highest LSN. Called once on startup."""
        max_lsn = 0
        try:
            for record in self.replay():
                if record.lsn > max_lsn:
                    max_lsn = record.lsn
        except (WALCorruptionError, EOFError, OSError):
            pass  # partial record at end is okay — recovery handles it
        return max_lsn

    def write(self, data: dict) -> int:
        """
        Append a record to the WAL.
        Fsync before returning — guarantees durability.
        Returns the LSN of the written record.
        """
        with self._lock:
            self._lsn += 1
            lsn = self._lsn

            payload = json.dumps(data, default=str).encode("utf-8")
            crc = zlib.crc32(payload) & 0xFFFFFFFF

            record = (
                struct.pack(">I", len(payload)) +
                payload +
                struct.pack(">I", crc) +
                struct.pack(">Q", lsn)
            )

            with open(self.path, "ab") as f:
                f.write(record)
                f.flush()
                os.fsync(f.fileno())

        return lsn

    def replay(
        self,
        start_lsn: int = 0
    ) -> Iterator[WALRecord]:
        """
        Iterate over all records with lsn > start_lsn.
        Validates CRC for each record.
        Stops cleanly at truncated records (crash recovery scenario).
        """
        with open(self.path, "rb") as f:
            # Verify header
            magic = f.read(4)
            if magic != WAL_MAGIC:
                raise WALCorruptionError(f"Invalid WAL magic: {magic}")
            f.read(4)  # version — skip for now

            while True:
                len_bytes = f.read(4)
                if len(len_bytes) < 4:
                    break  # clean EOF or truncation

                payload_len = struct.unpack(">I", len_bytes)[0]
                payload = f.read(payload_len)
                crc_bytes = f.read(4)
                lsn_bytes = f.read(8)

                if len(payload) < payload_len or len(crc_bytes) < 4 or len(lsn_bytes) < 8:
                    # Truncated record — stop here (crash during write)
                    break

                stored_crc = struct.unpack(">I", crc_bytes)[0]
                lsn = struct.unpack(">Q", lsn_bytes)[0]
                computed_crc = zlib.crc32(payload) & 0xFFFFFFFF

                if stored_crc != computed_crc:
                    raise WALCorruptionError(
                        f"CRC mismatch at LSN {lsn}: "
                        f"stored={stored_crc:#010x}, computed={computed_crc:#010x}"
                    )

                if lsn > start_lsn:
                    yield WALRecord(
                        lsn=lsn,
                        payload=json.loads(payload.decode("utf-8")),
                        crc=stored_crc,
                    )

    def checkpoint(self, checkpoint_lsn: int, checkpoint_path: Optional[str] = None):
        """
        Mark that all records up to checkpoint_lsn have been persisted
        to stable storage (flushed Memtable → SSTable).

        In a full implementation this would allow truncating the WAL
        before checkpoint_lsn. Here we just record the checkpoint LSN.
        """
        ckpt = {
            "checkpoint_lsn": checkpoint_lsn,
            "type": "_checkpoint",
        }
        self.write(ckpt)

    def current_lsn(self) -> int:
        return self._lsn

    def stats(self) -> dict:
        size = self.path.stat().st_size if self.path.exists() else 0
        return {
            "path": str(self.path),
            "size_bytes": size,
            "current_lsn": self._lsn,
        }