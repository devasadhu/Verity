"""
Append-only message queue with consumer offset tracking.

Design mirrors Kafka's core abstraction:
  - Messages are appended to an immutable log
  - Consumers track their own offset (position in log)
  - Multiple consumers can read the same log independently
  - No deletion — compaction policy handles old data

Format on disk (each record):
  [4 bytes: payload length][payload bytes][4 bytes: CRC32]

This is exactly how Kafka's log segment files work.
"""

import os
import struct
import zlib
import json
import threading
from pathlib import Path
from typing import Optional, Iterator
from dataclasses import dataclass


MAGIC_HEADER = b"VRTY"   # 4-byte file magic
VERSION = 1
HEADER_SIZE = 8           # magic(4) + version(4)
RECORD_OVERHEAD = 8       # length(4) + crc(4)


@dataclass
class Message:
    offset: int
    payload: bytes
    crc: int

    def to_dict(self) -> dict:
        return json.loads(self.payload.decode("utf-8"))


class CorruptedRecordError(Exception):
    pass


class AppendLog:
    """
    Single append-only log file.
    Thread-safe for one writer, multiple readers.
    """

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()
        self._file_size = 0

        if self.path.exists():
            self._file_size = self.path.stat().st_size
            self._verify_header()
        else:
            self._write_header()

    def _write_header(self):
        with open(self.path, "wb") as f:
            f.write(MAGIC_HEADER)
            f.write(struct.pack(">I", VERSION))
        self._file_size = HEADER_SIZE

    def _verify_header(self):
        with open(self.path, "rb") as f:
            magic = f.read(4)
            if magic != MAGIC_HEADER:
                raise ValueError(
                    f"Invalid log file at {self.path}. "
                    f"Expected magic {MAGIC_HEADER}, got {magic}"
                )
            version = struct.unpack(">I", f.read(4))[0]
            if version != VERSION:
                raise ValueError(f"Unsupported log version: {version}")

    def append(self, payload: bytes) -> int:
        """
        Appends payload to log.
        Returns the byte offset of this record within the file.
        Thread-safe.
        """
        crc = zlib.crc32(payload) & 0xFFFFFFFF
        record = struct.pack(">I", len(payload)) + payload + struct.pack(">I", crc)

        with self._write_lock:
            offset = self._file_size
            with open(self.path, "ab") as f:
                f.write(record)
                f.flush()
                os.fsync(f.fileno())  # durability guarantee
            self._file_size += len(record)

        return offset

    def read_at(self, offset: int) -> Message:
        """
        Read one record at the given byte offset.
        Validates CRC. Raises CorruptedRecordError on mismatch.
        """
        with open(self.path, "rb") as f:
            f.seek(offset)
            len_bytes = f.read(4)
            if len(len_bytes) < 4:
                raise EOFError(f"No record at offset {offset}")

            payload_len = struct.unpack(">I", len_bytes)[0]
            payload = f.read(payload_len)
            crc_bytes = f.read(4)

            if len(payload) < payload_len or len(crc_bytes) < 4:
                raise EOFError(f"Truncated record at offset {offset}")

            stored_crc = struct.unpack(">I", crc_bytes)[0]
            computed_crc = zlib.crc32(payload) & 0xFFFFFFFF

            if stored_crc != computed_crc:
                raise CorruptedRecordError(
                    f"CRC mismatch at offset {offset}: "
                    f"stored={stored_crc:#010x}, computed={computed_crc:#010x}"
                )

        return Message(
            offset=offset,
            payload=payload,
            crc=stored_crc,
        )

    def scan(self, start_offset: int = HEADER_SIZE) -> Iterator[Message]:
        """
        Iterate over all records from start_offset to end of file.
        Yields Message objects. Stops at EOF.
        """
        offset = start_offset
        while offset < self._file_size:
            try:
                msg = self.read_at(offset)
                yield msg
                offset += RECORD_OVERHEAD + len(msg.payload)
            except EOFError:
                break

    def size(self) -> int:
        return self._file_size

    def record_count(self) -> int:
        """Scan entire log and count records. O(n) — use sparingly."""
        return sum(1 for _ in self.scan())


class ConsumerOffset:
    """
    Persists consumer offsets to disk.
    Each consumer group tracks independently — mirrors Kafka consumer groups.

    Format: JSON file mapping consumer_group → last committed byte offset.
    """

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._offsets: dict[str, int] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        if self.path.exists():
            with open(self.path, "r") as f:
                self._offsets = json.load(f)

    def _save(self):
        with open(self.path, "w") as f:
            json.dump(self._offsets, f)

    def get(self, consumer_group: str) -> int:
        """Returns last committed offset, or start of data if new consumer."""
        with self._lock:
            return self._offsets.get(consumer_group, HEADER_SIZE)

    def commit(self, consumer_group: str, offset: int):
        """Commit offset after processing. Call after successful processing."""
        with self._lock:
            self._offsets[consumer_group] = offset
            self._save()

    def reset(self, consumer_group: str):
        """Reset consumer to beginning of log."""
        with self._lock:
            self._offsets[consumer_group] = HEADER_SIZE
            self._save()


class MessageQueue:
    """
    High-level interface combining AppendLog + ConsumerOffset.

    Usage:
        # Producer
        mq = MessageQueue("data/queue/transactions")
        mq.publish({"transaction_id": "txn_abc", "amount": 1500.0})

        # Consumer
        for msg in mq.consume("stream_processor"):
            data = msg.to_dict()
            process(data)
            mq.ack("stream_processor", msg)
    """

    def __init__(self, base_path: str):
        self.log = AppendLog(f"{base_path}.log")
        self.offsets = ConsumerOffset(f"{base_path}.offsets.json")

    def publish(self, data: dict) -> int:
        """Serialize dict to JSON and append to log. Returns byte offset."""
        payload = json.dumps(data, default=str).encode("utf-8")
        return self.log.append(payload)

    def consume(
        self,
        consumer_group: str,
        max_messages: Optional[int] = None
    ) -> Iterator[Message]:
        """
        Yield unprocessed messages for this consumer group.
        Does NOT auto-commit — caller must call ack() after processing.
        """
        start = self.offsets.get(consumer_group)
        count = 0
        for msg in self.log.scan(start_offset=start):
            yield msg
            count += 1
            if max_messages and count >= max_messages:
                break

    def ack(self, consumer_group: str, msg: Message):
        """
        Acknowledge a message. Advances consumer offset past this record.
        Call this after successfully processing the message.
        """
        next_offset = msg.offset + RECORD_OVERHEAD + len(msg.payload)
        self.offsets.commit(consumer_group, next_offset)

    def stats(self) -> dict:
        return {
            "log_size_bytes": self.log.size(),
            "log_path": str(self.log.path),
        }