import pytest
import tempfile
import os
from ingestion.message_queue import MessageQueue, AppendLog, HEADER_SIZE


def test_append_and_read():
    with tempfile.TemporaryDirectory() as d:
        log = AppendLog(f"{d}/test.log")
        offset = log.append(b"hello world")
        msg = log.read_at(offset)
        assert msg.payload == b"hello world"


def test_crc_corruption_detected():
    from ingestion.message_queue import CorruptedRecordError
    with tempfile.TemporaryDirectory() as d:
        path = f"{d}/test.log"
        log = AppendLog(path)
        offset = log.append(b"important data")

        # Corrupt one byte in the payload
        with open(path, "r+b") as f:
            f.seek(offset + 4 + 3)  # skip length(4), corrupt 4th payload byte
            f.write(b"\xff")

        with pytest.raises(CorruptedRecordError):
            log.read_at(offset)


def test_consumer_offset_independence():
    """Two consumers reading the same log maintain independent offsets."""
    with tempfile.TemporaryDirectory() as d:
        mq = MessageQueue(f"{d}/txns")
        for i in range(5):
            mq.publish({"id": i})

        # Consumer A reads 2
        msgs_a = list(mq.consume("group_a", max_messages=2))
        for m in msgs_a:
            mq.ack("group_a", m)

        # Consumer B reads all 5
        msgs_b = list(mq.consume("group_b", max_messages=10))
        for m in msgs_b:
            mq.ack("group_b", m)

        # Consumer A resumes from offset 2
        remaining_a = list(mq.consume("group_a", max_messages=10))
        assert len(remaining_a) == 3
        assert len(msgs_b) == 5


def test_publish_and_consume_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        mq = MessageQueue(f"{d}/txns")
        data = {"transaction_id": "txn_abc", "amount": 1500.0, "is_fraud": False}
        mq.publish(data)

        msgs = list(mq.consume("test_consumer", max_messages=1))
        assert len(msgs) == 1
        assert msgs[0].to_dict()["transaction_id"] == "txn_abc"
        assert msgs[0].to_dict()["amount"] == 1500.0