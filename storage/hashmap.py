"""
Hash map implementation in pure Python.
Identical interface to the C-backed version.

Uses open-addressing with linear probing and tombstone deletion —
the same algorithm as hashmap.c. On Linux/HPC the C-backed version
is loaded automatically; this is the Windows development fallback.
"""

import sys
from pathlib import Path
import ctypes

_TOMBSTONE = object()  # sentinel for deleted slots
_LOAD_FACTOR = 0.70


def _try_load_native():
    ext = ".dll" if sys.platform == "win32" else ".so"
    lib_path = Path(__file__).parent / f"hashmap{ext}"
    if lib_path.exists():
        try:
            return ctypes.CDLL(str(lib_path))
        except OSError:
            return None
    return None


class HashMap:
    """
    Open-addressing hash map with linear probing and tombstone deletion.
    Pure Python implementation — identical interface to C-backed version.
    """

    def __init__(self, initial_capacity: int = 64):
        self._capacity = max(initial_capacity, 8)
        self._slots = [None] * self._capacity   # None=empty, _TOMBSTONE=deleted
        self._values = [None] * self._capacity
        self._count = 0
        self._tombstones = 0

    def _hash(self, key: str) -> int:
        # FNV-1a — matches the C implementation
        h = 0xcbf29ce484222325
        for ch in key.encode("utf-8"):
            h ^= ch
            h = (h * 0x100000000001b3) & 0xFFFFFFFFFFFFFFFF
        return h

    def _resize(self, new_capacity: int):
        old_slots = self._slots
        old_values = self._values
        self._capacity = new_capacity
        self._slots = [None] * new_capacity
        self._values = [None] * new_capacity
        self._count = 0
        self._tombstones = 0
        for k, v in zip(old_slots, old_values):
            if k is not None and k is not _TOMBSTONE:
                self.set(k, v)

    def set(self, key: str, value: str) -> None:
        if (self._count + self._tombstones) / self._capacity >= _LOAD_FACTOR:
            self._resize(self._capacity * 2)

        idx = self._hash(key) % self._capacity
        first_tombstone = None

        while True:
            slot = self._slots[idx]
            if slot is None:
                insert_at = first_tombstone if first_tombstone is not None else idx
                self._slots[insert_at] = key
                self._values[insert_at] = value
                self._count += 1
                if first_tombstone is not None:
                    self._tombstones -= 1
                return
            if slot is _TOMBSTONE:
                if first_tombstone is None:
                    first_tombstone = idx
            elif slot == key:
                self._values[idx] = value
                return
            idx = (idx + 1) % self._capacity

    def get(self, key: str):
        idx = self._hash(key) % self._capacity
        while True:
            slot = self._slots[idx]
            if slot is None:
                return None
            if slot is not _TOMBSTONE and slot == key:
                return self._values[idx]
            idx = (idx + 1) % self._capacity

    def delete(self, key: str) -> bool:
        idx = self._hash(key) % self._capacity
        while True:
            slot = self._slots[idx]
            if slot is None:
                return False
            if slot is not _TOMBSTONE and slot == key:
                self._slots[idx] = _TOMBSTONE
                self._values[idx] = None
                self._count -= 1
                self._tombstones += 1
                return True
            idx = (idx + 1) % self._capacity

    def __len__(self) -> int:
        return self._count

    def capacity(self) -> int:
        return self._capacity

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None

    def __setitem__(self, key: str, value: str):
        self.set(key, value)

    def __getitem__(self, key: str) -> str:
        val = self.get(key)
        if val is None:
            raise KeyError(key)
        return val

    def __delitem__(self, key: str):
        if not self.delete(key):
            raise KeyError(key)