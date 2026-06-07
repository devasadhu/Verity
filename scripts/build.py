"""
Build script for Windows (replaces make).
Run from project root: python scripts\build.py
"""
import subprocess
import sys
from pathlib import Path
import sys

import storage
ext = ".dll" if sys.platform == "win32" else ".so"
out = storage / f"hashmap{ext}"
src = storage / "hashmap.c"

def build():
    storage = Path("storage")
    out = storage / f"hashmap{ext}"
    src = storage / "hashmap.c"

    if not src.exists():
        print(f"ERROR: {src} not found. Run from project root.")
        sys.exit(1)

    cmd = [
    "gcc", "-O2", "-Wall", "-fPIC", "-shared",
    "-o", str(out), str(src)
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("BUILD FAILED:")
        print(result.stderr)
        sys.exit(1)

    print(f"Built: {out} ({out.stat().st_size} bytes)")

if __name__ == "__main__":
    build()