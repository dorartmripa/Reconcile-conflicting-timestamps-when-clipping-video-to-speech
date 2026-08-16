"""Identify the bundled sample video without trusting transcript text alone."""

from __future__ import annotations

import hashlib
from pathlib import Path

CHUNK = 64 * 1024


def file_fingerprint(path: Path) -> str:
    """Stable identity: size plus head/tail bytes (full file if small)."""

    data = path.read_bytes()
    digest = hashlib.sha256()
    digest.update(str(len(data)).encode("utf-8"))
    digest.update(b"\0")
    digest.update(data[:CHUNK])
    if len(data) > CHUNK:
        digest.update(data[-CHUNK:])
    return digest.hexdigest()


def is_same_media(left: Path, right: Path) -> bool:
    if not left.exists() or not right.exists():
        return False
    if left.resolve() == right.resolve():
        return True
    return file_fingerprint(left) == file_fingerprint(right)
