"""Cryptographic hashing utilities for Vision Model Factory."""

import hashlib
import re
from pathlib import Path
from typing import Union

SHA256_REGEX = re.compile(r"^[0-9a-f]{64}$")


def is_valid_sha256(value: str) -> bool:
    """Check whether a string is a valid 64-character lowercase hex SHA-256 digest."""
    return bool(SHA256_REGEX.match(value))


def compute_sha256_bytes(data: bytes) -> str:
    """Compute the SHA-256 hex digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def compute_sha256_str(data: str, encoding: str = "utf-8") -> str:
    """Compute the SHA-256 hex digest of a string."""
    return compute_sha256_bytes(data.encode(encoding))


def compute_sha256_file(file_path: Union[str, Path], chunk_size: int = 65536) -> str:
    """Compute the SHA-256 hex digest of a file on disk."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found for hashing: {path}")

    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()
