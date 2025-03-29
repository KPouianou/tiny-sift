"""
Hashing functions for TinySift.

This module provides efficient hash implementations that require no external dependencies.
These functions are optimized for performance and distribution quality, not cryptographic security.
"""

from typing import Any, Union


def murmurhash3_32(key: Any, seed: int = 0) -> int:
    """
    Pure Python implementation of MurmurHash3 (32-bit variant).

    MurmurHash is a non-cryptographic hash function suitable for general hash-based
    lookup. It's known for being fast, having good distribution, and minimizing collisions.

    Args:
        key: The key to hash (will be converted to bytes if not already)
        seed: Optional seed for the hash

    Returns:
        32-bit hash value
    """
    # Convert key to bytes if it's not already
    if isinstance(key, str):
        key_bytes = key.encode("utf-8")
    elif isinstance(key, bytes):
        key_bytes = key
    else:
        # For other types, convert to string first and then to bytes
        # Use repr() to get a more unique string for various objects
        key_bytes = repr(key).encode("utf-8")

    length = len(key_bytes)

    # MurmurHash3 constants
    c1 = 0xCC9E2D51
    c2 = 0x1B873593

    # Initialize hash with seed
    h = seed

    # Process 4 bytes at a time
    nblocks = length // 4
    for i in range(nblocks):
        # Extract 4-byte block and convert to integer
        k = (
            key_bytes[i * 4]
            | (key_bytes[i * 4 + 1] << 8)
            | (key_bytes[i * 4 + 2] << 16)
            | (key_bytes[i * 4 + 3] << 24)
        )

        # MurmurHash3 block mixing function
        k = (k * c1) & 0xFFFFFFFF
        k = ((k << 15) | (k >> 17)) & 0xFFFFFFFF  # rotl32(k, 15)
        k = (k * c2) & 0xFFFFFFFF

        # Update hash
        h ^= k
        h = ((h << 13) | (h >> 19)) & 0xFFFFFFFF  # rotl32(h, 13)
        h = (h * 5 + 0xE6546B64) & 0xFFFFFFFF

    # Handle remaining bytes (tail)
    k = 0
    idx = nblocks * 4

    # Process remaining bytes based on how many are left (0-3)
    if length & 3 >= 3:
        k ^= key_bytes[idx + 2] << 16
    if length & 3 >= 2:
        k ^= key_bytes[idx + 1] << 8
    if length & 3 >= 1:
        k ^= key_bytes[idx]
        k = (k * c1) & 0xFFFFFFFF
        k = ((k << 15) | (k >> 17)) & 0xFFFFFFFF  # rotl32(k, 15)
        k = (k * c2) & 0xFFFFFFFF
        h ^= k

    # Finalization mixing
    h ^= length
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & 0xFFFFFFFF
    h ^= h >> 16

    # Ensure unsigned 32-bit integer
    return h & 0xFFFFFFFF


def fnv1a_32(key: Any, seed: int = 0) -> int:
    """
    Pure Python implementation of FNV-1a hash (32-bit variant).

    FNV-1a is a simple but effective non-cryptographic hash function.
    It's slightly faster than MurmurHash3 but with slightly less uniform distribution.

    Args:
        key: The key to hash (will be converted to bytes if not already)
        seed: Optional seed value (modifies the initial hash value)

    Returns:
        32-bit hash value
    """
    # Convert key to bytes if it's not already
    if isinstance(key, str):
        key_bytes = key.encode("utf-8")
    elif isinstance(key, bytes):
        key_bytes = key
    else:
        key_bytes = repr(key).encode("utf-8")

    # FNV-1a constants
    FNV_PRIME = 16777619
    FNV_OFFSET_BASIS = 2166136261

    # Initialize hash with offset basis and seed
    h = (FNV_OFFSET_BASIS ^ seed) & 0xFFFFFFFF

    # Process each byte
    for byte in key_bytes:
        h ^= byte
        h = (h * FNV_PRIME) & 0xFFFFFFFF

    return h
