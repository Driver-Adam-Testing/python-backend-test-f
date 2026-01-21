"""Thread-safe caching with TTL expiration and optional S3 persistence.

TTLCache: In-memory cache with automatic TTL-based expiration.
S3BackedTTLCache: Two-tier cache (L1 memory + L2 S3) with coordinated loading
                  to prevent thundering herd on cold containers.
"""

import asyncio
import os
import threading
import time
from typing import Any

import boto3
from shared.inspector.utils.io import (
    delete_cache_from_s3,
    download_cache_from_s3,
    upload_cache_to_s3,
)


class TTLCache:
    """Thread-safe cache with TTL-based expiration."""

    def __init__(self, default_ttl_seconds: int = 28800) -> None:
        self._cache: dict[str, tuple[Any, float]] = {}
        self._lock = threading.Lock()
        self._default_ttl = default_ttl_seconds

    def put(
        self, key: str, value: Any, ttl_seconds: int | None = None
    ) -> tuple[str, list[str]]:
        now = time.time()
        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl
        expires = now + ttl

        with self._lock:
            evicted_keys = self._evict_expired(now)
            self._cache[key] = (value, expires)

        return key, evicted_keys

    def get(self, key: str) -> Any:
        now = time.time()

        with self._lock:
            value, expires = self._cache[key]  # Raises KeyError if missing
            if expires < now:
                del self._cache[key]
                raise KeyError(key)
            return value

    def delete(self, key: str) -> None:
        with self._lock:
            self._cache.pop(key, None)

    def _evict_expired(self, now: float) -> list[str]:
        keys_to_delete = [k for k, (_, exp) in self._cache.items() if exp < now]
        for k in keys_to_delete:
            del self._cache[k]
        return keys_to_delete


class S3BackedTTLCache:
    """Thread-safe cache with TTL-based expiration and S3 persistence.

    This cache survives container reschedules by persisting to S3.
    On get, if the key is not in memory, it attempts to load from S3.
    Uses coordinated loading to prevent multiple concurrent S3 downloads.

    Provides both sync and async methods:
    - Sync methods (put, get, delete): Use for sync Hatchet tasks
    - Async methods (aput, aget, adelete): Use for async code to avoid blocking event loop
    """

    def __init__(self, cache_type: str, default_ttl_seconds: int = 28800) -> None:
        self._l1 = TTLCache(default_ttl_seconds)
        self._cache_type = cache_type
        self._default_ttl = default_ttl_seconds
        # Threading lock for sync coordination
        self._load_locks: dict[str, threading.Lock] = {}
        self._load_locks_guard = threading.Lock()
        # Async lock for async coordination
        self._async_load_locks: dict[str, asyncio.Lock] = {}
        self._async_load_locks_guard = asyncio.Lock()

    def _get_s3_client_and_bucket(self) -> tuple[Any, str]:
        s3_client = boto3.client(
            "s3", endpoint_url=os.environ.get("AWS_S3_ENDPOINT_URL")
        )
        bucket_name = os.environ["INSPECTOR_BUCKET_NAME"]
        return s3_client, bucket_name

    def _upload_to_s3(self, key: str, value: Any) -> None:
        s3_client, bucket_name = self._get_s3_client_and_bucket()
        upload_cache_to_s3(
            cache_type=self._cache_type,
            key=key,
            value=value,
            s3_client=s3_client,
            bucket_name=bucket_name,
        )

    def _download_from_s3(self, key: str) -> Any:
        s3_client, bucket_name = self._get_s3_client_and_bucket()
        return download_cache_from_s3(
            cache_type=self._cache_type,
            key=key,
            s3_client=s3_client,
            bucket_name=bucket_name,
        )

    def _delete_from_s3(self, key: str) -> None:
        s3_client, bucket_name = self._get_s3_client_and_bucket()
        delete_cache_from_s3(
            cache_type=self._cache_type,
            key=key,
            s3_client=s3_client,
            bucket_name=bucket_name,
        )

    def put(self, key: str, value: Any, ttl_seconds: int | None = None) -> str:
        _, evicted_keys = self._l1.put(key, value, ttl_seconds)

        if evicted_keys:
            with self._load_locks_guard:
                for evicted_key in evicted_keys:
                    self._load_locks.pop(evicted_key, None)

        try:
            self._upload_to_s3(key, value)
        except Exception as e:
            print(f"Warning: Failed to persist cache [{self._cache_type}] to S3: {e}")

        return key

    def get(self, key: str) -> Any:
        try:
            return self._l1.get(key)
        except KeyError:
            pass

        with self._load_locks_guard:
            if key not in self._load_locks:
                self._load_locks[key] = threading.Lock()
            lock = self._load_locks[key]

        with lock:
            # Double-check L1 (another thread may have loaded it)
            try:
                return self._l1.get(key)
            except KeyError:
                pass

            print(
                f"Cache [{self._cache_type}] not in memory, loading from S3 for key {key}"
            )
            value = self._download_from_s3(key)
            self._l1.put(key, value)
            return value

    def delete(self, key: str) -> None:
        self._l1.delete(key)

        with self._load_locks_guard:
            self._load_locks.pop(key, None)

        # NOTE: Do NOT delete from L2 (S3) in case we resume inspection and must refetch the cache result

    async def aput(self, key: str, value: Any, ttl_seconds: int | None = None) -> str:
        _, evicted_keys = self._l1.put(key, value, ttl_seconds)

        if evicted_keys:
            async with self._async_load_locks_guard:
                for evicted_key in evicted_keys:
                    self._async_load_locks.pop(evicted_key, None)

        try:
            await asyncio.to_thread(self._upload_to_s3, key, value)
        except Exception as e:
            print(f"Warning: Failed to persist cache [{self._cache_type}] to S3: {e}")

        return key

    async def aget(self, key: str) -> Any:
        try:
            return self._l1.get(key)
        except KeyError:
            pass

        async with self._async_load_locks_guard:
            if key not in self._async_load_locks:
                self._async_load_locks[key] = asyncio.Lock()
            lock = self._async_load_locks[key]

        async with lock:
            # Double-check L1
            try:
                return self._l1.get(key)
            except KeyError:
                pass

            print(
                f"Cache [{self._cache_type}] not in memory, loading from S3 for key {key}"
            )
            value = await asyncio.to_thread(self._download_from_s3, key)
            self._l1.put(key, value)
            return value

    async def adelete(self, key: str) -> None:
        self._l1.delete(key)

        # Clean up associated locks to prevent memory leak
        async with self._async_load_locks_guard:
            self._async_load_locks.pop(key, None)

        # NOTE: Do NOT delete from L2 (S3) in case we resume inspection and must refetch the cache result
