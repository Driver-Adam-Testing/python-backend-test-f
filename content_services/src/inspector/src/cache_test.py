"""Tests for TTLCache and S3BackedTTLCache.

Focuses on:
- Thread safety under concurrent access
- TTL expiration behavior
- L1/L2 (memory/S3) fallback with coordinated loading
"""

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest
from inspector.src.cache import S3BackedTTLCache, TTLCache


class TestTTLCache:
    """Unit tests for the in-memory TTLCache."""

    def test_put_and_get_basic(self):
        """Basic put/get operations work."""
        cache = TTLCache(default_ttl_seconds=60)
        cache.put("key1", "value1")

        assert cache.get("key1") == "value1"

    def test_get_missing_key_raises_keyerror(self):
        """Getting a missing key raises KeyError."""
        cache = TTLCache(default_ttl_seconds=60)

        with pytest.raises(KeyError):
            cache.get("nonexistent")

    def test_ttl_expiration(self):
        """Keys expire after TTL."""
        cache = TTLCache(default_ttl_seconds=60)
        cache.put("key1", "value1", ttl_seconds=0)

        time.sleep(0.01)

        with pytest.raises(KeyError):
            cache.get("key1")

    def test_delete_removes_key(self):
        """Delete removes key from cache."""
        cache = TTLCache(default_ttl_seconds=60)
        cache.put("key1", "value1")
        cache.delete("key1")

        with pytest.raises(KeyError):
            cache.get("key1")

    def test_delete_nonexistent_key_is_safe(self):
        """Deleting a nonexistent key doesn't raise."""
        cache = TTLCache(default_ttl_seconds=60)
        cache.delete("nonexistent")

    def test_put_returns_evicted_keys(self):
        """Put returns list of keys evicted due to TTL."""
        cache = TTLCache(default_ttl_seconds=60)
        cache.put("old_key", "old_value", ttl_seconds=0)

        time.sleep(0.01)

        _, evicted = cache.put("new_key", "new_value")

        assert "old_key" in evicted

    def test_overwrite_existing_key(self):
        """Overwriting an existing key updates the value."""
        cache = TTLCache(default_ttl_seconds=60)
        cache.put("key1", "value1")
        cache.put("key1", "value2")

        assert cache.get("key1") == "value2"

    def test_thread_safety_concurrent_writes(self):
        """Concurrent writes from multiple threads don't corrupt state."""
        cache = TTLCache(default_ttl_seconds=60)
        num_threads = 10
        writes_per_thread = 100

        def writer(thread_id: int):
            for i in range(writes_per_thread):
                cache.put(f"thread_{thread_id}_key_{i}", f"value_{i}")

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(writer, i) for i in range(num_threads)]
            for f in futures:
                f.result()

        for thread_id in range(num_threads):
            for i in range(writes_per_thread):
                assert cache.get(f"thread_{thread_id}_key_{i}") == f"value_{i}"

    def test_thread_safety_concurrent_reads_and_writes(self):
        """Concurrent reads and writes don't cause race conditions."""
        cache = TTLCache(default_ttl_seconds=60)
        cache.put("shared_key", "initial_value")

        errors = []
        stop_flag = threading.Event()

        def reader():
            while not stop_flag.is_set():
                try:
                    value = cache.get("shared_key")
                    if not value.startswith("value_") and value != "initial_value":
                        errors.append(f"Unexpected value: {value}")
                except KeyError:
                    pass

        def writer():
            for i in range(100):
                cache.put("shared_key", f"value_{i}")

        reader_threads = [threading.Thread(target=reader) for _ in range(5)]
        for t in reader_threads:
            t.start()

        writer_thread = threading.Thread(target=writer)
        writer_thread.start()
        writer_thread.join()

        stop_flag.set()
        for t in reader_threads:
            t.join()

        assert len(errors) == 0, f"Race condition errors: {errors}"


class TestS3BackedTTLCacheSyncMethods:
    """Tests for S3BackedTTLCache sync methods (put/get/delete)."""

    def test_l1_cache_hit_skips_s3(self):
        """When key is in L1, S3 is not called."""
        cache = S3BackedTTLCache(cache_type="test", default_ttl_seconds=60)

        with (
            patch.object(cache, "_upload_to_s3"),
            patch.object(cache, "_download_from_s3") as mock_download,
        ):
            cache.put("key1", "value1")

            result = cache.get("key1")
            assert result == "value1"
            mock_download.assert_not_called()

    def test_l1_miss_falls_back_to_s3(self):
        """When key is not in L1, falls back to S3."""
        cache = S3BackedTTLCache(cache_type="test", default_ttl_seconds=60)

        with patch.object(cache, "_download_from_s3", return_value="s3_value"):
            result = cache.get("missing_key")

            assert result == "s3_value"

    def test_l1_miss_s3_value_cached_in_l1(self):
        """Value loaded from S3 is cached in L1 for subsequent gets."""
        cache = S3BackedTTLCache(cache_type="test", default_ttl_seconds=60)

        with patch.object(
            cache, "_download_from_s3", return_value="s3_value"
        ) as mock_download:
            result1 = cache.get("key1")
            assert result1 == "s3_value"
            assert mock_download.call_count == 1

            result2 = cache.get("key1")
            assert result2 == "s3_value"
            assert mock_download.call_count == 1

    def test_s3_upload_failure_does_not_crash(self):
        """S3 upload failure logs warning but doesn't raise."""
        cache = S3BackedTTLCache(cache_type="test", default_ttl_seconds=60)

        with patch.object(cache, "_upload_to_s3", side_effect=Exception("S3 error")):
            key = cache.put("key1", "value1")
            assert key == "key1"

            assert cache.get("key1") == "value1"

    def test_delete_removes_from_l1_only(self):
        """Delete removes from L1 but NOT from S3 (for resumption)."""
        cache = S3BackedTTLCache(cache_type="test", default_ttl_seconds=60)

        with (
            patch.object(cache, "_upload_to_s3"),
            patch.object(cache, "_delete_from_s3") as mock_s3_delete,
        ):
            cache.put("key1", "value1")
            cache.delete("key1")

            mock_s3_delete.assert_not_called()

    def test_coordinated_loading_prevents_concurrent_s3_downloads(self):
        """Multiple threads requesting same missing key only trigger one S3 download."""
        cache = S3BackedTTLCache(cache_type="test", default_ttl_seconds=60)

        download_call_count = 0
        download_started = threading.Event()
        download_can_finish = threading.Event()

        def slow_download(key: str):
            nonlocal download_call_count
            download_call_count += 1
            download_started.set()
            download_can_finish.wait(timeout=5)
            return "s3_value"

        with patch.object(cache, "_download_from_s3", side_effect=slow_download):
            results = []
            errors = []

            def get_key():
                try:
                    results.append(cache.get("contested_key"))
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=get_key) for _ in range(5)]
            for t in threads:
                t.start()

            download_started.wait(timeout=5)
            time.sleep(0.1)

            download_can_finish.set()

            for t in threads:
                t.join(timeout=5)

            assert len(errors) == 0, f"Errors: {errors}"
            assert all(r == "s3_value" for r in results)
            assert (
                download_call_count == 1
            ), f"Expected 1 S3 download, got {download_call_count}"

    def test_lock_cleanup_on_eviction(self):
        """Locks for evicted keys are cleaned up to prevent memory leaks."""
        cache = S3BackedTTLCache(cache_type="test", default_ttl_seconds=60)

        with patch.object(cache, "_upload_to_s3"):
            cache.put("old_key", "old_value", ttl_seconds=0)
            cache._load_locks["old_key"] = threading.Lock()

            time.sleep(0.01)

            cache.put("new_key", "new_value")

            assert "old_key" not in cache._load_locks


class TestS3BackedTTLCacheAsyncMethods:
    """Tests for S3BackedTTLCache async methods (aput/aget/adelete)."""

    @pytest.mark.asyncio
    async def test_async_l1_cache_hit_skips_s3(self):
        """Async: When key is in L1, S3 is not called."""
        cache = S3BackedTTLCache(cache_type="test", default_ttl_seconds=60)

        with (
            patch.object(cache, "_upload_to_s3"),
            patch.object(
                cache, "_download_from_s3", return_value="s3_value"
            ) as mock_download,
        ):
            await cache.aput("key1", "value1")

            result = await cache.aget("key1")
            assert result == "value1"
            mock_download.assert_not_called()

    @pytest.mark.asyncio
    async def test_async_l1_miss_falls_back_to_s3(self):
        """Async: When key is not in L1, falls back to S3."""
        cache = S3BackedTTLCache(cache_type="test", default_ttl_seconds=60)

        with patch.object(cache, "_download_from_s3", return_value="s3_value"):
            result = await cache.aget("missing_key")

            assert result == "s3_value"

    @pytest.mark.asyncio
    async def test_async_coordinated_loading_prevents_concurrent_downloads(self):
        """Async: Multiple coroutines requesting same key only trigger one S3 download."""
        cache = S3BackedTTLCache(cache_type="test", default_ttl_seconds=60)

        download_call_count = 0
        download_started = threading.Event()
        download_can_finish = threading.Event()

        def slow_download(key: str):
            nonlocal download_call_count
            download_call_count += 1
            download_started.set()
            download_can_finish.wait(timeout=5)
            return "s3_value"

        with patch.object(cache, "_download_from_s3", side_effect=slow_download):
            results = []

            async def get_key():
                result = await cache.aget("contested_key")
                results.append(result)

            tasks = [asyncio.create_task(get_key()) for _ in range(5)]

            # Wait for download to start (runs in thread pool)
            await asyncio.to_thread(download_started.wait, 5)
            await asyncio.sleep(0.05)

            download_can_finish.set()

            await asyncio.gather(*tasks)

            assert all(r == "s3_value" for r in results)
            assert (
                download_call_count == 1
            ), f"Expected 1 download, got {download_call_count}"

    @pytest.mark.asyncio
    async def test_async_lock_cleanup_on_eviction(self):
        """Async: Locks for evicted keys are cleaned up from unified lock dict."""
        cache = S3BackedTTLCache(cache_type="test", default_ttl_seconds=60)

        with patch.object(cache, "_upload_to_s3"):
            await cache.aput("old_key", "old_value", ttl_seconds=0)
            cache._load_locks["old_key"] = threading.Lock()

            await asyncio.sleep(0.01)

            await cache.aput("new_key", "new_value")

            assert "old_key" not in cache._load_locks


class TestS3BackedTTLCacheMixedSyncAsync:
    """Tests for mixed sync/async usage - verifies unified lock infrastructure."""

    @pytest.mark.asyncio
    async def test_mixed_sync_async_no_lock_leak(self):
        """Mixed sync/async usage cleans up all locks properly."""
        cache = S3BackedTTLCache(cache_type="test", default_ttl_seconds=60)

        with patch.object(cache, "_upload_to_s3"):
            # Async put creates entry
            await cache.aput("key1", "value1")

            # Sync get (L1 hit)
            result = cache.get("key1")
            assert result == "value1"

            # Async delete should clean up locks
            await cache.adelete("key1")

            # No orphaned locks
            assert "key1" not in cache._load_locks

    @pytest.mark.asyncio
    async def test_sync_delete_cleans_lock_after_async_get(self):
        """Sync delete properly cleans lock created by async get path."""
        cache = S3BackedTTLCache(cache_type="test", default_ttl_seconds=60)

        with patch.object(cache, "_download_from_s3", return_value="s3_value"):
            # Async get creates lock in _load_locks
            result = await cache.aget("key1")
            assert result == "s3_value"
            assert "key1" in cache._load_locks

            # Sync delete should clean the same lock
            cache.delete("key1")
            assert "key1" not in cache._load_locks

    @pytest.mark.asyncio
    async def test_async_delete_cleans_lock_after_sync_get(self):
        """Async delete properly cleans lock created by sync get path."""
        cache = S3BackedTTLCache(cache_type="test", default_ttl_seconds=60)

        with patch.object(cache, "_download_from_s3", return_value="s3_value"):
            # Sync get creates lock in _load_locks
            result = cache.get("key1")
            assert result == "s3_value"
            assert "key1" in cache._load_locks

            # Async delete should clean the same lock
            await cache.adelete("key1")
            assert "key1" not in cache._load_locks

    @pytest.mark.asyncio
    async def test_cross_boundary_coordinated_loading(self):
        """Concurrent sync get() and async aget() for missing key only trigger one S3 download."""
        cache = S3BackedTTLCache(cache_type="test", default_ttl_seconds=60)

        download_call_count = 0
        download_started = threading.Event()
        download_can_finish = threading.Event()

        def slow_download(key: str):
            nonlocal download_call_count
            download_call_count += 1
            download_started.set()
            download_can_finish.wait(timeout=5)
            return "s3_value"

        with patch.object(cache, "_download_from_s3", side_effect=slow_download):
            results = []
            errors = []

            def sync_get():
                try:
                    results.append(("sync", cache.get("contested_key")))
                except Exception as e:
                    errors.append(e)

            async def async_get():
                try:
                    results.append(("async", await cache.aget("contested_key")))
                except Exception as e:
                    errors.append(e)

            # Start sync getter in thread
            sync_thread = threading.Thread(target=sync_get)
            sync_thread.start()

            # Wait for download to start
            download_started.wait(timeout=5)
            await asyncio.sleep(0.05)

            # Start async getter (should wait on same lock)
            async_task = asyncio.create_task(async_get())
            await asyncio.sleep(0.05)

            # Release the download
            download_can_finish.set()

            sync_thread.join(timeout=5)
            await async_task

            assert len(errors) == 0, f"Errors: {errors}"
            assert len(results) == 2
            assert all(r[1] == "s3_value" for r in results)
            assert (
                download_call_count == 1
            ), f"Expected 1 S3 download, got {download_call_count}"

    @pytest.mark.asyncio
    async def test_async_operations_do_not_block_event_loop(self):
        """Async cache operations don't block the event loop during S3 operations."""
        cache = S3BackedTTLCache(cache_type="test", default_ttl_seconds=60)

        download_started = threading.Event()
        download_can_finish = threading.Event()

        def slow_download(key: str):
            download_started.set()
            download_can_finish.wait(timeout=5)
            return "s3_value"

        heartbeat_times: list[float] = []
        stop_heartbeat = asyncio.Event()

        async def heartbeat():
            """Records timestamps - if event loop is blocked, gaps will appear."""
            import contextlib

            while not stop_heartbeat.is_set():
                heartbeat_times.append(time.time())
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(stop_heartbeat.wait(), timeout=0.01)

        with (
            patch.object(cache, "_download_from_s3", side_effect=slow_download),
            patch.object(cache, "_upload_to_s3"),
        ):
            # Start heartbeat task
            heartbeat_task = asyncio.create_task(heartbeat())

            # Start async cache operation that will block in thread pool
            cache_task = asyncio.create_task(cache.aget("test_key"))

            # Wait for download to start (proves we're in the slow S3 path)
            await asyncio.to_thread(download_started.wait, 5)

            # Let heartbeat run for a bit while cache op is blocked
            await asyncio.sleep(0.1)

            # Release the download
            download_can_finish.set()
            result = await cache_task

            # Stop heartbeat
            stop_heartbeat.set()
            await heartbeat_task

            assert result == "s3_value"

            # Verify heartbeat kept running (event loop wasn't blocked)
            # Should have many heartbeats during the 0.1s sleep
            assert (
                len(heartbeat_times) >= 5
            ), f"Event loop appears blocked - only {len(heartbeat_times)} heartbeats"

            # Verify no large gaps between heartbeats (max ~50ms tolerance)
            if len(heartbeat_times) > 1:
                gaps = [
                    heartbeat_times[i + 1] - heartbeat_times[i]
                    for i in range(len(heartbeat_times) - 1)
                ]
                max_gap = max(gaps)
                assert max_gap < 0.05, f"Event loop blocked for {max_gap:.3f}s"
