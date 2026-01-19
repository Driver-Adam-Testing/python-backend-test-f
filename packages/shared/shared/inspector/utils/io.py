import contextlib
import logging
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import quote

logger = logging.getLogger(__name__)


def get_prompt_template(f: Path | str) -> str:
    p = Path(f)
    with p.open("r") as pt:
        return pt.read()


def download_source_file(
    s3_client: Any,
    bucket_name: str,
    primary_asset_id: str,
    version_id: str,
    node_rel_path: str,
    download_root: Path,
) -> Path:
    s3_key = f"{primary_asset_id}/{version_id}/{node_rel_path}"
    local_download_path = download_root / node_rel_path
    local_download_path.parent.mkdir(parents=True, exist_ok=True)

    s3_client.download_file(bucket_name, s3_key, str(local_download_path))
    return local_download_path


def download_all_source_files_in_parallel(
    s3_client: Any,
    bucket_name: str,
    primary_asset_id: str,
    version_id: str,
    node_rel_paths: list[str],
    download_root: Path,
    max_workers: int,
) -> list[Path]:
    paths: list[Path] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for node_path in node_rel_paths:
            futures.append(
                executor.submit(
                    download_source_file,
                    s3_client,
                    bucket_name,
                    primary_asset_id,
                    version_id,
                    node_path,
                    download_root,
                )
            )

        for future in as_completed(futures):
            # Any exception raised within download_source_file will be re-raised here.
            paths.append(future.result())

    return paths


def _get_symbol_table_s3_key(version_id: str) -> str:
    return f"{version_id}_symbol_table.pkl"


def _get_symbol_table_cache_path(version_id: str) -> Path:
    return Path(f"/tmp/{version_id}_symbol_table.pkl")


def upload_symbol_table_to_s3(
    symbol_table: dict[str, Any],
    s3_client: Any,
    bucket_name: str,
    version_id: str,
) -> str:
    import os
    import tempfile

    start_time = time.time()
    s3_key = _get_symbol_table_s3_key(version_id)

    fd, temp_path = tempfile.mkstemp(suffix=".pkl")
    try:
        with os.fdopen(fd, "wb") as temp_file:
            pickle.dump(symbol_table, temp_file)

        file_size_mb = Path(temp_path).stat().st_size / (1024 * 1024)
        s3_client.upload_file(temp_path, bucket_name, s3_key)

        total_time = time.time() - start_time
        print(
            f"Symbol table saved and uploaded ({file_size_mb:.2f}MB) in {total_time:.2f}s to {bucket_name}/{s3_key}"
        )
    except Exception as e:
        Path(temp_path).unlink(missing_ok=True)
        raise RuntimeError("Failed to upload symbol table to S3") from e
    finally:
        Path(temp_path).unlink(missing_ok=True)

    return s3_key


def download_symbol_table_from_s3_with_cache(
    s3_client: Any,
    bucket_name: str,
    version_id: str,
) -> dict[str, Any]:
    cache_path = _get_symbol_table_cache_path(version_id)

    try:
        start_time = time.time()
        with open(cache_path, "rb") as f:
            symbol_table = pickle.load(f)
        file_size_mb = cache_path.stat().st_size / (1024 * 1024)
        cache_time = time.time() - start_time
        print(
            f"Symbol table loaded from cache ({file_size_mb:.2f}MB) in {cache_time:.2f}s"
        )
        # Delete local file after loading - we rely on in-memory cache, not local files
        cache_path.unlink(missing_ok=True)
        return symbol_table
    except FileNotFoundError:
        pass
    except Exception:
        cache_path.unlink(missing_ok=True)

    start_time = time.time()
    s3_key = _get_symbol_table_s3_key(version_id)
    try:
        s3_client.download_file(bucket_name, s3_key, str(cache_path))
    except Exception as e:
        raise RuntimeError("Failed to download symbol table from S3") from e

    with open(cache_path, "rb") as f:
        symbol_table = pickle.load(f)

    file_size_mb = cache_path.stat().st_size / (1024 * 1024)
    download_time = time.time() - start_time
    print(
        f"Symbol table downloaded from S3 ({file_size_mb:.2f}MB) in {download_time:.2f}s"
    )
    # Delete local file after loading - we rely on in-memory cache, not local files
    cache_path.unlink(missing_ok=True)
    return symbol_table


def cleanup_symbol_table_cache(version_id: str) -> None:
    cache_path = _get_symbol_table_cache_path(version_id)
    with contextlib.suppress(Exception):
        cache_path.unlink(missing_ok=True)


# Generic cache S3 persistence functions
# Used for large caches that need to survive container reschedules


def _get_cache_s3_key(cache_type: str, key: str) -> str:
    """Generate S3 key for a cache entry. Uses 'cache/' prefix for lifecycle rules."""
    safe_key = quote(key, safe="")
    return f"cache/{cache_type}/{safe_key}.pkl"


def _get_cache_local_path(cache_type: str, key: str) -> Path:
    """Generate local file path for cache entry."""
    safe_key = quote(key, safe="")
    return Path(f"/tmp/inspector_cache_{cache_type}_{safe_key}.pkl")


def upload_cache_to_s3(
    cache_type: str,
    key: str,
    value: Any,
    s3_client: Any,
    bucket_name: str,
) -> str:
    """Upload a cache entry to S3 with pickle serialization."""
    import os
    import tempfile

    start_time = time.time()
    s3_key = _get_cache_s3_key(cache_type, key)

    fd, temp_path = tempfile.mkstemp(suffix=".pkl")
    try:
        with os.fdopen(fd, "wb") as temp_file:
            pickle.dump(value, temp_file)

        file_size_mb = Path(temp_path).stat().st_size / (1024 * 1024)
        s3_client.upload_file(temp_path, bucket_name, s3_key)

        total_time = time.time() - start_time
        print(
            f"Cache [{cache_type}] uploaded ({file_size_mb:.2f}MB) in {total_time:.2f}s to {bucket_name}/{s3_key}"
        )
    except Exception as e:
        Path(temp_path).unlink(missing_ok=True)
        raise RuntimeError(f"Failed to upload cache [{cache_type}] to S3") from e
    finally:
        Path(temp_path).unlink(missing_ok=True)

    return s3_key


def download_cache_from_s3(
    cache_type: str,
    key: str,
    s3_client: Any,
    bucket_name: str,
) -> Any:
    """Download a cache entry from S3, with local file cache fallback."""
    cache_path = _get_cache_local_path(cache_type, key)

    # Try local file cache first (if it exists from a previous download)
    try:
        start_time = time.time()
        with open(cache_path, "rb") as f:
            value = pickle.load(f)
        file_size_mb = cache_path.stat().st_size / (1024 * 1024)
        cache_time = time.time() - start_time
        print(
            f"Cache [{cache_type}] loaded from local file ({file_size_mb:.2f}MB) in {cache_time:.2f}s"
        )
        # Delete local file after loading - we rely on in-memory cache, not local files
        cache_path.unlink(missing_ok=True)
        return value
    except FileNotFoundError:
        pass
    except Exception:
        cache_path.unlink(missing_ok=True)

    # Download from S3
    start_time = time.time()
    s3_key = _get_cache_s3_key(cache_type, key)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        s3_client.download_file(bucket_name, s3_key, str(cache_path))
    except Exception as e:
        logger.error(
            f"Error downloading cache [{cache_type}] key '{s3_key}' from S3: {e}"
        )
        raise KeyError(f"Cache [{cache_type}] key '{s3_key}' not found in S3") from e

    with open(cache_path, "rb") as f:
        value = pickle.load(f)

    file_size_mb = cache_path.stat().st_size / (1024 * 1024)
    download_time = time.time() - start_time
    print(
        f"Cache [{cache_type}] downloaded from S3 ({file_size_mb:.2f}MB) in {download_time:.2f}s"
    )
    # Delete local file after loading - we rely on in-memory cache, not local files
    cache_path.unlink(missing_ok=True)
    return value


def delete_cache_from_s3(
    cache_type: str,
    key: str,
    s3_client: Any,
    bucket_name: str,
) -> None:
    """Delete a cache entry from S3 and local file cache."""
    # Clean up local file
    cache_path = _get_cache_local_path(cache_type, key)
    with contextlib.suppress(Exception):
        cache_path.unlink(missing_ok=True)

    # Clean up S3 (best effort, don't fail if not found)
    s3_key = _get_cache_s3_key(cache_type, key)
    with contextlib.suppress(Exception):
        s3_client.delete_object(Bucket=bucket_name, Key=s3_key)
