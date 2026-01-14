import asyncio
import os
import threading
import time
import uuid
from typing import Any

from shared.inspector.inspection.files import comprehend_file_top_down
from shared.inspector.utils.dag import LiteNode


class TTLCache:
    """Thread-safe cache with TTL-based expiration"""

    def __init__(self, default_ttl_seconds: int = 28800) -> None:
        self._cache: dict[str, tuple[Any, float]] = {}
        self._lock = threading.Lock()
        self._default_ttl = default_ttl_seconds

    def put(self, key: str, value: Any, ttl_seconds: int | None = None) -> str:
        now = time.time()
        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl
        expires = now + ttl

        with self._lock:
            self._evict_expired(now)
            self._cache[key] = (value, expires)

        return key

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

    def _evict_expired(self, now: float) -> None:
        keys_to_delete = [k for k, (_, exp) in self._cache.items() if exp < now]
        for k in keys_to_delete:
            del self._cache[k]


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
        """Get S3 client and bucket name from environment."""
        import boto3

        s3_client = boto3.client(
            "s3", endpoint_url=os.environ.get("AWS_S3_ENDPOINT_URL")
        )
        bucket_name = os.environ["INSPECTOR_BUCKET_NAME"]
        return s3_client, bucket_name

    def _upload_to_s3(self, key: str, value: Any) -> None:
        """Internal sync S3 upload."""
        from shared.inspector.utils.io import upload_cache_to_s3

        s3_client, bucket_name = self._get_s3_client_and_bucket()
        upload_cache_to_s3(
            cache_type=self._cache_type,
            key=key,
            value=value,
            s3_client=s3_client,
            bucket_name=bucket_name,
        )

    def _download_from_s3(self, key: str) -> Any:
        """Internal sync S3 download."""
        from shared.inspector.utils.io import download_cache_from_s3

        s3_client, bucket_name = self._get_s3_client_and_bucket()
        return download_cache_from_s3(
            cache_type=self._cache_type,
            key=key,
            s3_client=s3_client,
            bucket_name=bucket_name,
        )

    def _delete_from_s3(self, key: str) -> None:
        """Internal sync S3 delete."""
        from shared.inspector.utils.io import delete_cache_from_s3

        s3_client, bucket_name = self._get_s3_client_and_bucket()
        delete_cache_from_s3(
            cache_type=self._cache_type,
            key=key,
            s3_client=s3_client,
            bucket_name=bucket_name,
        )

    # ==================== Sync methods ====================

    def put(self, key: str, value: Any, ttl_seconds: int | None = None) -> str:
        """Put value in L1 (memory) and L2 (S3). Blocking - use aput() in async code."""
        # Write to L1
        self._l1.put(key, value, ttl_seconds)

        # Write to L2 (S3)
        try:
            self._upload_to_s3(key, value)
        except Exception as e:
            print(f"Warning: Failed to persist cache [{self._cache_type}] to S3: {e}")

        return key

    def get(self, key: str) -> Any:
        """Get from L1, falling back to L2 (S3). Blocking - use aget() in async code."""
        # Fast path: try L1 first
        try:
            return self._l1.get(key)
        except KeyError:
            pass

        # Slow path: load from S3 with coordination
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

            # Load from S3
            print(
                f"Cache [{self._cache_type}] not in memory, loading from S3 for key {key}"
            )
            value = self._download_from_s3(key)
            self._l1.put(key, value)
            return value

    def delete(self, key: str) -> None:
        """Delete from L1 and L2. Blocking - use adelete() in async code."""
        # Delete from L1
        self._l1.delete(key)

        # Delete from L2 (S3)
        try:
            self._delete_from_s3(key)
        except Exception as e:
            print(f"Warning: Failed to delete cache [{self._cache_type}] from S3: {e}")

    # ==================== Async methods ====================

    async def aput(self, key: str, value: Any, ttl_seconds: int | None = None) -> str:
        """Async put - runs S3 upload in thread to avoid blocking event loop."""
        # Write to L1 (fast, no I/O)
        self._l1.put(key, value, ttl_seconds)

        # Write to L2 (S3) in thread
        try:
            await asyncio.to_thread(self._upload_to_s3, key, value)
        except Exception as e:
            print(f"Warning: Failed to persist cache [{self._cache_type}] to S3: {e}")

        return key

    async def aget(self, key: str) -> Any:
        """Async get - runs S3 download in thread to avoid blocking event loop."""
        # Fast path: try L1 first (no I/O)
        try:
            return self._l1.get(key)
        except KeyError:
            pass

        # Slow path: load from S3 with async coordination
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

            # Load from S3 in thread
            print(
                f"Cache [{self._cache_type}] not in memory, loading from S3 for key {key}"
            )
            value = await asyncio.to_thread(self._download_from_s3, key)
            self._l1.put(key, value)
            return value

    async def adelete(self, key: str) -> None:
        """Async delete - runs S3 delete in thread to avoid blocking event loop."""
        # Delete from L1 (fast)
        self._l1.delete(key)

        # Delete from L2 (S3) in thread
        try:
            await asyncio.to_thread(self._delete_from_s3, key)
        except Exception as e:
            print(f"Warning: Failed to delete cache [{self._cache_type}] from S3: {e}")


# Create cache instances (each with its own lock)
# Symbol table uses TTLCache with coordinated S3 loading via get_or_load_symbol_table()
_symbol_table_cache = TTLCache()

# All caches are S3-backed for container reschedule resilience
_top_level_cache = S3BackedTTLCache(cache_type="top_level")
_tags_cache = S3BackedTTLCache(cache_type="tags")
_source_code_cache = S3BackedTTLCache(cache_type="source_code")
_tech_doc_output_cache = S3BackedTTLCache(cache_type="tech_doc_output")
_diff_content_cache = S3BackedTTLCache(cache_type="diff_content")
_folder_child_nodes_to_docs_cache = S3BackedTTLCache(cache_type="folder_child_nodes")

# Coordinated loading infrastructure for symbol table
# Prevents multiple concurrent downloads when cache is cold after container reschedule
_symbol_table_load_locks: dict[str, threading.Lock] = {}
_symbol_table_load_locks_guard = threading.Lock()
_symbol_table_async_load_locks: dict[str, asyncio.Lock] = {}
_symbol_table_async_load_locks_guard = asyncio.Lock()


def _download_symbol_table_sync(version_id: str) -> dict:
    """Internal sync function to download symbol table from S3."""
    import boto3
    from shared.inspector.utils.io import download_symbol_table_from_s3_with_cache

    s3_client = boto3.client("s3", endpoint_url=os.environ.get("AWS_S3_ENDPOINT_URL"))
    bucket_name = os.environ["INSPECTOR_BUCKET_NAME"]

    return download_symbol_table_from_s3_with_cache(
        s3_client=s3_client,
        bucket_name=bucket_name,
        version_id=version_id,
    )


# Public API - keeps existing interface intact
def put_symbol_table_cache(key: str, value: dict, ttl_seconds: int = 28800) -> str:
    return _symbol_table_cache.put(key, value, ttl_seconds)


def get_symbol_table_cache(key: str) -> dict:
    return _symbol_table_cache.get(key)


def delete_symbol_table_cache(key: str) -> None:
    _symbol_table_cache.delete(key)


def get_or_load_symbol_table(version_id: str) -> dict:
    """Get symbol table from cache, or load from S3 with coordination (sync version).

    This function ensures that when multiple tasks need the symbol table on a
    cold container (after reschedule), only ONE task downloads from S3 while
    others wait. This prevents memory blowup from concurrent downloads.

    Use get_or_load_symbol_table_async() in async code to avoid blocking the event loop.
    """
    # Fast path: already in cache
    try:
        return _symbol_table_cache.get(version_id)
    except KeyError:
        pass

    # Slow path: need to load from S3
    # Get or create a lock for this specific version_id
    with _symbol_table_load_locks_guard:
        if version_id not in _symbol_table_load_locks:
            _symbol_table_load_locks[version_id] = threading.Lock()
        lock = _symbol_table_load_locks[version_id]

    with lock:
        # Double-check: another thread may have loaded it while we waited
        try:
            return _symbol_table_cache.get(version_id)
        except KeyError:
            pass

        # We're the loader - download from S3
        print(f"Symbol table not in cache, loading from S3 for version {version_id}")
        symbol_table = _download_symbol_table_sync(version_id)
        _symbol_table_cache.put(version_id, symbol_table)
        print(f"Symbol table loaded and cached for version {version_id}")
        return symbol_table


async def get_or_load_symbol_table_async(version_id: str) -> dict:
    """Get symbol table from cache, or load from S3 with coordination (async version).

    This function ensures that when multiple tasks need the symbol table on a
    cold container (after reschedule), only ONE task downloads from S3 while
    others wait. Runs S3 download in thread to avoid blocking event loop.
    """
    # Fast path: already in cache
    try:
        return _symbol_table_cache.get(version_id)
    except KeyError:
        pass

    # Slow path: need to load from S3 with async coordination
    async with _symbol_table_async_load_locks_guard:
        if version_id not in _symbol_table_async_load_locks:
            _symbol_table_async_load_locks[version_id] = asyncio.Lock()
        lock = _symbol_table_async_load_locks[version_id]

    async with lock:
        # Double-check: another coroutine may have loaded it while we waited
        try:
            return _symbol_table_cache.get(version_id)
        except KeyError:
            pass

        # We're the loader - download from S3 in thread to avoid blocking
        print(f"Symbol table not in cache, loading from S3 for version {version_id}")
        symbol_table = await asyncio.to_thread(_download_symbol_table_sync, version_id)
        _symbol_table_cache.put(version_id, symbol_table)
        print(f"Symbol table loaded and cached for version {version_id}")
        return symbol_table


# Top level cache - S3-backed (sync versions)
def put_top_level_cache(key: str, value: dict, ttl_seconds: int = 28800) -> str:
    return _top_level_cache.put(key, value, ttl_seconds)


def get_top_level_cache(key: str) -> dict:
    return _top_level_cache.get(key)


def delete_top_level_cache(key: str) -> None:
    _top_level_cache.delete(key)


# Top level cache - async versions
async def put_top_level_cache_async(
    key: str, value: dict, ttl_seconds: int = 28800
) -> str:
    return await _top_level_cache.aput(key, value, ttl_seconds)


async def get_top_level_cache_async(key: str) -> dict:
    return await _top_level_cache.aget(key)


async def delete_top_level_cache_async(key: str) -> None:
    await _top_level_cache.adelete(key)


# Tags cache - S3-backed (sync versions)
def put_tags_cache(key: str, value: dict, ttl_seconds: int = 28800) -> str:
    return _tags_cache.put(key, value, ttl_seconds)


def get_tags_cache(key: str) -> dict:
    return _tags_cache.get(key)


def delete_tags_cache(key: str) -> None:
    _tags_cache.delete(key)


# Tags cache - async versions
async def put_tags_cache_async(key: str, value: dict, ttl_seconds: int = 28800) -> str:
    return await _tags_cache.aput(key, value, ttl_seconds)


async def get_tags_cache_async(key: str) -> dict:
    return await _tags_cache.aget(key)


async def delete_tags_cache_async(key: str) -> None:
    await _tags_cache.adelete(key)


# Diff content cache - S3-backed (sync versions)
def put_diff_content_cache(key: str, value: dict, ttl_seconds: int = 28800) -> str:
    return _diff_content_cache.put(key, value, ttl_seconds)


def get_diff_content_cache(key: str) -> dict:
    return _diff_content_cache.get(key)


def delete_diff_content_cache(key: str) -> None:
    _diff_content_cache.delete(key)


# Diff content cache - async versions for async code
async def put_diff_content_cache_async(
    key: str, value: dict, ttl_seconds: int = 28800
) -> str:
    return await _diff_content_cache.aput(key, value, ttl_seconds)


async def get_diff_content_cache_async(key: str) -> dict:
    return await _diff_content_cache.aget(key)


async def delete_diff_content_cache_async(key: str) -> None:
    await _diff_content_cache.adelete(key)


# Source code cache - S3-backed (sync versions)
def put_source_code_cache(key: str, value: str, ttl_seconds: int = 28800) -> str:
    return _source_code_cache.put(key, value, ttl_seconds)


def get_source_code_cache(key: str) -> str:
    return _source_code_cache.get(key)


def delete_source_code_cache(key: str) -> None:
    _source_code_cache.delete(key)


# Source code cache - async versions
async def put_source_code_cache_async(
    key: str, value: str, ttl_seconds: int = 28800
) -> str:
    return await _source_code_cache.aput(key, value, ttl_seconds)


async def get_source_code_cache_async(key: str) -> str:
    return await _source_code_cache.aget(key)


async def delete_source_code_cache_async(key: str) -> None:
    await _source_code_cache.adelete(key)


# Tech doc output cache - S3-backed (sync versions)
def put_tech_doc_output_cache(key: str, value: dict, ttl_seconds: int = 28800) -> str:
    return _tech_doc_output_cache.put(key, value, ttl_seconds)


def get_tech_doc_output_cache(key: str) -> dict:
    return _tech_doc_output_cache.get(key)


def delete_tech_doc_output_cache(key: str) -> None:
    _tech_doc_output_cache.delete(key)


# Tech doc output cache - async versions
async def put_tech_doc_output_cache_async(
    key: str, value: dict, ttl_seconds: int = 28800
) -> str:
    return await _tech_doc_output_cache.aput(key, value, ttl_seconds)


async def get_tech_doc_output_cache_async(key: str) -> dict:
    return await _tech_doc_output_cache.aget(key)


async def delete_tech_doc_output_cache_async(key: str) -> None:
    await _tech_doc_output_cache.adelete(key)


# Folder child nodes cache - S3-backed (sync versions)
def put_folder_child_nodes_to_docs_cache(
    key: str, value: dict, ttl_seconds: int = 28800
) -> str:
    return _folder_child_nodes_to_docs_cache.put(key, value, ttl_seconds)


def get_folder_child_nodes_to_docs_cache(key: str) -> dict:
    return _folder_child_nodes_to_docs_cache.get(key)


def delete_folder_child_nodes_to_docs_cache(key: str) -> None:
    _folder_child_nodes_to_docs_cache.delete(key)


# Folder child nodes cache - async versions for async code
async def put_folder_child_nodes_to_docs_cache_async(
    key: str, value: dict, ttl_seconds: int = 28800
) -> str:
    return await _folder_child_nodes_to_docs_cache.aput(key, value, ttl_seconds)


async def get_folder_child_nodes_to_docs_cache_async(key: str) -> dict:
    return await _folder_child_nodes_to_docs_cache.aget(key)


async def delete_folder_child_nodes_to_docs_cache_async(key: str) -> None:
    await _folder_child_nodes_to_docs_cache.adelete(key)


def make_tech_doc(
    node: LiteNode,
    codebase_name: str,
    version_id: str,
) -> tuple[bool, dict, LiteNode]:
    from shared.agent.chat_openai import ChatOpenAI

    print(f"Processing tech docs ({node})")

    raise_hard_errors = False
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        request_timeout=FILE_TECH_DOC_LLM_TIMEOUT,
    )

    full_symbol_table = get_or_load_symbol_table(version_id)

    source_code = get_source_code_cache(f"{version_id}:{node.root_rel_path}")
    reified_symbols = full_symbol_table.get(node.root_rel_path, None)

    file_docs_successful, file_doc = comprehend_file_top_down(
        llm=llm,
        node=node,
        source_code=source_code,
        codebase_name=codebase_name,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        compression_loop_max_itr=COMPRESSION_LOOP_MAX_ITR,
        max_num_chunks=MAX_NUM_CHUNKS_FILE,
        reified_symbols=reified_symbols,
        raise_hard_errors=raise_hard_errors,
    )
    print(f"Tech docs created for ({node})")
    return {"success": file_docs_successful, "file_doc": file_doc, "node": node}


def make_symbol_docs(
    node: LiteNode,
    source_code: str,
    file_description_paragraph: str,
    symbol_count_limit: int | None = None,
) -> list[dict[str, any]]:
    from shared.inspector.inspection.symbols import document_symbols_in_file

    print(f"Processing symbol docs ({node})")
    symbols = document_symbols_in_file(
        file_node=node,
        source_code=source_code,
        file_description_paragraph=file_description_paragraph,
        symbol_count_limit=symbol_count_limit,
    )
    print(f"Symbol docs created for {node}")
    return {"symbols": symbols}


def make_folder_tech_doc(
    codebase_name: str,
    node: LiteNode,
    child_nodes_to_docs: dict[LiteNode, dict],
    previous_content: dict[str, str] | None = None,
) -> dict[str, any]:
    from shared.agent.chat_openai import ChatOpenAI
    from shared.inspector.inspection.folders import comprehend_folder_top_down

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        request_timeout=FOLDER_TECH_DOC_LLM_TIMEOUT,
    )

    print(f"Processing folder tech docs for ({node.root_rel_path.name})")
    folder_docs = comprehend_folder_top_down(
        llm=llm,
        codebase_name=codebase_name,
        node=node,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        max_workers=1,
        child_nodes_to_docs=child_nodes_to_docs,
        compression_loop_max_itr=COMPRESSION_LOOP_MAX_ITR,
        previous_content=previous_content,
    )
    print(f"Folder tech docs created for ({node})")
    return folder_docs


def make_toplevel_tech_docs(
    codebase_name: str, nodes_to_docs: dict[LiteNode, dict]
) -> dict[str, any]:
    from shared.agent.chat_openai import ChatOpenAI
    from shared.inspector.inspection.toplevel import comprehend_codebase_top_down

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        request_timeout=TOP_LEVEL_DOC_LLM_TIMEOUT,
    )

    print(f"Processing top-level docs for `{codebase_name}`")
    top_level_docs = comprehend_codebase_top_down(
        llm=llm,
        docs=nodes_to_docs,
        codebase_name=codebase_name,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        max_workers=20,
        include_exploratory_generation=False,
        compression_loop_max_itr=COMPRESSION_LOOP_MAX_ITR,
    )
    print(f"Processed top-level docs for `{codebase_name}`")
    return top_level_docs


def make_codebase_tags(
    codebase_name: str,
    nodes_to_docs: dict[LiteNode, dict],
    content_kinds: set,
) -> dict[str, any]:
    from shared.agent.chat_openai import ChatOpenAI
    from shared.inspector.inspection.toplevel import tag_codebase

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        request_timeout=TOP_LEVEL_DOC_LLM_TIMEOUT,
    )

    print(f"Processing tags for `{codebase_name}`")
    tags = tag_codebase(
        llm=llm,
        docs=nodes_to_docs,
        content_kinds=content_kinds,
    )
    print(f"Processed tags for `{codebase_name}`")
    return tags


def export_tech_docs_to_zip(
    version_id: uuid.UUID,
    install_id: str | None = None,
) -> None:
    import hashlib
    import tempfile
    from pathlib import Path
    from shutil import make_archive

    import boto3
    from database.db import engine
    from database.models import DerivedContent, Node, Version, VersionNode
    from database.models_enums import ContentKind, NodeKind
    from shared.inspector.utils.export_utils import (
        replace_driver_compatible_links_with_markdown_links,
    )
    from sqlalchemy.orm import selectinload
    from sqlmodel import Session, select

    with Session(engine) as session:
        long_desc_query = (
            select(
                VersionNode,
                Node,
                DerivedContent,
                # Node.relative_path, DerivedContent.content, Node.kind, Node.depth
            )
            .join(Node, VersionNode.node_id == Node.id)
            .join(DerivedContent, DerivedContent.node_id == Node.id)
            .where(
                VersionNode.version_id == version_id,
                DerivedContent.content_kind == ContentKind.LONG_DESCRIPTION,
            )
        )

        short_desc_query = (
            select(
                VersionNode,
                Node,
                DerivedContent,
            )
            .join(Node, VersionNode.node_id == Node.id)
            .join(DerivedContent, DerivedContent.node_id == Node.id)
            .where(
                VersionNode.version_id == version_id,
                DerivedContent.content_kind == ContentKind.SHORT_SENTENCE_DESCRIPTION,
            )
        )

        version_query = (
            select(Version)
            .where(Version.id == version_id)
            .options(
                selectinload(Version.primary_asset),
            )
        )
        version_result = session.exec(version_query)
        version_row = version_result.one()
        primary_asset_id = version_row.primary_asset_id
        auto_commit_docs = version_row.primary_asset.codebase_settings_auto_commit_docs
        org_id = version_row.primary_asset.organization_id
        org_id_hash = hashlib.sha256(org_id.encode()).hexdigest()[:63]

        long_desc_result = session.exec(long_desc_query)
        long_desc_rows = long_desc_result.all()

        short_desc_result = session.exec(short_desc_query)
        short_desc_rows = short_desc_result.all()

        node_to_short_desc = {
            node.id: derived_content for _, node, derived_content in short_desc_rows
        }

        node_path_to_kind = {
            Path(version_node.relative_path): node.kind
            for version_node, node, _ in long_desc_rows
        }
    with (
        tempfile.TemporaryDirectory() as temp_dir,
    ):
        for version_node, node, long_desc_dc in long_desc_rows:
            node_path = Path(version_node.relative_path)
            if node.kind == NodeKind.CODEBASE_FILE:
                link_destination_path = node_path.with_suffix(node_path.suffix + ".md")
                file_path = Path(temp_dir) / link_destination_path
            elif node.kind == NodeKind.CODEBASE_DIRECTORY:
                if node_path.name == ".github":
                    # NOTE: we special case .github here, because Github priotizes displaying
                    # the README.md file from the .github folder over the README.md file in the root of the repo
                    # See: https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes
                    link_destination_path = node_path / "README_.md"
                    file_path = Path(temp_dir) / link_destination_path
                else:
                    link_destination_path = node_path / "README.md"
                    file_path = Path(temp_dir) / link_destination_path
                # doc_file_path = node_path.with_suffix(".driver.md")

            short_desc_dc = node_to_short_desc.get(node.id)

            # Combine short and long descriptions as we do in the frontend display
            content = ""
            if short_desc_dc and short_desc_dc.content:
                content += short_desc_dc.content + "\n\n"
            content += long_desc_dc.content

            content = replace_driver_compatible_links_with_markdown_links(
                content,
                Path(*link_destination_path.parts[1:]),
                node_path.suffix,
                node_path_to_kind,
            )
            file_path.parent.mkdir(parents=True, exist_ok=True)
            comment = (
                "<!--------------------------------------------------------------------------------->\n"
                "<!-- IMPORTANT: This file is auto-generated by Driver (https://driver.ai). -------->\n"
                "<!-- Manual edits may be overwritten on future commits. --------------------------->\n"
                "<!--------------------------------------------------------------------------------->\n\n"
            )
            end_comment = "\n---\nMade with ❤️ by [Driver](https://www.driver.ai/)"
            content = comment + content + end_comment
            file_path.write_text(content)
        make_archive(f"{version_id}_tech_docs", "zip", Path(temp_dir))

        s3_resource = boto3.resource(
            "s3", endpoint_url=os.environ.get("AWS_S3_ENDPOINT_URL")
        )
        s3_dest = f"{primary_asset_id}/{version_id}/{version_id}_tech_docs.zip"
        s3_bucket = s3_resource.Bucket(org_id_hash)
        if install_id is not None:
            s3_bucket.upload_file(
                Path(f"{version_id}_tech_docs.zip"),
                s3_dest,
                ExtraArgs={"Metadata": {"install_id": install_id}},
            )
        else:
            s3_bucket.upload_file(
                Path(f"{version_id}_tech_docs.zip"),
                s3_dest,
            )
        print(f"Uploaded tech docs zip to S3: {s3_dest}")
        if auto_commit_docs:
            if install_id is not None:
                print("PRing exported docs")
                from shared.inspector.onboarding.push_bot import push_docs

                push_docs(version_id)
            else:
                # TODO: better handling of install_id rather than attaching to S3 metadata
                print("Unable to PR - install id is not available")
        else:
            print("PR disabled for this codebase.")
        # Delete zip after upload
        os.remove(f"{version_id}_tech_docs.zip")


CHUNK_SIZE = 64_000
CHUNK_OVERLAP = 3_000
COMPRESSION_LOOP_MAX_ITR = 10
MAX_NUM_CHUNKS_FILE = 10
FILE_TECH_DOC_LLM_TIMEOUT = 500
FOLDER_TECH_DOC_LLM_TIMEOUT = 500
TOP_LEVEL_DOC_LLM_TIMEOUT = 500
