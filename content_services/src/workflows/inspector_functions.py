import uuid
from datetime import timedelta
from pathlib import Path
from typing import Any

from hatchet_client import hatchet
from hatchet_sdk import Context, remove_null_unicode_character
from hatchet_sdk.runnables.types import (
    ConcurrencyExpression,
    ConcurrencyLimitStrategy,
)
from inspector.src.deep_context_docs import deep_context_docs
from inspector.src.hatchet_funcs import (
    diff_content_cache,
    export_tech_docs_to_zip,
    folder_child_nodes_cache,
    make_codebase_tags,
    make_folder_tech_doc,
    make_symbol_docs,
    make_tech_doc,
    make_toplevel_tech_docs,
    tags_cache,
    tech_doc_output_cache,
    top_level_cache,
)
from pydantic import BaseModel
from shared.inspector.utils.dag import LiteNode, NodeKind
from shared.inspector.utils.synthesis.deep_context import (
    DeepContextDoc,
)


class TechDocInput(BaseModel):
    node: LiteNode
    codebase_name: str
    version_id: str


class FolderDocInput(BaseModel):
    node: LiteNode
    codebase_name: str
    version_id: str
    previous_content: dict[str, str] | None


class SymbolDocInput(BaseModel):
    node: LiteNode
    source_code: str
    file_description_paragraph: str
    symbol_count_limit: int | None


class TopLevelDocInput(BaseModel):
    codebase_name: str
    version_node_id: str


class DeepContextDocsInput(BaseModel):
    old_version_id: uuid.UUID | None
    old_version_content: list | None
    new_version_id: uuid.UUID
    install_id: str | None


class ExportDocsInput(BaseModel):
    version_id: uuid.UUID
    install_id: str | None


class CodebaseTagsInput(BaseModel):
    codebase_name: str
    version_node_id: str
    content_kinds: set


@hatchet.task(
    name="codebase-tags-workflow",
    execution_timeout=timedelta(minutes=60),
    concurrency=ConcurrencyExpression(
        max_runs=1,
        expression="'codebase-tags-workflow'",  # NOTE: must be a string literal to be evaluated as a constant task name
        limit_strategy=ConcurrencyLimitStrategy.GROUP_ROUND_ROBIN,
    ),
)
def codebase_tags_task(input: CodebaseTagsInput, ctx: Context) -> dict[str, str]:
    print("starting codebase tags task")
    # nodes_to_docs = {}
    # for node, doc in input.nodes_to_docs:
    #     node_kind = NodeKind(node["kind"])
    #     node_root_rel_path = Path(node["root_rel_path"])
    #     node_status = node["status"]
    #     node = LiteNode(
    #         kind=node_kind,
    #         root_rel_path=node_root_rel_path,
    #         status=node_status,
    #     )
    #     nodes_to_docs[node] = doc
    nodes_to_docs = tags_cache.get(input.version_node_id)
    tags = make_codebase_tags(
        input.codebase_name,
        nodes_to_docs,
        input.content_kinds,
    )
    tags_cache.delete(input.version_node_id)
    print("executed codebase tags task")
    return tags


@hatchet.task(
    name="export-tech-docs-workflow",
    execution_timeout=timedelta(minutes=15),
    concurrency=ConcurrencyExpression(
        max_runs=5,
        expression="'export-tech-docs-workflow'",  # NOTE: must be a string literal to be evaluated as a constant task name
        limit_strategy=ConcurrencyLimitStrategy.GROUP_ROUND_ROBIN,
    ),
)
def export_tech_docs_task(input: ExportDocsInput, ctx: Context) -> dict[str, str]:
    print("starting export tech docs task")
    # Call the function to export tech docs to zip
    export_tech_docs_to_zip(
        input.version_id,
        input.install_id,
    )
    print("executed export tech docs task")
    return {"status": "export complete"}


@hatchet.task(
    name="deep-context-docs-workflow",
    execution_timeout=timedelta(minutes=480),
    concurrency=ConcurrencyExpression(
        max_runs=3,
        expression="'deep-context-docs-workflow'",  # NOTE: must be a string literal to be evaluated as a constant task name
        limit_strategy=ConcurrencyLimitStrategy.GROUP_ROUND_ROBIN,
    ),
)
async def deep_context_docs_task(input: DeepContextDocsInput, ctx: Context) -> dict:
    print("starting deep context docs task")
    # Call the function to generate deep context docs
    # Reconstruct the old version content and diff dag
    old_version_content = None
    if input.old_version_content is not None:
        old_version_content = []
        for doc in input.old_version_content:
            old_version_content.append(DeepContextDoc.model_validate(doc))
    code_diff = None
    if input.old_version_id is not None:
        code_diff = diff_content_cache.get(input.old_version_id)

    await deep_context_docs(
        input.old_version_id,
        old_version_content,
        code_diff,
        input.new_version_id,
        input.install_id,
    )
    if input.old_version_id is not None:
        diff_content_cache.delete(input.old_version_id)
    print("executed deep context docs task")
    return {"status": "completed"}


@hatchet.task(
    name="tech-doc-workflow",
    execution_timeout=timedelta(minutes=180),
    concurrency=ConcurrencyExpression(
        max_runs=72,
        expression="'tech-doc-workflow'",  # NOTE: must be a string literal to be evaluated as a constant task name
        limit_strategy=ConcurrencyLimitStrategy.GROUP_ROUND_ROBIN,
    ),
    schedule_timeout=timedelta(minutes=60),
)
def tech_doc_task(input: TechDocInput, ctx: Context) -> dict[str, str]:
    print("starting tech doc task")
    # Call the function to generate tech docs
    node_kind = NodeKind(input.node["kind"])
    node_root_rel_path = Path(input.node["root_rel_path"])
    node_status = input.node["status"]
    node = LiteNode(
        kind=node_kind,
        root_rel_path=node_root_rel_path,
        status=node_status,
    )
    tech_docs = make_tech_doc(
        node,
        input.codebase_name,
        input.version_id,
    )
    cleaned_tech_docs = remove_null_unicode_character(data=tech_docs)
    tech_doc_output_cache.put(
        f"{input.version_id}:{node.root_rel_path}", cleaned_tech_docs
    )
    print("executed tech doc task")
    return {"success": True}


@hatchet.task(
    name="folder-doc-workflow",
    execution_timeout=timedelta(minutes=30),
    concurrency=ConcurrencyExpression(
        max_runs=60,
        expression="'folder-doc-workflow'",  # NOTE: must be a string literal to be evaluated as a constant task name
        limit_strategy=ConcurrencyLimitStrategy.GROUP_ROUND_ROBIN,
    ),
)
def folder_doc_task(input: FolderDocInput, ctx: Context) -> dict[str, str]:
    print("starting folder doc task")
    # Call the function to generate folder docs

    node_kind = NodeKind(input.node["kind"])
    node_root_rel_path = Path(input.node["root_rel_path"])
    node_status = input.node["status"]
    node = LiteNode(
        kind=node_kind,
        root_rel_path=node_root_rel_path,
        status=node_status,
    )
    print(node.root_rel_path.name)
    child_nodes_to_docs = folder_child_nodes_cache.get(
        f"{input.version_id}:{node.root_rel_path}"
    )
    folder_docs = make_folder_tech_doc(
        input.codebase_name,
        node,
        child_nodes_to_docs,
        input.previous_content,
    )
    cleaned_folder_docs = remove_null_unicode_character(data=folder_docs)
    tech_doc_output_cache.put(
        f"{input.version_id}:{node.root_rel_path}", cleaned_folder_docs
    )
    print("executed folder doc task")
    return {"success": True}


@hatchet.task(
    name="symbol-doc-workflow",
    execution_timeout=timedelta(minutes=120),
    concurrency=ConcurrencyExpression(
        max_runs=72,
        expression="'symbol-doc-workflow'",  # NOTE: must be a string literal to be evaluated as a constant task name
        limit_strategy=ConcurrencyLimitStrategy.GROUP_ROUND_ROBIN,
    ),
)
def symbol_doc_task(input: SymbolDocInput, ctx: Context) -> list[dict[str, Any]]:
    print("starting symbol doc task")
    # Call the function to generate symbol docs
    node_kind = NodeKind(input.node["kind"])
    node_root_rel_path = Path(input.node["root_rel_path"])
    node_status = input.node["status"]
    node = LiteNode(
        kind=node_kind,
        root_rel_path=node_root_rel_path,
        status=node_status,
    )
    symbol_docs = make_symbol_docs(
        node,
        input.source_code,
        input.file_description_paragraph,
        input.symbol_count_limit,
    )
    print("executed symbol doc task")
    return symbol_docs


@hatchet.task(
    name="toplevel-doc-workflow",
    execution_timeout=timedelta(minutes=60),
    concurrency=ConcurrencyExpression(
        max_runs=3,
        expression="'toplevel-doc-workflow'",  # NOTE: must be a string literal to be evaluated as a constant task name
        limit_strategy=ConcurrencyLimitStrategy.GROUP_ROUND_ROBIN,
    ),
)
def toplevel_doc_task(input: TopLevelDocInput, ctx: Context) -> dict[str, Any]:
    print("starting toplevel doc task")
    # Call the function to generate toplevel docs
    # nodes_to_docs = {}
    # for node, doc in input.nodes_to_docs:
    #     node_kind = NodeKind(node["kind"])
    #     node_root_rel_path = Path(node["root_rel_path"])
    #     node_status = node["status"]
    #     node = LiteNode(
    #         kind=node_kind,
    #         root_rel_path=node_root_rel_path,
    #         status=node_status,
    #     )
    #     nodes_to_docs[node] = doc
    nodes_to_docs = top_level_cache.get(input.version_node_id)
    toplevel_docs = make_toplevel_tech_docs(
        input.codebase_name,
        nodes_to_docs,
    )
    top_level_cache.delete(input.version_node_id)
    print("executed toplevel doc task")
    cleaned_top_level_docs = remove_null_unicode_character(data=toplevel_docs)
    return cleaned_top_level_docs
