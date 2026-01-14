import asyncio
import hashlib
import os
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from pathlib import Path
from uuid import UUID

from hatchet_sdk import TriggerWorkflowOptions
from shared.inspector.utils.dag import (
    FileTreeDag,
    Node,
    NodeKind,
    NodeStatus,
)
from shared.inspector.utils.task import TaskManager

from .hatchet_funcs import delete_symbol_table_cache, put_diff_content_cache_async
from .tasks import (
    CodebaseTaggingTask,
    CSymbolTableTask,
    EmbeddingTask,
    EmbeddingTaskType,
    FileTechDocTask,
    FolderTechDocTask,
    TopLevelDocsTask,
)

# TODO considering using concurrent inputs when we're just calling open AI. This should
# save some cost (though costs are negligible today)

TECH_DOC_THREAD_POOL = ThreadPoolExecutor(max_workers=2)


class InspectionMode(Enum):
    NORMAL = "normal"
    RESUME = "resume"
    RERUN = "rerun"

    @classmethod
    def from_str(cls, mode_str: str) -> "InspectionMode":
        try:
            return cls(mode_str.lower())
        except ValueError:
            valid_modes = ", ".join([mode.value for mode in cls])
            raise ValueError(
                f"Invalid mode '{mode_str}'. Must be one of: {valid_modes}."
            )


async def get_result_loading_config(
    inspection_mode: InspectionMode,
    version_id: uuid.UUID,
    previous_version_id: uuid.UUID | None = None,
) -> list[tuple[uuid.UUID, set[NodeStatus]]]:
    from shared.inspector.utils.db import try_get_latest_run_from_version_id

    result_loading_config = []
    is_diff = previous_version_id is not None

    match inspection_mode:
        case InspectionMode.NORMAL:
            existing_run_id_for_prev_version = (
                await try_get_latest_run_from_version_id(previous_version_id)
                if is_diff
                else None
            )
            existing_run_id_for_current_version = None
        case InspectionMode.RESUME:
            existing_run_id_for_prev_version = (
                await try_get_latest_run_from_version_id(previous_version_id)
                if is_diff
                else None
            )
            existing_run_id_for_current_version = (
                await try_get_latest_run_from_version_id(version_id)
            )
        case InspectionMode.RERUN:
            existing_run_id_for_prev_version = (
                await try_get_latest_run_from_version_id(previous_version_id)
                if is_diff
                else None
            )
            existing_run_id_for_current_version = None
        case _:
            raise ValueError("Invalid inspection mode")

    if existing_run_id_for_prev_version:
        result_loading_config.append(
            (existing_run_id_for_prev_version, {NodeStatus.UNMODIFIED})
        )
    if existing_run_id_for_current_version:
        result_loading_config.append(
            (existing_run_id_for_current_version, set(NodeStatus))
        )

    return result_loading_config


async def inspect_db(
    version_id: uuid.UUID,
    inspection_mode: InspectionMode = InspectionMode.RESUME,
) -> None:
    import tempfile

    import boto3
    from database.models_enums import NodeKind as DbNodeKind
    from database.models_enums import VersionStatus

    # from .modal_funcs import export_tech_docs_to_zip
    from shared.inspector.onboarding.onboard_utils import (
        process_and_upload_all_files_in_parallel,
        set_codebase_status,
        unpack_archive_to_finalized_path,
    )
    from shared.inspector.utils.db import (
        create_inspector_run,
        get_analyzable_version_nodes_by_version_id,
        get_version_by_id,
        try_get_prev_version,
    )
    from shared.inspector.utils.git_diff import (
        CodeDiffParams,
        InsufficientBalanceError,
        compute_and_log_code_diff_size_in_bytes,
    )
    from shared.inspector.utils.io import (
        download_all_source_files_in_parallel,
    )
    # from .utils.synthesis.deep_context import DeepContextDoc, DeepContextDocKind

    try:
        # Get the Version and check if it has previous_version_id
        version = await get_version_by_id(version_id)

        if version.status == VersionStatus.GENERATION_ERROR:
            set_codebase_status(version_id, VersionStatus.GENERATING)

        org_id = version.primary_asset.organization_id
        org_hashed_id = hashlib.sha256(org_id.encode()).hexdigest()[:63]

        previous_version = await try_get_prev_version(version_id)
        previous_version_id = previous_version.id if previous_version else None
        previous_version_root_version_node_id = (
            previous_version.root_version_node.id if previous_version else None
        )
        flat_topo_file_diff_dag = None

        codebase_name = version.primary_asset.display_name

        # TODO: rethink result loading config, may end up only persisting the symbol table task to S3
        result_loading_config = await get_result_loading_config(
            inspection_mode, version_id, previous_version_id
        )
        print("Result loading config: ", result_loading_config)

        run_id = await create_inspector_run(version_id)

        # Get content records for version_id
        db_file_version_nodes = await get_analyzable_version_nodes_by_version_id(
            version_id, {DbNodeKind.CODEBASE_FILE}
        )

        db_all_codebase_version_nodes = (
            await get_analyzable_version_nodes_by_version_id(
                version_id, {DbNodeKind.CODEBASE_FILE, DbNodeKind.CODEBASE_DIRECTORY}
            )
        )

        # Get content records for previous_version_id if available
        if previous_version is not None:
            db_previous_file_version_nodes = (
                await get_analyzable_version_nodes_by_version_id(
                    previous_version_id, {DbNodeKind.CODEBASE_FILE}
                )
            )
        # Download s3 for version_id (and previous if available)
        s3_client = boto3.client(
            "s3", endpoint_url=os.environ.get("AWS_S3_ENDPOINT_URL")
        )
        with (
            tempfile.TemporaryDirectory() as download_dir,
            tempfile.TemporaryDirectory() as previous_download_dir,
        ):
            download_root = Path(download_dir)
            if (
                version.status == VersionStatus.CONNECTED
                or version.status == VersionStatus.GENERATING
            ):
                # TODO: check usage before switching to generating
                # if it's in the connected state, must upload the individual files to S3
                download_archive_key = (
                    f"{version.primary_asset_id}/{version_id}/{version_id}_source.zip"
                )
                download_path = Path(download_dir) / f"{version_id}.zip"
                print(f"downloading zip to {download_path}")
                metadata = await asyncio.to_thread(
                    s3_client.head_object,
                    Bucket=org_hashed_id,
                    Key=download_archive_key,
                )
                install_id = metadata["Metadata"].get("install_id")
                await asyncio.to_thread(
                    s3_client.download_file,
                    org_hashed_id,
                    download_archive_key,
                    download_path,
                )

                extracted_path = await asyncio.to_thread(
                    unpack_archive_to_finalized_path,
                    archive_path=download_path,
                    extraction_root=Path(download_dir),
                    override_codebase_name=codebase_name,
                )
                print(f"Extracted archive to {extracted_path}")

                db_version_node_paths = {
                    version_node.relative_path for version_node in db_file_version_nodes
                }
                file_paths = await asyncio.to_thread(
                    process_and_upload_all_files_in_parallel,
                    s3_client=s3_client,
                    org_hashed_id=org_hashed_id,
                    primary_asset_id=version.primary_asset_id,
                    version_id=version_id,
                    extracted_path=extracted_path,
                    download_dir=download_dir,
                    db_node_paths=db_version_node_paths,
                    max_workers=10,
                )

                if version.status == VersionStatus.CONNECTED:
                    set_codebase_status(version_id, VersionStatus.GENERATING)
            else:
                print("Downloading all source files for codebase from s3...")
                file_paths = await asyncio.to_thread(
                    download_all_source_files_in_parallel,
                    s3_client=s3_client,
                    bucket_name=org_hashed_id,
                    primary_asset_id=str(version.primary_asset.id),
                    version_id=str(version_id),
                    node_rel_paths=[
                        node.relative_path for node in db_file_version_nodes
                    ],
                    download_root=download_root,
                    max_workers=8,
                )
                install_id = None  # TODO: install id is attached to the zip, and is not available on rerun/resume
                # NOTE: can still achieve PR of docs by running export_tech_docs_to_zip manually with install_id via local entrypoint
                print("Download complete")

            codebase_dag: FileTreeDag = build_dag(
                root_path=download_root, file_paths=file_paths
            )

            print("======= Nodes from current codebase processed =======")
            for node in codebase_dag.topological_sort():
                print(node.root_rel_path, node.status)

            db_all_codebase_prev_version_nodes = None
            if previous_version is not None:
                # TODO: given new snapshot model, is this how we still want to charge for bytes used?
                # If so - we still need to do all of this stuff
                previous_download_root = Path(previous_download_dir)
                print("Downloading all source files for previous codebase from s3...")
                previous_file_paths = await asyncio.to_thread(
                    download_all_source_files_in_parallel,
                    s3_client=s3_client,
                    bucket_name=org_hashed_id,
                    primary_asset_id=str(previous_version.primary_asset.id),
                    version_id=str(previous_version.id),
                    node_rel_paths=[
                        prev_node.relative_path
                        for prev_node in db_previous_file_version_nodes
                    ],
                    download_root=previous_download_root,
                    max_workers=8,
                )
                db_all_codebase_prev_version_nodes = (
                    await get_analyzable_version_nodes_by_version_id(
                        previous_version.id,
                        {DbNodeKind.CODEBASE_FILE, DbNodeKind.CODEBASE_DIRECTORY},
                    )
                )
                print("Download complete for previous version of code")

                previous_codebase_dag: FileTreeDag = build_dag(
                    root_path=previous_download_root,
                    file_paths=previous_file_paths,
                )
                print("======= Nodes from previous codebase =======")
                for node in previous_codebase_dag.topological_sort():
                    print(node.root_rel_path, node.status)

                diff_dag = codebase_dag.compute_diff(
                    previous_codebase_dag, delete_file_nodes=False
                )

                # TODO: this is effectively computing the diff dag twice (this calls `compute_diff` underneath the hood).
                flat_topo_file_diff_dag = codebase_dag.into_flat_diff_dag(
                    old=previous_codebase_dag
                )
                print("Diff dag computed")

                print("======= Nodes from diff dag =======")
                for node in diff_dag.topological_sort():
                    print(node.root_rel_path, node.status)

                print("======= Computing diff size in bytes =======")
                changed_nodes = diff_dag.topological_sort(
                    changed_nodes_only=True, files_only=True
                )
                try:
                    # @andrew: We calculate the diff size in bytes, log it while not turning on billing for code diffs
                    compute_and_log_code_diff_size_in_bytes(
                        CodeDiffParams(
                            codebase_name=codebase_name,
                            version_id=str(version.id),
                            primary_asset_id=str(version.primary_asset_id),
                            org_id=org_id,
                            previous_download_root=previous_download_root,
                            download_root=download_root,
                            changed_nodes=changed_nodes,
                        )
                    )
                except InsufficientBalanceError as ibe:
                    print(
                        f"Insufficient balance for org {org_id} to process codebase {codebase_name} {ibe}"
                    )
                    set_codebase_status(version_id, VersionStatus.INSUFFICIENT_BALANCE)
                    # I chose to return here vs re-raising the error because it will get caught and swalloed by the outer try/catch
                    return

            sorted_nodes = codebase_dag.topological_sort()
            path_to_db_node_id = {
                Path(db_node.relative_path): db_node.id
                for db_node in db_all_codebase_version_nodes
            }
            version_node_id_to_node_id = {
                db_version_node.id: db_version_node.node_id
                for db_version_node in db_all_codebase_version_nodes
            }

            nodes_with_id: list[tuple[Node, uuid.UUID | None]] = [
                (node, path_to_db_node_id[node.root_rel_path])
                for node in sorted_nodes
                if node.root_rel_path != Path(".") and node.status != NodeStatus.REMOVED
            ]
            prev_version_path_to_db_node_id = (
                {
                    Path(db_node.relative_path): db_node.id
                    for db_node in db_all_codebase_prev_version_nodes
                }
                if db_all_codebase_prev_version_nodes
                else {}
            )

            print("======= Nodes with source content id =======")
            for node, sc_id in nodes_with_id:
                print(node.root_rel_path, sc_id)

            await inspect_files(
                version_id=version_id,
                codebase_root=download_root,
                nodes_with_id=nodes_with_id,
                codebase_name=codebase_name,
                run_id=run_id,
                result_loading_config=result_loading_config,
                rel_path_to_previous_version_db_node_ids=prev_version_path_to_db_node_id,
                version_node_id_to_node_id=version_node_id_to_node_id,
            )
    except Exception as e:
        exception_type = type(e).__name__
        exc_tb = e.__traceback__
        filename = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        exception_details = (
            f"Exception type: {exception_type}\nFile: {filename}\nLine: {line_number}"
        )
        print(f"Error while processing version {version_id}: {e}")
        send_exception_email(exception_details)
        set_codebase_status(version_id, VersionStatus.GENERATION_ERROR)
        delete_symbol_table_cache(version_id)
        raise
    else:
        delete_symbol_table_cache(version_id)
        from database.models_enums import ContentKind
        from shared.inspector.utils.db import get_all_derived_content_by_version_node_id
        from shared.inspector.utils.synthesis.deep_context import (
            DeepContextDoc,
            DeepContextDocKind,
        )
        from workflows.inspector_functions import (
            DeepContextDocsInput,
            ExportDocsInput,
            deep_context_docs_task,
            export_tech_docs_task,
        )

        # TODO: Implement checkpoint-based statuses for formalized multi-stage compiler
        # architecture, then uncomment the following line to represent completion of
        # stage 1.
        set_codebase_status(version_id, VersionStatus.GENERATION_COMPLETE)
        print("Changes detected exporting tech docs to zip...")
        await export_tech_docs_task.aio_run(
            ExportDocsInput(
                version_id=version_id,
                install_id=install_id,
            )
        )

        print("Spawning off deep context docs generation...")
        # TODO: do deep context doc specific I/O or further analysis.

        if previous_version_root_version_node_id is not None:
            update_set = {
                ContentKind.DEEP_CONTEXT_ARCHITECTURE,
                ContentKind.DEEP_CONTEXT_LLM_ONBOARDING,
            }
            previous_version_root_content = (
                await get_all_derived_content_by_version_node_id(
                    version_node_id=previous_version_root_version_node_id
                )
            )
            previous_version_content = [
                DeepContextDoc(
                    doc_kind=DeepContextDocKind.from_content_kind(
                        content_kind=c.content_kind
                    ),
                    name=None,
                    user_context={"desired_length": "SHORT"},
                    sources=[],
                    config_content="",
                    doc_content=c.content,
                )
                for c in previous_version_root_content
                if c.content_kind in update_set
            ]
        else:
            previous_version_root_content = None
            previous_version_content = None

        if previous_version_id is not None:
            await put_diff_content_cache_async(
                str(previous_version_id),
                flat_topo_file_diff_dag,
            )
        deep_context_docs_input = DeepContextDocsInput(
            old_version_id=previous_version_id,
            old_version_content=previous_version_content,
            new_version_id=version_id,
            install_id=install_id,
        )
        await deep_context_docs_task.aio_run(
            deep_context_docs_input, options=TriggerWorkflowOptions(sticky=True)
        )


def hash_file(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def build_dag(root_path: Path, file_paths: list[Path]) -> FileTreeDag:
    print("Building DAG...")
    dag = FileTreeDag(root_abs_path=root_path)
    for p in file_paths:
        if p.is_file():
            dag.add_file(p, change_status=False, file_hash=hash_file(p))
    return dag


async def inspect_files(
    version_id: uuid.UUID,
    codebase_root: Path,
    nodes_with_id: list[tuple[Node, uuid.UUID | None]],
    codebase_name: str,
    run_id: UUID,
    result_loading_config: list[tuple[UUID, set[NodeStatus]]] | None,
    rel_path_to_previous_version_db_node_ids: dict[Path, uuid.UUID],
    version_node_id_to_node_id: dict[str, str],
) -> None:
    from shared.inspector.utils.db import get_all_derived_content_by_version_node_id

    print("---------- All nodes ----------")

    for node, _ in nodes_with_id:
        print(node)

    # c_files = [
    #     codebase_root / node.root_rel_path
    #     for node, _ in nodes_with_id
    #     if node.root_rel_path.suffix.lower() in [".c"]
    # ]
    # h_files = [
    #     codebase_root / node.root_rel_path
    #     for node, _ in nodes_with_id
    #     if node.root_rel_path.suffix.lower() in [".h"]
    # ]
    # c_and_h_files = c_files + h_files

    # if any(c_files):
    #     index = build_c_project_index(c_and_h_files, codebase_root / codebase_name)
    #     print("C symbol index built")
    #     # Build index here put as single dict key. This is obviously not prod ready. We would ideally name the dict
    #     # by unique id (or ephemeral) and pass in a dict handle  the downstream functions that need shared data
    #     d = modal.Dict.from_name("temp", create_if_missing=True)
    #     d["symbol_table"] = index

    tasks = []
    c_symbol_table_task = CSymbolTableTask(
        root_node=nodes_with_id[-1][0],
        task_name="CSymbolTableTask",
        codebase_name=codebase_name,
        codebase_root=codebase_root,
        nodes_relative_paths=[
            node.root_rel_path
            for node, _ in nodes_with_id
            if node.kind == NodeKind.FILE
        ],
        version_id=str(version_id),
    )
    tasks.append(c_symbol_table_task)
    node_id_to_file_task = {}
    node_id_to_folder_task = {}
    for node, db_version_node_id in nodes_with_id:
        lite_node = node.into_lite_node()

        if node.kind in {NodeKind.SUB_FOLDER, NodeKind.ROOT_FOLDER}:
            child_doc_tasks = tuple(
                {
                    t
                    for t in tasks
                    if isinstance(t, FileTechDocTask | FolderTechDocTask)
                    and t.node.root_rel_path.as_posix() in node.children
                }
            )
            if node.root_rel_path in rel_path_to_previous_version_db_node_ids:
                prev_db_node_id = rel_path_to_previous_version_db_node_ids[
                    node.root_rel_path
                ]
                prev_folder_derived_contents = (
                    await get_all_derived_content_by_version_node_id(prev_db_node_id)
                )
                previous_contents = {
                    dc.content_kind: dc.content for dc in prev_folder_derived_contents
                }
            else:
                previous_contents = None
            node_id = version_node_id_to_node_id[db_version_node_id]
            if node_id in node_id_to_folder_task:
                # NOTE: this is a duplicate node, so we reference the existing task to avoid recomputation
                folder_tech_docs_task = FolderTechDocTask(
                    node=lite_node,
                    task_name=f"FolderTechDoc {node.root_rel_path}",
                    child_docs_tasks=child_doc_tasks,
                    codebase_name=codebase_name,
                    version_id=str(version_id),
                    db_version_node_id=db_version_node_id,
                    previous_content=previous_contents,
                    deduped_node_task=node_id_to_folder_task[node_id],
                )
                tasks.append(folder_tech_docs_task)
                continue
            folder_tech_docs_task = FolderTechDocTask(
                node=lite_node,
                task_name=f"FolderTechDoc {node.root_rel_path}",
                child_docs_tasks=child_doc_tasks,
                codebase_name=codebase_name,
                version_id=str(version_id),
                db_version_node_id=db_version_node_id,
                previous_content=previous_contents,
            )
            node_id_to_folder_task[node_id] = folder_tech_docs_task
            folder_embedding_task = EmbeddingTask(
                node=node,
                task_name=f"Embedding TechDoc (Folder) {node.root_rel_path}",
                embedding_task_type=EmbeddingTaskType.FOLDER_TECH_DOC,
                dependent_tasks=[folder_tech_docs_task],
                db_version_node_id=db_version_node_id,
            )
            tasks.extend([folder_tech_docs_task, folder_embedding_task])
        else:  # File
            node_id = version_node_id_to_node_id[db_version_node_id]
            source_code = get_file_content(codebase_root / lite_node.root_rel_path)
            if node_id in node_id_to_file_task:
                # NOTE: this is a duplicate node, so we reference the existing task to avoid recomputation
                file_tech_docs_task = FileTechDocTask(
                    codebase_name=codebase_name,
                    source_code=source_code,
                    node=lite_node,
                    task_name=f"TechDoc {node.root_rel_path}",
                    db_version_node_id=db_version_node_id,
                    version_id=str(version_id),
                    symbol_table_task=c_symbol_table_task,
                    deduped_node_task=node_id_to_file_task[node_id],
                    thread_pool=TECH_DOC_THREAD_POOL,
                )
                tasks.append(file_tech_docs_task)
                continue

            source_file_embedding_task = EmbeddingTask(
                node=node,
                task_name=f"Embedding Source Code {node.root_rel_path}",
                embedding_task_type=EmbeddingTaskType.SOURCE_CODE,
                source_code=source_code,
                db_version_node_id=db_version_node_id,
                dependent_tasks=[c_symbol_table_task],
            )
            file_tech_docs_task = FileTechDocTask(
                codebase_name=codebase_name,
                source_code=source_code,
                node=lite_node,
                task_name=f"TechDoc {node.root_rel_path}",
                db_version_node_id=db_version_node_id,
                version_id=str(version_id),
                symbol_table_task=c_symbol_table_task,
                thread_pool=TECH_DOC_THREAD_POOL,
            )
            node_id_to_file_task[node_id] = file_tech_docs_task
            file_tech_docs_embedding_task = EmbeddingTask(
                node=node,
                task_name=f"Embedding TechDoc (File) {node.root_rel_path}",
                embedding_task_type=EmbeddingTaskType.FILE_TECH_DOC,
                source_code=None,
                db_version_node_id=db_version_node_id,
                dependent_tasks=[file_tech_docs_task],
            )
            tasks.extend(
                [
                    source_file_embedding_task,
                    file_tech_docs_task,
                    file_tech_docs_embedding_task,
                ]
            )

    root_node, root_db_node_id = nodes_with_id[-1]

    # TODO check propagation of changes to root node is working properly such that these tasks are appropriatly triggered
    # on rerun case.

    all_tech_docs_tasks = tuple(
        t for t in tasks if isinstance(t, FileTechDocTask | FolderTechDocTask)
    )
    top_level_tech_docs_task = TopLevelDocsTask(
        node=root_node,
        codebase_name=codebase_name,
        ordered_tech_docs_tasks=all_tech_docs_tasks,  # TODO where does source content go here?
        db_version_node_id=root_db_node_id,
    )

    has_previous_version = (
        root_node.root_rel_path in rel_path_to_previous_version_db_node_ids
    )
    if has_previous_version:
        prev_db_root_node_id = rel_path_to_previous_version_db_node_ids[
            root_node.root_rel_path
        ]
        prev_root_node_derived_contents = (
            await get_all_derived_content_by_version_node_id(prev_db_root_node_id)
        )
        previous_root_node_metadata = defaultdict(list)
        for dc in prev_root_node_derived_contents:
            previous_root_node_metadata[dc.content_kind].append(dc.misc_metadata)
    codebase_tagging_task = CodebaseTaggingTask(
        root_node=root_node,
        codebase_name=codebase_name,
        ordered_tech_docs_tasks=all_tech_docs_tasks,
        db_root_version_node_id=root_db_node_id,
        previous_root_node_metadata=previous_root_node_metadata
        if has_previous_version
        else None,
    )
    tasks.extend([top_level_tech_docs_task, codebase_tagging_task])

    print("\n---------- All tasks ----------")
    for t in tasks:
        print("=> ", t)

    print("\n---------- Running tasks ----------")
    task_manager = TaskManager.with_db_persistence(tasks=tasks, serial_exe=False)

    print(f"Starting inspection with {len(tasks)} tasks")
    await task_manager.run_tasks(run_id, result_loading_config=result_loading_config)

    print(
        f"Inspection completed! Final progress: {task_manager.progress_state.percent_complete:.1f}%"
    )


def get_file_content(path: Path) -> str:
    return Path(path).read_text()


def send_exception_email(exception_details: str) -> None:
    import sendgrid
    from sendgrid.helpers.mail import Content, Email, Mail, To

    env_name = os.environ.get("ENV_NAME")
    sendgrid_api_key = os.environ.get("SENDGRID_API_KEY")

    sg = sendgrid.SendGridAPIClient(api_key=sendgrid_api_key)
    from_email = Email("support@driverai.com")  # Replace with your email
    to_email = To("support@driverai.com")  # Replace with recipient's email
    subject = f"MODAL {env_name}: Exception Occurred"
    content = Content("text/plain", f"An exception occurred: {exception_details}")
    mail = Mail(from_email, to_email, subject, content)

    try:
        response = sg.send(mail)
        print(f"Email sent: {response.status_code}")
    except Exception as e:
        print(f"Error sending email: {e}")
