import abc
import asyncio
import concurrent
import hashlib
import json
import pickle
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar
from uuid import UUID

from shared.inspector.utils.dag import LiteNode, NodeStatus

TaskName = str


@dataclass
class ProgressState:
    total_work_units: int = 0
    completed_work_units: int = 0
    task_count: int = 0
    completed_task_count: int = 0

    @property
    def percent_complete(self) -> float:
        """Calculate percentage completion based on work units"""
        if self.total_work_units == 0:
            return 0.0
        return (self.completed_work_units / self.total_work_units) * 100.0

    @property
    def task_percent_complete(self) -> float:
        """Calculate percentage completion based on task count"""
        if self.task_count == 0:
            return 0.0
        return (self.completed_task_count / self.task_count) * 100.0


class TaskWorkUnits:
    """Defines work unit constants for each task type"""

    SYMBOL_TABLE = 30
    FILE_TECH_DOC = 10
    FOLDER_TECH_DOC = 10
    TOP_LEVEL_DOCS = 20
    SYMBOLS = 10
    EMBEDDING = 1
    TAGS = 20


class SerializationMethod(str, Enum):
    JSON = "json"  # NOTE: we use JSON here in case of python version upgrade, we maintain backwards compatibility.
    PICKLE = "pickle"  # NOTE: use this sparingly, due to concern about backwards compatibilty with python upgrade


@dataclass(frozen=True)
class TaskResult:
    data: Any
    serialization: SerializationMethod

    def serialize(self) -> str | bytes:
        if self.serialization is SerializationMethod.JSON:
            return json.dumps(self.data)  # str
        if self.serialization is SerializationMethod.PICKLE:
            return pickle.dumps(self.data)  # bytes
        raise ValueError(f"Unsupported serialization method: {self.serialization}")

    @classmethod
    def deserialize(
        cls, data: str | bytes, method: SerializationMethod
    ) -> "TaskResult":
        if method is SerializationMethod.JSON and isinstance(data, str):
            value = json.loads(data)
        elif method is SerializationMethod.PICKLE and isinstance(data, bytes):
            value = pickle.loads(data)
        else:
            raise ValueError("Invalid data type for the given serialization method")
        return cls(data=value, serialization=method)


# TODO check exception handling and propagation is correct!
# If a task fails, the dependent tasks should not run, but we can still run other tasks if desired.


class TaskResultPersistence(ABC):
    _extension_for_method: ClassVar[dict[SerializationMethod, str]] = {
        SerializationMethod.JSON: ".json",
        SerializationMethod.PICKLE: ".pkl",
    }

    _method_for_extension: ClassVar[dict[str, SerializationMethod]] = {
        v: k for k, v in _extension_for_method.items()
    }

    @abstractmethod
    def save_task_result(
        self, run_id: str, result: TaskResult, task: type["Task"]
    ) -> None:
        pass

    @abstractmethod
    def load_task_result(self, run_id: str, task: type["Task"]) -> None | TaskResult:
        pass


class LocalDiskTaskResultPersistence(TaskResultPersistence):
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_file_base_path(self, run_id: str, task_id: str) -> Path:
        return self.base_dir / run_id / f"{task_id}"

    def _get_run_dir(self, run_id: str) -> Path:
        return self.base_dir / run_id

    def save_task_result(
        self, run_id: str, result: TaskResult, task: type["Task"]
    ) -> None:
        run_dir = self._get_run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        file_path = self._get_file_base_path(run_id, task.hashed_stable_id)
        ext = self._extension_for_method[result.serialization]
        full_path = file_path.with_suffix(ext)

        data = result.serialize()
        if result.serialization is SerializationMethod.JSON:
            full_path.write_text(data)
        elif result.serialization is SerializationMethod.PICKLE:
            full_path.write_bytes(data)
        else:
            raise ValueError(
                f"Serialization must be either JSON or PICKLE, found {result.serialization}"
            )

    def load_task_result(self, run_id: str, task: type["Task"]) -> None | TaskResult:
        base_path = self._get_file_base_path(run_id, task.hashed_stable_id)

        for serialization_method, ext in self._extension_for_method.items():
            file_path = base_path.with_suffix(ext)
            if file_path.exists():
                data = (
                    file_path.read_text() if ext == ".json" else file_path.read_bytes()
                )
                return TaskResult.deserialize(data, serialization_method)

        return None


class DbTaskResultPersistence(TaskResultPersistence):
    def save_task_result(
        self, run_id: str, result: TaskResult, task: type["Task"]
    ) -> None:
        # Saving all tasks via database and existing flows
        pass

    def load_task_result(self, run_id: str, task: type["Task"]) -> None | TaskResult:
        return task.load_result()


@dataclass
class Task(abc.ABC):
    task_name: str
    node: LiteNode
    dependencies: tuple[type["Task"], ...] = field(default_factory=tuple)

    async def run(
        self,
        dependent_results: dict["Task", TaskResult],
    ) -> TaskResult:
        return await self.run_implementation(dependent_results)

    @abstractmethod
    async def run_implementation(
        self, dependent_results: dict["Task", TaskResult]
    ) -> TaskResult:
        raise NotImplementedError

    @abstractmethod
    async def post_run_io(
        self,
        task_result: TaskResult,
    ) -> dict[str, any]:
        raise NotImplementedError

    @abstractmethod
    def load_result(self) -> None | TaskResult:
        """Load previously saved result for this task from storage/database."""
        raise NotImplementedError

    @property
    @abstractmethod
    def work_units(self) -> int:
        raise NotImplementedError

    # TODO this could get really long, but does it matter?
    @property
    def stable_id(self) -> str:
        id_str = f"{self.__class__.__name__}_{self.node.stable_id}"
        if self.dependencies:
            dep_str = "_".join([dep.stable_id for dep in self.dependencies])
            id_str += f"_{dep_str}"
        return id_str

    @property
    def hashed_stable_id(self) -> str:
        return hashlib.sha256(self.stable_id.encode()).hexdigest()

    def __hash__(self) -> int:
        return hash(self.stable_id)

    def __eq__(self, other: "Task") -> bool:
        if isinstance(other, Task):
            return hash(self) == hash(other)
        return False

    def __str__(self) -> str:
        return f"{self.__class__.__name__} for node: {self.node.root_rel_path}, node_status: {self.node.status}"


@dataclass
class TaskManager:
    tasks: list[type[Task]] = field(default_factory=list)
    serial_exe: bool = False
    task_results: dict[type[Task], TaskResult] = field(
        default_factory=dict
    )  # TODO remove TaskResult until used...
    task_io_results: dict[type[Task], dict[str, any]] = field(default_factory=dict)
    task_to_asynctask: dict[type[Task], asyncio.Task] = field(default_factory=dict)
    persistence: None | TaskResultPersistence = field(
        default_factory=lambda: DbTaskResultPersistence()  # TODO make configurable!!
    )
    write_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    write_executor: ThreadPoolExecutor = field(
        default_factory=lambda: ThreadPoolExecutor(max_workers=5)
    )
    progress_state: ProgressState = field(default_factory=ProgressState)

    @classmethod
    def with_db_persistence(
        cls,
        *args: Any,
        **kwargs: Any,
    ) -> "TaskManager":
        return cls(*args, persistence=DbTaskResultPersistence(), **kwargs)

    def _initialize_progress(self) -> None:
        self.progress_state.total_work_units = sum(
            task.work_units for task in self.tasks
        )
        self.progress_state.task_count = len(self.tasks)
        self.progress_state.completed_work_units = 0
        self.progress_state.completed_task_count = 0
        print(
            f"Initialized progress tracking: {self.progress_state.total_work_units} total work units, {self.progress_state.task_count} tasks"
        )

    def _update_progress(self, completed_task: "Task", threshold: float = 0.1) -> None:
        self.progress_state.completed_work_units += completed_task.work_units
        self.progress_state.completed_task_count += 1

        # Only print progress if we've crossed a percentage threshold
        if (
            self.progress_state.percent_complete % threshold
            < (
                self.progress_state.percent_complete
                - completed_task.work_units
                * 100.0
                / self.progress_state.total_work_units
            )
            % threshold
        ):
            color = "\033[92m"
            reset = "\033[0m"
            print(
                f"{color}[PROGRESS]: {self.progress_state.percent_complete:.1f}% ({self.progress_state.completed_work_units}/{self.progress_state.total_work_units} work units, {self.progress_state.completed_task_count}/{self.progress_state.task_count} tasks){reset}"
            )

    async def run_tasks(
        self,
        run_id: UUID,
        result_loading_config: list[tuple[UUID, set[NodeStatus]]] | None = None,
    ) -> dict[type[Task], TaskResult]:
        result_loading_config = result_loading_config or []

        self._initialize_progress()

        if self.persistence:  # ALWAYS try to load as long as persistence is set
            # We can block the event loop with blocking IO when loading the state we aren't running
            # anything concurrent yet
            self.load_persisted_results(result_loading_config)
        # return

        if self.persistence:
            writer_task = asyncio.create_task(self._write_task_results(str(run_id)))

        try:
            if self.serial_exe:
                for task in self.tasks:
                    await self._schedule_and_await_task(task)
            else:
                await asyncio.gather(
                    *[self._schedule_and_await_task(task) for task in self.tasks]
                )
        finally:
            # Stop the writer task
            await self.write_queue.put(None)
            if self.persistence:
                await writer_task

        return self.task_results

    def load_persisted_results(
        self, result_loading_config: list[tuple[str, set[NodeStatus]]]
    ) -> None:
        flattened_tasks = self.tasks

        task_by_id: dict[str, Task] = {t.hashed_stable_id: t for t in flattened_tasks}
        print(
            "Loading persisted results for resumption (threadpool, preserving order)..."
        )

        loaded_results: dict[Task, TaskResult] = {}

        for run_id, _ in result_loading_config:
            with ThreadPoolExecutor(max_workers=25) as pool:
                future_map = {
                    pool.submit(
                        self.persistence.load_task_result, run_id, t
                    ): t.hashed_stable_id
                    for t in flattened_tasks
                }

                for future in concurrent.futures.as_completed(future_map):
                    task_id = future_map[future]
                    try:
                        result = future.result()
                        if result and task_id in task_by_id:
                            loaded_results[task_by_id[task_id]] = result
                    except Exception as e:
                        for task in flattened_tasks:
                            if task.hashed_stable_id == task_id:
                                print(
                                    f"Failed to load result for task '{task.task_name}' from storage: {e}"
                                )
                        print(
                            f"Failed to load S3 result for {task_id} from run {run_id}: {e}"
                        )
        self.task_results.update(loaded_results)
        print(
            f"Loaded results successfully for {len(loaded_results)} tasks from storage"
        )

    # This is non-parallelized code we previously used for reference. Consider deleting...

    # def load_persisted_results(
    #     self, result_loading_config: list[tuple[str, set[NodeStatus]]]
    # ) -> None:
    #     print("Loading persisted results for resumption...")
    #     flattened_tasks = self.tasks
    #     print("Total tasks:", len(flattened_tasks))
    #
    #     loaded_results = {}
    #     for run_id, node_statuses in result_loading_config:
    #         persisted_results = self.persistence.load_all_results(run_id)
    #         for task in flattened_tasks:
    #             if (
    #                 task.node.status in node_statuses
    #                 and task.hashed_stable_id in persisted_results
    #             ):
    #                 print(f"Using results for task '{task.task_name}' from storage")
    #                 # Note that potential overwriting here is intentional.
    #                 loaded_results[task] = persisted_results[task.hashed_stable_id]
    #
    #     self.task_results.update(loaded_results)
    #     print(
    #         f"Loaded results successfully for {len(loaded_results)} tasks from storage"
    #     )

    async def _schedule_and_await_task(self, task: type[Task]) -> asyncio.Task:
        asynctask = self._schedule_task(task)
        return await asynctask

    def _schedule_task(self, task: type[Task]) -> asyncio.Task:
        if task not in self.task_to_asynctask:
            task_coroutine = self._run_task(task)
            self.task_to_asynctask[task] = asyncio.create_task(task_coroutine)
        return self.task_to_asynctask[task]

    def _can_skip_task(self, task: type[Task]) -> bool:
        task_result = self.task_results.get(task, None)
        return task_result is not None

    async def _run_task(self, task: type[Task]) -> TaskResult:
        """
        Run the task, ensuring that all dependencies run first.
        """
        if self.serial_exe:
            for dep_task in task.dependencies:
                await self._schedule_and_await_task(dep_task)
        else:
            dependent_tasks = [
                self._schedule_and_await_task(dep_task)
                for dep_task in task.dependencies
            ]
            await asyncio.gather(*dependent_tasks)

        if self._can_skip_task(task):
            print(f"Skipping task '{task.task_name}'...")
            result = self.task_results[task]
        else:
            print(f"Running task '{task.task_name}'...")
            result = await task.run(
                dependent_results={
                    dep_task: self.task_results[dep_task]
                    for dep_task in task.dependencies
                }
            )
            self.task_results[task] = result

            print(
                f"Running post-run IO for task '{task.task_name}' since task was not skipped..."
            )
            io_result = await task.post_run_io(
                task_result=result,
            )
            self.task_io_results[task] = io_result

        task_hash_str = task.hashed_stable_id
        await self.write_queue.put((task_hash_str, result, task))

        self._update_progress(task)

        return result

    async def _write_task_results(self, run_id: str) -> None:
        while True:
            item = await self.write_queue.get()
            if item is None:
                break
            task_id, result, task = item
            await asyncio.get_running_loop().run_in_executor(
                self.write_executor,
                self.persistence.save_task_result,
                run_id,
                result,
                task,
            )
        print("Writer task done")


def flatten_tasks(tasks: list[type[Task]]) -> list[type[Task]]:
    def _flatten(task: type[Task], seen: set[type[Task]]) -> list[type[Task]]:
        if task in seen:
            return []
        seen.add(task)
        return [task] + [t for dep in task.dependencies for t in _flatten(dep, seen)]

    seen = set()
    return [t for task in tasks for t in _flatten(task, seen)]
