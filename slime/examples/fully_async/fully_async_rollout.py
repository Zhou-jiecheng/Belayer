import asyncio
import atexit
import logging
import queue
import threading
import time

# Import core functions from sglang_rollout directly to avoid code duplication
from slime.rollout.sglang_rollout import GenerateState, generate_and_rm_group
from slime.utils.async_utils import run
from slime.utils.types import Sample

# Global worker manager
_global_worker = None
_worker_lock = threading.Lock()
logger = logging.getLogger(__name__)


def get_global_worker(args, data_buffer):
    """Get or create global worker"""
    global _global_worker
    with _worker_lock:
        if _global_worker is None or not _global_worker.worker_thread.is_alive():
            print("Creating new global async worker...")
            _global_worker = AsyncRolloutWorker(args, data_buffer, concurrency=args.sglang_server_concurrency)
            _global_worker.start()
        return _global_worker


def stop_global_worker():
    """Stop global worker"""
    global _global_worker
    with _worker_lock:
        if _global_worker is not None:
            _global_worker.stop()
            _global_worker = None


class AsyncRolloutWorker:
    """
    Simplified asynchronous rollout worker, using threads instead of processes
    Supports continuous running, independent of rollout function lifecycle
    """

    def __init__(self, args, data_buffer, concurrency=10):
        self.args = args
        self.data_buffer = data_buffer  # Directly save data_buffer reference
        self.concurrency = concurrency
        self.running = True
        self.output_queue = queue.Queue(maxsize=1000)  # Continuous output queue
        self.worker_thread = None
        self.state = GenerateState(args)
        self._stats_lock = threading.Lock()
        self._inflight_groups: dict[int, dict] = {}
        self._launched_groups = 0
        self._completed_groups = 0
        self._failed_groups = 0
        self._callback_block_events = 0
        self._last_launch_ts = 0.0
        self._last_completion_ts = 0.0
        self._last_empty_fetch_log_ts = 0.0
        self._last_loop_snapshot_log_ts = 0.0

    def _mark_group_launched(self, group_id: int, group: list[Sample], active_task_count: int) -> None:
        now = time.time()
        first_index = group[0].index if group else None
        with self._stats_lock:
            self._launched_groups += 1
            self._last_launch_ts = now
            self._inflight_groups[group_id] = {
                "launched_at": now,
                "size": len(group),
                "first_index": first_index,
                "active_task_count_on_launch": active_task_count,
            }

    def _mark_group_finished(self, group_id: int, failed: bool = False) -> dict | None:
        now = time.time()
        with self._stats_lock:
            metadata = self._inflight_groups.pop(group_id, None)
            if failed:
                self._failed_groups += 1
            else:
                self._completed_groups += 1
                self._last_completion_ts = now
            return metadata

    def _record_callback_block(self) -> int:
        with self._stats_lock:
            self._callback_block_events += 1
            return self._callback_block_events

    def get_debug_snapshot(self) -> dict:
        now = time.time()
        with self._stats_lock:
            inflight = list(self._inflight_groups.items())
            oldest_group_id = None
            oldest_age_s = 0.0
            oldest_first_index = None
            if inflight:
                oldest_group_id, oldest_meta = min(inflight, key=lambda item: item[1]["launched_at"])
                oldest_age_s = now - oldest_meta["launched_at"]
                oldest_first_index = oldest_meta["first_index"]
            return {
                "inflight_groups": len(inflight),
                "oldest_inflight_group_id": oldest_group_id,
                "oldest_inflight_age_s": round(oldest_age_s, 1),
                "oldest_inflight_first_index": oldest_first_index,
                "launched_groups": self._launched_groups,
                "completed_groups": self._completed_groups,
                "failed_groups": self._failed_groups,
                "callback_block_events": self._callback_block_events,
                "output_queue_size": self.output_queue.qsize(),
                "last_launch_ago_s": round(now - self._last_launch_ts, 1) if self._last_launch_ts else None,
                "last_completion_ago_s": round(now - self._last_completion_ts, 1) if self._last_completion_ts else None,
            }

    async def continuous_worker_loop(self):
        """Continuous work loop - constantly get data from data_buffer and process"""
        logger.info("Continuous async rollout worker started")

        active_tasks = set()
        max_concurrent_tasks = self.args.rollout_batch_size
        group_id_counter = 0

        while self.running:
            try:
                # Clean up completed tasks
                if active_tasks:
                    done_tasks = {task for task in active_tasks if task.done()}
                    for task in done_tasks:
                        try:
                            task.result()  # Results are already handled in callbacks
                        except Exception as e:
                            print(f"Task failed with exception: {e}")
                    active_tasks -= done_tasks

                # If active task count hasn't reached limit, try to get new data and start tasks
                while len(active_tasks) < max_concurrent_tasks and self.running:
                    fetch_start = time.time()
                    samples = self.data_buffer.get_samples(1)
                    fetch_elapsed = time.time() - fetch_start

                    if not samples:
                        current_time = time.time()
                        if current_time - self._last_empty_fetch_log_ts >= 30:
                            logger.info(
                                "Async worker got no samples from data buffer for %.1fs "
                                "(active_tasks=%s, snapshot=%s)",
                                fetch_elapsed,
                                len(active_tasks),
                                self.get_debug_snapshot(),
                            )
                            self._last_empty_fetch_log_ts = current_time
                        break

                    for group in samples:
                        group_id = group_id_counter
                        group_id_counter += 1
                        self._mark_group_launched(group_id, group, len(active_tasks))

                        # Create new async task
                        task = asyncio.create_task(
                            generate_and_rm_group(
                                self.args,
                                group,
                                sampling_params=self.state.sampling_params.copy(),
                                evaluation=False,
                            )
                        )

                        # Add completion callback
                        def make_callback(gid):
                            def task_done_callback(done_task):
                                callback_start = time.time()
                                try:
                                    result = done_task.result()
                                except Exception:
                                    metadata = self._mark_group_finished(gid, failed=True)
                                    logger.exception(
                                        "Async worker task failed for group_id=%s metadata=%s snapshot=%s",
                                        gid,
                                        metadata,
                                        self.get_debug_snapshot(),
                                    )
                                    return

                                metadata = self._mark_group_finished(gid, failed=False)
                                total_elapsed = None
                                if metadata is not None:
                                    total_elapsed = time.time() - metadata["launched_at"]

                                if self.output_queue.full():
                                    block_count = self._record_callback_block()
                                    logger.warning(
                                        "Async worker output queue is full before enqueuing group_id=%s "
                                        "(block_event=%s, snapshot=%s, task_elapsed=%.2fs)",
                                        gid,
                                        block_count,
                                        self.get_debug_snapshot(),
                                        total_elapsed or -1.0,
                                    )

                                self.output_queue.put((gid, result))

                                statuses = {}
                                for sample in result:
                                    status_name = getattr(sample.status, "name", str(sample.status))
                                    statuses[status_name] = statuses.get(status_name, 0) + 1

                            return task_done_callback

                        task.add_done_callback(make_callback(group_id))
                        active_tasks.add(task)
                        break

                # Brief sleep to avoid busy waiting
                current_time = time.time()
                if current_time - self._last_loop_snapshot_log_ts >= 60:
                    logger.info(
                        "Async worker heartbeat: active_tasks=%s/%s snapshot=%s",
                        len(active_tasks),
                        max_concurrent_tasks,
                        self.get_debug_snapshot(),
                    )
                    self._last_loop_snapshot_log_ts = current_time
                await asyncio.sleep(1)

            except Exception as e:
                logger.exception(f"Error in continuous worker loop: {e}")
                await asyncio.sleep(1)

        if active_tasks:
            logger.info(f"Waiting for {len(active_tasks)} continuous tasks to complete...")
            await asyncio.wait(active_tasks)

        logger.info("Continuous async rollout worker stopped")

    def worker_thread_func(self):
        """Worker function running in independent thread"""
        asyncio.run(self.continuous_worker_loop())

    def start(self):
        """Start continuous work mode"""
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(target=self.worker_thread_func, daemon=True)
            self.worker_thread.start()
            logger.info("Started continuous async worker thread")

    def stop(self):
        """Stop worker thread"""
        self.running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)
        logger.info("Stopped async worker thread")

    def get_completed_groups(self) -> list[tuple]:
        """Get completed sample groups"""
        completed = []
        while True:
            try:
                result = self.output_queue.get_nowait()
                completed.append(result)
            except queue.Empty:
                break
        return completed

    def get_queue_size(self) -> int:
        """Get current output queue size"""
        return self.output_queue.qsize()


async def generate_rollout_async(args, rollout_id: int, data_buffer) -> list[list[Sample]]:
    """
    Simplified asynchronous rollout generation - using global continuous worker
    """
    assert args.rollout_global_dataset

    # Get global worker, which will run continuously
    worker = get_global_worker(args, data_buffer)

    # Simplified: directly use rollout_batch_size as target
    target_data_size = args.rollout_batch_size

    data = []
    completed_groups = {}
    do_print = True

    logger.info(f"Starting async rollout generation for {target_data_size} groups")
    logger.info("Global worker snapshot at rollout start: %s", worker.get_debug_snapshot())

    # Main loop: collect results from global worker's output queue
    start_time = time.time()
    last_progress_time = start_time
    no_progress_timeout = 30.0  # Warn if no progress for 30 seconds

    while len(data) < target_data_size:
        # Collect completed results
        completed = worker.get_completed_groups()

        made_progress = False
        for group_id, group in completed:
            completed_groups[group_id] = group
            made_progress = True

        if made_progress:
            last_progress_time = time.time()

        # Process completed groups in order (try to maintain order, but not strict requirement)
        processed_any = False

        # Process all available completed groups
        available_ids = list(completed_groups.keys())
        for group_id in available_ids:
            if len(data) >= target_data_size:
                break

            group = completed_groups.pop(group_id)

            # If any sample in the group was aborted, return the whole group to the data buffer
            # and do not forward it to the training engine.
            try:
                any_aborted = any([sample.status == Sample.Status.ABORTED for sample in group])
            except Exception:
                any_aborted = False

            if any_aborted:
                try:
                    # add back to buffer so it can be retried or handled by buffer policy
                    data_buffer.add_samples([group])
                    logger.info(f"Returned aborted group {group_id} to data buffer")
                except Exception as e:
                    logger.warning(f"Failed to return aborted group {group_id} to buffer: {e}")
                # don't count as processed for training
                continue

            if do_print:
                logger.info(
                    f"First rollout sample: {[group[0].prompt + group[0].response]}, "
                    f"label: {group[0].label}, reward: {group[0].reward}"
                )
                do_print = False

            # Simplified: directly add samples, no filters used
            data.append(group)
            processed_any = True

        # Check progress
        current_time = time.time()
        if current_time - last_progress_time > no_progress_timeout:
            logger.warning(
                "No progress for %.1fs. Collected=%s/%s completed_groups_buffered=%s worker_snapshot=%s",
                no_progress_timeout,
                len(data),
                target_data_size,
                len(completed_groups),
                worker.get_debug_snapshot(),
            )
            last_progress_time = current_time

        # If no results were processed, brief sleep to avoid busy waiting
        if not processed_any:
            await asyncio.sleep(0.01)

    duration = time.time() - start_time
    logger.info(
        "Rollout completed in %.2fs! Global worker snapshot at rollout end: %s",
        duration,
        worker.get_debug_snapshot(),
    )

    if data:
        logger.info(
            f"Finish rollout: {[data[-1][0].prompt + data[-1][0].response]}, "
            f"label: {data[-1][0].label}, reward: {data[-1][0].reward}"
        )

    data = sorted(data, key=lambda group: group[0].index)
    return data


def generate_rollout_fully_async(args, rollout_id, data_buffer, evaluation=False):
    if evaluation:
        raise ValueError("Evaluation mode not supported in simple async rollout")

    completed_samples = run(generate_rollout_async(args, rollout_id, data_buffer))
    return completed_samples


# Register exit cleanup function

atexit.register(stop_global_worker)
