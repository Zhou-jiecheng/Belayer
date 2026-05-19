import logging
import os
import threading
import time

import ray


logger = logging.getLogger(__name__)


class RolloutHealthMonitor:
    """Health monitor for rollout engines.

    The monitor runs continuously once started, but can be paused/resumed
    based on whether the engines are offloaded (cannot health check when offloaded).

    Lifecycle:
    - start(): Start the monitor thread (called once during initialization)
    - pause(): Pause health checking (called when offloading engines)
    - resume(): Resume health checking (called when onloading engines)
    - stop(): Stop the monitor thread completely (called during dispose)
    """

    def __init__(self, rollout_manager, args):
        # TODO may remove this dependency after refactoring
        self._rollout_manager = rollout_manager

        self._thread = None
        self._stop_event = None
        self._pause_event = None  # When set, health checking is paused
        self._check_interval = args.rollout_health_check_interval
        self._check_timeout = args.rollout_health_check_timeout
        self._check_first_wait = args.rollout_health_check_first_wait
        self._need_first_wait = True  # Need to wait after each resume
        self._is_checking_enabled = False  # Track if health checking should be active
        self._health_check_enabled = os.getenv("SLIME_ROLLOUT_ENABLE_HEALTH_CHECK", "1").lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        self._failure_threshold = max(1, int(os.getenv("SLIME_ROLLOUT_HEALTH_CHECK_FAILURE_THRESHOLD", "5")))
        self._consecutive_failures: dict[int, int] = {}
        self._suppressed_rollout_engines: set[int] = set()
        self._suppressed_rollout_engine_reasons: dict[int, str] = {}
        self._suppression_lock = threading.Lock()

    def start(self) -> bool:
        """Start the health monitor thread. Called once during initialization.

        Returns:
            True if the monitor was started, False if there are no engines to monitor.
        """
        if not self._health_check_enabled:
            logger.warning("Rollout health checks are disabled by SLIME_ROLLOUT_ENABLE_HEALTH_CHECK=0")
            return False

        if not self._rollout_manager.all_rollout_engines:
            return False

        if self._thread is not None:
            logger.warning("Health monitor thread is already running.")
            return True

        logger.info("Starting RolloutHealthMonitor...")
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # Start in paused state until resume() is called
        self._thread = threading.Thread(
            target=self._health_monitor_loop,
            name="RolloutHealthMonitor",
            daemon=True,
        )
        self._thread.start()
        logger.info("RolloutHealthMonitor started (in paused state).")
        return True

    def stop(self) -> None:
        """Stop the health monitor thread completely. Called during dispose."""
        if not self._thread:
            return

        logger.info("Stopping RolloutHealthMonitor...")
        assert self._stop_event is not None
        self._stop_event.set()
        # Also clear pause to let the thread exit
        if self._pause_event:
            self._pause_event.clear()
        timeout = self._check_timeout + self._check_interval + 5
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            logging.warning("Rollout health monitor thread did not terminate within %.1fs", timeout)
        else:
            logger.info("RolloutHealthMonitor stopped.")

        self._thread = None
        self._stop_event = None
        self._pause_event = None
        self._is_checking_enabled = False

    def pause(self) -> None:
        """Pause health checking. Called when engines are offloaded."""
        if self._pause_event is None:
            return
        logger.info("Pausing health monitor...")
        self._pause_event.set()
        self._is_checking_enabled = False

    def resume(self) -> None:
        """Resume health checking. Called when engines are onloaded."""
        if self._pause_event is None:
            return
        if not self._pause_event.is_set() and self._is_checking_enabled:
            logger.debug("Health monitor resume skipped because checks are already enabled.")
            return
        logger.info("Resuming health monitor...")
        self._need_first_wait = True  # Need to wait after each resume
        self._pause_event.clear()
        self._is_checking_enabled = True

    def is_checking_enabled(self) -> bool:
        """Return whether health checking is currently enabled (not paused)."""
        return self._is_checking_enabled

    def suppress_rollout_engine(self, rollout_engine_id: int, reason: str) -> None:
        with self._suppression_lock:
            already_suppressed = rollout_engine_id in self._suppressed_rollout_engines
            self._suppressed_rollout_engines.add(rollout_engine_id)
            self._suppressed_rollout_engine_reasons[rollout_engine_id] = str(reason)
        self._consecutive_failures[rollout_engine_id] = 0
        if already_suppressed:
            logger.info(
                "Health monitor suppression refreshed for rollout engine %s due to %s",
                rollout_engine_id,
                reason,
            )
        else:
            logger.info(
                "Health monitor suppression enabled for rollout engine %s due to %s",
                rollout_engine_id,
                reason,
            )

    def unsuppress_rollout_engine(self, rollout_engine_id: int, reason: str) -> None:
        with self._suppression_lock:
            was_suppressed = rollout_engine_id in self._suppressed_rollout_engines
            previous_reason = self._suppressed_rollout_engine_reasons.pop(rollout_engine_id, None)
            self._suppressed_rollout_engines.discard(rollout_engine_id)
        self._consecutive_failures[rollout_engine_id] = 0
        if was_suppressed:
            logger.info(
                "Health monitor suppression disabled for rollout engine %s due to %s (previous_reason=%s)",
                rollout_engine_id,
                reason,
                previous_reason,
            )

    def is_rollout_engine_suppressed(self, rollout_engine_id: int) -> bool:
        with self._suppression_lock:
            return rollout_engine_id in self._suppressed_rollout_engines

    def _health_monitor_loop(self) -> None:
        assert self._stop_event is not None
        assert self._pause_event is not None

        while not self._stop_event.is_set():
            # Wait while paused
            while self._pause_event.is_set() and not self._stop_event.is_set():
                self._stop_event.wait(timeout=0.5)

            if self._stop_event.is_set():
                break

            # Do first wait after each resume (for large MoE models to be ready)
            if self._need_first_wait:
                logger.info(f"Health monitor doing first wait after resume: {self._check_first_wait}s")
                if self._stop_event.wait(self._check_first_wait):
                    logger.info("Health monitor stopped during first wait.")
                    break
                if self._pause_event.is_set():
                    # Got paused during first wait, skip this round and wait again next resume
                    logger.info("Health monitor paused during first wait, will wait again next resume.")
                    continue
                self._need_first_wait = False

            # Run health checks
            if not self._pause_event.is_set() and not self._stop_event.is_set():
                self._run_health_checks()

            # Wait for next check interval
            if self._stop_event.wait(self._check_interval):
                break

    def _run_health_checks(self) -> None:
        logger.debug(
            "Running rollout health checks for %s engine slots (enabled=%s)",
            len(self._rollout_manager.rollout_engines),
            self._is_checking_enabled,
        )
        rollout_engine_ids_to_recover: list[int] = []
        for rollout_engine_id, engine in enumerate(self._rollout_manager.rollout_engines):
            if self._stop_event is not None and self._stop_event.is_set():
                break
            if self._pause_event is not None and self._pause_event.is_set():
                break
            if self._check_engine_health(rollout_engine_id, engine):
                rollout_engine_ids_to_recover.append(rollout_engine_id)

        if rollout_engine_ids_to_recover:
            self._recover_engine_groups(rollout_engine_ids_to_recover)

    def _check_engine_health(self, rollout_engine_id, engine) -> bool:
        if self.is_rollout_engine_suppressed(rollout_engine_id):
            logger.debug("Skipping health check for rollout engine %s because it is suppressed", rollout_engine_id)
            self._consecutive_failures[rollout_engine_id] = 0
            return False

        if engine is None:
            failures = self._consecutive_failures.get(rollout_engine_id, 0) + 1
            self._consecutive_failures[rollout_engine_id] = failures
            logger.warning(
                "Health check found rollout engine %s missing (engine=None). consecutive_failures=%s threshold=%s",
                rollout_engine_id,
                failures,
                self._failure_threshold,
            )
            if failures < self._failure_threshold:
                return False
            return True

        start_ts = time.monotonic()
        try:
            healthy = ray.get(engine.check_health.remote(timeout=self._check_timeout))
            if not healthy:
                raise RuntimeError("engine.check_health returned unhealthy status")
        except Exception as e:
            elapsed = time.monotonic() - start_ts
            failures = self._consecutive_failures.get(rollout_engine_id, 0) + 1
            self._consecutive_failures[rollout_engine_id] = failures
            logger.error(
                "Health check failed for rollout engine %s (ray timeout or error). "
                "consecutive_failures=%s threshold=%s elapsed=%.2fs timeout=%ss paused=%s checking_enabled=%s exception=%s",
                rollout_engine_id,
                failures,
                self._failure_threshold,
                elapsed,
                self._check_timeout,
                self._pause_event.is_set() if self._pause_event is not None else None,
                self._is_checking_enabled,
                e,
            )
            if failures < self._failure_threshold:
                return False
            if self._try_shadow_handover(rollout_engine_id):
                self._consecutive_failures[rollout_engine_id] = 0
                logger.warning(
                    "Shadow-worker handover succeeded for rollout engine %s; rollout manager will collect reconnect state from engines",
                    rollout_engine_id,
                )
                return False
            return True
        else:
            self._consecutive_failures[rollout_engine_id] = 0
            elapsed = time.monotonic() - start_ts
            if elapsed >= max(5.0, float(self._check_timeout) * 0.5):
                logger.info(
                    "Health check passed slowly for rollout engine %s (elapsed=%.2fs timeout=%ss)",
                    rollout_engine_id,
                    elapsed,
                    self._check_timeout,
                )
            else:
                logger.debug(
                    "Health check passed for rollout engine %s (elapsed=%.2fs)",
                    rollout_engine_id,
                    elapsed,
                )
            return False

    def _try_shadow_handover(self, rollout_engine_id: int) -> bool:
        start = rollout_engine_id * self._rollout_manager.nodes_per_engine
        end = (rollout_engine_id + 1) * self._rollout_manager.nodes_per_engine
        engines = self._rollout_manager.all_rollout_engines[start:end]
        if not engines or any(engine is None for engine in engines):
            logger.warning(
                "Shadow-worker handover skipped for rollout engine %s because engine group is incomplete (start=%s, end=%s)",
                rollout_engine_id,
                start,
                end,
            )
            return False

        logger.info(
            "Attempting shadow-worker handover for rollout engine %s covering engine indices [%s, %s)",
            rollout_engine_id,
            start,
            end,
        )
        try:
            results = ray.get([engine.promote_shadow_worker.remote() for engine in engines])
        except Exception as e:
            logger.warning("Shadow-worker handover failed for rollout engine %s: %s", rollout_engine_id, e)
            return False

        logger.info(
            "Shadow-worker handover results for rollout engine %s: %s",
            rollout_engine_id,
            results,
        )
        return all(results)

    def _recover_engine_groups(self, rollout_engine_ids: list[int]):
        unique_rollout_engine_ids = sorted(set(rollout_engine_ids))
        if not unique_rollout_engine_ids:
            return

        logger.info("Killing engine groups %s...", unique_rollout_engine_ids)
        for rollout_engine_id in unique_rollout_engine_ids:
            self.suppress_rollout_engine(
                rollout_engine_id,
                reason="batched kill-and-recover sequence started",
            )

        kill_start_ts = time.monotonic()
        for rollout_engine_id in unique_rollout_engine_ids:
            for i in range(
                rollout_engine_id * self._rollout_manager.nodes_per_engine,
                (rollout_engine_id + 1) * self._rollout_manager.nodes_per_engine,
            ):
                engine = self._rollout_manager.all_rollout_engines[i]
                if engine:
                    logger.info("Shutting down and killing engine at index %s", i)
                    one_engine_start_ts = time.monotonic()
                    try:
                        ray.get(engine.shutdown.remote())
                        ray.kill(engine)
                        logger.info(
                            "Successfully killed engine at index %s (elapsed=%.1fs)",
                            i,
                            time.monotonic() - one_engine_start_ts,
                        )
                    except Exception as e:
                        logger.warning("Fail to kill engine at index %s (e: %s)", i, e)
                else:
                    logger.info("Engine at index %s is already None", i)
                self._rollout_manager.all_rollout_engines[i] = None

        logger.info(
            "Finished kill loop for rollout engine groups %s (elapsed=%.1fs), entering immediate recovery",
            unique_rollout_engine_ids,
            time.monotonic() - kill_start_ts,
        )

        try:
            recover_start_ts = time.monotonic()
            _, _, recovered = self._rollout_manager.recover_rollout_engines()
            logger.info(
                "Immediate non-fast-restart recovery completed for rollout engine groups %s (recovered=%s, elapsed=%.1fs, total_elapsed=%.1fs)",
                unique_rollout_engine_ids,
                recovered,
                time.monotonic() - recover_start_ts,
                time.monotonic() - kill_start_ts,
            )
        except Exception:
            logger.exception(
                "Immediate non-fast-restart recovery failed for rollout engine groups %s; "
                "will rely on external recovery path",
                unique_rollout_engine_ids,
            )
            unsuppress_reason = "batched kill-and-recover sequence failed"
        else:
            unsuppress_reason = "batched kill-and-recover sequence completed"

        for rollout_engine_id in unique_rollout_engine_ids:
            self.unsuppress_rollout_engine(
                rollout_engine_id,
                reason=unsuppress_reason,
            )
