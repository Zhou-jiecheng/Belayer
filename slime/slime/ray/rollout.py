import itertools
import logging
import multiprocessing
import os
import random
import threading
import time
from copy import copy
from pathlib import Path
from typing import Any

import numpy as np
import ray
import torch
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH, GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

from slime.backends.sglang_utils.sglang_engine import SGLangEngine
from slime.rollout.base_types import call_rollout_fn
from slime.utils import logging_utils
from slime.utils.health_monitor import RolloutHealthMonitor
from slime.utils.http_utils import _wrap_ipv6, find_available_port, get_host_info, init_http_client
from slime.utils.logging_utils import configure_logger, init_tracking
from slime.utils.metric_utils import (
    MetricChecker,
    compute_pass_rate,
    compute_rollout_step,
    compute_statistics,
    dict_add_prefix,
)
from slime.utils.misc import Box, group_by, load_function
from slime.utils.seqlen_balancing import get_seqlen_balanced_partitions
from slime.utils.types import Sample

from ..utils.metric_utils import has_repetition
from .utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST, Lock

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

_CURRENT_ROLLOUT_MANAGER = None


def _get_ci_fault_injection_mode() -> str:
    return os.getenv("SLIME_CI_FAULT_INJECTION_MODE", "pre_generate").strip().lower()


def _get_ci_fault_injection_rollout_id_threshold() -> int:
    return int(os.getenv("SLIME_CI_FAULT_INJECTION_ROLLOUT_ID_THRESHOLD", "2"))


def _get_ci_fault_injection_engine_indices() -> list[int]:
    raw_value = os.getenv("SLIME_CI_FAULT_INJECTION_ENGINE_INDEX", "0")
    indices: list[int] = []
    seen: set[int] = set()
    for token in raw_value.split(","):
        token = token.strip()
        if not token:
            continue
        index = int(token)
        if index in seen:
            continue
        seen.add(index)
        indices.append(index)
    if not indices:
        raise ValueError("SLIME_CI_FAULT_INJECTION_ENGINE_INDEX must contain at least one engine index")
    return indices


def _try_acquire_ci_injection_global_once() -> bool:
    if os.getenv("SLIME_CI_FAULT_INJECTION_GLOBAL_ONCE", "1").lower() in {"0", "false", "no", "off"}:
        return True
    lock_path = os.getenv("SLIME_CI_FAULT_INJECTION_LOCK_PATH", "/tmp/slime_ci_fault_injection_once.lock")
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        return True
    except FileExistsError:
        return False
    except Exception:
        # Fail open to avoid blocking CI test if filesystem is unavailable.
        return True


def maybe_trigger_ci_fault_injection_mid_generate(rollout_id: int, completed_groups: int, target_groups: int) -> bool:
    manager = _CURRENT_ROLLOUT_MANAGER
    if manager is None:
        return False
    if not getattr(manager, "_ci_mid_generate_check_logged", False):
        logger.info(
            "CI mid-generate check active: mode=%s rollout_id=%s threshold=%s progress_fraction=%s completed=%s/%s",
            _get_ci_fault_injection_mode(),
            rollout_id,
            _get_ci_fault_injection_rollout_id_threshold(),
            os.getenv("SLIME_CI_FAULT_INJECTION_PROGRESS_FRACTION", "0.5"),
            completed_groups,
            target_groups,
        )
        manager._ci_mid_generate_check_logged = True
    if not (manager.args.ci_test and manager.args.use_fault_tolerance):
        if not getattr(manager, "_ci_mid_generate_skip_logged", False):
            logger.info(
                "CI mid-generate skipped: ci_test=%s use_fault_tolerance=%s",
                manager.args.ci_test,
                manager.args.use_fault_tolerance,
            )
            manager._ci_mid_generate_skip_logged = True
        return False
    mode = _get_ci_fault_injection_mode()
    if mode != "mid_generate":
        if not getattr(manager, "_ci_mid_generate_skip_logged", False):
            logger.info("CI mid-generate skipped: mode=%s (expected mid_generate)", mode)
            manager._ci_mid_generate_skip_logged = True
        return False
    threshold = _get_ci_fault_injection_rollout_id_threshold()
    if rollout_id < threshold:
        if not getattr(manager, "_ci_mid_generate_skip_logged", False):
            logger.info(
                "CI mid-generate skipped: rollout_id=%s < threshold=%s",
                rollout_id,
                threshold,
            )
            manager._ci_mid_generate_skip_logged = True
        return False
    if target_groups <= 0:
        return False

    progress_fraction = float(os.getenv("SLIME_CI_FAULT_INJECTION_PROGRESS_FRACTION", "0.5"))
    midpoint = max(1, int(np.ceil(target_groups * progress_fraction)))
    if completed_groups < midpoint:
        return False

    return manager._try_ci_fault_injection(
        wait_for_detection=False,
        trigger=f"mid_generate completed_groups={completed_groups}/{target_groups}",
    )


@ray.remote
class RolloutManager:
    """The class to run rollout and convert rollout data to training data."""

    def __init__(self, args, pg, prm_pg=None):
        configure_logger()
        init_start = time.perf_counter()
        logger.info("[RolloutManager.__init__] start")
        global _CURRENT_ROLLOUT_MANAGER
        _CURRENT_ROLLOUT_MANAGER = self

        self.args = args
        self.pg = pg
        self.prm_pg = prm_pg
        self._num_new_engines_lock = threading.Lock()
        self._pending_shadow_handover_reconnects = 0
        self._pending_shadow_handover_reconnect_reasons: list[str] = []
        step_start = time.perf_counter()
        _start_router(args, router_ip_attr="sglang_router_ip", router_port_attr="sglang_router_port")
        logger.info("[RolloutManager.__init__] router started in %.2fs", time.perf_counter() - step_start)
        if self.args.prm_enable and self.args.prm_num_gpus > 0:
            step_start = time.perf_counter()
            _start_router(args, router_ip_attr="prm_router_ip", router_port_attr="prm_router_port")
            logger.info("[RolloutManager.__init__] prm router started in %.2fs", time.perf_counter() - step_start)
        # TODO make args immutable
        step_start = time.perf_counter()
        init_tracking(args, primary=False, router_addr=f"http://{args.sglang_router_ip}:{args.sglang_router_port}")
        init_http_client(args)
        logger.info("[RolloutManager.__init__] tracking/http client initialized in %.2fs", time.perf_counter() - step_start)

        step_start = time.perf_counter()
        data_source_cls = load_function(self.args.data_source_path)
        self.data_source = data_source_cls(args)
        logger.info("[RolloutManager.__init__] data source initialized in %.2fs", time.perf_counter() - step_start)

        step_start = time.perf_counter()
        self.generate_rollout = load_function(self.args.rollout_function_path)
        self.eval_generate_rollout = load_function(self.args.eval_function_path)
        self.custom_reward_post_process_func = None
        if self.args.custom_reward_post_process_path is not None:
            self.custom_reward_post_process_func = load_function(self.args.custom_reward_post_process_path)
        self.custom_convert_samples_to_train_data_func = None
        if self.args.custom_convert_samples_to_train_data_path is not None:
            self.custom_convert_samples_to_train_data_func = load_function(
                self.args.custom_convert_samples_to_train_data_path
            )
        logger.info("[RolloutManager.__init__] rollout/eval functions loaded in %.2fs", time.perf_counter() - step_start)
        logger.info(f"import {self.args.rollout_function_path} as generate_rollout function.")
        logger.info(f"import {self.args.eval_function_path} as eval_generate_rollout function.")

        if self.args.debug_train_only:
            self.all_rollout_engines = []
        else:
            num_gpu_per_engine = min(args.rollout_num_gpus_per_engine, args.num_gpus_per_node)
            num_engines = args.rollout_num_gpus // num_gpu_per_engine
            self.all_rollout_engines = [None] * num_engines
        step_start = time.perf_counter()
        self.num_new_engines = init_rollout_engines(args, pg, self.all_rollout_engines)
        logger.info(
            "[RolloutManager.__init__] rollout engines initialized in %.2fs (num_new_engines=%s)",
            time.perf_counter() - step_start,
            self.num_new_engines,
        )
        if self.args.prm_enable and self.args.prm_num_gpus > 0:
            prm_num_gpu_per_engine = min(args.prm_num_gpus_per_engine, args.num_gpus_per_node)
            prm_num_engines = args.prm_num_gpus // prm_num_gpu_per_engine
            self.all_prm_engines = [None] * prm_num_engines
            step_start = time.perf_counter()
            self.num_new_prm_engines = init_prm_engines(args, prm_pg, self.all_prm_engines)
            logger.info(
                "[RolloutManager.__init__] prm engines initialized in %.2fs (num_new_prm_engines=%s)",
                time.perf_counter() - step_start,
                self.num_new_prm_engines,
            )
        else:
            self.all_prm_engines = []
            self.num_new_prm_engines = 0
        self.nodes_per_engine = max(1, args.rollout_num_gpus_per_engine // args.num_gpus_per_node)
        self.rollout_engine_lock = Lock.options(num_cpus=1, num_gpus=0).remote()
        self.rollout_id = -1
        self._startup_ready_checked = False
        # Rollout-only mode does not always call set_train_parallel_config().
        # Keep a safe default so generate() can always split data.
        self.train_parallel_config = {"dp_size": 1}

        self._metric_checker = MetricChecker.maybe_create(args)
        self._health_monitor = None
        if self.args.use_fault_tolerance:
            self._health_monitor = RolloutHealthMonitor(self, args)
            self._health_monitor.start()  # Start the monitor thread (in paused state)
            self._ci_fault_injection_pending = self.args.ci_test  # Flag for CI fault injection
            self._ci_fault_injection_reserved_indices: set[int] = set()
            self._ci_mid_generate_check_logged = False
            self._ci_mid_generate_skip_logged = False
            logger.info(
                "CI fault injection config: pending=%s mode=%s threshold=%s progress_fraction=%s engine_indices=%s delay_sec=%s global_once=%s lock_path=%s",
                self._ci_fault_injection_pending,
                _get_ci_fault_injection_mode(),
                _get_ci_fault_injection_rollout_id_threshold(),
                os.getenv("SLIME_CI_FAULT_INJECTION_PROGRESS_FRACTION", "0.5"),
                _get_ci_fault_injection_engine_indices(),
                os.getenv("SLIME_CI_FAULT_INJECTION_DELAY_SEC", "0"),
                os.getenv("SLIME_CI_FAULT_INJECTION_GLOBAL_ONCE", "1"),
                os.getenv("SLIME_CI_FAULT_INJECTION_LOCK_PATH", "/tmp/slime_ci_fault_injection_once.lock"),
            )

        logger.info("[RolloutManager.__init__] done in %.2fs", time.perf_counter() - init_start)

    def ready(self):
        if not self._startup_ready_checked:
            self._wait_for_rollout_engines_startup_ready(self.rollout_engines)
            self._startup_ready_checked = True
        logger.info("[RolloutManager.ready] ready")
        return True

    def _arm_ci_mid_generate_fallback_timer(self, rollout_id: int) -> None:
        fallback_seconds = float(os.getenv("SLIME_CI_FAULT_INJECTION_MID_FALLBACK_SEC", "120"))
        if fallback_seconds <= 0:
            logger.info("CI mid-generate fallback timer disabled (<=0s)")
            return

        logger.info(
            "CI mid-generate fallback timer armed: rollout_id=%s delay=%.1fs",
            rollout_id,
            fallback_seconds,
        )

        def _timer_worker(expected_rollout_id: int, delay_seconds: float):
            time.sleep(delay_seconds)
            if self.rollout_id != expected_rollout_id:
                logger.info(
                    "CI mid-generate fallback timer skipped: rollout_id changed (expected=%s, current=%s)",
                    expected_rollout_id,
                    self.rollout_id,
                )
                return
            if not self._ci_fault_injection_pending:
                logger.info("CI mid-generate fallback timer skipped: injection already consumed")
                return
            injected = self._try_ci_fault_injection(
                wait_for_detection=False,
                trigger=f"mid_generate_fallback_timer after {delay_seconds:.1f}s",
            )
            if injected:
                logger.info("CI mid-generate fallback timer fired successfully")
            else:
                logger.info("CI mid-generate fallback timer fired but no injection happened")

        threading.Thread(
            target=_timer_worker,
            args=(rollout_id, fallback_seconds),
            daemon=True,
            name=f"ci-mid-fallback-{rollout_id}",
        ).start()

    def _wait_for_rollout_engines_startup_ready(self, engines, *, wait_for_flush_cache: bool = True):
        engines = [engine for engine in engines if engine is not None]
        if not engines:
            return

        need_shadow_ready = bool(
            self.args.use_fault_tolerance
            and getattr(self.args, "sglang_enable_fast_restart", False)
            and not self.args.rollout_external
        )
        shadow_timeout = float(getattr(self.args, "sglang_shadow_worker_ready_timeout_seconds", 600.0))
        stabilization_seconds = float(getattr(self.args, "sglang_shadow_worker_stabilization_seconds", 0.0))
        logger.info(
            "Waiting for %d rollout engines to become startup-ready (require_shadow=%s, wait_for_flush_cache=%s, timeout=%.1fs, stabilization=%.1fs)",
            len(engines),
            need_shadow_ready,
            wait_for_flush_cache,
            shadow_timeout,
            stabilization_seconds,
        )
        start_ts = time.monotonic()
        ray.get(
            [
                engine.wait_until_ready.remote(
                    wait_for_shadow=need_shadow_ready,
                    shadow_timeout=shadow_timeout,
                    stabilization_seconds=stabilization_seconds,
                    wait_for_flush_cache=wait_for_flush_cache,
                )
                for engine in engines
            ]
        )
        logger.info(
            "All %d rollout engines reported startup-ready (require_shadow=%s, wait_for_flush_cache=%s, elapsed=%.1fs)",
            len(engines),
            need_shadow_ready,
            wait_for_flush_cache,
            time.monotonic() - start_ts,
        )

    def _try_ci_fault_injection(self, wait_for_detection: bool = True, trigger: str = "pre_generate"):
        """Inject CI fault by killing selected rollout engines only.

        Detection, handover, and recovery are intentionally delegated to the
        health monitor and the normal fault-tolerance pipeline.
        """
        if not self._ci_fault_injection_pending:
            return False

        if not _try_acquire_ci_injection_global_once():
            self._ci_fault_injection_pending = False
            logger.info("CI Fault Injection skipped: global-once lock already acquired by another manager")
            return False

        # Only inject fault once
        self._ci_fault_injection_pending = False
        logger.info(
            "CI Fault Injection: preparing pure-kill rollout-engine crash(es) (trigger=%s, wait_for_detection=%s)",
            trigger,
            wait_for_detection,
        )

        target_engine_indices = _get_ci_fault_injection_engine_indices()
        available_engine_indices = [
            index
            for index in target_engine_indices
            if 0 <= index < len(self.all_rollout_engines) and self.all_rollout_engines[index] is not None
        ]
        unavailable_engine_indices = [index for index in target_engine_indices if index not in available_engine_indices]

        if not available_engine_indices:
            logger.warning(
                "CI Fault Injection skipped: target engine indices %s are unavailable (num_engines=%s)",
                target_engine_indices,
                len(self.all_rollout_engines),
            )
            return False

        if unavailable_engine_indices:
            logger.warning(
                "CI Fault Injection: skipping unavailable engine indices %s; proceeding with available indices %s",
                unavailable_engine_indices,
                available_engine_indices,
            )

        self._ci_fault_injection_reserved_indices.update(available_engine_indices)

        delay_before_crash = float(os.getenv("SLIME_CI_FAULT_INJECTION_DELAY_SEC", "0"))
        if trigger.startswith("mid_generate"):
            delay_before_crash = float(os.getenv("SLIME_CI_FAULT_INJECTION_MID_DELAY_SEC", "0"))
        if delay_before_crash > 0:
            logger.info(
                "CI Fault Injection: Waiting %.1fs before simulating crash so shadow worker can finish warmup",
                delay_before_crash,
            )
            time.sleep(delay_before_crash)

        logger.info("CI Fault Injection: Simulating crash on engine indices %s during generate", available_engine_indices)

        try:
            for engine_position, engine_index in enumerate(available_engine_indices):
                logger.info(
                    "CI Fault Injection: simulating crash on engine %s (%s/%s)",
                    engine_index,
                    engine_position + 1,
                    len(available_engine_indices),
                )
                # This will cause the ray actor to exit.
                self.all_rollout_engines[engine_index].simulate_crash.remote()
            logger.info(
                "CI Fault Injection: pure-kill injection finished for engine indices %s without waiting for prior recovery; detection and recovery are delegated to the health monitor",
                available_engine_indices,
            )
            return True
        except Exception as e:
            logger.warning(f"CI Fault Injection failed: {e}")
            return False

    def _wait_for_ci_fault_recovery(self, engine_index: int) -> bool:
        rollout_engine_id = engine_index // max(1, self.nodes_per_engine)
        group_start = (engine_index // self.nodes_per_engine) * self.nodes_per_engine
        group_end = min(group_start + self.nodes_per_engine, len(self.all_rollout_engines))
        original_group = {
            index: self.all_rollout_engines[index]
            for index in range(group_start, group_end)
        }

        timeout_sec = float(os.getenv("SLIME_CI_FAULT_INJECTION_RECOVERY_TIMEOUT_SEC", "1800"))
        poll_interval_sec = float(os.getenv("SLIME_CI_FAULT_INJECTION_RECOVERY_POLL_SEC", "2"))
        health_timeout_sec = float(
            os.getenv(
                "SLIME_CI_FAULT_INJECTION_RECOVERY_HEALTH_TIMEOUT_SEC",
                str(max(5.0, getattr(self.args, "rollout_health_check_timeout", 10.0))),
            )
        )
        start_ts = time.monotonic()
        last_log_ts = start_ts - 60.0

        while time.monotonic() - start_ts < timeout_sec:
            current_group = {
                index: self.all_rollout_engines[index]
                for index in range(group_start, group_end)
            }
            all_replaced = all(
                current_group[index] is not None and current_group[index] != original_group[index]
                for index in range(group_start, group_end)
            )
            recovery_in_progress = self.is_rollout_engine_health_check_suppressed(rollout_engine_id)
            if all_replaced and not recovery_in_progress:
                try:
                    health_results = ray.get(
                        [
                            current_group[index].check_health.remote(timeout=health_timeout_sec)
                            for index in range(group_start, group_end)
                        ]
                    )
                except Exception as exc:
                    logger.info(
                        "CI Fault Injection: health check while waiting for recovery of engine group [%s, %s) is not ready yet: %s",
                        group_start,
                        group_end,
                        exc,
                    )
                    health_results = None
                if health_results and all(health_results):
                    logger.info(
                        "CI Fault Injection: rollout engine recovery finished for engine %s via group [%s, %s) in %.1fs",
                        engine_index,
                        group_start,
                        group_end,
                        time.monotonic() - start_ts,
                    )
                    return True

            now = time.monotonic()
            if now - last_log_ts >= 15.0:
                replacement_state = {
                    index: {
                        "replaced": current_group[index] is not None and current_group[index] != original_group[index],
                        "present": current_group[index] is not None,
                    }
                    for index in range(group_start, group_end)
                }
                logger.info(
                    "CI Fault Injection: waiting for recovery of engine %s via group [%s, %s) (elapsed=%.1fs timeout=%.1fs recovery_in_progress=%s state=%s)",
                    engine_index,
                    group_start,
                    group_end,
                    now - start_ts,
                    timeout_sec,
                    recovery_in_progress,
                    replacement_state,
                )
                last_log_ts = now

            time.sleep(poll_interval_sec)

        logger.warning(
            "CI Fault Injection: timed out waiting %.1fs for recovery of engine %s via group [%s, %s)",
            timeout_sec,
            engine_index,
            group_start,
            group_end,
        )
        return False

    def dispose(self):
        if self._metric_checker is not None:
            self._metric_checker.dispose()
        if self._health_monitor is not None:
            self._health_monitor.stop()

    # TODO maybe rename "rollout_engines" and "all_rollout_engines" later
    @property
    def rollout_engines(self):
        # when doing multi-node serving, we will only send request to node-0 for each engine.
        return self.all_rollout_engines[:: self.nodes_per_engine]

    def _collect_engine_side_shadow_handover_reconnects(self) -> int:
        indexed_engines = [(index, engine) for index, engine in enumerate(self.all_rollout_engines) if engine is not None]
        if not indexed_engines:
            return 0

        try:
            results = ray.get(
                [engine.consume_shadow_handover_reconnect_event.remote() for _, engine in indexed_engines]
            )
        except Exception:
            logger.exception("Failed to collect engine-side shadow handover reconnect events")
            return 0

        groups_to_reconnect: dict[int, list[str]] = {}
        for (index, _), result in zip(indexed_engines, results):
            if not result.get("pending"):
                continue
            rollout_engine_id = index // max(1, self.nodes_per_engine)
            reason = result.get("reason") or f"engine index {index}"
            groups_to_reconnect.setdefault(rollout_engine_id, []).append(str(reason))

        if not groups_to_reconnect:
            return 0

        group_ids = sorted(groups_to_reconnect)
        reason_parts = [
            f"rollout_engine={group_id} reasons={groups_to_reconnect[group_id]}"
            for group_id in group_ids
        ]
        self.mark_rollout_engines_need_weight_update_reconnect(
            count=len(group_ids),
            reason="engine-side shadow handover: " + "; ".join(reason_parts),
        )
        logger.info(
            "Collected engine-side shadow handover reconnect events for rollout engines %s",
            group_ids,
        )
        return len(group_ids)

    def get_rollout_engines_and_lock(self, include_prm=False):
        self._collect_engine_side_shadow_handover_reconnects()
        engines = list(self.rollout_engines)
        with self._num_new_engines_lock:
            num_new = self.num_new_engines
            if include_prm:
                prm_engines = [e for e in getattr(self, "all_prm_engines", []) if e is not None]
                engines.extend(prm_engines)
                num_new += getattr(self, "num_new_prm_engines", 0)
        return engines, self.rollout_engine_lock, num_new

    def get_weight_update_reconnect_debug_state(self) -> dict[str, Any]:
        self._collect_engine_side_shadow_handover_reconnects()
        with self._num_new_engines_lock:
            return {
                "num_new_engines": self.num_new_engines,
                "pending_shadow_handover_reconnects": self._pending_shadow_handover_reconnects,
                "pending_shadow_handover_reconnect_reasons": list(self._pending_shadow_handover_reconnect_reasons),
            }

    def ack_shadow_handover_weight_update_reconnect(self) -> None:
        with self._num_new_engines_lock:
            before = self._pending_shadow_handover_reconnects
            reasons = list(self._pending_shadow_handover_reconnect_reasons)
            self._pending_shadow_handover_reconnects = 0
            self._pending_shadow_handover_reconnect_reasons.clear()
        logger.info(
            "Acknowledged shadow-handover weight-update reconnect obligation (before=%s, reasons=%s)",
            before,
            reasons,
        )

    def get_num_rollout_per_epoch(self):
        assert self.args.rollout_global_dataset
        return len(self.data_source.dataset) // self.args.rollout_batch_size

    def generate(self, rollout_id):
        start_time = time.time()
        self.rollout_id = rollout_id
        self.health_monitoring_resume()
        if (
            self.args.ci_test
            and self.args.use_fault_tolerance
            and _get_ci_fault_injection_mode() == "mid_generate"
            and rollout_id >= _get_ci_fault_injection_rollout_id_threshold()
        ):
            self._arm_ci_mid_generate_fallback_timer(rollout_id)
        if (
            self.args.ci_test
            and self.args.use_fault_tolerance
            and _get_ci_fault_injection_mode() == "pre_generate"
            and rollout_id >= _get_ci_fault_injection_rollout_id_threshold()
        ):
            self._try_ci_fault_injection(wait_for_detection=True, trigger="pre_generate")
        data, metrics = self._get_rollout_data(rollout_id=rollout_id)
        self._save_debug_rollout_data(data, rollout_id=rollout_id, evaluation=False)
        total_rollout_time = time.time() - start_time
        effective_rollout_time = total_rollout_time
        if rollout_id == 0:
            metrics = metrics or {}
            first_sample_wait_time = metrics.get("perf/rollout_time_before_first_sample")
            metrics["perf/rollout_time_total"] = total_rollout_time
            if (
                isinstance(first_sample_wait_time, (int, float))
                and 0 <= first_sample_wait_time < total_rollout_time
            ):
                effective_rollout_time = max(1e-6, total_rollout_time - first_sample_wait_time)
                metrics["perf/rollout_time_after_first_sample"] = effective_rollout_time
                logger.info(
                    "Adjusted perf/rollout_time for rollout_id=0: total=%.2fs, before_first_sample=%.2fs, effective=%.2fs",
                    total_rollout_time,
                    first_sample_wait_time,
                    effective_rollout_time,
                )
        _log_rollout_data(rollout_id, self.args, data, metrics, effective_rollout_time)
        data = self._convert_samples_to_train_data(data)
        return self._split_train_data_by_dp(data, self.train_parallel_config["dp_size"])

    def eval(self, rollout_id):
        if self.args.debug_train_only:
            # if debug train only, we don't generate evaluation data
            return
        self.health_monitoring_resume()

        result = call_rollout_fn(self.eval_generate_rollout, self.args, rollout_id, self.data_source, evaluation=True)
        data = result.data
        self._save_debug_rollout_data(data, rollout_id=rollout_id, evaluation=True)
        metrics = _log_eval_rollout_data(rollout_id, self.args, data, result.metrics)
        if self._metric_checker is not None:
            self._metric_checker.on_eval(metrics)

    def save(self, rollout_id):
        self.data_source.save(rollout_id)

    def load(self, rollout_id=None):
        self.data_source.load(rollout_id)

    def offload(self):
        self.health_monitoring_pause()
        return ray.get(
            [engine.release_memory_occupation.remote() for engine in self.rollout_engines if engine is not None]
        )

    def onload(self, tags: list[str] | None = None):
        return ray.get(
            [
                engine.resume_memory_occupation.remote(tags=tags)
                for engine in self.rollout_engines
                if engine is not None
            ]
        )

    def onload_weights(self):
        self.onload(tags=[GPU_MEMORY_TYPE_WEIGHTS])

    def onload_kv(self):
        self.onload(tags=[GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_CUDA_GRAPH])

    def _rollout_worker_type_for_index(self, index: int) -> str:
        if self.args.prefill_num_servers is None:
            return "regular"
        num_gpu_per_engine = min(self.args.rollout_num_gpus_per_engine, self.args.num_gpus_per_node)
        prefill_num_servers = self.args.prefill_num_servers * self.args.rollout_num_gpus_per_engine // num_gpu_per_engine
        return "prefill" if index < prefill_num_servers else "decode"

    def _build_non_fast_restart_remote_instance_overrides(self, dead_indices: list[int]) -> dict[int, dict]:
        if getattr(self.args, "sglang_enable_fast_restart", False):
            return {}
        if self.args.rollout_external:
            logger.info("Skip remote-instance recovery override because rollout_external=True")
            return {}
        if self.nodes_per_engine != 1:
            logger.info(
                "Skip remote-instance recovery override because nodes_per_engine=%s (only single-node engines supported)",
                self.nodes_per_engine,
            )
            return {}
        if len(dead_indices) >= 3:
            logger.warning(
                "Skip remote-instance recovery override because dead rollout engine count=%s "
                "is large enough to risk overloading the remaining seed worker(s); "
                "falling back to cold load from storage for dead_indices=%s",
                len(dead_indices),
                dead_indices,
            )
            return {}

        excluded_indices = set(dead_indices)
        excluded_indices.update(getattr(self, "_ci_fault_injection_reserved_indices", set()))
        seed_health_timeout = float(os.getenv("SLIME_REMOTE_INSTANCE_RECOVERY_SEED_HEALTH_TIMEOUT_SEC", "2"))

        healthy_candidates: list[tuple[int, object]] = [
            (index, engine)
            for index, engine in enumerate(self.all_rollout_engines)
            if engine is not None and index not in excluded_indices
        ]
        if not healthy_candidates:
            logger.warning(
                "No healthy rollout engine available as remote-instance seed; falling back to cold load "
                "(dead_indices=%s excluded_indices=%s)",
                dead_indices,
                sorted(excluded_indices),
            )
            return {}

        overrides: dict[int, dict] = {}
        for dead_index in dead_indices:
            target_worker_type = self._rollout_worker_type_for_index(dead_index)
            candidate_pool = [
                (index, engine)
                for index, engine in healthy_candidates
                if self._rollout_worker_type_for_index(index) == target_worker_type
            ] or healthy_candidates

            seed_info = None
            seed_index = None
            for candidate_index, candidate_engine in candidate_pool:
                try:
                    ray.get(candidate_engine.health_generate.remote(timeout=seed_health_timeout))
                    seed_info = ray.get(candidate_engine.get_remote_instance_weight_loader_seed_info.remote())
                    seed_index = candidate_index
                    break
                except Exception as exc:
                    logger.warning(
                        "Failed to use rollout engine index=%s as remote-instance seed for dead index=%s "
                        "(health_timeout=%.1fs): %s",
                        candidate_index,
                        dead_index,
                        seed_health_timeout,
                        exc,
                    )
            if seed_info is None:
                logger.warning(
                    "No usable remote-instance seed found for dead rollout engine index=%s; falling back to cold load",
                    dead_index,
                )
                continue

            overrides[dead_index] = seed_info
            logger.info(
                "Remote-instance recovery override prepared for dead rollout engine index=%s using healthy index=%s seed=%s:%s ports=%s",
                dead_index,
                seed_index,
                seed_info["seed_instance_ip"],
                seed_info["seed_instance_service_port"],
                seed_info["send_weights_group_ports"],
            )

        return overrides

    def recover_rollout_engines(self):
        """Restart any dead rollout engines and update num_new_engines for update_weights detection."""
        if self.rollout_id == -1:
            with self._num_new_engines_lock:
                pending_num_new = self.num_new_engines
            return self.rollout_engines, self.rollout_engine_lock, pending_num_new

        dead_indices = [i for i, engine in enumerate(self.all_rollout_engines) if engine is None]
        if not dead_indices:
            with self._num_new_engines_lock:
                pending_num_new = self.num_new_engines
            logger.info(
                "No dead rollout engines detected during recovery; pending num_new_engines=%s",
                pending_num_new,
            )
            return self.rollout_engines, self.rollout_engine_lock, pending_num_new

        recover_start_ts = time.monotonic()
        logger.info("Recovery started for dead rollout engine indices=%s", dead_indices)
        recovery_overrides = self._build_non_fast_restart_remote_instance_overrides(dead_indices)
        old_recovery_overrides = getattr(self.args, "_sglang_recovery_remote_instance_overrides", None)
        self.args._sglang_recovery_remote_instance_overrides = recovery_overrides
        try:
            recovered_num_new = init_rollout_engines(self.args, self.pg, self.all_rollout_engines)
        finally:
            if old_recovery_overrides is None:
                try:
                    delattr(self.args, "_sglang_recovery_remote_instance_overrides")
                except AttributeError:
                    pass
            else:
                self.args._sglang_recovery_remote_instance_overrides = old_recovery_overrides
        with self._num_new_engines_lock:
            self.num_new_engines = recovered_num_new
        logger.info(
            "Engine init phase finished for dead indices=%s (num_new_engines=%s, remote_instance_overrides=%s, elapsed=%.1fs)",
            dead_indices,
            recovered_num_new,
            sorted(recovery_overrides.keys()),
            time.monotonic() - recover_start_ts,
        )
        if hasattr(self, "_ci_fault_injection_reserved_indices"):
            self._ci_fault_injection_reserved_indices.difference_update(dead_indices)
        if dead_indices:
            ready_wait_start_ts = time.monotonic()
            self._wait_for_rollout_engines_startup_ready(
                [self.all_rollout_engines[i] for i in dead_indices],
                wait_for_flush_cache=False,
            )
            logger.info(
                "Startup-ready phase finished for dead indices=%s (elapsed=%.1fs)",
                dead_indices,
                time.monotonic() - ready_wait_start_ts,
            )
        logger.info(
            "Recovered %s dead rollout engines (dead_indices=%s, total_elapsed=%.1fs)",
            recovered_num_new,
            dead_indices,
            time.monotonic() - recover_start_ts,
        )
        assert recovered_num_new == len(dead_indices), "num_new_engines does not match dead_indices length"
        if self.args.offload_rollout and dead_indices:
            new_engines = [self.all_rollout_engines[i] for i in dead_indices]
            ray.get([engine.release_memory_occupation.remote() for engine in new_engines])
            ray.get([engine.resume_memory_occupation.remote(tags=[GPU_MEMORY_TYPE_WEIGHTS]) for engine in new_engines])

        return self.rollout_engines, self.rollout_engine_lock, recovered_num_new

    def clear_num_new_engines(self, consumed: int | None = None):
        # This counter tracks rollout engines whose weight-update connections must be rebuilt,
        # including normal restarts and shadow-worker handovers.
        #
        # Use decrement semantics when `consumed` is provided so we don't lose
        # newly-arrived reconnect events that may race with an in-flight
        # update_weights cycle.
        if consumed is None:
            with self._num_new_engines_lock:
                consumed = self.num_new_engines
        consumed = max(0, int(consumed))
        with self._num_new_engines_lock:
            before = self.num_new_engines
            self.num_new_engines = max(0, self.num_new_engines - consumed)
            after = self.num_new_engines
        logger.info(
            "Cleared rollout reconnect counter by consumed=%s (before=%s, after=%s)",
            consumed,
            before,
            after,
        )
        if hasattr(self, "num_new_prm_engines"):
            # Keep existing behavior for PRM engines (full clear), since PRM
            # reconnection currently isn't consumed independently in training.
            self.num_new_prm_engines = 0

    def mark_rollout_engines_need_weight_update_reconnect(self, count: int = 1, reason: str = "unknown") -> int:
        if count <= 0:
            with self._num_new_engines_lock:
                return self.num_new_engines
        reason_text = str(reason)
        with self._num_new_engines_lock:
            self.num_new_engines += count
            pending = self.num_new_engines
            if "handover" in reason_text.lower():
                self._pending_shadow_handover_reconnects += count
                self._pending_shadow_handover_reconnect_reasons.append(reason_text)
                pending_shadow = self._pending_shadow_handover_reconnects
            else:
                pending_shadow = self._pending_shadow_handover_reconnects
        logger.info(
            "Marked %s rollout engine(s) for weight-update reconnection due to %s; pending reconnect count=%s pending_shadow_handover_reconnects=%s",
            count,
            reason_text,
            pending,
            pending_shadow,
        )
        return pending

    def health_monitoring_pause(self) -> None:
        if self._health_monitor is not None:
            self._health_monitor.pause()

    def health_monitoring_resume(self) -> None:
        if self._health_monitor is not None:
            self._health_monitor.resume()

    def is_rollout_engine_health_check_suppressed(self, rollout_engine_id: int) -> bool:
        if self._health_monitor is None:
            return False
        return self._health_monitor.is_rollout_engine_suppressed(rollout_engine_id)

    def check_weights(self, action: str):
        return ray.get([engine.check_weights.remote(action=action) for engine in self.rollout_engines])

    def _get_rollout_data(self, rollout_id):
        if self.args.load_debug_rollout_data:
            data = torch.load(
                self.args.load_debug_rollout_data.format(rollout_id=rollout_id),
                weights_only=False,
            )["samples"]
            data = [Sample.from_dict(sample) for sample in data]
            if (ratio := self.args.load_debug_rollout_data_subsample) is not None:
                original_num_rows = len(data)
                rough_subsample_num_rows = int(original_num_rows * ratio)
                data = data[: rough_subsample_num_rows // 2] + data[-rough_subsample_num_rows // 2 :]
                logger.info(
                    f"Subsample loaded debug rollout data using {ratio=} and change num rows {original_num_rows} -> {len(data)}"
                )
            metrics = None
        else:
            data = call_rollout_fn(self.generate_rollout, self.args, rollout_id, self.data_source, evaluation=False)
            metrics = data.metrics
            data = data.samples
            # flatten the data if it is a list of lists
            while isinstance(data[0], list):
                data = list(itertools.chain.from_iterable(data))

            if not self.args.disable_rollout_trim_samples:
                global_batch_size = self.args.global_batch_size
                target_steps_per_rollout = getattr(self.args, "num_steps_per_rollout", None)
                # dynamic_history can expand one rollout into many step-wise samples.
                # In that case, honor num_steps_per_rollout by deriving a per-rollout
                # dynamic global batch size from the actual collected sample count.
                auto_dynamic_for_history = getattr(self.args, "dynamic_history", False) and target_steps_per_rollout is not None
                use_dynamic_gbs = self.args.use_dynamic_global_batch_size or auto_dynamic_for_history
                dynamic_target_steps = target_steps_per_rollout if auto_dynamic_for_history else None
                if use_dynamic_gbs:
                    logger.info(f"Collected {len(data)} samples from rollout to train with dynamic global batch size")
                    # TODO: this is a temporary solution, we should directly save dynamic_global_batch_size to rollout data
                    self._dynamic_global_batch_size = self._compute_dynamic_global_batch_size(
                        len(data), target_steps=dynamic_target_steps
                    )
                    global_batch_size = self._dynamic_global_batch_size

                if len(data) % global_batch_size != 0:
                    trim_len = (len(data) // global_batch_size) * global_batch_size
                    if trim_len == 0:
                        raise ValueError(f"Not enough samples {len(data)} for global_batch_size {global_batch_size}")
                    origin_data_length = len(data)
                    data = data[:trim_len]
                    logger.info(f"trim number of samples from {origin_data_length} to {trim_len}")
                logger.info(f"Final collected {len(data)} samples from rollout to train")

        return data, metrics

    def _compute_dynamic_global_batch_size(self, num_samples: int, target_steps: int | None = None) -> int:
        """Calculate dynamic global_batch_size from actual per-rollout samples.

        If target_steps is provided, choose global_batch_size to keep the realized
        number of training steps per rollout close to target_steps.
        Otherwise fallback to one-step behavior for backward compatibility.
        """
        dp_size = self.train_parallel_config["dp_size"]
        original_gbs = self.args.global_batch_size

        desired_steps = int(target_steps) if target_steps is not None and target_steps > 0 else 1
        # Target per-step samples, then round down to a multiple of dp_size.
        per_step_target = max(1, num_samples // desired_steps)
        dynamic_gbs = (per_step_target // dp_size) * dp_size

        if dynamic_gbs == 0:
            # Too few samples, use at least dp_size.
            dynamic_gbs = dp_size
            logger.warning(f"num_samples={num_samples} < dp_size={dp_size}, using dp_size as global_batch_size")

        realized_steps = max(1, num_samples // dynamic_gbs)
        # Calculate how many samples will be discarded after trim.
        wasted = num_samples % dynamic_gbs

        if dynamic_gbs != original_gbs or wasted > 0 or realized_steps != desired_steps:
            logger.info(
                f"Dynamic global_batch_size: {original_gbs} -> {dynamic_gbs} "
                f"(num_samples={num_samples}, dp_size={dp_size}, "
                f"target_steps={desired_steps}, realized_steps={realized_steps}, wasted={wasted})"
            )

        return dynamic_gbs

    def _save_debug_rollout_data(self, data, rollout_id, evaluation: bool):
        # TODO to be refactored (originally Buffer._set_data)
        if (path_template := self.args.save_debug_rollout_data) is not None:
            path = Path(path_template.format(rollout_id=("eval_" if evaluation else "") + str(rollout_id)))
            logger.info(f"Save debug rollout data to {path}")
            path.parent.mkdir(parents=True, exist_ok=True)

            # TODO may improve the format
            if evaluation:
                dump_data = dict(
                    samples=[sample.to_dict() for dataset_name, info in data.items() for sample in info["samples"]]
                )
            else:
                dump_data = dict(
                    samples=[sample.to_dict() for sample in data],
                )

            torch.save(dict(rollout_id=rollout_id, **dump_data), path)

    def _post_process_rewards(self, samples: list[Sample] | list[list[Sample]]):
        if self.custom_reward_post_process_func is not None:
            return self.custom_reward_post_process_func(self.args, samples)

        raw_rewards = [sample.get_reward_value(self.args) for sample in samples]
        if self.args.advantage_estimator in ["grpo", "gspo"] and self.args.rewards_normalization:
            if getattr(self.args, "dynamic_history", False):
                # dynamic_history + GRPO:
                # normalize one outcome per trajectory inside each task(group),
                # then broadcast that normalized value to all step samples in
                # the same trajectory.
                traj_reward_by_key: dict[tuple[int, int], float] = {}
                group_to_keys: dict[int, list[tuple[int, int]]] = {}
                key_by_sample: list[tuple[int, int]] = []
                for i, sample in enumerate(samples):
                    group_idx = int(sample.group_index) if sample.group_index is not None else -1
                    traj_idx = int(sample.index) if sample.index is not None else i
                    key = (group_idx, traj_idx)
                    key_by_sample.append(key)
                    if key not in traj_reward_by_key:
                        traj_reward_by_key[key] = float(raw_rewards[i])
                        group_to_keys.setdefault(group_idx, []).append(key)

                normalized_by_key: dict[tuple[int, int], float] = {}
                for _, keys in group_to_keys.items():
                    vals = torch.tensor([traj_reward_by_key[k] for k in keys], dtype=torch.float32)
                    vals = vals - vals.mean(dim=-1, keepdim=True)
                    if self.args.grpo_std_normalization:
                        if len(keys) > 1:
                            vals = vals / (vals.std(dim=-1, keepdim=True) + 1e-6)
                        else:
                            vals = torch.zeros_like(vals)
                    for j, key in enumerate(keys):
                        normalized_by_key[key] = float(vals[j].item())

                rewards = [normalized_by_key[key] for key in key_by_sample]
                return raw_rewards, rewards

            # non-dynamic_history + GRPO/GSPO:
            # normalize reward directly inside each task(group).
            group_to_indices: dict[int, list[int]] = {}
            for i, sample in enumerate(samples):
                group_idx = int(sample.group_index) if sample.group_index is not None else -1
                group_to_indices.setdefault(group_idx, []).append(i)

            rewards = list(raw_rewards)
            for _, idxs in group_to_indices.items():
                vals = torch.tensor([raw_rewards[i] for i in idxs], dtype=torch.float32)
                vals = vals - vals.mean(dim=-1, keepdim=True)
                if self.args.grpo_std_normalization:
                    if len(idxs) > 1:
                        vals = vals / (vals.std(dim=-1, keepdim=True) + 1e-6)
                    else:
                        vals = torch.zeros_like(vals)
                for j, sample_idx in enumerate(idxs):
                    rewards[sample_idx] = float(vals[j].item())
            return raw_rewards, rewards

        if self.args.advantage_estimator in ["reinforce_plus_plus_baseline"] and self.args.rewards_normalization:
            # group norm
            rewards = torch.tensor(raw_rewards, dtype=torch.float)
            if rewards.shape[-1] == self.args.n_samples_per_prompt * self.args.rollout_batch_size:
                rewards = rewards.reshape(-1, self.args.n_samples_per_prompt)
            else:
                # when samples count are not equal in each group
                rewards = rewards.view(-1, rewards.shape[-1])
            mean = rewards.mean(dim=-1, keepdim=True)
            rewards = rewards - mean

            if self.args.advantage_estimator in ["grpo", "gspo"] and self.args.grpo_std_normalization:
                std = rewards.std(dim=-1, keepdim=True)
                rewards = rewards / (std + 1e-6)

            return raw_rewards, rewards.flatten().tolist()

        return raw_rewards, raw_rewards

    def _drop_constant_reward_groups(self, samples: list[Sample]) -> list[Sample]:
        """Drop GRPO/GSPO groups whose rewards are all identical.

        Keep at least one group to avoid empty training data.
        """
        if not samples:
            return samples
        if self.args.advantage_estimator not in ["grpo", "gspo"] or not self.args.rewards_normalization:
            return samples

        raw_rewards = [float(sample.get_reward_value(self.args)) for sample in samples]
        group_to_indices: dict[int, list[int]] = {}
        for i, sample in enumerate(samples):
            group_idx = int(sample.group_index) if sample.group_index is not None else -1
            group_to_indices.setdefault(group_idx, []).append(i)

        constant_groups: list[int] = []
        for group_idx, idxs in group_to_indices.items():
            vals = [raw_rewards[i] for i in idxs]
            if len(vals) == 0:
                continue
            if max(vals) - min(vals) <= 1e-12:
                constant_groups.append(group_idx)

        if not constant_groups:
            return samples

        keep_groups = [g for g in group_to_indices.keys() if g not in set(constant_groups)]
        dropped_groups = list(constant_groups)
        if not keep_groups:
            # Keep one full group so batch is never empty.
            keep_group = next(iter(group_to_indices.keys()))
            keep_groups = [keep_group]
            dropped_groups = [g for g in constant_groups if g != keep_group]

        keep_set = set(keep_groups)
        filtered_samples = []
        for sample in samples:
            group_idx = int(sample.group_index) if sample.group_index is not None else -1
            if group_idx in keep_set:
                filtered_samples.append(sample)

        if len(filtered_samples) != len(samples):
            logger.warning(
                "Dropped constant-reward groups for %s: dropped=%s kept=%s samples %d -> %d",
                self.args.advantage_estimator,
                dropped_groups,
                keep_groups,
                len(samples),
                len(filtered_samples),
            )
        return filtered_samples

    def _post_process_step_wise_rewards(
        self, samples: list[Sample]
    ) -> tuple[list[list[float]], list[list[list[int]]], list[list[int]], list[int]]:
        """Build and normalize step-wise rewards metadata for training.

        Returns:
            step_wise_step_rewards, step_wise_step_token_spans, step_wise_step_indices, group_indices
        """
        step_wise_step_rewards: list[list[float]] = []
        step_wise_step_token_spans: list[list[list[int]]] = []
        step_wise_step_indices: list[list[int]] = []
        group_indices: list[int] = []

        for sample in samples:
            metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
            step_wise_meta = metadata.get("step_wise", {}) if isinstance(metadata, dict) else {}
            if not isinstance(step_wise_meta, dict):
                step_wise_meta = {}

            # Preferred field for step_wise: reward for each step has already
            # been composed as (step_prm + outcome) in reward_func.
            raw_step_rewards = step_wise_meta.get("step_scores_with_outcome", None)
            if raw_step_rewards is None:
                # Backward-compatible fallback for old metadata.
                raw_step_rewards = step_wise_meta.get("step_scores", [])
            raw_step_spans = step_wise_meta.get("step_token_spans", [])
            raw_step_indices = step_wise_meta.get("step_indices", None)

            if isinstance(raw_step_rewards, (tuple, list)):
                step_rewards = [float(x) for x in raw_step_rewards]
            else:
                step_rewards = []
            if isinstance(raw_step_indices, (tuple, list)):
                step_indices = [int(x) for x in raw_step_indices]
            else:
                step_indices = list(range(len(step_rewards)))

            step_spans = []
            if isinstance(raw_step_spans, (tuple, list)):
                for span in raw_step_spans:
                    if (
                        isinstance(span, (tuple, list))
                        and len(span) == 2
                        and span[0] is not None
                        and span[1] is not None
                    ):
                        step_spans.append([int(span[0]), int(span[1])])

            # dynamic_history mode may intentionally omit explicit step spans.
            # In that case, infer one span from loss_mask (first/last 1).
            if len(step_spans) == 0 and getattr(self.args, "dynamic_history", False) and sample.loss_mask is not None:
                active_positions = [i for i, m in enumerate(sample.loss_mask) if int(m) == 1]
                if active_positions:
                    step_spans = [[active_positions[0], active_positions[-1] + 1]]
                    if len(step_rewards) == 0:
                        step_rewards = [float(sample.get_reward_value(self.args))]
                    if len(step_indices) == 0:
                        step_indices = [0]

            if not (len(step_rewards) == len(step_spans) == len(step_indices)):
                aligned_len = min(len(step_rewards), len(step_spans), len(step_indices))
                logger.warning(
                    "Step-wise metadata length mismatch for sample %s: rewards=%s spans=%s indices=%s, trim to %s",
                    sample.index,
                    len(step_rewards),
                    len(step_spans),
                    len(step_indices),
                    aligned_len,
                )
                step_rewards = step_rewards[:aligned_len]
                step_spans = step_spans[:aligned_len]
                step_indices = step_indices[:aligned_len]

            step_wise_step_rewards.append(step_rewards)
            step_wise_step_token_spans.append(step_spans)
            step_wise_step_indices.append(step_indices)
            group_indices.append(int(sample.group_index) if sample.group_index is not None else -1)

        # step_wise normalization is done in rollout for clarity:
        # normalize within same (task group, step_index) across trajectories.
        if self.args.rewards_normalization:
            stats: dict[tuple[int, int], tuple[float, float, int]] = {}
            for i, rewards_i in enumerate(step_wise_step_rewards):
                group_idx = int(group_indices[i])
                indices_i = step_wise_step_indices[i]
                aligned_len = min(len(rewards_i), len(indices_i))
                for pos in range(aligned_len):
                    key = (group_idx, int(indices_i[pos]))
                    v = float(rewards_i[pos])
                    sum_v, sum_sq_v, count_v = stats.get(key, (0.0, 0.0, 0))
                    stats[key] = (sum_v + v, sum_sq_v + v * v, count_v + 1)

            # Drop constant normalization groups (same reward in one
            # (group_index, step_index) bucket). Keep at least one group.
            all_keys = list(stats.keys())
            constant_keys: set[tuple[int, int]] = set()
            for key, (sum_v, sum_sq_v, count_v) in stats.items():
                if count_v <= 1:
                    # Keep single-sample buckets: no normalization, use raw reward.
                    continue
                mean_v = sum_v / count_v
                var_v = max(sum_sq_v / count_v - mean_v * mean_v, 0.0)
                if var_v <= 1e-12:
                    constant_keys.add(key)

            kept_keys = [k for k in all_keys if k not in constant_keys]
            dropped_keys = list(constant_keys)
            if not kept_keys and all_keys:
                keep_key = all_keys[0]
                kept_keys = [keep_key]
                dropped_keys = [k for k in dropped_keys if k != keep_key]

            kept_key_set = set(kept_keys)

            for i, rewards_i in enumerate(step_wise_step_rewards):
                group_idx = int(group_indices[i])
                indices_i = step_wise_step_indices[i]
                spans_i = step_wise_step_token_spans[i]
                aligned_len = min(len(rewards_i), len(indices_i))
                normalized_i = []
                filtered_indices_i = []
                filtered_spans_i = []
                for pos in range(aligned_len):
                    key = (group_idx, int(indices_i[pos]))
                    if key not in kept_key_set:
                        continue
                    sum_v, sum_sq_v, count_v = stats[key]
                    v = float(rewards_i[pos])
                    if count_v > 1:
                        mean_v = sum_v / count_v
                        var_v = max(sum_sq_v / count_v - mean_v * mean_v, 0.0)
                        std_v = var_v**0.5
                        v = (v - mean_v) / (std_v + 1e-6)
                    normalized_i.append(v)
                    filtered_indices_i.append(int(indices_i[pos]))
                    filtered_spans_i.append(spans_i[pos])
                step_wise_step_rewards[i] = normalized_i
                step_wise_step_indices[i] = filtered_indices_i
                step_wise_step_token_spans[i] = filtered_spans_i

                # If this sample loses all step entries, mark it as non-trainable.
                if len(normalized_i) == 0:
                    samples[i].remove_sample = True

            if dropped_keys:
                logger.warning(
                    "Dropped constant step_wise groups: dropped=%s kept=%s",
                    dropped_keys,
                    kept_keys,
                )

        return step_wise_step_rewards, step_wise_step_token_spans, step_wise_step_indices, group_indices

    def _convert_samples_to_train_data(self, samples: list[Sample] | list[list[Sample]]):
        """
        Convert inference generated samples to training data.
        """
        if self.custom_convert_samples_to_train_data_func is not None:
            return self.custom_convert_samples_to_train_data_func(self.args, samples)

        samples = self._drop_constant_reward_groups(samples)
        raw_rewards, rewards = self._post_process_rewards(samples)

        assert len(raw_rewards) == len(samples)
        assert len(rewards) == len(samples)

        train_data = {
            "tokens": [sample.tokens for sample in samples],
            "response_lengths": [sample.response_length for sample in samples],
            # some reward model, e.g. remote rm, may return multiple rewards,
            # we could use key to select the reward.
            "rewards": rewards,
            "raw_reward": raw_rewards,
            "truncated": [1 if sample.status == Sample.Status.TRUNCATED else 0 for sample in samples],
            "sample_indices": [sample.index for sample in samples],
        }
        if self.args.advantage_estimator in ["grpo", "gspo"]:
            train_data["group_indices"] = [int(sample.group_index) if sample.group_index is not None else -1 for sample in samples]

        if self.args.advantage_estimator == "step_wise":
            (
                step_wise_step_rewards,
                step_wise_step_token_spans,
                step_wise_step_indices,
                group_indices,
            ) = self._post_process_step_wise_rewards(samples)

            train_data["step_wise_step_rewards"] = step_wise_step_rewards
            train_data["step_wise_step_token_spans"] = step_wise_step_token_spans
            train_data["step_wise_step_indices"] = step_wise_step_indices
            train_data["group_indices"] = group_indices

        # loss mask
        # TODO: compress the loss mask
        loss_masks = []
        for sample in samples:
            # always instantiate loss_mask if not provided
            if sample.loss_mask is None:
                sample.loss_mask = [1] * sample.response_length

            assert (
                len(sample.loss_mask) == sample.response_length
            ), f"loss mask length {len(sample.loss_mask)} != response length {sample.response_length}"
            if sample.remove_sample:
                sample.loss_mask = [0] * sample.response_length
            loss_masks.append(sample.loss_mask)
        train_data["loss_masks"] = loss_masks

        # overwriting the raw reward
        if samples[0].metadata and "raw_reward" in samples[0].metadata:
            train_data["raw_reward"] = [sample.metadata["raw_reward"] for sample in samples]

        # For rollout buffer
        if samples[0].metadata and "round_number" in samples[0].metadata:
            train_data["round_number"] = [sample.metadata["round_number"] for sample in samples]

        # Add rollout log probabilities for off-policy correction
        if samples[0].rollout_log_probs is not None:
            train_data["rollout_log_probs"] = [sample.rollout_log_probs for sample in samples]

        if samples[0].rollout_routed_experts is not None:
            train_data["rollout_routed_experts"] = [sample.rollout_routed_experts for sample in samples]

        if samples[0].train_metadata is not None:
            train_data["metadata"] = [sample.train_metadata for sample in samples]

        has_any_mm = any(s.multimodal_train_inputs is not None for s in samples)
        if has_any_mm:
            missing_mm_indices = []
            for i, sample in enumerate(samples):
                if sample.multimodal_train_inputs is None and not sample.remove_sample:
                    sample.remove_sample = True
                    sample.loss_mask = [0] * sample.response_length
                    loss_masks[i] = sample.loss_mask
                    missing_mm_indices.append(sample.index)
            if missing_mm_indices:
                logger.warning(
                    "Dropped %d samples with missing multimodal_train_inputs: indices=%s",
                    len(missing_mm_indices),
                    missing_mm_indices[:20],
                )
            train_data["multimodal_train_inputs"] = [sample.multimodal_train_inputs for sample in samples]

        if "teacher_log_probs" in samples[0].__dict__:
            train_data["teacher_log_probs"] = [sample.teacher_log_probs for sample in samples]

        if "teacher_topk_log_probs" in samples[0].__dict__:
            train_data["teacher_topk_log_probs"] = [sample.teacher_topk_log_probs for sample in samples]

        if "teacher_topk_indices" in samples[0].__dict__:
            train_data["teacher_topk_indices"] = [sample.teacher_topk_indices for sample in samples]

        return train_data

    def set_train_parallel_config(self, config: dict):
        logger.info("[RolloutManager.set_train_parallel_config] enter config=%s", config)
        self.train_parallel_config = config
        logger.info("[RolloutManager.set_train_parallel_config] exit")

    def _split_train_data_by_dp(self, data, dp_size):
        """Split the train data by data parallel size."""
        rollout_data = {}

        if "prompt" in data:
            rollout_data["prompt"] = data["prompt"]

        total_lengths = [len(t) for t in data["tokens"]]
        data["total_lengths"] = total_lengths

        if self.args.balance_data:
            # Equal-size partitioning requires divisibility by dp_size.
            # Dynamic rollout/history can produce tail batches that violate this.
            use_equal_size = (len(total_lengths) % dp_size) == 0
            if not use_equal_size:
                logger.warning(
                    "balance-data fallback: num_samples=%d is not divisible by dp_size=%d; "
                    "using unequal-size seqlen balancing for this rollout step.",
                    len(total_lengths),
                    dp_size,
                )
            partitions = get_seqlen_balanced_partitions(total_lengths, dp_size, equal_size=use_equal_size)
        else:
            partitions = [range(i, len(total_lengths), dp_size) for i in range(dp_size)]

        rollout_data_refs = []

        for i in range(dp_size):
            rollout_data = {}
            partition = partitions[i]
            rollout_data["partition"] = partition
            for key in [
                "tokens",
                "multimodal_train_inputs",
                "response_lengths",
                "rewards",
                "truncated",
                "loss_masks",
                "round_number",
                "sample_indices",
                "rollout_log_probs",
                "rollout_routed_experts",
                "prompt",
                "teacher_log_probs",
                "teacher_topk_log_probs",
                "teacher_topk_indices",
                "step_wise_step_rewards",
                "step_wise_step_token_spans",
                "step_wise_step_indices",
                "group_indices",
            ]:
                if key not in data:
                    continue
                val = [data[key][j] for j in partition]
                rollout_data[key] = val
            # keys that need to be splited at train side
            for key in [
                "raw_reward",
                "total_lengths",
            ]:
                if key not in data:
                    continue
                rollout_data[key] = data[key]
            # Pass dynamic global_batch_size to training side
            if hasattr(self, "_dynamic_global_batch_size"):
                rollout_data["dynamic_global_batch_size"] = self._dynamic_global_batch_size
            rollout_data_refs.append(Box(ray.put(rollout_data)))
        return rollout_data_refs


def init_rollout_engines(args, pg, all_rollout_engines):
    if args.debug_train_only:
        return 0

    num_gpu_per_engine = min(args.rollout_num_gpus_per_engine, args.num_gpus_per_node)
    num_engines = args.rollout_num_gpus // num_gpu_per_engine
    assert len(all_rollout_engines) == num_engines
    if args.prefill_num_servers is not None:
        prefill_num_servers = args.prefill_num_servers * args.rollout_num_gpus_per_engine // num_gpu_per_engine
        assert (
            num_engines > prefill_num_servers
        ), f"num_engines {num_engines} should be larger than prefill_num_servers {prefill_num_servers}"

    pg, reordered_bundle_indices, reordered_gpu_ids = pg

    RolloutRayActor = ray.remote(SGLangEngine)

    rollout_engines = []
    for i in range(num_engines):
        if all_rollout_engines[i] is not None:
            continue

        num_gpus = 0.2
        num_cpus = num_gpus

        # Get the base GPU ID from placement group
        base_gpu_id = int(reordered_gpu_ids[i * num_gpu_per_engine])

        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=reordered_bundle_indices[i * num_gpu_per_engine],
        )

        env_vars = {name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST} | {
            key: os.environ.get(key, default_val)
            for key, default_val in {
                "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "false",
                "SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK": "true",
                "SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK": "true",
                "SGLANG_MEMORY_SAVER_CUDA_GRAPH": "true",
                "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT": "true",
                "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "false",
                "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE": "false",
            }.items()
        }
        if getattr(args, "sglang_shadow_worker_kv_cache_socket_path", None):
            env_vars["SGLANG_KV_CACHE_SOCKET_PATH"] = args.sglang_shadow_worker_kv_cache_socket_path
        elif os.environ.get("SGLANG_KV_CACHE_SOCKET_PATH"):
            env_vars["SGLANG_KV_CACHE_SOCKET_PATH"] = os.environ["SGLANG_KV_CACHE_SOCKET_PATH"]
        if getattr(args, "sglang_shadow_worker_weight_server_base_port", None) is not None:
            env_vars["WEIGHT_SERVER_BASE_PORT"] = str(args.sglang_shadow_worker_weight_server_base_port)
        elif os.environ.get("WEIGHT_SERVER_BASE_PORT"):
            env_vars["WEIGHT_SERVER_BASE_PORT"] = os.environ["WEIGHT_SERVER_BASE_PORT"]
        if getattr(args, "sglang_shadow_worker_min_gpu_id", None) is not None:
            env_vars["SGLANG_MIN_GPU_ID"] = str(args.sglang_shadow_worker_min_gpu_id)
        elif os.environ.get("SGLANG_MIN_GPU_ID"):
            env_vars["SGLANG_MIN_GPU_ID"] = os.environ["SGLANG_MIN_GPU_ID"]

        worker_type = "regular"
        if args.prefill_num_servers is not None:
            if i < prefill_num_servers:
                worker_type = "prefill"
            else:
                worker_type = "decode"

        rollout_engine = RolloutRayActor.options(
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            scheduling_strategy=scheduling_strategy,
            runtime_env={
                "env_vars": env_vars,
            },
        ).remote(args, rank=i, worker_type=worker_type, base_gpu_id=base_gpu_id)

        rollout_engines.append((i, rollout_engine))
        all_rollout_engines[i] = rollout_engine

    num_new_engines = len(rollout_engines)

    if num_new_engines == 0:
        return num_new_engines

    if args.rollout_external:
        addr_and_ports = _allocate_rollout_engine_addr_and_ports_external(args=args, rollout_engines=rollout_engines)
    else:
        addr_and_ports = _allocate_rollout_engine_addr_and_ports_normal(
            args=args, num_engines=num_engines, rollout_engines=rollout_engines
        )

    # TODO: don't ray.get here to overlap train actor init with rollout engine init.
    # somehow if we don't sync here, the --debug-rollout-only mode will crash.
    init_handles = [engine.init.remote(**(addr_and_ports[rank])) for rank, engine in rollout_engines]
    init_wait_start_ts = time.monotonic()
    logger.info(
        "Waiting for rollout engine init handles: count=%s, ranks=%s",
        len(init_handles),
        [rank for rank, _ in rollout_engines],
    )
    ray.get(init_handles)
    logger.info(
        "Rollout engine init handles completed: count=%s, elapsed=%.1fs",
        len(init_handles),
        time.monotonic() - init_wait_start_ts,
    )

    return num_new_engines


def init_prm_engines(args, pg, all_prm_engines):
    if not args.prm_enable or args.prm_num_gpus <= 0:
        return 0
    assert pg is not None, "PRM placement group is required when PRM is enabled."

    num_gpu_per_engine = min(args.prm_num_gpus_per_engine, args.num_gpus_per_node)
    num_engines = args.prm_num_gpus // num_gpu_per_engine
    assert len(all_prm_engines) == num_engines

    pg, reordered_bundle_indices, reordered_gpu_ids = pg
    RolloutRayActor = ray.remote(SGLangEngine)

    prm_engines = []
    for i in range(num_engines):
        if all_prm_engines[i] is not None:
            continue

        num_gpus = 0.2
        num_cpus = num_gpus
        base_gpu_id = int(reordered_gpu_ids[i * num_gpu_per_engine])
        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=reordered_bundle_indices[i * num_gpu_per_engine],
        )

        env_vars = {name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST} | {
            key: os.environ.get(key, default_val)
            for key, default_val in {
                "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "false",
                "SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK": "true",
                "SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK": "true",
                "SGLANG_MEMORY_SAVER_CUDA_GRAPH": "true",
                "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT": "true",
                "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "false",
                "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE": "false",
            }.items()
        }

        prm_engine = RolloutRayActor.options(
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            scheduling_strategy=scheduling_strategy,
            runtime_env={"env_vars": env_vars},
        ).remote(args, rank=i, worker_type="regular", base_gpu_id=base_gpu_id, engine_role="prm")

        prm_engines.append((i, prm_engine))
        all_prm_engines[i] = prm_engine

    num_new_engines = len(prm_engines)
    if num_new_engines == 0:
        return num_new_engines

    addr_and_ports = _allocate_prm_engine_addr_and_ports(
        args=args,
        num_engines=num_engines,
        prm_engines=prm_engines,
    )
    init_handles = [engine.init.remote(**(addr_and_ports[rank])) for rank, engine in prm_engines]
    ray.get(init_handles)
    return num_new_engines


def _allocate_rollout_engine_addr_and_ports_external(args, rollout_engines):
    addr_and_ports = []
    for rank, _ in rollout_engines:
        addr = args.rollout_external_engine_addrs[rank]
        [host, port] = addr.split(":")
        addr_and_ports.append(
            dict(
                dist_init_addr=addr,
                nccl_port=None,
                host=host,
                port=int(port),
            )
        )
    return addr_and_ports


def _allocate_prm_engine_addr_and_ports(*, args, num_engines, prm_engines):
    # mirror rollout allocator but use PRM engine parallel settings.
    num_engines_per_node = max(1, min(args.num_gpus_per_node, args.prm_num_gpus) // args.prm_num_gpus_per_engine)
    addr_and_ports = [{} for _ in range(num_engines)]

    visited_nodes = set()
    for rank, engine in prm_engines:
        if rank // num_engines_per_node in visited_nodes:
            continue
        visited_nodes.add(rank // num_engines_per_node)
        num_engines_on_this_node = num_engines_per_node - (rank % num_engines_per_node)

        def get_addr_and_ports(engine):
            start_port = 25000

            def port(consecutive=1):
                nonlocal start_port
                _, port = ray.get(
                    engine._get_current_node_ip_and_free_port.remote(
                        start_port=start_port,
                        consecutive=consecutive,
                    )
                )
                start_port = port + consecutive
                return port

            def addr():
                addr, _ = ray.get(engine._get_current_node_ip_and_free_port.remote())
                return addr

            return addr, port

        get_addr, get_port = get_addr_and_ports(engine)
        for i in range(num_engines_on_this_node):
            current_rank = rank + i
            addr_and_ports[current_rank]["host"] = get_addr()
            addr_and_ports[current_rank]["port"] = get_port()
            addr_and_ports[current_rank]["nccl_port"] = get_port()

        if args.prm_num_gpus_per_engine > args.num_gpus_per_node:
            num_node_per_engine = args.prm_num_gpus_per_engine // args.num_gpus_per_node
            if rank % num_node_per_engine == 0:
                dist_init_addr = f"{get_addr()}:{get_port(30 + args.sglang_dp_size)}"
                for i in range(num_node_per_engine):
                    addr_and_ports[rank + i]["dist_init_addr"] = dist_init_addr
        else:
            for i in range(num_engines_on_this_node):
                addr_and_ports[rank + i]["dist_init_addr"] = f"{get_addr()}:{get_port(30 + args.sglang_dp_size)}"

    for i, _ in prm_engines:
        for key in ["port", "nccl_port", "dist_init_addr"]:
            assert key in addr_and_ports[i], f"PRM Engine {i} {key} is not set."
        logger.info(f"Ports for PRM engine {i}: {addr_and_ports[i]}")
    return addr_and_ports


def _allocate_rollout_engine_addr_and_ports_normal(*, args, num_engines, rollout_engines):
    # get ports
    # there are 4 ports we need to allocate
    # 1. server port
    # 2. nccl port
    # 3. dist_init_addr port
    # 4. other ports for dp_attention, which is of size 4 + dp_size
    num_engines_per_node = max(
        1, min(args.num_gpus_per_node, args.rollout_num_gpus) // args.rollout_num_gpus_per_engine
    )
    addr_and_ports = [{} for _ in range(num_engines)]

    # Calculate prefill limit to identify prefill engines
    prefill_limit = 0
    if args.prefill_num_servers is not None:
        num_gpu_per_engine = min(args.rollout_num_gpus_per_engine, args.num_gpus_per_node)
        prefill_limit = args.prefill_num_servers * args.rollout_num_gpus_per_engine // num_gpu_per_engine

    visited_nodes = set()
    for rank, engine in rollout_engines:
        if rank // num_engines_per_node in visited_nodes:
            continue
        visited_nodes.add(rank // num_engines_per_node)
        # TODO: currently when restarting engines, we will set port for all engines on this node starting with this rank.
        # e.g. for 8 gpus, if we are restarting engine on gpu 3, we will set port for engine 3,4,5,6,7 on this node.
        num_engines_on_this_node = num_engines_per_node - (rank % num_engines_per_node)

        def get_addr_and_ports(engine):
            # use small ports to prevent ephemeral port between 32768 and 65536.
            # also, ray uses port 10002-19999, thus we avoid near-10002 to avoid racing condition
            start_port = 15000

            def port(consecutive=1):
                nonlocal start_port
                _, port = ray.get(
                    engine._get_current_node_ip_and_free_port.remote(
                        start_port=start_port,
                        consecutive=consecutive,
                    )
                )
                start_port = port + consecutive
                return port

            def addr():
                addr, _ = ray.get(engine._get_current_node_ip_and_free_port.remote())
                return addr

            return addr, port

        get_addr, get_port = get_addr_and_ports(engine)

        for i in range(num_engines_on_this_node):
            current_rank = rank + i
            addr_and_ports[current_rank]["host"] = get_addr()
            addr_and_ports[current_rank]["port"] = get_port()
            addr_and_ports[current_rank]["nccl_port"] = get_port()

            if args.prefill_num_servers is not None and current_rank < prefill_limit:
                addr_and_ports[current_rank]["disaggregation_bootstrap_port"] = get_port()

        if args.rollout_num_gpus_per_engine > args.num_gpus_per_node:
            num_node_per_engine = args.rollout_num_gpus_per_engine // args.num_gpus_per_node
            if rank % num_node_per_engine == 0:
                # this is the first node in the engine, we need to allocate the dist_init_addr port
                dist_init_addr = f"{get_addr()}:{get_port(30 + args.sglang_dp_size)}"
                for i in range(num_node_per_engine):
                    addr_and_ports[rank + i]["dist_init_addr"] = dist_init_addr
        else:
            for i in range(num_engines_on_this_node):
                addr_and_ports[rank + i]["dist_init_addr"] = f"{get_addr()}:{get_port(30 + args.sglang_dp_size)}"

    for i, _ in rollout_engines:
        for key in ["port", "nccl_port", "dist_init_addr"]:
            assert key in addr_and_ports[i], f"Engine {i} {key} is not set."
        logger.info(f"Ports for engine {i}: {addr_and_ports[i]}")

    return addr_and_ports


def _start_router(args, router_ip_attr: str, router_port_attr: str):
    """Start a router for rollout or PRM engines."""
    if getattr(args, router_ip_attr, None) is not None:
        return

    setattr(args, router_ip_attr, _wrap_ipv6(get_host_info()[1]))
    if getattr(args, router_port_attr, None) is None:
        setattr(args, router_port_attr, find_available_port(random.randint(3000, 4000)))

    if args.use_slime_router:
        if router_ip_attr == "sglang_router_ip":
            assert args.prefill_num_servers is None, "slime router does not support prefill_num_servers."
        from slime.router.router import run_router

        router_args = copy(args)
        router_args.sglang_router_ip = getattr(args, router_ip_attr)
        router_args.sglang_router_port = getattr(args, router_port_attr)

    else:
        from sglang_router.launch_router import RouterArgs

        from slime.utils.http_utils import run_router

        router_args = RouterArgs.from_cli_args(args, use_router_prefix=True)
        router_args.host = getattr(args, router_ip_attr)
        router_args.port = getattr(args, router_port_attr)
        router_args.prometheus_port = find_available_port(random.randint(4000, 5000))
        router_args.log_level = "warn"
        router_args.request_timeout_secs = args.sglang_router_request_timeout_secs

        if router_ip_attr == "sglang_router_ip" and args.prefill_num_servers is not None:
            router_args.pd_disaggregation = True

        logger.info(f"Launch router with args: {router_args}")

    process = multiprocessing.Process(
        target=run_router,
        args=(router_args,),
    )
    process.daemon = True  # Set the process as a daemon
    process.start()
    # Wait 3 seconds
    time.sleep(3)
    assert process.is_alive()
    logger.info(f"Router launched at {getattr(args, router_ip_attr)}:{getattr(args, router_port_attr)}")


def _log_eval_rollout_data(rollout_id, args, data, extra_metrics: dict[str, Any] | None = None):
    if args.custom_eval_rollout_log_function_path is not None:
        custom_log_func = load_function(args.custom_eval_rollout_log_function_path)
        if custom_log_func(rollout_id, args, data, extra_metrics):
            return

    log_dict = extra_metrics or {}
    for key in data.keys():
        rewards = data[key]["rewards"]
        log_dict[f"eval/{key}"] = sum(rewards) / len(rewards)
        if (samples := data[key].get("samples")) is not None:
            sample_metrics = compute_metrics_from_samples(args, samples)
            sample_metrics = {k: v for k, v in sample_metrics.items() if not k.startswith("zero_std/")}
            log_dict |= dict_add_prefix(sample_metrics, f"eval/{key}/")
            aborted = [s for s in samples if s.status == Sample.Status.ABORTED]
            log_dict[f"eval/{key}-aborted_ratio"] = len(aborted) / len(samples)
        if "truncated" in data[key]:
            truncated = data[key]["truncated"]
            log_dict[f"eval/{key}-truncated_ratio"] = sum(truncated) / len(truncated)
        if args.log_passrate:
            log_dict |= dict_add_prefix(
                compute_pass_rate(
                    flat_rewards=rewards,
                    group_size=args.n_samples_per_eval_prompt,
                ),
                f"eval/{key}-",
            )

    logger.info(f"eval {rollout_id}: {log_dict}")

    step = compute_rollout_step(args, rollout_id)
    log_dict["eval/step"] = step
    logging_utils.log(args, log_dict, step_key="eval/step")

    return log_dict


def _log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    if args.custom_rollout_log_function_path is not None:
        custom_log_func = load_function(args.custom_rollout_log_function_path)
        if custom_log_func(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
            return

    if args.load_debug_rollout_data:
        return

    log_dict = {**(rollout_extra_metrics or {})}
    log_dict |= dict_add_prefix(compute_metrics_from_samples(args, samples), "rollout/")
    log_dict |= dict_add_prefix(compute_perf_metrics_from_samples(args, samples, rollout_time), "perf/")
    logger.info(f"perf {rollout_id}: {log_dict}")
    step = compute_rollout_step(args, rollout_id)
    log_dict["rollout/step"] = step
    logging_utils.log(args, log_dict, step_key="rollout/step")


def compute_metrics_from_samples(args, samples):
    response_lengths = [sample.effective_response_length for sample in samples]

    log_dict = {}
    log_dict |= dict_add_prefix(compute_statistics(response_lengths), "response_len/")
    log_dict |= _compute_zero_std_metrics(args, samples)
    log_dict |= _compute_reward_cat_metrics(args, samples)
    log_dict["repetition_frac"] = np.mean([int(has_repetition(s.response)) for s in samples]).item()
    log_dict["truncated_ratio"] = np.mean([int(s.status == Sample.Status.TRUNCATED) for s in samples]).item()
    return log_dict


def compute_perf_metrics_from_samples(args, samples, rollout_time):
    non_generation_time = [sample.non_generation_time for sample in samples]

    log_dict = {}
    log_dict["rollout_time"] = rollout_time
    if max(non_generation_time) > 0:
        log_dict |= dict_add_prefix(compute_statistics(non_generation_time), "non_generation_time/")

    def token_perf(response_lengths, non_generation_time, key=""):
        max_response_length = max(response_lengths)
        if args.rollout_num_gpus:
            log_dict[f"{key}tokens_per_gpu_per_sec"] = sum(response_lengths) / rollout_time / args.rollout_num_gpus
        log_dict[f"longest_{key}sample_tokens_per_sec"] = max_response_length / rollout_time

        if max(non_generation_time) == 0:
            return

        non_generation_time = [
            t for t, length in zip(non_generation_time, response_lengths, strict=True) if length == max_response_length
        ]
        mean_non_generation_time = sum(non_generation_time) / len(non_generation_time)

        log_dict[f"longest_{key}sample_non_generation_time"] = mean_non_generation_time
        log_dict[f"longest_{key}sample_tokens_per_sec_without_non_generation"] = max_response_length / (
            rollout_time - mean_non_generation_time
        )

    token_perf([sample.response_length for sample in samples], non_generation_time, key="")
    token_perf([sample.effective_response_length for sample in samples], non_generation_time, key="effective_")

    return log_dict


def _compute_zero_std_metrics(args, all_samples: list[Sample]):
    # only compute in GRPO-like algorithms where one prompt has multiple responses
    if args.advantage_estimator == "ppo":
        return {}

    all_samples = [sample for sample in all_samples if sample.reward is not None]
    if not all_samples:
        return {}

    def _is_zero_std(samples: list[Sample]):
        rewards = [sample.get_reward_value(args) for sample in samples]
        return len(rewards) == 0 or all(rewards[0] == r for r in rewards)

    all_sample_groups = group_by(all_samples, lambda s: s.group_index)
    interesting_sample_groups = [g for g in all_sample_groups.values() if _is_zero_std(g)]

    interesting_rewards = [str(round(g[0].get_reward_value(args), 1)) for g in interesting_sample_groups]

    return {f"zero_std/count_{reward}": len(items) for reward, items in group_by(interesting_rewards).items()}


def _compute_spec_metrics(args, all_samples: list[Sample]):
    if args.sglang_speculative_algorithm is None:
        return {}
    num_samples = len(all_samples)
    metrics = {}
    metrics["spec_accept_rate"] = sum(sample.spec_info.spec_accept_rate for sample in all_samples) / num_samples
    metrics["spec_accept_length"] = sum(sample.spec_info.spec_accept_length for sample in all_samples) / num_samples
    return metrics


def _compute_prefix_cache_metrics(args, all_samples: list[Sample]):
    num_samples = len(all_samples)
    metrics = {}
    total_cached_tokens = sum(sample.prefix_cache_info.cached_tokens for sample in all_samples)
    total_prompt_tokens = sum(sample.prefix_cache_info.total_prompt_tokens for sample in all_samples)

    metrics["prefix_cache_hit_rate"] = total_cached_tokens / total_prompt_tokens if total_prompt_tokens > 0 else 0.0
    metrics["avg_cached_tokens_per_sample"] = total_cached_tokens / num_samples
    return metrics


def _compute_reward_cat_metrics(args, all_samples: list[Sample]):
    reward_cat_key = args.log_reward_category
    if reward_cat_key is None:
        return {}

    samples_of_reward_cat = group_by(all_samples, lambda s: s.reward[reward_cat_key])

    return {f"error_cat/{reward_cat}": len(s) / len(all_samples) for reward_cat, s in samples_of_reward_cat.items()}
