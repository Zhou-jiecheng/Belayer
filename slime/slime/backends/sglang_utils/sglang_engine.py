import dataclasses
import ipaddress
import logging
import multiprocessing
import os
import threading
import time
from urllib.parse import quote

import requests
import sglang_router
from packaging.version import parse
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import kill_process_tree
from urllib3.exceptions import NewConnectionError

from slime.ray.ray_actor import RayActor
from slime.utils.http_utils import get_host_info

logger = logging.getLogger(__name__)

_SHADOW_PORT_OFFSET = 110
_SHADOW_NCCL_PORT_OFFSET = 120
_SHADOW_DIST_INIT_PORT_OFFSET = 130
_SHADOW_SUPPORTED_HEALTH_CODES = {200, 202}
_SERVER_ARG_NAMES = {field.name for field in dataclasses.fields(ServerArgs)}
_REMOTE_INSTANCE_RECOVERY_ARG_NAMES = {
    "remote_instance_weight_loader_seed_instance_ip",
    "remote_instance_weight_loader_seed_instance_service_port",
    "remote_instance_weight_loader_send_weights_group_ports",
    "remote_instance_weight_loader_backend",
}


def _is_not_found(response: requests.Response | None) -> bool:
    return response is not None and response.status_code == 404


def _filter_server_args_for_current_sglang(server_args_dict: dict, *, context: str) -> dict:
    filtered = dict(server_args_dict)
    unsupported = sorted(set(filtered) - _SERVER_ARG_NAMES)
    if unsupported:
        logger.warning(
            "Dropping unsupported ServerArgs keys for %s because current sglang does not accept them: %s",
            context,
            unsupported,
        )
        for key in unsupported:
            filtered.pop(key, None)
    return filtered


def get_base_gpu_id(args, rank):
    num_gpus = min(args.num_gpus_per_node, args.rollout_num_gpus_per_engine)
    if args.colocate:
        start_index = (rank * num_gpus) % args.num_gpus_per_node
    else:
        num_actor_gpus = 0 if args.debug_rollout_only else args.actor_num_gpus_per_node * args.actor_num_nodes
        start_index = (num_actor_gpus + rank * num_gpus) % args.num_gpus_per_node
        if args.use_critic:
            num_critic_gpus = args.critic_num_gpus_per_node * args.critic_num_nodes
            start_index = (num_actor_gpus + num_critic_gpus + rank * num_gpus) % args.num_gpus_per_node
    return start_index


def _get_weight_server_base_port(args) -> int | None:
    explicit = getattr(args, "sglang_shadow_worker_weight_server_base_port", None)
    if explicit is not None:
        return explicit
    value = os.environ.get("WEIGHT_SERVER_BASE_PORT")
    return int(value) if value is not None else None


def _get_weight_server_min_gpu_id(args) -> int:
    explicit = getattr(args, "sglang_shadow_worker_min_gpu_id", None)
    if explicit is not None:
        return explicit
    value = os.environ.get("SGLANG_MIN_GPU_ID")
    return int(value) if value is not None else 0


def _compute_weight_load_port(args, physical_base_gpu_id: int | None, engine_role: str) -> int | None:
    if physical_base_gpu_id is None:
        return None

    base_port = _get_weight_server_base_port(args)
    if base_port is None:
        return None

    gpus_per_engine = args.prm_num_gpus_per_engine if engine_role == "prm" else args.rollout_num_gpus_per_engine
    return base_port + (physical_base_gpu_id - _get_weight_server_min_gpu_id(args)) // gpus_per_engine


def _get_recovery_remote_instance_override(args, rank: int) -> dict | None:
    overrides = getattr(args, "_sglang_recovery_remote_instance_overrides", None)
    if not overrides:
        return None
    override = overrides.get(rank)
    if not isinstance(override, dict):
        return None
    required_keys = {
        "seed_instance_ip",
        "seed_instance_service_port",
        "send_weights_group_ports",
    }
    if not required_keys.issubset(override.keys()):
        logger.warning(
            "Ignore incomplete remote-instance recovery override for rank=%s: keys=%s",
            rank,
            sorted(override.keys()),
        )
        return None
    return override


def _is_remote_instance_recovery_launch(server_args_dict: dict) -> bool:
    return (
        server_args_dict.get("load_format") == "remote_instance"
        and "remote_instance_weight_loader_seed_instance_ip" in server_args_dict
    )


def _get_remote_instance_seed_health_timeout_sec() -> float:
    return max(0.1, float(os.getenv("SLIME_REMOTE_INSTANCE_RECOVERY_SEED_HEALTH_TIMEOUT_SEC", "2")))


def _is_remote_instance_seed_service_healthy(server_args_dict: dict) -> bool:
    seed_ip = server_args_dict.get("remote_instance_weight_loader_seed_instance_ip")
    seed_port = server_args_dict.get("remote_instance_weight_loader_seed_instance_service_port")
    if not seed_ip or seed_port is None:
        return False

    timeout = _get_remote_instance_seed_health_timeout_sec()
    try:
        response = requests.get(f"http://{seed_ip}:{seed_port}/health_generate", timeout=timeout)
        response.raise_for_status()
        return True
    except requests.RequestException as exc:
        logger.warning(
            "Remote-instance recovery seed health probe failed for %s:%s (timeout=%.1fs): %s",
            seed_ip,
            seed_port,
            timeout,
            exc,
        )
        return False


def _build_storage_fallback_server_args(server_args_dict: dict) -> dict:
    fallback = dict(server_args_dict)
    for key in _REMOTE_INSTANCE_RECOVERY_ARG_NAMES:
        fallback.pop(key, None)
    fallback.pop("weight_load_port", None)
    fallback["load_format"] = "auto"
    return fallback


def _to_local_gpu_id(physical_gpu_id: int) -> int:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not cvd:
        return physical_gpu_id  # no remapping
    # CUDA_VISIBLE_DEVICES can be like "4,5,6,7"
    visible = [int(x) for x in cvd.split(",") if x.strip() != ""]
    # In a remapped process, valid torch device indices are 0..len(visible)-1
    if physical_gpu_id in visible:
        return visible.index(physical_gpu_id)
    # If we're already getting local IDs, allow them
    if 0 <= physical_gpu_id < len(visible):
        return physical_gpu_id
    raise RuntimeError(
        f"GPU id {physical_gpu_id} is not valid under CUDA_VISIBLE_DEVICES={cvd}. "
        f"Expected one of {visible} (physical) or 0..{len(visible)-1} (local)."
    )


def launch_server_process(server_args: ServerArgs, *, wait_healthy: bool = True) -> multiprocessing.Process:
    from sglang.srt.entrypoints.http_server import launch_server

    multiprocessing.set_start_method("spawn", force=True)
    server_args.host = server_args.host.strip("[]")
    p = multiprocessing.Process(target=launch_server, args=(server_args,))
    p.start()

    if server_args.node_rank != 0 or not wait_healthy:
        return p

    try:
        _wait_server_healthy(
            base_url=server_args.url(),
            api_key=server_args.api_key,
            is_process_alive=lambda: p.is_alive(),
        )
    except Exception:
        if p.is_alive():
            kill_process_tree(p.pid)
        raise

    return p


def _wait_server_healthy(base_url, api_key, is_process_alive, *, wait_for_flush_cache: bool = True):
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Authorization": f"Bearer {api_key}",
    }

    start_time = time.monotonic()
    health_attempt = 0
    flush_attempt = 0

    with requests.Session() as session:
        while True:
            health_attempt += 1
            status_code = None
            try:
                response = session.get(f"{base_url}/health_generate", headers=headers)
                status_code = response.status_code
                if response.status_code == 200:
                    logger.info(
                        "Server health_generate ready at %s after %.1fs (attempt=%s)",
                        base_url,
                        time.monotonic() - start_time,
                        health_attempt,
                    )
                    break
            except requests.RequestException:
                pass

            if not is_process_alive():
                raise Exception("Server process terminated unexpectedly.")

            if health_attempt == 1 or health_attempt % 10 == 0:
                logger.info(
                    "Waiting for health_generate at %s (elapsed=%.1fs, attempt=%s, last_status=%s)",
                    base_url,
                    time.monotonic() - start_time,
                    health_attempt,
                    status_code,
                )

            time.sleep(2)

        if not wait_for_flush_cache:
            logger.info("Skip flush_cache readiness wait at %s", base_url)
            return

        # use flush_cache to make sure the working queue is empty, so that we can do offload
        while True:
            flush_attempt += 1
            status_code = None
            try:
                response = session.get(f"{base_url}/flush_cache", headers=headers)
                status_code = response.status_code
                if response.status_code == 200:
                    logger.info(
                        "Server flush_cache ready at %s after %.1fs (attempt=%s)",
                        base_url,
                        time.monotonic() - start_time,
                        flush_attempt,
                    )
                    break

            except requests.RequestException:
                pass

            if not is_process_alive():
                raise Exception("Server process terminated unexpectedly.")

            if flush_attempt == 1 or flush_attempt % 10 == 0:
                logger.info(
                    "Waiting for flush_cache at %s (elapsed=%.1fs, attempt=%s, last_status=%s)",
                    base_url,
                    time.monotonic() - start_time,
                    flush_attempt,
                    status_code,
                )

            time.sleep(2)


class SGLangEngine(RayActor):
    shadow_worker: multiprocessing.Process | None = None

    def __init__(
        self,
        args,
        rank: int,
        worker_type: str = "regular",
        base_gpu_id: int | None = None,
        engine_role: str = "rollout",
    ):
        self.args = args
        self.rank = rank
        self.worker_type = worker_type
        self.base_gpu_id = base_gpu_id
        self.engine_role = engine_role
        self.server_args = None
        self.shadow_worker = None
        self.shadow_server_args = None
        self.shadow_worker_enabled = False
        self._shadow_disable_reason = None
        self._shadow_handover_lock = threading.Lock()
        self._shadow_failover_stop_event = threading.Event()
        self._shadow_failover_thread = None
        self._pending_shadow_handover_reconnect = False
        self._pending_shadow_handover_reason = None

    def init(self, dist_init_addr, port, nccl_port, host=None, disaggregation_bootstrap_port=None):
        if self.engine_role == "prm":
            self.router_ip = self.args.prm_router_ip
            self.router_port = self.args.prm_router_port
        else:
            self.router_ip = self.args.sglang_router_ip
            self.router_port = self.args.sglang_router_port

        host = host or get_host_info()[1]

        def _format_v6_uri(addr):
            if not addr or addr.startswith("["):
                return addr
            try:
                if ipaddress.ip_address(addr).version == 6:
                    return f"[{addr}]"
            except ValueError:
                pass
            return addr

        host = _format_v6_uri(host)
        ip_part, port_part = dist_init_addr.rsplit(":", 1)
        dist_init_addr = f"{_format_v6_uri(ip_part)}:{port_part}"

        server_args_dict, external_engine_need_check_fields = _compute_server_args(
            self.args,
            self.rank,
            dist_init_addr,
            nccl_port,
            host,
            port,
            self.worker_type,
            disaggregation_bootstrap_port,
            base_gpu_id=self.base_gpu_id,
            engine_role=self.engine_role,
        )

        self.node_rank = server_args_dict["node_rank"]
        self.server_host = server_args_dict["host"]  # with [] if ipv6
        self.server_port = server_args_dict["port"]
        self.disaggregation_bootstrap_port = server_args_dict.get("disaggregation_bootstrap_port")

        if self.args.rollout_external:
            self._init_external(server_args_dict, external_engine_need_check_fields=external_engine_need_check_fields)
        else:
            self._init_normal(server_args_dict)

    def _init_external(self, expect_server_args, external_engine_need_check_fields):
        logger.info(f"Use external SGLang engine (rank={self.rank}, expect_server_args={expect_server_args})")

        def _get_actual_server_args():
            response = requests.get(f"http://{self.server_host}:{self.server_port}/get_server_info")
            response.raise_for_status()
            return response.json()

        def _sanity_check_server_args(actual_server_args, expect_server_args):
            for name in external_engine_need_check_fields:
                expect_value = expect_server_args.get(name)
                actual_value = actual_server_args.get(name)
                assert (
                    actual_value == expect_value
                ), f"{name=} {expect_value=} {actual_value=} {expect_server_args=} {actual_server_args=}"

        _wait_server_healthy(
            base_url=f"http://{self.server_host}:{self.server_port}",
            api_key=None,
            is_process_alive=lambda: True,
        )
        actual_server_args = _get_actual_server_args()
        _sanity_check_server_args(actual_server_args, expect_server_args)

    def _init_normal(self, server_args_dict):
        logger.info(f"Launch HttpServerEngineAdapter at: {self.server_host}:{self.server_port}")
        filtered_server_args = _filter_server_args_for_current_sglang(
            server_args_dict,
            context=f"engine rank={self.rank} role={self.engine_role}",
        )
        launch_context = f"engine rank={self.rank} role={self.engine_role}"
        try:
            if _is_remote_instance_recovery_launch(filtered_server_args) and not _is_remote_instance_seed_service_healthy(
                filtered_server_args
            ):
                raise RuntimeError("remote-instance recovery seed is not healthy")
            self.server_args = ServerArgs(**filtered_server_args)
            self.process = launch_server_process(self.server_args)
        except Exception as exc:
            if not _is_remote_instance_recovery_launch(filtered_server_args):
                raise

            logger.warning(
                "Remote-instance recovery launch failed for %s at %s:%s; falling back to storage load. error=%s",
                launch_context,
                self.server_host,
                self.server_port,
                exc,
            )
            process = getattr(self, "process", None)
            if process is not None and getattr(process, "pid", None) is not None and process.is_alive():
                kill_process_tree(process.pid)
            self.process = None

            fallback_server_args = _filter_server_args_for_current_sglang(
                _build_storage_fallback_server_args(server_args_dict),
                context=f"{launch_context} storage-fallback",
            )
            self.server_args = ServerArgs(**fallback_server_args)
            self.process = launch_server_process(self.server_args)

        self._register_worker(self._worker_url(self.server_host, self.server_port))
        self._init_shadow_worker()

    def _worker_url(self, host: str, port: int) -> str:
        return f"http://{host}:{port}"

    def get_remote_instance_weight_loader_seed_info(self) -> dict:
        if getattr(self, "server_args", None) is None:
            raise RuntimeError(f"Engine rank={self.rank} is not initialized yet")
        if self.node_rank != 0:
            raise RuntimeError(
                f"Only node_rank=0 can serve as remote-instance seed (rank={self.rank}, node_rank={self.node_rank})"
            )
        if int(getattr(self.server_args, "nnodes", 1)) != 1:
            raise RuntimeError(
                "Remote-instance recovery is only supported for single-node rollout engines in the current integration"
            )

        tp_size = max(1, int(getattr(self.server_args, "tp_size", 1)))
        seed_ip, start_port = self._get_current_node_ip_and_free_port(
            start_port=max(int(self.server_port) + 1000, 20000),
            consecutive=tp_size,
        )
        seed_info = {
            "seed_instance_ip": seed_ip,
            "seed_instance_service_port": int(self.server_port),
            "send_weights_group_ports": [start_port + offset for offset in range(tp_size)],
            "backend": "nccl",
        }
        logger.info(
            "Prepared remote-instance seed info for engine rank=%s: seed=%s:%s tp_size=%s ports=%s",
            self.rank,
            seed_info["seed_instance_ip"],
            seed_info["seed_instance_service_port"],
            tp_size,
            seed_info["send_weights_group_ports"],
        )
        return seed_info

    def _worker_key(self) -> str:
        return f"{self.worker_type}:rank={self.rank}:node_rank={self.node_rank}"

    def _register_worker(self, worker_url: str) -> None:
        if self.node_rank != 0 or not self.router_ip or not self.router_port:
            return

        logger.info(
            "Registering worker %s to router %s:%s (engine_rank=%s, worker_type=%s)",
            worker_url,
            self.router_ip,
            self.router_port,
            self.rank,
            self.worker_type,
        )
        worker_key = self._worker_key()
        payload = {
            "url": worker_url,
            "worker_key": worker_key,
            "worker_type": self.worker_type,
        }
        if self.worker_type == "prefill" and self.disaggregation_bootstrap_port is not None:
            payload["bootstrap_port"] = self.disaggregation_bootstrap_port

        response = None
        prefer_legacy_endpoint = parse(sglang_router.__version__) <= parse("0.2.1") or self.args.use_slime_router
        if prefer_legacy_endpoint:
            assert self.worker_type == "regular", "pd disaggregation is not supported in old router or slime router."
            response = requests.post(
                f"http://{self.router_ip}:{self.router_port}/add_worker",
                params={"url": worker_url, "worker_key": worker_key},
            )
            if _is_not_found(response) and not self.args.use_slime_router:
                logger.info(
                    "Router %s:%s does not expose /add_worker; retrying registration via /workers",
                    self.router_ip,
                    self.router_port,
                )
                response = requests.post(
                    f"http://{self.router_ip}:{self.router_port}/workers",
                    json=payload,
                )
        else:
            response = requests.post(
                f"http://{self.router_ip}:{self.router_port}/workers",
                json=payload,
            )
        response.raise_for_status()
        logger.info("Registered worker %s to router successfully", worker_url)

    def _unregister_worker(self, worker_url: str) -> None:
        if self.node_rank != 0 or not self.router_ip or not self.router_port:
            return

        logger.info(
            "Unregistering worker %s from router %s:%s (engine_rank=%s)",
            worker_url,
            self.router_ip,
            self.router_port,
            self.rank,
        )
        worker_key = self._worker_key()

        response = None
        prefer_legacy_endpoint = parse(sglang_router.__version__) <= parse("0.2.1") or self.args.use_slime_router
        if prefer_legacy_endpoint:
            response = requests.post(
                f"http://{self.router_ip}:{self.router_port}/remove_worker",
                params={"url": worker_url, "worker_key": worker_key},
            )
            if _is_not_found(response) and not self.args.use_slime_router:
                logger.info(
                    "Router %s:%s does not expose /remove_worker; retrying removal via /workers/{url}",
                    self.router_ip,
                    self.router_port,
                )
                encoded_worker_url = quote(worker_url, safe="")
                response = requests.delete(f"http://{self.router_ip}:{self.router_port}/workers/{encoded_worker_url}")
        elif parse(sglang_router.__version__) < parse("0.3.0"):
            encoded_worker_url = quote(worker_url, safe="")
            response = requests.delete(f"http://{self.router_ip}:{self.router_port}/workers/{encoded_worker_url}")
        else:
            try:
                all_workers = requests.get(f"http://{self.router_ip}:{self.router_port}/workers").json()["workers"]
                for worker in all_workers:
                    if worker["url"] == worker_url:
                        response = requests.delete(f"http://{self.router_ip}:{self.router_port}/workers/{worker['id']}")
                        break
                else:
                    logger.warning("Worker %s not found in router during removal.", worker_url)
                    return
            except Exception as e:
                logger.warning("Failed to fetch workers list or remove worker %s: %s", worker_url, e)
                return

        if response is not None:
            response.raise_for_status()
        logger.info("Unregistered worker %s from router successfully", worker_url)

    def _shadow_kv_cache_socket_path(self) -> str | None:
        explicit = getattr(self.args, "sglang_shadow_worker_kv_cache_socket_path", None)
        if explicit:
            os.environ["SGLANG_KV_CACHE_SOCKET_PATH"] = explicit
            return explicit
        return os.environ.get("SGLANG_KV_CACHE_SOCKET_PATH")

    def _shadow_weight_server_base_port(self) -> int | None:
        return _get_weight_server_base_port(self.args)

    def _shadow_min_gpu_id(self) -> int:
        return _get_weight_server_min_gpu_id(self.args)

    def _disable_shadow_worker(self, reason: str) -> None:
        self.shadow_worker_enabled = False
        self._shadow_disable_reason = reason
        logger.info("Shadow-worker fast restart disabled for engine rank=%s: %s", self.rank, reason)

    def _ensure_shadow_failover_watcher_started(self) -> None:
        if self._shadow_failover_thread is not None and self._shadow_failover_thread.is_alive():
            return
        self._shadow_failover_stop_event.clear()
        self._shadow_failover_thread = threading.Thread(
            target=self._shadow_failover_watcher_loop,
            name=f"shadow-failover-rank-{self.rank}",
            daemon=True,
        )
        self._shadow_failover_thread.start()
        logger.info("Started shadow failover watcher for engine rank=%s", self.rank)

    def _shadow_failover_watcher_loop(self) -> None:
        poll_interval_seconds = 0.2
        last_dead_pid = None
        while not self._shadow_failover_stop_event.wait(poll_interval_seconds):
            if not self.shadow_worker_enabled or self.shadow_worker is None or self.shadow_server_args is None:
                last_dead_pid = None
                continue

            process = getattr(self, "process", None)
            process_pid = getattr(process, "pid", None)
            if process is None or process.is_alive():
                last_dead_pid = None
                continue

            if last_dead_pid != process_pid:
                logger.warning(
                    "Detected primary worker process exit for engine rank=%s (pid=%s); attempting immediate shadow handover",
                    self.rank,
                    process_pid,
                )
                last_dead_pid = process_pid

            promoted = self._promote_shadow_worker_impl(
                reason=f"shadow failover watcher detected primary process exit pid={process_pid}",
                readiness_timeout=1.0,
            )
            if promoted:
                last_dead_pid = None

    def _promote_shadow_worker_impl(self, *, reason: str, readiness_timeout: float = 10.0) -> bool:
        with self._shadow_handover_lock:
            if self._pending_shadow_handover_reconnect:
                logger.info(
                    "Shadow handover already completed for engine rank=%s; reusing pending reconnect reason=%s",
                    self.rank,
                    self._pending_shadow_handover_reason,
                )
                return True

            if not self._shadow_worker_ready_for_handover(timeout=readiness_timeout):
                logger.warning(
                    "Shadow worker is not ready for handover for engine rank=%s (reason=%s)",
                    self.rank,
                    reason,
                )
                return False

            old_worker_url = self._worker_url(self.server_host, self.server_port)
            new_worker_url = self._worker_url(self.shadow_server_args.host, self.shadow_server_args.port)
            logger.info(
                "Promoting shadow worker for engine rank=%s (reason=%s, old_url=%s, new_url=%s, old_pid=%s, shadow_pid=%s)",
                self.rank,
                reason,
                old_worker_url,
                new_worker_url,
                getattr(self.process, "pid", None),
                getattr(self.shadow_worker, "pid", None),
            )
            self._register_worker(new_worker_url)
            try:
                self._unregister_worker(old_worker_url)
            except Exception:
                logger.warning(
                    "Failed to unregister previous worker %s during shadow handover.",
                    old_worker_url,
                    exc_info=True,
                )

            if getattr(self, "process", None):
                kill_process_tree(self.process.pid)

            self.process = self.shadow_worker
            self.shadow_worker = None
            self.server_args = self.shadow_server_args
            self.shadow_server_args = None
            self.server_host = self.server_args.host
            self.server_port = self.server_args.port
            self.shadow_worker_enabled = False
            self._pending_shadow_handover_reconnect = True
            self._pending_shadow_handover_reason = reason
            logger.info("Promoted shadow worker for engine rank=%s; active server is now %s", self.rank, new_worker_url)
            return True

    def _shadow_worker_requested(self) -> bool:
        return bool(
            getattr(self.args, "sglang_enable_fast_restart", False)
            and getattr(self.args, "use_fault_tolerance", False)
            and not self.args.rollout_external
        )

    def _compute_shadow_weight_load_port(self) -> int | None:
        return _compute_weight_load_port(self.args, self.base_gpu_id, self.engine_role)

    def _replace_port(self, addr: str, port: int) -> str:
        host, _ = addr.rsplit(":", 1)
        return f"{host}:{port}"

    def _init_shadow_worker(self) -> None:
        if not self._shadow_worker_requested():
            self._disable_shadow_worker("feature flag not enabled")
            return
        if self.worker_type != "regular":
            self._disable_shadow_worker("only regular rollout engines currently support fast restart")
            return
        if self._shadow_kv_cache_socket_path() is None:
            self._disable_shadow_worker("SGLANG_KV_CACHE_SOCKET_PATH is not configured")
            return

        weight_load_port = self._compute_shadow_weight_load_port()
        if weight_load_port is None:
            self._disable_shadow_worker("weight server base port or GPU mapping is unavailable")
            return

        _, shadow_port = self._get_current_node_ip_and_free_port(start_port=self.server_port + _SHADOW_PORT_OFFSET)
        _, shadow_nccl_port = self._get_current_node_ip_and_free_port(
            start_port=self.server_args.nccl_port + _SHADOW_NCCL_PORT_OFFSET
        )
        _, shadow_dist_init_port = self._get_current_node_ip_and_free_port(
            start_port=int(self.server_args.dist_init_addr.rsplit(":", 1)[1]) + _SHADOW_DIST_INIT_PORT_OFFSET
        )

        shadow_kwargs = dataclasses.asdict(self.server_args)
        shadow_kwargs.update(
            {
                "port": shadow_port,
                "nccl_port": shadow_nccl_port,
                "dist_init_addr": self._replace_port(self.server_args.dist_init_addr, shadow_dist_init_port),
                "enable_memory_saver": True,
                "load_format": "weight_deamon",
                "weight_load_port": weight_load_port,
                "skeleton_worker": True,
            }
        )
        shadow_kwargs = _filter_server_args_for_current_sglang(
            shadow_kwargs,
            context=f"shadow worker rank={self.rank} role={self.engine_role}",
        )
        self.shadow_server_args = ServerArgs(**shadow_kwargs)
        logger.info(
            "Launch shadow worker for engine rank=%s at %s:%s (weight_load_port=%s, kv_socket=%s, base_gpu_id=%s)",
            self.rank,
            self.shadow_server_args.host,
            self.shadow_server_args.port,
            weight_load_port,
            self._shadow_kv_cache_socket_path(),
            self.base_gpu_id,
        )
        self.shadow_worker = launch_server_process(self.shadow_server_args, wait_healthy=False)
        self.shadow_worker_enabled = True
        logger.info(
            "Shadow worker process started for engine rank=%s with pid=%s",
            self.rank,
            getattr(self.shadow_worker, "pid", None),
        )
        self._ensure_shadow_failover_watcher_started()

    def _shadow_worker_ready_for_handover(self, timeout: float = 10.0) -> bool:
        if not self.shadow_worker_enabled or self.shadow_worker is None or self.shadow_server_args is None:
            logger.debug("Shadow worker readiness probe skipped for engine rank=%s because shadow worker is unavailable", self.rank)
            return False
        if not self.shadow_worker.is_alive():
            logger.warning(
                "Shadow worker readiness probe failed for engine rank=%s because process pid=%s is not alive",
                self.rank,
                getattr(self.shadow_worker, "pid", None),
            )
            return False

        url = f"http://{self.shadow_server_args.host}:{self.shadow_server_args.port}/health"
        for _ in range(2):
            try:
                response = requests.get(url, timeout=timeout)
                if response.status_code in _SHADOW_SUPPORTED_HEALTH_CODES:
                    logger.debug(
                        "Shadow worker readiness probe succeeded for engine rank=%s with status=%s",
                        self.rank,
                        response.status_code,
                    )
                    return True
                if response.status_code != 503:
                    logger.warning(
                        "Shadow worker readiness probe got unexpected status for engine rank=%s: %s",
                        self.rank,
                        response.status_code,
                    )
                    return False
            except requests.RequestException:
                logger.debug("Shadow worker readiness probe request failed for engine rank=%s", self.rank, exc_info=True)
                return False
            time.sleep(2)
        return False

    def wait_until_ready(
        self,
        wait_for_shadow: bool = False,
        shadow_timeout: float = 600.0,
        stabilization_seconds: float = 0.0,
        wait_for_flush_cache: bool = True,
    ) -> bool:
        if self.node_rank != 0:
            return True

        if getattr(self, "server_args", None) is not None:
            logger.info(
                "Waiting for primary worker to become healthy for engine rank=%s at %s",
                self.rank,
                self.server_args.url(),
            )
            _wait_server_healthy(
                base_url=self.server_args.url(),
                api_key=self.server_args.api_key,
                is_process_alive=lambda: getattr(self, "process", None) is not None and self.process.is_alive(),
                wait_for_flush_cache=wait_for_flush_cache,
            )
            logger.info("Primary worker is healthy for engine rank=%s at %s", self.rank, self.server_args.url())

        if not wait_for_shadow or not self.shadow_worker_enabled:
            return True

        logger.info(
            "Waiting for shadow worker readiness for engine rank=%s (timeout=%.1fs, stabilization=%.1fs, shadow_url=%s)",
            self.rank,
            shadow_timeout,
            stabilization_seconds,
            self._worker_url(self.shadow_server_args.host, self.shadow_server_args.port),
        )
        deadline = time.monotonic() + max(0.0, shadow_timeout)
        attempt = 0
        while time.monotonic() < deadline:
            attempt += 1
            remaining = deadline - time.monotonic()
            probe_timeout = max(1.0, min(5.0, remaining))
            if attempt == 1 or attempt % 5 == 0:
                logger.info(
                    "Shadow readiness probe attempt=%s for engine rank=%s (remaining=%.1fs, probe_timeout=%.1fs, pid=%s)",
                    attempt,
                    self.rank,
                    remaining,
                    probe_timeout,
                    getattr(self.shadow_worker, "pid", None),
                )
            if self._shadow_worker_ready_for_handover(timeout=probe_timeout):
                if stabilization_seconds > 0:
                    logger.info(
                        "Shadow worker for engine rank=%s is ready; waiting %.1fs before allowing rollout traffic",
                        self.rank,
                        stabilization_seconds,
                    )
                    time.sleep(stabilization_seconds)
                return True

            if self.shadow_worker is None or not self.shadow_worker.is_alive():
                raise RuntimeError(f"Shadow worker for engine rank={self.rank} exited before becoming ready")

            time.sleep(min(2.0, max(0.1, remaining)))

        raise TimeoutError(
            f"Timed out waiting {shadow_timeout:.1f}s for shadow worker of engine rank={self.rank} to become ready"
        )

    def promote_shadow_worker(self) -> bool:
        return self._promote_shadow_worker_impl(reason="explicit promote_shadow_worker call")

    def consume_shadow_handover_reconnect_event(self) -> dict:
        with self._shadow_handover_lock:
            pending = self._pending_shadow_handover_reconnect
            reason = self._pending_shadow_handover_reason
            self._pending_shadow_handover_reconnect = False
            self._pending_shadow_handover_reason = None
        return {
            "pending": pending,
            "reason": reason,
        }

    def _make_request(self, endpoint: str, payload: dict | None = None):
        """Make a POST request to the specified endpoint with the given payload.

        Args:
            endpoint: The API endpoint to call
            payload: The JSON payload to send (default: empty dict)

        Returns:
            The JSON response from the server
        """
        if self.node_rank != 0:
            return

        url = f"http://{self.server_host}:{self.server_port}/{endpoint}"
        response = requests.post(url, json=payload or {})
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            e.add_note(f"{response.text=}")
            raise
        return response.json()

    def health_generate(self, timeout: float = 5.0) -> bool:
        """Run /health_generate on the underlying SGLang HTTP server.

        Args:
            timeout: Timeout for the health request in seconds.

        Returns:
            True if the server responds with HTTP 200.

        Raises:
            requests.RequestException: If the request fails for any reason, including timeout.
        """
        if self.node_rank != 0:
            return True

        response = requests.get(
            f"http://{self.server_host}:{self.server_port}/health_generate",
            timeout=timeout,
        )
        response.raise_for_status()
        return True

    def check_health(self, timeout: float = 5.0) -> bool:
        """Run /health on the underlying SGLang HTTP server.

        This check is intentionally lightweight and is used by rollout health monitor.

        Args:
            timeout: Timeout for the health request in seconds.

        Returns:
            True if the server process is alive and responds with HTTP 200.
        """
        node_rank = getattr(self, "node_rank", None)
        if node_rank is None:
            logger.info(
                "Health check skipped because engine rank=%s has not finished init yet (node_rank is unavailable)",
                getattr(self, "rank", None),
            )
            return False

        if node_rank != 0:
            return True

        process = getattr(self, "process", None)
        if process is None or not process.is_alive():
            logger.warning(
                "Health check failed for engine rank=%s because process is not alive (pid=%s)",
                self.rank,
                getattr(process, "pid", None),
            )
            return False

        if not hasattr(self, "_health_check_non_200_failures"):
            self._health_check_non_200_failures = 0

        server_host = getattr(self, "server_host", None)
        server_port = getattr(self, "server_port", None)
        if not server_host or server_port is None:
            logger.info(
                "Health check skipped because engine rank=%s is missing server endpoint info (host=%s port=%s)",
                getattr(self, "rank", None),
                server_host,
                server_port,
            )
            return False

        try:
            response = requests.get(
                f"http://{server_host}:{server_port}/health",
                timeout=timeout,
            )
        except requests.RequestException:
            return False

        if response.status_code == 200:
            if self._health_check_non_200_failures > 0:
                logger.info(
                    "Health /health recovered for engine rank=%s after %s consecutive non-200 responses",
                    self.rank,
                    self._health_check_non_200_failures,
                )
            self._health_check_non_200_failures = 0
            return True

        self._health_check_non_200_failures += 1
        if self._health_check_non_200_failures >= 5:
            logger.warning(
                "Health /health failed for engine rank=%s with %s consecutive non-200 responses (last_status=%s)",
                self.rank,
                self._health_check_non_200_failures,
                response.status_code,
            )
            return False

        logger.info(
            "Health /health got non-200 for engine rank=%s (status=%s, consecutive=%s/5); treating as transient",
            self.rank,
            response.status_code,
            self._health_check_non_200_failures,
        )
        return True

    def update_weights_from_tensor(
        self,
        serialized_named_tensors: list[str],
        load_format: str | None = None,
        flush_cache: bool = False,
        weight_version: str | None = None,
    ):
        """
        Update model weights from tensor data. The HTTP server will only post meta data, and the real weights will be copied directly from GPUs.

        Note: The model should be on GPUs rather than CPU for this functionality to work properly.
        If you encounter issues, ensure your model is loaded on GPU devices rather than CPU.
        """
        payload = {
            "serialized_named_tensors": serialized_named_tensors,
            "load_format": load_format,
            "flush_cache": flush_cache,
        }
        if weight_version is not None:
            payload["weight_version"] = weight_version
        return self._make_request(
            "update_weights_from_tensor",
            payload,
        )

    def flush_cache(self):
        """Flush the cache of the server."""
        if self.node_rank != 0:
            return
        # flush cache will not return status_code 200 when there are pending requests
        for _ in range(60):
            try:
                response = requests.get(f"http://{self.server_host}:{self.server_port}/flush_cache")
                if response.status_code == 200:
                    break
            except NewConnectionError as e:
                raise e
            except Exception as e:
                logger.info(f"Error flushing cache: {e}")
                time.sleep(1)
                continue
        else:
            raise TimeoutError("Timeout while flushing cache.")

    def shutdown(self):
        if self.args.rollout_external:
            return

        logger.info(f"Shutdown engine {self.server_host}:{self.server_port}...")
        self._shadow_failover_stop_event.set()
        self._unregister_worker(self._worker_url(self.server_host, self.server_port))
        kill_process_tree(self.process.pid)
        if self.shadow_worker is not None:
            kill_process_tree(self.shadow_worker.pid)
            self.shadow_worker = None
            self.shadow_server_args = None
            self.shadow_worker_enabled = False

    def get_weight_version(self):
        if self.node_rank != 0:
            return
        url = f"http://{self.server_host}:{self.server_port}/get_weight_version"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()["weight_version"]

    def get_weights_by_name(self, name: str, truncate_size: int = 4):
        return self._make_request(
            "get_weights_by_name",
            {
                "name": name,
                "truncate_size": truncate_size,
            },
        )

    def release_memory_occupation(self):
        self.flush_cache()
        return self._make_request("release_memory_occupation")

    def resume_memory_occupation(self, tags: list[str] = None):
        """
        Available tags for multi-stage resume: weights, kv_cache
        """
        return self._make_request(
            "resume_memory_occupation",
            {"tags": tags},
        )

    def check_weights(self, action: str):
        return self._make_request("weights_checker", {"action": action})

    def init_weights_update_group(self, master_address, master_port, rank_offset, world_size, group_name, backend):
        return self._make_request(
            "init_weights_update_group",
            {
                "master_address": master_address,
                "master_port": master_port,
                "rank_offset": rank_offset,
                "world_size": world_size,
                "group_name": group_name,
                "backend": backend,
            },
        )

    def destroy_weights_update_group(self, group_name):
        try:
            return self._make_request(
                "destroy_weights_update_group",
                {
                    "group_name": group_name,
                },
            )
        except requests.exceptions.RequestException:
            # catch the case there the engine is just created and does not have the group.
            pass

    def update_weights_from_distributed(
        self, names, dtypes, shapes, group_name, flush_cache=False, weight_version: str | None = None
    ):
        payload = {
            "names": names,
            "dtypes": [str(dtype).replace("torch.", "") for dtype in dtypes],
            "shapes": shapes,
            "group_name": group_name,
            "flush_cache": flush_cache,
        }
        if weight_version is not None:
            payload["weight_version"] = weight_version
        return self._make_request(
            "update_weights_from_distributed",
            payload,
        )

    def pause_generation(self):
        response = requests.post(f"http://{self.server_host}:{self.server_port}/pause_generation", json={})
        response.raise_for_status()
        return response

    def continue_generation(self):
        response = requests.post(f"http://{self.server_host}:{self.server_port}/continue_generation", json={})
        response.raise_for_status()
        return response

    def post_process_weights(
        self,
        restore_weights_before_load: bool = False,
        post_process_quantization: bool = False,
    ):
        """
        Update model weights from tensor data. The HTTP server will only post meta data, and the real weights will be copied directly from GPUs.
        Note: The model should be on GPUs rather than CPU for this functionality to work properly.
        If you encounter issues, ensure your model is loaded on GPU devices rather than CPU.
        """

        return self._make_request(
            "post_process_weights",
            {
                "restore_weights_before_load": restore_weights_before_load,
                "post_process_quantization": post_process_quantization,
            },
        )

    def start_profile(
        self,
        # The output directory
        output_dir: str | None = None,
        # If set, it profile as many as this number of steps.
        # If it is set, profiling is automatically stopped after this step, and
        # the caller doesn't need to run stop_profile.
        start_step: int | None = None,
        num_steps: int | None = None,
        activities: list[str] | None = None,
        profile_by_stage: bool = False,
        with_stack: bool | None = None,
        record_shapes: bool | None = None,
    ):
        response = requests.post(
            f"http://{self.server_host}:{self.server_port}/start_profile",
            json={
                "output_dir": output_dir,
                "start_step": start_step,
                "num_steps": num_steps,
                "activities": activities,
                "profile_by_stage": profile_by_stage,
                "with_stack": with_stack,
                "record_shapes": record_shapes,
            },
        )
        response.raise_for_status()
        return response

    def stop_profile(self):
        response = requests.post(f"http://{self.server_host}:{self.server_port}/stop_profile", json={})
        response.raise_for_status()
        return response

    def simulate_crash(self):
        if self.args.rollout_external or not getattr(self, "process", None):
            logger.info(
                "simulate_crash called but no local engine process exists (rollout_external=%s); skip kill",
                self.args.rollout_external,
            )
            return

        logger.info(
            "Simulating crash on engine %s:%s (pid=%s, shadow_enabled=%s, shadow_pid=%s)",
            self.server_host,
            self.server_port,
            getattr(self.process, "pid", None),
            self.shadow_worker_enabled,
            getattr(self.shadow_worker, "pid", None) if self.shadow_worker is not None else None,
        )
        if self.shadow_worker_enabled and self.shadow_worker is not None:
            kill_process_tree(self.process.pid)
            return
        self.shutdown()


def _compute_server_args(
    args,
    rank,
    dist_init_addr,
    nccl_port,
    host,
    port,
    worker_type: str = "regular",
    disaggregation_bootstrap_port: int | None = None,
    base_gpu_id: int | None = None,
    engine_role: str = "rollout",
):
    is_prm = engine_role == "prm"
    gpus_per_engine = args.prm_num_gpus_per_engine if is_prm else args.rollout_num_gpus_per_engine
    model_path = args.prm_model_path if is_prm else args.hf_checkpoint

    nnodes = max(1, gpus_per_engine // args.num_gpus_per_node)
    node_rank = rank % nnodes
    physical_base = base_gpu_id if base_gpu_id is not None else get_base_gpu_id(args, rank)
    base = _to_local_gpu_id(physical_base)
    kwargs = {
        "model_path": model_path,
        "trust_remote_code": True,
        "random_seed": args.seed + rank,
        # memory
        "enable_memory_saver": args.offload_rollout if not is_prm else False,
        # distributed
        "host": host,
        "port": port,
        "nccl_port": nccl_port,
        "nnodes": nnodes,
        "node_rank": node_rank,
        "dist_init_addr": dist_init_addr,
        "gpu_id_step": 1,
        "base_gpu_id": base,
        # parallel
        "tp_size": gpus_per_engine // args.sglang_pp_size,
        "dp_size": args.sglang_dp_size,
        "pp_size": args.sglang_pp_size,
        "ep_size": args.sglang_ep_size,
        # always skip warmup to prevent warmup timeout.
        "skip_server_warmup": True,
        # always enable draft weights cpu backup so that we run training without mtp weights.
        "enable_draft_weights_cpu_backup": True,
    }

    if worker_type == "prefill" and not is_prm:
        kwargs["disaggregation_mode"] = "prefill"
        kwargs["load_balance_method"] = "round_robin"
        assert (
            disaggregation_bootstrap_port is not None
        ), "disaggregation_bootstrap_port must be set for prefill worker"
        kwargs["disaggregation_bootstrap_port"] = disaggregation_bootstrap_port
    elif worker_type == "decode" and not is_prm:
        kwargs["disaggregation_mode"] = "decode"
        kwargs["prefill_round_robin_balance"] = True

    if args.use_rollout_routing_replay:
        kwargs["enable_return_routed_experts"] = True
    if args.fp16:
        kwargs["dtype"] = "float16"
    external_engine_need_check_fields = [k for k in kwargs.keys() if k not in _EXTERNAL_ENGINE_SKIP_CHECK_FIELDS]

    unused_keys = set(kwargs.keys())
    for attr in dataclasses.fields(ServerArgs):
        if worker_type == "decode" and attr.name == "enable_hierarchical_cache":
            continue
        if hasattr(args, f"sglang_{attr.name}") and attr.name not in kwargs:
            kwargs[attr.name] = getattr(args, f"sglang_{attr.name}")
        unused_keys.discard(attr.name)

    recovery_remote_instance = _get_recovery_remote_instance_override(args, rank)
    if recovery_remote_instance is not None:
        kwargs["load_format"] = "remote_instance"
        kwargs["remote_instance_weight_loader_seed_instance_ip"] = recovery_remote_instance["seed_instance_ip"]
        kwargs["remote_instance_weight_loader_seed_instance_service_port"] = recovery_remote_instance[
            "seed_instance_service_port"
        ]
        kwargs["remote_instance_weight_loader_send_weights_group_ports"] = recovery_remote_instance[
            "send_weights_group_ports"
        ]
        kwargs["remote_instance_weight_loader_backend"] = recovery_remote_instance.get("backend", "nccl")
        kwargs.pop("weight_load_port", None)
        logger.info(
            "Use remote-instance recovery weight loading for rank=%s from seed=%s:%s ports=%s",
            rank,
            kwargs["remote_instance_weight_loader_seed_instance_ip"],
            kwargs["remote_instance_weight_loader_seed_instance_service_port"],
            kwargs["remote_instance_weight_loader_send_weights_group_ports"],
        )
    else:
        if kwargs.get("weight_load_port") is None:
            inferred_weight_load_port = _compute_weight_load_port(args, physical_base, engine_role)
            if inferred_weight_load_port is not None:
                kwargs["weight_load_port"] = inferred_weight_load_port

        if kwargs.get("weight_load_port") is not None and kwargs.get("load_format") == "auto":
            kwargs["load_format"] = "weight_deamon"

    # for compatibility with old args
    if len(unused_keys) > 0:
        logger.info(f"Warning: The following arguments is not supported in the current sglang: {unused_keys}.")
        for key in unused_keys:
            kwargs.pop(key)

    return kwargs, external_engine_need_check_fields


_EXTERNAL_ENGINE_SKIP_CHECK_FIELDS = [
    "model_path",
    "trust_remote_code",
    "random_seed",
    "nccl_port",
    "dist_init_addr",
    "skip_server_warmup",
    "enable_draft_weights_cpu_backup",
    "mem_fraction_static",
]
