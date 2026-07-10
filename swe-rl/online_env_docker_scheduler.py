from __future__ import annotations

import asyncio
import contextlib
import json
import os
import random
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib import request as urllib_request

try:
    from loguru import logger
except Exception:  # pragma: no cover - fallback for lightweight test envs
    import logging

    logger = logging.getLogger("swe.online_env_docker_scheduler")

GIB = 1024 ** 3
_UNBOUNDED_RESOURCE_BYTES = 1 << 60
_UNBOUNDED_CPU_PERCENT = 1e9


def _get_json_blocking(url: str, timeout: float = 30.0) -> dict[str, Any]:
    req = urllib_request.Request(url, method="GET")
    with urllib_request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise RuntimeError(f"GET {url} returned non-dict payload: {payload!r}")
    return payload


@dataclass(slots=True)
class ResourceVector:
    memory_bytes: float
    cpu_percent: float
    disk_read_bytes: float
    disk_write_bytes: float

    def plus(self, other: "ResourceVector") -> "ResourceVector":
        return ResourceVector(
            memory_bytes=self.memory_bytes + other.memory_bytes,
            cpu_percent=self.cpu_percent + other.cpu_percent,
            disk_read_bytes=self.disk_read_bytes + other.disk_read_bytes,
            disk_write_bytes=self.disk_write_bytes + other.disk_write_bytes,
        )

    def minus(self, other: "ResourceVector") -> "ResourceVector":
        return ResourceVector(
            memory_bytes=max(0.0, self.memory_bytes - other.memory_bytes),
            cpu_percent=max(0.0, self.cpu_percent - other.cpu_percent),
            disk_read_bytes=max(0.0, self.disk_read_bytes - other.disk_read_bytes),
            disk_write_bytes=max(0.0, self.disk_write_bytes - other.disk_write_bytes),
        )


@dataclass(slots=True)
class RepoResourceProfile:
    sample_count: int = 0
    peak_memory_bytes: float = 0.0
    avg_cpu_percent: float = 0.0
    avg_disk_read_bytes: float = 0.0
    avg_disk_write_bytes: float = 0.0
    avg_total_time_sec: float = 0.0
    last_updated: float = 0.0


@dataclass(slots=True)
class PromptResourceSummary:
    prompt_id: str
    repo: str
    data_key: str
    sample_count: int
    peak_memory_bytes: float
    avg_cpu_percent: float
    disk_read_bytes: float
    disk_write_bytes: float
    started_at: float
    finished_at: float
    total_time_sec: float
    cpu_sample_count: int = 0
    lease_count: int = 0


@dataclass(slots=True)
class SchedulerConfig:
    enabled: bool = False
    sampling_interval_sec: float = 10.0
    scheduler_safety_margin: float = 0.9
    memory_oversell_ratio: float = 1.0
    memory_peak_scale: float = 0.7
    cpu_oversell_ratio: float = 2.0
    disk_read_oversell_ratio: float = 10.0
    disk_write_oversell_ratio: float = 10.0
    max_unknown_repo_concurrency: int = 128
    cold_start_memory_multiplier: float = 2.0
    cold_start_cpu_multiplier: float = 1.5
    max_active_prompts: int = 0
    startup_max_active_prompts: int = 0
    startup_cap_duration_sec: float = 0.0

    default_memory_bytes: float = float(2 * GIB)
    default_cpu_percent: float = 30.0
    default_disk_read_bytes: float = float(2 * GIB)
    default_disk_write_bytes: float = float(2 * GIB)

    profile_json_path: str = ""
    resource_stats_dir: str = ""
    resource_stats_refresh_sec: float = 60
    use_resource_stats_dir: bool = True
    disable_live_stats_polling: bool = False
    enable_live_profile_updates: bool = False
    min_live_profile_samples: int = 2
    min_profile_memory_bytes: float = 0.0
    max_stats_requests_per_round: int = 16
    stats_request_spacing_sec: float = 0.02
    use_realtime_server_memory: bool = True
    use_realtime_server_cpu: bool = True
    use_realtime_server_disk: bool = True
    server_memory_refresh_sec: float = 30.0
    verbose_logging: bool = True
    blocked_log_interval_sec: float = 5.0
    verbose_log_interval_sec: float = 10.0
    preserve_prompt_order: bool = False
    head_block_requeue_threshold: int = 3
    head_block_requeue_offset: int = 5
    max_requests_per_resource_update: int = 64
    realtime_local_active_discount: float = 0.7
    duration_priority_weight: float = 0.6
    duration_priority_ref_sec: float = 300.0
    random_seed: int = 0

    @classmethod
    def from_env(cls) -> "SchedulerConfig":
        enabled = _env_flag("SWE_ENABLE_ONLINE_ENV_DOCKER_SCHEDULER", False)
        repo_root = Path(__file__).resolve().parent.parent

        default_memory_bytes = float(os.getenv("SWE_SCHED_DEFAULT_MEMORY_BYTES", str(0.09 * GIB)))
        default_cpu_percent = float(os.getenv("SWE_SCHED_DEFAULT_CPU_PERCENT", "3.6"))
        default_disk_read_bytes = float(os.getenv("SWE_SCHED_DEFAULT_DISK_READ_BYTES", str(0.1 * GIB)))
        default_disk_write_bytes = float(os.getenv("SWE_SCHED_DEFAULT_DISK_WRITE_BYTES", str(0.1 * GIB)))

        profile_json_path = os.getenv(
            "SWE_REPO_RESOURCE_STATS_PATH",
            str((repo_root / "output" / "swe_rollouts" / "repo_resource_stats.json")),
        )
        resource_stats_dir = os.getenv(
            "SWE_RESOURCE_STATS_DIR",
            str((repo_root / "export" / "replay_image_resource_profile_20260505_184106")),
        )

        return cls(
            enabled=enabled,
            sampling_interval_sec=float(os.getenv("SWE_SCHED_SAMPLING_INTERVAL_SEC", "10.0")),
            scheduler_safety_margin=float(os.getenv("SWE_SCHED_SAFETY_MARGIN", "0.9")),
            memory_oversell_ratio=float(os.getenv("SWE_SCHED_MEMORY_OVERSELL_RATIO", "1.0")),
            memory_peak_scale=float(os.getenv("SWE_SCHED_MEMORY_PEAK_SCALE", "0.7")),
            cpu_oversell_ratio=float(os.getenv("SWE_SCHED_CPU_OVERSELL_RATIO", "2.0")),
            disk_read_oversell_ratio=float(os.getenv("SWE_SCHED_DISK_READ_OVERSELL_RATIO", "10.0")),
            disk_write_oversell_ratio=float(os.getenv("SWE_SCHED_DISK_WRITE_OVERSELL_RATIO", "10.0")),
            max_unknown_repo_concurrency=int(os.getenv("SWE_SCHED_MAX_UNKNOWN_REPO_CONCURRENCY", "192")),
            cold_start_memory_multiplier=float(os.getenv("SWE_SCHED_COLD_START_MEMORY_MULTIPLIER", "2.0")),
            cold_start_cpu_multiplier=float(os.getenv("SWE_SCHED_COLD_START_CPU_MULTIPLIER", "1.5")),
            max_active_prompts=int(os.getenv("SWE_SCHED_MAX_ACTIVE_PROMPTS", "0")),
            startup_max_active_prompts=int(os.getenv("SWE_SCHED_STARTUP_MAX_ACTIVE_PROMPTS", "0")),
            startup_cap_duration_sec=float(os.getenv("SWE_SCHED_STARTUP_CAP_DURATION_SEC", "0")),
            default_memory_bytes=default_memory_bytes,
            default_cpu_percent=default_cpu_percent,
            default_disk_read_bytes=default_disk_read_bytes,
            default_disk_write_bytes=default_disk_write_bytes,
            profile_json_path=profile_json_path,
            resource_stats_dir=resource_stats_dir,
            resource_stats_refresh_sec=float(os.getenv("SWE_SCHED_RESOURCE_STATS_REFRESH_SEC", "30")),
            use_resource_stats_dir=_env_flag("SWE_SCHED_USE_RESOURCE_STATS_DIR", True),
            disable_live_stats_polling=_env_flag("SWE_SCHED_DISABLE_LIVE_STATS_POLLING", False),
            enable_live_profile_updates=_env_flag("SWE_SCHED_ENABLE_LIVE_PROFILE_UPDATES", False),
            min_live_profile_samples=int(os.getenv("SWE_SCHED_MIN_LIVE_PROFILE_SAMPLES", "2")),
            min_profile_memory_bytes=float(
                os.getenv("SWE_SCHED_MIN_PROFILE_MEMORY_BYTES", str(default_memory_bytes))
            ),
            max_stats_requests_per_round=int(os.getenv("SWE_SCHED_MAX_STATS_REQUESTS_PER_ROUND", "16")),
            stats_request_spacing_sec=float(os.getenv("SWE_SCHED_STATS_REQUEST_SPACING_SEC", "0.02")),
            use_realtime_server_memory=_env_flag("SWE_SCHED_USE_REALTIME_SERVER_MEMORY", True),
            use_realtime_server_cpu=_env_flag("SWE_SCHED_USE_REALTIME_SERVER_CPU", True),
            use_realtime_server_disk=_env_flag("SWE_SCHED_USE_REALTIME_SERVER_DISK", True),
            server_memory_refresh_sec=float(os.getenv("SWE_SCHED_SERVER_MEMORY_REFRESH_SEC", "30")),
            verbose_logging=_env_flag("SWE_SCHED_VERBOSE_LOGGING", True),
            blocked_log_interval_sec=float(os.getenv("SWE_SCHED_BLOCKED_LOG_INTERVAL_SEC", "5")),
            verbose_log_interval_sec=float(os.getenv("SWE_SCHED_VERBOSE_LOG_INTERVAL_SEC", "10")),
            preserve_prompt_order=_env_flag("SWE_SCHED_PRESERVE_PROMPT_ORDER", False),
            head_block_requeue_threshold=int(os.getenv("SWE_SCHED_HEAD_BLOCK_REQUEUE_THRESHOLD", "100")),
            head_block_requeue_offset=int(os.getenv("SWE_SCHED_HEAD_BLOCK_REQUEUE_OFFSET", "1")),
            max_requests_per_resource_update=int(
                os.getenv("SWE_SCHED_MAX_REQUESTS_PER_RESOURCE_UPDATE", "8")
            ),
            realtime_local_active_discount=float(
                os.getenv("SWE_SCHED_REALTIME_LOCAL_ACTIVE_DISCOUNT", "0.7")
            ),
            duration_priority_weight=float(os.getenv("SWE_SCHED_DURATION_PRIORITY_WEIGHT", "0.4")),
            duration_priority_ref_sec=float(os.getenv("SWE_SCHED_DURATION_PRIORITY_REF_SEC", "300")),
            random_seed=int(os.getenv("SWE_SCHED_RANDOM_SEED", os.getenv("SWE_REPLAY_SEED", "0"))),
        )


@dataclass(slots=True)
class PromptAdmissionTicket:
    prompt_id: str
    repo: str
    rollout_id: int | None
    predicted: ResourceVector


@dataclass(slots=True)
class _PendingPrompt:
    prompt_id: str
    repo: str
    data_key: str
    rollout_id: int | None
    group_index: int | None
    sample_index: int | None
    predicted: ResourceVector
    avg_total_time_sec: float
    has_history: bool
    created_at: float
    admitted: bool = False
    blocked_log_ts: float = 0.0
    head_block_count: int = 0


@dataclass(slots=True)
class _LeaseResourceStats:
    sample_count: int = 0
    cpu_sample_count: int = 0
    peak_memory_bytes: float = 0.0
    avg_cpu_percent: float | None = None
    disk_read_bytes: float = 0.0
    disk_write_bytes: float = 0.0


@dataclass(slots=True)
class _ActivePrompt:
    prompt_id: str
    repo: str
    data_key: str
    rollout_id: int | None
    predicted: ResourceVector
    has_history: bool
    started_at: float
    sample_count: int = 0
    cpu_sample_count: int = 0
    latest_cpu_percent: float | None = None
    cpu_latest_by_lease: dict[str, float] = field(default_factory=dict)
    lease_stats_by_lease: dict[str, _LeaseResourceStats] = field(default_factory=dict)
    peak_memory_bytes: float = 0.0
    disk_read_bytes: float = 0.0
    disk_write_bytes: float = 0.0
    leases: set[str] = field(default_factory=set)
    lease_errors: dict[str, int] = field(default_factory=dict)


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def format_bytes(value: float | int) -> str:
    v = float(value)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if v < 1024 or unit == "TiB":
            return f"{v:.1f}{unit}" if unit != "B" else f"{int(v)}B"
        v /= 1024
    return f"{int(value)}B"


def infer_rollout_id(sample: Any, rollout_batch_size: int | None) -> int | None:
    if rollout_batch_size is None or rollout_batch_size <= 0:
        return None
    group_index = getattr(sample, "group_index", None)
    if group_index is None:
        return None
    return int(group_index) // int(rollout_batch_size)


def extract_repo_key(
    *,
    sample_metadata: dict | None = None,
    image_name: str | None = None,
    instance_id: str | None = None,
    fallback: str = "unknown",
) -> str:
    metadata = sample_metadata or {}

    # Priority 1: explicit repo fields in metadata.
    for key in ["repo", "repository", "repo_name"]:
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return _normalize_repo(value)

    instance = metadata.get("instance") if isinstance(metadata.get("instance"), dict) else {}
    for key in ["repo", "repository", "repo_name"]:
        value = instance.get(key)
        if isinstance(value, str) and value.strip():
            return _normalize_repo(value)

    # Priority 2: explicit instance_id.
    iid = instance_id or (instance.get("instance_id") if isinstance(instance, dict) else None)
    repo_from_iid = _repo_from_instance_id(iid)
    if repo_from_iid:
        return repo_from_iid

    # Priority 3: image name parsing.
    repo_from_image = _repo_from_image_name(image_name)
    if repo_from_image:
        return repo_from_image

    return fallback


def extract_data_key(
    *,
    sample_metadata: dict | None = None,
    image_name: str | None = None,
    instance_id: str | None = None,
    fallback: str = "unknown",
) -> str:
    metadata = sample_metadata or {}
    instance = metadata.get("instance") if isinstance(metadata.get("instance"), dict) else {}

    iid = instance_id or (instance.get("instance_id") if isinstance(instance, dict) else None)
    if isinstance(iid, str) and iid.strip():
        return iid.strip().lower()

    # Backward-compatible fallback when instance_id is unavailable.
    return extract_repo_key(
        sample_metadata=sample_metadata,
        image_name=image_name,
        instance_id=instance_id,
        fallback=fallback,
    )


def _normalize_repo(repo: str) -> str:
    text = repo.strip().strip("/")
    if "/" in text:
        return text.split("/")[-1].lower()
    return text.lower()


def _repo_from_instance_id(instance_id: str | None) -> str | None:
    if not instance_id or not isinstance(instance_id, str):
        return None
    if "__" in instance_id:
        parts = instance_id.split("__", 1)
        tail = parts[1]
    else:
        tail = instance_id

    if "-" in tail:
        maybe_repo = tail.rsplit("-", 1)[0]
    else:
        maybe_repo = tail

    maybe_repo = maybe_repo.strip("_").strip()
    if not maybe_repo:
        return None
    return maybe_repo.lower()


def _repo_from_image_name(image_name: str | None) -> str | None:
    if not image_name or not isinstance(image_name, str):
        return None

    # Typical: docker.io/swebench/sweb.eval.x86_64.pallets_s_sphinx-12345:latest
    core = image_name.split(":", 1)[0].rsplit(".", 1)[-1]
    core = core.replace("_1776_", "__").replace("_s_", "__")
    return _repo_from_instance_id(core)


def _data_key_from_prompt_id(prompt_id: str | None) -> str | None:
    """Recover stable instance-level key from prompt_id if possible.

    prompt_id format:
      <instance_id>__g<group_index>__i<sample_index>__<rand8>
    """
    if not prompt_id or not isinstance(prompt_id, str):
        return None
    marker = "__g"
    idx = prompt_id.find(marker)
    if idx <= 0:
        return None
    stem = prompt_id[:idx].strip().lower()
    if not stem or stem == "unknown":
        return None
    return stem


def _data_key_from_image_name(image_name: str | None) -> str | None:
    if not image_name or not isinstance(image_name, str):
        return None

    # For SWE images, tag usually encodes instance id:
    # registry/.../swe-rl:getmoto_s_moto-5258 -> getmoto__moto-5258
    tag = image_name.rsplit(":", 1)[-1].strip().lower() if ":" in image_name else ""
    if tag and tag != "latest":
        decoded = tag.replace("_1776_", "__").replace("_s_", "__")
        if decoded and decoded != "swe-rl":
            return decoded

    # Fallback to image basename.
    name = image_name.rsplit("/", 1)[-1].split(":", 1)[0].strip().lower()
    if not name:
        return None
    decoded = name.replace("_1776_", "__").replace("_s_", "__")
    if "." in decoded:
        decoded = decoded.rsplit(".", 1)[-1]
    return decoded or None


def _nested_float(payload: dict[str, Any], key: str, field: str, default: float = 0.0) -> float:
    try:
        value = payload.get(key, {})
        if isinstance(value, dict):
            return float(value.get(field, default) or default)
        return float(default)
    except Exception:
        return float(default)


def _update_aggregated_profile(
    aggregated: dict[str, RepoResourceProfile],
    *,
    data_key: str,
    peak_memory_bytes: float,
    avg_cpu_percent: float,
    avg_disk_read_bytes: float,
    avg_disk_write_bytes: float,
    duration_sec: float,
    ts: float,
) -> None:
    profile = aggregated.get(data_key)
    if profile is None:
        profile = RepoResourceProfile()
        aggregated[data_key] = profile

    n = profile.sample_count
    profile.sample_count = n + 1
    profile.peak_memory_bytes = max(profile.peak_memory_bytes, max(0.0, float(peak_memory_bytes)))
    profile.avg_cpu_percent = _rolling_avg(profile.avg_cpu_percent, n, max(0.0, float(avg_cpu_percent)))
    profile.avg_disk_read_bytes = _rolling_avg(
        profile.avg_disk_read_bytes, n, max(0.0, float(avg_disk_read_bytes))
    )
    profile.avg_disk_write_bytes = _rolling_avg(
        profile.avg_disk_write_bytes, n, max(0.0, float(avg_disk_write_bytes))
    )
    profile.avg_total_time_sec = _rolling_avg(profile.avg_total_time_sec, n, max(0.0, float(duration_sec)))
    profile.last_updated = max(profile.last_updated, float(ts))


class OnlineEnvDockerScheduler:
    def __init__(self, env_client: Any, config: SchedulerConfig):
        self.env_client = env_client
        self.config = config
        # NOTE: kept legacy name for compatibility; keys are now per-data data_key.
        self.repo_resource_stats: dict[str, RepoResourceProfile] = {}

        self._active_predicted = ResourceVector(0.0, 0.0, 0.0, 0.0)
        self._pending: list[_PendingPrompt] = []
        self._active_prompts: dict[str, _ActivePrompt] = {}
        self._lease_to_prompt: dict[str, str] = {}
        self._sampling_cursor = 0

        self._lock = asyncio.Lock()
        self._cond = asyncio.Condition(self._lock)
        self._sampler_task: asyncio.Task | None = None
        self._server_refresh_task: asyncio.Task | None = None
        self._stop_sampling = False
        self._round_logged: set[int] = set()
        self._scheduler_start_ts = time.time()
        self._resource_stats_last_refresh_ts = 0.0
        self._server_memory_available_bytes: float | None = None
        self._server_memory_total_bytes: float | None = None
        self._server_cpu_available_percent: float | None = None
        self._server_cpu_total_percent: float | None = None
        self._server_disk_read_available_bps: float | None = None
        self._server_disk_read_total_bps: float | None = None
        self._server_disk_write_available_bps: float | None = None
        self._server_disk_write_total_bps: float | None = None
        self._server_memory_last_refresh_ts = 0.0
        self._server_memory_refresh_lock = asyncio.Lock()
        self._admitted_since_resource_refresh = 0
        self._last_stall_log_ts = 0.0
        self._last_server_resource_log_ts = 0.0
        self._rng = random.Random(int(self.config.random_seed))

        self._profile_path = (
            Path(self.config.profile_json_path)
            if str(self.config.profile_json_path or "").strip()
            else None
        )
        loaded = 0
        if self.config.use_resource_stats_dir:
            loaded = self._refresh_resource_profiles(force=True)
        if loaded <= 0 and self._profile_path is not None:
            self._load_profiles()
        if self.config.verbose_logging:
            logger.info(
                "[SWE-SCHED][VERBOSE] init enabled={} preserve_order={} active_cap={} live_stats_polling={} live_profile_updates={} min_live_profile_samples={} min_profile_mem={} stats_round_cap={} stats_spacing={}s realtime(mem={},cpu={},disk={}) refresh={}s blocked_log={}s verbose_log={}s",
                self.config.enabled,
                self.config.preserve_prompt_order,
                self.config.max_active_prompts,
                not self.config.disable_live_stats_polling,
                self.config.enable_live_profile_updates,
                self.config.min_live_profile_samples,
                format_bytes(self.config.min_profile_memory_bytes),
                self.config.max_stats_requests_per_round,
                self.config.stats_request_spacing_sec,
                self.config.use_realtime_server_memory,
                self.config.use_realtime_server_cpu,
                self.config.use_realtime_server_disk,
                self.config.server_memory_refresh_sec,
                self.config.blocked_log_interval_sec,
                self.config.verbose_log_interval_sec,
            )

    def _use_realtime_server_budget(self) -> bool:
        return True

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    async def admit_prompt(
        self,
        *,
        sample: Any,
        image_name: str,
        rollout_batch_size: int | None,
    ) -> PromptAdmissionTicket:
        await self._ensure_server_refresh_task()
        metadata = sample.metadata if isinstance(getattr(sample, "metadata", None), dict) else {}
        iid = metadata.get("instance", {}).get("instance_id") if isinstance(metadata.get("instance"), dict) else None
        repo = extract_repo_key(sample_metadata=metadata, image_name=image_name, instance_id=iid)
        data_key = extract_data_key(sample_metadata=metadata, image_name=image_name, instance_id=iid)

        predicted, has_history, avg_total_time_sec = self._predict_for_key(data_key)
        prompt_id = self._build_prompt_id(sample, iid=iid)
        rollout_id = infer_rollout_id(sample, rollout_batch_size)

        request = _PendingPrompt(
            prompt_id=prompt_id,
            repo=repo,
            data_key=data_key,
            rollout_id=rollout_id,
            group_index=getattr(sample, "group_index", None),
            sample_index=getattr(sample, "index", None),
            predicted=predicted,
            avg_total_time_sec=avg_total_time_sec,
            has_history=has_history,
            created_at=time.time(),
        )

        async with self._cond:
            self._pending.append(request)
            if self.config.verbose_logging:
                logger.info(
                    "[SWE-SCHED][VERBOSE] enqueue prompt={} repo={} data_key={} rollout={} predicted={} source={} pending={} active={}",
                    request.prompt_id,
                    request.repo,
                    request.data_key,
                    request.rollout_id,
                    self._fmt_resource(request.predicted),
                    "history" if request.has_history else "default",
                    len(self._pending),
                    len(self._active_prompts),
                )

        blocked_log_interval = max(0.1, float(self.config.blocked_log_interval_sec))
        wait_timeout_sec = blocked_log_interval

        while True:
            async with self._cond:
                admitted = self._admit_next_unlocked()
                if admitted is not None:
                    self._cond.notify_all()

                if request.admitted:
                    active = self._active_prompts.get(prompt_id)
                    if active is None:
                        active = _ActivePrompt(
                            prompt_id=prompt_id,
                            repo=repo,
                            data_key=data_key,
                            rollout_id=rollout_id,
                            predicted=request.predicted,
                            has_history=request.has_history,
                            started_at=time.time(),
                        )
                        self._active_prompts[prompt_id] = active
                        logger.warning(
                            "[SWE-SCHED] repaired missing active prompt record prompt={} repo={} rollout={}",
                            prompt_id,
                            repo,
                            rollout_id,
                        )
                    admitted_predicted = active.predicted
                    logger.info(
                        "[SWE-SCHED] admitted prompt={} repo={} rollout={} predicted(mem={},cpu={:.1f}%,r={},w={}) active(mem={},cpu={:.1f}%)",
                        prompt_id,
                        repo,
                        rollout_id,
                        format_bytes(admitted_predicted.memory_bytes),
                        admitted_predicted.cpu_percent,
                        format_bytes(admitted_predicted.disk_read_bytes),
                        format_bytes(admitted_predicted.disk_write_bytes),
                        format_bytes(self._active_predicted.memory_bytes),
                        self._active_predicted.cpu_percent,
                    )
                    if self.config.verbose_logging:
                        logger.info(
                            "[SWE-SCHED][VERBOSE] admit snapshot prompt={} {}",
                            prompt_id,
                            self._budget_snapshot_unlocked(request=request.predicted),
                        )
                    return PromptAdmissionTicket(
                        prompt_id=prompt_id,
                        repo=repo,
                        rollout_id=rollout_id,
                        predicted=admitted_predicted,
                    )

                now = time.time()
                if now - request.blocked_log_ts >= blocked_log_interval:
                    request.blocked_log_ts = now
                    reason = self._build_budget_block_reason(request.predicted, has_history=request.has_history)
                    if self.config.verbose_logging:
                        logger.info(
                            "[SWE-SCHED] delayed prompt={} repo={} rollout={} reason={} req={} {}",
                            request.prompt_id,
                            request.repo,
                            request.rollout_id,
                            reason,
                            self._fmt_resource(request.predicted),
                            self._budget_snapshot_unlocked(request=request.predicted),
                        )
                    else:
                        logger.info(
                            "[SWE-SCHED] delayed prompt={} repo={} rollout={} reason={}",
                            request.prompt_id,
                            request.repo,
                            request.rollout_id,
                            reason,
                        )

                try:
                    await asyncio.wait_for(self._cond.wait(), timeout=wait_timeout_sec)
                except asyncio.TimeoutError:
                    pass

    async def attach_lease(
        self,
        *,
        prompt_id: str,
        lease_id: str,
    ) -> None:
        if self.config.disable_live_stats_polling:
            return
        if not lease_id:
            return
        async with self._lock:
            active = self._active_prompts.get(prompt_id)
            if active is None:
                return
            active.leases.add(lease_id)
            self._lease_to_prompt[lease_id] = prompt_id
            self._ensure_sampler_locked()

    async def detach_lease(self, lease_id: str, *, reason: str = "") -> None:
        if self.config.disable_live_stats_polling:
            return
        if not lease_id:
            return
        async with self._lock:
            prompt_id = self._lease_to_prompt.pop(lease_id, None)
            if prompt_id is None:
                return
            active = self._active_prompts.get(prompt_id)
            if active is not None:
                active.leases.discard(lease_id)
                active.lease_errors.pop(lease_id, None)
        if reason:
            logger.info("[SWE-SCHED] detached lease={} reason={}", lease_id, reason)

    async def finish_prompt(self, prompt_id: str) -> PromptResourceSummary | None:
        async with self._cond:
            active = self._active_prompts.pop(prompt_id, None)
            if active is None:
                return None

            for lease_id in list(active.leases):
                self._lease_to_prompt.pop(lease_id, None)
            active.leases.clear()

            self._active_predicted = self._active_predicted.minus(active.predicted)
            self._cond.notify_all()

            sample_count = active.sample_count
            if sample_count <= 0:
                # Fall back to admission prediction if stats were unavailable.
                peak_memory_bytes = active.predicted.memory_bytes
                avg_cpu_percent = active.predicted.cpu_percent
                disk_read_bytes = active.predicted.disk_read_bytes
                disk_write_bytes = active.predicted.disk_write_bytes
                sample_count = 1
                lease_count = 0
            else:
                lease_stats = [
                    item
                    for item in active.lease_stats_by_lease.values()
                    if item.sample_count > 0
                ]
                lease_count = len(lease_stats)
                if lease_stats:
                    peak_memory_bytes = sum(item.peak_memory_bytes for item in lease_stats) / len(lease_stats)
                    disk_read_bytes = sum(item.disk_read_bytes for item in lease_stats) / len(lease_stats)
                    disk_write_bytes = sum(item.disk_write_bytes for item in lease_stats) / len(lease_stats)
                    cpu_values = [
                        float(item.avg_cpu_percent)
                        for item in lease_stats
                        if item.cpu_sample_count > 0 and item.avg_cpu_percent is not None
                    ]
                    if cpu_values:
                        avg_cpu_percent = sum(cpu_values) / len(cpu_values)
                    elif active.latest_cpu_percent is not None:
                        avg_cpu_percent = active.latest_cpu_percent
                    else:
                        avg_cpu_percent = active.predicted.cpu_percent
                else:
                    peak_memory_bytes = active.peak_memory_bytes
                    if active.latest_cpu_percent is not None:
                        avg_cpu_percent = active.latest_cpu_percent
                    elif active.cpu_latest_by_lease:
                        avg_cpu_percent = sum(active.cpu_latest_by_lease.values()) / len(active.cpu_latest_by_lease)
                    else:
                        avg_cpu_percent = active.predicted.cpu_percent
                    disk_read_bytes = active.disk_read_bytes
                    disk_write_bytes = active.disk_write_bytes

            finished_at = time.time()
            total_time_sec = max(0.0, finished_at - active.started_at)
            summary = PromptResourceSummary(
                prompt_id=prompt_id,
                repo=active.repo,
                data_key=active.data_key,
                sample_count=sample_count,
                peak_memory_bytes=peak_memory_bytes,
                avg_cpu_percent=avg_cpu_percent,
                disk_read_bytes=disk_read_bytes,
                disk_write_bytes=disk_write_bytes,
                started_at=active.started_at,
                finished_at=finished_at,
                total_time_sec=total_time_sec,
                cpu_sample_count=active.cpu_sample_count,
                lease_count=lease_count,
            )

            if self.config.enable_live_profile_updates:
                self._update_repo_profile_unlocked(summary)
                self._refresh_pending_predictions_for_key_unlocked(active.data_key)
            self._reset_admission_window_unlocked()
            self._cond.notify_all()
            if self.config.enable_live_profile_updates:
                self._persist_profiles_unlocked()

            logger.info(
                "[SWE-SCHED] prompt summary prompt={} repo={} peak_mem={} avg_cpu={:.2f}% disk(r={},w={}) elapsed={:.1f}s samples={} cpu_samples={} leases={}",
                summary.prompt_id,
                summary.repo,
                format_bytes(summary.peak_memory_bytes),
                summary.avg_cpu_percent,
                format_bytes(summary.disk_read_bytes),
                format_bytes(summary.disk_write_bytes),
                summary.total_time_sec,
                summary.sample_count,
                summary.cpu_sample_count,
                summary.lease_count,
            )
            return summary

    async def close(self) -> None:
        async with self._lock:
            self._stop_sampling = True
            task = self._sampler_task
            self._sampler_task = None
            refresh_task = self._server_refresh_task
            self._server_refresh_task = None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        if refresh_task is not None:
            refresh_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await refresh_task

    def get_repo_resource_stats(self) -> dict[str, dict[str, Any]]:
        self._refresh_resource_profiles()
        out = {}
        for data_key, profile in self.repo_resource_stats.items():
            out[data_key] = {
                "sample_count": profile.sample_count,
                "peak_memory_bytes": profile.peak_memory_bytes,
                "avg_cpu_percent": profile.avg_cpu_percent,
                "avg_disk_read_bytes": profile.avg_disk_read_bytes,
                "avg_disk_write_bytes": profile.avg_disk_write_bytes,
                "avg_total_time_sec": profile.avg_total_time_sec,
                "last_updated": profile.last_updated,
            }
        return out

    def record_prompt_summary(self, summary: PromptResourceSummary) -> None:
        self._update_repo_profile_unlocked(summary)
        self._persist_profiles_unlocked()

    def predict_resources_for_repo(self, repo: str) -> ResourceVector:
        return self._predict_for_key(repo)[0]

    def plan_prompt_order(self, prompt_specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Plan prompt order with resource-constrained greedy packing.

        Args:
            prompt_specs: each item should include `repo` and optional
                `group_index`, `sample_index`.
        """
        now = time.time()
        budget = self._effective_budget()
        remaining: list[tuple[dict[str, Any], _PendingPrompt]] = []
        for idx, spec in enumerate(prompt_specs):
            repo = str(spec.get("repo", "unknown"))
            data_key = str(spec.get("data_key") or spec.get("instance_id") or repo)
            predicted, has_history, avg_total_time_sec = self._predict_for_key(data_key)
            pending = _PendingPrompt(
                prompt_id=f"plan-{idx}",
                repo=repo,
                data_key=data_key,
                rollout_id=spec.get("rollout_id"),
                group_index=spec.get("group_index"),
                sample_index=spec.get("sample_index"),
                predicted=predicted,
                avg_total_time_sec=avg_total_time_sec,
                has_history=has_history,
                created_at=now + idx * 1e-3,
            )
            remaining.append((spec, pending))

        ordered: list[dict[str, Any]] = []
        virtual_active = ResourceVector(0, 0.0, 0, 0)

        while remaining:
            residual = budget.minus(virtual_active)
            fit = [
                (spec, req)
                for spec, req in remaining
                if self._fits_virtual(virtual_active, req.predicted, budget)
            ]
            if not fit:
                # Start a new wave if current wave is full.
                virtual_active = ResourceVector(0, 0.0, 0, 0)
                residual = budget
                fit = [
                    (spec, req)
                    for spec, req in remaining
                    if self._fits_virtual(virtual_active, req.predicted, budget)
                ]
                if not fit:
                    fit = [min(remaining, key=lambda x: x[1].created_at)]

            fit.sort(key=lambda x: self._packing_score(x[1], residual, budget), reverse=True)
            chosen_spec, chosen_req = fit[0]
            ordered.append(chosen_spec)
            remaining.remove((chosen_spec, chosen_req))
            if self._fits_virtual(virtual_active, chosen_req.predicted, budget):
                virtual_active = virtual_active.plus(chosen_req.predicted)

        return ordered

    def _predict_for_key(self, data_key: str) -> tuple[ResourceVector, bool, float]:
        self._refresh_resource_profiles()
        profile = self.repo_resource_stats.get(data_key)
        if profile is None or profile.sample_count <= 0:
            mem_mul = max(1.0, float(self.config.cold_start_memory_multiplier))
            cpu_mul = max(1.0, float(self.config.cold_start_cpu_multiplier))
            return (
                ResourceVector(
                    memory_bytes=max(1, float(self.config.default_memory_bytes * mem_mul)),
                    cpu_percent=max(0.1, float(self.config.default_cpu_percent * cpu_mul)),
                    disk_read_bytes=self.config.default_disk_read_bytes,
                    disk_write_bytes=self.config.default_disk_write_bytes,
                ),
                False,
                0.0,
            )

        mem_mul = max(1.0, float(self.config.cold_start_memory_multiplier))
        cpu_mul = max(1.0, float(self.config.cold_start_cpu_multiplier))
        min_profile_memory = max(0.0, float(self.config.min_profile_memory_bytes))
        profile_prediction = ResourceVector(
            memory_bytes=max(
                1.0,
                min_profile_memory,
                float(profile.peak_memory_bytes)
                * min(1.0, max(0.0, float(self.config.memory_peak_scale))),
            ),
            cpu_percent=max(0.1, float(profile.avg_cpu_percent)),
            disk_read_bytes=max(1.0, float(profile.avg_disk_read_bytes)),
            disk_write_bytes=max(1.0, float(profile.avg_disk_write_bytes)),
        )

        min_samples = max(0, int(self.config.min_live_profile_samples))
        if profile.sample_count < min_samples:
            return (
                ResourceVector(
                    memory_bytes=max(
                        profile_prediction.memory_bytes,
                        float(self.config.default_memory_bytes * mem_mul),
                    ),
                    cpu_percent=max(
                        profile_prediction.cpu_percent,
                        float(self.config.default_cpu_percent * cpu_mul),
                    ),
                    disk_read_bytes=max(
                        profile_prediction.disk_read_bytes,
                        float(self.config.default_disk_read_bytes),
                    ),
                    disk_write_bytes=max(
                        profile_prediction.disk_write_bytes,
                        float(self.config.default_disk_write_bytes),
                    ),
                ),
                False,
                max(0.0, float(profile.avg_total_time_sec)),
            )

        return (
            profile_prediction,
            True,
            max(0.0, float(profile.avg_total_time_sec)),
        )

    def _refresh_pending_predictions_for_key_unlocked(self, data_key: str) -> None:
        predicted, has_history, avg_total_time_sec = self._predict_for_key(data_key)
        remaining = max(1, int(self.config.max_requests_per_resource_update))
        for req in self._pending:
            if req.data_key != data_key:
                continue
            req.predicted = predicted
            req.has_history = has_history
            req.avg_total_time_sec = avg_total_time_sec
            req.blocked_log_ts = 0.0
            req.head_block_count = 0
            remaining -= 1
            if remaining <= 0:
                break

    def _build_prompt_id(self, sample: Any, *, iid: str | None) -> str:
        gid = getattr(sample, "group_index", None)
        sid = getattr(sample, "index", None)
        stem = iid or "unknown"
        return f"{stem}__g{gid if gid is not None else 'na'}__i{sid if sid is not None else 'na'}__{uuid.uuid4().hex[:8]}"

    def _effective_budget(self) -> ResourceVector:
        m = max(0.01, self.config.scheduler_safety_margin)
        memory_oversell = max(1.0, float(self.config.memory_oversell_ratio))
        cpu_oversell = max(1.0, float(self.config.cpu_oversell_ratio))
        disk_read_oversell = max(1.0, float(self.config.disk_read_oversell_ratio))
        disk_write_oversell = max(1.0, float(self.config.disk_write_oversell_ratio))
        memory_budget_bytes = (
            max(1.0, float(self._server_memory_available_bytes))
            if self.config.use_realtime_server_memory and self._server_memory_available_bytes is not None
            else (0.0 if self.config.use_realtime_server_memory else float(_UNBOUNDED_RESOURCE_BYTES))
        )
        cpu_budget_percent = (
            max(0.1, float(self._server_cpu_available_percent))
            if self.config.use_realtime_server_cpu and self._server_cpu_available_percent is not None
            else (0.0 if self.config.use_realtime_server_cpu else _UNBOUNDED_CPU_PERCENT)
        )
        disk_read_budget_bytes = (
            max(1.0, float(self._server_disk_read_available_bps))
            if self.config.use_realtime_server_disk and self._server_disk_read_available_bps is not None
            else (0.0 if self.config.use_realtime_server_disk else float(_UNBOUNDED_RESOURCE_BYTES))
        )
        disk_write_budget_bytes = (
            max(1.0, float(self._server_disk_write_available_bps))
            if self.config.use_realtime_server_disk and self._server_disk_write_available_bps is not None
            else (0.0 if self.config.use_realtime_server_disk else float(_UNBOUNDED_RESOURCE_BYTES))
        )
        return ResourceVector(
            memory_bytes=memory_budget_bytes * m * memory_oversell,
            cpu_percent=cpu_budget_percent * m * cpu_oversell,
            disk_read_bytes=disk_read_budget_bytes * m * disk_read_oversell,
            disk_write_bytes=disk_write_budget_bytes * m * disk_write_oversell,
        )

    @staticmethod
    def _fmt_resource(v: ResourceVector) -> str:
        return (
            f"mem={format_bytes(v.memory_bytes)},"
            f"cpu={v.cpu_percent:.1f}%,"
            f"r={format_bytes(v.disk_read_bytes)},"
            f"w={format_bytes(v.disk_write_bytes)}"
        )

    def _budget_snapshot_unlocked(self, request: ResourceVector | None = None) -> str:
        budget = self._effective_budget()
        active = self._active_predicted
        active_for_budget = self._active_for_budget(active)
        residual = budget.minus(active_for_budget)
        req = request or ResourceVector(0, 0.0, 0, 0)
        return (
            f"active[{self._fmt_resource(active)}] "
            f"active_for_budget[{self._fmt_resource(active_for_budget)}] "
            f"req[{self._fmt_resource(req)}] "
            f"residual[{self._fmt_resource(residual)}] "
            f"budget[{self._fmt_resource(budget)}] "
            f"pending={len(self._pending)} active_prompts={len(self._active_prompts)}"
        )

    def _active_for_budget(self, active: ResourceVector) -> ResourceVector:
        scaled_active = self._scaled_active_for_budget_unlocked(active)
        return ResourceVector(
            memory_bytes=(
                scaled_active.memory_bytes
                if self.config.use_realtime_server_memory
                else active.memory_bytes
            ),
            cpu_percent=(
                scaled_active.cpu_percent
                if self.config.use_realtime_server_cpu
                else active.cpu_percent
            ),
            disk_read_bytes=(
                scaled_active.disk_read_bytes
                if self.config.use_realtime_server_disk
                else active.disk_read_bytes
            ),
            disk_write_bytes=(
                scaled_active.disk_write_bytes
                if self.config.use_realtime_server_disk
                else active.disk_write_bytes
            ),
        )

    def _scaled_active_for_budget_unlocked(self, active: ResourceVector) -> ResourceVector:
        # Use scaled full active reservations as a conservative hedge on top of the
        # realtime remaining resources: effective_budget = remaining * oversell - active.
        discount = max(0.0, float(self.config.realtime_local_active_discount))
        if discount == 1.0:
            return active
        return ResourceVector(
            memory_bytes=max(0.0, active.memory_bytes * discount),
            cpu_percent=max(0.0, active.cpu_percent * discount),
            disk_read_bytes=max(0.0, active.disk_read_bytes * discount),
            disk_write_bytes=max(0.0, active.disk_write_bytes * discount),
        )

    async def _refresh_server_memory_budget_if_needed(self, force: bool = False) -> bool:
        if not self._use_realtime_server_budget():
            return False

        now = time.time()
        refresh_interval = max(1.0, float(self.config.server_memory_refresh_sec))
        if not force and now - self._server_memory_last_refresh_ts < refresh_interval:
            return False

        async with self._server_memory_refresh_lock:
            now = time.time()
            if not force and now - self._server_memory_last_refresh_ts < refresh_interval:
                return False

            base_url = str(getattr(self.env_client, "base_url", "") or "").rstrip("/")
            if not base_url:
                self._server_memory_last_refresh_ts = now
                return False

            try:
                prev_available = self._server_memory_available_bytes
                prev_total = self._server_memory_total_bytes
                prev_cpu_available = self._server_cpu_available_percent
                prev_cpu_total = self._server_cpu_total_percent
                prev_disk_read_available = self._server_disk_read_available_bps
                prev_disk_read_total = self._server_disk_read_total_bps
                prev_disk_write_available = self._server_disk_write_available_bps
                prev_disk_write_total = self._server_disk_write_total_bps
                status_fn = getattr(self.env_client, "status", None)
                if callable(status_fn):
                    payload = await status_fn()
                else:
                    payload = await asyncio.to_thread(_get_json_blocking, f"{base_url}/status", 30.0)
                if not isinstance(payload, dict):
                    raise RuntimeError(f"status returned non-dict payload: {payload!r}")
                pool = payload.get("pool", {})
                if not isinstance(pool, dict):
                    raise RuntimeError(f"status payload missing dict pool field: {payload!r}")
                if "cluster_memory_available_bytes" in pool:
                    self._server_memory_available_bytes = max(
                        0.0, float(pool.get("cluster_memory_available_bytes", 0.0) or 0.0)
                    )
                if "cluster_memory_total_bytes" in pool:
                    self._server_memory_total_bytes = max(
                        0.0, float(pool.get("cluster_memory_total_bytes", 0.0) or 0.0)
                    )
                if "cluster_cpu_available_percent" in pool:
                    self._server_cpu_available_percent = max(
                        0.0, float(pool.get("cluster_cpu_available_percent", 0.0) or 0.0)
                    )
                if "cluster_cpu_total_percent" in pool:
                    self._server_cpu_total_percent = max(0.0, float(pool.get("cluster_cpu_total_percent", 0.0) or 0.0))
                if "cluster_disk_read_available_bytes_per_sec" in pool:
                    self._server_disk_read_available_bps = max(
                        0.0, float(pool.get("cluster_disk_read_available_bytes_per_sec", 0.0) or 0.0)
                    )
                if "cluster_disk_read_total_bytes_per_sec" in pool:
                    self._server_disk_read_total_bps = max(
                        0.0, float(pool.get("cluster_disk_read_total_bytes_per_sec", 0.0) or 0.0)
                    )
                if "cluster_disk_write_available_bytes_per_sec" in pool:
                    self._server_disk_write_available_bps = max(
                        0.0, float(pool.get("cluster_disk_write_available_bytes_per_sec", 0.0) or 0.0)
                    )
                if "cluster_disk_write_total_bytes_per_sec" in pool:
                    self._server_disk_write_total_bps = max(
                        0.0, float(pool.get("cluster_disk_write_total_bytes_per_sec", 0.0) or 0.0)
                    )
                self._server_memory_last_refresh_ts = now
                if self.config.verbose_logging:
                    changed = (
                        prev_available != self._server_memory_available_bytes
                        or prev_total != self._server_memory_total_bytes
                        or prev_cpu_available != self._server_cpu_available_percent
                        or prev_cpu_total != self._server_cpu_total_percent
                        or prev_disk_read_available != self._server_disk_read_available_bps
                        or prev_disk_read_total != self._server_disk_read_total_bps
                        or prev_disk_write_available != self._server_disk_write_available_bps
                        or prev_disk_write_total != self._server_disk_write_total_bps
                    )
                    if changed or (
                        now - self._last_server_resource_log_ts
                        >= max(1.0, float(self.config.server_memory_refresh_sec))
                    ):
                        self._last_server_resource_log_ts = now
                        logger.info(
                            (
                                "[SWE-SCHED][VERBOSE] server resources refreshed "
                                "mem(avail={},total={}) cpu(avail={:.1f}%,total={:.1f}%) "
                                "disk_r(avail={},total={}) disk_w(avail={},total={}) ts={:.0f}"
                            ),
                            format_bytes(self._server_memory_available_bytes or 0),
                            format_bytes(self._server_memory_total_bytes or 0),
                            float(self._server_cpu_available_percent or 0.0),
                            float(self._server_cpu_total_percent or 0.0),
                            format_bytes(self._server_disk_read_available_bps or 0),
                            format_bytes(self._server_disk_read_total_bps or 0),
                            format_bytes(self._server_disk_write_available_bps or 0),
                            format_bytes(self._server_disk_write_total_bps or 0),
                            self._server_memory_last_refresh_ts,
                        )
                return True
            except Exception as e:
                self._server_memory_last_refresh_ts = now
                logger.warning(f"[SWE-SCHED] refresh server memory failed: {e}")
                return False

    async def _ensure_server_refresh_task(self) -> None:
        if not self._use_realtime_server_budget():
            return
        async with self._lock:
            task = self._server_refresh_task
            if task is not None and not task.done():
                return
            self._server_refresh_task = asyncio.create_task(
                self._background_server_refresh_loop(),
                name="swe_sched_server_refresh",
            )

    async def _background_server_refresh_loop(self) -> None:
        refresh_interval = max(1.0, float(self.config.server_memory_refresh_sec))
        try:
            while True:
                refreshed = await self._refresh_server_memory_budget_if_needed(force=True)
                async with self._cond:
                    if refreshed:
                        self._reset_admission_window_unlocked()
                    self._cond.notify_all()
                await asyncio.sleep(refresh_interval)
        except asyncio.CancelledError:
            raise

    def _startup_active_cap_unlocked(self) -> int:
        cap = int(self.config.startup_max_active_prompts)
        if cap <= 0:
            return 0
        duration = max(0.0, float(self.config.startup_cap_duration_sec))
        if duration <= 0:
            return 0
        if time.time() - self._scheduler_start_ts <= duration:
            return cap
        return 0

    def _admit_next_unlocked(self) -> _PendingPrompt | None:
        if not self._pending:
            return None

        request_cap = max(0, int(self.config.max_requests_per_resource_update))
        if request_cap > 0 and self._admitted_since_resource_refresh >= request_cap:
            return None

        active_cap = max(0, int(self.config.max_active_prompts))
        if active_cap > 0 and len(self._active_prompts) >= active_cap:
            return None

        startup_cap = self._startup_active_cap_unlocked()
        if startup_cap > 0 and len(self._active_prompts) >= startup_cap:
            return None

        budget = self._effective_budget()
        active_for_budget = self._active_for_budget(self._active_predicted)
        residual = budget.minus(active_for_budget)
        unknown_cap = int(self.config.max_unknown_repo_concurrency)
        active_unknown = sum(1 for x in self._active_prompts.values() if not x.has_history)

        if self.config.preserve_prompt_order:
            chosen = self._admit_head_of_line_unlocked(
                budget=budget,
                active_for_budget=active_for_budget,
                unknown_cap=unknown_cap,
                active_unknown=active_unknown,
            )
            if chosen is None and self.config.verbose_logging:
                now = time.time()
                if now - self._last_stall_log_ts >= max(0.1, float(self.config.verbose_log_interval_sec)):
                    self._last_stall_log_ts = now
                    logger.info(
                        "[SWE-SCHED][VERBOSE] head-of-line waiting {}",
                        self._budget_snapshot_unlocked(),
                    )
            return chosen

        candidates: list[tuple[float, _PendingPrompt]] = []
        oversized_waiting: list[_PendingPrompt] = []
        for req in self._pending:
            if not req.has_history and unknown_cap > 0 and active_unknown >= unknown_cap:
                continue
            if self._fits(req.predicted, budget):
                score = self._packing_score(req, residual, budget)
                candidates.append((score, req))
            elif (
                active_for_budget.memory_bytes == 0
                and active_for_budget.cpu_percent == 0
                and active_for_budget.disk_read_bytes == 0
                and active_for_budget.disk_write_bytes == 0
                and self._is_oversized(req.predicted, budget)
            ):
                oversized_waiting.append(req)

        if not candidates and oversized_waiting:
            # Never deadlock the pipeline on a single heavy task.
            chosen = min(oversized_waiting, key=lambda x: x.created_at)
            logger.warning(
                "[SWE-SCHED] forcing oversized prompt={} repo={} predicted(mem={},cpu={:.1f}%,r={},w={}) budget(mem={},cpu={:.1f}%,r={},w={})",
                chosen.prompt_id,
                chosen.repo,
                format_bytes(chosen.predicted.memory_bytes),
                chosen.predicted.cpu_percent,
                format_bytes(chosen.predicted.disk_read_bytes),
                format_bytes(chosen.predicted.disk_write_bytes),
                format_bytes(budget.memory_bytes),
                budget.cpu_percent,
                format_bytes(budget.disk_read_bytes),
                format_bytes(budget.disk_write_bytes),
            )
            self._mark_admitted(chosen)
            return chosen

        if not candidates:
            if self.config.verbose_logging:
                now = time.time()
                if now - self._last_stall_log_ts >= max(0.1, float(self.config.verbose_log_interval_sec)):
                    self._last_stall_log_ts = now
                    logger.info(
                        "[SWE-SCHED][VERBOSE] no candidate admitted (unknown_cap={} active_unknown={}) {}",
                        unknown_cap,
                        active_unknown,
                        self._budget_snapshot_unlocked(),
                    )
            return None

        candidates.sort(key=lambda x: x[0], reverse=True)
        chosen = candidates[0][1]

        if chosen.rollout_id is not None and chosen.rollout_id not in self._round_logged:
            self._round_logged.add(chosen.rollout_id)
            self._log_round_reorder_unlocked(chosen.rollout_id, budget)

        self._mark_admitted(chosen)
        return chosen

    def _admit_head_of_line_unlocked(
        self,
        *,
        budget: ResourceVector,
        active_for_budget: ResourceVector,
        unknown_cap: int,
        active_unknown: int,
    ) -> _PendingPrompt | None:
        while self._pending:
            req = self._pending[0]
            if not req.has_history and unknown_cap > 0 and active_unknown >= unknown_cap:
                return None

            if self._fits(req.predicted, budget):
                req.head_block_count = 0
                self._mark_admitted(req)
                return req

            if (
                active_for_budget.memory_bytes == 0
                and active_for_budget.cpu_percent == 0
                and active_for_budget.disk_read_bytes == 0
                and active_for_budget.disk_write_bytes == 0
                and self._is_oversized(req.predicted, budget)
            ):
                logger.warning(
                    "[SWE-SCHED] forcing oversized prompt={} repo={} predicted(mem={},cpu={:.1f}%,r={},w={}) budget(mem={},cpu={:.1f}%,r={},w={})",
                    req.prompt_id,
                    req.repo,
                    format_bytes(req.predicted.memory_bytes),
                    req.predicted.cpu_percent,
                    format_bytes(req.predicted.disk_read_bytes),
                    format_bytes(req.predicted.disk_write_bytes),
                    format_bytes(budget.memory_bytes),
                    budget.cpu_percent,
                    format_bytes(budget.disk_read_bytes),
                    format_bytes(budget.disk_write_bytes),
                )
                req.head_block_count = 0
                self._mark_admitted(req)
                return req

            req.head_block_count += 1
            requeue_threshold = max(0, int(self.config.head_block_requeue_threshold))
            requeue_offset = max(1, int(self.config.head_block_requeue_offset))
            if requeue_threshold > 0 and req.head_block_count >= requeue_threshold and len(self._pending) > 1:
                self._defer_head_request_unlocked(req, requeue_offset)
                continue

            return None

        return None

    def _defer_head_request_unlocked(self, req: _PendingPrompt, offset: int) -> None:
        # Give smaller/fittable prompts a chance when the head item repeatedly hits the budget ceiling.
        req.head_block_count = 0
        req.blocked_log_ts = 0.0
        head = self._pending.pop(0)
        insert_idx = min(max(1, int(offset)), len(self._pending))
        self._pending.insert(insert_idx, head)
        logger.info(
            "[SWE-SCHED] requeue head prompt={} repo={} shift={} pending={} reason=repeated_budget_block",
            req.prompt_id,
            req.repo,
            insert_idx,
            len(self._pending),
        )

    def _mark_admitted(self, req: _PendingPrompt) -> None:
        if req.admitted:
            return
        req.admitted = True
        self._pending = [x for x in self._pending if x.prompt_id != req.prompt_id]
        self._active_predicted = self._active_predicted.plus(req.predicted)
        self._active_prompts[req.prompt_id] = _ActivePrompt(
            prompt_id=req.prompt_id,
            repo=req.repo,
            data_key=req.data_key,
            rollout_id=req.rollout_id,
            predicted=req.predicted,
            has_history=req.has_history,
            started_at=time.time(),
        )
        self._admitted_since_resource_refresh += 1

    def _reset_admission_window_unlocked(self) -> None:
        self._admitted_since_resource_refresh = 0

    def _fits(self, request: ResourceVector, budget: ResourceVector) -> bool:
        active_for_budget = self._active_for_budget(self._active_predicted)
        return (
            active_for_budget.memory_bytes + request.memory_bytes <= budget.memory_bytes
            and active_for_budget.cpu_percent + request.cpu_percent <= budget.cpu_percent
            and active_for_budget.disk_read_bytes + request.disk_read_bytes <= budget.disk_read_bytes
            and active_for_budget.disk_write_bytes + request.disk_write_bytes <= budget.disk_write_bytes
        )

    def _is_oversized(self, request: ResourceVector, budget: ResourceVector) -> bool:
        return (
            request.memory_bytes > budget.memory_bytes
            or request.cpu_percent > budget.cpu_percent
            or request.disk_read_bytes > budget.disk_read_bytes
            or request.disk_write_bytes > budget.disk_write_bytes
        )

    def _packing_score(self, request: _PendingPrompt, residual: ResourceVector, budget: ResourceVector) -> float:
        # Resource-constrained greedy packing score:
        # 1) fill current residual hole; 2) consider dominant resource; 3) mild aging.
        fill_mem = request.predicted.memory_bytes / max(1.0, float(residual.memory_bytes))
        fill_cpu = request.predicted.cpu_percent / max(1.0, residual.cpu_percent)
        fill_r = request.predicted.disk_read_bytes / max(1.0, float(residual.disk_read_bytes))
        fill_w = request.predicted.disk_write_bytes / max(1.0, float(residual.disk_write_bytes))

        dom = max(
            request.predicted.memory_bytes / max(1.0, float(budget.memory_bytes)),
            request.predicted.cpu_percent / max(1.0, budget.cpu_percent),
            request.predicted.disk_read_bytes / max(1.0, float(budget.disk_read_bytes)),
            request.predicted.disk_write_bytes / max(1.0, float(budget.disk_write_bytes)),
        )

        age_bonus = min(1.0, max(0.0, (time.time() - request.created_at) / 20.0))
        duration_ref = max(1.0, float(self.config.duration_priority_ref_sec))
        duration_weight = max(0.0, float(self.config.duration_priority_weight))
        duration_bonus = min(2.0, max(0.0, float(request.avg_total_time_sec)) / duration_ref)
        unknown_penalty = 0.08 if not request.has_history else 0.0
        return (
            (fill_mem + fill_cpu + fill_r + fill_w)
            + 0.6 * dom
            + 0.4 * age_bonus
            # + duration_weight * duration_bonus
            # - unknown_penalty
        )

    def _build_budget_block_reason(self, request: ResourceVector, *, has_history: bool) -> str:
        request_cap = max(0, int(self.config.max_requests_per_resource_update))
        if request_cap > 0 and self._admitted_since_resource_refresh >= request_cap:
            return (
                "resource_refresh_request_cap "
                f"{self._admitted_since_resource_refresh}>={request_cap}"
            )

        active_cap = max(0, int(self.config.max_active_prompts))
        if active_cap > 0 and len(self._active_prompts) >= active_cap:
            return f"active_prompt_cap {len(self._active_prompts)}>={active_cap}"

        startup_cap = self._startup_active_cap_unlocked()
        if startup_cap > 0 and len(self._active_prompts) >= startup_cap:
            elapsed = time.time() - self._scheduler_start_ts
            return f"startup_active_cap {len(self._active_prompts)}>={startup_cap} elapsed={elapsed:.1f}s"

        unknown_cap = int(self.config.max_unknown_repo_concurrency)
        if not has_history and unknown_cap > 0:
            active_unknown = sum(1 for x in self._active_prompts.values() if not x.has_history)
            if active_unknown >= unknown_cap:
                return f"unknown_repo_concurrency {active_unknown}>={unknown_cap}"

        budget = self._effective_budget()
        active_for_budget = self._active_for_budget(self._active_predicted)
        checks = [
            (
                "memory",
                active_for_budget.memory_bytes + request.memory_bytes,
                budget.memory_bytes,
                format_bytes,
            ),
            (
                "cpu",
                active_for_budget.cpu_percent + request.cpu_percent,
                budget.cpu_percent,
                lambda x: f"{x:.1f}%",
            ),
            (
                "disk_read",
                active_for_budget.disk_read_bytes + request.disk_read_bytes,
                budget.disk_read_bytes,
                format_bytes,
            ),
            (
                "disk_write",
                active_for_budget.disk_write_bytes + request.disk_write_bytes,
                budget.disk_write_bytes,
                format_bytes,
            ),
        ]
        failed = []
        for name, req_total, limit, formatter in checks:
            if req_total > limit:
                failed.append(f"{name} {formatter(req_total)}>{formatter(limit)}")
        return ", ".join(failed) if failed else "waiting_for_admission"

    def _log_round_reorder_unlocked(self, rollout_id: int, budget: ResourceVector) -> None:
        round_pending = [x for x in self._pending if x.rollout_id == rollout_id]
        if not round_pending:
            return

        virtual_active = self._active_predicted
        remaining = list(round_pending)
        plan: list[_PendingPrompt] = []
        while remaining:
            residual = budget.minus(virtual_active)
            fit = [x for x in remaining if self._fits_virtual(virtual_active, x.predicted, budget)]
            if not fit:
                # put the oldest one to avoid no-progress in planning output
                fit = [min(remaining, key=lambda x: x.created_at)]
            fit.sort(key=lambda x: self._packing_score(x, residual, budget), reverse=True)
            chosen = fit[0]
            plan.append(chosen)
            remaining.remove(chosen)
            virtual_active = virtual_active.plus(chosen.predicted)

        short_plan = [f"{p.repo}#g{p.group_index if p.group_index is not None else 'na'}" for p in plan[:12]]
        logger.info(
            "[SWE-SCHED] round={} reorder plan (first {}): {}",
            rollout_id,
            len(short_plan),
            short_plan,
        )

    @staticmethod
    def _fits_virtual(active: ResourceVector, request: ResourceVector, budget: ResourceVector) -> bool:
        return (
            active.memory_bytes + request.memory_bytes <= budget.memory_bytes
            and active.cpu_percent + request.cpu_percent <= budget.cpu_percent
            and active.disk_read_bytes + request.disk_read_bytes <= budget.disk_read_bytes
            and active.disk_write_bytes + request.disk_write_bytes <= budget.disk_write_bytes
        )

    def _ensure_sampler_locked(self) -> None:
        if self.config.disable_live_stats_polling:
            return
        if self._sampler_task is not None and not self._sampler_task.done():
            return
        self._stop_sampling = False
        self._sampler_task = asyncio.create_task(self._sampling_loop(), name="swe-online-docker-sampler")

    async def _sampling_loop(self) -> None:
        idle_rounds = 0
        while not self._stop_sampling:
            async with self._lock:
                leases = self._select_sampling_targets_unlocked()
                interval = max(0.2, float(self.config.sampling_interval_sec))
                spacing = max(0.0, float(self.config.stats_request_spacing_sec))

            if not leases:
                idle_rounds += 1
                if idle_rounds >= 3:
                    break
                await asyncio.sleep(interval)
                continue

            idle_rounds = 0
            await self._sample_leases(leases, spacing=spacing)
            await asyncio.sleep(interval)

        async with self._lock:
            self._sampler_task = None

    def _select_sampling_targets_unlocked(self) -> list[tuple[str, str]]:
        candidates: list[tuple[str, str]] = []
        for prompt_id, active in self._active_prompts.items():
            if not active.leases:
                continue
            candidates.append((next(iter(active.leases)), prompt_id))

        if not candidates:
            return []

        if len(candidates) > 1:
            rotate = self._sampling_cursor % len(candidates)
            if rotate:
                candidates = candidates[rotate:] + candidates[:rotate]
            head = candidates[: min(len(candidates), 4)]
            self._rng.shuffle(head)
            candidates[: len(head)] = head

        limit = max(1, int(self.config.max_stats_requests_per_round))
        if len(candidates) <= limit:
            self._sampling_cursor = (self._sampling_cursor + len(candidates)) % max(1, len(candidates))
            return candidates

        selected = candidates[:limit]
        self._sampling_cursor = (self._sampling_cursor + len(selected)) % len(candidates)
        return selected

    async def _sample_leases(self, leases: list[tuple[str, str]], *, spacing: float = 0.0) -> None:
        stats_batch = getattr(self.env_client, "stats_batch", None)
        if callable(stats_batch) and leases:
            try:
                payload = await stats_batch([lease_id for lease_id, _ in leases])
                stats_by_lease = payload.get("stats") if isinstance(payload, dict) else None
                if not isinstance(stats_by_lease, dict):
                    raise RuntimeError(f"invalid stats_batch payload: {payload!r}")
                for lease_id, prompt_id in leases:
                    item = stats_by_lease.get(lease_id)
                    if not isinstance(item, dict):
                        await self._record_lease_stats_error(
                            lease_id,
                            prompt_id,
                            RuntimeError("missing lease stats in batch response"),
                        )
                        continue
                    if not item.get("ok", False):
                        await self._record_lease_stats_error(
                            lease_id,
                            prompt_id,
                            RuntimeError(str(item.get("error") or item)),
                        )
                        continue
                    await self._record_lease_stats_payload(lease_id, prompt_id, item)
                return
            except Exception as exc:
                logger.debug(
                    "[SWE-SCHED] batch stats failed for {} leases, falling back to per-lease stats: {}",
                    len(leases),
                    exc,
                )

        tasks: list[asyncio.Task] = []
        for idx, (lease_id, prompt_id) in enumerate(leases):
            tasks.append(
                asyncio.create_task(
                    self._sample_one_lease(lease_id, prompt_id),
                    name=f"swe-online-docker-sample-{idx}",
                )
            )
            if spacing > 0.0 and idx + 1 < len(leases):
                await asyncio.sleep(spacing)
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _sample_one_lease(self, lease_id: str, prompt_id: str) -> None:
        try:
            payload = await self.env_client.stats(lease_id=lease_id)
        except Exception as exc:
            await self._record_lease_stats_error(lease_id, prompt_id, exc)
            return
        await self._record_lease_stats_payload(lease_id, prompt_id, payload)

    async def _record_lease_stats_error(self, lease_id: str, prompt_id: str, exc: Exception) -> None:
        error_text = str(exc).lower()
        is_unknown_lease = "unknown lease_id" in error_text
        async with self._lock:
            active = self._active_prompts.get(prompt_id)
            if active is None:
                return
            failures = int(active.lease_errors.get(lease_id, 0)) + 1
            active.lease_errors[lease_id] = failures
            if is_unknown_lease or failures >= 3:
                active.leases.discard(lease_id)
                self._lease_to_prompt.pop(lease_id, None)
                if is_unknown_lease:
                    logger.info(
                        "[SWE-SCHED] lease={} no longer exists, stop tracking lease ({})",
                        lease_id,
                        exc,
                    )
                else:
                    logger.warning(
                        "[SWE-SCHED] lease={} stats unavailable for 3 samples, stop tracking lease ({})",
                        lease_id,
                        exc,
                    )

    async def _record_lease_stats_payload(self, lease_id: str, prompt_id: str, payload: dict[str, Any]) -> None:
        memory_bytes = float(
            payload.get("peak_memory_usage_bytes", payload.get("memory_usage_bytes", 0.0)) or 0.0
        )
        cpu_percent = float(payload.get("avg_cpu_percent", payload.get("cpu_percent", 0.0)) or 0.0)
        if "cpu_sample_valid" in payload:
            cpu_sample_valid = bool(payload.get("cpu_sample_valid"))
        else:
            cpu_sample_valid = cpu_percent > 0.0
        disk_read_bytes = float(
            payload.get("peak_disk_read_bytes", payload.get("disk_read_bytes", 0.0)) or 0.0
        )
        disk_write_bytes = float(
            payload.get("peak_disk_write_bytes", payload.get("disk_write_bytes", 0.0)) or 0.0
        )
        async with self._lock:
            active = self._active_prompts.get(prompt_id)
            if active is None:
                return
            active.sample_count += 1
            lease_stats = active.lease_stats_by_lease.setdefault(lease_id, _LeaseResourceStats())
            lease_stats.sample_count += 1
            lease_stats.peak_memory_bytes = max(lease_stats.peak_memory_bytes, memory_bytes)
            lease_stats.disk_read_bytes = max(lease_stats.disk_read_bytes, disk_read_bytes)
            lease_stats.disk_write_bytes = max(lease_stats.disk_write_bytes, disk_write_bytes)
            if cpu_sample_valid:
                active.cpu_sample_count += 1
                active.latest_cpu_percent = cpu_percent
                active.cpu_latest_by_lease[lease_id] = cpu_percent
                lease_stats.cpu_sample_count += 1
                lease_stats.avg_cpu_percent = cpu_percent
            active.peak_memory_bytes = max(active.peak_memory_bytes, memory_bytes)
            active.disk_read_bytes = max(active.disk_read_bytes, disk_read_bytes)
            active.disk_write_bytes = max(active.disk_write_bytes, disk_write_bytes)
            active.lease_errors.pop(lease_id, None)

    def _update_repo_profile_unlocked(self, summary: PromptResourceSummary) -> None:
        data_key = (summary.data_key or "").strip().lower()
        repo_key = (summary.repo or "").strip().lower()

        # Enforce per-data aggregation: if key degraded to repo-level or unknown,
        # recover instance-level key from prompt_id when available.
        if not data_key or data_key == "unknown" or (repo_key and data_key == repo_key):
            inferred = _data_key_from_prompt_id(summary.prompt_id)
            if inferred:
                data_key = inferred

        if not data_key:
            data_key = summary.prompt_id or summary.repo

        profile = self.repo_resource_stats.get(data_key)
        if profile is None:
            profile = RepoResourceProfile()
            self.repo_resource_stats[data_key] = profile

        n = profile.sample_count
        profile.sample_count = n + 1
        profile.peak_memory_bytes = max(profile.peak_memory_bytes, summary.peak_memory_bytes)
        profile.avg_cpu_percent = _rolling_avg(profile.avg_cpu_percent, n, summary.avg_cpu_percent)
        profile.avg_disk_read_bytes = _rolling_avg(profile.avg_disk_read_bytes, n, float(summary.disk_read_bytes))
        profile.avg_disk_write_bytes = _rolling_avg(profile.avg_disk_write_bytes, n, float(summary.disk_write_bytes))
        profile.avg_total_time_sec = _rolling_avg(profile.avg_total_time_sec, n, float(summary.total_time_sec))
        profile.last_updated = time.time()

        logger.info(
            "[SWE-SCHED] data profile updated key={} repo={} count={} peak_mem={} avg_cpu={:.2f}% avg_disk(r={},w={}) avg_elapsed={:.1f}s",
            data_key,
            summary.repo,
            profile.sample_count,
            format_bytes(profile.peak_memory_bytes),
            profile.avg_cpu_percent,
            format_bytes(profile.avg_disk_read_bytes),
            format_bytes(profile.avg_disk_write_bytes),
            profile.avg_total_time_sec,
        )

    def _load_profiles(self) -> None:
        if self._profile_path is None:
            return
        if not self._profile_path.exists():
            return
        try:
            data = json.loads(self._profile_path.read_text())
            if not isinstance(data, dict):
                return
            for data_key, profile_data in data.items():
                if not isinstance(profile_data, dict):
                    continue
                self.repo_resource_stats[data_key] = RepoResourceProfile(
                    sample_count=int(profile_data.get("sample_count", 0)),
                    peak_memory_bytes=float(profile_data.get("peak_memory_bytes", 0.0)),
                    avg_cpu_percent=float(profile_data.get("avg_cpu_percent", 0.0)),
                    avg_disk_read_bytes=float(profile_data.get("avg_disk_read_bytes", 0.0)),
                    avg_disk_write_bytes=float(profile_data.get("avg_disk_write_bytes", 0.0)),
                    avg_total_time_sec=float(profile_data.get("avg_total_time_sec", 0.0)),
                    last_updated=float(profile_data.get("last_updated", 0.0)),
                )
            logger.info(
                "[SWE-SCHED] loaded resource profiles from {} (entries={})",
                self._profile_path,
                len(self.repo_resource_stats),
            )
        except Exception as exc:
            logger.warning("[SWE-SCHED] failed to load repo profile json {}: {}", self._profile_path, exc)

    def _refresh_resource_profiles(self, *, force: bool = False) -> int:
        if not self.config.use_resource_stats_dir:
            return 0
        now = time.time()
        refresh_sec = max(1.0, float(self.config.resource_stats_refresh_sec))
        if not force and now - self._resource_stats_last_refresh_ts < refresh_sec:
            return 0
        self._resource_stats_last_refresh_ts = now

        stats_dir = Path(self.config.resource_stats_dir)
        if not stats_dir.exists() or not stats_dir.is_dir():
            return 0

        aggregated: dict[str, RepoResourceProfile] = {}
        file_count = 0
        replay_per_run_dir = stats_dir / "per_run"
        replay_files = (
            sorted(replay_per_run_dir.glob("*.resource_profile.json"))
            if replay_per_run_dir.exists() and replay_per_run_dir.is_dir()
            else sorted(stats_dir.glob("*.resource_profile.json"))
        )

        if replay_files:
            for stat_file in replay_files:
                try:
                    payload = json.loads(stat_file.read_text())
                except Exception:
                    continue
                if not isinstance(payload, dict):
                    continue

                data_key = (
                    str(payload.get("instance_id") or "").strip().lower()
                    or _data_key_from_image_name(payload.get("image"))
                )
                if not data_key:
                    continue

                mem_mb_max = _nested_float(payload, "mem_mb", "max", 0.0)
                cpu_avg = _nested_float(payload, "cpu_percent", "avg", 0.0)
                disk_read_mb_max = _nested_float(payload, "disk_read_mb", "max", 0.0)
                disk_write_mb_max = _nested_float(payload, "disk_write_mb", "max", 0.0)
                duration_sec = max(
                    0.0,
                    float(payload.get("duration_sec", 0.0) or 0.0),
                    float(payload.get("end_time", 0.0) or 0.0) - float(payload.get("start_time", 0.0) or 0.0),
                )
                ts = float(payload.get("end_time", 0.0) or 0.0)

                _update_aggregated_profile(
                    aggregated,
                    data_key=data_key,
                    peak_memory_bytes=max(0.0, mem_mb_max) * 1024 * 1024,
                    avg_cpu_percent=cpu_avg,
                    avg_disk_read_bytes=max(0.0, disk_read_mb_max) * 1024 * 1024,
                    avg_disk_write_bytes=max(0.0, disk_write_mb_max) * 1024 * 1024,
                    duration_sec=duration_sec,
                    ts=ts,
                )
                file_count += 1
        else:
            for stat_file in stats_dir.glob("resource-usage-*.json"):
                try:
                    payload = json.loads(stat_file.read_text())
                except Exception:
                    continue
                if not isinstance(payload, dict):
                    continue
                image = payload.get("image")
                data_key = _data_key_from_image_name(image)
                if not data_key:
                    continue

                mem_mb_max = _nested_float(payload, "mem_mb", "max", 0.0)
                cpu_avg = _nested_float(payload, "cpu_percent", "avg", 0.0)
                disk_read_mb_max = _nested_float(payload, "disk_read_mb", "max", 0.0)
                disk_write_mb_max = _nested_float(payload, "disk_write_mb", "max", 0.0)
                duration_sec = max(
                    0.0,
                    float(payload.get("duration_sec", 0.0) or 0.0),
                    float(payload.get("end_time", 0.0) or 0.0) - float(payload.get("start_time", 0.0) or 0.0),
                )
                ts = float(payload.get("end_time", 0.0) or 0.0)

                _update_aggregated_profile(
                    aggregated,
                    data_key=data_key,
                    peak_memory_bytes=max(0.0, mem_mb_max) * 1024 * 1024,
                    avg_cpu_percent=cpu_avg,
                    avg_disk_read_bytes=max(0.0, disk_read_mb_max) * 1024 * 1024,
                    avg_disk_write_bytes=max(0.0, disk_write_mb_max) * 1024 * 1024,
                    duration_sec=duration_sec,
                    ts=ts,
                )
                file_count += 1

        if aggregated:
            self.repo_resource_stats.update(aggregated)
            logger.info(
                "[SWE-SCHED] loaded resource stats from {} (files={}, keys={}, format={})",
                stats_dir,
                file_count,
                len(aggregated),
                "replay_per_run" if replay_files else "legacy_resource_usage",
            )
        return len(aggregated)

    def _persist_profiles_unlocked(self) -> None:
        if self._profile_path is None:
            return
        try:
            self._profile_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {data_key: asdict(profile) for data_key, profile in self.repo_resource_stats.items()}
            self._profile_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
        except Exception as exc:
            logger.warning("[SWE-SCHED] failed to persist repo profile json {}: {}", self._profile_path, exc)



def _rolling_avg(old_value: float, old_count: int, new_value: float) -> float:
    if old_count <= 0:
        return float(new_value)
    return (float(old_value) * old_count + float(new_value)) / float(old_count + 1)
