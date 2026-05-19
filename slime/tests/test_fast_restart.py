import threading
import sys
import types
from dataclasses import dataclass
from types import SimpleNamespace

if "sglang_router" not in sys.modules:
    sys.modules["sglang_router"] = types.SimpleNamespace(__version__="0.3.0")

if "packaging.version" not in sys.modules:
    packaging_module = types.ModuleType("packaging")
    packaging_version_module = types.ModuleType("packaging.version")
    packaging_version_module.parse = lambda value: value
    packaging_module.version = packaging_version_module
    sys.modules["packaging"] = packaging_module
    sys.modules["packaging.version"] = packaging_version_module

if "sglang.srt.server_args" not in sys.modules:
    sglang_module = types.ModuleType("sglang")
    srt_module = types.ModuleType("sglang.srt")
    server_args_module = types.ModuleType("sglang.srt.server_args")
    utils_module = types.ModuleType("sglang.srt.utils")

    @dataclass
    class _ServerArgs:
        host: str = "127.0.0.1"
        port: int = 0
        node_rank: int = 0
        api_key: str | None = None
        nccl_port: int = 0
        dist_init_addr: str = "127.0.0.1:0"

        def url(self):
            return f"http://{self.host}:{self.port}"

    server_args_module.ServerArgs = _ServerArgs
    utils_module.kill_process_tree = lambda pid: None

    sys.modules["sglang"] = sglang_module
    sys.modules["sglang.srt"] = srt_module
    sys.modules["sglang.srt.server_args"] = server_args_module
    sys.modules["sglang.srt.utils"] = utils_module

if "ray" not in sys.modules:
    ray_module = types.ModuleType("ray")
    ray_module.get = lambda obj, timeout=None: obj
    ray_module.kill = lambda actor: None
    sys.modules["ray"] = ray_module

if "slime.ray.ray_actor" not in sys.modules:
    ray_actor_module = types.ModuleType("slime.ray.ray_actor")

    class _RayActor:
        @staticmethod
        def _get_current_node_ip_and_free_port(start_port=10000, consecutive=1):
            del consecutive
            return "127.0.0.1", start_port

    ray_actor_module.RayActor = _RayActor
    sys.modules["slime.ray.ray_actor"] = ray_actor_module

if "slime.utils.http_utils" not in sys.modules:
    http_utils_module = types.ModuleType("slime.utils.http_utils")
    http_utils_module.get_host_info = lambda: ("localhost", "127.0.0.1")
    sys.modules["slime.utils.http_utils"] = http_utils_module

if "httpx" not in sys.modules:
    httpx_module = types.ModuleType("httpx")

    class _AsyncClient:
        def __init__(self, *args, **kwargs):
            del args, kwargs

    class _HTTPStatusError(Exception):
        pass

    class _Limits:
        def __init__(self, *args, **kwargs):
            del args, kwargs

    class _Timeout:
        def __init__(self, *args, **kwargs):
            del args, kwargs

    httpx_module.AsyncClient = _AsyncClient
    httpx_module.HTTPStatusError = _HTTPStatusError
    httpx_module.Limits = _Limits
    httpx_module.Timeout = _Timeout
    sys.modules["httpx"] = httpx_module

from slime.backends.sglang_utils.sglang_engine import SGLangEngine, _compute_server_args
from slime.utils.health_monitor import RolloutHealthMonitor


class _RemoteCall:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        return lambda: self._fn(*args, **kwargs)


def _fake_ray_get(obj, timeout=None):
    del timeout
    if isinstance(obj, list):
        return [_fake_ray_get(item) for item in obj]
    if callable(obj):
        return obj()
    return obj


class _FakeEngine:
    def __init__(self, *, handover_result):
        self.health_generate = _RemoteCall(self._health_generate)
        self.promote_shadow_worker = _RemoteCall(lambda: handover_result)

    @staticmethod
    def _health_generate(timeout):
        del timeout
        raise RuntimeError("main worker is down")


def test_health_monitor_prefers_shadow_handover(monkeypatch):
    marked = []
    engine = _FakeEngine(handover_result=True)
    manager = SimpleNamespace(
        all_rollout_engines=[engine],
        rollout_engines=[engine],
        nodes_per_engine=1,
        mark_rollout_engines_need_weight_update_reconnect=lambda count, reason: marked.append((count, reason)),
    )
    args = SimpleNamespace(
        rollout_health_check_interval=1.0,
        rollout_health_check_timeout=1.0,
        rollout_health_check_first_wait=0.0,
    )
    monitor = RolloutHealthMonitor(manager, args)
    monitor._stop_event = threading.Event()
    monitor._pause_event = threading.Event()

    killed = []
    monkeypatch.setattr("slime.utils.health_monitor.ray.get", _fake_ray_get)
    monkeypatch.setattr(monitor, "_kill_engine", lambda rollout_engine_id: killed.append(rollout_engine_id))

    monitor._check_engine_health(0, engine)

    assert killed == []
    assert marked == [(1, "shadow handover on rollout engine 0")]


def test_health_monitor_falls_back_to_kill_when_handover_unavailable(monkeypatch):
    engine = _FakeEngine(handover_result=False)
    manager = SimpleNamespace(
        all_rollout_engines=[engine],
        rollout_engines=[engine],
        nodes_per_engine=1,
    )
    args = SimpleNamespace(
        rollout_health_check_interval=1.0,
        rollout_health_check_timeout=1.0,
        rollout_health_check_first_wait=0.0,
    )
    monitor = RolloutHealthMonitor(manager, args)
    monitor._stop_event = threading.Event()
    monitor._pause_event = threading.Event()

    killed = []
    monkeypatch.setattr("slime.utils.health_monitor.ray.get", _fake_ray_get)
    monkeypatch.setattr(monitor, "_kill_engine", lambda rollout_engine_id: killed.append(rollout_engine_id))

    monitor._check_engine_health(0, engine)

    assert killed == [0]


def test_rollout_manager_tracks_shadow_handover_reconnect_obligation():
    manager = object.__new__(RolloutManager)
    manager._num_new_engines_lock = threading.Lock()
    manager.num_new_engines = 0
    manager.num_new_prm_engines = 0
    manager._pending_shadow_handover_reconnects = 0
    manager._pending_shadow_handover_reconnect_reasons = []

    pending = RolloutManager.mark_rollout_engines_need_weight_update_reconnect(
        manager,
        count=1,
        reason="shadow handover on rollout engine 0",
    )
    assert pending == 1

    state = RolloutManager.get_weight_update_reconnect_debug_state(manager)
    assert state["num_new_engines"] == 1
    assert state["pending_shadow_handover_reconnects"] == 1
    assert state["pending_shadow_handover_reconnect_reasons"] == ["shadow handover on rollout engine 0"]

    RolloutManager.clear_num_new_engines(manager, consumed=1)
    state = RolloutManager.get_weight_update_reconnect_debug_state(manager)
    assert state["num_new_engines"] == 0
    assert state["pending_shadow_handover_reconnects"] == 1

    RolloutManager.ack_shadow_handover_weight_update_reconnect(manager)
    state = RolloutManager.get_weight_update_reconnect_debug_state(manager)
    assert state["pending_shadow_handover_reconnects"] == 0
    assert state["pending_shadow_handover_reconnect_reasons"] == []


def test_promote_shadow_worker_switches_active_server(monkeypatch):
    assert hasattr(SGLangEngine, "shadow_worker")

    args = SimpleNamespace(
        rollout_external=False,
        use_slime_router=False,
        sglang_enable_fast_restart=True,
        use_fault_tolerance=True,
    )
    engine = SGLangEngine(args=args, rank=0)
    engine.node_rank = 0
    engine.router_ip = None
    engine.router_port = None
    engine.server_host = "127.0.0.1"
    engine.server_port = 31000
    engine.server_args = SimpleNamespace(
        host="127.0.0.1",
        port=31000,
        nccl_port=32000,
        dist_init_addr="127.0.0.1:33000",
    )
    engine.shadow_server_args = SimpleNamespace(
        host="127.0.0.1",
        port=41000,
        nccl_port=42000,
        dist_init_addr="127.0.0.1:43000",
    )
    engine.process = SimpleNamespace(pid=111)
    engine.shadow_worker = SimpleNamespace(pid=222, is_alive=lambda: True)
    engine.shadow_worker_enabled = True

    killed_pids = []
    monkeypatch.setattr(engine, "_shadow_worker_ready_for_handover", lambda timeout=10.0: True)
    monkeypatch.setattr(
        "slime.backends.sglang_utils.sglang_engine.kill_process_tree",
        lambda pid: killed_pids.append(pid),
    )

    assert engine.promote_shadow_worker() is True
    assert killed_pids == [111]
    assert engine.server_port == 41000
    assert engine.process.pid == 222
    assert engine.shadow_worker is None


def test_wait_until_ready_blocks_for_shadow_stabilization(monkeypatch):
    args = SimpleNamespace(
        rollout_external=False,
        use_slime_router=False,
        sglang_enable_fast_restart=True,
        use_fault_tolerance=True,
    )
    engine = SGLangEngine(args=args, rank=0)
    engine.node_rank = 0
    engine.server_args = SimpleNamespace(url=lambda: "http://127.0.0.1:31000", api_key=None)
    engine.process = SimpleNamespace(is_alive=lambda: True)
    engine.shadow_worker = SimpleNamespace(is_alive=lambda: True)
    engine.shadow_worker_enabled = True

    waits = []
    ready_checks = iter([False, True])
    monkeypatch.setattr(
        "slime.backends.sglang_utils.sglang_engine._wait_server_healthy",
        lambda base_url, api_key, is_process_alive: waits.append(("main", base_url, api_key, is_process_alive())),
    )
    monkeypatch.setattr(
        engine,
        "_shadow_worker_ready_for_handover",
        lambda timeout=10.0: waits.append(("shadow_probe", timeout)) or next(ready_checks),
    )
    monkeypatch.setattr(
        "slime.backends.sglang_utils.sglang_engine.time.sleep",
        lambda seconds: waits.append(("sleep", seconds)),
    )

    assert engine.wait_until_ready(wait_for_shadow=True, shadow_timeout=30.0, stabilization_seconds=60.0) is True
    assert waits[0][0] == "main"
    assert ("sleep", 2.0) in waits
    assert ("sleep", 60.0) in waits


def test_main_worker_infers_weight_load_port_from_base_mapping(monkeypatch):
    monkeypatch.setenv("WEIGHT_SERVER_BASE_PORT", "5556")
    monkeypatch.setenv("SGLANG_MIN_GPU_ID", "0")

    args = SimpleNamespace(
        prm_num_gpus_per_engine=4,
        rollout_num_gpus_per_engine=4,
        num_gpus_per_node=8,
        colocate=False,
        debug_rollout_only=True,
        actor_num_gpus_per_node=8,
        actor_num_nodes=1,
        use_critic=False,
        critic_num_gpus_per_node=0,
        critic_num_nodes=0,
        hf_checkpoint="/tmp/model",
        prm_model_path="/tmp/prm",
        seed=1234,
        offload_rollout=False,
        sglang_pp_size=1,
        sglang_dp_size=1,
        sglang_ep_size=1,
        fp16=False,
        use_rollout_routing_replay=False,
        sglang_shadow_worker_weight_server_base_port=None,
        sglang_shadow_worker_min_gpu_id=None,
    )

    kwargs0, _ = _compute_server_args(
        args,
        rank=0,
        dist_init_addr="127.0.0.1:30000",
        nccl_port=30001,
        host="127.0.0.1",
        port=30002,
        base_gpu_id=0,
        engine_role="rollout",
    )
    kwargs1, _ = _compute_server_args(
        args,
        rank=1,
        dist_init_addr="127.0.0.1:30010",
        nccl_port=30011,
        host="127.0.0.1",
        port=30012,
        base_gpu_id=4,
        engine_role="rollout",
    )

    assert kwargs0["weight_load_port"] == 5556
    assert kwargs1["weight_load_port"] == 5557
    assert kwargs0["load_format"] == "weight_deamon"
    assert kwargs1["load_format"] == "weight_deamon"
