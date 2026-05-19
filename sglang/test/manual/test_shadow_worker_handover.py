import os
import signal
import socket
import subprocess
import sys
import time
import unittest
from pathlib import Path
from typing import Any, Optional

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]


def _log(message: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[shadow-handover][{ts}] {message}", flush=True)


def _is_true(value: str) -> bool:
    return value.lower() in ("1", "true", "yes", "on")


def _wait_for_tcp(
    host: str,
    port: int,
    timeout_s: float,
    process: Optional[subprocess.Popen] = None,
):
    _log(f"Waiting for TCP ready: {host}:{port} (timeout={timeout_s}s)")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"Process exited unexpectedly with code {process.poll()}")

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            if sock.connect_ex((host, port)) == 0:
                _log(f"TCP ready: {host}:{port}")
                return
        time.sleep(0.2)
    raise TimeoutError(f"Timed out waiting for TCP {host}:{port}")


def _wait_for_health(
    base_url: str,
    timeout_s: float,
    process: Optional[subprocess.Popen] = None,
):
    _log(f"Waiting for health endpoint: {base_url}/health_generate (timeout={timeout_s}s)")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"Process exited unexpectedly with code {process.poll()}")

        try:
            resp = requests.get(f"{base_url}/health_generate", timeout=5)
            if resp.status_code == 200:
                _log(f"Health ready: {base_url}/health_generate")
                return
        except requests.RequestException:
            pass
        time.sleep(0.5)
    raise TimeoutError(f"Timed out waiting for {base_url}/health_generate")


def _wait_for_generate(
    base_url: str,
    payload: dict[str, Any],
    timeout_s: float,
    process: Optional[subprocess.Popen] = None,
) -> requests.Response:
    _log(f"Generate payload: {payload}")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"Process exited unexpectedly with code {process.poll()}")

        try:
            resp = requests.post(
                f"{base_url}/generate",
                json=payload,
                timeout=30,
            )
            if resp.status_code == 200:
                _log(f"Generate succeeded: {base_url}/generate")
                return resp
        except requests.RequestException:
            pass
        time.sleep(0.5)
    raise TimeoutError(f"Timed out waiting for {base_url}/generate success")


def _launch_process(command: list[str], env: dict[str, str]) -> subprocess.Popen:
    _log(f"Launching process: {' '.join(command)}")
    proc = subprocess.Popen(
        command,
        env=env,
        stdout=None,
        stderr=None,
        start_new_session=True,
    )
    _log(f"Launched PID={proc.pid}")
    return proc


def _find_available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _kill_process_group(proc: Optional[subprocess.Popen]):
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


class TestShadowWorkerHandoverE2E(unittest.TestCase):
    """
    Manual E2E for shadow-worker handover.

    Run example:
        SGLANG_SHADOW_MANUAL_E2E=1 \
        SGLANG_SHADOW_TEST_MODEL_PATH=/path/to/Qwen3-or-Qwen3-VL-model \
        CUDA_VISIBLE_DEVICES=0 \
        python3 -m pytest sglang/test/manual/test_shadow_worker_handover.py -q -s

    Required env:
    - SGLANG_SHADOW_MANUAL_E2E=1
    - SGLANG_SHADOW_TEST_MODEL_PATH

    Optional env:
    - SGLANG_SHADOW_TEST_SERVER_CKPTS (defaults to model path)
    - SGLANG_SHADOW_TEST_GPU_IDS (defaults to CUDA_VISIBLE_DEVICES or "0")
    - SGLANG_SHADOW_TEST_TP (defaults to 1)
    - SGLANG_SHADOW_HEARTBEAT_TIMEOUT (defaults to 1.0 for this test)
    """

    @classmethod
    def setUpClass(cls):
        if not _is_true(os.environ.get("SGLANG_SHADOW_MANUAL_E2E", "0")):
            raise unittest.SkipTest(
                "Set SGLANG_SHADOW_MANUAL_E2E=1 to run this manual shadow-worker E2E test."
            )

        try:
            import torch

            if not torch.cuda.is_available():
                raise unittest.SkipTest("CUDA is required for this test.")
        except Exception as exc:
            raise unittest.SkipTest(f"PyTorch/CUDA unavailable: {exc}")

        cls.model_path = os.environ.get("SGLANG_SHADOW_TEST_MODEL_PATH")
        if not cls.model_path:
            raise unittest.SkipTest(
                "Set SGLANG_SHADOW_TEST_MODEL_PATH to a local model path."
            )
        cls.server_ckpts = os.environ.get("SGLANG_SHADOW_TEST_SERVER_CKPTS", cls.model_path)
        _log(f"Using model_path={cls.model_path}")
        _log(f"Using server_ckpts={cls.server_ckpts}")

        workspace_root = Path(__file__).resolve().parents[3]
        checkpoint_engine_root = workspace_root / "checkpoint-engine"
        cls.ps_script = checkpoint_engine_root / "examples" / "persistent_ps_example.py"
        if not cls.ps_script.exists():
            raise unittest.SkipTest(f"Missing script: {cls.ps_script}")

        cls.tp = int(os.environ.get("SGLANG_SHADOW_TEST_TP", "1"))
        default_gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        cls.gpu_ids = os.environ.get("SGLANG_SHADOW_TEST_GPU_IDS", default_gpu_ids)
        cls.primary_gpu_id = cls.gpu_ids.split(",")[0].strip()

        cls.weight_port = int(os.environ.get("SGLANG_SHADOW_TEST_WEIGHT_PORT", "0")) or _find_available_port()
        cls.main_port = int(os.environ.get("SGLANG_SHADOW_TEST_MAIN_PORT", "0")) or _find_available_port()
        cls.shadow_port = int(os.environ.get("SGLANG_SHADOW_TEST_SHADOW_PORT", "0")) or _find_available_port()
        cls.kv_socket_path = os.environ.get(
            "SGLANG_SHADOW_TEST_KV_SOCKET_PATH", f"/tmp/kv_cache_shadow_{cls.main_port}.sock"
        )

        cls.main_url = f"http://127.0.0.1:{cls.main_port}"
        cls.shadow_url = f"http://127.0.0.1:{cls.shadow_port}"
        cls.heartbeat_file = f"/tmp/sglang_heartbeat_{cls.primary_gpu_id}_0.log"
        cls.main_process: Optional[subprocess.Popen] = None
        cls.shadow_process: Optional[subprocess.Popen] = None
        cls.extra_processes: list[subprocess.Popen] = []

        test_env = os.environ.copy()
        py_path = test_env.get("PYTHONPATH", "")
        python_paths = [str(REPO_ROOT / "python"), str(checkpoint_engine_root)]
        if py_path:
            python_paths.append(py_path)
        test_env["PYTHONPATH"] = ":".join(python_paths)
        test_env["SGLANG_KV_CACHE_SOCKET_PATH"] = cls.kv_socket_path
        test_env["SGLANG_SHADOW_HEARTBEAT_TIMEOUT"] = os.environ.get(
            "SGLANG_SHADOW_HEARTBEAT_TIMEOUT", "1.0"
        )
        if "SGLANG_SHADOW_TEST_GPU_IDS" in os.environ:
            test_env["CUDA_VISIBLE_DEVICES"] = cls.gpu_ids

        if os.path.exists(cls.kv_socket_path):
            os.remove(cls.kv_socket_path)
        if os.path.exists(cls.heartbeat_file):
            os.remove(cls.heartbeat_file)

        ps_cmd = [
            sys.executable,
            str(cls.ps_script),
            "--server-ckpts",
            cls.server_ckpts,
            str(cls.tp),
            str(cls.weight_port),
        ]
        ps_proc = _launch_process(ps_cmd, test_env)
        cls.extra_processes.append(ps_proc)
        _wait_for_tcp("127.0.0.1", cls.weight_port, timeout_s=60, process=ps_proc)
        _log("Persistent parameter server is ready.")

        kv_cmd = [
            sys.executable,
            "-m",
            "sglang.srt.mem_cache.kv_cache_server",
            "--socket-path",
            cls.kv_socket_path,
            "--gpu-id",
            cls.gpu_ids,
            "--model-path",
            cls.server_ckpts,
            "--mem-fraction-static",
            os.environ.get("SGLANG_SHADOW_TEST_MEM_FRACTION_STATIC", "0.8"),
            "--page-size",
            os.environ.get("SGLANG_SHADOW_TEST_PAGE_SIZE", "1"),
            "--dtype",
            os.environ.get("SGLANG_SHADOW_TEST_DTYPE", "bfloat16"),
        ]
        kv_proc = _launch_process(kv_cmd, test_env)
        cls.extra_processes.append(kv_proc)
        _wait_for_tcp("127.0.0.1", cls.weight_port, timeout_s=5, process=ps_proc)
        deadline = time.time() + 120
        while time.time() < deadline:
            if kv_proc.poll() is not None:
                raise RuntimeError(
                    f"KV cache server exited unexpectedly with code {kv_proc.poll()}"
                )
            if os.path.exists(cls.kv_socket_path):
                break
            time.sleep(0.5)
        else:
            raise TimeoutError(f"Timed out waiting for socket {cls.kv_socket_path}")
        _log(f"KV cache server is ready. socket={cls.kv_socket_path}")

        shadow_cmd = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            cls.model_path,
            "--tp",
            str(cls.tp),
            "--host",
            "127.0.0.1",
            "--port",
            str(cls.shadow_port),
            "--load-format",
            "weight_deamon",
            "--weight_load_port",
            str(cls.weight_port),
            "--enable-memory-saver",
            "--skeleton-worker",
            "--disable-custom-all-reduce",
            "--skip-server-warmup",
        ]
        cls.shadow_process = _launch_process(shadow_cmd, test_env)
        _wait_for_health(cls.shadow_url, timeout_s=180, process=cls.shadow_process)
        _log(f"Shadow worker startup completed at {cls.shadow_url}")

        main_cmd = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            cls.model_path,
            "--tp",
            str(cls.tp),
            "--host",
            "127.0.0.1",
            "--port",
            str(cls.main_port),
            "--load-format",
            "weight_deamon",
            "--weight_load_port",
            str(cls.weight_port),
            "--disable-custom-all-reduce",
            "--skip-server-warmup",
        ]
        cls.main_process = _launch_process(main_cmd, test_env)
        _wait_for_health(cls.main_url, timeout_s=180, process=cls.main_process)
        _log(f"Main worker startup completed at {cls.main_url}")

    @classmethod
    def tearDownClass(cls):
        for proc in [cls.main_process, cls.shadow_process, *cls.extra_processes]:
            _kill_process_group(proc)

        if hasattr(cls, "kv_socket_path") and os.path.exists(cls.kv_socket_path):
            os.remove(cls.kv_socket_path)
        if hasattr(cls, "heartbeat_file") and os.path.exists(cls.heartbeat_file):
            os.remove(cls.heartbeat_file)

    def test_main_shadow_handover(self):
        generate_payload = {
            "text": os.environ.get(
                "SGLANG_SHADOW_TEST_PROMPT",
                "The capital of France is",
            ),
            "sampling_params": {
                "temperature": float(os.environ.get("SGLANG_SHADOW_TEST_TEMPERATURE", "0.0")),
                "max_new_tokens": int(os.environ.get("SGLANG_SHADOW_TEST_MAX_NEW_TOKENS", "16")),
                "top_p": float(os.environ.get("SGLANG_SHADOW_TEST_TOP_P", "1.0")),
                "top_k": int(os.environ.get("SGLANG_SHADOW_TEST_TOP_K", "-1")),
                "sampling_seed": int(os.environ.get("SGLANG_SHADOW_TEST_SAMPLING_SEED", "42")),
            },
        }
        expected_substr = os.environ.get("SGLANG_SHADOW_TEST_EXPECTED_SUBSTR", "").strip()
        strict_semantic = _is_true(
            os.environ.get("SGLANG_SHADOW_TEST_STRICT_SEMANTIC", "0")
        )

        # Validate request parameter correctness explicitly.
        sp = generate_payload["sampling_params"]
        self.assertGreaterEqual(float(sp["temperature"]), 0.0)
        self.assertGreater(float(sp["top_p"]), 0.0)
        self.assertLessEqual(float(sp["top_p"]), 1.0)
        self.assertTrue(int(sp["top_k"]) == -1 or int(sp["top_k"]) > 0)
        self.assertGreater(int(sp["max_new_tokens"]), 0)

        # Main worker should serve normally before failover.
        _log("Sending initial request to main worker...")
        main_resp = _wait_for_generate(
            self.main_url,
            payload=generate_payload,
            timeout_s=120,
            process=self.__class__.main_process,
        )
        self.assertEqual(main_resp.status_code, 200, main_resp.text)
        main_json = main_resp.json()
        self.assertIn("text", main_json)
        _log(f"Main generate response: status={main_resp.status_code}, body={main_json}")
        self.assertIn("output_ids", main_json)
        self.assertIn("meta_info", main_json)
        self.assertLessEqual(
            len(main_json["output_ids"]),
            int(generate_payload["sampling_params"]["max_new_tokens"]),
        )
        if expected_substr:
            hit = expected_substr in main_json["text"]
            _log(
                f"Main semantic check expected_substr={expected_substr!r} hit={hit} "
                f"(strict={strict_semantic})"
            )
            if strict_semantic:
                self.assertIn(
                    expected_substr,
                    main_json["text"],
                    f"Main output does not contain expected substring={expected_substr!r}",
                )

        # Kill main worker and verify shadow worker takes over after heartbeat timeout.
        _log("Killing main worker to trigger handover...")
        _kill_process_group(self.__class__.main_process)
        self.__class__.main_process = None
        _log("Main worker terminated. Waiting for shadow takeover...")

        takeover_start = time.time()
        shadow_resp = _wait_for_generate(
            self.shadow_url,
            payload=generate_payload,
            timeout_s=120,
            process=self.__class__.shadow_process,
        )
        handover_latency = time.time() - takeover_start

        self.assertEqual(shadow_resp.status_code, 200, shadow_resp.text)
        shadow_json = shadow_resp.json()
        self.assertIn("text", shadow_json)
        _log(f"Shadow generate response: status={shadow_resp.status_code}, body={shadow_json}")
        _log(f"Handover latency: {handover_latency:.2f}s")
        self.assertIn("output_ids", shadow_json)
        self.assertIn("meta_info", shadow_json)
        self.assertLessEqual(
            len(shadow_json["output_ids"]),
            int(generate_payload["sampling_params"]["max_new_tokens"]),
        )
        if expected_substr:
            hit = expected_substr in shadow_json["text"]
            _log(
                f"Shadow semantic check expected_substr={expected_substr!r} hit={hit} "
                f"(strict={strict_semantic})"
            )
            if strict_semantic:
                self.assertIn(
                    expected_substr,
                    shadow_json["text"],
                    f"Shadow output does not contain expected substring={expected_substr!r}",
                )

        # Core correctness for handover: same prompt + same sampling params => same output.
        self.assertEqual(
            main_json["output_ids"],
            shadow_json["output_ids"],
            "Main and shadow output_ids mismatch under identical deterministic sampling params.",
        )
        self.assertEqual(
            main_json["text"],
            shadow_json["text"],
            "Main and shadow text mismatch under identical deterministic sampling params.",
        )
        self.assertLess(
            handover_latency,
            30,
            f"Shadow takeover too slow: {handover_latency:.2f}s",
        )


if __name__ == "__main__":
    unittest.main()
