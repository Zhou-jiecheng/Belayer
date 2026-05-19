from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import functools
import hashlib
import importlib.util
import json
import logging
import multiprocessing
import os
import queue
import re
import sys
import threading
import time
import types
from pathlib import Path
import traceback

from slime.utils.types import Sample

DEFAULT_PRIME_CODE_DIR = "/mnt/shared-storage-user/ailab-sys/zhoujiecheng/projs/robust_rl/verl/verl/utils/reward_score/prime_code"
REWARD_THREAD_POOL_WORKERS_ENV = "PRIME_CODE_SHARED_REWARD_THREAD_POOL_WORKERS"
DEFAULT_REWARD_THREAD_POOL_WORKERS = 8
THINK_REWARD_BONUS = 0.5
TOOL_CALL_REWARD_BONUS = 0.5
THINK_BLOCK_RE = re.compile(r"<think>\s*.+?\s*</think>", re.DOTALL)
TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
logger = logging.getLogger(__name__)
_PRIME_RUN_TESTS: dict[str, object] = {}
_PRIME_RUN_TEST_LOCK = threading.Lock()
_PRIME_REWARD_EXECUTORS: dict[int, concurrent.futures.ThreadPoolExecutor] = {}
_PRIME_REWARD_EXECUTOR_LOCK = threading.Lock()


def _sample_log_id(sample: Sample) -> str:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    for key in ("index", "sample_index", "rollout_id", "question_id", "task_id", "id"):
        value = metadata.get(key)
        if value is not None:
            return f"{key}={value}"
    return "sample=unknown"


def _parse_positive_int_env(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError:
        logger.warning("Invalid int env %s=%r, falling back to %d.", name, raw_value, default)
        return default
    if value <= 0:
        logger.warning("Env %s=%r must be > 0, falling back to %d.", name, raw_value, default)
        return default
    return value


def _get_prime_reward_executor() -> concurrent.futures.ThreadPoolExecutor:
    max_workers = _parse_positive_int_env(REWARD_THREAD_POOL_WORKERS_ENV, DEFAULT_REWARD_THREAD_POOL_WORKERS)
    with _PRIME_REWARD_EXECUTOR_LOCK:
        executor = _PRIME_REWARD_EXECUTORS.get(max_workers)
        if executor is None:
            executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="prime-code-reward",
            )
            _PRIME_REWARD_EXECUTORS[max_workers] = executor
        return executor


def _ensure_prime_pyext_available():
    if "pyext" not in sys.modules:
        try:
            import pyext  # type: ignore  # noqa: F401
        except Exception:
            pyext_module = types.ModuleType("pyext")

            class RuntimeModule:
                @staticmethod
                def from_string(module_name: str, _unused_path: str, code: str):
                    module = types.ModuleType(module_name)
                    exec(code, module.__dict__)
                    return module

            pyext_module.RuntimeModule = RuntimeModule
            sys.modules["pyext"] = pyext_module


def _load_prime_run_test(prime_code_dir: str):
    package_dir = Path(prime_code_dir)
    testing_util_file = package_dir / "testing_util.py"
    if not testing_util_file.exists():
        raise FileNotFoundError(f"PRIME code testing utility not found: {testing_util_file}")

    resolved_dir = str(package_dir.resolve())
    with _PRIME_RUN_TEST_LOCK:
        cached = _PRIME_RUN_TESTS.get(resolved_dir)
        if cached is not None:
            return cached

        _ensure_prime_pyext_available()
        module_name = f"_prime_code_reward_testing_util_{hashlib.sha1(resolved_dir.encode('utf-8')).hexdigest()[:12]}"
        module = sys.modules.get(module_name)
        if module is None:
            spec = importlib.util.spec_from_file_location(module_name, testing_util_file)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load PRIME code testing utility from {testing_util_file}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

        run_test = getattr(module, "run_test", None)
        if not callable(run_test):
            raise ImportError(f"PRIME code testing utility missing callable run_test: {testing_util_file}")
        _PRIME_RUN_TESTS[resolved_dir] = run_test
        return run_test


def _prime_check_worker(
    prime_code_dir: str,
    sample: dict | None,
    generation: str,
    debug: bool,
    timeout: int | float,
    result_queue,
):
    with open(os.devnull, "w") as devnull:
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            run_test = _load_prime_run_test(prime_code_dir)
            res, metadata = run_test(in_outs=sample, test=generation, debug=debug, timeout=timeout)
            result_queue.put((res, metadata))
        except Exception:
            traceback.print_exc(10)
            fallback_len = len((sample or {}).get("inputs") or [])
            result_queue.put(([-1 for _ in range(fallback_len)], {}))


def _prime_check_correctness_isolated(
    prime_code_dir: str,
    in_outs: dict | None,
    generation: str,
    timeout: int | float = 10,
    debug: bool = True,
):
    ctx = multiprocessing.get_context()
    result_queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(
        target=_prime_check_worker,
        args=(prime_code_dir, in_outs, generation, debug, timeout, result_queue),
    )
    result_payload = None
    try:
        proc.start()
        proc.join(timeout=timeout + 1)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=1)

        with contextlib.suppress(queue.Empty):
            result_payload = result_queue.get(timeout=0.1)
    finally:
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=1)
        with contextlib.suppress(Exception):
            proc.close()
        with contextlib.suppress(Exception):
            result_queue.close()
        with contextlib.suppress(Exception):
            result_queue.join_thread()

    if result_payload is None:
        fallback_len = len((in_outs or {}).get("inputs") or [])
        if debug:
            print("global timeout")
        return [-1 for _ in range(fallback_len)], []

    res, metadata = result_payload
    return res, [metadata]


def _normalize_test_cases(test_cases):
    if not isinstance(test_cases, dict):
        test_cases = json.loads(test_cases)
    return test_cases


def _extract_solution(completion: str | None) -> str:
    completion = completion or ""
    return completion.split("```python")[-1].split("```")[0]


def _has_execute_code_tool_call(completion: str | None) -> bool:
    completion = completion or ""
    for match in TOOL_CALL_BLOCK_RE.finditer(completion):
        payload = match.group(1)
        try:
            tool_call = json.loads(payload)
        except Exception:
            continue
        arguments = tool_call.get("arguments") or {}
        if tool_call.get("name") == "execute_code" and isinstance(arguments.get("code"), str):
            return True
    return False


def _compute_format_and_tool_bonus(completion: str | None, *, apply_bonus: bool = True) -> tuple[float, dict[str, float]]:
    completion = completion or ""
    components = {
        "base_reward": 0.0,
        "think_bonus": THINK_REWARD_BONUS if apply_bonus and THINK_BLOCK_RE.search(completion) else 0.0,
        "tool_call_bonus": TOOL_CALL_REWARD_BONUS if apply_bonus and _has_execute_code_tool_call(completion) else 0.0,
    }
    bonus = components["think_bonus"] + components["tool_call_bonus"]
    return bonus, components


def _compute_score_local(prime_code_dir: str, completion, test_cases, continuous=False, apply_bonus=True):
    solution = _extract_solution(completion)
    bonus, reward_components = _compute_format_and_tool_bonus(completion, apply_bonus=apply_bonus)
    try:
        test_cases = _normalize_test_cases(test_cases)

        try:
            res, metadata = _prime_check_correctness_isolated(
                prime_code_dir,
                in_outs=test_cases,
                generation=solution,
                timeout=5,
                debug=False,
            )
            metadata = dict(enumerate(metadata))[0]
            base_score = 1.0 if all(map(lambda x: x is True, res)) else 0.0
            if base_score:
                reward_components["base_reward"] = base_score
                metadata["reward_components"] = dict(reward_components)
                metadata["reward"] = base_score + bonus
                return base_score + bonus, metadata
        except Exception:
            pass

        test_cases_list = []
        inputs = test_cases["inputs"]
        outputs = test_cases["outputs"]
        for i in range(len(inputs)):
            test_cases_list.append({"inputs": [inputs[i]], "outputs": [outputs[i]]})

        metadata_list = None
        if continuous:
            metadata_list = []
            res_list = []
            for test_case_id, test_case in enumerate(test_cases_list):
                res, metadata = _prime_check_correctness_isolated(
                    prime_code_dir,
                    in_outs=test_case,
                    generation=solution,
                    timeout=10,
                    debug=False,
                )
                try:
                    metadata = dict(enumerate(metadata))[0]
                except Exception:
                    metadata = {}
                metadata["test_case"] = {}
                metadata["test_case"]["input"] = str(test_case["inputs"][0])
                metadata["test_case"]["output"] = str(test_case["outputs"][0])
                metadata["test_case"]["res"] = str(res)
                metadata_list.append(metadata)
                res_list.extend(res)

                if test_case_id >= 9:
                    break
            res_count = len(res_list) if len(res_list) > 0 else 1
            base_score = sum(map(lambda x: x is True, res_list)) / res_count
            reward_components["base_reward"] = base_score
            for item in metadata_list:
                item["reward_components"] = dict(reward_components)
                item["reward"] = base_score + bonus
        else:
            base_score = 0.0
    except Exception:
        traceback.print_exc(10)
        base_score = 0.0
        metadata_list = None
    reward_components["base_reward"] = base_score
    return base_score + bonus, metadata_list


def _load_prime_compute_score(prime_code_dir: str):
    package_dir = Path(prime_code_dir)
    init_file = package_dir / "__init__.py"
    if not init_file.exists():
        raise FileNotFoundError(f"PRIME code reward package not found: {init_file}")
    return functools.partial(_compute_score_local, str(package_dir))


async def batched_prime_code_reward(args, samples: Sample | list[Sample], **kwargs) -> list[float]:
    evaluation = bool(kwargs.get("evaluation", False))
    batch_start = time.perf_counter()
    compute_score = _load_prime_compute_score(
        str(getattr(args, "code_execution_prime_code_dir", DEFAULT_PRIME_CODE_DIR))
    )
    loop = asyncio.get_running_loop()
    reward_executor = _get_prime_reward_executor()

    if isinstance(samples, Sample):
        samples = [samples]

    logger.info("prime_code reward_batch_start samples=%d", len(samples))
    rewards: list[float] = []
    for sample in samples:
        sample_start = time.perf_counter()
        logger.info("prime_code reward_sample_start %s", _sample_log_id(sample))
        score, _metadata = await loop.run_in_executor(
            reward_executor,
            functools.partial(
                compute_score,
                sample.response,
                sample.label,
                continuous=not evaluation,
                apply_bonus=not evaluation,
            ),
        )
        bonus, reward_components = _compute_format_and_tool_bonus(sample.response, apply_bonus=not evaluation)
        reward_components["reward"] = float(score)
        rewards.append(float(score))
        logger.info(
            "prime_code reward_sample_done %s elapsed=%.3fs reward=%.4f base=%.4f think_bonus=%.4f tool_call_bonus=%.4f bonus=%.4f evaluation=%s",
            _sample_log_id(sample),
            time.perf_counter() - sample_start,
            float(score),
            float(score) - bonus,
            reward_components["think_bonus"],
            reward_components["tool_call_bonus"],
            bonus,
            evaluation,
        )
    logger.info(
        "prime_code reward_batch_done samples=%d elapsed=%.3fs",
        len(samples),
        time.perf_counter() - batch_start,
    )
    return rewards


async def prime_code_reward(args, sample: Sample, **kwargs) -> float:
    rewards = await batched_prime_code_reward(args, [sample], **kwargs)
    return float(rewards[0])
