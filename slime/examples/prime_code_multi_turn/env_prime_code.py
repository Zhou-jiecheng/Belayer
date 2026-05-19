from __future__ import annotations

import concurrent.futures
import json
import importlib
import importlib.util
import logging
import math
import os
from pathlib import Path
import random
import re
import threading
import subprocess
import sys
import tempfile
import time
import types
from copy import deepcopy
from typing import Any

try:
    import orjson  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    orjson = None

from examples.prime_code_multi_turn.base_env import BaseInteractionEnv
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
TOOL_CALL_OPEN_RE = re.compile(r"<tool_call>\s*(.*)", re.DOTALL)
CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
SUPPORTED_TOOL_NAMES = {"execute_code"}
DEFAULT_PRIME_CODE_DIR = "/mnt/shared-storage-user/ailab-sys/zhoujiecheng/projs/robust_rl/verl/verl/utils/reward_score/prime_code"
ERROR_INJECTION_ENABLED_ENV = "PRIME_CODE_ERROR_INJECTION_ENABLED"
ERROR_INJECTION_PROB_ENV = "PRIME_CODE_ERROR_INJECTION_PROB"
STOP_ON_FIRST_FAILURE_BATCH_SIZE_ENV = "PRIME_CODE_STOP_ON_FIRST_FAILURE_BATCH_SIZE"
EVAL_THREAD_POOL_WORKERS_ENV = "PRIME_CODE_EVAL_THREAD_POOL_WORKERS"
SHARED_JUDGE_THREAD_POOL_WORKERS_ENV = "PRIME_CODE_SHARED_JUDGE_THREAD_POOL_WORKERS"
DEFAULT_ERROR_INJECTION_ENABLED = True
DEFAULT_ERROR_INJECTION_PROB = 0.1
DEFAULT_STOP_ON_FIRST_FAILURE_BATCH_SIZE = 8
DEFAULT_EVAL_THREAD_POOL_WORKERS = 8

_PRIME_JUDGE_EXECUTOR_LOCK = threading.Lock()
_PRIME_JUDGE_EXECUTORS: dict[int, concurrent.futures.ThreadPoolExecutor] = {}


def _parse_bool_env(name: str, default: bool) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    logger.warning("Invalid boolean env %s=%r, falling back to %s.", name, raw_value, default)
    return default


def _parse_probability_env(name: str, default: float) -> float:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        value = float(raw_value)
    except ValueError:
        logger.warning("Invalid float env %s=%r, falling back to %.3f.", name, raw_value, default)
        return default
    if not 0.0 <= value <= 1.0:
        clamped = min(1.0, max(0.0, value))
        logger.warning("Env %s=%r is outside [0, 1], clamped to %.3f.", name, raw_value, clamped)
        return clamped
    return value


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


def _get_prime_judge_executor(requested_workers: int) -> concurrent.futures.ThreadPoolExecutor:
    max_workers = _parse_positive_int_env(SHARED_JUDGE_THREAD_POOL_WORKERS_ENV, requested_workers)
    with _PRIME_JUDGE_EXECUTOR_LOCK:
        executor = _PRIME_JUDGE_EXECUTORS.get(max_workers)
        if executor is None:
            executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="prime-code-judge",
            )
            _PRIME_JUDGE_EXECUTORS[max_workers] = executor
        return executor


def _load_prime_check_correctness(prime_code_dir: str):
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

    package_dir = Path(prime_code_dir)
    init_file = package_dir / "__init__.py"
    if not init_file.exists():
        raise FileNotFoundError(f"PRIME code judge package not found: {init_file}")

    package_name = "_prime_code_reference"
    if package_name in sys.modules:
        package = sys.modules[package_name]
    else:
        spec = importlib.util.spec_from_file_location(
            package_name,
            init_file,
            submodule_search_locations=[str(package_dir)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load PRIME code judge package from {package_dir}")
        package = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = package
        spec.loader.exec_module(package)

    utils_module = importlib.import_module(f"{package_name}.utils")
    return utils_module.check_correctness


def _canonicalize_test_cases(candidate: dict[str, Any]) -> dict[str, Any]:
    inputs = list(candidate.get("inputs") or [])
    outputs = list(candidate.get("outputs") or [])
    normalized: dict[str, Any] = {"inputs": [], "outputs": []}

    if candidate.get("fn_name") is not None:
        normalized["fn_name"] = str(candidate["fn_name"])
        for item in inputs:
            args = item if isinstance(item, list) else [item]
            normalized["inputs"].append("\n".join(json.dumps(arg, ensure_ascii=False) for arg in args))
        for item in outputs:
            normalized["outputs"].append(json.dumps(item, ensure_ascii=False))
        return normalized

    normalized["inputs"] = [str(item) for item in inputs]
    normalized["outputs"] = [str(item) for item in outputs]
    return normalized


class PrimeCodeEnv(BaseInteractionEnv):
    """
    Multi-turn environment for PRIME coding tasks.

    The actor emits a single tool call that contains Python code. The env executes
    that code against the sample's hidden tests and returns execution feedback.
    """

    def __init__(
        self,
        *,
        test_cases: dict[str, list[str]] | None = None,
        max_turns: int | None = None,
        python_bin: str = "python3",
        exec_timeout: float = 2.0,
        memory_limit_mb: int = 1024,
        max_test_cases: int | None = None,
        max_output_chars: int = 1200,
        stop_on_first_failure: bool = True,
        stop_on_first_failure_batch_size: int = DEFAULT_STOP_ON_FIRST_FAILURE_BATCH_SIZE,
        eval_thread_pool_workers: int = DEFAULT_EVAL_THREAD_POOL_WORKERS,
        prime_code_dir: str = DEFAULT_PRIME_CODE_DIR,
    ):
        self.test_cases = test_cases or {"inputs": [], "outputs": []}
        self.max_turns = max_turns
        self.python_bin = python_bin
        self.exec_timeout = float(exec_timeout)
        self.memory_limit_mb = int(memory_limit_mb)
        self.max_test_cases = max_test_cases
        self.max_output_chars = max_output_chars
        self.stop_on_first_failure = stop_on_first_failure
        self.stop_on_first_failure_batch_size = max(1, int(stop_on_first_failure_batch_size))
        self.eval_thread_pool_workers = max(1, int(eval_thread_pool_workers))
        self.check_correctness = _load_prime_check_correctness(prime_code_dir)
        self.error_injection_enabled = _parse_bool_env(
            ERROR_INJECTION_ENABLED_ENV, DEFAULT_ERROR_INJECTION_ENABLED
        )
        self.error_injection_prob = _parse_probability_env(
            ERROR_INJECTION_PROB_ENV, DEFAULT_ERROR_INJECTION_PROB
        )

        self.turn = 0
        self.tool_calls: list[dict[str, Any]] = []
        self.last_execution: dict[str, Any] | None = None

        logger.info(
            "create PrimeCodeEnv with tests=%d max_turns=%s python_bin=%s timeout=%.2f memory_limit_mb=%d max_test_cases=%s stop_on_first_failure=%s stop_on_first_failure_batch_size=%d eval_thread_pool_workers=%d error_injection_enabled=%s error_injection_prob=%.3f",
            len(self.test_cases.get("inputs", [])),
            self.max_turns,
            self.python_bin,
            self.exec_timeout,
            self.memory_limit_mb,
            self.max_test_cases,
            self.stop_on_first_failure,
            self.stop_on_first_failure_batch_size,
            self.eval_thread_pool_workers,
            self.error_injection_enabled,
            self.error_injection_prob,
        )

    def reset(self):
        self.turn = 0
        self.tool_calls.clear()
        self.last_execution = None
        return {}, {"num_test_cases": len(self.test_cases.get("inputs", []))}

    def close(self):
        return

    def _parse_tool_payload(self, raw_json: str) -> dict[str, Any] | None:
        loader = orjson.loads if orjson is not None else json.loads
        try:
            return loader(raw_json)
        except Exception:
            return None

    def _decode_code_string(self, raw_code: str) -> str:
        try:
            return bytes(raw_code, "utf-8").decode("unicode_escape")
        except Exception:
            return raw_code

    def _sanitize_code_string(self, raw_code: str) -> str:
        cleaned = raw_code.strip()
        cleaned = re.sub(r'(<\|im_end\|>|<\|im_start\|>.*)$', "", cleaned, flags=re.DOTALL).strip()
        cleaned = re.sub(r'^"+\s*python\s*\n', "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^'+\s*python\s*\n", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^```(?:python)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        return cleaned.strip()

    def _extract_code_argument_lenient(self, raw_json: str) -> str:
        greedy_match = re.search(r'"code"\s*:\s*"(.*)"\s*}\s*}?', raw_json, re.DOTALL)
        if greedy_match:
            return self._sanitize_code_string(greedy_match.group(1))

        next_field_match = re.search(r'"code"\s*:\s*"(.*)"\s*,\s*"[A-Za-z_][^"]*"\s*:', raw_json, re.DOTALL)
        if next_field_match:
            return self._sanitize_code_string(next_field_match.group(1))

        strict_match = re.search(r'"code"\s*:\s*"((?:\\.|[^"])*)"', raw_json, re.DOTALL)
        if strict_match:
            return self._sanitize_code_string(self._decode_code_string(strict_match.group(1)))

        start_match = re.search(r'"code"\s*:\s*"', raw_json, re.DOTALL)
        if not start_match:
            return ""

        raw_code = raw_json[start_match.end() :]
        raw_code = re.sub(r'"\s*}\s*}?[\s\n\r\t]*(?:</tool_call>)?[\s\n\r\t]*(?:<\|im_end\|>)?[\s\n\r\t]*$', "", raw_code, flags=re.DOTALL)
        return self._sanitize_code_string(raw_code)

    def _extract_tool_call_payload(self, text: str) -> str | None:
        matches = list(TOOL_CALL_RE.finditer(text))
        if matches:
            return matches[-1].group(1).strip()

        open_matches = list(TOOL_CALL_OPEN_RE.finditer(text))
        if not open_matches:
            return None

        raw_payload = open_matches[-1].group(1)
        terminators = ["</tool_call>", "<|im_end|>", "<|im_start|>user", "<|im_start|>assistant"]
        end = len(raw_payload)
        for marker in terminators:
            idx = raw_payload.find(marker)
            if idx != -1:
                end = min(end, idx)
        return raw_payload[:end].strip()

    def _extract_tool_call_fallback(self, raw_json: str) -> dict[str, Any] | None:
        name_match = re.search(r'"name"\s*:\s*"([^"]+)"', raw_json)
        if not name_match:
            return None

        arguments: dict[str, Any] = {}
        raw_code = self._extract_code_argument_lenient(raw_json)
        if raw_code.strip():
            arguments["code"] = raw_code

        return {"name": name_match.group(1), "arguments": arguments}

    def _extract_tool_call(self, text: str) -> dict[str, Any] | None:
        raw_payload = self._extract_tool_call_payload(text)
        if raw_payload is None:
            return None
        payload = self._parse_tool_payload(raw_payload)
        if payload is None:
            payload = self._extract_tool_call_fallback(raw_payload)
            if payload is None:
                preview = self._preview(raw_payload)
                logger.warning("Failed to decode tool call payload and fallback recovery failed. payload preview=%s", preview)
                return None

        name = payload.get("name") or payload.get("function", {}).get("name")
        arguments = payload.get("arguments") or payload.get("function", {}).get("arguments") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                recovered = self._extract_tool_call_fallback(raw_payload)
                if recovered is None:
                    logger.warning("Tool call arguments are not valid JSON; rejecting tool call.")
                    return None
                arguments = recovered.get("arguments") or arguments

        if not name:
            return None

        return {"name": str(name), "arguments": arguments}

    def _extract_code(self, tool_call: dict[str, Any], response_text: str) -> str:
        arguments = tool_call.get("arguments") or {}
        for key in ("code", "source", "python_code"):
            value = arguments.get(key)
            if isinstance(value, str) and value.strip():
                return self._strip_code_fence(value)

        matches = CODE_BLOCK_RE.findall(response_text)
        if matches:
            return matches[-1].strip()
        return ""

    def _strip_code_fence(self, text: str) -> str:
        stripped = self._sanitize_code_string(text)
        fence_match = CODE_BLOCK_RE.fullmatch(stripped)
        if fence_match:
            return fence_match.group(1).strip()
        return stripped

    def _preview(self, text: str) -> str:
        compact = text.replace("\r\n", "\n").strip()
        if len(compact) <= self.max_output_chars:
            return compact
        return compact[: self.max_output_chars - 3] + "..."

    def _normalize_output(self, text: str) -> str:
        return "\n".join(line.rstrip() for line in text.replace("\r\n", "\n").strip().split("\n")).strip()

    def _parse_float(self, token: str) -> float | None:
        try:
            if token.lower() in {"nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}:
                return None
            return float(token)
        except Exception:
            return None

    def _outputs_match(self, expected: str, actual: str) -> bool:
        expected_norm = self._normalize_output(expected)
        actual_norm = self._normalize_output(actual)
        if expected_norm == actual_norm:
            return True

        expected_tokens = expected_norm.split()
        actual_tokens = actual_norm.split()
        if len(expected_tokens) != len(actual_tokens):
            return False

        for exp, got in zip(expected_tokens, actual_tokens, strict=True):
            if exp == got:
                continue
            exp_float = self._parse_float(exp)
            got_float = self._parse_float(got)
            if exp_float is None or got_float is None:
                return False
            if not math.isclose(exp_float, got_float, rel_tol=1e-4, abs_tol=1e-6):
                return False
        return True

    def _values_match(self, expected: Any, actual: Any) -> bool:
        if isinstance(expected, str) and isinstance(actual, str):
            return self._outputs_match(expected, actual)
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            return math.isclose(float(expected), float(actual), rel_tol=1e-4, abs_tol=1e-6)
        if isinstance(expected, list) and len(expected) == 1 and not isinstance(actual, list):
            return self._values_match(expected[0], actual)
        if isinstance(expected, list) and isinstance(actual, list):
            if len(expected) != len(actual):
                return False
            return all(self._values_match(exp, got) for exp, got in zip(expected, actual, strict=True))
        if isinstance(expected, dict) and isinstance(actual, dict):
            if set(expected) != set(actual):
                return False
            return all(self._values_match(expected[key], actual[key]) for key in expected)
        return expected == actual

    def _build_preexec_fn(self):
        if os.name != "posix":
            return None

        memory_limit_bytes = self.memory_limit_mb * 1024 * 1024
        cpu_limit_seconds = max(1, int(math.ceil(self.exec_timeout)))

        def _limit_resources():
            import resource

            resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit_seconds, cpu_limit_seconds + 1))
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))
            resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 1024 * 1024, 8 * 1024 * 1024))

        return _limit_resources

    def _run_code_once(self, code: str, stdin_text: str) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="prime_code_env_") as temp_dir:
            script_path = os.path.join(temp_dir, "solution.py")
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(code)

            try:
                proc = subprocess.run(
                    [self.python_bin, script_path],
                    input=stdin_text,
                    capture_output=True,
                    text=True,
                    cwd=temp_dir,
                    timeout=self.exec_timeout,
                    preexec_fn=self._build_preexec_fn(),
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                )
            except subprocess.TimeoutExpired as exc:
                return {
                    "status": "timeout",
                    "stdout": exc.stdout or "",
                    "stderr": exc.stderr or f"Timed out after {self.exec_timeout:.2f} seconds.",
                    "returncode": None,
                }
            except Exception as exc:  # pragma: no cover - defensive
                return {
                    "status": "error",
                    "stdout": "",
                    "stderr": str(exc),
                    "returncode": None,
                }

            status = "ok" if proc.returncode == 0 else "runtime_error"
            return {
                "status": status,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "returncode": proc.returncode,
            }

    def _run_function_once(self, code: str, fn_name: str, fn_input: Any) -> dict[str, Any]:
        payload = json.dumps(fn_input)
        wrapper = (
            f"{code}\n\n"
            "import json\n"
            "import sys\n"
            f"_fn = {fn_name}\n"
            f"_payload = json.loads({payload!r})\n"
            "if isinstance(_payload, list):\n"
            "    _result = _fn(*_payload)\n"
            "else:\n"
            "    _result = _fn(_payload)\n"
            "print(json.dumps(_result, ensure_ascii=False))\n"
        )
        return self._run_code_once(wrapper, "")

    def _classify_prime_status(self, res: list[Any], metadata: dict[str, Any]) -> str:
        if all(item is True for item in res):
            return "ok"
        if any(item == -2 for item in res):
            return "compile_error"
        if any(item == -1 for item in res):
            return "runtime_error"
        if metadata.get("error_message") == "Wrong Answer":
            return "wrong_answer"
        return "failed"

    def _run_prime_check(self, code: str, test_spec: dict[str, Any]) -> tuple[list[Any], dict[str, Any]]:
        start_time = time.perf_counter()
        res, metadata_list = self.check_correctness(
            in_outs=test_spec,
            generation=code,
            timeout=max(1, int(math.ceil(self.exec_timeout))),
            debug=False,
        )
        metadata = metadata_list[0] if metadata_list else {}
        elapsed = time.perf_counter() - start_time
        num_cases = min(len(test_spec.get("inputs") or []), len(test_spec.get("outputs") or []))
        logger.info(
            "prime_code judge_check turn=%d elapsed=%.3fs cases=%d timeout=%ss status=%s",
            self.turn,
            elapsed,
            num_cases,
            max(1, int(math.ceil(self.exec_timeout))),
            self._classify_prime_status(list(res), metadata),
        )
        return list(res), metadata

    def _build_test_spec(
        self,
        inputs: list[str],
        outputs: list[str],
        start: int,
        end: int,
        fn_name: str | None,
    ) -> dict[str, Any]:
        test_spec: dict[str, Any] = {
            "inputs": inputs[start:end],
            "outputs": outputs[start:end],
        }
        if fn_name is not None:
            test_spec["fn_name"] = fn_name
        return test_spec

    def _run_prime_checks_parallel(
        self,
        code: str,
        jobs: list[tuple[int, int, dict[str, Any]]],
    ) -> list[tuple[int, int, list[Any], dict[str, Any]]]:
        if not jobs:
            return []

        start_time = time.perf_counter()
        if len(jobs) == 1 or self.eval_thread_pool_workers <= 1:
            start, end, test_spec = jobs[0]
            res, metadata = self._run_prime_check(code, test_spec)
            logger.info(
                "prime_code judge_batch_parallel turn=%d jobs=%d workers=%d shared_executor=%s elapsed=%.3fs cases=%d",
                self.turn,
                1,
                1,
                False,
                time.perf_counter() - start_time,
                end - start,
            )
            return [(start, end, res, metadata)]

        max_workers = min(self.eval_thread_pool_workers, len(jobs))
        ordered_results: list[tuple[int, int, list[Any], dict[str, Any]] | None] = [None] * len(jobs)
        executor = _get_prime_judge_executor(max_workers)
        future_to_index = {
            executor.submit(self._run_prime_check, code, test_spec): job_index
            for job_index, (_start, _end, test_spec) in enumerate(jobs)
        }
        for future in concurrent.futures.as_completed(future_to_index):
            job_index = future_to_index[future]
            start, end, _ = jobs[job_index]
            res, metadata = future.result()
            ordered_results[job_index] = (start, end, res, metadata)

        logger.info(
            "prime_code judge_batch_parallel turn=%d jobs=%d workers=%d shared_executor=%s elapsed=%.3fs total_cases=%d",
            self.turn,
            len(jobs),
            max_workers,
            True,
            time.perf_counter() - start_time,
            sum(end - start for start, end, _spec in jobs),
        )
        return [item for item in ordered_results if item is not None]

    def _locate_first_failure_metadata(
        self,
        code: str,
        inputs: list[str],
        outputs: list[str],
        failed_index: int,
        fn_name: str | None,
    ) -> dict[str, Any]:
        case_spec = self._build_test_spec(inputs, outputs, failed_index, failed_index + 1, fn_name)
        _res, metadata = self._run_prime_check(code, case_spec)
        return metadata

    def _evaluate_stop_on_first_failure(
        self,
        code: str,
        inputs: list[str],
        outputs: list[str],
        total_to_run: int,
        fn_name: str | None,
    ) -> tuple[list[Any], dict[str, Any]]:
        if total_to_run <= 0:
            return [], {}

        start_time = time.perf_counter()
        batch_size = min(self.stop_on_first_failure_batch_size, total_to_run)
        jobs = [
            (
                batch_start,
                min(total_to_run, batch_start + batch_size),
                self._build_test_spec(inputs, outputs, batch_start, min(total_to_run, batch_start + batch_size), fn_name),
            )
            for batch_start in range(0, total_to_run, batch_size)
        ]
        batch_results = self._run_prime_checks_parallel(code, jobs)
        coarse_elapsed = time.perf_counter() - start_time
        logger.info(
            "prime_code evaluate_stop_on_first_failure turn=%d stage=coarse elapsed=%.3fs total_to_run=%d batch_size=%d batches=%d",
            self.turn,
            coarse_elapsed,
            total_to_run,
            batch_size,
            len(jobs),
        )
        res: list[Any] = []
        metadata: dict[str, Any] = {}

        for batch_start, batch_end, batch_res, batch_metadata in batch_results:
            expected_batch_len = batch_end - batch_start
            batch_passed = len(batch_res) == expected_batch_len and all(item is True for item in batch_res)
            if batch_passed:
                res.extend(batch_res)
                metadata = batch_metadata
                continue

            # A failing batch is refined one case at a time so we still report the first failing case accurately,
            # while avoiding one subprocess launch per case on the common passing path.
            case_jobs = [
                (
                    case_index,
                    case_index + 1,
                    self._build_test_spec(inputs, outputs, case_index, case_index + 1, fn_name),
                )
                for case_index in range(batch_start, batch_end)
            ]
            refine_start = time.perf_counter()
            case_results = self._run_prime_checks_parallel(code, case_jobs)
            logger.info(
                "prime_code evaluate_stop_on_first_failure turn=%d stage=refine elapsed=%.3fs batch_range=[%d,%d)",
                self.turn,
                time.perf_counter() - refine_start,
                batch_start,
                batch_end,
            )
            for case_index, _case_end, case_res, case_metadata in case_results:
                metadata = case_metadata
                if not case_res:
                    res.append(-1)
                    logger.info(
                        "prime_code evaluate_stop_on_first_failure turn=%d first_failure_case=%d total_elapsed=%.3fs",
                        self.turn,
                        case_index,
                        time.perf_counter() - start_time,
                    )
                    return res, metadata
                res.extend(case_res)
                if any(item is not True for item in case_res):
                    logger.info(
                        "prime_code evaluate_stop_on_first_failure turn=%d first_failure_case=%d total_elapsed=%.3fs",
                        self.turn,
                        case_index,
                        time.perf_counter() - start_time,
                    )
                    return res, metadata

            # Defensive fallback: if the coarse batch failed but refinement did not find a failing case,
            # preserve the batch metadata and stop instead of silently masking an inconsistency.
            metadata = batch_metadata
            if batch_res:
                res.extend(batch_res[: max(0, expected_batch_len - len(res) + batch_start)])
            logger.info(
                "prime_code evaluate_stop_on_first_failure turn=%d inconsistent_batch batch_range=[%d,%d) total_elapsed=%.3fs",
                self.turn,
                batch_start,
                batch_end,
                time.perf_counter() - start_time,
            )
            return res, metadata

        logger.info(
            "prime_code evaluate_stop_on_first_failure turn=%d completed elapsed=%.3fs executed_cases=%d",
            self.turn,
            time.perf_counter() - start_time,
            len(res),
        )
        return res, metadata

    def _evaluate_all_cases_parallel(
        self,
        code: str,
        inputs: list[str],
        outputs: list[str],
        total_to_run: int,
        fn_name: str | None,
    ) -> tuple[list[Any], dict[str, Any]]:
        if total_to_run <= 0:
            return [], {}

        start_time = time.perf_counter()
        batch_size = min(self.stop_on_first_failure_batch_size, total_to_run)
        jobs = [
            (
                batch_start,
                min(total_to_run, batch_start + batch_size),
                self._build_test_spec(inputs, outputs, batch_start, min(total_to_run, batch_start + batch_size), fn_name),
            )
            for batch_start in range(0, total_to_run, batch_size)
        ]
        batch_results = self._run_prime_checks_parallel(code, jobs)
        logger.info(
            "prime_code evaluate_all_cases turn=%d elapsed=%.3fs total_to_run=%d batch_size=%d batches=%d",
            self.turn,
            time.perf_counter() - start_time,
            total_to_run,
            batch_size,
            len(jobs),
        )

        res: list[Any] = []
        metadata_by_batch: list[dict[str, Any]] = []
        for _batch_start, _batch_end, batch_res, batch_metadata in batch_results:
            res.extend(batch_res)
            metadata_by_batch.append(batch_metadata)

        first_failed_index = next((idx for idx, item in enumerate(res) if item is not True), None)
        if first_failed_index is not None:
            return res, self._locate_first_failure_metadata(code, inputs, outputs, first_failed_index, fn_name)

        return res, metadata_by_batch[0] if metadata_by_batch else {}

    def _random_hex_address(self, rng: random.Random) -> str:
        return f"0x{rng.getrandbits(48):012x}"

    def _build_injected_error_trace(
        self, error_kind: str, code: str, case_input: str, rng: random.Random
    ) -> tuple[str, int]:
        code_lines = max(1, len(code.splitlines()))
        top_line = min(code_lines, max(1, code_lines // 3))
        inner_line = min(code_lines, max(top_line, (2 * code_lines) // 3))
        pid = rng.randint(10000, 99999)
        temp_dir = os.path.join(tempfile.gettempdir(), f"prime_code_env_{pid}_{rng.randint(1000, 9999)}")
        script_path = os.path.join(temp_dir, "solution.py")
        py_bin_name = Path(self.python_bin).name or "python3"
        vm_kb = max(262144, self.memory_limit_mb * 1024 + rng.randint(1024, 65536))
        rss_kb = max(32768, min(vm_kb, int(self.memory_limit_mb * 768) + rng.randint(2048, 32768)))

        if error_kind == "oom_killed":
            stderr = "\n".join(
                [
                    "Traceback (most recent call last):",
                    f'  File "{script_path}", line {inner_line}, in solve',
                    "    dp = [[0] * (n + 1) for _ in range(n + 1)]",
                    "MemoryError",
                    "",
                    "During handling of the above exception, another exception occurred:",
                    "",
                    "Traceback (most recent call last):",
                    f'  File "{script_path}", line {code_lines}, in <module>',
                    "    solve()",
                    f'  File "{script_path}", line {top_line}, in solve',
                    "    raise SystemExit(main())",
                    "SystemExit: 1",
                    "",
                    f"{py_bin_name}: line 1: {pid} Killed                  {self.python_bin} {script_path}",
                    f"[  +0.000091] Memory cgroup out of memory: Killed process {pid} ({py_bin_name}) total-vm:{vm_kb}kB, anon-rss:{rss_kb}kB, file-rss:512kB, shmem-rss:0kB, UID:0 pgtables:1024kB oom_score_adj:0",
                    f"[  +0.000014] oom_reaper: reaped process {pid} ({py_bin_name}), now anon-rss:0kB, file-rss:0kB, shmem-rss:0kB",
                ]
            )
            return stderr, 137

        libc_frame = f"/lib/x86_64-linux-gnu/libc.so.6(+0x{rng.randint(0x10000, 0x8FFFF):x}) [{self._random_hex_address(rng)}]"
        python_frame = f"/usr/bin/{py_bin_name}(PyEval_EvalFrameDefault+0x{rng.randint(0x100, 0x8000):x}) [{self._random_hex_address(rng)}]"

        if error_kind == "segmentation_fault":
            stderr = "\n".join(
                [
                    "Fatal Python error: Segmentation fault",
                    "",
                    "Current thread 0x00007f4a2c7fe740 (most recent call first):",
                    f'  File "{script_path}", line {inner_line}, in solve',
                    f'  File "{script_path}", line {code_lines}, in <module>',
                    "",
                    "Extension modules: _json, math (total: 2)",
                    "",
                    "C stack trace (most recent call first):",
                    f"  {libc_frame}",
                    f"  {python_frame}",
                    f"  /usr/bin/{py_bin_name}(PyObject_Call+0x{rng.randint(0x40, 0x400):x}) [{self._random_hex_address(rng)}]",
                    f"  /usr/bin/{py_bin_name}(PyRun_SimpleFileExFlags+0x{rng.randint(0x40, 0x300):x}) [{self._random_hex_address(rng)}]",
                    f"{py_bin_name}: line 1: {pid} Segmentation fault (core dumped) {self.python_bin} {script_path}",
                ]
            )
            return stderr, -11

        if error_kind == "bus_error":
            stderr = "\n".join(
                [
                    "Fatal Python error: Bus error",
                    "",
                    "Current thread 0x00007fb03d12a740 (most recent call first):",
                    f'  File "{script_path}", line {inner_line}, in solve',
                    f'  File "{script_path}", line {code_lines}, in <module>',
                    "",
                    "Native stack trace:",
                    f"  {libc_frame}",
                    f"  {python_frame}",
                    f"  /usr/bin/{py_bin_name}(Py_BytesMain+0x{rng.randint(0x40, 0x300):x}) [{self._random_hex_address(rng)}]",
                    f"[  +0.000018] traps: {py_bin_name}[{pid}] general protection fault ip:{self._random_hex_address(rng)} sp:{self._random_hex_address(rng)} error:0 in libpython3.10.so.1.0[{self._random_hex_address(rng)}+0x{rng.randint(0x1000, 0x9000):x}]",
                    f"{py_bin_name}: line 1: {pid} Bus error               (core dumped) {self.python_bin} {script_path}",
                ]
            )
            return stderr, -7

        stderr = "\n".join(
            [
                "Fatal Python error: Illegal instruction",
                "",
                "Current thread 0x00007f8e9da42740 (most recent call first):",
                f'  File "{script_path}", line {inner_line}, in solve',
                f'  File "{script_path}", line {code_lines}, in <module>',
                "",
                "Native stack trace:",
                f"  {libc_frame}",
                f"  {python_frame}",
                f"  /usr/bin/{py_bin_name}(_PyEval_EvalFrameDefault+0x{rng.randint(0x100, 0x6000):x}) [{self._random_hex_address(rng)}]",
                f"[  +0.000012] traps: {py_bin_name}[{pid}] invalid opcode ip:{self._random_hex_address(rng)} sp:{self._random_hex_address(rng)} error:0 in libpython3.10.so.1.0[{self._random_hex_address(rng)}+0x{rng.randint(0x1000, 0x9000):x}]",
                f"{py_bin_name}: line 1: {pid} Illegal instruction     (core dumped) {self.python_bin} {script_path}",
            ]
        )
        return stderr, -4

    def _maybe_build_injected_execution(
        self, code: str, total_cases: int, total_to_run: int, inputs: list[str], outputs: list[str]
    ) -> dict[str, Any] | None:
        if total_to_run == 0 or not self.error_injection_enabled:
            return None
        if random.random() >= self.error_injection_prob:
            return None

        error_kind = random.choice(("oom_killed", "segmentation_fault", "bus_error", "illegal_instruction"))
        rng = random.Random(f"{self.turn}:{total_cases}:{len(code)}:{error_kind}:{random.random()}")
        failed_index = 0
        stderr, returncode = self._build_injected_error_trace(
            error_kind, code, inputs[failed_index] if inputs else "", rng
        )
        execution_status = {
            "oom_killed": "oom_killed",
            "segmentation_fault": "segmentation_fault",
            "bus_error": "bus_error",
            "illegal_instruction": "illegal_instruction",
        }[error_kind]
        return {
            "total_cases": total_cases,
            "executed_cases": 1,
            "passed_cases": 0,
            "evaluated_all_cases": total_to_run == total_cases and total_cases == 1,
            "all_passed": False,
            "execution_status": execution_status,
            "error_injected": True,
            "injected_error_type": error_kind,
            "prime_metadata": {
                "error_injected": True,
                "error_type": error_kind,
                "returncode": returncode,
            },
            "first_failure": {
                "case_index": failed_index,
                "status": execution_status,
                "returncode": returncode,
                "stdin": self._preview(inputs[failed_index] if inputs else ""),
                "expected": self._preview(outputs[failed_index] if outputs else ""),
                "actual": "",
                "stderr": stderr,
                "stderr_full": stderr,
            },
        }

    def _evaluate_code(self, code: str) -> dict[str, Any]:
        start_time = time.perf_counter()
        inputs = list(self.test_cases.get("inputs") or [])
        outputs = list(self.test_cases.get("outputs") or [])
        total_cases = min(len(inputs), len(outputs))
        if self.max_test_cases is not None:
            total_to_run = min(total_cases, self.max_test_cases)
        else:
            total_to_run = total_cases

        result: dict[str, Any] = {
            "total_cases": total_cases,
            "executed_cases": 0,
            "passed_cases": 0,
            "evaluated_all_cases": total_to_run == total_cases,
            "all_passed": False,
            "first_failure": None,
            "execution_status": "ok",
        }

        if total_cases == 0:
            result["all_passed"] = False
            result["execution_status"] = "missing_tests"
            logger.info("prime_code evaluate_code turn=%d elapsed=%.3fs total_cases=0 status=missing_tests", self.turn, time.perf_counter() - start_time)
            return result

        injected_result = self._maybe_build_injected_execution(code, total_cases, total_to_run, inputs, outputs)
        if injected_result is not None:
            logger.info(
                "Injected synthetic system error type=%s turn=%d total_cases=%d elapsed=%.3fs",
                injected_result.get("injected_error_type"),
                self.turn,
                total_cases,
                time.perf_counter() - start_time,
            )
            return injected_result

        fn_name = self.test_cases.get("fn_name")
        if self.stop_on_first_failure:
            res, metadata = self._evaluate_stop_on_first_failure(code, inputs, outputs, total_to_run, fn_name)
        else:
            res, metadata = self._evaluate_all_cases_parallel(code, inputs, outputs, total_to_run, fn_name)

        executed_cases = len(res)
        passed_cases = sum(item is True for item in res)

        result["executed_cases"] = executed_cases
        result["passed_cases"] = passed_cases
        result["all_passed"] = executed_cases > 0 and passed_cases == executed_cases
        result["execution_status"] = self._classify_prime_status(res, metadata)
        result["prime_metadata"] = metadata

        if not result["all_passed"]:
            failed_index = next((idx for idx, item in enumerate(res) if item is not True), 0)
            result["first_failure"] = {
                "case_index": failed_index,
                "status": result["execution_status"],
                "returncode": None,
                "stdin": self._preview(str(metadata.get("inputs", inputs[failed_index] if failed_index < len(inputs) else ""))),
                "expected": self._preview(str(metadata.get("expected", outputs[failed_index] if failed_index < len(outputs) else ""))),
                "actual": self._preview(str(metadata.get("output", ""))),
                "stderr": self._preview(str(metadata.get("traceback", metadata.get("error", "")))),
            }
        logger.info(
            "prime_code evaluate_code turn=%d elapsed=%.3fs total_cases=%d run_cases=%d executed_cases=%d passed_cases=%d all_passed=%s status=%s stop_on_first_failure=%s",
            self.turn,
            time.perf_counter() - start_time,
            total_cases,
            total_to_run,
            executed_cases,
            passed_cases,
            result["all_passed"],
            result["execution_status"],
            self.stop_on_first_failure,
        )
        return result

    def _build_tool_feedback(self, execution: dict[str, Any]) -> str:
        turn_idx = self.turn - 1
        last_warning_turn = None
        if self.max_turns is not None:
            last_warning_turn = self.max_turns - 2 if self.max_turns >= 2 else self.max_turns - 1
        is_last_tool_turn = last_warning_turn is not None and turn_idx >= last_warning_turn

        total_cases = execution["total_cases"]
        executed_cases = execution["executed_cases"]
        passed_cases = execution["passed_cases"]

        if execution["execution_status"] == "missing_tests":
            return (
                "execute_code result: no test cases were available in the sample metadata, so correctness could not be checked. "
                "Do not rely on this tool result."
            )

        if execution["all_passed"] and execution["evaluated_all_cases"]:
            return (
                f"execute_code result: passed {passed_cases}/{executed_cases} tests. "
                "The current Python code is correct on all available tests. "
                "Do not call the tool again. Provide your final answer as a single ```python``` code block."
            )

        if execution["all_passed"]:
            return (
                f"execute_code result: passed {passed_cases}/{executed_cases} executed tests, "
                f"but only a subset of {total_cases} total tests was checked. "
                "The code looks promising but is not fully verified yet."
            )

        failure = execution["first_failure"] or {}
        feedback = [
            f"execute_code result: passed {passed_cases}/{executed_cases} executed tests.",
            f"First failing case index: {failure.get('case_index', -1)}.",
            f"Execution status: {failure.get('status', 'unknown')}.",
            f"stdin preview:\n{failure.get('stdin', '')}",
            f"expected stdout:\n{failure.get('expected', '')}",
            f"actual stdout:\n{failure.get('actual', '')}",
        ]
        stderr_text = failure.get("stderr_full") or failure.get("stderr")
        if stderr_text:
            feedback.append(f"stderr:\n{stderr_text}")
        feedback.append("Revise the code and call the tool again if needed.")
        return "\n".join(feedback)

    def step(self, response_text: str):
        self.turn += 1
        is_final_turn = self.max_turns is not None and self.turn >= self.max_turns

        tool_call = self._extract_tool_call(response_text)
        info: dict[str, Any] = {"tool_call": deepcopy(tool_call)}

        if not tool_call:
            info["tool_executed"] = False
            obs = {
                "obs_str": "No tool call detected; ending the episode.",
                "role": "tool",
            }
            return obs, True, info

        name = (tool_call.get("name") or "").strip()
        if name not in SUPPORTED_TOOL_NAMES:
            obs = {
                "obs_str": (
                    f"Tool `{name}` is not supported. "
                    'Use exactly one tool named `execute_code` and emit exactly this shape: '
                    '<tool_call>{"name": "execute_code", "arguments": {"code": "..."}}</tool_call>.'
                ),
                "role": "tool",
            }
            info["tool_executed"] = False
            return obs, is_final_turn, info

        code = self._extract_code(tool_call, response_text)
        if not code.strip():
            obs = {
                "obs_str": (
                    "Tool call detected but no Python code was provided. "
                    'Use exactly this format: <tool_call>{"name": "execute_code", "arguments": {"code": "..."}}</tool_call>.'
                ),
                "role": "tool",
            }
            info["tool_executed"] = False
            info["code_missing"] = True
            return obs, is_final_turn, info

        execution = self._evaluate_code(code)
        self.last_execution = execution
        tool_record = {
            "name": name,
            "code_preview": self._preview(code),
            "passed_cases": execution["passed_cases"],
            "executed_cases": execution["executed_cases"],
            "all_passed": execution["all_passed"],
        }
        self.tool_calls.append(tool_record)
        info.update(tool_record)
        info["tool_executed"] = True
        info["execution"] = execution

        obs = {
            "obs_str": self._build_tool_feedback(execution),
            "role": "tool",
            "tool_score": 1.0 if execution["all_passed"] and execution["evaluated_all_cases"] else 0.0,
        }
        return obs, is_final_turn, info


def _parse_test_cases(sample: Sample | None) -> dict[str, list[str]] | None:
    if sample is None:
        return None

    metadata = sample.metadata or {}
    candidate = metadata.get("test_cases")
    if candidate is None:
        candidate = metadata.get("test_cases_json")
    if candidate is None and sample.label is not None:
        candidate = sample.label

    if candidate is None:
        return None

    if isinstance(candidate, str):
        loader = orjson.loads if orjson is not None else json.loads
        candidate = loader(candidate)

    if not isinstance(candidate, dict):
        return None

    normalized = _canonicalize_test_cases(candidate)
    inputs = list(normalized.get("inputs") or [])
    outputs = list(normalized.get("outputs") or [])
    if not inputs or not outputs:
        return None
    return normalized


def build_env(
    sample: Sample | None = None,
    args: Any | None = None,
    evaluation: bool = False,
    **_: Any,
) -> PrimeCodeEnv:
    max_turns = getattr(args, "max_turns", None)
    if max_turns is None:
        raise ValueError("max_turns must be set via --custom-config-path in the custom config file.")

    test_cases = _parse_test_cases(sample)
    if test_cases is None:
        logger.warning("No executable test cases were found in sample metadata or label.")

    max_test_cases = getattr(args, "code_execution_max_cases", None)
    if evaluation:
        max_test_cases = getattr(args, "code_execution_eval_max_cases", max_test_cases)

    return PrimeCodeEnv(
        test_cases=test_cases,
        max_turns=max_turns,
        python_bin=getattr(args, "code_execution_python_bin", "python3"),
        exec_timeout=float(getattr(args, "code_execution_timeout", 2.0)),
        memory_limit_mb=int(getattr(args, "code_execution_memory_mb", 1024)),
        max_test_cases=max_test_cases,
        max_output_chars=int(getattr(args, "code_execution_max_output_chars", 1200)),
        stop_on_first_failure=bool(getattr(args, "code_execution_stop_on_first_failure", True)),
        stop_on_first_failure_batch_size=int(
            getattr(
                args,
                "code_execution_stop_on_first_failure_batch_size",
                _parse_positive_int_env(
                    STOP_ON_FIRST_FAILURE_BATCH_SIZE_ENV, DEFAULT_STOP_ON_FIRST_FAILURE_BATCH_SIZE
                ),
            )
        ),
        eval_thread_pool_workers=int(
            getattr(
                args,
                "code_execution_eval_thread_pool_workers",
                _parse_positive_int_env(EVAL_THREAD_POOL_WORKERS_ENV, DEFAULT_EVAL_THREAD_POOL_WORKERS),
            )
        ),
        prime_code_dir=str(getattr(args, "code_execution_prime_code_dir", DEFAULT_PRIME_CODE_DIR)),
    )
