from __future__ import annotations

import json
import logging
import random
import os
import re
import socket
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

try:
    import orjson  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    orjson = None
from examples.geo3k_vlm_multi_turn.base_env import BaseInteractionEnv

from slime.rollout.rm_hub import grade_answer_verl
from slime.rollout.rm_hub.math_utils import extract_answer as extract_boxed_answer
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

# Matches the JSON payload emitted between <tool_call> ... </tool_call> tags.
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
# Accept either name; verl uses `calc_geo3k_reward` while the instruction refers to `calc_score`.
SUPPORTED_TOOL_NAMES = {"calc_score", "calc_geo3k_reward"}
FAULT_TYPES = ("oom", "segmentation_fault", "bus_error", "illegal_instruction", "rpc_timeout")


class Geo3kEnv(BaseInteractionEnv):
    """
    Minimal interaction environment for multi-turn geo3k with a scoring tool.

    The model is expected to emit a <tool_call>{...}</tool_call> payload that includes
    an `answer` argument. We run the math reward checker against the ground truth and
    return the score as the next observation. The episode ends immediately after each
    step; responses are provided but no further turns are taken.
    """

    def __init__(
        self,
        *,
        ground_truth: str | None = None,
        max_turns: int | None = None,
        fault_injection_enabled: bool = True,
        fault_injection_prob: float = 0.1,
    ):
        self.ground_truth = str(ground_truth) if ground_truth is not None else None
        self.tool_calls: list[dict[str, Any]] = []
        self.last_tool_score: float | None = None
        self.turn = 0
        self.max_turns = max_turns
        self.fault_injection_enabled = fault_injection_enabled
        self.fault_injection_prob = max(0.0, min(1.0, float(fault_injection_prob)))
        self.total_step_calls = 0
        logger.info("create Geo3kEnv with ground_truth=%s, max_turns=%s, fault_injection_enabled=%s, fault_injection_prob=%.3f",
                self.ground_truth, self.max_turns, self.fault_injection_enabled, self.fault_injection_prob)

    def reset(self):
        self.tool_calls.clear()
        self.last_tool_score = None
        self.turn = 0
        # No initial observation is needed; the question lives in the prompt.
        observation: dict[str, Any] = {}
        reset_info = {"ground_truth_available": self.ground_truth is not None}
        return observation, reset_info

    def close(self):
        """No resources to release."""
        return

    def _extract_tool_call(self, text: str) -> dict[str, Any] | None:
        """
        Parse the latest tool call payload from the assistant response.
        Supports the <tool_call>{...}</tool_call> convention used in the
        SGLang multi-turn templates. Tool tags are mandatory.
        """
        matches = list(TOOL_CALL_RE.finditer(text))
        raw_json = None
        if matches:
            raw_json = matches[-1].group(1).strip()

        if raw_json is None:
            return None

        payload = self._parse_tool_payload(raw_json)
        if payload is None:
            return None

        name = payload.get("name") or payload.get("function", {}).get("name")
        arguments = payload.get("arguments") or payload.get("function", {}).get("arguments") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                logger.warning("Tool call arguments are not valid JSON; rejecting tool call.")
                return None

        if not name:
            return None
        return {"name": name, "arguments": arguments}

    def _score_answer(self, answer: str) -> float:
        """
        Use the same logic as the single-turn math reward model.
        We accept either boxed or raw numeric strings by retrying with a boxed wrapper.
        """
        if not self.ground_truth:
            return 0.0

        answer = answer.strip()
        candidates = [answer]
        if "\\boxed" not in answer:
            candidates.append(f"\\boxed{{{answer}}}")

        for candidate in candidates:
            try:
                if grade_answer_verl(candidate, self.ground_truth):
                    return 1.0
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("grade_answer_verl failed on %s: %s", candidate, exc)
                continue
        return 0.0

    def _extract_answer_from_text(self, text: str) -> str | None:
        """
        Prefer a concise answer by pulling the last \\boxed{} chunk; fall back to the last
        non-empty line (capped) to avoid echoing the whole response body.
        """
        boxed = extract_boxed_answer(text)
        if boxed:
            return str(boxed).strip()
        for line in reversed(text.splitlines()):
            cleaned = line.strip()
            if cleaned:
                return cleaned[:512]
        trimmed = text.strip()
        return trimmed[:512] if trimmed else None

    def _extract_balanced_json(self, text: str, start: int) -> str | None:
        """
        Best-effort balanced brace extraction starting at `start` (index of an opening '{').
        Keeps string-awareness to avoid terminating inside quoted braces.
        """
        depth = 0
        in_string = False
        escaped = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if ch == "\\" and not escaped:
                escaped = True
                continue
            if ch == '"' and not escaped:
                in_string = not in_string
            if not in_string:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start : idx + 1]
            escaped = False
        return None

    def _build_tool_feedback(self, score: float, parsed_answer: str) -> str:
        """
        Provide concise feedback for the model to continue reasoning.
        """
        turn_idx = self.turn - 1  # zero-based
        # Send the final reminder one turn before the true last turn so the model sees it in time.
        last_warning_turn = None
        if self.max_turns is not None:
            if self.max_turns >= 2:
                last_warning_turn = self.max_turns - 2
            else:
                last_warning_turn = self.max_turns - 1
        is_final_turn = last_warning_turn is not None and turn_idx >= last_warning_turn

        if score == 1.0:
            return (
                f"calc_score result: {score}. Parsed answer '{parsed_answer}' matches the reference. "
                "You can now stop reasoning and provide the final solution in \\boxed{}."
            )
        if score == 0.0:
            if is_final_turn:
                return (
                    f"calc_score result: {score}. Parsed answer '{parsed_answer}' does not match the reference. "
                    "Your answer is wrong. You may need to reason in a different way. Don't repeat your answer unless necessary. "
                    "Since you only have one chance to answer, don't call tool again. You should provide your final answer in the form below Answer: \\boxed{$Answer} where $Answer is your fiinal answer to this problem."
                )
            return (
                f"calc_score result: {score}. Parsed answer '{parsed_answer}' does not match the reference. "
                "Your answer is wrong. You may need to reason in a different way. Don't repeat your answer unless necessary."
            )

    def _preview_text(self, text: str, limit: int = 120) -> str:
        compact = " ".join(text.split())
        if len(compact) <= limit:
            return compact
        return compact[: limit - 3] + "..."

    def _utc_timestamp(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    def _random_hex_address(self) -> str:
        return f"0x{random.getrandbits(48):012x}"

    def _build_fault_message(self, tool_name: str, parsed_answer: str, response_text: str) -> tuple[str, dict[str, Any]]:
        fault_type = random.choice(FAULT_TYPES)
        request_id = f"geo3k-{uuid.uuid4().hex[:12]}"
        worker_id = f"rm-worker-{random.randint(0, 7)}"
        hostname = socket.gethostname()
        pid = random.randint(12000, 65000)
        answer_preview = self._preview_text(parsed_answer or "<empty>", limit=96)
        response_preview = self._preview_text(response_text, limit=160)
        timestamp = self._utc_timestamp()
        turn_idx = self.turn
        step_idx = self.total_step_calls

        lines = [
            f"[{timestamp}] INFO geo3k.reward_client request_id={request_id} step={step_idx} turn={turn_idx} tool={tool_name}",
            f"[{timestamp}] INFO geo3k.reward_client answer_preview={answer_preview!r}",
            f"[{timestamp}] DEBUG geo3k.reward_client response_preview={response_preview!r}",
            f"[{timestamp}] INFO geo3k.reward_client dispatch target={worker_id}@{hostname} pid={pid}",
        ]

        if fault_type == "oom":
            lines.extend(
                [
                    f"[{timestamp}] ERROR geo3k.reward_worker request_id={request_id} status=RESOURCE_EXHAUSTED detail='scoring worker ran out of memory while normalizing candidate expression tree'",
                    "Traceback (most recent call last):",
                    '  File "/opt/geo3k/reward_server/worker.py", line 218, in handle_calc_score',
                    "    normalized = normalize_expression(parsed_answer)",
                    '  File "/opt/geo3k/reward_server/symbolic.py", line 147, in normalize_expression',
                    "    cache = np.empty((268435456,), dtype=np.float64)",
                    "numpy.core._exceptions._ArrayMemoryError: Unable to allocate 2.00 GiB for an array with shape (268435456,) and data type float64",
                    "",
                    "During handling of the above exception, another exception occurred:",
                    "",
                    "Traceback (most recent call last):",
                    '  File "/opt/geo3k/reward_server/main.py", line 94, in dispatch_request',
                    "    return worker.handle_calc_score(payload)",
                    '  File "/opt/geo3k/reward_server/worker.py", line 226, in handle_calc_score',
                    "    raise RuntimeError('calc_score backend OOM')",
                    "RuntimeError: calc_score backend OOM",
                    f"[  +0.000021] Memory cgroup out of memory: Killed process {pid} (python3) total-vm:{random.randint(2200000, 3200000)}kB, anon-rss:{random.randint(800000, 1500000)}kB, file-rss:512kB, shmem-rss:0kB, UID:0 pgtables:4096kB oom_score_adj:0",
                ]
            )
            return "\n".join(lines), {
                "fault_type": fault_type,
                "request_id": request_id,
                "worker_id": worker_id,
                "pid": pid,
            }

        if fault_type == "segmentation_fault":
            lines.extend(
                [
                    f"[{timestamp}] ERROR geo3k.reward_worker request_id={request_id} status=INTERNAL detail='worker crashed during symbolic equivalence check'",
                    "Fatal Python error: Segmentation fault",
                    "",
                    "Current thread 0x00007f8d6c7fe740 (most recent call first):",
                    '  File "/opt/geo3k/reward_server/symbolic.py", line 312, in compare_symbolic_forms',
                    '  File "/opt/geo3k/reward_server/worker.py", line 241, in handle_calc_score',
                    '  File "/opt/geo3k/reward_server/main.py", line 94, in dispatch_request',
                    "",
                    "Native stack trace:",
                    f"  /lib/x86_64-linux-gnu/libc.so.6(+0x{random.randint(0x10000, 0x8FFFF):x}) [{self._random_hex_address()}]",
                    f"  /usr/bin/python3(PyEval_EvalFrameDefault+0x{random.randint(0x100, 0x7000):x}) [{self._random_hex_address()}]",
                    f"  /usr/bin/python3(PyObject_Call+0x{random.randint(0x40, 0x500):x}) [{self._random_hex_address()}]",
                    f"python3: line 1: {pid} Segmentation fault (core dumped) /usr/bin/python3 /opt/geo3k/reward_server/main.py --worker-id {worker_id}",
                ]
            )
            return "\n".join(lines), {
                "fault_type": fault_type,
                "request_id": request_id,
                "worker_id": worker_id,
                "pid": pid,
            }

        if fault_type == "bus_error":
            lines.extend(
                [
                    f"[{timestamp}] ERROR geo3k.reward_worker request_id={request_id} status=INTERNAL detail='shared-memory page fault while deserializing scoring request'",
                    "Fatal Python error: Bus error",
                    "",
                    "Current thread 0x00007f1bb43fe740 (most recent call first):",
                    '  File "/opt/geo3k/reward_server/ipc.py", line 88, in load_request',
                    '  File "/opt/geo3k/reward_server/worker.py", line 203, in handle_calc_score',
                    "",
                    "Native stack trace:",
                    f"  /lib/x86_64-linux-gnu/libc.so.6(+0x{random.randint(0x10000, 0x8FFFF):x}) [{self._random_hex_address()}]",
                    f"  /usr/bin/python3(Py_BytesMain+0x{random.randint(0x40, 0x500):x}) [{self._random_hex_address()}]",
                    f"[  +0.000018] traps: python3[{pid}] general protection fault ip:{self._random_hex_address()} sp:{self._random_hex_address()} error:0 in libpython3.10.so.1.0[{self._random_hex_address()}+0x{random.randint(0x1000, 0x9000):x}]",
                    f"python3: line 1: {pid} Bus error               (core dumped) /usr/bin/python3 /opt/geo3k/reward_server/main.py --worker-id {worker_id}",
                ]
            )
            return "\n".join(lines), {
                "fault_type": fault_type,
                "request_id": request_id,
                "worker_id": worker_id,
                "pid": pid,
            }

        if fault_type == "illegal_instruction":
            lines.extend(
                [
                    f"[{timestamp}] ERROR geo3k.reward_worker request_id={request_id} status=INTERNAL detail='unexpected CPU feature mismatch while evaluating numeric fallback'",
                    "Fatal Python error: Illegal instruction",
                    "",
                    "Current thread 0x00007f4af2dfe740 (most recent call first):",
                    '  File "/opt/geo3k/reward_server/numeric.py", line 154, in numeric_equivalence',
                    '  File "/opt/geo3k/reward_server/worker.py", line 248, in handle_calc_score',
                    "",
                    "Native stack trace:",
                    f"  /lib/x86_64-linux-gnu/libc.so.6(+0x{random.randint(0x10000, 0x8FFFF):x}) [{self._random_hex_address()}]",
                    f"  /usr/bin/python3(_PyEval_EvalFrameDefault+0x{random.randint(0x100, 0x7000):x}) [{self._random_hex_address()}]",
                    f"[  +0.000012] traps: python3[{pid}] invalid opcode ip:{self._random_hex_address()} sp:{self._random_hex_address()} error:0 in libpython3.10.so.1.0[{self._random_hex_address()}+0x{random.randint(0x1000, 0x9000):x}]",
                    f"python3: line 1: {pid} Illegal instruction     (core dumped) /usr/bin/python3 /opt/geo3k/reward_server/main.py --worker-id {worker_id}",
                ]
            )
            return "\n".join(lines), {
                "fault_type": fault_type,
                "request_id": request_id,
                "worker_id": worker_id,
                "pid": pid,
            }

        lines.extend(
            [
                f"[{timestamp}] WARNING geo3k.reward_client request_id={request_id} rpc_deadline_ms=2500",
                f"[{timestamp}] ERROR geo3k.reward_client request_id={request_id} status=DEADLINE_EXCEEDED detail='calc_score backend timed out waiting for worker response'",
                "Traceback (most recent call last):",
                '  File "/opt/geo3k/reward_client/client.py", line 131, in call_calc_score',
                "    response = stub.CalcScore(request, timeout=2.5)",
                '  File "/usr/local/lib/python3.10/site-packages/grpc/_channel.py", line 1166, in __call__',
                "    return _end_unary_response_blocking(state, call, False, None)",
                "grpc._channel._InactiveRpcError: <_InactiveRpcError of RPC that terminated with:",
                "    status = StatusCode.DEADLINE_EXCEEDED",
                "    details = \"calc_score backend timed out waiting for worker response\"",
                f"    debug_error_string = \"UNKNOWN:Error received from peer ipv4:127.0.0.1:{random.randint(20000, 40000)} {{grpc_status:4, grpc_message:\\\"calc_score backend timed out waiting for worker response\\\"}}\"",
                ">",
            ]
        )
        return "\n".join(lines), {
            "fault_type": fault_type,
            "request_id": request_id,
            "worker_id": worker_id,
            "pid": pid,
        }

    # Called during rollout after receiving a model response
    def step(self, response_text: str):
        self.turn += 1
        self.total_step_calls += 1
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
        arguments = tool_call.get("arguments") or {}
        if name not in SUPPORTED_TOOL_NAMES:
            obs = {
                "obs_str": (
                    f"Tool `{name}` is not supported. "
                    'Call `calc_score` (or `calc_geo3k_reward`) via <tool_call>{"name": "calc_score", "arguments": {"answer": "<digits>"}}</tool_call> (format must be <tool_call>(JSON)</tool_call>)'
                    "to check your solution."
                ),
                "role": "tool",
            }
            info["tool_executed"] = False
            return obs, is_final_turn, info

        raw_answer = arguments.get("answer", None)
        parsed_answer = "" if raw_answer is None else str(raw_answer)
        if not parsed_answer.strip():
            obs = {
                "obs_str": (
                    "Tool call detected but no `answer` was provided. "
                    'Call `calc_score` (or `calc_geo3k_reward`) via <tool_call>{"name": "calc_score", "arguments": {"answer": "<digits>"}}</tool_call> '
                    "to check your solution."
                ),
                "role": "tool",
            }
            info["tool_executed"] = False
            info["answer_missing"] = True
            return obs, is_final_turn, info

        score = self._score_answer(parsed_answer)
        self.last_tool_score = score
        tool_record = {"name": name, "answer": parsed_answer, "score": score}
        self.tool_calls.append(tool_record)
        info.update(tool_record)
        info["tool_executed"] = True

        obs = {
            "obs_str": self._build_tool_feedback(score, parsed_answer),
            "role": "tool",
            "tool_score": score,
        }
        if score == 1 and self.fault_injection_enabled and random.random() < 0.5: # all fault
            injected_error, fault_meta = self._build_fault_message(name, parsed_answer, response_text)
            obs = {
                "obs_str": injected_error,
                "role": "tool",
            }
            info.update(
                {
                    "tool_executed": False,
                    **fault_meta,
                }
            )
            logger.info(
                "Injecting fault into step %d request_id=%s type=%s tool=%s",
                self.total_step_calls,
                fault_meta["request_id"],
                fault_meta["fault_type"],
                name,
            )
            return obs, is_final_turn, info

        return obs, is_final_turn, info

    def _parse_tool_payload(self, raw_json: str) -> dict[str, Any] | None:
        """Parse tool payload strictly as JSON. Malformed payloads are rejected."""
        loader = orjson.loads if orjson is not None else json.loads
        try:
            return loader(raw_json)
        except Exception as exc:
            logger.warning("Failed to decode tool call payload: %s", exc)
            return None


def _extract_ground_truth(sample: Sample | None) -> str | None:
    """Resolve the ground-truth answer from label or metadata."""
    if sample is None:
        return None
    if sample.label is not None:
        return str(sample.label)
    # metadata = sample.metadata
    # for key in ("answer", "ground_truth", "label"):
    #     if key in metadata and metadata[key] is not None:
    #         return str(metadata[key])
    return None


def build_env(
    sample: Sample | None = None,
    args: Any | None = None,
    evaluation: bool = False,
    **_: Any,
) -> Geo3kEnv:
    """
    Construct a Geo3kEnv. Ground truth is pulled from sample.label or metadata.
    """
    ground_truth = _extract_ground_truth(sample)
    max_turns = args.max_turns
    if max_turns is None:
        raise ValueError("max_turns must be set via --custom-config-path in the custom config file.")
    if ground_truth is None:
        logger.warning("Ground truth answer missing; calc_score tool will always return 0.")
    fault_injection_enabled = os.environ.get("ENV_FAULT_INJECTION_ENABLED", str(True)).lower() in ("1", "true", "yes")
    fault_injection_prob = float(os.environ.get("ENV_FAULT_INJECTION_PROB", "0.1"))
    if args is not None:
        if getattr(args, "fault_injection_enabled", None) is not None:
            fault_injection_enabled = bool(args.fault_injection_enabled)
        if getattr(args, "fault_injection_prob", None) is not None:
            fault_injection_prob = float(args.fault_injection_prob)
    # Keep training controlled by env vars/args, but never inject faults during eval rollout.
    if evaluation:
        fault_injection_enabled = False

    return Geo3kEnv(
        ground_truth=ground_truth,
        max_turns=max_turns,
        fault_injection_enabled=fault_injection_enabled,
        fault_injection_prob=fault_injection_prob,
    )
