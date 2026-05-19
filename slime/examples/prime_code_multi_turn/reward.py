from __future__ import annotations

import importlib
import importlib.util
import json
import logging
import re
import sys
import time
import types
from pathlib import Path

from slime.utils.types import Sample

DEFAULT_PRIME_CODE_DIR = "/mnt/shared-storage-user/ailab-sys/zhoujiecheng/projs/robust_rl/verl/verl/utils/reward_score/prime_code"
THINK_REWARD_BONUS = 0.5
TOOL_CALL_REWARD_BONUS = 0.5
THINK_BLOCK_RE = re.compile(r"<think>\s*.+?\s*</think>", re.DOTALL)
TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
logger = logging.getLogger(__name__)


def _sample_log_id(sample: Sample) -> str:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    for key in ("index", "sample_index", "rollout_id", "question_id", "task_id", "id"):
        value = metadata.get(key)
        if value is not None:
            return f"{key}={value}"
    return "sample=unknown"


def _load_prime_compute_score(prime_code_dir: str):
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
        raise FileNotFoundError(f"PRIME code reward package not found: {init_file}")

    package_name = "_prime_code_reward_reference"
    if package_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            package_name,
            init_file,
            submodule_search_locations=[str(package_dir)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load PRIME code reward package from {package_dir}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = module
        spec.loader.exec_module(module)

    package = importlib.import_module(package_name)
    return package.compute_score


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
        "think_bonus": THINK_REWARD_BONUS if apply_bonus and THINK_BLOCK_RE.search(completion) else 0.0,
        "tool_call_bonus": TOOL_CALL_REWARD_BONUS if apply_bonus and _has_execute_code_tool_call(completion) else 0.0,
    }
    bonus = components["think_bonus"] + components["tool_call_bonus"]
    return bonus, components


async def batched_prime_code_reward(args, samples: Sample | list[Sample], **kwargs) -> list[float]:
    evaluation = bool(kwargs.get("evaluation", False))
    batch_start = time.perf_counter()
    compute_score = _load_prime_compute_score(
        str(getattr(args, "code_execution_prime_code_dir", DEFAULT_PRIME_CODE_DIR))
    )

    if isinstance(samples, Sample):
        samples = [samples]

    logger.info("prime_code reward_batch_start samples=%d", len(samples))
    rewards: list[float] = []
    for sample in samples:
        sample_start = time.perf_counter()
        base_score, _metadata = compute_score(sample.response, sample.label, continuous=not evaluation)
        bonus, reward_components = _compute_format_and_tool_bonus(sample.response, apply_bonus=not evaluation)
        score = float(base_score) + bonus
        rewards.append(float(score))
        logger.info(
            "prime_code reward_sample_done %s elapsed=%.3fs reward=%.4f base=%.4f think_bonus=%.4f tool_call_bonus=%.4f bonus=%.4f evaluation=%s",
            _sample_log_id(sample),
            time.perf_counter() - sample_start,
            float(score),
            float(base_score),
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
