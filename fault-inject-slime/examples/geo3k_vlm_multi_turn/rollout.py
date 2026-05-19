from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import importlib
import importlib.util
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from examples.geo3k_vlm_multi_turn.base_env import BaseInteractionEnv

# When executed as a module: python -m examples.vlm_multi_turn.rollout
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.processing_utils import encode_image_for_rollout_engine
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

DEFAULT_ENV_MODULE = "examples.vlm_multi_turn.env_geo3k"
ENV_STEP_THREAD_POOL_WORKERS_ENV = "SLIME_ENV_STEP_THREAD_POOL_WORKERS"
ENV_STEP_MAX_IN_FLIGHT_ENV = "SLIME_ENV_STEP_MAX_IN_FLIGHT"
ENV_STEP_QUEUE_TIMEOUT_SEC_ENV = "SLIME_ENV_STEP_QUEUE_TIMEOUT_SEC"
INTERACTION_HEARTBEAT_SEC_ENV = "SLIME_INTERACTION_HEARTBEAT_SEC"
INTERACTION_LLM_TIMEOUT_SEC_ENV = "SLIME_INTERACTION_LLM_TIMEOUT_SEC"
INTERACTION_ENV_STEP_TIMEOUT_SEC_ENV = "SLIME_INTERACTION_ENV_STEP_TIMEOUT_SEC"
DEFAULT_ENV_STEP_THREAD_POOL_WORKERS = 128
DEFAULT_ENV_STEP_MAX_IN_FLIGHT = max(1, DEFAULT_ENV_STEP_THREAD_POOL_WORKERS)
DEFAULT_INTERACTION_HEARTBEAT_SEC = 30.0
DEFAULT_INTERACTION_LLM_TIMEOUT_SEC = 900.0
DEFAULT_INTERACTION_ENV_STEP_TIMEOUT_SEC = 900.0
ENV_STEP_TIMEOUT_OBSERVATION = (
    "The code execution timed out. Check your code and invoke the tool call again later."
)
ENV_STEP_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=max(1, int(os.getenv(ENV_STEP_THREAD_POOL_WORKERS_ENV, str(DEFAULT_ENV_STEP_THREAD_POOL_WORKERS)))),
    thread_name_prefix="interaction-env-step",
)
ENV_STEP_MAX_IN_FLIGHT = max(1, int(os.getenv(ENV_STEP_MAX_IN_FLIGHT_ENV, str(DEFAULT_ENV_STEP_MAX_IN_FLIGHT))))
ENV_STEP_SEMAPHORE: asyncio.Semaphore | None = None

# Dummy messages used for calculating trim length in chat template encoding
DUMMY_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "I am a user."},
]


def _parse_positive_float_env(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = float(raw_value)
    except ValueError:
        logger.warning("Invalid float env %s=%r, falling back to %.3f", name, raw_value, default)
        return default
    if value <= 0:
        logger.warning("Env %s=%r must be > 0, falling back to %.3f", name, raw_value, default)
        return default
    return value


def _parse_optional_timeout_env(name: str, default: float | None) -> float | None:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"0", "none", "off", "false", "no"}:
        return None
    try:
        value = float(raw_value)
    except ValueError:
        logger.warning("Invalid timeout env %s=%r, falling back to %r", name, raw_value, default)
        return default
    if value <= 0:
        return None
    return value


INTERACTION_HEARTBEAT_SEC = _parse_positive_float_env(
    INTERACTION_HEARTBEAT_SEC_ENV, DEFAULT_INTERACTION_HEARTBEAT_SEC
)
INTERACTION_LLM_TIMEOUT_SEC = _parse_optional_timeout_env(
    INTERACTION_LLM_TIMEOUT_SEC_ENV, DEFAULT_INTERACTION_LLM_TIMEOUT_SEC
)
INTERACTION_ENV_STEP_TIMEOUT_SEC = _parse_optional_timeout_env(
    INTERACTION_ENV_STEP_TIMEOUT_SEC_ENV, DEFAULT_INTERACTION_ENV_STEP_TIMEOUT_SEC
)
ENV_STEP_QUEUE_TIMEOUT_SEC = _parse_optional_timeout_env(ENV_STEP_QUEUE_TIMEOUT_SEC_ENV, None)


def _sample_log_id(sample_metadata: dict | None) -> str:
    if not sample_metadata:
        return "sample=unknown"
    for key in ("index", "sample_index", "rollout_id", "question_id", "task_id", "id"):
        value = sample_metadata.get(key)
        if value is not None:
            return f"{key}={value}"
    return "sample=unknown"


def _sample_status_name(status: Any) -> str:
    return getattr(status, "value", str(status))


def _get_env_step_semaphore() -> asyncio.Semaphore:
    global ENV_STEP_SEMAPHORE
    if ENV_STEP_SEMAPHORE is None:
        ENV_STEP_SEMAPHORE = asyncio.Semaphore(ENV_STEP_MAX_IN_FLIGHT)
    return ENV_STEP_SEMAPHORE


def _load_env_module(env_path: str | None):
    """Load the interaction environment module from a module path or a file path."""
    target = env_path or DEFAULT_ENV_MODULE
    module_path = Path(target)
    if module_path.suffix == ".py" and module_path.exists():
        spec = importlib.util.spec_from_file_location(f"rollout_env_{module_path.stem}", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import environment module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    return importlib.import_module(target)


def _build_env(env_module, sample: Sample, args: Any, evaluation: bool = False):
    """Instantiate the interaction environment using the provided module."""
    build_fn = env_module.build_env
    if not callable(build_fn):
        raise ValueError("Environment module must expose a callable `build_env(sample, args)`.")
    try:
        return build_fn(sample=sample, args=args, evaluation=evaluation)
    except TypeError:
        # Fallback to positional signature
        return build_fn(sample, args)


def _encode_observation_for_generation(
    tokenizer,
    processor,
    message: dict,
    metadata: dict | None,
    apply_chat_template: bool,
    apply_chat_template_kwargs: dict | None,
):
    """
    Encode a single observation turn that may include images/videos in the content list.
    Trim out the system/tool preamble added by the chat template so only the observation tokens remain.
    """
    tools = metadata.get("tools") if metadata else None
    apply_kwargs = apply_chat_template_kwargs or {}

    trim_length = 0

    if apply_chat_template:
        dummy_prompt = tokenizer.apply_chat_template(
            DUMMY_MESSAGES,
            tools=tools,
            tokenize=False,
            add_generation_prompt=False,
            **apply_kwargs,
        )
        formatted_prompt = tokenizer.apply_chat_template(
            DUMMY_MESSAGES + [message],
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            **apply_kwargs,
        )
        trim_length = len(tokenizer.encode(dummy_prompt, add_special_tokens=False))
    else:
        formatted_prompt = [message]

    multimodal_inputs = None
    multimodal_train_inputs = None
    if processor:
        # Convert content-embedded images/videos into multimodal inputs for the processor.
        from qwen_vl_utils import process_vision_info

        images, videos = process_vision_info([message])
        multimodal_inputs = {"images": images, "videos": videos}
        processor_output = processor(text=formatted_prompt, **multimodal_inputs)
        prompt_ids = processor_output["input_ids"][0]
        multimodal_train_inputs = {
            k: v for k, v in processor_output.items() if k not in ["input_ids", "attention_mask"]
        } or None
    else:
        prompt_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)

    if trim_length:
        prompt_ids = prompt_ids[trim_length:]

    image_data = []
    if multimodal_inputs and multimodal_inputs.get("images"):
        image_data = [encode_image_for_rollout_engine(img) for img in multimodal_inputs["images"]]
    return prompt_ids, image_data, multimodal_inputs, multimodal_train_inputs


def _merge_multimodal_train_inputs(chunks: list[dict | None]) -> dict | None:
    """
    Merge per-turn multimodal_train_inputs with a single concat per key.

    Note: Only torch.Tensor values are merged; non-tensor fields are ignored by design.
    """
    if not chunks:
        return None

    values_by_key = {}
    for chunk in chunks:
        if not chunk:
            continue
        for key, val in chunk.items():
            if val is None:
                continue
            values_by_key.setdefault(key, []).append(val)

    merged = {}
    for key, values in values_by_key.items():
        if all(isinstance(v, torch.Tensor) for v in values):
            merged[key] = torch.cat(values, dim=0)

    return merged


def _initialize_resources(args: Any, sample: Sample, evaluation: bool = False):
    env_module = _load_env_module(args.rollout_interaction_env_path)
    max_turns = args.max_turns
    if max_turns is None:
        raise ValueError("max_turns must be set via --custom-config-path in the custom config file.")
    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    sample.metadata = sample.metadata or {}
    env = _build_env(env_module, sample, args, evaluation=evaluation)
    config = {"max_turns": max_turns}
    return env, env_module, config, state, url


def _prepare_initial_inputs(sample: Sample, processor, tokenizer):
    if processor:
        processor_output = processor(text=sample.prompt, **(sample.multimodal_inputs or {}))
        prompt_ids = processor_output["input_ids"][0]
        sample.multimodal_train_inputs = {
            k: v for k, v in processor_output.items() if k not in ["input_ids", "attention_mask"]
        } or None
    else:
        prompt_ids = tokenizer.encode(sample.prompt, add_special_tokens=False)

    image_data = []
    if sample.multimodal_inputs and sample.multimodal_inputs.get("images"):
        image_data = [encode_image_for_rollout_engine(img) for img in sample.multimodal_inputs["images"]]
    return prompt_ids, image_data, sample.multimodal_train_inputs


def _prepare_start_state(sample: Sample, state, args: Any, sampling_params: dict):
    prompt_ids, image_data, init_mm_train = _prepare_initial_inputs(sample, state.processor, state.tokenizer)
    current_image_data = image_data
    multimodal_train_inputs_buffer: list[dict | None] = []
    if init_mm_train:
        multimodal_train_inputs_buffer.append(init_mm_train)

    if not sample.tokens:
        sample.tokens = list(prompt_ids)
    response_tokens: list[int] = sample.tokens[len(prompt_ids) :] if len(sample.tokens) >= len(prompt_ids) else []
    sample.loss_mask = sample.loss_mask or []
    sample.rollout_log_probs = sample.rollout_log_probs or []
    sample.response_length = len(response_tokens)

    budget = None
    if args.rollout_max_context_len is not None:
        budget = args.rollout_max_context_len - len(sample.tokens)
    elif sampling_params.get("max_new_tokens") is not None:
        budget = sampling_params["max_new_tokens"] - len(sample.tokens)
    return current_image_data, response_tokens, budget, multimodal_train_inputs_buffer


async def _run_inference_step(url: str, tokens: list[int], sampling_params: dict, image_data, tokenizer):
    payload = {
        "input_ids": tokens,
        "sampling_params": sampling_params,
        "return_logprob": True,
    }
    if image_data:
        payload["image_data"] = image_data

    output = await post(url, payload)
    response_text = output["text"]
    if "output_token_logprobs" in output["meta_info"]:
        new_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
        new_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
    else:
        new_tokens, new_log_probs = [], []
    finish_type = output["meta_info"]["finish_reason"]["type"]
    return response_text, new_tokens, new_log_probs, finish_type


def _process_env_step_sync(env: BaseInteractionEnv, response_text: str, tokenizer, processor, args, sample_metadata):
    sample_id = _sample_log_id(sample_metadata)
    env_step_start = time.perf_counter()
    observation, done, _ = env.step(response_text)
    env_step_elapsed = time.perf_counter() - env_step_start
    logger.info(
        "interaction env.step %s turn=%s elapsed=%.3fs done=%s response_chars=%d",
        sample_id,
        getattr(env, "turn", "unknown"),
        env_step_elapsed,
        done,
        len(response_text),
    )
    if done:
        return None, None, None, None, True

    next_user_message = env.format_observation(observation)
    encode_start = time.perf_counter()
    obs_prompt_ids, obs_image_data, obs_multimodal_inputs, obs_multimodal_train_inputs = (
        _encode_observation_for_generation(
            tokenizer,
            processor,
            next_user_message,
            sample_metadata,
            args.apply_chat_template,
            args.apply_chat_template_kwargs,
        )
    )
    encode_elapsed = time.perf_counter() - encode_start

    bos_id = tokenizer.bos_token_id
    if bos_id is not None and obs_prompt_ids and obs_prompt_ids[0] == bos_id:
        obs_prompt_ids = obs_prompt_ids[1:]

    return obs_prompt_ids, obs_image_data, obs_multimodal_inputs, obs_multimodal_train_inputs, False


async def _process_env_step(env: BaseInteractionEnv, response_text: str, tokenizer, processor, args, sample_metadata):
    # Run env.step and observation encoding off the event loop so expensive local tool execution
    # from one sample does not stall other in-flight interaction environments.
    loop = asyncio.get_running_loop()
    sample_id = _sample_log_id(sample_metadata)
    semaphore = _get_env_step_semaphore()
    queue_start = time.perf_counter()
    semaphore_acquired = False
    semaphore_release_managed_by_future = False
    try:
        if ENV_STEP_QUEUE_TIMEOUT_SEC is None:
            await semaphore.acquire()
        else:
            await asyncio.wait_for(semaphore.acquire(), timeout=ENV_STEP_QUEUE_TIMEOUT_SEC)
        semaphore_acquired = True
        queue_elapsed = time.perf_counter() - queue_start
        if queue_elapsed >= 0.05:
            logger.info(
                "interaction env_step_gate %s turn=%s queued=%.3fs in_flight_limit=%d",
                sample_id,
                getattr(env, "turn", "unknown"),
                queue_elapsed,
                ENV_STEP_MAX_IN_FLIGHT,
            )
        executor_start = time.perf_counter()
        executor_future = loop.run_in_executor(
            ENV_STEP_EXECUTOR,
            _process_env_step_sync,
            env,
            response_text,
            tokenizer,
            processor,
            args,
            sample_metadata,
        )
        # Keep the in-flight permit tied to the actual executor work instead of the awaiting coroutine.
        # If the outer task times out and gets cancelled, the threadpool job keeps running; releasing here
        # would allow unbounded executor backlog to accumulate behind a misleadingly small semaphore count.
        executor_future.add_done_callback(lambda _future: semaphore.release())
        semaphore_release_managed_by_future = True
        result = await executor_future
        return result
    except asyncio.TimeoutError:
        queue_elapsed = time.perf_counter() - queue_start
        logger.error(
            "interaction env_step_queue_timeout %s turn=%s queued=%.3fs queue_timeout=%s in_flight_limit=%d",
            sample_id,
            getattr(env, "turn", "unknown"),
            queue_elapsed,
            f"{ENV_STEP_QUEUE_TIMEOUT_SEC:.3f}s" if ENV_STEP_QUEUE_TIMEOUT_SEC is not None else "disabled",
            ENV_STEP_MAX_IN_FLIGHT,
        )
        raise
    finally:
        if semaphore_acquired and not semaphore_release_managed_by_future:
            semaphore.release()


async def _await_with_heartbeat(
    awaitable,
    *,
    sample_id: str,
    turn: int,
    phase: str,
    heartbeat_sec: float,
    timeout_sec: float | None,
):
    task = asyncio.create_task(awaitable)
    start_time = time.perf_counter()
    next_heartbeat_at = heartbeat_sec
    try:
        while True:
            elapsed = time.perf_counter() - start_time
            wait_timeout = heartbeat_sec
            if timeout_sec is not None:
                remaining = timeout_sec - elapsed
                if remaining <= 0:
                    raise asyncio.TimeoutError
                wait_timeout = min(wait_timeout, remaining)

            done, _pending = await asyncio.wait({task}, timeout=wait_timeout, return_when=asyncio.FIRST_COMPLETED)
            if done:
                return await task

            elapsed = time.perf_counter() - start_time
            if timeout_sec is not None and elapsed >= timeout_sec:
                raise asyncio.TimeoutError

            if elapsed >= next_heartbeat_at:
                logger.warning(
                    "interaction heartbeat %s turn=%d phase=%s elapsed=%.3fs timeout=%s",
                    sample_id,
                    turn,
                    phase,
                    elapsed,
                    f"{timeout_sec:.3f}s" if timeout_sec is not None else "disabled",
                )
                while next_heartbeat_at <= elapsed:
                    next_heartbeat_at += heartbeat_sec
    except asyncio.TimeoutError:
        elapsed = time.perf_counter() - start_time
        logger.error(
            "interaction timeout %s turn=%d phase=%s elapsed=%.3fs timeout=%s",
            sample_id,
            turn,
            phase,
            elapsed,
            f"{timeout_sec:.3f}s" if timeout_sec is not None else "disabled",
        )
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        raise


def _mark_sample_timeout(sample: Sample, *, phase: str, turn: int, timeout_sec: float | None) -> None:
    sample.status = Sample.Status.ABORTED
    sample.reward = 0.0
    sample.metadata = sample.metadata or {}
    sample.metadata["interaction_timeout"] = {
        "phase": phase,
        "turn": turn,
        "timeout_sec": timeout_sec,
    }


def _build_env_step_timeout_observation(
    env: BaseInteractionEnv,
    *,
    phase: str,
    turn: int,
    timeout_sec: float | None,
):
    observation = {
        "obs_str": ENV_STEP_TIMEOUT_OBSERVATION,
        "role": "tool",
        "interaction_timeout": {
            "phase": phase,
            "turn": turn,
            "timeout_sec": timeout_sec,
        },
    }
    return env.format_observation(observation)


def _append_to_sample(
    sample: Sample,
    response_tokens: list[int],
    tokens_to_add: list[int],
    logprobs: list[float],
    loss_mask_val: int,
) -> None:
    sample.tokens.extend(tokens_to_add)
    response_tokens.extend(tokens_to_add)
    sample.loss_mask.extend([loss_mask_val] * len(tokens_to_add))
    sample.rollout_log_probs.extend(logprobs)
    sample.response_length = len(response_tokens)


def _update_multimodal_state(
    sample: Sample,
    current_image_data,
    obs_image_data,
    obs_multimodal_inputs,
    obs_multimodal_train_inputs,
    multimodal_train_inputs_buffer: list[dict | None],
):
    if obs_image_data:
        current_image_data = (current_image_data or []) + obs_image_data

    if obs_multimodal_inputs:
        if not sample.multimodal_inputs:
            sample.multimodal_inputs = obs_multimodal_inputs
        elif isinstance(sample.multimodal_inputs, dict) and isinstance(obs_multimodal_inputs, dict):
            for key, val in obs_multimodal_inputs.items():
                if val is None:
                    continue
                if (
                    key in sample.multimodal_inputs
                    and isinstance(sample.multimodal_inputs[key], list)
                    and isinstance(val, list)
                ):
                    sample.multimodal_inputs[key].extend(val)
        else:
            sample.multimodal_inputs = obs_multimodal_inputs

    if obs_multimodal_train_inputs:
        multimodal_train_inputs_buffer.append(obs_multimodal_train_inputs)

    return current_image_data


def _should_stop_on_finish(sample: Sample, finish_type: str) -> bool:
    match finish_type:
        case "length":
            sample.status = Sample.Status.TRUNCATED
            return True
        case "abort":
            sample.status = Sample.Status.ABORTED
            return True
    return False


def _update_budget(budget, consumed: int):
    if budget is None:
        return None
    return budget - consumed


def _finalize_sample(sample: Sample, tokenizer, response_tokens, multimodal_train_inputs_buffer):
    sample.multimodal_train_inputs = _merge_multimodal_train_inputs(multimodal_train_inputs_buffer)
    sample.response = tokenizer.decode(response_tokens, skip_special_tokens=False)
    sample.response_length = len(response_tokens)
    if sample.status is None:
        sample.status = Sample.Status.COMPLETED
    return sample


async def generate(args: Any, sample: Sample, sampling_params, evaluation: bool = False) -> Sample:
    """Custom multi-turn rollout that interacts with a pluggable environment."""
    assert not args.partial_rollout, "Partial rollout is not supported for interaction rollouts."

    env, env_module, config, state, url = _initialize_resources(args, sample, evaluation=evaluation)
    sampling_params = sampling_params.copy()
    current_image_data, response_tokens, budget, multimodal_train_inputs_buffer = _prepare_start_state(
        sample, state, args, sampling_params
    )
    sample_id = _sample_log_id(sample.metadata)
    current_phase = "initialize"
    current_turn = -1
    try:
        env.reset()
        if budget is not None and budget <= 0:
            sample.status = Sample.Status.TRUNCATED
            return sample

        cur_sampling_params = sampling_params
        for turn_idx in range(config["max_turns"]):
            current_turn = turn_idx
            if budget is not None:
                cur_sampling_params["max_new_tokens"] = budget

            inference_start = time.perf_counter()
            current_phase = "llm_generate"
            try:
                response_text, new_response_tokens, new_response_log_probs, finish_type = await _await_with_heartbeat(
                    _run_inference_step(url, sample.tokens, cur_sampling_params, current_image_data, state.tokenizer),
                    sample_id=sample_id,
                    turn=turn_idx,
                    phase="llm_generate",
                    heartbeat_sec=INTERACTION_HEARTBEAT_SEC,
                    timeout_sec=INTERACTION_LLM_TIMEOUT_SEC,
                )
            except asyncio.TimeoutError:
                _mark_sample_timeout(
                    sample,
                    phase="llm_generate",
                    turn=turn_idx,
                    timeout_sec=INTERACTION_LLM_TIMEOUT_SEC,
                )
                break
            _append_to_sample(sample, response_tokens, new_response_tokens, new_response_log_probs, loss_mask_val=1)
            budget = _update_budget(budget, len(new_response_tokens))

            if _should_stop_on_finish(sample, finish_type):
                break
            if budget is not None and budget <= 0:
                sample.status = Sample.Status.TRUNCATED
                break

            turn_env_step_start = time.perf_counter()
            current_phase = "env_step"
            try:
                obs_prompt_ids, obs_image_data, obs_multimodal_inputs, obs_multimodal_train_inputs, done = (
                    await _await_with_heartbeat(
                        _process_env_step(env, response_text, state.tokenizer, state.processor, args, sample.metadata),
                        sample_id=sample_id,
                        turn=turn_idx,
                        phase="env_step",
                        heartbeat_sec=INTERACTION_HEARTBEAT_SEC,
                        timeout_sec=INTERACTION_ENV_STEP_TIMEOUT_SEC,
                    )
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "interaction env_step_timeout_observation %s turn=%d timeout=%s",
                    sample_id,
                    turn_idx,
                    f"{INTERACTION_ENV_STEP_TIMEOUT_SEC:.3f}s"
                    if INTERACTION_ENV_STEP_TIMEOUT_SEC is not None
                    else "disabled",
                )
                sample.metadata = sample.metadata or {}
                sample.metadata["interaction_timeout"] = {
                    "phase": "env_step",
                    "turn": turn_idx,
                    "timeout_sec": INTERACTION_ENV_STEP_TIMEOUT_SEC,
                }
                timeout_message = _build_env_step_timeout_observation(
                    env,
                    phase="env_step",
                    turn=turn_idx,
                    timeout_sec=INTERACTION_ENV_STEP_TIMEOUT_SEC,
                )
                obs_prompt_ids, obs_image_data, obs_multimodal_inputs, obs_multimodal_train_inputs = (
                    _encode_observation_for_generation(
                        state.tokenizer,
                        state.processor,
                        timeout_message,
                        sample.metadata,
                        args.apply_chat_template,
                        args.apply_chat_template_kwargs,
                    )
                )
                done = False
            if done:
                sample.status = Sample.Status.COMPLETED
                break

            obs_log_probs = [0.0] * len(obs_prompt_ids)
            _append_to_sample(sample, response_tokens, obs_prompt_ids, obs_log_probs, loss_mask_val=0)
            budget = _update_budget(budget, len(obs_prompt_ids))

            current_image_data = _update_multimodal_state(
                sample,
                current_image_data,
                obs_image_data,
                obs_multimodal_inputs,
                obs_multimodal_train_inputs,
                multimodal_train_inputs_buffer,
            )

            if budget is not None and budget <= 0:
                sample.status = Sample.Status.TRUNCATED
                break
            if turn_idx + 1 >= config["max_turns"]:
                sample.status = Sample.Status.COMPLETED
                break

        current_phase = "finalize"
        finalized_sample = _finalize_sample(sample, state.tokenizer, response_tokens, multimodal_train_inputs_buffer)
        logger.info(
            "interaction generate_done %s turn=%d status=%s response_tokens=%d reward=%s",
            sample_id,
            current_turn,
            _sample_status_name(finalized_sample.status),
            finalized_sample.response_length,
            finalized_sample.reward,
        )
        return finalized_sample
    except Exception:
        logger.exception(
            "interaction generate_error %s turn=%d phase=%s status=%s",
            sample_id,
            current_turn,
            current_phase,
            _sample_status_name(sample.status),
        )
        raise
    finally:
        try:
            env.close()
        except Exception:
            logger.exception("Failed to close interaction environment cleanly for %s", sample_id)
