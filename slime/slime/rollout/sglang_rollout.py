import asyncio
import copy
import inspect
import logging
import os
import time
from argparse import Namespace
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import numpy as np
import pybase64
import sglang_router
from packaging.version import parse
from tqdm import tqdm

from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from slime.rollout.filter_hub.base_types import MetricGatherer, call_dynamic_filter
from slime.utils.async_utils import run
from slime.utils.data import Dataset
from slime.utils.eval_config import EvalDatasetConfig
from slime.utils.http_utils import get, post, post_stream
from slime.utils.misc import SingletonMeta, load_function
from slime.utils.processing_utils import encode_image_for_rollout_engine, load_processor, load_tokenizer
from slime.utils.types import Sample

from .rm_hub import async_rm, batched_async_rm

__all__ = ["generate_rollout"]

logger = logging.getLogger(__name__)


async def _list_router_worker_urls(args: Namespace) -> list[str]:
    prefer_legacy_endpoint = parse(sglang_router.__version__) <= parse("0.2.1") or args.use_slime_router

    if prefer_legacy_endpoint:
        try:
            response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/list_workers")
            return response["urls"]
        except Exception as exc:
            if args.use_slime_router or "404" not in str(exc):
                raise
            logger.info(
                "Router %s:%s does not expose /list_workers; retrying via /workers",
                args.sglang_router_ip,
                args.sglang_router_port,
            )

    response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/workers")
    workers = response.get("workers", response if isinstance(response, list) else [])
    return [worker["url"] for worker in workers]


def _rolling_refill_enabled() -> bool:
    return os.getenv("SLIME_ENABLE_ROLLING_OVERSAMPLING_REFILL", "0").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _effective_pending_group_target(args: Namespace, target_data_size: int, rolling_refill_enabled: bool) -> int:
    target = max(1, int(target_data_size))
    requested_group_window = max(1, int(getattr(args, "over_sampling_batch_size", target_data_size)))

    if rolling_refill_enabled:
        return max(target, requested_group_window)
    return target


class GenerateState(metaclass=SingletonMeta):
    """
    The global state for the generation process.
    """

    def __init__(self, args: Namespace) -> None:
        # persistent state for the generation process
        self.args = args
        self.tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        self.processor = load_processor(args.hf_checkpoint, trust_remote_code=True)

        base_concurrency = (
            args.sglang_server_concurrency * args.rollout_num_gpus // args.rollout_num_gpus_per_engine
        )
        self.semaphore = asyncio.Semaphore(max(1, base_concurrency))
        self.sampling_params: dict[str, Any] = dict(
            temperature=args.rollout_temperature,
            top_p=args.rollout_top_p,
            top_k=args.rollout_top_k,
            max_new_tokens=args.rollout_max_response_len,
            stop=args.rollout_stop,
            stop_token_ids=args.rollout_stop_token_ids,
            skip_special_tokens=args.rollout_skip_special_tokens,
            no_stop_trim=True,
            spaces_between_special_tokens=False,
        )

        if getattr(args, "sglang_enable_deterministic_inference", False):
            sampling_seed_base = args.rollout_seed
            self.group_sampling_seeds = [sampling_seed_base + i for i in range(args.n_samples_per_prompt)]

        # dp rank balancing
        self.dp_counts = [0] * (args.sglang_dp_size or 1)
        self.dp_rank = 0

        self.reset()

    @contextmanager
    def dp_rank_context(self):
        candidates = [i for i, count in enumerate(self.dp_counts) if count == min(self.dp_counts)]
        dp_rank = int(np.random.choice(candidates))
        self.dp_counts[dp_rank] += 1
        self.dp_rank = dp_rank
        try:
            yield dp_rank
        finally:
            self.dp_counts[dp_rank] -= 1
            assert self.dp_counts[dp_rank] >= 0

    def reset(self) -> None:
        self.remaining_batch_size = 0
        self.pendings = set()
        self.aborted = False

    def submit_generate_tasks(self, samples: list[list[Sample]]) -> None:
        for group in samples:
            self.pendings.add(
                asyncio.create_task(
                    # submit a group of samples as a single task.
                    generate_and_rm_group(
                        self.args,
                        group,
                        sampling_params=self.sampling_params.copy(),
                        evaluation=False,
                    )
                )
            )
        self.remaining_batch_size += len(samples)

    def submit_generate_tasks_bfs(self, samples: list[list[Sample]]) -> None:
        max_group_size = max((len(group) for group in samples), default=0)
        for sample_offset in range(max_group_size):
            for group in samples:
                if sample_offset >= len(group):
                    continue

                sampling_params = self.sampling_params.copy()
                if getattr(self.args, "sglang_enable_deterministic_inference", False):
                    sampling_params["sampling_seed"] = self.group_sampling_seeds[sample_offset]

                self.pendings.add(
                    asyncio.create_task(
                        generate_and_rm(
                            self.args,
                            group[sample_offset],
                            sampling_params=sampling_params,
                            evaluation=False,
                        )
                    )
                )
        self.remaining_batch_size += len(samples)


async def generate(args: Namespace, sample: Sample, sampling_params: dict[str, Any]) -> Sample:
    """Generate using traditional SGLang router with token-based workflow"""
    if args.ci_test:
        assert isinstance(sample.prompt, str)

    state = GenerateState(args)
    router_generate_path = os.getenv("SLIME_ROUTER_GENERATE_PATH", "/generate")
    if not router_generate_path.startswith("/"):
        router_generate_path = f"/{router_generate_path}"
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}{router_generate_path}"

    assert (
        sample.status == Sample.Status.PENDING or sample.status == Sample.Status.ABORTED
    ), f"Sample status is {sample.status}"

    if state.processor:
        processor_output = state.processor(text=sample.prompt, **sample.multimodal_inputs)
        prompt_ids = processor_output["input_ids"][0]
        sample.multimodal_train_inputs = {
            k: v for k, v in processor_output.items() if k not in ["input_ids", "attention_mask"]
        } or None
    else:
        prompt_ids = state.tokenizer.encode(sample.prompt, add_special_tokens=False)

    if len(sample.response) > 0:
        sampling_params["max_new_tokens"] -= len(sample.tokens) - len(prompt_ids)

    assert (
        sampling_params["max_new_tokens"] >= 0
    ), f"max_new_tokens: {sampling_params['max_new_tokens']} should not be less than 0"
    if sampling_params["max_new_tokens"] == 0:
        sample.status = Sample.Status.TRUNCATED
        return sample

    # Prepare payload for sglang server
    payload = {
        "sampling_params": sampling_params,
        "return_logprob": True,
    }

    if args.use_rollout_routing_replay:
        payload["return_routed_experts"] = True

    if sample.multimodal_inputs and sample.multimodal_inputs["images"]:
        image_data = sample.multimodal_inputs["images"]
        payload["image_data"] = [encode_image_for_rollout_engine(image) for image in image_data]

    # Use existing tokens for multi-turn or tokenize the new prompt
    if len(sample.response) > 0:
        payload["input_ids"] = sample.tokens
    else:
        payload["input_ids"] = prompt_ids
        if not sample.tokens:  # Initialize sample.tokens for the first turn
            sample.tokens = prompt_ids

    use_stream_receiver = (
        not args.use_slime_router
        and parse(sglang_router.__version__) <= parse("0.2.1")
        and router_generate_path == "/generate"
    )
    if use_stream_receiver:
        logger.info(
            "Using streaming receiver for legacy sglang-router generate response: router_version=%s sample_index=%s",
            sglang_router.__version__,
            sample.index,
        )
        output = await post_stream(url, payload)
    else:
        output = await post(url, payload)

    if args.use_slime_router and "RadixTreeMiddleware" in args.slime_router_middleware_paths:
        from slime.router.middleware_hub.radix_tree_middleware import postprocess_sample_with_radix_tree

        sample = await postprocess_sample_with_radix_tree(args, sample, output)
    else:
        if "output_token_logprobs" in output["meta_info"]:
            new_response_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
            new_response_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
        else:
            new_response_tokens = output.get("output_ids") or []
            new_response_log_probs = []

        # Update sample with tokens directly - avoiding re-tokenization
        sample.tokens = sample.tokens + new_response_tokens
        sample.response_length += len(new_response_tokens)
        sample.response += output["text"]

        # When partial rollout and masking off policy is enabled, update the loss mask
        if sample.loss_mask is not None:
            assert args.partial_rollout and args.mask_offpolicy_in_partial_rollout
            sample.loss_mask += [1] * len(new_response_tokens)

        if sample.rollout_log_probs is None:
            sample.rollout_log_probs = []
        sample.rollout_log_probs += new_response_log_probs

    if "routed_experts" in output["meta_info"]:
        sample.rollout_routed_experts = np.frombuffer(
            pybase64.b64decode(output["meta_info"]["routed_experts"].encode("ascii")),
            dtype=np.int32,
        ).reshape(
            len(sample.tokens) - 1,
            args.num_layers,
            args.moe_router_topk,
        )

    sample.update_from_meta_info(args, output["meta_info"])

    return sample


async def generate_and_rm(
    args: Namespace,
    sample: Sample | list[Sample],
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Sample | list[Sample]:
    # mask previous off-policy generation for partial rollout
    if args.partial_rollout and args.mask_offpolicy_in_partial_rollout and sample.response_length > 0:
        sample.loss_mask = [0] * sample.response_length

    # For samples with existing response, check if they're complete
    if sample.status == Sample.Status.COMPLETED or sample.status == Sample.Status.TRUNCATED:
        assert sample.response is not None
        if not args.group_rm:
            assert sample.reward is not None
        return sample

    state = GenerateState(args)

    # generate
    async with state.semaphore:
        if state.aborted:
            sample.status = Sample.Status.ABORTED
            return sample

        with state.dp_rank_context() as _:
            # Check sample.generate_function_path for per-sample custom_generate_function_path (e.g., from eval dataset config)
            custom_func_path = getattr(sample, "generate_function_path", None) or args.custom_generate_function_path

            if custom_func_path is not None:
                custom_generate_func = load_function(custom_func_path)
                # if signature has evaluation, pass evaluation
                if "evaluation" in inspect.signature(custom_generate_func).parameters:
                    sample = await custom_generate_func(args, sample, sampling_params, evaluation=evaluation)
                else:
                    sample = await custom_generate_func(args, sample, sampling_params)
            else:
                sample = await generate(args, sample, sampling_params)

    # for the rm that need the whole group, we will not do the rm here
    if args.group_rm:
        return sample

    # multi samples
    if isinstance(sample, list):
        samples = sample
        if any([sample.status == Sample.Status.ABORTED for sample in samples]):
            return samples

        # for multi agent system, the reward of some sample is calculated during generation.
        samples_need_reward = [sample for sample in samples if sample.reward is None]
        if samples_need_reward:
            rm_start_time = time.time()
            logger.info(
                "generate_and_rm entering batched_async_rm batch_size=%s sample_indices=%s",
                len(samples_need_reward),
                [sample.index for sample in samples_need_reward[: min(len(samples_need_reward), 8)]],
            )
        rewards = await batched_async_rm(args, samples_need_reward, evaluation=evaluation)
        for sample, reward in zip(samples_need_reward, rewards, strict=False):
            sample.reward = reward
        if samples_need_reward:
            logger.info(
                "generate_and_rm finished batched_async_rm batch_size=%s elapsed=%.2fs",
                len(samples_need_reward),
                time.time() - rm_start_time,
            )
        return samples
    else:
        if sample.status == Sample.Status.ABORTED:
            return sample
        # for multi-turn environment, a reward could be assigned to the agent.
        if sample.reward is None:
            rm_start_time = time.time()
            logger.info(
                "generate_and_rm entering async_rm sample_index=%s response_len=%s",
                sample.index,
                len(sample.response) if sample.response is not None else None,
            )
            sample.reward = await async_rm(args, sample, evaluation=evaluation)
            logger.info(
                "generate_and_rm finished async_rm sample_index=%s elapsed=%.2fs reward=%s",
                sample.index,
                time.time() - rm_start_time,
                sample.reward,
            )

    return sample


async def generate_and_rm_group(
    args: Namespace, group: list[Sample], sampling_params: dict[str, Any], evaluation: bool = False
) -> list[Sample]:
    state = GenerateState(args)

    if state.aborted:
        return group

    group_start_time = time.time()
    tasks = []
    for idx, sample in enumerate(group):
        current_sampling_params = sampling_params.copy()
        if getattr(args, "sglang_enable_deterministic_inference", False):
            seed = state.group_sampling_seeds[idx]
            current_sampling_params["sampling_seed"] = seed
        tasks.append(
            asyncio.create_task(generate_and_rm(args, sample, current_sampling_params, evaluation=evaluation))
        )

    log_interval = float(os.getenv("SLIME_GROUP_TASK_DEBUG_INTERVAL_SEC", "30"))
    next_log_time = group_start_time + log_interval
    pending_tasks = set(tasks)
    while pending_tasks:
        timeout = max(0.0, next_log_time - time.time())
        done, pending_tasks = await asyncio.wait(
            pending_tasks,
            timeout=timeout,
            return_when=asyncio.ALL_COMPLETED if timeout == 0 else asyncio.FIRST_COMPLETED,
        )
        if pending_tasks and time.time() >= next_log_time:
            logger.info(
                "generate_and_rm_group still running after %.1fs: group_size=%s completed=%s pending=%s sample_indices=%s",
                time.time() - group_start_time,
                len(group),
                len(tasks) - len(pending_tasks),
                len(pending_tasks),
                [sample.index for sample in group[: min(len(group), 8)]],
            )
            next_log_time = time.time() + log_interval

    group = [task.result() for task in tasks]

    # for the rm that need the whole group, we will do the rm here
    if not state.aborted and args.group_rm:
        rewards = await batched_async_rm(args, group, evaluation=evaluation)
        for sample, reward in zip(group, rewards, strict=False):
            sample.reward = reward

    elapsed = time.time() - group_start_time
    if elapsed >= log_interval:
        statuses = {}
        for sample in group:
            status_name = getattr(sample.status, "name", str(sample.status))
            statuses[status_name] = statuses.get(status_name, 0) + 1
        logger.info(
            "generate_and_rm_group completed after %.1fs: group_size=%s statuses=%s sample_indices=%s",
            elapsed,
            len(group),
            statuses,
            [sample.index for sample in group[: min(len(group), 8)]],
        )

    return group


async def abort(args: Namespace, rollout_id: int) -> list[list[Sample]]:
    aborted_samples = []

    state = GenerateState(args)
    assert not state.aborted
    state.aborted = True

    async def _list_worker_urls() -> list[str]:
        return await _list_router_worker_urls(args)

    async def _abort_urls(urls: list[str]) -> list[Exception]:
        if not urls:
            return []
        logger.info(f"Abort request for {urls}")
        results = await asyncio.gather(
            *[post(f"{url}/abort_request", {"abort_all": True}, max_retries=3) for url in urls],
            return_exceptions=True,
        )
        failures = [result for result in results if isinstance(result, Exception)]
        for url, result in zip(urls, results, strict=False):
            if isinstance(result, Exception):
                logger.warning("Abort request failed for %s: %s", url, result)
        return failures

    urls = await _list_worker_urls()
    failures = await _abort_urls(urls)
    if failures:
        refreshed_urls = await _list_worker_urls()
        refreshed_urls = [url for url in refreshed_urls if url not in urls] or refreshed_urls
        if refreshed_urls:
            logger.info("Retry abort against refreshed worker list: %s", refreshed_urls)
            await _abort_urls(refreshed_urls)

    if not args.partial_rollout:
        pending_tasks = tuple(state.pendings)
        if pending_tasks:
            cancel_timeout = float(os.getenv("SLIME_ABORT_CANCEL_TIMEOUT_SEC", "10"))
            logger.info(
                "Cancelling %d pending rollout tasks after target batch is reached (timeout=%ss)",
                len(pending_tasks),
                cancel_timeout,
            )
            for task in pending_tasks:
                task.cancel()
            try:
                await asyncio.wait_for(
                    asyncio.gather(*pending_tasks, return_exceptions=True),
                    timeout=cancel_timeout,
                )
            except (asyncio.TimeoutError, TimeoutError):
                logger.warning(
                    "Timed out while draining cancelled rollout tasks; %d tasks still pending",
                    sum(1 for task in pending_tasks if not task.done()),
                )
            state.pendings = {task for task in pending_tasks if not task.done()}
        return aborted_samples

    # make sure all the pending tasks are finished
    count = 0
    while state.pendings:
        done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)

        if not args.partial_rollout:
            continue

        # for partial rollout, collect the partial samples into the data buffer
        for task in done:
            result = task.result()
            group = result if isinstance(result, list) else [result]
            for sample in group:
                if sample.response and "start_rollout_id" not in sample.metadata:
                    sample.metadata["start_rollout_id"] = rollout_id
            aborted_samples.append(group)
            count += len(group)

    if args.partial_rollout:
        logger.info(f"Collected {count} partial samples into the data buffer")

    return aborted_samples


async def generate_rollout_async(
    args: Namespace, rollout_id: int, data_source: Callable[[int], list[list[Sample]]]
) -> tuple[RolloutFnTrainOutput, list[list[Sample]]]:
    """An example to implement the generate_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        data_source: the data source to fetch

    Returns:
        tuple[RolloutFnTrainOutput, list[list[Sample]]]:
            - data: a list of groups of samples generated by the rollout, length equals `rollout_batch_size`
            - aborted_samples: any partial groups collected during abort when partial_rollout is enabled
    """
    assert args.rollout_global_dataset

    state = GenerateState(args)

    # instantiate data filters
    dynamic_filter = (
        load_function(args.dynamic_sampling_filter_path) if args.dynamic_sampling_filter_path is not None else None
    )

    metric_gatherer = MetricGatherer()
    rollout_start_time = time.time()
    first_sample_time = None
    pending_group_samples: dict[int, list[Sample]] = {}

    # target_data_size is the total number of valid samples to get
    target_data_size = args.rollout_batch_size
    rolling_refill_enabled = _rolling_refill_enabled()
    pending_group_target = _effective_pending_group_target(args, target_data_size, rolling_refill_enabled)
    if pending_group_target != target_data_size:
        logger.info(
            "Pending window expanded: target_groups=%d -> pending_groups=%d (rolling_refill=%s)",
            target_data_size,
            pending_group_target,
            rolling_refill_enabled,
        )
    else:
        logger.info("Pending window: target_groups=%d (rolling_refill=%s)", target_data_size, rolling_refill_enabled)

    def refill_pending_groups() -> None:
        while len(data) < target_data_size and state.remaining_batch_size < pending_group_target:
            if rolling_refill_enabled:
                # Top up only the missing in-flight groups so oversampling becomes a rolling window
                # instead of repeatedly over-issuing full batches.
                refill_size = min(
                    pending_group_target - state.remaining_batch_size,
                    args.over_sampling_batch_size,
                )
            else:
                refill_size = args.over_sampling_batch_size
            samples = data_source(refill_size)
            state.submit_generate_tasks_bfs(samples)

    data = []
    all_data = []
    do_print = True
    pbar = tqdm(total=target_data_size * args.n_samples_per_prompt, desc="Rollout generation")
    while len(data) < target_data_size:
        refill_pending_groups()

        # wait for the generation to finish
        done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            result = task.result()
            if isinstance(result, list):
                completed_samples = result
            else:
                completed_samples = [result]

            if do_print:
                first_sample_time = time.time()
                sample = completed_samples[0]
                logger.info(
                    f"First rollout sample: {[str(sample.prompt) + sample.response]}, label: {str(sample.label)[:100]}, reward: {sample.reward}",
                )
                do_print = False

            for sample in completed_samples:
                assert sample.group_index is not None
                group = pending_group_samples.setdefault(sample.group_index, [])
                group.append(sample)
                if len(group) != args.n_samples_per_prompt:
                    continue

                group.sort(key=lambda item: item.index)
                del pending_group_samples[sample.group_index]

                if args.group_rm:
                    rewards = await batched_async_rm(args, group, evaluation=False)
                    for group_sample, reward in zip(group, rewards, strict=False):
                        group_sample.reward = reward

                all_data.append(group)
                dynamic_filter_output = call_dynamic_filter(dynamic_filter, args, group)
                if not dynamic_filter_output.keep:
                    metric_gatherer.on_dynamic_filter_drop(reason=dynamic_filter_output.reason)
                    if rolling_refill_enabled:
                        state.remaining_batch_size -= 1
                    continue

                if rolling_refill_enabled:
                    state.remaining_batch_size -= 1

                # add the samples to the data
                # NOTE: here we have not stored all the unused samples back to the data buffer.
                if len(data) < target_data_size:
                    data.append(group)
                    pbar.update(args.n_samples_per_prompt)

    pbar.close()
    sample = data[-1][0][0] if isinstance(data[-1][0], list) else data[-1][0]
    logger.info(
        f"Finish rollout: {[str(sample.prompt) + sample.response]}, label: {str(sample.label)[:100]}, reward: {sample.reward}",
    )

    # there are still some unfinished requests, abort them
    aborted_samples = await abort(args, rollout_id)

    assert len(data) == args.rollout_batch_size, f"Got {len(data)} samples, expected {args.rollout_batch_size}"
    data = sorted(data, key=lambda group: group[0][0].index if isinstance(group[0], list) else group[0].index)
    all_samples = sorted(
        all_data, key=lambda group: group[0][0].index if isinstance(group[0], list) else group[0].index
    )

    # reset the global state to prevent effects on the next rollout or eval.
    state.reset()
    if args.rollout_sample_filter_path is not None:
        filter_func = load_function(args.rollout_sample_filter_path)
        filter_func(args, data)

    # There can be circumstances where users want to process all samples including filtered ones.
    if args.rollout_all_samples_process_path is not None:
        process_func = load_function(args.rollout_all_samples_process_path)
        process_func(args, all_samples, data_source)

    metrics = metric_gatherer.collect()
    if first_sample_time is not None:
        metrics["perf/rollout_time_before_first_sample"] = first_sample_time - rollout_start_time

    return RolloutFnTrainOutput(samples=data, metrics=metrics), aborted_samples


EVAL_PROMPT_DATASET = {}


async def eval_rollout(args: Namespace, rollout_id: int) -> tuple[dict[str, dict[str, list[Any]]], list[list[Sample]]]:
    assert not args.group_rm, "Group RM is not supported for eval rollout"

    coros = []
    for dataset_cfg in getattr(args, "eval_datasets", []) or []:
        coros.append(eval_rollout_single_dataset(args, rollout_id, dataset_cfg))
    results_list = await asyncio.gather(*coros)
    results = {}
    for r in results_list:
        results.update(r)
    return RolloutFnEvalOutput(data=results), []


async def eval_rollout_single_dataset(
    args: Namespace, rollout_id: int, dataset_cfg: EvalDatasetConfig
) -> dict[str, dict[str, list[Any]]]:
    """An example to implement the eval_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        dataset_cfg: configuration of the dataset
    """
    assert not args.group_rm, "Group RM is not supported for eval rollout"

    global EVAL_PROMPT_DATASET

    cache_key = dataset_cfg.cache_key + (args.hf_checkpoint, args.apply_chat_template)
    if cache_key not in EVAL_PROMPT_DATASET:
        tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        processor = load_processor(args.hf_checkpoint, trust_remote_code=True)
        EVAL_PROMPT_DATASET[cache_key] = Dataset(
            path=dataset_cfg.path,
            tokenizer=tokenizer,
            processor=processor,
            max_length=args.eval_max_prompt_len,
            prompt_key=dataset_cfg.input_key,
            label_key=dataset_cfg.label_key,
            multimodal_keys=args.multimodal_keys,
            metadata_key=dataset_cfg.metadata_key,
            tool_key=dataset_cfg.tool_key,
            apply_chat_template=args.apply_chat_template,
            apply_chat_template_kwargs=args.apply_chat_template_kwargs,
        )
    dataset = EVAL_PROMPT_DATASET[cache_key]

    base_sampling_params = dict(
        temperature=dataset_cfg.temperature,
        top_p=dataset_cfg.top_p,
        top_k=dataset_cfg.top_k,
        max_new_tokens=dataset_cfg.max_response_len,
        stop=args.rollout_stop,
        stop_token_ids=args.rollout_stop_token_ids,
        skip_special_tokens=args.rollout_skip_special_tokens,
        no_stop_trim=True,
        spaces_between_special_tokens=False,
    )

    tasks = []
    # do multiple samples for eval prompts
    sample_index = 0
    for _i, prompt_sample in enumerate(dataset.samples):
        for j in range(dataset_cfg.n_samples_per_eval_prompt):
            # use the same prompt for multiple samples
            sample = copy.deepcopy(prompt_sample)
            sample.index = sample_index
            sample_index += 1
            sample.metadata = dataset_cfg.inject_metadata(getattr(sample, "metadata", None))
            sample.generate_function_path = getattr(dataset_cfg, "custom_generate_function_path", None)
            sampling_params = base_sampling_params
            if getattr(args, "sglang_enable_deterministic_inference", False):
                sampling_params = base_sampling_params.copy()
                sampling_params["sampling_seed"] = args.rollout_seed + j
            tasks.append(
                asyncio.create_task(
                    generate_and_rm(
                        args,
                        sample,
                        sampling_params=sampling_params,
                        evaluation=True,
                    )
                )
            )

    data = []
    do_print = True
    pbar = tqdm(total=len(tasks), desc=f"Eval {dataset_cfg.name}", disable=not do_print)
    for coro in asyncio.as_completed(tasks):
        sample = await coro
        if do_print:
            logger.info(
                "eval_rollout_single_dataset example data: "
                f"{[str(sample.prompt) + sample.response]} "
                f"reward={sample.reward}"
            )
            do_print = False
        if isinstance(sample, list):
            data.extend(sample)
        else:
            data.append(sample)
        pbar.update(1)
    pbar.close()

    data.sort(key=lambda sample: sample.index)

    reward_key = args.eval_reward_key or args.reward_key

    def _reward_for_output(sample: Sample):
        if not reward_key or not isinstance(sample.reward, dict):
            return sample.reward
        return sample.reward[reward_key]

    return {
        dataset_cfg.name: {
            "rewards": [_reward_for_output(sample) for sample in data],
            "truncated": [sample.status == Sample.Status.TRUNCATED for sample in data],
            "samples": data,
        }
    }


def generate_rollout(
    args: Namespace, rollout_id: int, data_source: Any, evaluation: bool = False
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    """An example to implement the generate_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        data_buffer: the data buffer to store the generated samples
        evaluation: bool, whether the rollout is for evaluation or not

    Returns:
        list[list[Sample]]: a list of list of samples generated by the rollout
    """
    assert args.rollout_global_dataset
    if evaluation:
        output, _ = run(eval_rollout(args, rollout_id))
        return output

    output, aborted_samples = run(generate_rollout_async(args, rollout_id, data_source.get_samples))
    data_source.add_samples(aborted_samples)
    return output
