import logging
import socket
import time
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence

import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
from ray import ObjectRef
from ray.actor import ActorHandle
from tqdm import tqdm

from slime.utils.distributed_utils import get_gloo_group, init_process_group

from ..megatron_to_hf import convert_to_hf
from .common import all_gather_param, named_params_and_buffers

logger = logging.getLogger(__name__)


class UpdateWeightFromDistributed:
    """
    Update distributed engines via NCCL. Each PP rank: group "slime-pp_{pp_rank}",
    only DP=TP=0 broadcasts. Non-expert (TP) and expert (EP) params separate.
    """

    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        weights_getter: Callable[[], Mapping[str, torch.Tensor]],
        *,
        model_name: str,
        quantization_config: dict[str, int | str | list[str]] | None,
    ) -> None:
        """
        Initialize. Groups created in connect_rollout_engines.
        """
        self.args = args
        self.model = model
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.weight_version = 0
        self._model_update_groups = None

    def _should_verify_rollout_weight_update(self) -> bool:
        return bool(getattr(self.args, "verify_rollout_weight_update", False))

    def _is_supported_verify_param_name(self, name: str) -> bool:
        return name.endswith(".self_attn.o_proj.weight") or name.endswith(".mlp.down_proj.weight")

    def _record_verify_samples(
        self,
        verify_samples: list[tuple[str, torch.Tensor]],
        seen_names: set[str],
        named_tensors: Sequence[tuple[str, torch.Tensor]],
    ) -> None:
        if not self._should_verify_rollout_weight_update():
            return

        target_num_params = max(0, int(getattr(self.args, "verify_rollout_weight_update_num_params", 0)))
        target_num_values = max(1, int(getattr(self.args, "verify_rollout_weight_update_num_values", 16)))
        if len(verify_samples) >= target_num_params:
            return

        def maybe_append(name: str, param: torch.Tensor) -> bool:
            if (
                name in seen_names
                or not torch.is_floating_point(param)
                or not self._is_supported_verify_param_name(name)
            ):
                return False
            verify_samples.append((name, param.detach().float().flatten()[:target_num_values].cpu().clone()))
            seen_names.add(name)
            return True

        for name, param in named_tensors:
            if len(verify_samples) >= target_num_params:
                return
            maybe_append(name, param)

    def _verify_rollout_engine_weights(self, verify_samples: Sequence[tuple[str, torch.Tensor]]) -> None:
        if not self._should_verify_rollout_weight_update() or dist.get_rank() != 0:
            return
        if not getattr(self, "rollout_engines", None):
            logger.warning(
                "verify_rollout_weight_update is enabled, but no rollout engines are attached to the updater."
            )
            return

        target_version = str(self.weight_version)
        truncate_size = max(1, int(getattr(self.args, "verify_rollout_weight_update_truncate_size", 4)))
        compare_num_values = max(1, int(getattr(self.args, "verify_rollout_weight_update_num_values", 16)))

        for engine_index, engine in enumerate(self.rollout_engines):
            engine_version = ray.get(engine.get_weight_version.remote())
            if str(engine_version) != target_version:
                raise RuntimeError(
                    f"Rollout engine[{engine_index}] weight version mismatch: "
                    f"engine={engine_version}, expected={target_version}"
                )

            sample_summaries: list[str] = []
            for name, expected_prefix in verify_samples:
                actual = ray.get(engine.get_weights_by_name.remote(name=name, truncate_size=truncate_size))
                if actual is None:
                    raise RuntimeError(
                        f"Rollout engine[{engine_index}] returned no data for sampled weight '{name}' "
                        f"while verifying weight_version={target_version}"
                    )

                actual_prefix = torch.tensor(actual, dtype=torch.float32).flatten()[:compare_num_values].cpu()
                expected_prefix = expected_prefix[:compare_num_values].cpu()
                compare_len = min(actual_prefix.numel(), expected_prefix.numel())
                if compare_len == 0:
                    raise RuntimeError(
                        f"Rollout engine[{engine_index}] returned an empty tensor prefix for sampled weight '{name}' "
                        f"while verifying weight_version={target_version}"
                    )

                actual_prefix = actual_prefix[:compare_len]
                expected_prefix = expected_prefix[:compare_len]
                max_abs_diff = torch.max(torch.abs(actual_prefix - expected_prefix)).item()
                mean_abs_diff = torch.mean(torch.abs(actual_prefix - expected_prefix)).item()
                if not torch.allclose(actual_prefix, expected_prefix, atol=5e-3, rtol=1e-3):
                    raise RuntimeError(
                        f"Rollout engine[{engine_index}] sampled weight mismatch for '{name}' at "
                        f"weight_version={target_version}: compare_len={compare_len}, "
                        f"max_abs_diff={max_abs_diff:.6g}, mean_abs_diff={mean_abs_diff:.6g}, "
                        f"expected={expected_prefix.tolist()}, actual={actual_prefix.tolist()}"
                    )

                sample_summaries.append(
                    f"{name}(n={compare_len}, max_abs_diff={max_abs_diff:.3g}, mean_abs_diff={mean_abs_diff:.3g})"
                )

            logger.info(
                "Verified rollout engine[%s] weight update: weight_version=%s samples=%s",
                engine_index,
                target_version,
                sample_summaries,
            )

    def connect_rollout_engines(
        self, rollout_engines: Sequence[ActorHandle], rollout_engine_lock: ActorHandle
    ) -> None:
        """
        Create NCCL "slime-pp_{pp_rank}" if PP source (DP=TP=0). Lock prevents concurrent broadcasts.
        """
        self.rollout_engines = rollout_engines
        self.rollout_engine_lock = rollout_engine_lock

        # For TP:
        #   1. AllGather parameters to rank 0
        #   2. Broadcast parameters from rank 0 to all sglang engines
        self._is_pp_src_rank = (
            mpu.get_data_parallel_rank(with_context_parallel=True) == 0 and mpu.get_tensor_model_parallel_rank() == 0
        )
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        if self._is_pp_src_rank:
            self._group_name = f"slime-pp_{pp_rank}"

        if self._is_pp_src_rank:
            if self._model_update_groups is not None:
                disconnect_rollout_engines_from_distributed(
                    self.args, self._group_name, self._model_update_groups, self.rollout_engines
                )
            self._model_update_groups = connect_rollout_engines_from_distributed(
                self.args, self._group_name, rollout_engines
            )

    @torch.no_grad()
    def update_weights(self) -> None:
        """
        Pause → flush → non-expert (TP) → expert (EP) → continue. Progress on PP source.
        """
        self.weight_version += 1
        verify_samples: list[tuple[str, torch.Tensor]] = []
        verify_sample_names: set[str] = set()

        if dist.get_rank() == 0:
            ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])
            ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])

            # int4/fp4 pre_process
            if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
                post_process_weights(
                    restore_weights_before_load=True,
                    post_process_quantization=False,
                    rollout_engines=self.rollout_engines,
                )
        dist.barrier(group=get_gloo_group())

        buffer_size = 0
        converted_named_tensors = []
        # non expert params
        pbar = tqdm(desc=f"[{self._group_name}] Update weights", total=0) if self._is_pp_src_rank else None

        for name, param in named_params_and_buffers(self.args, self.model):
            if ".experts." in name:
                continue
            buffer_size = self._update_weight_from_distributed(
                name,
                param,
                converted_named_tensors,
                buffer_size,
                verify_samples,
                verify_sample_names,
                pbar=pbar,
            )

        if converted_named_tensors:
            self._update_bucket_weights_from_distributed(converted_named_tensors, pbar=pbar)

        dist.barrier(group=get_gloo_group())

        buffer_size = 0
        named_tensors = []
        for name, param in named_params_and_buffers(self.args, self.model):
            if ".experts." not in name:
                continue
            buffer_size = self._update_expert_weight_from_distributed(
                name, param, named_tensors, buffer_size, pbar=pbar
            )

        if named_tensors:
            self._update_expert_bucket_weights_from_distributed(named_tensors, pbar=pbar)

        dist.barrier(group=get_gloo_group())
        if dist.get_rank() == 0:
            # int4/fp4 post_process
            if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
                post_process_weights(
                    restore_weights_before_load=False,
                    post_process_quantization=True,
                    rollout_engines=self.rollout_engines,
                )
            self._verify_rollout_engine_weights(verify_samples)
            ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())

    def _update_weight_from_distributed(
        self,
        name: str,
        param: torch.nn.Parameter,
        converted_named_tensors: list[tuple[str, torch.Tensor]],
        buffer_size: int,
        verify_samples: list[tuple[str, torch.Tensor]],
        verify_sample_names: set[str],
        pbar: tqdm | None = None,
    ) -> int | None:
        """
        Non-expert: gather TP → rm pad → HF → buffer (flush if full). All gather, PP source buffers.
        Returns updated bytes on source, None on non-source.
        """
        param = all_gather_param(name, param)
        if not self._is_pp_src_rank:
            return

        param_size = param.numel() * param.element_size()
        if buffer_size + param_size > self.args.update_weight_buffer_size:
            self._update_bucket_weights_from_distributed(converted_named_tensors, pbar=pbar)
            buffer_size = 0
        converted = convert_to_hf(self.args, self.model_name, name, param, self.quantization_config)
        self._record_verify_samples(verify_samples, verify_sample_names, converted)
        converted_named_tensors += converted
        buffer_size += param_size
        return buffer_size

    def _update_expert_weight_from_distributed(
        self,
        name: str,
        param: torch.nn.Parameter,
        named_tensors: list[tuple[str, torch.Tensor]],
        buffer_size: int,
        pbar: tqdm | None = None,
    ) -> int:
        """
        Expert: gather TP → rm pad → buffer. EP gather + HF deferred. Threshold × EP size.
        """
        param = all_gather_param(name, param)

        param_size = param.numel() * param.element_size()
        if (
            buffer_size + param_size
        ) * mpu.get_expert_model_parallel_world_size() > self.args.update_weight_buffer_size:
            self._update_expert_bucket_weights_from_distributed(named_tensors, pbar=pbar)
            buffer_size = 0

        named_tensors.append((name, param))
        buffer_size += param_size
        return buffer_size

    def _update_expert_bucket_weights_from_distributed(
        self, named_tensors: list[tuple[str, torch.Tensor]], pbar: tqdm | None = None
    ) -> None:
        """
        Gather EP → HF → broadcast. Clears buffer.
        """
        names = [name for name, _ in named_tensors]
        all_names = [None] * mpu.get_expert_model_parallel_world_size()
        dist.all_gather_object(all_names, names, group=mpu.get_expert_model_parallel_group())

        for names in all_names:
            assert len(named_tensors) == len(names), f"mismatch names length: {len(named_tensors)} != {len(names)}"

        all_gathered_params = [[] for _ in range(mpu.get_expert_model_parallel_world_size())]
        handles = []
        for i, (_name, param) in enumerate(named_tensors):
            params = [
                torch.empty_like(param.data, device=torch.cuda.current_device())
                for _ in range(mpu.get_expert_model_parallel_world_size())
            ]
            handle = dist.all_gather(params, param.data, group=mpu.get_expert_model_parallel_group(), async_op=True)
            handles.append(handle)
            for ep_rank, names in enumerate(all_names):
                all_gathered_params[ep_rank].append((names[i], params[ep_rank]))
        for handle in handles:
            handle.wait()

        named_tensors.clear()
        if not self._is_pp_src_rank:
            return

        all_gathered_params = sum(all_gathered_params, [])
        converted_hf_tensors = []
        for name, param in all_gathered_params:
            converted_hf_tensors += convert_to_hf(self.args, self.model_name, name, param, self.quantization_config)

        self._update_bucket_weights_from_distributed(converted_hf_tensors, pbar)

    def _update_bucket_weights_from_distributed(
        self, converted_named_tensors: list[tuple[str, torch.Tensor]], pbar: tqdm | None = None
    ) -> None:
        """
        Lock → broadcast → clear → unlock → pbar++. Lock prevents NCCL deadlock.
        """
        # lock the rollout engines to prevent dead lock on broadcast.
        while not ray.get(self.rollout_engine_lock.acquire.remote()):
            time.sleep(0.1)

        refs = update_weights_from_distributed(
            self._group_name,
            self._model_update_groups,
            self.weight_version,
            self.rollout_engines,
            converted_named_tensors,
        )

        ray.get(refs)
        converted_named_tensors.clear()
        ray.get(self.rollout_engine_lock.release.remote())
        pbar.update(1)


def connect_rollout_engines_from_distributed(
    args: Namespace, group_name: str, rollout_engines: Sequence[ActorHandle]
) -> dist.ProcessGroup:
    """
    Create NCCL group: training rank 0 + all engine GPUs. Blocks until joined.
    """
    master_address = ray._private.services.get_node_ip_address()
    with socket.socket() as sock:
        sock.bind(("", 0))
        master_port = sock.getsockname()[1]
    world_size = len(rollout_engines) * args.rollout_num_gpus_per_engine + 1

    refs = [
        engine.init_weights_update_group.remote(
            master_address,
            master_port,
            i * args.rollout_num_gpus_per_engine + 1,
            world_size,
            group_name,
            backend="nccl",
        )
        for i, engine in enumerate(rollout_engines)
    ]
    model_update_groups = init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_address}:{master_port}",
        world_size=world_size,
        rank=0,
        group_name=group_name,
    )
    ray.get(refs)
    return model_update_groups


def disconnect_rollout_engines_from_distributed(args, group_name, model_update_groups, rollout_engines):
    """
    Destroy NCCL on training and engines.
    """
    refs = [engine.destroy_weights_update_group.remote(group_name) for engine in rollout_engines]
    dist.destroy_process_group(model_update_groups)
    ray.get(refs)


def update_weights_from_distributed(
    group_name: str,
    group: dist.ProcessGroup,
    weight_version: int,
    rollout_engines: Sequence[ActorHandle],
    converted_named_tensors: Sequence[tuple[str, torch.Tensor]],
) -> list[ObjectRef]:
    """
    Send metadata (Ray), broadcast tensors (NCCL rank 0 → engines).
    """
    refs = [
        engine.update_weights_from_distributed.remote(
            names=[name for name, _ in converted_named_tensors],
            dtypes=[param.dtype for _, param in converted_named_tensors],
            shapes=[param.shape for _, param in converted_named_tensors],
            group_name=group_name,
            weight_version=str(weight_version),
        )
        for engine in rollout_engines
    ]

    handles = []
    for _, param in converted_named_tensors:
        handles.append(dist.broadcast(param.data, 0, group=group, async_op=True))
    for handle in handles:
        handle.wait()

    return refs


def post_process_weights(
    restore_weights_before_load: bool,
    post_process_quantization: bool,
    rollout_engines: Sequence[ActorHandle],
):
    """
    Trigger post-process for int4/fp4 quantization on all rollout engines.
    """
    ray.get(
        [
            engine.post_process_weights.remote(
                restore_weights_before_load=restore_weights_before_load,
                post_process_quantization=post_process_quantization,
            )
            for engine in rollout_engines
        ]
    )
