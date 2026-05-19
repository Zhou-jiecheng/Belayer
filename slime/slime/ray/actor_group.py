import os
import time

import ray
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from slime.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST


import logging


logger = logging.getLogger(__name__)


class RayTrainGroup:
    """
    A group of ray actors
    Functions start with 'async' should return list of object refs

    Args:
        args (Namespace): Arguments for the actor group.
        num_nodes (int): Number of nodes for this actor group.
        num_gpus_per_node (int): Number of gpus for this actor group.
        pg (PlacementGroup, optional): Placement group to schedule actor on.
            If none, create new placement group automatically. Defaults to None.
        num_gpus_per_actor (float, optional): Number of gpus allocated for each actor.
            If < 1.0, multiple models can share same gpu. Defaults to 1.
        resources (Dict[str, float], optional): Custom resources to allocate for each actor.
            See https://docs.ray.io/en/latest/ray-core/scheduling/resources.html
        num_resources_per_node (int, optional): Number of custom resources to allocate for each node.
            See https://docs.ray.io/en/latest/ray-core/scheduling/resources.html
    """

    def __init__(
        self,
        args,
        num_nodes,
        num_gpus_per_node,
        pg: tuple[PlacementGroup, list[int], list[int]],
        num_gpus_per_actor: float = 1,
        role: str = "actor",
    ) -> None:
        self.args = args
        self._num_nodes = num_nodes
        self._num_gpus_per_node = num_gpus_per_node
        self.role = role

        # Allocate the GPUs for actors w/o instantiating them
        self._allocate_gpus_for_actor(pg, num_gpus_per_actor)

    def _allocate_gpus_for_actor(self, pg, num_gpus_per_actor):
        world_size = self._num_nodes * self._num_gpus_per_node

        # Use placement group to lock resources for models of same type
        assert pg is not None
        pg, reordered_bundle_indices, _reordered_gpu_ids = pg

        env_vars = {
            # because sglang will always set NCCL_CUMEM_ENABLE to 0
            # we need also set it to 0 to prevent nccl error.
            "NCCL_CUMEM_ENABLE": os.environ.get("NCCL_CUMEM_ENABLE", "0"),
            "NVTE_FP8_BLOCK_SCALING_FP32_SCALES": os.environ.get("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", "1"),
            **{name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST},
            **self.args.train_env_vars,
        }

        if self.args.offload_train and self.args.train_backend == "megatron":
            import torch_memory_saver

            dynlib_path = os.path.join(
                os.path.dirname(os.path.dirname(torch_memory_saver.__file__)),
                "torch_memory_saver_hook_mode_preload.abi3.so",
            )
            assert os.path.exists(dynlib_path), f"LD_PRELOAD so file {dynlib_path} does not exist."

            env_vars["LD_PRELOAD"] = dynlib_path
            env_vars["TMS_INIT_ENABLE"] = "1"
            env_vars["TMS_INIT_ENABLE_CPU_BACKUP"] = "1"

        # We cannot do routing replay for critic.
        if self.args.use_routing_replay and self.role == "actor":
            env_vars["ENABLE_ROUTING_REPLAY"] = "1"

        backend = self.args.train_backend
        if backend == "megatron":
            from slime.backends.megatron_utils.actor import MegatronTrainRayActor

            actor_impl = MegatronTrainRayActor

        else:
            from slime.backends.fsdp_utils import FSDPTrainRayActor

            actor_impl = FSDPTrainRayActor

        TrainRayActor = ray.remote(num_gpus=1, runtime_env={"env_vars": env_vars})(actor_impl)

        # Create worker actors
        self._actor_handlers = []
        master_addr, master_port = None, None
        for rank in range(world_size):
            actor = TrainRayActor.options(
                num_cpus=num_gpus_per_actor,
                num_gpus=num_gpus_per_actor,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=reordered_bundle_indices[rank],
                ),
            ).remote(world_size, rank, master_addr, master_port)
            if rank == 0:
                master_addr, master_port = ray.get(actor.get_master_addr_and_port.remote())
            self._actor_handlers.append(actor)

    def async_init(self, args, role, with_ref=False):
        """
        Allocate GPU resourced and initialize model, optimzier, local ckpt, etc.
        """
        self.args = args
        return [actor.init.remote(args, role, with_ref=with_ref) for actor in self._actor_handlers]

    def async_train(self, rollout_id, rollout_data_ref):
        """Do one rollout training"""
        return [actor.train.remote(rollout_id, rollout_data_ref) for actor in self._actor_handlers]

    def save_model(self, rollout_id, force_sync=False):
        """Save actor model"""
        return ray.get([actor.save_model.remote(rollout_id, force_sync=force_sync) for actor in self._actor_handlers])

    def update_weights(self):
        """Broadcast weights from rank 0 to all other ranks."""
        return ray.get([actor.update_weights.remote() for actor in self._actor_handlers])

    def onload(self):
        return ray.get([actor.wake_up.remote() for actor in self._actor_handlers])

    def offload(self):
        return ray.get([actor.sleep.remote() for actor in self._actor_handlers])

    def clear_memory(self):
        return ray.get([actor.clear_memory.remote() for actor in self._actor_handlers])

    def connect(self, critic_group):
        return ray.get(
            [
                actor.connect_actor_critic.remote(critic)
                for actor, critic in zip(self._actor_handlers, critic_group._actor_handlers, strict=False)
            ]
        )

    def set_rollout_manager(self, rollout_manager):
        total = len(self._actor_handlers)
        logger.info("[RayTrainGroup.set_rollout_manager] submitting to %d actors...", total)
        refs = [actor.set_rollout_manager.remote(rollout_manager) for actor in self._actor_handlers]
        ref_to_rank = {ref: rank for rank, ref in enumerate(refs)}

        pending = list(refs)
        completed = 0
        start = time.perf_counter()

        while pending:
            ready, pending = ray.wait(pending, num_returns=1, timeout=30)
            if not ready:
                logger.info(
                    "[RayTrainGroup.set_rollout_manager] still waiting: completed=%d/%d, pending=%d, elapsed=%.2fs",
                    completed,
                    total,
                    len(pending),
                    time.perf_counter() - start,
                )
                continue

            ref = ready[0]
            rank = ref_to_rank[ref]
            ray.get(ref)
            completed += 1
            logger.info(
                "[RayTrainGroup.set_rollout_manager] actor rank %d done (%d/%d), elapsed=%.2fs",
                rank,
                completed,
                total,
                time.perf_counter() - start,
            )

        logger.info(
            "[RayTrainGroup.set_rollout_manager] all actors done in %.2fs",
            time.perf_counter() - start,
        )
        return None

    def get_train_parallel_config(self):
        logger.info("[RayTrainGroup.get_train_parallel_config] fetching config from rank 0 actor...")
        config = ray.get(self._actor_handlers[0].get_train_parallel_config.remote())
        logger.info("[RayTrainGroup.get_train_parallel_config] fetched config: %s", config)
        return config
