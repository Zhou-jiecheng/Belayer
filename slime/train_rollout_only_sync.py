import ray

from slime.ray.placement_group import create_placement_groups, create_rollout_manager
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger, init_tracking
from slime.utils.misc import should_run_periodic_action


def train(args):
    assert args.debug_rollout_only, (
        "train_rollout_only_sync.py requires --debug-rollout-only so argument "
        "normalization and GPU placement match rollout-only mode."
    )
    assert not args.colocate, "Colocation is not supported for sync rollout-only mode."

    configure_logger()
    pgs = create_placement_groups(args)
    init_tracking(args)

    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"], pgs.get("prm"))

    if args.start_rollout_id is None:
        args.start_rollout_id = 0

    if args.num_rollout == 0 and args.eval_interval is not None:
        ray.get(rollout_manager.eval.remote(rollout_id=0))

    try:
        for rollout_id in range(args.start_rollout_id, args.num_rollout):
            ray.get(rollout_manager.generate.remote(rollout_id))

            if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):
                if args.rollout_global_dataset:
                    ray.get(rollout_manager.save.remote(rollout_id))

            if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
                ray.get(rollout_manager.eval.remote(rollout_id))
    finally:
        ray.get(rollout_manager.dispose.remote())


if __name__ == "__main__":
    args = parse_args()
    train(args)
