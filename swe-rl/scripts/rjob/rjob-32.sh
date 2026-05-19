rjob submit --name=swe-rl-adaptive-checkpoint-32-gpu --gpu=8 --memory=1500000 --cpu=128 \
--charged-group=stu --private-machine=group \
--mount=gpfs://gpfs1/ailab-sys:/mnt/shared-storage-user/ailab-sys \
--image=registry.h.pjlab.org.cn/ailab-sys-sys_gpu/swe-rl:litellm_1_18_1 \
-P 4 \
--host-network=true \
--negative-tags node/gpu-lg-cmc-h-h200-1113.host.h.pjlab.org.cn \
--custom-resources rdma/mlnx_shared=8 \
--custom-resources mellanox.com/mlnx_rdma=1 \
-e DISTRIBUTED_JOB=true \
-- bash -exc /mnt/shared-storage-user/ailab-sys/zhoujiecheng/projs/robust_rl/OpenClaw-RL/swe-rl/scripts/launch_script/run_swe_rl_online_scheduler_adaptive_checkpoint_32gpu_qwen3-32b.sh

# while true; do echo "$(date '+%Y-%m-%d %H:%M:%S') $(docker ps | wc -l)" >> ./logs/online_scheduler_32gpu_docker_count.log; sleep 5; done