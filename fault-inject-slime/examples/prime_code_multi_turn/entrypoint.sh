# rm -rf ./slime/  && cp -r /mnt/shared-storage-user/ailab-sys/zhoujiecheng/projs/robust_rl/OpenClaw-RL/fault-inject-slime/ ./slime/
# rm -rf ./examples/geo3k_vlm_multi_turn/  && cp -r /mnt/shared-storage-user/ailab-sys/zhoujiecheng/projs/robust_rl/OpenClaw-RL/slime/examples/geo3k_vlm_multi_turn/ ./examples/geo3k_vlm_multi_turn/
cd ~
rm -rf ./slime/  && cp -r /mnt/shared-storage-user/ailab-sys/zhoujiecheng/projs/robust_rl/OpenClaw-RL/fault-inject-slime/ ./slime/
cd ~/slime


TENSORBOARD_DIR=/mnt/shared-storage-user/ailab-sys/zhoujiecheng/projs/robust_rl/OpenClaw-RL/slime/tensorboard_022/prime_code_multi_turn/Qwen2.5-7B-Instruct-no-fault-inject-max-turn-1-w-format-reward PRIME_CODE_ERROR_INJECTION_ENABLED=0 PRIME_CODE_ERROR_INJECTION_PROB=0.0 PRIME_CODE_SHARED_JUDGE_THREAD_POOL_WORKERS=64 PRIME_CODE_SHARED_REWARD_THREAD_POOL_WORKERS=16 SLIME_ENV_STEP_THREAD_POOL_WORKERS=128 SLIME_ENV_STEP_MAX_IN_FLIGHT=128 SLIME_ENV_STEP_QUEUE_TIMEOUT_SEC=120 SLIME_ROLLOUT_SAMPLE_TIMEOUT_SEC=2400 SLIME_ROLLOUT_GROUP_TIMEOUT_SEC=2400 SLIME_SCRIPT_MODE=normal python ./examples/prime_code_multi_turn/run_prime_multi_turn.py 2>&1 | tee  /mnt/shared-storage-user/ailab-sys/zhoujiecheng/projs/robust_rl/OpenClaw-RL/slime/logs/prime_multi_turn/Qwen2.5_7B_Instruct_prime_multi_turn-no-fault-inject-max-turn-1-w-format-reward-eval.log
