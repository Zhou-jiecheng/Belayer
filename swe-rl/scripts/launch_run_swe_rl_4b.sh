source /mnt/shared-storage-user/ailab-sys/zhoujiecheng/bashrc.sh

proxy_off

export SWE_EXEC_SERVER_URLS="http://100.101.233.137:5000"
export HF_CKPT=/mnt/shared-storage-user/ailab-sys/zhoujiecheng/projs/models/Qwen3-4B
export PROMPT_DATA=/mnt/shared-storage-user/ailab-sys/zhoujiecheng/projs/robust_rl/OpenClaw-RL/swe-rl/data/train.jsonl
export MEGATRON_LM_PATH=/mnt/shared-storage-user/ailab-sys/zhoujiecheng/projs/robust_rl/OpenClaw-RL/Megatron-LM

bash /mnt/shared-storage-user/ailab-sys/zhoujiecheng/projs/robust_rl/OpenClaw-RL/swe-rl/scripts/run_swe_rl.sh