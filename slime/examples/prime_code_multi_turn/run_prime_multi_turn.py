import os
from pathlib import Path

import slime.utils.external_utils.command_utils as U

os.environ["PYTHONUNBUFFERED"] = "1"

MODEL_PATH = os.environ.get(
    "SLIME_SCRIPT_MODEL_PATH",
    "/mnt/shared-storage-user/ailab-sys/zhoujiecheng/projs/models/Qwen2.5-3B-Instruct",
)
MODE = os.environ.get("SLIME_SCRIPT_MODE", "normal")
NUM_GPUS = int(os.environ.get("SLIME_SCRIPT_NUM_GPUS", "8"))
TRAIN_BACKEND = os.environ.get("SLIME_SCRIPT_TRAIN_BACKEND", "megatron").lower()
assert MODE in {"normal", "debug_minimal", "debug_one_sample"}
assert TRAIN_BACKEND in {"fsdp", "megatron"}

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path("/mnt/shared-storage-user/ailab-sys/zhoujiecheng/projs/robust_rl/OpenClaw-RL/data")
TRAIN_RAW_PATH = DATA_ROOT / "train_coding.parquet"
VAL_RAW_PATH = DATA_ROOT / "validation_coding.parquet"
TRAIN_DATA_PATH = DATA_ROOT / "train_coding_multi_turn.parquet"
VAL_DATA_PATH = DATA_ROOT / "validation_coding_multi_turn.parquet"
PREP_SCRIPT = REPO_ROOT / "slime/examples/prime_code_multi_turn/prepare_prime_code_multi_turn_dataset.py"
CONFIG_PATH = "examples/prime_code_multi_turn/prime_code_multi_turn_config.yaml"
PYTHON_BIN = os.environ.get(
    "SLIME_SCRIPT_PYTHON_BIN",
    "/mnt/shared-storage-user/ailab-sys/zhoujiecheng/miniconda/envs/osworld/bin/python",
)
TENSORBOARD_DIR="/mnt/shared-storage-user/ailab-sys/zhoujiecheng/projs/robust_rl/OpenClaw-RL/slime/tensorboard_022/prime_code_multi_turn/Qwen2.5-3B-Instruct"
INJECTION_PROB = os.environ.get("PRIME_CODE_ERROR_INJECTION_PROB", "0.0")
IF_INJECTION = os.environ.get("PRIME_CODE_ERROR_INJECTION_ENABLED", "0").lower() in {"1", "true", "yes"}

def get_model_name() -> str:
    return Path(MODEL_PATH).name


def get_megatron_model_type() -> str:
    return "qwen2.5-3B"


def prepare():
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Model path not found: {MODEL_PATH}")
    if not TRAIN_RAW_PATH.exists():
        raise FileNotFoundError(f"Raw train dataset not found: {TRAIN_RAW_PATH}")
    if not VAL_RAW_PATH.exists():
        raise FileNotFoundError(f"Raw validation dataset not found: {VAL_RAW_PATH}")

    force_prepare = os.environ.get("SLIME_SCRIPT_FORCE_PREPARE", "0").lower() in {"1", "true", "yes"}

    if force_prepare or not TRAIN_DATA_PATH.exists():
        U.exec_command(
            f"PYTHONUNBUFFERED=1 {PYTHON_BIN} {PREP_SCRIPT} --input {TRAIN_RAW_PATH} --output {TRAIN_DATA_PATH}"
        )
    if force_prepare or not VAL_DATA_PATH.exists():
        U.exec_command(
            f"PYTHONUNBUFFERED=1 {PYTHON_BIN} {PREP_SCRIPT} --input {VAL_RAW_PATH} --output {VAL_DATA_PATH}"
        )


def execute():
    model_name = get_model_name()
    run_name = f"{model_name}_prime_code_multi_turn_{'fault_inject_' + INJECTION_PROB if IF_INJECTION else 'no_fault_injection'}"
    rollout_max_context_len = 16000
    rollout_max_prompt_len = 2048
    rollout_max_response_len = 4096

    ckpt_args = (
        f"--hf-checkpoint {MODEL_PATH} "
        f"--save /mnt/shared-storage-user/ailab-sys/zhoujiecheng/projs/robust_rl/OpenClaw-RL/slime/checkpoints/prime_code_multi_turn/{run_name} "
        "--save-interval 20 "
    )

    rollout_args = (
        f"--prompt-data {TRAIN_DATA_PATH} "
        "--input-key messages "
        "--label-key label "
        "--metadata-key metadata "
        "--apply-chat-template "
        "--custom-generate-function-path examples.prime_code_multi_turn.rollout.generate "
        "--custom-config-path examples/prime_code_multi_turn/prime_code_multi_turn_config.yaml "
        "--custom-rm-path examples.prime_code_multi_turn.reward.prime_code_reward "
        "--rollout-shuffle "
        f"--num-rollout {4 if MODE == 'debug_one_sample' else 20 if MODE == 'debug_minimal' else 300} "
        f"--rollout-batch-size {1 if MODE == 'debug_one_sample' else 4 if MODE == 'debug_minimal' else 32} "
        f"--n-samples-per-prompt {1 if MODE == 'debug_one_sample' else 2 if MODE == 'debug_minimal' else 8} "
        f"--rollout-max-context-len {rollout_max_context_len} "
        f"--rollout-max-prompt-len {rollout_max_prompt_len} "
        f"--rollout-max-response-len {rollout_max_response_len} "
        "--rollout-temperature 1 "
        "--rollout-top-p 1 "
        f"--global-batch-size {1 if MODE == 'debug_one_sample' else 8 if MODE == 'debug_minimal' else 256} "
        "--save-debug-rollout-data /mnt/shared-storage-user/ailab-sys/zhoujiecheng/projs/robust_rl/OpenClaw-RL/slime/debug_data/prime_code/Qwen2.5-3B-Instruct/data_{rollout_id}.pt "
    )

    eval_args = (
        ""
        if MODE != "normal"
        else (
            "--eval-interval 20 "
            f"--eval-prompt-data prime_code_eval {VAL_DATA_PATH} "
            "--n-samples-per-eval-prompt 1 "
            f"--eval-max-context-len {rollout_max_context_len} "
            f"--eval-max-response-len {rollout_max_response_len} "
            "--eval-top-k 1 "
        )
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    sglang_args = (
        "--rollout-num-gpus-per-engine 1 "
        # "--use-slime-router "
        f"--sglang-mem-fraction-static {0.7 if MODE == 'debug_one_sample' else 0.6} "
        f"--sglang-cuda-graph-bs {' '.join(map(str, [1, 2, 4, 8] + list(range(16, 257, 8))))} "
    )

    fsdp_args = (
        "--train-backend fsdp "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 8192 "
        "--gradient-checkpointing "
        "--update-weight-buffer-size 536870912 "
    )

    megatron_args = (
        "--train-backend megatron "
        f"--load {MODEL_PATH} "
        f"--tensor-model-parallel-size {2 if NUM_GPUS >= 4 else 1} "
        # "--rotary-base 5000000 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 8192 "
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--megatron-to-hf-mode bridge "
    )

    misc_args = (
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {NUM_GPUS} "
        f"--rollout-num-gpus {NUM_GPUS} "
        "--colocate "
    )

    logging_args = (
        (
            "--use-wandb "
            "--wandb-project slime-dev "
            "--wandb-group prime_code_multi_turn "
            f"--wandb-key '{wandb_api_key}' "
            "--disable-wandb-random-suffix "
        )
        if (wandb_api_key := os.environ.get("WANDB_API_KEY"))
        else (
            "--use-tensorboard "
            "--tb-project-name prime_code_multi_turn "
            f"--tb-experiment-name {run_name} "
        )
    )

    if TRAIN_BACKEND == "megatron":
        backend_args = megatron_args
        megatron_model_type = get_megatron_model_type()
    else:
        backend_args = fsdp_args
        megatron_model_type = None

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{sglang_args} "
        f"{backend_args} "
        f"{misc_args} "
        f"{logging_args} "
        # f"{eval_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=megatron_model_type,
        extra_env_vars=({}),
    )


if __name__ == "__main__":
    prepare()
    execute()
