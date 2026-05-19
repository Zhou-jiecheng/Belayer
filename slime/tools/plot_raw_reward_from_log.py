#!/usr/bin/env python3
"""Plot raw_reward curves from fault-inject and normal logs on one figure."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_FAULT_LOG_PATH = Path(
    "/mnt/shared-storage-user/ailab-sys/zhoujiecheng/projs/robust_rl/OpenClaw-RL/"
    "slime/logs/prime_multi_turn/"
    "Qwen2.5_7B_Instruct_prime_multi_turn-1-fault-inject-0-1-part-1.log"
)
DEFAULT_NORMAL_LOG_PATH = Path(
    "/mnt/shared-storage-user/ailab-sys/zhoujiecheng/projs/robust_rl/OpenClaw-RL/"
    "slime/logs/prime_multi_turn/"
    "Qwen2.5_7B_Instruct_prime_multi_turn-no-fault-inject-max-turn-1.log"
)
DEFAULT_OUTPUT_PATH = Path(
    "/mnt/shared-storage-user/ailab-sys/zhoujiecheng/projs/robust_rl/OpenClaw-RL/"
    "slime/logs/prime_multi_turn/raw_reward_compare.png"
)

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")
ROLLOUT_RE = re.compile(r"rollout\s+(\d+):")
RAW_REWARD_RE = re.compile(r"'rollout/raw_reward':\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract rollout/raw_reward from two logs and draw them on one figure."
    )
    parser.add_argument(
        "--fault-log-path",
        type=Path,
        default=DEFAULT_FAULT_LOG_PATH,
        help=f"Fault-inject log path. Default: {DEFAULT_FAULT_LOG_PATH}",
    )
    parser.add_argument(
        "--normal-log-path",
        type=Path,
        default=DEFAULT_NORMAL_LOG_PATH,
        help=f"Normal log path. Default: {DEFAULT_NORMAL_LOG_PATH}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output figure path. Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--use-rollout-index",
        action="store_true",
        help="Use rollout index in the log as x-axis. Otherwise use 0-based extraction order.",
    )
    parser.add_argument(
        "--dump-json",
        type=Path,
        default=None,
        help="Optional JSON path to save both extracted series.",
    )
    return parser.parse_args()


def extract_raw_rewards(log_path: Path, use_rollout_index: bool) -> tuple[list[int], list[float]]:
    steps: list[int] = []
    rewards: list[float] = []

    with log_path.open("r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            clean_line = ANSI_ESCAPE_RE.sub("", line)
            raw_reward_match = RAW_REWARD_RE.search(clean_line)
            if raw_reward_match is None:
                continue

            if use_rollout_index:
                rollout_match = ROLLOUT_RE.search(clean_line)
                step = int(rollout_match.group(1)) if rollout_match else len(rewards)
            else:
                step = len(rewards)

            steps.append(step)
            rewards.append(float(raw_reward_match.group(1)))

    if not rewards:
        raise ValueError(f"No rollout/raw_reward found in log file: {log_path}")

    return steps, rewards

def moving_average(values: list[float], window_size: int) -> list[float]:
      if window_size <= 1 or window_size > len(values):
          return values[:]

      smoothed = []
      window_sum = sum(values[:window_size])

      for idx in range(len(values)):
          if idx < window_size - 1:
              smoothed.append(values[idx])
              continue

          if idx == window_size - 1:
              smoothed.append(window_sum / window_size)
              continue

          window_sum += values[idx] - values[idx - window_size]
          smoothed.append(window_sum / window_size)

      return smoothed

def plot_two_curves(
    steps_fault_inject_0_1: list[int],
    rewards_fault_inject_0_1: list[float],
    steps_normal: list[int],
    rewards_normal: list[float],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    steps_num = min(len(steps_fault_inject_0_1), len(steps_normal))
    steps = steps_fault_inject_0_1[:steps_num]

    rewards_fault_inject_0_1 = rewards_fault_inject_0_1[:steps_num]
    rewards_normal = rewards_normal[:steps_num]

    rewards_fault_inject_0_1 = moving_average(rewards_fault_inject_0_1, window_size=10)
    rewards_normal = moving_average(rewards_normal, window_size=10)

    plt.figure(figsize=(12, 6))
    plt.plot(
        steps,
        rewards_fault_inject_0_1,
        label="fault_inject_0_1",
        linewidth=1.8,
        color="#d62728",
    )
    plt.plot(
        steps,
        rewards_normal,
        label="normal",
        linewidth=1.8,
        color="#1f77b4",
    )
    plt.xlabel("step")
    plt.ylabel("rollout/raw_reward")
    plt.title("rollout/raw_reward Comparison")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()

    steps_fault_inject_0_1, rewards_fault_inject_0_1 = extract_raw_rewards(
        args.fault_log_path,
        use_rollout_index=False,
    )
    step_fault_inject_2_0, rewards_fault_inject_2_0 = extract_raw_rewards(
        Path("/mnt/shared-storage-user/ailab-sys/zhoujiecheng/projs/robust_rl/OpenClaw-RL/slime/logs/prime_multi_turn/Qwen2.5_7B_Instruct_prime_multi_turn-1-fault-inject-0-1-part-2.log"),
        use_rollout_index=False,
    )
    steps_normal, rewards_normal = extract_raw_rewards(
        args.normal_log_path,
        use_rollout_index=False,
    )

    rewards_fault_inject = rewards_fault_inject_0_1 + rewards_fault_inject_2_0
    steps_fault_inject = list(range(len(rewards_fault_inject)))


    plot_two_curves(
        steps_fault_inject_0_1=steps_fault_inject,
        rewards_fault_inject_0_1=rewards_fault_inject,
        steps_normal=steps_normal,
        rewards_normal=rewards_normal,
        output_path=args.output,
    )

    print(f"fault_inject_0_1 points: {len(rewards_fault_inject)}")
    print(f"normal points: {len(rewards_normal)}")
    print(f"figure saved to: {args.output}")
    if args.dump_json is not None:
        print(f"json saved to: {args.dump_json}")


if __name__ == "__main__":
    main()
