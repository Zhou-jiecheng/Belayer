# PRIME Code Multi-Turn

This example adapts the geo3k multi-turn rollout contract to PRIME coding data.
It exposes exactly one tool, `execute_code`, and the prepared dataset injects a strict system instruction that forces the model to emit only the supported tool-call format.

## What is included

- `env_prime_code.py`: parses `<tool_call>{...}</tool_call>`, executes Python code, runs hidden tests, and returns pass/fail feedback to the model.
- `rollout.py`: reuses the generic multi-turn rollout from `examples.geo3k_vlm_multi_turn.rollout`.
- `prime_code_multi_turn_config.yaml`: configures `max_turns` and code execution limits.
- `prime_code_multi_turn_config.yaml`: also points to the reference PRIME judge implementation under `verl/utils/reward_score/prime_code`.
- `prepare_prime_code_multi_turn_dataset.py`: converts PRIME coding parquet into a SLIME-friendly parquet with `messages`, `label`, and `metadata.test_cases`.

## Prepare dataset

```bash
python slime/examples/prime_code_multi_turn/prepare_prime_code_multi_turn_dataset.py \
  --input data/train_coding.parquet \
  --output data/train_coding_multi_turn.parquet
python slime/examples/prime_code_multi_turn/prepare_prime_code_multi_turn_dataset.py \
  --input data/validation_coding.parquet \
  --output data/validation_coding_multi_turn.parquet
```

The prepared dataset schema is:

- `messages`: conversation prompt for `--apply-chat-template`
- `label`: raw test-case JSON string
- `metadata`: includes `test_cases_json`, `test_case_format`, `data_source`, `ability`, and `extra_info`

## Training hook-up

Use:

- `--input-key messages`
- `--label-key label`
- `--metadata-key metadata`
- `--apply-chat-template`
- `--custom-generate-function-path examples.prime_code_multi_turn.rollout.generate`
- `--custom-config-path examples/prime_code_multi_turn/prime_code_multi_turn_config.yaml`

This patch focuses on the environment interaction path. You still need to choose or implement the final reward logic for training.
