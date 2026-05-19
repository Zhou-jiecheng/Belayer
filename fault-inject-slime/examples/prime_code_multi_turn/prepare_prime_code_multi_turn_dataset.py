#!/usr/bin/env python3
"""Prepare PRIME coding parquet into SLIME multi-turn prompt format."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

try:
    import orjson  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    orjson = None

"""
problem: You are a math/geometry expert. Solve the user's question carefully and verify your work.

Follow this protocol:
1) First, reason step by step as an internal monologue wrapped inside <think>...</think> tags.
2) After you finish the reasoning, you must call the `calc_score` (aka `calc_geo3k_reward`) tool at least once with your parsed numeric answer to check correctness before finalizing.
   - Emit tool calls in the below format:
    <tool_call>{"name": "calc_score", "arguments": {"answer": "<digits>"}}</tool_call>
    Always include the <tool_call></tool_call> tag, and object within the tag must be JSON.
    Right after emitting tool call (in the same message), you must always also include the answer in the form Answer: \boxed{$Answer} so the answer can be extracted.
3) Use the tool feedback to refine your solution if needed.
4) Provide the final answer in the form Answer: \boxed{$Answer} where $Answer is the answer to the problem.

Solve the following math problem step by step. The last line of your response should be of the form Answer: \boxed{$Answer} where $Answer is the answer to the problem.
"""


BASE_SYSTEM_PROMPT = (
    "You are a coding expert. Solve the user's question carefully and verify your work.\n\n"
    "Follow this protocol:\n"
    "1) First, reason step by step as an internal monologue wrapped inside <think>...</think> tags.\n"
    "   - Keep the reasoning concise and focused on producing a full candidate Python solution.\n"
    "2) After you finish the reasoning and generate the code, you must call the `execute_code` tool at least once with your code to check correctness before finalizing.\n"
    "   - Emit tool calls in the below format:\n"
    '    <tool_call>{"name": "execute_code", "arguments": {"code": "..."}}</tool_call>\n'
    "    Always include the <tool_call></tool_call> tag, and the object within the tag must be JSON.\n"
    "    The `code` field must be a string containing the full Python program to execute. Do not emit partial code or just the answer.\n"
    "    In the same assistant message, first write <think>...</think>, then immediately emit exactly one <tool_call>...</tool_call> block.\n"
    "    Do not put any extra text after the tool call.\n"
    "3) After receiving tool feedback, either send another message in the same format (<think> followed by exactly one <tool_call>), or provide the final answer.\n"
    "4) Final answer rules:\n"
    "   Final answer must be code only.\n"
    "   Final answer must be a single ```python``` block.\n"
    "   Do not include any prose before or after the final code block.\n"
    "5) Never use <tool_call> tags except for an actual execute_code tool invocation.\n"
    "6) Do not stop after reasoning only. In a non-final attempt, you must actually emit the execute_code tool call in that same message.\n"
    "Solve the following coding problem step by step. "
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare PRIME coding data for SLIME multi-turn rollout.")
    parser.add_argument("--input", required=True, help="Input PRIME parquet path.")
    parser.add_argument("--output", required=True, help="Output parquet path.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of records to write.")
    parser.add_argument(
        "--ability",
        nargs="*",
        default=None,
        help="Optional ability filter, e.g. --ability code coding",
    )
    return parser


def load_json(text: str) -> Any:
    loader = orjson.loads if orjson is not None else json.loads
    return loader(text)


def inject_tool_instruction(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    new_messages = copy.deepcopy(messages)
    normalized_messages: list[dict[str, Any]] = []

    if new_messages and new_messages[0].get("role") == "system":
        first_system = copy.deepcopy(new_messages[0])
        first_system["content"] = BASE_SYSTEM_PROMPT
        normalized_messages.append(first_system)
        new_messages = new_messages[1:]
    else:
        normalized_messages.append({"role": "system", "content": BASE_SYSTEM_PROMPT})
    normalized_messages.extend(new_messages)
    return normalized_messages


def canonicalize_test_cases(test_cases: dict[str, Any]) -> dict[str, Any]:
    inputs = list(test_cases.get("inputs") or [])
    outputs = list(test_cases.get("outputs") or [])
    normalized: dict[str, Any] = {"inputs": [], "outputs": []}

    if test_cases.get("fn_name") is not None:
        normalized["fn_name"] = str(test_cases["fn_name"])
        for item in inputs:
            args = item if isinstance(item, list) else [item]
            normalized["inputs"].append("\n".join(json.dumps(arg, ensure_ascii=False) for arg in args))
        for item in outputs:
            normalized["outputs"].append(json.dumps(item, ensure_ascii=False))
        return normalized

    normalized["inputs"] = [str(item) for item in inputs]
    normalized["outputs"] = [str(item) for item in outputs]
    return normalized


def convert_row(row: dict[str, Any]) -> dict[str, Any]:
    raw_ground_truth = row["reward_model"]["ground_truth"]
    test_cases = load_json(raw_ground_truth) if isinstance(raw_ground_truth, str) else raw_ground_truth
    test_cases = canonicalize_test_cases(test_cases)
    ground_truth = json.dumps(test_cases, ensure_ascii=False)
    test_case_format = "function_call" if isinstance(test_cases, dict) and test_cases.get("fn_name") else "stdin"
    return {
        "messages": inject_tool_instruction(row["prompt"]),
        "label": ground_truth,
        "metadata": {
            "data_source": row.get("data_source"),
            "ability": row.get("ability"),
            "style": (row.get("reward_model") or {}).get("style"),
            "extra_info": row.get("extra_info") or {},
            "test_cases_json": ground_truth,
            "test_case_format": test_case_format,
        },
    }


def main() -> int:
    args = build_parser().parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {input_path}")

    parquet_file = pq.ParquetFile(input_path)
    writer: pq.ParquetWriter | None = None
    written = 0
    ability_filter = set(args.ability) if args.ability else None

    try:
        for row_group_index in range(parquet_file.metadata.num_row_groups):
            rows = parquet_file.read_row_group(row_group_index).to_pylist()
            prepared_rows = []
            for row in rows:
                if ability_filter and str(row.get("ability")) not in ability_filter:
                    continue
                prepared_rows.append(convert_row(row))
                if args.limit is not None and written + len(prepared_rows) >= args.limit:
                    prepared_rows = prepared_rows[: args.limit - written]
                    break

            if not prepared_rows:
                if args.limit is not None and written >= args.limit:
                    break
                continue

            table = pa.Table.from_pylist(prepared_rows)
            if writer is None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                writer = pq.ParquetWriter(output_path, table.schema)
            writer.write_table(table)
            written += len(prepared_rows)

            if args.limit is not None and written >= args.limit:
                break
    finally:
        if writer is not None:
            writer.close()

    print(f"input={input_path}")
    print(f"output={output_path}")
    print(f"written_rows={written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
