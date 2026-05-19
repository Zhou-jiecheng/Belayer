#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


def _sanitize_sys_path_for_torch() -> None:
    filtered = []
    for entry in sys.path:
        normalized = str(Path(entry).resolve()) if entry else ""
        if normalized.endswith("/projs/pytorch") or normalized.endswith("/pytorch"):
            continue
        filtered.append(entry)
    sys.path[:] = filtered


def _import_torch():
    _sanitize_sys_path_for_torch()
    import torch  # type: ignore

    return torch


def _to_jsonable(value: Any, *, torch_module) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    if dataclasses.is_dataclass(value):
        return _to_jsonable(dataclasses.asdict(value), torch_module=torch_module)

    if isinstance(value, dict):
        return {str(k): _to_jsonable(v, torch_module=torch_module) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item, torch_module=torch_module) for item in value]

    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return {
                "__type__": "bytes",
                "encoding": "base64",
                "length": len(value),
                "data": base64.b64encode(value).decode("ascii"),
            }

    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _to_jsonable(value.to_dict(), torch_module=torch_module)

    if torch_module is not None and isinstance(value, torch_module.Tensor):
        return {
            "__type__": "tensor",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "numel": int(value.numel()),
        }

    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return value.tolist()
        except Exception:
            pass

    return repr(value)


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2)


def _decode_from_tokens(tokenizer, sample: dict[str, Any]) -> tuple[str | None, int | None, str | None]:
    tokens = sample.get("tokens")
    prompt = sample.get("prompt")
    if not isinstance(tokens, list):
        return None, None, None

    decoded_tokens = tokenizer.decode(tokens, skip_special_tokens=False)

    prompt_token_count = None
    decoded_response = None
    if isinstance(prompt, str):
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_token_count = len(prompt_tokens)
        if prompt_token_count <= len(tokens):
            decoded_response = tokenizer.decode(tokens[prompt_token_count:], skip_special_tokens=False)

    return decoded_tokens, prompt_token_count, decoded_response


def _build_txt_content(sample_index: int, sample: dict[str, Any], tokenizer) -> str:
    decoded_tokens, prompt_token_count, decoded_response = _decode_from_tokens(tokenizer, sample)

    prompt = _as_text(sample.get("prompt"))
    response = _as_text(sample.get("response"))
    reward = _as_text(sample.get("reward"))
    status = _as_text(sample.get("status"))
    metadata = _as_text(sample.get("metadata"))

    return (
        f"SAMPLE_INDEX: {sample_index}\n"
        f"GROUP_INDEX: {_as_text(sample.get('group_index'))}\n"
        f"INDEX: {_as_text(sample.get('index'))}\n"
        f"STATUS: {status}\n"
        f"REWARD: {reward}\n"
        f"PROMPT_TOKEN_COUNT: {_as_text(prompt_token_count)}\n"
        f"RESPONSE_LENGTH: {_as_text(sample.get('response_length'))}\n"
        f"REMOVE_SAMPLE: {_as_text(sample.get('remove_sample'))}\n"
        "\n===== PROMPT =====\n"
        f"{prompt}\n"
        "\n===== RESPONSE =====\n"
        f"{response}\n"
        # "\n===== DECODED_RESPONSE_FROM_TOKENS =====\n"
        # f"{_as_text(decoded_response)}\n"
        # "\n===== DECODED_TOKENS =====\n"
        # f"{_as_text(decoded_tokens)}\n"
        "\n===== METADATA =====\n"
        f"{metadata}\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export a slime debug .pt file directly into per-sample readable TXT files."
    )
    parser.add_argument("input_pt", help="Input .pt file path")
    parser.add_argument("model_path", help="Tokenizer/model path used for token decoding")
    parser.add_argument(
        "-o",
        "--output-dir",
        help="Output directory for sample_*.txt files. Defaults to <input>.txt_samples",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading tokenizer.",
    )
    args = parser.parse_args()

    input_pt = Path(args.input_pt).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else input_pt.with_suffix(input_pt.suffix + ".txt_samples")
    )

    if not input_pt.exists():
        raise FileNotFoundError(f"Input file not found: {input_pt}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    torch = _import_torch()
    raw_obj = torch.load(str(input_pt), map_location="cpu", weights_only=False)
    data = _to_jsonable(raw_obj, torch_module=torch)

    if not isinstance(data, dict) or "samples" not in data or not isinstance(data["samples"], list):
        raise ValueError("Expected top-level dict with a 'samples' list.")

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=args.trust_remote_code,
    )

    samples = data["samples"]
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "source_file": str(input_pt),
        "model_path": str(model_path),
        "output_dir": str(output_dir),
        "num_samples": len(samples),
        "files": [],
    }

    for idx, sample in enumerate(samples):
        if not isinstance(sample, dict):
            sample = _to_jsonable(sample, torch_module=torch)
        txt_path = output_dir / f"sample_{idx:06d}.txt"
        content = _build_txt_content(idx, sample, tokenizer)
        with txt_path.open("w", encoding="utf-8") as f:
            f.write(content)
        manifest["files"].append(txt_path.name)

    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Wrote {len(samples)} txt files to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
