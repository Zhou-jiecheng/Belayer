#!/usr/bin/env python3
"""Export only PRIME coding samples into multi-turn parquet files."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

_PREP_PATH = Path(__file__).with_name("prepare_prime_code_multi_turn_dataset.py")
_PREP_SPEC = importlib.util.spec_from_file_location("_prime_prepare_module", _PREP_PATH)
if _PREP_SPEC is None or _PREP_SPEC.loader is None:
    raise ImportError(f"Cannot load prepare module from {_PREP_PATH}")
_PREP_MODULE = importlib.util.module_from_spec(_PREP_SPEC)
_PREP_SPEC.loader.exec_module(_PREP_MODULE)
convert_row = _PREP_MODULE.convert_row


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export PRIME ability=code samples to multi-turn parquet files.")
    parser.add_argument("--input-dir", default="./data", help="Directory containing train.parquet and validation.parquet")
    parser.add_argument(
        "--output-dir",
        default="./data",
        help="Directory to write train_coding_multi_turn.parquet and validation_coding_multi_turn.parquet",
    )
    parser.add_argument(
        "--output-prefix",
        default="coding_multi_turn",
        help="Output suffix, producing train_<suffix>.parquet and validation_<suffix>.parquet",
    )
    parser.add_argument(
        "--ability",
        nargs="+",
        default=["code"],
        help="Ability values to keep. Default: code",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of rows per split.")
    return parser


def export_split(input_path: Path, output_path: Path, ability_filter: set[str], limit: int | None) -> int:
    parquet_file = pq.ParquetFile(input_path)
    writer: pq.ParquetWriter | None = None
    written = 0

    try:
        for row_group_index in range(parquet_file.metadata.num_row_groups):
            rows = parquet_file.read_row_group(row_group_index).to_pylist()
            prepared_rows = []
            for row in rows:
                if str(row.get("ability")) not in ability_filter:
                    continue
                prepared_rows.append(convert_row(row))
                if limit is not None and written + len(prepared_rows) >= limit:
                    prepared_rows = prepared_rows[: limit - written]
                    break

            if not prepared_rows:
                if limit is not None and written >= limit:
                    break
                continue

            table = pa.Table.from_pylist(prepared_rows)
            if writer is None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                writer = pq.ParquetWriter(output_path, table.schema)
            writer.write_table(table)
            written += len(prepared_rows)

            if limit is not None and written >= limit:
                break
    finally:
        if writer is not None:
            writer.close()

    return written


def main() -> int:
    args = build_parser().parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    ability_filter = set(args.ability)

    for split in ("train", "validation"):
        input_path = input_dir / f"{split}.parquet"
        output_path = output_dir / f"{split}_{args.output_prefix}.parquet"
        if not input_path.exists():
            raise FileNotFoundError(f"Input parquet not found: {input_path}")

        written = export_split(input_path, output_path, ability_filter, args.limit)
        print(f"split={split}")
        print(f"input={input_path}")
        print(f"output={output_path}")
        print(f"written_rows={written}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
